import pyvene as pv
import torch.nn as nn
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollator,
    DataCollatorForSeq2Seq,
    AutoTokenizer
)
from transformers.trainer_utils import (
    EvalPrediction,
    has_length,
    denumpify_detensorize
)
from datasets import Dataset
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union, Iterable

from tqdm import tqdm
import os
import torch
import re

import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.utils import logging
import torch.distributed as dist

logger = logging.get_logger(__name__)

@dataclass
class ReftDataCollator(object):
    """Collate examples for ReFT."""

    data_collator: DataCollator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch_inputs = self.data_collator(instances)
        max_seq_length = batch_inputs["input_ids"].shape[-1]
        batch_inputs["intervention_locations"] = batch_inputs["intervention_locations"][..., :max_seq_length]
        return batch_inputs


def make_data_collator(tokenizer, model) -> ReftDataCollator:
    data_collator_fn = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest",
        max_length=2048,
    )
    return ReftDataCollator(data_collator=data_collator_fn)


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    collate_fn: DataCollatorForSeq2Seq,
    shuffle: bool,
    sampler: Union[Sampler, Iterable, None]=None
) -> DataLoader:
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)


class ReftTrainer(Trainer):
    def save_model(self, output_dir, _internal_call=False, **kwargs):
        # Handle CPU training and non-distributed cases
        try:
            is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        except (RuntimeError, AttributeError) as e:  # Catches case when torch.distributed is not available or other dist errors
            logger.error(f"Error checking distributed training status: {str(e)}")
            is_main_process = True
        
        if is_main_process:
            target_dir = f"{output_dir}/intervenable_model"
            # Log warning if target directory exists and has content
            if os.path.exists(target_dir) and os.listdir(target_dir):
                logger.warning(
                    f"Directory {target_dir} already exists and contains files. "
                    "Skipping save to prevent overwriting existing model."
                )
                return
                
            try:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                self.model.save_intervention(
                    save_directory=target_dir, 
                    include_model=True
                )
            except Exception as e:
                logger.error(f"Error saving model to {target_dir}: {str(e)}")
                raise  # Re-raise the exception after logging

    def _load_best_model(self, **kwargs):
        logger.warning(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        self.model.load_intervention(
            f"{self.state.best_model_checkpoint}/intervenable_model", 
            include_model=True
        )
    
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None, **kwargs):
        if model is None:
            model = self.model

        logger.warning(f"Loading checkpoint from {resume_from_checkpoint}.")
        model.load_intervention(
            f"{resume_from_checkpoint}/intervenable_model", 
            include_model=True
        )

    def compute_loss(
        self,
        intervenable: pv.IntervenableModel,
        inputs,
        return_outputs=False,
        **kwargs
    ):
        # run intervened forward pass
        unit_locations = None
        if "intervention_locations" in inputs:
            if inputs["intervention_locations"].dim() == 3:
                unit_locations={"sources->base": (
                    None,
                    inputs["intervention_locations"].permute(1, 0, 2).tolist()
                )}
            else:
                # this is dummy for lora only baseline
                unit_locations={"sources->base": (None, 0)}
        base_outputs, cf_outputs = intervenable(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            },
            unit_locations=unit_locations,
            labels=inputs["labels"],
            subspaces=inputs["subspaces"].permute(1, 0, 2).tolist() if "subspaces" in inputs else None
        )
        # return
        output = cf_outputs
        if cf_outputs is None:
            output = base_outputs # in case of lora only training

        return (output, output) if return_outputs else output.loss

class ReftTrainerForCausalLM(ReftTrainer):
    def get_train_dataloader(self) -> DataLoader:
        return make_dataloader(self.train_dataset, self._train_batch_size, self.data_collator, shuffle=True)

class ReftTrainerForCausalLMDistributed(ReftTrainer):
    def save_model(self, output_dir, _internal_call=False):
        if dist.get_rank() == 0:
            super().save_model(output_dir, _internal_call)

    def get_train_dataloader(self) -> DataLoader:
        return make_dataloader(
            self.train_dataset,
            self._train_batch_size,
            self.data_collator,
            shuffle=False,
            sampler=DistributedSampler(self.train_dataset, shuffle=True),
        )
    
class ReftTrainerForSequenceClassification(ReftTrainer):
    def compute_loss(
        self,
        intervenable: pv.IntervenableModel,
        inputs,
        return_outputs=False
    ):
        # run intervened forward pass
        unit_locations = None
        if "intervention_locations" in inputs:
            unit_locations={"sources->base": (
                None,
                inputs["intervention_locations"].permute(1, 0, 2).tolist()
            )}
            
        _, cf_outputs = intervenable(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            },
            unit_locations=unit_locations,
            labels=inputs["labels"],
            subspaces=inputs["subspaces"].permute(1, 0, 2).tolist() if "subspaces" in inputs else None
        )
        # classification loss on counterfactual labels
        logits = cf_outputs.logits
        labels = inputs["labels"]

        if self.model.model.config.problem_type is None:
            if self.model.model.num_labels == 1:
                problem_type = "regression"
            elif self.model.model.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                problem_type = "single_label_classification"
            else:
                problem_type = "multi_label_classification"
        else:
            problem_type = self.model.model.config.problem_type
            
        if problem_type == "regression":
            loss_fct = MSELoss()
            if self.model.model.num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze().to(torch.bfloat16))
            else:
                loss = loss_fct(logits, labels.to(torch.bfloat16))
        elif problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.model.num_labels), labels.view(-1))
        elif problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        # return
        return (loss, cf_outputs) if return_outputs else loss
    
    def evaluate(
        self, ignore_keys,
    ):

        # ensure everything is in eval mode
        self.model.model.eval()
        for k,v in  self.model.interventions.items():
            _ = v[0].eval()
        
        batch_size = self.args.eval_batch_size
        data_collator = self.data_collator
        eval_dataset = self.eval_dataset
        intervenable = self.model
        
        dataloader = make_dataloader(
            eval_dataset, batch_size, data_collator, shuffle=False)

        logger.info(f"***** Running In-Training Evaluation *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        eval_iterator = tqdm(dataloader, position=0, leave=True)
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for step, inputs in enumerate(eval_iterator):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.model.get_device())
                
                # [layers, batch_size, positions]
                intervention_locations = inputs["intervention_locations"].permute(1, 0, 2).tolist()
                _, cf_outputs = intervenable(
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
                    unit_locations={"sources->base": (None, intervention_locations)})
            
                all_preds += [cf_outputs.logits]
                all_labels += [inputs["labels"]]
        all_preds = torch.cat(all_preds, dim=0).cpu().to(torch.float32)
        all_labels = torch.cat(all_labels, dim=0).cpu().to(torch.float32)
        metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        metrics = denumpify_detensorize(metrics)
        
        metric_key_prefix = "eval"
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)
        
        return metrics


class ReftTrainerForGRPO(ReftTrainer):
    """
    ReFT Trainer for Group Relative Policy Optimization (GRPO).
    
    GRPO is a reinforcement learning algorithm that computes advantages relative
    to other samples in the same group (same prompt). This trainer implements
    GRPO while using ReFT interventions as the policy's action space.
    
    Key features:
    - Group-relative advantage estimation (no separate value network needed)
    - Policy gradient loss with KL divergence penalty
    - Support for external reward functions
    - Only updates ReFT intervention parameters
    
    Args:
        model: The ReFT model (IntervenableModel)
        args: Training arguments
        data_collator: Data collator for GRPO batches
        train_dataset: Training dataset with prompts, responses, and rewards
        tokenizer: Tokenizer for the model
        beta: KL divergence penalty coefficient (default: 0.1)
        group_size: Number of samples per prompt group (default: inferred from data)
        epsilon: Small constant for numerical stability (default: 1e-8)
    """
    
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        beta: float = 0.1,
        group_size: Optional[int] = None,
        epsilon: float = 1e-8,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **kwargs
        )
        self.beta = beta
        self.group_size = group_size
        self.epsilon = epsilon
    
    def get_train_dataloader(self) -> DataLoader:
        """Create dataloader for GRPO training."""
        return make_dataloader(
            self.train_dataset, 
            self._train_batch_size, 
            self.data_collator, 
            shuffle=True
        )
    
    def _get_batch_logps(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probabilities of the labels given the logits.
        
        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            labels: Target token ids [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Log probabilities per sequence [batch_size]
        """
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        attention_mask = attention_mask[:, 1:].clone()
        
        # Replace padding tokens in labels with 0 to avoid index errors
        loss_mask = (labels != -100) & (attention_mask == 1)
        labels[labels == -100] = 0
        
        # Compute per-token log probabilities
        per_token_logps = torch.gather(
            logits.log_softmax(-1), 
            dim=2, 
            index=labels.unsqueeze(2)
        ).squeeze(2)
        
        # Mask and sum to get sequence log probabilities
        per_token_logps = per_token_logps * loss_mask
        return per_token_logps.sum(-1)
    
    def _compute_group_advantages(
        self,
        rewards: torch.Tensor,
        group_size: int
    ) -> torch.Tensor:
        """
        Compute group-relative advantages.
        
        For each group of samples (same prompt), compute advantages as:
        advantage = (reward - group_mean) / (group_std + epsilon)
        
        Args:
            rewards: Reward values [batch_size]
            group_size: Number of samples per group
            
        Returns:
            Normalized advantages [batch_size]
        """
        batch_size = rewards.shape[0]
        num_groups = batch_size // group_size
        
        # Reshape to [num_groups, group_size]
        rewards_grouped = rewards.view(num_groups, group_size)
        
        # Compute group statistics
        group_mean = rewards_grouped.mean(dim=1, keepdim=True)
        group_std = rewards_grouped.std(dim=1, keepdim=True)
        
        # Normalize within groups
        advantages = (rewards_grouped - group_mean) / (group_std + self.epsilon)
        
        # Flatten back to [batch_size]
        return advantages.view(-1)
    
    def compute_loss(
        self,
        intervenable: pv.IntervenableModel,
        inputs,
        return_outputs=False
    ):
        """
        Compute GRPO loss.
        
        The GRPO loss consists of:
        1. Policy gradient loss: -log_prob * advantage
        2. KL divergence penalty: beta * KL(policy || reference)
        
        Args:
            intervenable: The ReFT model
            inputs: Batch inputs containing:
                - input_ids: Token ids [batch_size, seq_len]
                - attention_mask: Attention mask [batch_size, seq_len]
                - labels: Target labels [batch_size, seq_len]
                - intervention_locations: Intervention positions
                - rewards: Reward values [batch_size]
                - group_ids: Group identifiers [batch_size] (optional)
            return_outputs: Whether to return model outputs
            
        Returns:
            Loss value (and optionally outputs)
        """
        # Get intervention locations
        unit_locations = None
        if "intervention_locations" in inputs:
            if inputs["intervention_locations"].dim() == 3:
                unit_locations = {"sources->base": (
                    None,
                    inputs["intervention_locations"].permute(1, 0, 2).tolist()
                )}
            else:
                unit_locations = {"sources->base": (None, 0)}
        
        # Forward pass with interventions (policy)
        _, cf_outputs = intervenable(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            },
            unit_locations=unit_locations,
            subspaces=inputs["subspaces"].permute(1, 0, 2).tolist() if "subspaces" in inputs else None
        )
        
        # Compute policy log probabilities
        policy_logps = self._get_batch_logps(
            cf_outputs.logits,
            inputs["labels"],
            inputs["attention_mask"]
        )
        
        # Get rewards and compute advantages
        rewards = inputs["rewards"].to(policy_logps.device)
        
        # Determine group size
        if self.group_size is not None:
            group_size = self.group_size
        elif "group_ids" in inputs:
            # Infer group size from group_ids
            unique_groups = inputs["group_ids"].unique()
            group_size = inputs["input_ids"].shape[0] // len(unique_groups)
        else:
            # Default: treat entire batch as one group
            group_size = inputs["input_ids"].shape[0]
        
        # Compute group-relative advantages
        advantages = self._compute_group_advantages(rewards, group_size)
        
        # Policy gradient loss: -log_prob * advantage
        pg_loss = -(policy_logps * advantages).mean()
        
        # KL divergence penalty (against reference/base model)
        # Compute reference log probabilities (without interventions)
        with torch.no_grad():
            base_outputs = intervenable.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            ref_logps = self._get_batch_logps(
                base_outputs.logits,
                inputs["labels"],
                inputs["attention_mask"]
            )
        
        # KL divergence: policy_logp - ref_logp (approximation)
        kl_div = (policy_logps - ref_logps).mean()
        
        # Total loss
        loss = pg_loss + self.beta * kl_div
        
        if return_outputs:
            return loss, {
                "cf_outputs": cf_outputs,
                "policy_logps": policy_logps,
                "ref_logps": ref_logps,
                "advantages": advantages,
                "pg_loss": pg_loss,
                "kl_div": kl_div,
            }
        return loss
    
    def prediction_step(
        self,
        model: pv.IntervenableModel,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        """
        Perform a prediction step for evaluation.
        
        Args:
            model: The ReFT model
            inputs: Batch inputs
            prediction_loss_only: Whether to only return loss
            ignore_keys: Keys to ignore in outputs
            
        Returns:
            Tuple of (loss, logits, labels)
        """
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        
        loss = loss.detach().cpu()
        
        if prediction_loss_only:
            return (loss, None, None)
        
        # Return policy log probs as logits for metrics computation
        logits = outputs["policy_logps"].detach().cpu()
        labels = inputs["rewards"].detach().cpu()
        
        return (loss, logits, labels)
        
