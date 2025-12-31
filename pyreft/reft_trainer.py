"""Training utilities for Representation Fine-Tuning (ReFT) models.

This module provides specialized trainer classes for training ReFT interventions
on transformer models. The trainers extend HuggingFace's Trainer class to handle
the unique requirements of ReFT training, including intervention-aware loss
computation and model saving.

Classes:
    ReftDataCollator: Data collator that handles intervention locations.
    ReftTrainer: Base trainer class for ReFT models.
    ReftTrainerForCausalLM: Trainer for causal language modeling tasks.
    ReftTrainerForCausalLMDistributed: Distributed training variant.
    ReftTrainerForSequenceClassification: Trainer for classification tasks.

Functions:
    make_data_collator: Create a ReftDataCollator for a model.
    make_dataloader: Create a DataLoader for ReFT training.
"""

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
from typing import Dict, Optional, Sequence, Union, Iterable, Any, Tuple

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
class ReftDataCollator:
    """Data collator that handles intervention locations for ReFT training.

    This collator wraps a standard HuggingFace data collator and ensures that
    intervention_locations are properly truncated to match the maximum sequence
    length in each batch. This is necessary because intervention locations may
    extend beyond the actual sequence length after padding.

    Args:
        data_collator: The underlying data collator (typically DataCollatorForSeq2Seq)
            that handles standard collation tasks like padding.

    Attributes:
        data_collator: The wrapped data collator instance.

    Example:
        >>> from transformers import DataCollatorForSeq2Seq
        >>> base_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
        >>> reft_collator = ReftDataCollator(data_collator=base_collator)
    """

    data_collator: DataCollator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of examples for ReFT training.

        Args:
            instances: A sequence of dictionaries, each containing input_ids,
                attention_mask, labels, and intervention_locations.

        Returns:
            A dictionary containing batched and padded tensors, with
            intervention_locations truncated to match the sequence length.
        """
        batch_inputs = self.data_collator(instances)
        max_seq_length = batch_inputs["input_ids"].shape[-1]
        batch_inputs["intervention_locations"] = batch_inputs["intervention_locations"][..., :max_seq_length]
        return batch_inputs


def make_data_collator(tokenizer: Any, model: Any) -> ReftDataCollator:
    """Create a ReftDataCollator for the given tokenizer and model.

    This is a convenience function that creates a properly configured
    ReftDataCollator with a DataCollatorForSeq2Seq as the underlying collator.

    Args:
        tokenizer: The tokenizer to use for padding.
        model: The model (used for determining padding behavior).

    Returns:
        A configured ReftDataCollator instance.

    Example:
        >>> collator = make_data_collator(tokenizer, model)
        >>> dataloader = DataLoader(dataset, collate_fn=collator)
    """
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
    sampler: Union[Sampler, Iterable, None] = None
) -> DataLoader:
    """Create a DataLoader for ReFT training.

    Args:
        dataset: The dataset to load from.
        batch_size: Number of samples per batch.
        collate_fn: The collation function (typically a ReftDataCollator).
        shuffle: Whether to shuffle the data. Ignored if sampler is provided.
        sampler: Optional sampler for distributed training.

    Returns:
        A configured DataLoader instance.

    Example:
        >>> dataloader = make_dataloader(
        ...     dataset=train_dataset,
        ...     batch_size=8,
        ...     collate_fn=reft_collator,
        ...     shuffle=True
        ... )
    """
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)


class ReftTrainer(Trainer):
    """Base trainer class for Representation Fine-Tuning (ReFT) models.

    ReftTrainer extends HuggingFace's Trainer to handle the unique requirements
    of training ReFT interventions. It overrides model saving, loading, and loss
    computation to work with pyvene's IntervenableModel.

    Key differences from standard Trainer:
        - Saves only intervention weights, not the full model
        - Computes loss through the intervened forward pass
        - Handles intervention locations in the input batch

    This is the base class; use ReftTrainerForCausalLM or
    ReftTrainerForSequenceClassification for specific task types.

    Args:
        model: A ReftModel instance (IntervenableModel with interventions).
        args: TrainingArguments for configuring training.
        data_collator: A ReftDataCollator for batching examples.
        train_dataset: The training dataset.
        eval_dataset: Optional evaluation dataset.
        tokenizer: The tokenizer (for saving with the model).
        **kwargs: Additional arguments passed to HuggingFace Trainer.

    Example:
        >>> trainer = ReftTrainer(
        ...     model=reft_model,
        ...     tokenizer=tokenizer,
        ...     args=training_args,
        ...     **data_module
        ... )
        >>> trainer.train()
    """

    def save_model(self, output_dir: str, _internal_call: bool = False, **kwargs: Any) -> None:
        """Save the ReFT intervention weights to disk.

        This method saves only the intervention parameters, not the full base model.
        The saved interventions can be loaded later with ReftModel.load().

        Args:
            output_dir: Directory to save the model to.
            _internal_call: Internal flag used by HuggingFace Trainer.
            **kwargs: Additional arguments (unused).

        Note:
            In distributed training, only the main process (rank 0) saves the model.
            The method will skip saving if the target directory already exists to
            prevent accidental overwrites.
        """
        try:
            is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        except (RuntimeError, AttributeError) as e:
            logger.error(f"Error checking distributed training status: {str(e)}")
            is_main_process = True
        
        if is_main_process:
            target_dir = f"{output_dir}/intervenable_model"
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
                raise

    def _load_best_model(self, **kwargs: Any) -> None:
        """Load the best model checkpoint during training.

        This method is called by the Trainer when load_best_model_at_end=True
        to restore the best performing checkpoint after training completes.
        """
        logger.warning(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        self.model.load_intervention(
            f"{self.state.best_model_checkpoint}/intervenable_model", 
            include_model=True
        )
    
    def _load_from_checkpoint(self, resume_from_checkpoint: str, model: Optional[Any] = None, **kwargs: Any) -> None:
        """Load model state from a checkpoint for resuming training.

        Args:
            resume_from_checkpoint: Path to the checkpoint directory.
            model: Optional model to load into. Uses self.model if not provided.
            **kwargs: Additional arguments (unused).
        """
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
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """Compute the training loss with interventions applied.

        This method runs the intervened forward pass and computes the loss.
        It handles extracting intervention locations from the input batch
        and applying them during the forward pass.

        Args:
            intervenable: The ReftModel (IntervenableModel) to compute loss for.
            inputs: Dictionary containing input_ids, attention_mask, labels,
                and intervention_locations.
            return_outputs: Whether to return model outputs along with loss.
            **kwargs: Additional arguments (unused).

        Returns:
            If return_outputs is False, returns the loss tensor.
            If return_outputs is True, returns a tuple of (loss, outputs).
        """
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
    """Trainer for causal language modeling tasks with ReFT.

    This trainer is optimized for autoregressive language modeling tasks like
    instruction following, text generation, and chat. It uses a shuffled
    DataLoader for training.

    Example:
        >>> trainer = ReftTrainerForCausalLM(
        ...     model=reft_model,
        ...     tokenizer=tokenizer,
        ...     args=training_args,
        ...     **data_module
        ... )
        >>> trainer.train()
    """

    def get_train_dataloader(self) -> DataLoader:
        """Create the training DataLoader with shuffling enabled."""
        return make_dataloader(self.train_dataset, self._train_batch_size, self.data_collator, shuffle=True)


class ReftTrainerForCausalLMDistributed(ReftTrainer):
    """Distributed trainer for causal language modeling tasks with ReFT.

    This trainer extends ReftTrainerForCausalLM with support for distributed
    training across multiple GPUs using PyTorch's DistributedDataParallel.
    It uses a DistributedSampler to ensure each GPU processes different data.

    Example:
        >>> trainer = ReftTrainerForCausalLMDistributed(
        ...     model=reft_model,
        ...     tokenizer=tokenizer,
        ...     args=training_args,
        ...     **data_module
        ... )
        >>> trainer.train()
    """

    def save_model(self, output_dir: str, _internal_call: bool = False) -> None:
        """Save model only on rank 0 to avoid conflicts in distributed training."""
        if dist.get_rank() == 0:
            super().save_model(output_dir, _internal_call)

    def get_train_dataloader(self) -> DataLoader:
        """Create the training DataLoader with DistributedSampler."""
        return make_dataloader(
            self.train_dataset,
            self._train_batch_size,
            self.data_collator,
            shuffle=False,
            sampler=DistributedSampler(self.train_dataset, shuffle=True),
        )
    

class ReftTrainerForSequenceClassification(ReftTrainer):
    """Trainer for sequence classification tasks with ReFT.

    This trainer is designed for classification tasks like sentiment analysis,
    natural language inference, and other GLUE-style tasks. It handles
    regression, single-label, and multi-label classification automatically
    based on the model configuration.

    The trainer computes appropriate loss functions:
        - MSELoss for regression tasks
        - CrossEntropyLoss for single-label classification
        - BCEWithLogitsLoss for multi-label classification

    Example:
        >>> trainer = ReftTrainerForSequenceClassification(
        ...     model=reft_model,
        ...     tokenizer=tokenizer,
        ...     args=training_args,
        ...     compute_metrics=compute_metrics_fn,
        ...     **data_module
        ... )
        >>> trainer.train()
        >>> metrics = trainer.evaluate()
    """

    def compute_loss(
        self,
        intervenable: pv.IntervenableModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """Compute classification loss with interventions applied.

        This method handles regression, single-label, and multi-label classification
        by automatically selecting the appropriate loss function based on the
        model's configuration.

        Args:
            intervenable: The ReftModel to compute loss for.
            inputs: Dictionary containing input_ids, attention_mask, labels,
                and intervention_locations.
            return_outputs: Whether to return model outputs along with loss.

        Returns:
            If return_outputs is False, returns the loss tensor.
            If return_outputs is True, returns a tuple of (loss, outputs).
        """
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

        return (loss, cf_outputs) if return_outputs else loss
    
    def evaluate(
        self, ignore_keys: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Run evaluation on the evaluation dataset.

        This method runs the model in evaluation mode with interventions applied,
        collecting predictions and computing metrics using the provided
        compute_metrics function.

        Args:
            ignore_keys: Keys to ignore in the output (unused, for API compatibility).

        Returns:
            Dictionary containing evaluation metrics with 'eval_' prefix.
        """
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
        
