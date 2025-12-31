"""ReFT model wrapper for applying interventions to transformer models.

This module provides the ReftModel class, which wraps a base transformer model
with ReFT interventions. It extends pyvene's IntervenableModel to provide
ReFT-specific functionality for training and inference.
"""

import pyvene as pv
from typing import Any, Optional, Tuple


def count_parameters(model: pv.TrainableIntervention) -> int:
    """Count the number of trainable parameters in a model or intervention.

    Args:
        model: A PyTorch module (typically a TrainableIntervention) to count
            parameters for.

    Returns:
        The total number of parameters that require gradients.

    Example:
        >>> intervention = LoreftIntervention(embed_dim=4096, low_rank_dimension=4)
        >>> num_params = count_parameters(intervention)
        >>> print(f"Trainable params: {num_params:,d}")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ReftModel(pv.IntervenableModel):
    """Wrapper model for Representation Fine-Tuning (ReFT) methods.

    ReftModel extends pyvene's IntervenableModel to provide a unified interface
    for training and inference with ReFT interventions. It wraps a base transformer
    model and applies learned interventions at specified layers and positions.

    The model supports:
        - Training ReFT interventions while keeping base model weights frozen
        - Saving and loading trained interventions separately from the base model
        - Generating text with interventions applied during inference
        - Combining multiple interventions across different layers

    Args:
        config: A ReftConfig specifying the intervention configuration.
        model: The base transformer model to wrap (e.g., from HuggingFace).
        **kwargs: Additional arguments passed to pyvene.IntervenableModel.

    Attributes:
        model: The wrapped base transformer model.
        interventions: Dictionary mapping intervention keys to intervention modules.
        config: The ReftConfig used to create this model.

    Example:
        Creating and using a ReftModel::

            import pyreft
            import transformers

            # Load base model
            model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

            # Create ReFT config
            reft_config = pyreft.ReftConfig(
                representations={
                    "layer": 15,
                    "component": "block_output",
                    "low_rank_dimension": 4,
                    "intervention": pyreft.LoreftIntervention(
                        embed_dim=model.config.hidden_size,
                        low_rank_dimension=4
                    )
                }
            )

            # Create ReFT model
            reft_model = pyreft.get_reft_model(model, reft_config)
            reft_model.print_trainable_parameters()

            # Generate with interventions
            output = reft_model.generate(
                inputs,
                unit_locations={"sources->base": (None, [[[position]]])},
                intervene_on_prompt=True
            )

    See Also:
        - ReftConfig: Configuration class for specifying interventions.
        - get_reft_model: Factory function for creating ReftModel instances.
        - pyvene.IntervenableModel: Parent class providing core intervention logic.
    """

    def __init__(self, config: "ReftConfig", model: Any, **kwargs: Any) -> None:
        super().__init__(config, model, **kwargs)

    @staticmethod
    def _convert_to_reft_model(intervenable_model: pv.IntervenableModel) -> "ReftModel":
        """Convert a pyvene IntervenableModel to a ReftModel.

        This internal method is used when loading saved ReFT models to ensure
        the returned object is a ReftModel instance with all ReFT-specific
        functionality.

        Args:
            intervenable_model: A pyvene IntervenableModel to convert.

        Returns:
            A ReftModel instance with all attributes copied from the input model.
        """
        reft_model = ReftModel(intervenable_model.config, intervenable_model.model)
        for attr in vars(intervenable_model):
            setattr(reft_model, attr, getattr(intervenable_model, attr))
        return reft_model

    @staticmethod
    def load(*args: Any, **kwargs: Any) -> "ReftModel":
        """Load a saved ReftModel from disk or HuggingFace Hub.

        This method loads a previously saved ReFT model, including the intervention
        weights and configuration. The base model must be provided separately.

        Args:
            *args: Positional arguments passed to pyvene.IntervenableModel.load().
                Typically includes the save directory path and the base model.
            **kwargs: Keyword arguments passed to pyvene.IntervenableModel.load().

        Returns:
            A ReftModel instance with loaded intervention weights.

        Example:
            >>> base_model = transformers.AutoModelForCausalLM.from_pretrained(
            ...     "meta-llama/Llama-2-7b-hf"
            ... )
            >>> reft_model = ReftModel.load("./saved_reft", base_model)
        """
        model = pv.IntervenableModel.load(*args, **kwargs)
        return ReftModel._convert_to_reft_model(model)

    def print_trainable_parameters(self) -> None:
        """Print a summary of trainable parameters in the model.

        This method prints statistics about the number of trainable parameters
        in both the intervention modules and the base model, helping users
        understand the parameter efficiency of their ReFT setup.

        The output includes:
            - Number of trainable intervention parameters
            - Number of trainable base model parameters (usually 0 for ReFT)
            - Total base model parameters
            - Percentage of parameters that are trainable

        Example:
            >>> reft_model.print_trainable_parameters()
            trainable intervention params: 32,772 || trainable model params: 0
            model params: 6,738,415,616 || trainable%: 0.00048634578018881287
        """
        _linked_key_set = set([])
        trainable_intervention_parameters = 0
        for k, v in self.interventions.items():
            if isinstance(v, pv.TrainableIntervention):
                if k in self._intervention_reverse_link:
                    if not self._intervention_reverse_link[k] in _linked_key_set:
                        _linked_key_set.add(self._intervention_reverse_link[k])
                        trainable_intervention_parameters += count_parameters(v)
                else:
                    trainable_intervention_parameters += count_parameters(v)

        trainable_model_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)

        all_model_parameters = sum(
            p.numel() for p in self.model.parameters())

        total_trainable_parameters = trainable_intervention_parameters + trainable_model_parameters
        
        print(
            f"trainable intervention params: {trainable_intervention_parameters:,d} || trainable model params: {trainable_model_parameters:,d}\n"
            f"model params: {all_model_parameters:,d} || trainable%: {100 * total_trainable_parameters / all_model_parameters}"
        )

