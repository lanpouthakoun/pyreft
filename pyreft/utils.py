"""Utility functions and enums for Representation Fine-Tuning (ReFT).

This module provides utility functions for creating ReFT models and enums
for specifying ReFT types and task types.

Functions:
    get_reft_model: Factory function for creating ReftModel instances.

Classes:
    ReftType: Enum for different ReFT intervention types.
    TaskType: Enum for different task types supported by ReFT.
"""

import enum
from typing import Any, Optional
from .reft_model import ReftModel


class ReftType(str, enum.Enum):
    """Enum class for the different types of ReFT interventions.

    This enum defines the available intervention types that can be used
    with ReFT models. Each type corresponds to a different intervention
    class in the interventions module.

    Attributes:
        LOREFT: Low-rank ReFT with orthogonal rotation (LoreftIntervention).
        NLOREFT: Low-rank ReFT without orthogonal constraint (NoreftIntervention).

    Example:
        >>> reft_type = ReftType.LOREFT
        >>> print(reft_type.value)
        'LOREFT'
    """

    LOREFT = "LOREFT"
    NLOREFT = "NOREFT"


class TaskType(str, enum.Enum):
    """Enum class for the different types of tasks supported by ReFT.

    This enum defines the task types that determine which trainer class
    should be used for training ReFT models.

    Attributes:
        SEQ_CLS: Sequence classification tasks (e.g., sentiment analysis, NLI).
            Use with ReftTrainerForSequenceClassification.
        CAUSAL_LM: Causal language modeling tasks (e.g., instruction following, chat).
            Use with ReftTrainerForCausalLM.

    Example:
        >>> task_type = TaskType.CAUSAL_LM
        >>> print(task_type.value)
        'CAUSAL_LM'
    """

    SEQ_CLS = "SEQ_CLS"
    CAUSAL_LM = "CAUSAL_LM"


def get_reft_model(
    model: Any, 
    reft_config: "ReftConfig", 
    set_device: bool = True, 
    disable_model_grads: bool = True
) -> ReftModel:
    """Create a ReftModel instance from a base model and configuration.

    This is the primary factory function for creating ReFT models. It wraps
    a base transformer model with the specified ReFT interventions and
    optionally configures the device and gradient settings.

    Args:
        model: The base transformer model to wrap (e.g., from HuggingFace).
        reft_config: A ReftConfig specifying the intervention configuration.
        set_device: Whether to set the ReftModel's device to match the base model.
            Defaults to True.
        disable_model_grads: Whether to disable gradients for the base model
            parameters, making only intervention parameters trainable.
            Defaults to True.

    Returns:
        A configured ReftModel instance ready for training or inference.

    Example:
        Basic usage::

            import pyreft
            import transformers

            # Load base model
            model = transformers.AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-2-7b-hf",
                torch_dtype=torch.bfloat16,
                device_map="cuda"
            )

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

        Training with LoRA (keeping base model gradients)::

            reft_model = pyreft.get_reft_model(
                model, 
                reft_config, 
                disable_model_grads=False
            )

    See Also:
        - ReftConfig: Configuration class for specifying interventions.
        - ReftModel: The model class returned by this function.
    """
    reft_model = ReftModel(reft_config, model)
    if set_device:
        reft_model.set_device(model.device)
    if disable_model_grads:
        reft_model.disable_model_gradients()    
    return reft_model
