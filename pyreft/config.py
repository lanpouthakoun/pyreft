import pyvene as pv
import json
from typing import Any, Dict, List, Optional, Union


class ReftConfig(pv.IntervenableConfig):
    """Configuration class for Representation Fine-Tuning (ReFT) methods.

    ReftConfig extends pyvene's IntervenableConfig to provide a simplified interface
    for configuring ReFT interventions on transformer models. It specifies where and
    how interventions should be applied during training and inference.

    This class inherits all functionality from pyvene.IntervenableConfig and can be
    used to configure single or multiple interventions across different layers and
    components of a transformer model.

    Args:
        representations: A dictionary or list of dictionaries specifying intervention
            configurations. Each dictionary should contain:
            - layer (int): The transformer layer index to intervene on.
            - component (str): The component to intervene on (e.g., "block_output",
              "mlp_output", or a custom path like "model.layers[0].output").
            - low_rank_dimension (int): The rank of the low-rank intervention.
            - intervention: An intervention instance (e.g., LoreftIntervention).
        **kwargs: Additional keyword arguments passed to pyvene.IntervenableConfig,
            including:
            - model_type (str, optional): The type of model being intervened on.
            - sorted_keys (list, optional): Keys for ordering interventions.
            - intervention_types (list, optional): Types of interventions to apply.

    Example:
        Basic usage with a single intervention::

            import pyreft

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

        Multiple interventions across layers::

            reft_config = pyreft.ReftConfig(
                representations=[
                    {
                        "layer": l,
                        "component": "block_output",
                        "low_rank_dimension": 4,
                        "intervention": pyreft.LoreftIntervention(
                            embed_dim=model.config.hidden_size,
                            low_rank_dimension=4
                        )
                    }
                    for l in [10, 15, 20]
                ]
            )

    See Also:
        - pyvene.IntervenableConfig: The parent configuration class.
        - LoreftIntervention: The most commonly used intervention type.
        - get_reft_model: Function to create a ReftModel from this config.
    """

    def __init__(
        self, **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
