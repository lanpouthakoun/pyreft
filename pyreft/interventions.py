"""Intervention modules for Representation Fine-Tuning (ReFT).

This module provides various intervention implementations that can be applied to
transformer model representations during training and inference. Each intervention
type offers different trade-offs between parameter efficiency and expressiveness.

The interventions are designed to work with the pyvene library and can be used
with ReftConfig to create ReftModel instances.

Classes:
    LowRankRotateLayer: A linear transformation with orthogonal initialization.
    LoreftIntervention: Low-rank linear subspace ReFT with orthogonal rotation.
    NoreftIntervention: Low-rank ReFT without orthogonal constraint.
    ConsreftIntervention: Constant source ReFT (bias-only intervention).
    LobireftIntervention: Low-rank bitfit ReFT variant.
    DireftIntervention: Direct ReFT with orthogonal rotation.
    NodireftIntervention: Direct ReFT without orthogonal constraint.
"""

import torch
from collections import OrderedDict
from typing import Optional, Dict, Any

from pyvene import (
    ConstantSourceIntervention,
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
)
from transformers.activations import ACT2FN


class LowRankRotateLayer(torch.nn.Module):
    """A linear transformation layer with orthogonal weight initialization.

    This layer implements a low-rank projection that maps from a high-dimensional
    space (n) to a lower-dimensional space (m). The weight matrix is initialized
    using orthogonal initialization to preserve gradient flow and ensure the
    projection captures meaningful directions in the representation space.

    This layer is a core component of LoReFT and other orthogonal ReFT variants,
    where it defines the subspace in which interventions are applied.

    Args:
        n: Input dimension (embedding dimension of the model).
        m: Output dimension (low-rank dimension for the intervention).
        init_orth: Whether to use orthogonal initialization. Defaults to True.
            When True, the weight matrix columns form an orthonormal basis.

    Attributes:
        weight: Learnable weight matrix of shape (n, m).

    Example:
        >>> rotate_layer = LowRankRotateLayer(n=4096, m=4)
        >>> x = torch.randn(32, 128, 4096)  # (batch, seq, embed_dim)
        >>> projected = rotate_layer(x)  # (batch, seq, 4)
    """

    def __init__(self, n: int, m: int, init_orth: bool = True) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(n, m), requires_grad=True)
        if init_orth:
            torch.nn.init.orthogonal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input tensor to lower-dimensional space.

        Args:
            x: Input tensor of shape (..., n) where n is the input dimension.

        Returns:
            Projected tensor of shape (..., m) where m is the output dimension.
        """
        return torch.matmul(x.to(self.weight.dtype), self.weight)


class LoreftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """Low-rank Linear Subspace ReFT (LoReFT) intervention with orthogonal rotation.

    LoReFT is the primary intervention method in the ReFT family. It learns to modify
    model representations within a low-rank orthogonal subspace, achieving strong
    performance with minimal trainable parameters (typically ~0.001% of model size).

    The intervention formula is: LoReFT(h) = h + R^T(Wh + b - Rh)

    Where:
        - h: Original hidden representation
        - R: Orthogonal rotation matrix (projects to low-rank subspace)
        - W, b: Learned linear transformation parameters
        - R^T: Transpose of R (projects back to full space)

    This formulation ensures that modifications are constrained to a learned subspace,
    providing both parameter efficiency and interpretability.

    Args:
        embed_dim (int): The embedding dimension of the model's hidden states.
        low_rank_dimension (int): The rank of the intervention subspace.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        dtype (torch.dtype, optional): Data type for parameters. Defaults to torch.bfloat16.
        act_fn (str, optional): Activation function name from transformers.activations.
            Defaults to "linear" (no activation).
        **kwargs: Additional arguments passed to parent intervention classes.

    Attributes:
        rotate_layer: Orthogonally-constrained rotation layer.
        learned_source: Linear layer for computing the intervention.
        dropout: Dropout layer for regularization.
        act_fn: Activation function applied to learned source.

    Example:
        >>> intervention = LoreftIntervention(
        ...     embed_dim=4096,
        ...     low_rank_dimension=4,
        ...     dropout=0.05
        ... )
        >>> h = torch.randn(32, 128, 4096)  # (batch, seq, embed_dim)
        >>> h_modified = intervention(h)  # Same shape as input
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base: torch.Tensor, source: Optional[torch.Tensor] = None, 
        subspaces: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply the LoReFT intervention to the input representation.

        Args:
            base: Input hidden states of shape (batch, seq_len, embed_dim).
            source: Unused. Present for API compatibility with pyvene.
            subspaces: Optional subspace indices for compositional interventions.

        Returns:
            Modified hidden states with the same shape as input.
        """
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - rotated_base), self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args: Any, **kwargs: Any) -> OrderedDict:
        """Return a minimal state dict containing only intervention parameters.

        This method is overridden for data efficiency, storing only the learned
        source parameters and rotation layer weights rather than the full model state.

        Returns:
            OrderedDict containing learned_source parameters and rotate_layer weights.
        """
        state_dict = OrderedDict()
        for k, v in self.learned_source.state_dict().items():
            state_dict[k] = v
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], *args: Any, **kwargs: Any) -> None:
        """Load intervention parameters from a state dict.

        This method handles the special loading requirements for the orthogonally-
        constrained rotation layer, which requires recreating the layer to properly
        restore the parametrization.

        Args:
            state_dict: Dictionary containing learned_source parameters and
                rotate_layer weights.
        """
        self.learned_source.load_state_dict(state_dict, strict=False)

        overload_w = state_dict["rotate_layer"].to(
            self.learned_source.weight.device)
        overload_w_width = overload_w.shape[-1]
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, overload_w_width, init_orth=True).to(
            self.learned_source.weight.device)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.rotate_layer.parametrizations.weight[0].base[:,:overload_w_width] = overload_w
        assert torch.allclose(self.rotate_layer.weight.data, overload_w.data) == True
        
        return


class NoreftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """Low-rank ReFT intervention without orthogonal constraint (NoReFT).

    NoReFT is a variant of LoReFT that removes the orthogonality constraint on the
    projection layer. This allows for more flexible learned projections at the cost
    of potentially less interpretable subspaces.

    The intervention formula is: NoReFT(h) = h + W2^T(W1h + b - W2h)

    Where:
        - h: Original hidden representation
        - W1, b: Learned linear transformation for computing the intervention
        - W2: Learned projection layer (not constrained to be orthogonal)
        - W2^T: Transpose of W2 (projects back to full space)

    Args:
        embed_dim (int): The embedding dimension of the model's hidden states.
        low_rank_dimension (int): The rank of the intervention subspace.
        add_bias (bool): Whether to include bias in the projection layer.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        dtype (torch.dtype, optional): Data type for parameters. Defaults to torch.bfloat16.
        act_fn (str, optional): Activation function name. Defaults to "linear".
        **kwargs: Additional arguments passed to parent intervention classes.

    Attributes:
        proj_layer: Linear projection layer (non-orthogonal).
        learned_source: Linear layer for computing the intervention.
        dropout: Dropout layer for regularization.
        act_fn: Activation function applied to learned source.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj_layer = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=kwargs["add_bias"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base: torch.Tensor, source: Optional[torch.Tensor] = None, 
        subspaces: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply the NoReFT intervention to the input representation.

        Args:
            base: Input hidden states of shape (batch, seq_len, embed_dim).
            source: Unused. Present for API compatibility with pyvene.
            subspaces: Optional subspace indices for compositional interventions.

        Returns:
            Modified hidden states with the same shape as input.
        """
        proj_base = self.proj_layer(base)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - proj_base), self.proj_layer.weight
        )
        return self.dropout(output.to(base.dtype))


class ConsreftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """Constant Source ReFT intervention (ConsReFT).

    ConsReFT is a minimal ReFT variant that uses only a learned constant bias vector
    instead of a learned linear transformation. This results in the fewest trainable
    parameters among ReFT variants, making it useful for extremely parameter-efficient
    fine-tuning or as a baseline.

    The intervention formula is: ConsReFT(h) = h + R^T(b - Rh)

    Where:
        - h: Original hidden representation
        - R: Orthogonal rotation matrix (projects to low-rank subspace)
        - b: Learned constant bias vector in the low-rank subspace
        - R^T: Transpose of R (projects back to full space)

    Args:
        embed_dim (int): The embedding dimension of the model's hidden states.
        low_rank_dimension (int): The rank of the intervention subspace.
        **kwargs: Additional arguments passed to parent intervention classes.

    Attributes:
        rotate_layer: Orthogonally-constrained rotation layer.
        learned_source: Learned constant bias parameter of shape (low_rank_dimension,).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Parameter(
            torch.rand(kwargs["low_rank_dimension"]), requires_grad=True)
        
    def forward(
        self, base: torch.Tensor, source: Optional[torch.Tensor] = None, 
        subspaces: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply the ConsReFT intervention to the input representation.

        Args:
            base: Input hidden states of shape (batch, seq_len, embed_dim).
            source: Unused. Present for API compatibility with pyvene.
            subspaces: Optional subspace indices for compositional interventions.

        Returns:
            Modified hidden states with the same shape as input.
        """
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.learned_source - rotated_base), self.rotate_layer.weight.T
        )
        return output.to(base.dtype)


class LobireftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """Low-rank Bitfit ReFT intervention (LobiReFT).

    LobiReFT is inspired by the BitFit method and adds a learned bias in a low-rank
    subspace. Unlike ConsReFT, it does not subtract the projected base representation,
    making it a pure additive bias intervention.

    The intervention formula is: LobiReFT(h) = h + R^T(b)

    Where:
        - h: Original hidden representation
        - R: Orthogonal rotation matrix (defines the subspace)
        - b: Learned constant bias vector in the low-rank subspace
        - R^T: Transpose of R (projects bias back to full space)

    This is the simplest additive intervention, adding a constant direction to all
    representations at the intervened positions.

    Args:
        embed_dim (int): The embedding dimension of the model's hidden states.
        low_rank_dimension (int): The rank of the intervention subspace.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        **kwargs: Additional arguments passed to parent intervention classes.

    Attributes:
        rotate_layer: Orthogonally-constrained rotation layer.
        learned_source: Learned constant bias parameter of shape (low_rank_dimension,).
        dropout: Dropout layer for regularization.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Parameter(
            torch.rand(kwargs["low_rank_dimension"]), requires_grad=True)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        
    def forward(
        self, base: torch.Tensor, source: Optional[torch.Tensor] = None, 
        subspaces: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply the LobiReFT intervention to the input representation.

        Args:
            base: Input hidden states of shape (batch, seq_len, embed_dim).
            source: Unused. Present for API compatibility with pyvene.
            subspaces: Optional subspace indices for compositional interventions.

        Returns:
            Modified hidden states with the same shape as input.
        """
        output = base + torch.matmul(
            self.learned_source, self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))


class DireftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """Direct ReFT intervention with orthogonal rotation (DiReFT).

    DiReFT directly adds a learned transformation of the input to the representation,
    without subtracting the projected base. This makes it a purely additive intervention
    that learns input-dependent modifications.

    The intervention formula is: DiReFT(h) = h + R^T(Wh + b)

    Where:
        - h: Original hidden representation
        - W, b: Learned linear transformation parameters
        - R: Orthogonal rotation matrix (defines the subspace)
        - R^T: Transpose of R (projects back to full space)

    Unlike LoReFT, DiReFT does not subtract the projected base representation,
    making the intervention purely additive rather than a replacement within the subspace.

    Args:
        embed_dim (int): The embedding dimension of the model's hidden states.
        low_rank_dimension (int): The rank of the intervention subspace.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        dtype (torch.dtype, optional): Data type for parameters. Defaults to torch.bfloat16.
        act_fn (str, optional): Activation function name. Defaults to "linear".
        **kwargs: Additional arguments passed to parent intervention classes.

    Attributes:
        rotate_layer: Orthogonally-constrained rotation layer.
        learned_source: Linear layer for computing the intervention.
        dropout: Dropout layer for regularization.
        act_fn: Activation function applied to learned source.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base: torch.Tensor, source: Optional[torch.Tensor] = None, 
        subspaces: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply the DiReFT intervention to the input representation.

        Args:
            base: Input hidden states of shape (batch, seq_len, embed_dim).
            source: Unused. Present for API compatibility with pyvene.
            subspaces: Optional subspace indices for compositional interventions.

        Returns:
            Modified hidden states with the same shape as input.
        """
        cast_base = base.to(self.learned_source.weight.dtype)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(cast_base))).to(self.rotate_layer.weight.dtype), self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))


class NodireftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """Direct ReFT intervention without orthogonal constraint (NodiReFT).

    NodiReFT is a variant of DiReFT that removes the orthogonality constraint on the
    projection layer. This allows for more flexible learned projections at the cost
    of potentially less interpretable subspaces.

    The intervention formula is: NodiReFT(h) = h + W2^T(W1h + b)

    Where:
        - h: Original hidden representation
        - W1, b: Learned linear transformation for computing the intervention
        - W2: Learned projection layer (not constrained to be orthogonal)
        - W2^T: Transpose of W2 (projects back to full space)

    This is the non-orthogonal variant of DiReFT, offering more flexibility
    in the learned projection at the cost of interpretability.

    Args:
        embed_dim (int): The embedding dimension of the model's hidden states.
        low_rank_dimension (int): The rank of the intervention subspace.
        add_bias (bool): Whether to include bias in the projection layer.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        dtype (torch.dtype, optional): Data type for parameters. Defaults to torch.bfloat16.
        act_fn (str, optional): Activation function name. Defaults to "linear".
        **kwargs: Additional arguments passed to parent intervention classes.

    Attributes:
        proj_layer: Linear projection layer (non-orthogonal).
        learned_source: Linear layer for computing the intervention.
        dropout: Dropout layer for regularization.
        act_fn: Activation function applied to learned source.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj_layer = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=kwargs["add_bias"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base: torch.Tensor, source: Optional[torch.Tensor] = None, 
        subspaces: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply the NodiReFT intervention to the input representation.

        Args:
            base: Input hidden states of shape (batch, seq_len, embed_dim).
            source: Unused. Present for API compatibility with pyvene.
            subspaces: Optional subspace indices for compositional interventions.

        Returns:
            Modified hidden states with the same shape as input.
        """
        output = base + torch.matmul(
            self.act_fn(self.learned_source(base)), self.proj_layer.weight
        )
        return self.dropout(output.to(base.dtype))

