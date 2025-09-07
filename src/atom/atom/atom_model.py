from typing import final, override

import torch
import torch.nn as nn
from atom.atom.activations import ReLU2, SwiGLU
from atom.training.config_options import FFNActivation, NormType, ValueResidualType, AttentionType, LiftingType, PositionalEncodingType, ProjectionType
from tensordict import TensorDict
from atom.atom.attentions import QuadraticHeterogenousCrossAttention, QuadraticSelfAttention, LinearHeterogenousCrossAttention
from atom.atom.mlps import MLP
from atom.atom.lifting_layers import StandardLift, EquivariantLift, EquivariantLiftTensorProduct, CanonicalizationLift
from atom.atom.projection_layers import EquivariantProject, EquivariantMoEProject, DecanonicalizationProject
import math


@final
class ATOMBlock(nn.Module):
    def __init__(
        self,
        lifting_dim: int,
        norm: NormType,
        activation: FFNActivation,
        num_heads: int,
        attention_type: AttentionType,
        num_timesteps: int,
        positional_encoding: PositionalEncodingType,
        rope_base: float,
        value_residual_type: ValueResidualType,
        learnable_attention_denom: bool,
    ) -> None:
        super().__init__()

        self.num_timesteps = num_timesteps
        self.attention_type = attention_type

        self.pre_norm: nn.Module
        match norm:
            case NormType.LAYER:
                self.pre_norm = nn.LayerNorm(normalized_shape=lifting_dim)
            case NormType.RMS:
                self.pre_norm = nn.RMSNorm(normalized_shape=lifting_dim)
            case _:
                raise ValueError(f"Invalid norm type: {norm}, select from one of {NormType.__members__.keys()}")  # type: ignore

        if lifting_dim % num_heads != 0:
            raise ValueError(f"Lifting (embedding) dim {lifting_dim} must be divisible by num_heads ({num_heads})")

        activation_fn: nn.Module
        match activation:
            case FFNActivation.RELU:
                activation_fn = nn.ReLU()
            case FFNActivation.RELU2:
                activation_fn = ReLU2()
            case FFNActivation.GELU:
                activation_fn = nn.GELU()
            case FFNActivation.SILU:
                activation_fn = nn.SiLU()
            case FFNActivation.SWIGLU:
                activation_fn = SwiGLU(input_dim=lifting_dim)
            case _:
                raise ValueError(f"Invalid activation function: {activation}, select from one of {FFNActivation.__members__.keys()}")

        # lifting_dim_irreps = get_lifting_dim_irreps(lifting_dim)
        # self.ffn = EquivariantMLP(
        #     in_irreps=lifting_dim_irreps,
        #     hidden_irreps=lifting_dim_irreps,
        #     out_irreps=lifting_dim_irreps,
        #     hidden_layers=2,
        #     activation=activation_fn,
        #     dropout_p=0.1,
        # )

        self.ffn = MLP(
            in_dim=lifting_dim,
            hidden_dim=lifting_dim,
            out_dim=lifting_dim,
            hidden_layers=2,
            activation=activation_fn,
            dropout_p=0.1,
        )

        self.attention: nn.Module
        match self.attention_type:
            case AttentionType.SELF:
                self.attention = QuadraticSelfAttention(
                    lifting_dim=lifting_dim,
                    num_heads=num_heads,
                    num_timesteps=self.num_timesteps,
                    positional_encoding=positional_encoding,
                    learnable_attention_denom=learnable_attention_denom,
                )
            case AttentionType.GHCA:
                self.attention = QuadraticHeterogenousCrossAttention(
                    lifting_dim=lifting_dim,
                    num_heads=num_heads,
                    num_timesteps=self.num_timesteps,
                    positional_encoding=positional_encoding,
                    rope_base=rope_base,
                    learnable_attention_denom=learnable_attention_denom,
                )
            case AttentionType.LINEAR_GHCA:
                self.attention = LinearHeterogenousCrossAttention(
                    lifting_dim=lifting_dim,
                    num_heads=num_heads,
                    num_timesteps=self.num_timesteps,
                    positional_encoding=positional_encoding,
                    rope_base=rope_base,
                    learnable_attention_denom=learnable_attention_denom,
                )
            case _:
                raise ValueError(f"Invalid heterogenous attention type: {attention_type}, select from one of {AttentionType.__members__.keys()}")  # type: ignore

        self.value_residual_type = value_residual_type

        self.lambda_v_residual: nn.Parameter | torch.Tensor
        match self.value_residual_type:
            case ValueResidualType.LEARNABLE:
                self.lambda_v_residual = nn.Parameter(torch.tensor(0.5))  # Initialize lambda to 0.5
            case ValueResidualType.FIXED:
                self.lambda_v_residual = torch.tensor(0.5)
            case ValueResidualType.NONE:
                self.lambda_v_residual = torch.empty(0)
            case _:
                raise ValueError(f"Invalid value residual type: {self.value_residual_type}, select from one of {ValueResidualType.__members__.keys()}")

    @override
    def forward(
        self,
        x_0: torch.Tensor,
        v_0: torch.Tensor,
        concatenated_features: torch.Tensor,
        q_data: torch.Tensor,
        mask: torch.Tensor | None,
        initial_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:  # None when value residual not yet set
        """Forward pass for the ATOM block.

        Parameters
        ----------
        x_0 : torch.Tensor
            Initial positions.
        v_0 : torch.Tensor
            Initial velocities.
        concatenated_features : torch.Tensor
            Concatenated features.
        q_data : torch.Tensor
            Query data.
        mask : torch.Tensor | None
            Padding mask.
        initial_v : torch.Tensor | None, optional
            Initial value for residual connection, by default None.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            The updated positions and the initial value for the next residual connection.
        """
        x_0 = self.pre_norm(x_0)
        v_0 = self.pre_norm(v_0)
        concatenated_features = self.pre_norm(concatenated_features)
        # q_data = self.pre_norm(q_data)

        if self.attention_type == AttentionType.SELF:
            attended_nodes: torch.Tensor = x_0 + self.attention(tensor=x_0, mask=mask)
        else:
            attended_nodes: torch.Tensor = x_0 + self.attention(x_0, v_0, concatenated_features, q_data=q_data, mask=mask)
        x_0 = attended_nodes + self.ffn(attended_nodes, mask)

        if self.value_residual_type == ValueResidualType.LEARNABLE:
            # Set initial_v if not provided (first layer); otherwise apply value residual
            if initial_v is None:
                initial_v = x_0.clone()
            else:
                lambda_val = torch.sigmoid(self.lambda_v_residual)
                x_0 = lambda_val * x_0 + (1 - lambda_val) * initial_v

        return x_0, initial_v


@final
class ATOM(nn.Module):
    def __init__(
        self,
        lifting_dim: int,
        norm: NormType,
        activation: FFNActivation,
        num_layers: int,
        num_heads: int,
        attention_type: AttentionType,
        output_heads: int,
        delta_update: bool,
        num_timesteps: int,
        positional_encoding: PositionalEncodingType,
        rope_base: float,
        lifting_type: LiftingType,
        projection_type: ProjectionType,
        rrwp_length: int,
        time_encoding_enabled: bool,
        value_residual_type: ValueResidualType,
        learnable_attention_denom: bool,
    ) -> None:
        """
        An ATOM model that always does T>1 predictions.

        ATOM is a graph transformer neural operator for predicting
        molecular dynamics trajectories.

        Parameters
        ----------
        lifting_dim : int
            Size of the lifted embedding dimension.
        norm : NormType
            Type of normalisation (e.g., NormType.LAYER).
        activation : FFNActivation
            Which feed-forward activation to use.
        num_layers : int
            Number of ATOM layers.
        num_heads : int
            Number of MHA heads.
        attention_type : AttentionType
            Type of attention mechanism.
        output_heads : int
            Number of output heads.
        delta_update : bool
            Whether to predict the delta of positions or absolute positions.
        num_timesteps : int
            The number of future steps (T) to predict.
        positional_encoding : PositionalEncodingType
            Type of positional encoding to use.
        rope_base : float
            Base for rotary positional embeddings.
        lifting_type : EquivariantLiftingType
            Type of equivariant lifting to use.
        rrwp_length : int
            Length of relative random walk positional encoding.
        value_residual_type : ValueResidualType
            Type of value residual connection.
        learnable_attention_denom : bool
            Whether the attention denominator is learnable.
        """
        super().__init__()

        assert num_timesteps > 1, f"num_timesteps must be greater than 1. Got {num_timesteps}"
        self.num_timesteps = num_timesteps
        self.use_equivariant_lifting = lifting_type
        self.lifting_dim = lifting_dim
        self.lifting_type = lifting_type
        self.projection_type = projection_type
        self.rrwp_length = rrwp_length
        self.output_heads = output_heads
        self.delta_update = delta_update
        self.time_encoding_enabled = time_encoding_enabled

        x_0_in_irreps, v_0_in_irreps, concat_feats_in_irreps = get_in_irreps(rrwp_length)
        lifting_dim_irreps: str = get_lifting_dim_irreps(lifting_dim)

        match lifting_type:
            case LiftingType.EQUIVARIANT:
                self.lifting_layer = EquivariantLiftTensorProduct(
                    x_0_in_irreps=x_0_in_irreps,
                    v_0_in_irreps=v_0_in_irreps,
                    concat_feats_in_irreps=concat_feats_in_irreps,
                    lifting_dim_irreps=lifting_dim_irreps,
                )
            case LiftingType.NO_TP:
                self.lifting_layer = EquivariantLift(
                    x_0_in_irreps=x_0_in_irreps,
                    v_0_in_irreps=v_0_in_irreps,
                    concat_feats_in_irreps=concat_feats_in_irreps,
                    lifting_dim_irreps=lifting_dim_irreps,
                )
            case LiftingType.NON_EQUIVARIANT:
                self.lifting_layer = StandardLift(
                    x_0_in_features=4,
                    v_0_in_features=4,
                    concat_feats_in_features=9 + rrwp_length,
                    lifting_dim=lifting_dim,
                )
            case LiftingType.CANONICALIZATION:
                self.lifting_layer = CanonicalizationLift(
                    x_0_in_irreps=x_0_in_irreps,
                    v_0_in_irreps=v_0_in_irreps,
                    concat_feats_in_irreps=concat_feats_in_irreps,
                    lifting_dim_irreps=lifting_dim_irreps,
                )
            case _:
                raise ValueError(f"Invalid equivariant lifting type: {lifting_type}, select from one of {bool.__members__.keys()}")

        self.transformer_blocks = nn.Sequential(
            *[
                ATOMBlock(
                    lifting_dim,
                    norm,
                    activation,
                    num_heads,
                    attention_type,
                    num_timesteps,
                    positional_encoding,
                    rope_base,
                    value_residual_type,
                    learnable_attention_denom,
                )
                for _ in range(num_layers)
            ]
        )

        # Final projection to (x, y, z)
        match projection_type:
            case ProjectionType.EQUIVARIANT:
                if self.output_heads > 1:
                    self.projection_layer = EquivariantMoEProject(lifting_dim, "1x1o", self.output_heads)
                else:
                    self.projection_layer = EquivariantProject(lifting_dim_irreps, "1x1o")
            case ProjectionType.DECANONICALIZATION:
                if self.output_heads > 1:
                    raise ValueError("Decanonicalization projection does not support multiple output heads")
                else:
                    self.projection_layer = DecanonicalizationProject(lifting_dim_irreps, "1x1o")
            case _:
                raise ValueError(f"Invalid projection type: {projection_type}, select from one of {ProjectionType.__members__.keys()}")

        self._initialise_weights(self)

    @override
    def forward(self, batch: TensorDict) -> torch.Tensor:
        """Forward pass for the ATOM model.

        Parameters
        ----------
        batch : TensorDict
            A TensorDict containing the input data.
            Expected keys: "x_0", "v_0", "concatenated_features".
            Optional key: "padded_nodes_mask".

        Returns
        -------
        torch.Tensor
            Predicted positions for N nodes over T timesteps, batched.
            Shape: (Batch, Timesteps, Nodes, 3)
        """
        # Batch: [Batch, Timesteps, Nodes, d]
        # Mask the inputs before applying the equivariant lifting layers
        mask: torch.Tensor | None = batch.get("padded_nodes_mask", None)

        if mask is not None:
            x_0: torch.Tensor = batch["x_0"] * mask
            v_0: torch.Tensor = batch["v_0"] * mask
            concat_features: torch.Tensor = batch["concatenated_features"] * mask
        else:
            x_0: torch.Tensor = batch["x_0"]
            v_0: torch.Tensor = batch["v_0"]
            concat_features: torch.Tensor = batch["concatenated_features"]

        ## Lift
        if self.lifting_type == LiftingType.CANONICALIZATION:
            lifted_x_0, lifted_v_0, lifted_concat_features, so3_matrix, x_0_mean = self.lifting_layer(x_0, v_0, concat_features, mask=mask)
        else:
            lifted_x_0, lifted_v_0, lifted_concat_features = self.lifting_layer(x_0, v_0, concat_features)
            so3_matrix = None  # type: ignore
            x_0_mean = None  # type: ignore

        # Add sinusoidal time encoding to token features (standard PE as elementwise addition)
        if self.time_encoding_enabled:
            # Require absolute time indices in the batch when enabled
            if "time_indices" not in batch:
                raise ValueError("time_encoding_enabled=True but 'time_indices' not found in batch. Ensure dataloader provides absolute time indices.")
            time_idx: torch.Tensor = batch["time_indices"].to(lifted_concat_features.device)
            if time_idx.dim() == 1:
                time_idx = time_idx.unsqueeze(0).expand(lifted_concat_features.shape[0], -1)

            # Build sinusoid from absolute indices on-the-fly to avoid fixed max_len
            bsz, tsteps = time_idx.shape
            D = self.lifting_dim
            position = time_idx.float().unsqueeze(-1)  # [B, T, 1]
            div_term = torch.exp(torch.arange(0, D, 2, device=position.device).float() * (-math.log(10000.0) / D))  # [D/2]
            pe = torch.zeros(bsz, tsteps, D, device=position.device)
            pe[..., 0::2] = torch.sin(position * div_term)
            pe[..., 1::2] = torch.cos(position * div_term)
            lifted_concat_features = lifted_concat_features + pe.unsqueeze(2)

        ## Kernel integral
        initial_v: torch.Tensor | None = None  # Value residual: Starts as none, becomes x_0 the first layer
        for layer in self.transformer_blocks:
            lifted_x_0, initial_v = layer(lifted_x_0, lifted_v_0, lifted_concat_features, q_data=lifted_concat_features, mask=mask, initial_v=initial_v)

        ## Project
        if self.projection_type == ProjectionType.DECANONICALIZATION:
            assert so3_matrix is not None and x_0_mean is not None, "Decanonicalization requires canonicalization outputs (Q and x_0_mean)."
            final_pred_pos: torch.Tensor = self.projection_layer(lifted_x_0, lifted_concat_features, so3_matrix, x_0_mean)
        else:
            final_pred_pos: torch.Tensor = self.projection_layer(lifted_x_0, lifted_concat_features)

        if self.delta_update:
            final_pred_pos = batch["x_0"][..., :3] + final_pred_pos

        return final_pred_pos  # Outputting the positions (x, y, z) for N nodes over T timesteps. Batched.

    @staticmethod
    def _initialise_weights(model: nn.Module) -> None:
        """Initialise the weights of the model.

        Uses Kaiming normal initialisation for linear layers and zeros for biases.

        Parameters
        ----------
        model : nn.Module
            The model to initialise.
        """
        for module in model.modules():
            if isinstance(module, nn.Linear):
                _ = nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
                if module.bias is not None:
                    _ = nn.init.zeros_(module.bias)

    def _build_time_positional_encoding(self, num_timesteps: int, dim: int) -> torch.Tensor:
        """Classic transformer sinusoidal positional encoding for time indices.

        Returns a tensor of shape [num_timesteps, dim].
        """
        pe = torch.zeros(num_timesteps, dim)
        position = torch.arange(0, num_timesteps, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


#     def _get_concat_feature_irreps(self) -> tuple[str, str]:
#         """
#         Returns the irreps for the concatenated features.

#         Returns
#         -------
#         tuple[str, str]
#             A tuple containing two strings representing the irreps
#             for the concatenated features. The first string is for
#             features derived from x_0 and v_0, and the second is for
#             features derived from v_0, Z, and optionally RRWP.
#         """
#         concat_irreps_1: str = "1x1o + 1x0e"  # (x,y,z, ||x||)
#         concat_irreps_2: str = "1x1o + 1x0e + 1x0e"  # (vx,vy,vz, ||v||, Z)
#         if self.rrwp_length > 0:
#             concat_irreps_2_rrwp: str = f"{concat_irreps_2} + {self.rrwp_length}x0e"
#         else:
#             concat_irreps_2_rrwp: str = concat_irreps_2

#         return concat_irreps_1, concat_irreps_2_rrwp


# def get_lifting_dim_irreps(lifting_dim: int) -> str:
#     """
#     Returns the irreps for the lifting dimension.
#     """
#     vector_lifting_dim_irreps: int = lifting_dim // 3
#     scalar_lifting_dim_irreps: int = lifting_dim - vector_lifting_dim_irreps * 3  # Remainder

#     lifting_dim_irreps: str = f"{vector_lifting_dim_irreps}x1o + {scalar_lifting_dim_irreps}x0e"
#     return lifting_dim_irreps


def get_in_irreps(rrwp_length: int) -> tuple[str, str, str]:
    x_0_in_irreps = "1x1o + 1x0e"  # (x,y,z, ||x||)
    v_0_in_irreps = "1x1o + 1x0e"  # (vx,vy,vz, ||v||)

    concat_feats_in_irreps = "1x1o + 1x0e + 1x1o + 1x0e + 1x0e"  # (x,y,z, ||x||, vx,vy,vz, ||v||, Z)
    if rrwp_length > 0:
        concat_feats_in_irreps += f" + {rrwp_length}x0e"

    return x_0_in_irreps, v_0_in_irreps, concat_feats_in_irreps


def get_lifting_dim_irreps(lifting_dim: int) -> str:
    vector_lifting_dim_irreps: int = lifting_dim // 3
    scalar_lifting_dim_irreps: int = lifting_dim - vector_lifting_dim_irreps * 3  # Remainder

    lifting_dim_irreps: str = f"{vector_lifting_dim_irreps}x1o + {scalar_lifting_dim_irreps}x0e"
    return lifting_dim_irreps
