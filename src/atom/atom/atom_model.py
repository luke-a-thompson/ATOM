from typing import final, override

import torch
import torch.nn as nn
from atom.atom.activations import ReLU2, SwiGLU
from atom.training.config_options import (
    FFNActivation,
    NormType,
    ValueResidualType,
    AttentionType,
    LiftingType,
    PositionalEncodingType,
    ProjectionType,
    OutputMode,
)
from tensordict import TensorDict
from atom.atom.attentions import (
    QuadraticHeterogenousCrossAttention,
    QuadraticSelfAttention,
    LinearHeterogenousCrossAttention,
    GATv2GraphAttention,
)
from atom.atom.mlps import MLP
from atom.atom.lifting_layers import (
    StandardLift,
    QuasiEquivariantLift,
    QuasiEquivariantTPLift,
    CanonicalizationLift,
)
from atom.atom.projection_layers import (
    EquivariantProjectFull,
    EquivariantProjectPosOnly,
    DecanonicalizationProject,
    DecanonicalizationProjectPosOnly,
)
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
        rope_tau: float,
    ) -> None:
        super().__init__()

        self.num_timesteps = num_timesteps
        self.attention_type = attention_type

        self.pre_norm: nn.Module
        match norm:
            case NormType.LAYER:
                self.norms = nn.ModuleList(
                    [nn.LayerNorm(normalized_shape=lifting_dim) for _ in range(3)]
                )
            case NormType.RMS:
                self.norms = nn.ModuleList(
                    [nn.RMSNorm(normalized_shape=lifting_dim) for _ in range(3)]
                )
            case _:
                raise ValueError(
                    f"Invalid norm type: {norm}, select from one of {NormType.__members__.keys()}"
                )  # type: ignore

        if lifting_dim % num_heads != 0:
            raise ValueError(
                f"Lifting (embedding) dim {lifting_dim} must be divisible by num_heads ({num_heads})"
            )

        activation_fn: nn.Module
        match activation:
            case FFNActivation.RELU:
                activation_fn = nn.ReLU()
            case FFNActivation.LEAKY_RELU:
                activation_fn = nn.LeakyReLU()
            case FFNActivation.RELU2:
                activation_fn = ReLU2()
            case FFNActivation.GELU:
                activation_fn = nn.GELU()
            case FFNActivation.SILU:
                activation_fn = nn.SiLU()
            case FFNActivation.SWIGLU:
                activation_fn = SwiGLU(input_dim=lifting_dim)
            case _:
                raise ValueError(
                    f"Invalid activation function: {activation}, select from one of {FFNActivation.__members__.keys()}"
                )

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
                    rope_base=rope_base,
                    rope_tau=rope_tau,
                )
            case AttentionType.GHCA:
                self.attention = QuadraticHeterogenousCrossAttention(
                    lifting_dim=lifting_dim,
                    num_heads=num_heads,
                    num_timesteps=self.num_timesteps,
                    positional_encoding=positional_encoding,
                    rope_base=rope_base,
                    rope_tau=rope_tau,
                )
            case AttentionType.LINEAR_GHCA:
                self.attention = LinearHeterogenousCrossAttention(
                    lifting_dim=lifting_dim,
                    num_heads=num_heads,
                    num_timesteps=self.num_timesteps,
                    positional_encoding=positional_encoding,
                    rope_base=rope_base,
                    rope_tau=rope_tau,
                )
            case AttentionType.GATV2:
                self.attention = GATv2GraphAttention(
                    lifting_dim=lifting_dim,
                    num_heads=num_heads,
                    num_timesteps=self.num_timesteps,
                    positional_encoding=positional_encoding,
                    rope_base=rope_base,
                )
            case _:
                raise ValueError(
                    f"Invalid heterogenous attention type: {attention_type}, select from one of {AttentionType.__members__.keys()}"
                )  # type: ignore

        self.value_residual_type = value_residual_type

        self.lambda_v_residual: nn.Parameter | torch.Tensor
        match self.value_residual_type:
            case ValueResidualType.LEARNABLE:
                self.lambda_v_residual = nn.Parameter(
                    torch.tensor(0.5)
                )  # Initialize lambda to 0.5
            case ValueResidualType.FIXED:
                self.lambda_v_residual = torch.tensor(0.5)
            case ValueResidualType.NONE:
                self.lambda_v_residual = torch.empty(0)
            case _:
                raise ValueError(
                    f"Invalid value residual type: {self.value_residual_type}, select from one of {ValueResidualType.__members__.keys()}"
                )

    @override
    def forward(
        self,
        x_0: torch.Tensor,
        v_0: torch.Tensor,
        concatenated_features: torch.Tensor,
        q_data: torch.Tensor,
        mask: torch.Tensor | None,
        time_increments: torch.Tensor | None = None,
        initial_v: torch.Tensor | None = None,
        edge_index: tuple[torch.Tensor, torch.Tensor] | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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
        x_0 = self.norms[0](x_0)
        v_0 = self.norms[1](v_0)
        concatenated_features = self.norms[2](concatenated_features)

        # q_data = self.pre_norm(q_data)

        if self.attention_type == AttentionType.SELF:
            attended_nodes = x_0 + self.attention(
                tensor=x_0, mask=mask, time_increments=time_increments
            )
        elif self.attention_type == AttentionType.GATV2:
            if edge_index is None:
                msg = "GATV2 attention requires 'edge_index' to be provided to ATOMBlock.forward."
                raise ValueError(msg)
            attended_nodes = x_0 + self.attention(
                tensor=x_0,
                mask=mask,
                edge_index=edge_index,
                edge_mask=edge_mask,
                time_increments=time_increments,
            )
        else:
            attended_nodes = x_0 + self.attention(
                x_0,
                v_0,
                concatenated_features,
                q_data=q_data,
                mask=mask,
                time_increments=time_increments,
            )
        x_0 = attended_nodes + self.ffn(attended_nodes, mask)

        if self.value_residual_type in (
            ValueResidualType.LEARNABLE,
            ValueResidualType.FIXED,
        ):
            # Set initial_v if not provided (first layer); otherwise apply value residual
            if initial_v is None:
                initial_v = x_0.clone()
            else:
                if self.value_residual_type == ValueResidualType.LEARNABLE:
                    lambda_val = torch.sigmoid(self.lambda_v_residual)
                else:
                    lambda_val = self.lambda_v_residual.to(
                        dtype=x_0.dtype, device=x_0.device
                    )
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
        rope_tau: float,
        lifting_type: LiftingType,
        projection_type: ProjectionType,
        rrwp_length: int,
        value_residual_type: ValueResidualType,
        output_mode: OutputMode = OutputMode.POS_ONLY,
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

        """
        super().__init__()

        assert num_timesteps > 1, (
            f"num_timesteps must be greater than 1. Got {num_timesteps}"
        )
        self.num_timesteps = num_timesteps
        self.use_equivariant_lifting = lifting_type
        self.lifting_dim = lifting_dim
        self.lifting_type = lifting_type
        self.projection_type = projection_type
        self.rrwp_length = rrwp_length
        self.output_heads = output_heads
        self.delta_update = delta_update
        self.positional_encoding_type = positional_encoding
        self.output_mode = output_mode
        self.attention_type: AttentionType = attention_type
        # Removed FiLM modulation

        x_0_in_irreps, v_0_in_irreps, concat_feats_in_irreps = get_in_irreps(
            rrwp_length
        )
        lifting_dim_irreps: str = get_lifting_dim_irreps(lifting_dim)

        match lifting_type:
            case LiftingType.QUASI_EQUIVARIANT_TP:
                self.lifting_layer = QuasiEquivariantTPLift(
                    x_0_in_irreps=x_0_in_irreps,
                    v_0_in_irreps=v_0_in_irreps,
                    concat_feats_in_irreps=concat_feats_in_irreps,
                    lifting_dim_irreps=lifting_dim_irreps,
                )
            case LiftingType.QUASI_EQUIVARIANT:
                self.lifting_layer = QuasiEquivariantLift(
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
                raise ValueError(
                    f"Invalid equivariant lifting type: {lifting_type}, select from one of {LiftingType.__members__.keys()}"
                )

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
                    rope_tau,
                )
                for _ in range(num_layers)
            ]
        )

        # Final projection to (x, y, z)
        match projection_type:
            case ProjectionType.EQUIVARIANT:
                if self.output_mode == OutputMode.POS_ONLY:
                    self.projection_layer = EquivariantProjectPosOnly(
                        lifting_dim_irreps, "1x1o"
                    )
                else:
                    self.projection_layer = EquivariantProjectFull(
                        lifting_dim_irreps, "1x1o"
                    )
            case ProjectionType.DECANONICALIZATION:
                if self.output_mode == OutputMode.POS_ONLY:
                    self.projection_layer = DecanonicalizationProjectPosOnly(
                        lifting_dim_irreps, "1x1o"
                    )
                else:
                    self.projection_layer = DecanonicalizationProject(
                        lifting_dim_irreps, "1x1o"
                    )
            case _:
                raise ValueError(
                    f"Invalid projection type: {projection_type}, select from one of {ProjectionType.__members__.keys()}"
                )

        self._initialise_weights(self)

    @override
    def forward(self, batch: TensorDict) -> dict[str, torch.Tensor]:
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
            x_0 = batch["x_0"] * mask
            v_0 = batch["v_0"] * mask
            concat_features = batch["concatenated_features"] * mask
        else:
            x_0 = batch["x_0"]
            v_0 = batch["v_0"]
            concat_features = batch["concatenated_features"]

        ## Lift
        if self.lifting_type == LiftingType.CANONICALIZATION:
            lifted_x_0, lifted_v_0, lifted_concat_features, so3_matrix, x_0_mean = (
                self.lifting_layer(x_0, v_0, concat_features, mask=mask)
            )
        else:
            lifted_x_0, lifted_v_0, lifted_concat_features = self.lifting_layer(
                x_0, v_0, concat_features
            )
            so3_matrix = None  # type: ignore
            x_0_mean = None  # type: ignore

        # Add sinusoidal PE with local positions within the batch window
        if self.positional_encoding_type == PositionalEncodingType.SINUSOIDAL:
            B, T, N, D = lifted_concat_features.shape
            pos = (
                torch.arange(
                    T,
                    device=lifted_concat_features.device,
                    dtype=lifted_concat_features.dtype,
                )
                .unsqueeze(0)
                .unsqueeze(-1)
                .expand(B, -1, -1)
            )
            div_term = torch.exp(
                torch.arange(
                    0,
                    D,
                    2,
                    device=lifted_concat_features.device,
                    dtype=lifted_concat_features.dtype,
                )
                * (-math.log(10000.0) / D)
            )
            pe = torch.zeros(
                B,
                T,
                D,
                device=lifted_concat_features.device,
                dtype=lifted_concat_features.dtype,
            )
            pe[..., 0::2] = torch.sin(pos * div_term)
            pe[..., 1::2] = torch.cos(pos * div_term)
            lifted_concat_features = lifted_concat_features + pe.unsqueeze(2)

        # Prepare optional edge data for graph-based attention types
        edge_index: tuple[torch.Tensor, torch.Tensor] | None = None
        edge_mask: torch.Tensor | None = None
        if self.attention_type == AttentionType.GATV2:
            if "source_node_indices" not in batch or "target_node_indices" not in batch:
                msg = "GATV2 attention requires 'source_node_indices' and 'target_node_indices' in the batch."
                raise ValueError(msg)
            edge_index = (batch["source_node_indices"], batch["target_node_indices"])
            edge_mask = batch.get("edge_mask", None)

        # Kernel integral
        initial_v: torch.Tensor | None = None
        for layer in self.transformer_blocks:
            lifted_x_0, initial_v = layer(
                lifted_x_0,
                lifted_v_0,
                lifted_concat_features,
                q_data=lifted_concat_features,
                mask=mask,
                time_increments=batch.get("time_increments", None),
                initial_v=initial_v,
                edge_index=edge_index,
                edge_mask=edge_mask,
            )

        ## Project
        final_pred_pos: torch.Tensor
        final_pred_vel: torch.Tensor
        energy_per_node: torch.Tensor
        energy_pred: torch.Tensor | None = None
        if self.projection_type == ProjectionType.DECANONICALIZATION:
            assert so3_matrix is not None and x_0_mean is not None, (
                "Decanonicalization requires canonicalization outputs (Q and x_0_mean)."
            )
            if self.output_mode == OutputMode.POS_ONLY:
                final_pred_pos = self.projection_layer(
                    lifted_x_0, lifted_concat_features, so3_matrix, x_0_mean
                )
                final_pred_vel = torch.empty(0, device=lifted_x_0.device)
                energy_per_node = torch.empty(0, device=lifted_x_0.device)
            else:
                final_pred_pos, final_pred_vel, energy_per_node = self.projection_layer(
                    lifted_x_0, lifted_concat_features, so3_matrix, x_0_mean
                )
        else:
            if self.output_mode == OutputMode.POS_ONLY:
                final_pred_pos = self.projection_layer(
                    lifted_x_0, lifted_concat_features
                )
                final_pred_vel = torch.empty(0, device=lifted_x_0.device)
                energy_per_node = torch.empty(0, device=lifted_x_0.device)
            else:
                final_pred_pos, final_pred_vel, energy_per_node = self.projection_layer(
                    lifted_x_0, lifted_concat_features
                )

        if self.delta_update:
            final_pred_pos = batch["x_0"][..., :3] + final_pred_pos
        # Velocities are predicted as absolute vectors (no delta update)

        # Aggregate per-node energy to per-molecule energy per timestep only when needed
        if self.output_mode != OutputMode.POS_ONLY:
            if mask is not None:
                energy_per_node_masked = energy_per_node * mask
                energy_pred = energy_per_node_masked.squeeze(-1).sum(dim=2)
            else:
                energy_pred = energy_per_node.squeeze(-1).sum(dim=2)

        if self.output_mode == OutputMode.POS_ONLY:
            return {"pos": final_pred_pos}
        assert energy_pred is not None
        return {"pos": final_pred_pos, "vel": final_pred_vel, "energy": energy_pred}

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

    def _build_time_positional_encoding(
        self, num_timesteps: int, dim: int
    ) -> torch.Tensor:
        """Classic transformer sinusoidal positional encoding for time indices.

        Returns a tensor of shape [num_timesteps, dim].
        """
        pe = torch.zeros(num_timesteps, dim)
        position = torch.arange(0, num_timesteps, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
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

    concat_feats_in_irreps = (
        "1x1o + 1x0e + 1x1o + 1x0e + 1x0e"  # (x,y,z, ||x||, vx,vy,vz, ||v||, Z)
    )
    if rrwp_length > 0:
        concat_feats_in_irreps += f" + {rrwp_length}x0e"

    return x_0_in_irreps, v_0_in_irreps, concat_feats_in_irreps


def get_lifting_dim_irreps(lifting_dim: int) -> str:
    vector_lifting_dim_irreps: int = lifting_dim // 3
    scalar_lifting_dim_irreps: int = (
        lifting_dim - vector_lifting_dim_irreps * 3
    )  # Remainder

    lifting_dim_irreps: str = (
        f"{vector_lifting_dim_irreps}x1o + {scalar_lifting_dim_irreps}x0e"
    )
    return lifting_dim_irreps
