from typing import final, override
import torch
import torch.nn as nn
from atom.atom.activations import ReLU2, SwiGLU
from atom.training.config_options import FFNActivation, NormType, ValueResidualType, AttentionType, EquivariantLiftingType
from tensordict import TensorDict
from atom.atom.attentions import QuadraticHeterogenousCrossAttention, QuadraticSelfAttention
from atom.atom.mlps import MLP
from e3nn import o3


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
        use_rope: bool,
        rope_base: float,
        use_spherical_harmonics: bool,
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
                    use_rope=use_rope,
                    use_spherical_harmonics=use_spherical_harmonics,
                    learnable_attention_denom=learnable_attention_denom,
                )
            case AttentionType.GHCA:
                self.attention = QuadraticHeterogenousCrossAttention(
                    num_hetero_feats=3,
                    lifting_dim=lifting_dim,
                    num_heads=num_heads,
                    num_timesteps=self.num_timesteps,
                    use_rope=use_rope,
                    rope_base=rope_base,
                    use_spherical_harmonics=use_spherical_harmonics,
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
        concatenated_features = self.pre_norm(concatenated_features)
        x_0 = self.pre_norm(x_0)
        v_0 = self.pre_norm(v_0)

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
        use_rope: bool,
        rope_base: float,
        use_spherical_harmonics: bool,
        use_equivariant_lifting: EquivariantLiftingType,
        rrwp_length: int,
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
        use_rope : bool
            Whether to use rotary positional embeddings.
        rope_base : float
            Base for rotary positional embeddings.
        use_spherical_harmonics : bool
            Whether to use spherical harmonics.
        use_equivariant_lifting : EquivariantLiftingType
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
        self.use_equivariant_lifting = use_equivariant_lifting
        self.lifting_dim = lifting_dim
        self.rrwp_length = rrwp_length
        self.output_heads = output_heads
        self.delta_update = delta_update

        concat_irreps_1, concat_irreps_2 = self._get_concat_feature_irreps()
        lifting_dim_irreps = get_lifting_dim_irreps(lifting_dim)

        if self.rrwp_length > 0:
            vz_irreps = f"1x1o + 1x0e + 1x0e + {self.rrwp_length}x0e"
        else:
            vz_irreps = "1x1o + 1x0e + 1x0e"

        match use_equivariant_lifting:
            case EquivariantLiftingType.EQUIVARIANT:
                self.lifting_layers = nn.ModuleDict(
                    {
                        "x_0": o3.Linear("1x1o + 1x0e", lifting_dim_irreps),  # In: (x,y,z, ||x||)
                        "v_0": o3.Linear("1x1o + 1x0e", lifting_dim_irreps),  # In: (vx,vy,vz, ||v||)
                        "vz_0": o3.Linear(vz_irreps, lifting_dim_irreps),  # In: (vx, vy, vz, ||v||, Z). If rrwp_length > 0, then (vx, vy, vz, ||v||, Z, rrwp_length)
                        "concatenated_features": o3.FullyConnectedTensorProduct(lifting_dim_irreps, lifting_dim_irreps, lifting_dim_irreps),
                    }
                )
            case EquivariantLiftingType.NO_TP:
                self.lifting_layers = nn.ModuleDict(
                    {
                        "x_0": o3.Linear("1x1o + 1x0e", lifting_dim_irreps),  # In: (x,y,z, ||x||)
                        "v_0": o3.Linear("1x1o + 1x0e", lifting_dim_irreps),  # In: (vx,vy,vz, ||v||)
                        "concatenated_features": o3.Linear(str(concat_irreps_1 + "+" + concat_irreps_2), lifting_dim_irreps),
                    }
                )
            case EquivariantLiftingType.NONE:
                self.lifting_layers = nn.ModuleDict(
                    {
                        "x_0": nn.Linear(4, lifting_dim),
                        "v_0": nn.Linear(4, lifting_dim),
                        "concatenated_features": nn.Linear(9 + rrwp_length, lifting_dim),
                    }
                )
            case _:
                raise ValueError(f"Invalid equivariant lifting type: {use_equivariant_lifting}, select from one of {bool.__members__.keys()}")

        self.transformer_blocks = nn.Sequential(
            *[
                ATOMBlock(
                    lifting_dim,
                    norm,
                    activation,
                    num_heads,
                    attention_type,
                    num_timesteps,
                    use_rope,
                    rope_base,
                    use_spherical_harmonics,
                    value_residual_type,
                    learnable_attention_denom,
                )
                for _ in range(num_layers)
            ]
        )

        # Final projection to (x, y, z)
        if self.output_heads > 1:
            self.weight_pred_gate_net = nn.Sequential(
                MLP(
                    in_dim=lifting_dim,
                    hidden_dim=lifting_dim // 4,
                    out_dim=self.output_heads,
                    hidden_layers=2,
                    activation=SwiGLU(lifting_dim // 4),
                    dropout_p=0.1,
                ),
                nn.Softmax(dim=-1),
            )

            self.projection_layers = nn.Sequential(
                *[o3.Linear(lifting_dim_irreps, "1x1o") for _ in range(self.output_heads)],
            )
        else:
            self.projection_layer = o3.Linear(lifting_dim_irreps, "1x1o")

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

        # Lift the inputs
        lifted_x_0: torch.Tensor = self.lifting_layers["x_0"](x_0)
        lifted_v_0: torch.Tensor = self.lifting_layers["v_0"](v_0)
        if self.use_equivariant_lifting == EquivariantLiftingType.EQUIVARIANT:
            lifted_vz_0: torch.Tensor = self.lifting_layers["vz_0"](concat_features[..., 4:])
            lifted_concat_features: torch.Tensor = self.lifting_layers["concatenated_features"](lifted_x_0, lifted_vz_0)
        else:
            lifted_concat_features: torch.Tensor = self.lifting_layers["concatenated_features"](concat_features)

        initial_v: torch.Tensor | None = None  # Value residual: Starts as none, becomes x_0 the first layer
        for layer in self.transformer_blocks:
            lifted_x_0, initial_v = layer(lifted_x_0, lifted_v_0, lifted_concat_features, q_data=lifted_concat_features, mask=mask, initial_v=initial_v)

        # Batch (x, y, z) + projection layer
        if self.output_heads > 1:
            # Decides which output heads should be emphasised
            head_weights: torch.Tensor = self.weight_pred_gate_net(lifted_concat_features.mean(dim=(1, 2)))  # mean pool over nodes and timesteps (molecule-level summary)
            # Project each head's predictions to the final output space
            pred_pos_per_head: list[torch.Tensor] = [self.projection_layers[i](lifted_x_0) for i in range(self.output_heads)]
            # Weighted sum of the heads
            final_pred_pos = torch.zeros_like(pred_pos_per_head[0])

            for i in range(self.output_heads):
                final_pred_pos = final_pred_pos + head_weights[:, i].view(-1, 1, 1, 1) * pred_pos_per_head[i]  # Weighted sum
        else:
            # Single-head prediction
            final_pred_pos: torch.Tensor = self.projection_layer(lifted_x_0)

        if self.delta_update:
            pred_pos = batch["x_0"][..., :3] + final_pred_pos
        else:
            pred_pos = final_pred_pos

        return pred_pos  # Outputting the positions (x, y, z) for N nodes over T timesteps. Batched.

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

    def _get_concat_feature_irreps(self) -> tuple[str, str]:
        """
        Returns the irreps for the concatenated features.

        Returns
        -------
        tuple[str, str]
            A tuple containing two strings representing the irreps
            for the concatenated features. The first string is for
            features derived from x_0 and v_0, and the second is for
            features derived from v_0, Z, and optionally RRWP.
        """
        concat_irreps_1: str = "1x1o + 1x0e"  # (x,y,z, ||x||)
        concat_irreps_2: str = "1x1o + 1x0e + 1x0e"  # (vx,vy,vz, ||v||, Z)
        if self.rrwp_length > 0:
            concat_irreps_2_rrwp: str = f"{concat_irreps_2} + {self.rrwp_length}x0e"
        else:
            concat_irreps_2_rrwp: str = concat_irreps_2

        return concat_irreps_1, concat_irreps_2_rrwp


def get_lifting_dim_irreps(lifting_dim: int) -> str:
    """
    Returns the irreps for the lifting dimension.
    """
    vector_lifting_dim_irreps: int = lifting_dim // 3
    scalar_lifting_dim_irreps: int = lifting_dim - vector_lifting_dim_irreps * 3  # Remainder

    lifting_dim_irreps: str = f"{vector_lifting_dim_irreps}x1o + {scalar_lifting_dim_irreps}x0e"
    return lifting_dim_irreps
