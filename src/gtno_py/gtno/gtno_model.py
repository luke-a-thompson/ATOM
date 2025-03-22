from typing import final, override
import torch
import torch.nn as nn
from gtno_py.gtno.activations import ReLU2, SwiGLU
from gtno_py.training.config_options import FFNActivation, NormType, ValueResidualType, GraphHeterogenousAttentionType
from tensordict import TensorDict
from gtno_py.gtno.cross_attentions import QuadraticHeterogenousCrossAttention
from gtno_py.gtno.mlps import MLP
from e3nn import o3


@final
class GTNOBlock(nn.Module):
    def __init__(
        self,
        lifting_dim: int,
        norm: NormType,
        activation: FFNActivation,
        num_heads: int,
        heterogenous_attention_type: GraphHeterogenousAttentionType,
        num_timesteps: int,
        use_rope: bool,
        use_spherical_harmonics: bool,
        value_residual_type: ValueResidualType,
        learnable_attention_denom: bool,
    ) -> None:
        super().__init__()

        self.num_timesteps = num_timesteps

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

        self.ffn = MLP(in_dim=lifting_dim, out_dim=lifting_dim, hidden_dim=lifting_dim, hidden_layers=2, activation=activation_fn, dropout_p=0.0)

        self.heterogenous_attention: nn.Module
        match heterogenous_attention_type:
            case GraphHeterogenousAttentionType.GHCA:
                self.heterogenous_attention = QuadraticHeterogenousCrossAttention(
                    num_hetero_feats=3,
                    lifting_dim=lifting_dim,
                    num_heads=num_heads,
                    num_timesteps=self.num_timesteps,
                    use_rope=use_rope,
                    use_spherical_harmonics=use_spherical_harmonics,
                    learnable_attention_denom=learnable_attention_denom,
                )
            case _:
                raise ValueError(f"Invalid heterogenous attention type: {heterogenous_attention_type}, select from one of {GraphHeterogenousAttentionType.__members__.keys()}")  # type: ignore

        self.value_residual_type = value_residual_type

        self.lambda_v_residual: nn.Parameter | torch.Tensor
        match self.value_residual_type:
            case ValueResidualType.LEARNABLE:
                self.lambda_v_residual = nn.Parameter(torch.tensor(0.5))  # Initialize lambda to 0.5
            case ValueResidualType.FIXED:
                self.lambda_v_residual = torch.tensor(0.5)
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
        concatenated_features = self.pre_norm(concatenated_features)
        x_0 = self.pre_norm(x_0)
        v_0 = self.pre_norm(v_0)

        hetero_attended_nodes: torch.Tensor = x_0 + self.heterogenous_attention(x_0, v_0, concatenated_features, q_data=q_data, mask=mask)
        x_0 = hetero_attended_nodes + self.ffn(hetero_attended_nodes, mask)

        if self.value_residual_type == ValueResidualType.LEARNABLE:
            # Set initial_v if not provided (first layer); otherwise apply value residual
            if initial_v is None:
                initial_v = x_0.clone()
            else:
                lambda_val = torch.sigmoid(self.lambda_v_residual)
                x_0 = lambda_val * x_0 + (1 - lambda_val) * initial_v

        return x_0, initial_v


@final
class GTNO(nn.Module):
    def __init__(
        self,
        lifting_dim: int,
        norm: NormType,
        activation: FFNActivation,
        num_layers: int,
        num_heads: int,
        heterogenous_attention_type: GraphHeterogenousAttentionType,
        output_heads: int,
        num_timesteps: int,
        use_rope: bool,
        use_spherical_harmonics: bool,
        use_equivariant_lifting: bool,
        rrwp_length: int,
        value_residual_type: ValueResidualType,
        learnable_attention_denom: bool,
    ) -> None:
        """
        A GTNO model that always does T>1 predictions. GTNO is a graph transformer neural operator for predicting molecular dynamics trajectories.

        Args:
            lifting_dim: size of the lifted embedding dimension
            norm: type of normalisation (e.g., NormType.LAYER)
            activation: which feed-forward activation to use
            num_layers: number of IMPGTNOBlock layers
            num_heads: number of MHA heads
            graph_attention_type: 'Unified MHA', 'Split MHA', or 'GRIT'
            heterogenous_attention_type: e.g. 'G-HNCA'
            num_timesteps: the number of future steps (T) to predict
        """
        super().__init__()

        assert num_timesteps > 1, f"num_timesteps must be greater than 1. Got {num_timesteps}"
        self.num_timesteps = num_timesteps
        self.use_equivariant_lifting = use_equivariant_lifting
        self.lifting_dim = lifting_dim
        self.rrwp_length = rrwp_length
        self.output_heads = output_heads

        concat_irreps_1, concat_irreps_2 = self._get_concat_feature_irreps()
        lifting_dim_irreps = self._get_lifting_dim_irreps()

        match use_equivariant_lifting:
            case True:
                self.lifting_layers = nn.ModuleDict(
                    {
                        "x_0": o3.Linear("1x1o + 1x0e", lifting_dim_irreps),  # In: (x,y,z, ||x||)
                        "v_0": o3.Linear("1x1o + 1x0e", lifting_dim_irreps),  # In: (vx,vy,vz, ||v||)
                        "concatenated_features": o3.FullyConnectedTensorProduct(concat_irreps_1, concat_irreps_2, lifting_dim_irreps),
                    }
                )
            case False:
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
                GTNOBlock(
                    lifting_dim,
                    norm,
                    activation,
                    num_heads,
                    heterogenous_attention_type,
                    num_timesteps,
                    use_rope,
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
                    out_dim=self.output_heads,
                    hidden_dim=lifting_dim // 4,
                    hidden_layers=2,
                    activation=SwiGLU(lifting_dim // 4),
                    dropout_p=0.0,
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
        match self.use_equivariant_lifting:
            case True:
                lifted_concat_features: torch.Tensor = self.lifting_layers["concatenated_features"](concat_features[..., :4], concat_features[..., 4:])
            case False:
                lifted_concat_features: torch.Tensor = self.lifting_layers["concatenated_features"](concat_features)
            case _:
                raise ValueError(f"Invalid equivariant lifting type: {self.use_equivariant_lifting}, select from one of {bool.__members__.keys()}")

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

        pred_pos: torch.Tensor = batch["x_0"][..., :3] + final_pred_pos  # Residual connection

        return pred_pos  # Outputting the positions (x, y, z) for N nodes over T timesteps. Batched.

    @staticmethod
    def _initialise_weights(model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                _ = nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
                if module.bias is not None:
                    _ = nn.init.zeros_(module.bias)

    def _get_lifting_dim_irreps(self) -> str:
        """
        Returns the irreps for the lifting dimension.
        """
        vector_lifting_dim_irreps: int = self.lifting_dim // 3
        scalar_lifting_dim_irreps: int = self.lifting_dim - vector_lifting_dim_irreps * 3  # Remainder

        lifting_dim_irreps: str = f"{vector_lifting_dim_irreps}x1o + {scalar_lifting_dim_irreps}x0e"
        return lifting_dim_irreps

    def _get_concat_feature_irreps(self) -> tuple[str, str]:
        """
        Returns the irreps for the concatenated features.
        """
        concat_irreps_1: str = "1x1o + 1x0e"  # (x,y,z, ||x||)
        concat_irreps_2: str = "1x1o + 1x0e + 1x0e"  # (vx,vy,vz, ||v||, Z)
        if self.rrwp_length > 0:
            concat_irreps_2_rrwp: str = f"{concat_irreps_2} + {self.rrwp_length}x0e"
        else:
            concat_irreps_2_rrwp: str = concat_irreps_2

        return concat_irreps_1, concat_irreps_2_rrwp
