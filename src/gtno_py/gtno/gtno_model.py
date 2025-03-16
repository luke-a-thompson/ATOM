from typing import final, override
import torch
import torch.nn as nn
from enum import Enum
from gtno_py.gtno.activations import FFNActivation, ReLU2, SwiGLU
from tensordict import TensorDict
from gtno_py.gtno.cross_attentions import QuadraticHeterogenousCrossAttention
from gtno_py.gtno.mlps import MLP
from e3nn import o3
from enum import StrEnum


class ModelType(StrEnum):
    GTNO = "GTNO"


class NormType(StrEnum):
    LAYER = "layer"
    RMS = "rms"


class ValueResidualType(StrEnum):
    NONE = "none"
    LEARNABLE = "learnable"
    FIXED = "fixed"


class GraphHeterogenousAttentionType(StrEnum):
    GHCA = "GHCA"


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

        self.ffn = MLP(in_features=lifting_dim, out_features=lifting_dim, hidden_features=lifting_dim, hidden_layers=2, activation=activation_fn, dropout_p=0.0)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        num_timesteps: int,
        use_rope: bool,
        use_spherical_harmonics: bool,
        use_equivariant_lifting: bool,
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

        match use_equivariant_lifting:
            case True:
                self.lifting_layers = nn.ModuleDict(
                    {
                        "x_0": o3.Linear("1x1o + 1x0e", "42x1o + 2x0e"),  # Use e3nn lifting for equivariant embedding
                        "v_0": o3.Linear("1x1o + 1x0e", "42x1o + 2x0e"),  # Same for velocity
                        "concatenated_features": o3.FullyConnectedTensorProduct("1x1o + 1x0e", "1x1o + 1x0e + 1x0e", "42x1o + 2x0e"),
                    }
                )
            case False:
                self.lifting_layers = nn.ModuleDict(
                    {
                        "x_0": nn.Linear(4, lifting_dim),
                        "v_0": nn.Linear(4, lifting_dim),
                        "concatenated_features": nn.Linear(9, lifting_dim),
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
        self.projection_layer = o3.Linear("42x1o + 2x0e", "1x1o")

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
        lifted_concat_features: torch.Tensor = self.lifting_layers["concatenated_features"](concat_features[..., :4], concat_features[..., 4:])

        initial_v: torch.Tensor | None = None  # Value residual: Starts as none, becomes x_0 the first layer
        for layer in self.transformer_blocks:
            lifted_x_0, initial_v = layer(lifted_x_0, lifted_v_0, lifted_concat_features, q_data=lifted_concat_features, mask=mask, initial_v=initial_v)

        # Batch (x, y, z) + projection layer
        pred_pos: torch.Tensor = batch["x_0"][..., :3] + self.projection_layer(lifted_x_0)

        return pred_pos  # Outputting the positions (x, y, z) for N nodes over T timesteps. Batched.

    @staticmethod
    def _initialise_weights(model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                _ = nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
                if module.bias is not None:
                    _ = nn.init.zeros_(module.bias)
