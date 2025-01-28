from typing import final, override
import torch
import torch.nn as nn
from enum import Enum
from gtno_py.gtno.activations import FFNActivation, ReLU2, SwiGLU
from tensordict import TensorDict
from gtno_py.gtno.cross_attentions import QuadraticHeterogenousCrossAttention, HamiltonianCrossAttentionVelPos
from gtno_py.gtno.mlps import MLP


class NormType(str, Enum):
    LAYER = "LayerNorm"
    RMS = "RMSNorm"


class GraphAttentionType(str, Enum):
    UNIFIED_MHA = "Unified MHA"
    SPLIT_MHA = "Split MHA"
    GRIT = "GRIT"


class GraphHeterogenousAttentionType(str, Enum):
    GHCA = "G-HCA"
    HAMILTONIAN = "Hamiltonian"


@final
class IMPGTNOBlock(nn.Module):
    def __init__(
        self,
        lifting_dim: int,
        norm: NormType,
        activation: FFNActivation,
        num_heads: int,
        graph_attention_type: GraphAttentionType,
        heterogenous_attention_type: GraphHeterogenousAttentionType,
        num_timesteps: int,
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
                activation_fn = SwiGLU()
            case _:
                raise ValueError(f"Invalid activation function: {activation}, select from one of {FFNActivation.__members__.keys()}")

        self.ffn = MLP(in_features=lifting_dim, out_features=lifting_dim, hidden_features=lifting_dim, hidden_layers=2, activation=activation_fn)

        self.heterogenous_attention: nn.Module
        match heterogenous_attention_type:
            case GraphHeterogenousAttentionType.GHCA:
                self.heterogenous_attention = QuadraticHeterogenousCrossAttention(
                    num_hetero_feats=4,
                    lifting_dim=lifting_dim,
                    num_heads=num_heads,
                    num_timesteps=self.num_timesteps,
                )
            case GraphHeterogenousAttentionType.HAMILTONIAN:
                self.heterogenous_attention = HamiltonianCrossAttentionVelPos(
                    lifting_dim=lifting_dim,
                    dt=0.001,
                    hidden_dim=64,
                )
            case _:
                raise ValueError(f"Invalid heterogenous attention type: {heterogenous_attention_type}, select from one of {GraphHeterogenousAttentionType.__members__.keys()}")  # type: ignore

    @override
    def forward(self, batch: dict[str, torch.Tensor], q_data: torch.Tensor) -> dict[str, torch.Tensor]:
        match self.heterogenous_attention:
            case QuadraticHeterogenousCrossAttention():
                batch["concatenated_features"] = self.pre_norm(batch["concatenated_features"])
                batch["x_0"] = self.pre_norm(batch["x_0"])
                batch["v_0"] = self.pre_norm(batch["v_0"])

                hetero_attended_nodes: torch.Tensor = batch["x_0"] + self.heterogenous_attention(batch, q_data=q_data)["x_0"]  # Residual connection
                batch["x_0"] = self.ffn(hetero_attended_nodes)

            case HamiltonianCrossAttentionVelPos():
                batch["concatenated_features"] = self.pre_norm(batch["concatenated_features"])
                batch["x_0"] = self.pre_norm(batch["x_0"])
                batch["v_0"] = self.pre_norm(batch["v_0"])

                out_dict = self.heterogenous_attention(batch, q_data=q_data)
                batch["x_0"] = batch["x_0"] + out_dict["x_0"]
            case _:
                raise ValueError(f"Invalid heterogenous attention type: {self.heterogenous_attention}, select from one of {GraphHeterogenousAttentionType.__members__.keys()}")

        return batch


@final
class IMPGTNO(nn.Module):
    def __init__(
        self,
        lifting_dim: int,
        norm: NormType,
        activation: FFNActivation,
        num_layers: int,
        num_heads: int,
        graph_attention_type: GraphAttentionType,
        heterogenous_attention_type: GraphHeterogenousAttentionType,
        num_timesteps: int,
    ) -> None:
        """
        Multi-step IMPGTNO model that always does T>1 predictions. IMPGTNO is a graph transformer neural operator for predicting molecular dynamics trajectories.
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

        self.elements_to_lift = ["concatenated_features", "x_0", "v_0", "edge_attr"]
        in_dims = {
            "concatenated_features": 9,
            "x_0": 4,
            "v_0": 4,
            "edge_attr": 5,
        }

        # Create one Linear layer per feature
        self.lifting_layers = nn.ModuleDict({key: nn.Linear(in_features=in_dims[key], out_features=lifting_dim) for key in self.elements_to_lift})

        self.layers = nn.Sequential(
            *[
                IMPGTNOBlock(
                    lifting_dim,
                    norm,
                    activation,
                    num_heads,
                    graph_attention_type,
                    heterogenous_attention_type,
                    num_timesteps,
                )
                for _ in range(num_layers)
            ]
        )

        # Final projection to (x, y, z)
        self.projection_layer = nn.Linear(in_features=lifting_dim, out_features=3)

        self._initialise_weights(self)

    @override
    def forward(self, batch: TensorDict) -> torch.Tensor:
        # Batch: [Batch, Nodes, 4]
        B, N, _ = batch["x_0"].shape

        batch = self._replicate_tensordict_BxT(batch, self.num_timesteps)  # [Batch * timesteps, Nodes, 4]
        # Project this batch feature from its original dimension to `lifting_dim`
        # Use the same "key" to pick the lifting layer from `self.lifting_layers` and the corresponding feature from the `batch` dict.
        for key in self.elements_to_lift:
            batch[key] = self.lifting_layers[key](batch[key])

        for layer in self.layers:
            batch = layer(batch, q_data=batch["concatenated_features"])

        out: torch.Tensor = self.projection_layer(batch["x_0"])
        assert out.shape[-1] == 3, f"Output shape must have last dimension of 3 (x, y, z). Got {out.shape}"

        # 6) Reshape to [B, N, T, 3]
        out = out.view(self.num_timesteps, B, N, 3).permute(1, 2, 0, 3).contiguous()
        return out  # Outputting the positions (x, y, z) for N nodes over T timesteps. Batched.

    @staticmethod
    def _initialise_weights(model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                _ = nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
                if module.bias is not None:
                    _ = nn.init.zeros_(module.bias)

    @staticmethod
    def _replicate_tensordict_BxT(batch: TensorDict, T: int) -> TensorDict:
        """
        Replicates the entire tensordict along the batch dimension T times.
        Resulting TensorDict has batch_size = [T * B].

        Specifically, if the original tensordict has batch_size = [B], then
        we generate a new tensordict with batch_size = [T * B], where each
        of the B entries is repeated T times.

        Args:
            batch (TensorDict): The input tensordict with batch_size = [B].
            T (int): The number of times to replicate the batch dimension.

        Returns:
            TensorDict: A new tensordict whose batch_size = [T * B], with each
            field in the original tensordict replicated T times.

        Example:
            >>> # Suppose 'batch' has batch_size [32].
            >>> # We want 8 future timesteps -> T=8.
            >>> # The returned tensordict will have batch_size [256].
            >>> expanded = replicate_tensordict(batch, 8)
            >>> print(expanded.batch_size)  # torch.Size([256])
        """
        B = batch.batch_size[0]
        new_shape = (T, B)  # We'll reshape to [T * B] eventually.

        # 1) Insert a new dimension at index 0 -> shape = [1, B].
        out: torch.Tensor = batch.unsqueeze(0)

        # 2) Expand along that new dimension T times -> shape = [T, B].
        out = out.expand(*new_shape)

        # 3) Make memory contiguous, then flatten the first two dims -> [T * B].
        out = out.contiguous().view(-1)

        return out
