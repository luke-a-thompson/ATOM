from typing import final, override
import torch
import torch.nn as nn
from enum import Enum
from gtno_py.gtno.activations import FFNActivation, ReLU2, SwiGLU
from tensordict import TensorDict
from gtno_py.gtno.cross_attentions import SubQuadraticHeterogenousCrossAttention
from gtno_py.gtno.graph_attentions import UnifiedInputMHA, SplitInputMHA


class NormType(str, Enum):
    LAYER = "LayerNorm"
    RMS = "RMSNorm"


class GraphAttentionType(str, Enum):
    UNIFIED_MHA = "Unified MHA"
    SPLIT_MHA = "Split MHA"
    GRIT = "GRIT"


class GraphHeterogenousAttentionType(str, Enum):
    GHCNA = "G-HNCA"


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

        self.graph_attention: nn.Module
        match graph_attention_type:
            ## Add independent attentions for x_0, v_0, Z -> Heterogenous learns to compose learned graph representations
            ### Context for position data
            ### Heavy feature learning left to G-HNCA. Graph represnetation learning is done here
            ### Possibly different graphormer-style priors for x_0, v_0, Z
            case GraphAttentionType.UNIFIED_MHA:
                # Add causal masking
                self.graph_attention = UnifiedInputMHA(lifting_dim, num_heads, self.num_timesteps, batch_first=True)
            case GraphAttentionType.SPLIT_MHA:
                self.graph_attention = SplitInputMHA(lifting_dim, num_heads, batch_first=True)
            case GraphAttentionType.GRIT:
                raise NotImplementedError("GRITAttention is not implemented")
            case _:
                raise ValueError(f"Invalid graph attention type: {graph_attention_type}, select from one of {GraphAttentionType.__members__.keys()}")

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

        self.ffn = nn.Sequential(
            nn.Linear(in_features=lifting_dim, out_features=lifting_dim),
            activation_fn,
            nn.Linear(in_features=lifting_dim, out_features=lifting_dim),
        )

        self.heterogenous_attention: nn.Module
        match heterogenous_attention_type:
            case GraphHeterogenousAttentionType.GHCNA:
                self.heterogenous_attention = SubQuadraticHeterogenousCrossAttention(
                    num_hetero_feats=2,
                    lifting_dim=lifting_dim,
                    num_heads=num_heads,
                    num_timesteps=self.num_timesteps,
                )
            case _:
                raise ValueError(f"Invalid heterogenous attention type: {heterogenous_attention_type}, select from one of {GraphHeterogenousAttentionType.__members__.keys()}")  # type: ignore

    @override
    def forward(self, batch: dict[str, torch.Tensor], q_data: torch.Tensor) -> dict[str, torch.Tensor]:
        # Graph attention as message passing
        match self.graph_attention:
            case UnifiedInputMHA():
                batch["concatenated_features"] = self.pre_norm(batch["concatenated_features"])
            case SplitInputMHA():
                batch["x_0"] = self.pre_norm(batch["x_0"])  # We probably shouldn't normalise position/velocity right?
                batch["v_0"] = self.pre_norm(batch["v_0"])
            case _:
                raise ValueError(f"Invalid graph attention type: {self.graph_attention}, select from one of {GraphAttentionType.__members__.keys()}")

        # Check this residual works
        match self.graph_attention:
            case UnifiedInputMHA():
                graph_attended_concat: torch.Tensor = batch["concatenated_features"] + self.graph_attention(batch)["concatenated_features"]  # Residual connection
                batch["concatenated_features"] = self.ffn(graph_attended_concat)
            case SplitInputMHA():
                graph_attended_pos: torch.Tensor = batch["x_0"] + self.graph_attention(batch)["x_0"]  # Residual connection (with normalised - DOUBLE CHECK THIS)
                graph_attended_vel: torch.Tensor = batch["v_0"] + self.graph_attention(batch)["v_0"]  # Residual connection

                batch["x_0"] = self.ffn(graph_attended_pos)
                batch["v_0"] = self.ffn(graph_attended_vel)
            case _:
                raise ValueError(f"Invalid graph attention type: {self.graph_attention}, select from one of {GraphAttentionType.__members__.keys()}")

        hetero_attended_nodes: torch.Tensor = batch["x_0"] + self.heterogenous_attention(batch, q_data=q_data)["x_0"]  # Residual connection
        batch["x_0"] = self.ffn(hetero_attended_nodes)

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
        Multi-step IMPGTNO model that always does T>1 predictions.
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

        # One-time lifting to unified embedding space at model init. Keys must be in the batch dict
        # We infer the shape of the features from the graph attention type - Unified MHA works on concatenated features, Split MHA works on x_0, v_0, Z
        match graph_attention_type:
            case GraphAttentionType.UNIFIED_MHA:
                self.elements_to_lift = ["concatenated_features", "x_0", "v_0", "edge_attr"]
                in_dims = {
                    "concatenated_features": 9,
                    "x_0": 4,
                    "v_0": 4,
                    "edge_attr": 4,
                }
                print("Message passing on concatenated features: x_0 (position) || v_0 (velocity) || edge_attr (bonds)")
            case GraphAttentionType.SPLIT_MHA:
                # We should find a way to infer this from the data
                self.elements_to_lift = ["concatenated_features", "x_0", "v_0", "edge_attr"]
                in_dims = {
                    "concatenated_features": 9,
                    "x_0": 4,
                    "v_0": 4,
                    "edge_attr": 4,
                }
                print("Message passing on each graph feature: x_0 (position), v_0 (velocity)")
            case GraphAttentionType.GRIT:
                raise NotImplementedError("GRITAttention is not implemented")
            case _:
                raise ValueError(f"Invalid graph attention type: {graph_attention_type}, select from one of {GraphAttentionType.__members__.keys()}")

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
                _ = nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
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


# import math


# def rope_rotation(x: torch.Tensor, times: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
#     """
#     Applies Rotary Positional Encoding (RoPE) to x, using 'times' as absolute
#     time indices. This ensures uniqueness across the entire dataset.

#     x: [B, T, num_heads, head_dim]
#     times: [T] (absolute dataset time indices, e.g. [234, 235, ..., 241])
#     base: base for frequency scaling
#     """
#     B, T, H, d = x.shape
#     half_d = d // 2
#     # Frequencies (standard RoPE approach: freq_i = base^(-2i/d))
#     freq_seq = torch.arange(half_d, device=x.device, dtype=x.dtype)
#     freqs = torch.exp(-math.log(base) * (2 * freq_seq / d))  # [half_d]

#     # Angles: [T, half_d]
#     angles = times.unsqueeze(-1).float() * freqs.unsqueeze(0)

#     sin = torch.sin(angles)  # [T, half_d]
#     cos = torch.cos(angles)  # [T, half_d]

#     # Broadcast sin, cos to [B, T, H, half_d]
#     sin = sin.unsqueeze(0).unsqueeze(2).expand(B, T, H, half_d)
#     cos = cos.unsqueeze(0).unsqueeze(2).expand(B, T, H, half_d)

#     x1, x2 = x[..., :half_d], x[..., half_d:]  # Split into halves
#     x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
#     return x_rot


# class RoPEMultiheadAttention(nn.MultiheadAttention):
#     """
#     A simple multi-head attention with RoPE for Q,K projections, using *global*
#     absolute time indices.
#     """

#     def __init__(self, embed_dim, num_heads, batch_first=True):
#         super().__init__(embed_dim, num_heads, batch_first=batch_first)

#     def forward(self, query, key, value, times: torch.Tensor, **kwargs):
#         """
#         Expects:
#           query, key, value: [B, T, embed_dim]
#           times: [T] (absolute times)
#         """
#         B, T, _ = query.shape
#         # 1) Project Q,K,V
#         q, k, v = self.in_proj_qkv(query)

#         # 2) Reshape Q,K to [B, T, num_heads, head_dim]
#         q = q.view(B, T, self.num_heads, self.head_dim)
#         k = k.view(B, T, self.num_heads, self.head_dim)

#         # 3) Apply global RoPE rotation
#         q = rope_rotation(q, times)  # pass absolute times
#         k = rope_rotation(k, times)

#         # 4) Flatten back to [B, T, embed_dim]
#         q = q.view(B, T, self.embed_dim)
#         k = k.view(B, T, self.embed_dim)

#         # 5) Forward to standard dot-product attention
#         attn_output, attn_weights = super().forward(q, k, value, **kwargs)
#         return attn_output, attn_weights
