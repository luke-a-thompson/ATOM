from typing import final, override
import torch
import torch.nn as nn
from enum import Enum
from gtno_py.modules.activations import FFNActivation, ReLU2, SwiGLU
from gtno_py.utils import get_context
from tensordict import TensorDict


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
class UnifiedInputMHA(nn.Module):
    """
    Implicit message passing via attention using concatenated node features of the graph: x_0 (position), v_0 (velocity), Z (atomic number)

    The idea is to construct a learned unified graph representation accounting for dependencies between these primary node features.
    This may learn some position-velocity dependence across time (i.e., Newton's second law).
    """

    def __init__(self, lifting_dim: int, num_heads: int, num_timesteps: int, batch_first: bool = True) -> None:
        super().__init__()

        self.num_timesteps = num_timesteps
        self.graph_attention = nn.MultiheadAttention(embed_dim=lifting_dim, num_heads=num_heads, batch_first=batch_first)

    @override
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        concatenated_features = batch["concatenated_features"]

        B_times_T, N, d = concatenated_features.shape
        B = concatenated_features.shape[0] // self.num_timesteps  # Get the batch size from B*T

        x_over_time = IMPGTNO.flatten_spatiotemporal(concatenated_features, B, N, self.num_timesteps)

        attn_output: torch.Tensor = self.graph_attention(x_over_time, x_over_time, x_over_time)[0]

        batch["concatenated_features"] = IMPGTNO.unflatten_spatiotemporal(attn_output, B, N, self.num_timesteps)

        return batch


@final
class SplitInputMHA(nn.Module):
    """
    Implict message passing via attention using separate MHA for x_0 (position), v_0 (velocity). We do not include Z (atomic number) as it is a scalar.

    The idea here is to construct learned graph representations for each of the primary node features.
    """

    def __init__(self, lifting_dim: int, num_heads: int, batch_first: bool = True) -> None:
        super().__init__()

        self.lifting_dim = lifting_dim

        self.position_attention = nn.MultiheadAttention(embed_dim=lifting_dim, num_heads=num_heads, batch_first=batch_first)
        self.velocity_attention = nn.MultiheadAttention(embed_dim=lifting_dim, num_heads=num_heads, batch_first=batch_first)

    @override
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        positions = batch["x_0"]
        velocities = batch["v_0"]
        assert positions.shape == velocities.shape, f"Positions and velocities must have the same shape. Got {positions.shape} and {velocities.shape}"
        assert positions.shape[-1] == self.lifting_dim, f"Positions must have last dim of {self.lifting_dim} when MHA called. Got {positions.shape}"
        assert velocities.shape[-1] == self.lifting_dim, f"Velocities must have last dim of {self.lifting_dim} when MHA called. Got {velocities.shape}"

        position_attn_output: torch.Tensor = self.position_attention(positions, positions, positions)[0]
        velocity_attn_output: torch.Tensor = self.velocity_attention(velocities, velocities, velocities)[0]

        batch["x_0"] = position_attn_output
        batch["v_0"] = velocity_attn_output

        return batch


@final
class GraphHeterogenousCrossAttention(nn.Module):
    def __init__(
        self,
        num_hetero_feats: int,
        lifting_dim: int,
        num_heads: int,
        num_timesteps: int,
    ) -> None:
        """
        Heterogenous graph cross attention. We construct separate K/V projections for each heterogeneous feature, then perform cross attention on queries generated from the q_data aka "trunk".

        Features must already be lifted to the same dimension.
        """
        super().__init__()

        self.num_heads = num_heads
        self.num_hetero_feats = num_hetero_feats
        self.lifting_dim = lifting_dim
        self.num_timesteps = num_timesteps
        # Query projection (applied to node embeddings)
        self.query = nn.Linear(lifting_dim, lifting_dim)

        # Keys/Values for heterogeneous features
        # Each input set gets its own distinct K/V projections
        self.keys = nn.ModuleList([nn.Linear(lifting_dim, lifting_dim) for _ in range(self.num_hetero_feats)])
        self.values = nn.ModuleList([nn.Linear(lifting_dim, lifting_dim) for _ in range(self.num_hetero_feats)])
        self.out_proj = nn.Linear(lifting_dim, lifting_dim)

    @override
    def forward(self, batch: dict[str, torch.Tensor], q_data: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        This is the heterogenous cross attention.

        Parameters:
            batch: The batch dictionary containing the node features, edge features, graph features, etc.
            q_data: The query data for generating the query for cross attention.

        Returns:
            The updated batch dictionary with the heterogenous cross attention applied to x_0.
        """
        # Recall batch["x_0"] has shape [B*T, N, d]
        B_times_T, N, d = batch["x_0"].shape
        B = B_times_T // self.num_timesteps  # Get the batch size from B*T

        # Lifted nodes
        assert (
            batch["x_0"].shape[-1] == self.lifting_dim and batch["edge_attr"].shape[-1] == self.lifting_dim
        ), f"{get_context(self)}: Lifted nodes and edge_attr embedding dim must match. Got {batch['x_0'].shape} and {batch['edge_attr'].shape}"

        assert (
            batch["x_0"].shape[0] == B * self.num_timesteps and batch["edge_attr"].shape[0] == B * self.num_timesteps
        ), f"{get_context(self)}: Batch size must match the number of timesteps. Got {batch['x_0'].shape[0]} and {B * self.num_timesteps}"

        # Query data
        assert (
            q_data.shape[0] == B * self.num_timesteps and q_data.shape[-1] == self.lifting_dim
        ), f"{get_context(self)}: Query data batch size and feature dimension must match. Got {q_data.shape}"

        # Put the heterogeneous embeddings into a list to loop over
        hetero_features: list[torch.Tensor] = [batch["x_0"], batch["edge_attr"]]
        assert (
            len(hetero_features) == self.num_hetero_feats
        ), f"Number of heterogeneous features must match the number of keys/values. Expected {self.num_hetero_feats}, got {len(hetero_features)}"

        # 2. Compute Q from nodes only (following your given pattern)
        # We compute the queries from the trunk
        q_data = IMPGTNO.flatten_spatiotemporal(q_data, B, N, self.num_timesteps)  # [B*T, N, d] -> [B, T*N, d]
        q_proj: torch.Tensor = self.query(q_data)  # [B, N*T, d]
        q: torch.Tensor = q_proj.view(B, self.num_heads, (N * self.num_timesteps), self.lifting_dim // self.num_heads)
        q: torch.Tensor = q.softmax(dim=-1)  # [B, num_heads, T*N, d_head]
        out = q

        # 3. Perform cross-attention over all heterogeneous inputs
        # Here, hetero_features = [lifted_nodes, lifted_edges, lifted_graph]
        # Drop for loop, have a tensor with first dim as num_hetero_feats

        # Add causal masking - Do we really want this? Assume our attention is |V| x |V|, then we'd be masking nodes.
        # In a way, we actually want an attention of timesteps * |V| x timesteps * |V|, then RoPE all V with the same timestep the same
        # then mask timestep-node blocks (i.e., mask 13 nodes at a time (1 timestep))
        # Rope is kind of like a temporal prior, that graphs far apart in time are less relevant to eachother attentionally. This is
        # somewhat like temporal convolution in EGNO
        # The dot product in RoPE is the usual query-key match, however rotating the vectors means the dot product also incorporates a positional prior (further = lower dot product)
        for i in range(self.num_hetero_feats):
            h_feat = hetero_features[i]
            # Determine the sequence length for this feature type
            B_times_T, N_or_E, d = h_feat.shape
            T = B_times_T // B

            # 1) Flatten to [B, (N_or_E * T), d]
            h_feat = IMPGTNO.flatten_spatiotemporal(h_feat, B, N_or_E, T)

            # 2) Project K/V to [B, (N_or_E * T), d]
            k_proj = self.keys[i](h_feat)  # => [B, (N_or_E*T), d]
            v_proj = self.values[i](h_feat)  # => [B, (N_or_E*T), d]

            # 3) Reshape to multihead dims [B, num_heads, (N_or_E * T), d_head]
            #    so each head sees the full spatiotemporal dimension
            k = k_proj.view(B, self.num_heads, (N_or_E * T), self.lifting_dim // self.num_heads)  # k: [B, num_heads, (N_or_E*T), d_head]
            v = v_proj.view(B, self.num_heads, (N_or_E * T), self.lifting_dim // self.num_heads)  # v: [B, num_heads, (N_or_E*T), d_head]

            k = k.softmax(dim=-1)

            # Normalisation step
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1.0 / torch.clamp_min((q * k_cumsum).sum(dim=-1, keepdim=True), 1e-8)
            out: torch.Tensor = q + (q @ (k.transpose(-2, -1) @ v)) * D_inv

        # 4. Project output back - DOUBLE CHECK THIS
        out = out.transpose(1, 2).contiguous().view(B, N * self.num_timesteps, self.lifting_dim)
        out = self.out_proj(out)
        out = IMPGTNO.unflatten_spatiotemporal(out, B, N, self.num_timesteps)

        batch["x_0"] = out

        return batch


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
                self.heterogenous_attention = GraphHeterogenousCrossAttention(
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
        # MHA returns (attn_output, attn_weights)
        # Everything modified on each layer must be scaled / normalised
        match self.graph_attention:
            case UnifiedInputMHA():
                batch["concatenated_features"] = self.pre_norm(batch["concatenated_features"])
            case SplitInputMHA():
                batch["x_0"] = self.pre_norm(batch["x_0"])  # We probably shouldn't normalise position/velocity right?
                batch["v_0"] = self.pre_norm(batch["v_0"])
            case _:
                raise ValueError(f"Invalid graph attention type: {self.graph_attention}, select from one of {GraphAttentionType.__members__.keys()}")

        # Currently this residual is broken as we are returning a TensorDict from self.graph_attention, but node_features is a tensor
        # We can consider elementwise residual via the dict
        # We can do this if we change the batch dict to a tensor dict:
        # https://pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict.add
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
        out = batch.unsqueeze(0)

        # 2) Expand along that new dimension T times -> shape = [T, B].
        out = out.expand(*new_shape)

        # 3) Make memory contiguous, then flatten the first two dims -> [T * B].
        out = out.contiguous().view(-1)

        return out

    @staticmethod
    def _replicate_tensordict_B_T(batch: TensorDict, T: int) -> TensorDict:
        """
        Replicates the entire tensordict along a new 'time' dimension so the
        final shape is [B, T], instead of [T * B].

        If batch.batch_size == [B], then out.batch_size == [B, T].
        """
        B = batch.batch_size[0]
        # 1) Insert a time dimension at index 1 => shape [B, 1]
        out = batch.unsqueeze(1)
        # 2) Expand that dimension => shape [B, T]
        out = out.expand(B, T)
        return out

    @staticmethod
    def flatten_spatiotemporal(x: torch.Tensor, B: int, N: int, T: int) -> torch.Tensor:
        """
        Takes [B*T, N, d] -> reshapes to [B, N*T, d].

        Where:
            B = batch size
            N = number of nodes
            T = number of timesteps
            d = feature dimension
        """
        # 1) Reshape [B*T, N, d] -> [B, T, N, d]
        x = x.view(B, T, N, -1)
        # 2) Permute to [B, N*T, d]
        x = x.permute(0, 2, 1, 3).contiguous().view(B, N * T, -1)
        return x

    @staticmethod
    def unflatten_spatiotemporal(x: torch.Tensor, B: int, N: int, T: int) -> torch.Tensor:
        """
        Takes [B, N*T, d] -> reshapes back to [B*T, N, d].

        Where:
            B = batch size
            N = number of nodes
            T = number of timesteps
            d = feature dimension
        """
        # 1) Reshape to [B, N, T, d]
        x = x.view(B, N, T, -1)
        # 2) Permute to [B, T, N, d] -> flatten to [B*T, N, d]
        x = x.permute(0, 2, 1, 3).contiguous().view(B * T, N, -1)
        return x


import math


def rope_rotation(x: torch.Tensor, times: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """
    Applies Rotary Positional Encoding (RoPE) to x, using 'times' as absolute
    time indices. This ensures uniqueness across the entire dataset.

    x: [B, T, num_heads, head_dim]
    times: [T] (absolute dataset time indices, e.g. [234, 235, ..., 241])
    base: base for frequency scaling
    """
    B, T, H, d = x.shape
    half_d = d // 2
    # Frequencies (standard RoPE approach: freq_i = base^(-2i/d))
    freq_seq = torch.arange(half_d, device=x.device, dtype=x.dtype)
    freqs = torch.exp(-math.log(base) * (2 * freq_seq / d))  # [half_d]

    # Angles: [T, half_d]
    angles = times.unsqueeze(-1).float() * freqs.unsqueeze(0)

    sin = torch.sin(angles)  # [T, half_d]
    cos = torch.cos(angles)  # [T, half_d]

    # Broadcast sin, cos to [B, T, H, half_d]
    sin = sin.unsqueeze(0).unsqueeze(2).expand(B, T, H, half_d)
    cos = cos.unsqueeze(0).unsqueeze(2).expand(B, T, H, half_d)

    x1, x2 = x[..., :half_d], x[..., half_d:]  # Split into halves
    x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rot


class RoPEMultiheadAttention(nn.MultiheadAttention):
    """
    A simple multi-head attention with RoPE for Q,K projections, using *global*
    absolute time indices.
    """

    def __init__(self, embed_dim, num_heads, batch_first=True):
        super().__init__(embed_dim, num_heads, batch_first=batch_first)

    def forward(self, query, key, value, times: torch.Tensor, **kwargs):
        """
        Expects:
          query, key, value: [B, T, embed_dim]
          times: [T] (absolute times)
        """
        B, T, _ = query.shape
        # 1) Project Q,K,V
        q, k, v = self.in_proj_qkv(query)

        # 2) Reshape Q,K to [B, T, num_heads, head_dim]
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)

        # 3) Apply global RoPE rotation
        q = rope_rotation(q, times)  # pass absolute times
        k = rope_rotation(k, times)

        # 4) Flatten back to [B, T, embed_dim]
        q = q.view(B, T, self.embed_dim)
        k = k.view(B, T, self.embed_dim)

        # 5) Forward to standard dot-product attention
        attn_output, attn_weights = super().forward(q, k, value, **kwargs)
        return attn_output, attn_weights
