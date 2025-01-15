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

    def __init__(self, lifting_dim: int, num_heads: int, batch_first: bool = True) -> None:
        super().__init__()

        self.graph_attention = nn.MultiheadAttention(embed_dim=lifting_dim, num_heads=num_heads, batch_first=batch_first)

    @override
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        concatenated_features = batch["concatenated_features"]

        attn_output: torch.Tensor = self.graph_attention(concatenated_features, concatenated_features, concatenated_features)[0]
        batch["concatenated_features"] = attn_output

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
    ) -> None:
        """
        Heterogenous graph cross attention. We construct separate K/V projections for each heterogeneous feature, then perform cross attention on queries generated from the q_data aka "trunk".

        Features must already be lifted to the same dimension.
        """
        super().__init__()

        self.num_heads = num_heads
        self.num_hetero_feats = num_hetero_feats
        self.lifting_dim = lifting_dim

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

        B, N, D = batch["x_0"].shape
        # Already lifted at model init (first fwd pass)
        lifted_nodes: torch.Tensor = batch["x_0"]  # (B, N, C)
        lifted_edges: torch.Tensor = batch["edge_attr"]  # (B, E, C)
        # Graph feature shape: [B] -> [B, 1] -> [B, 1, C]

        # Lifted nodes
        assert (
            lifted_nodes.shape[0] == B and lifted_nodes.shape[-1] == self.lifting_dim
        ), f"{get_context(self)}: Lifted nodes batch size and feature dimension must match. Got {lifted_nodes.shape}"

        # Lifted edges
        assert (
            lifted_edges.shape[0] == B and lifted_edges.shape[-1] == self.lifting_dim
        ), f"{get_context(self)}: Lifted edges batch size and feature dimension must match. Got {lifted_edges.shape}"

        # Query data
        assert q_data.shape[0] == B and q_data.shape[-1] == self.lifting_dim, f"{get_context(self)}: Query data batch size and feature dimension must match. Got {q_data.shape}"

        # Put the heterogeneous embeddings into a list to loop over
        hetero_features: list[torch.Tensor] = [lifted_nodes, lifted_edges]
        assert (
            len(hetero_features) == self.num_hetero_feats
        ), f"Number of heterogeneous features must match the number of keys/values. Expected {self.num_hetero_feats}, got {len(hetero_features)}"

        # 2. Compute Q from nodes only (following your given pattern)
        # We compute the queries from the trunk
        q: torch.Tensor = self.query(q_data).view(B, N, self.num_heads, self.lifting_dim // self.num_heads).transpose(1, 2)
        q = q.softmax(dim=-1)

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
            _, T, _ = h_feat.shape

            # We construct a view for each head
            k: torch.Tensor = self.keys[i](h_feat).view(B, T, self.num_heads, self.lifting_dim // self.num_heads).transpose(1, 2)
            v: torch.Tensor = self.values[i](h_feat).view(B, T, self.num_heads, self.lifting_dim // self.num_heads).transpose(1, 2)

            k = k.softmax(dim=-1)

            # Normalisation step
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1.0 / torch.clamp_min((q * k_cumsum).sum(dim=-1, keepdim=True), 1e-8)
            out: torch.Tensor = q + (q @ (k.transpose(-2, -1) @ v)) * D_inv

        # 4. Project output back - DOUBLE CHECK THIS
        out = out.transpose(1, 2).contiguous().view(B, N, self.lifting_dim)
        out = self.out_proj(out)

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
    ) -> None:
        super().__init__()

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
                self.graph_attention = UnifiedInputMHA(lifting_dim, num_heads, batch_first=True)
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
                graph_attended_nodes = batch["concatenated_features"] + self.graph_attention(batch)["concatenated_features"]  # Residual connection
                batch["concatenated_features"] = self.ffn(graph_attended_nodes)
            case SplitInputMHA():
                batch["x_0"] = batch["x_0"] + self.graph_attention(batch)["x_0"]  # Residual connection (with normalised - DOUBLE CHECK THIS)
                batch["v_0"] = batch["v_0"] + self.graph_attention(batch)["v_0"]  # Residual connection

                batch["x_0"] = self.ffn(batch["x_0"])
                batch["v_0"] = self.ffn(batch["v_0"])
            case _:
                raise ValueError(f"Invalid graph attention type: {self.graph_attention}, select from one of {GraphAttentionType.__members__.keys()}")

        hetero_attended_nodes = batch["x_0"] + self.heterogenous_attention(batch, q_data=q_data)["x_0"]  # Residual connection
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
                )
                for _ in range(num_layers)
            ]
        )

        # Final projection to (x, y, z)
        self.projection_layer = nn.Linear(in_features=lifting_dim, out_features=3)

        self._initialise_weights(self)

    @override
    def forward(self, batch: TensorDict) -> torch.Tensor:
        # Batch size, number of nodes, feature dimension
        B, N, _ = batch["x_0"].shape

        batch = self._replicate_tensordict(batch, self.num_timesteps)

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
        return out

    @staticmethod
    def _initialise_weights(model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                _ = nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    _ = nn.init.zeros_(module.bias)
            

    def _replicate_tensordict(self, batch: TensorDict, T: int) -> TensorDict:
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
