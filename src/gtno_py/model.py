from typing import final, override
import torch
import torch.nn as nn
from enum import Enum
from modules.activations import FFNActivation, ReLU2, SwiGLU


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
    The idea here is to construct a learned unified graph representation accounting for dependencies between the primary node features.
    This may learn some position-velocity dependence across time (i.e., Newton's second law).
    """

    def __init__(self, lifting_dim: int, num_heads: int, batch_first: bool = True) -> None:
        super().__init__()

        self.graph_attention = nn.MultiheadAttention(embed_dim=lifting_dim, num_heads=num_heads, batch_first=batch_first)

    @override
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        concatenated_features = batch["concatenated_features"]

        attn_output: torch.Tensor = self.graph_attention(concatenated_features, concatenated_features, concatenated_features)[0]
        batch["ConcatenatedFeatures"] = attn_output

        return batch


@final
class SplitInputMHA(nn.Module):
    """
    The idea here is to construct learned graph representations for each of the primary node features.
    """

    def __init__(self, lifting_dim: int, num_heads: int, batch_first: bool = True) -> None:
        super().__init__()

        self.position_attention = nn.MultiheadAttention(embed_dim=lifting_dim, num_heads=num_heads, batch_first=batch_first)
        self.velocity_attention = nn.MultiheadAttention(embed_dim=lifting_dim, num_heads=num_heads, batch_first=batch_first)
        self.feature_attention = nn.MultiheadAttention(embed_dim=lifting_dim, num_heads=num_heads, batch_first=batch_first)

    @override
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        positions = batch["x_0"]
        velocities = batch["v_0"]
        features = batch["Z"]

        position_attn_output: torch.Tensor = self.position_attention(positions, positions, positions)[0]
        velocity_attn_output: torch.Tensor = self.velocity_attention(velocities, velocities, velocities)[0]
        feature_attn_output: torch.Tensor = self.feature_attention(features, features, features)[0]

        batch["x_0"] = position_attn_output
        batch["v_0"] = velocity_attn_output
        batch["Z"] = feature_attn_output

        return batch


@final
class GraphHeterogenousCrossAttention(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        graph_feature_dim: int,
        num_hetero_feats: int,
        lifting_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.num_hetero_feats = num_hetero_feats
        self.lifting_dim = lifting_dim

        # Lift raw features into a unified embedding space once
        self.node_lifting = nn.Linear(node_feature_dim, lifting_dim)
        self.edge_lifting = nn.Linear(edge_feature_dim, lifting_dim)
        self.graph_lifting = nn.Linear(graph_feature_dim, lifting_dim)

        # Query projection (applied to node embeddings)
        self.query = nn.Linear(lifting_dim, lifting_dim)

        # Keys/Values for heterogeneous features
        # Each input set gets its own distinct K/V projections
        self.keys = nn.ModuleList([nn.Linear(lifting_dim, lifting_dim) for _ in range(self.num_hetero_feats)])
        self.values = nn.ModuleList([nn.Linear(lifting_dim, lifting_dim) for _ in range(self.num_hetero_feats)])

        self.out_proj = nn.Linear(lifting_dim, lifting_dim)

    @override
    def forward(self, batch: dict[str, torch.Tensor], q_data: torch.Tensor) -> dict[str, torch.Tensor]:
        B, N, D = batch["node_features"].shape
        # Already lifted at model init (first fwd pass)
        lifted_nodes: torch.Tensor = batch["node_features"]  # (B, N, C)
        lifted_edges: torch.Tensor = batch["edge_features"]  # (B, E, C)
        # Graph feature shape: [B] -> [B, 1] -> [B, 1, C]
        lifted_graph: torch.Tensor = batch["energy"].unsqueeze(-1).unsqueeze(1)  # (B, 1, C)

        assert lifted_nodes.shape == (B, N, self.lifting_dim), f"Lifted nodes shape must be ({B}, {N}, {self.lifting_dim}). Got {lifted_nodes.shape}"
        assert lifted_edges.shape == (B, N, self.lifting_dim), f"Lifted edges shape must be ({B}, {N}, {self.lifting_dim}). Got {lifted_edges.shape}"
        assert lifted_graph.shape == (B, 1, self.lifting_dim), f"Lifted graph shape must be ({B}, {1}, {self.lifting_dim}). Got {lifted_graph.shape}"
        # Data for generating the query for cross attention
        assert q_data.shape == (B, N, self.lifting_dim), f"Query generating data shape must be ({B}, {N}, {self.lifting_dim}). Got {q_data.shape}"

        # Put the heterogeneous embeddings into a list to loop over
        hetero_features: list[torch.Tensor] = [lifted_nodes, lifted_edges, lifted_graph]
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

        # 4. Project output back
        out = out.transpose(1, 2).contiguous().view(B, N, self.lifting_dim)
        out = self.out_proj(out)

        batch["node_features"] = out

        return batch


@final
class IMPGTNOBlock(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        graph_feature_dim: int,
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
                    node_feature_dim=node_feature_dim,
                    edge_feature_dim=edge_feature_dim,
                    graph_feature_dim=graph_feature_dim,
                    num_hetero_feats=3,
                    lifting_dim=lifting_dim,
                    num_heads=num_heads,
                )
            case _:
                raise ValueError(
                    f"Invalid heterogenous attention type: {heterogenous_attention_type}, select from one of {GraphHeterogenousAttentionType.__members__.keys()}"
                )

    @override
    def forward(self, batch: dict[str, torch.Tensor], q_data: torch.Tensor) -> dict[str, torch.Tensor]:
        # Graph attention as message passing
        # MHA returns (attn_output, attn_weights)
        # Everything modified on each layer must be scaled / normalised
        node_features = self.pre_norm(batch["node_features"])  # We probably shouldn't normalise position/velocity right?

        # Currently this residual is broken as we are returning a graph from self.graph_attention, but node_features is a tensor
        # We can consider elementwise residual via the dict
        # We can do this if we change the batch dict to a tensor dict:
        # https://pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict.add
        graph_attended_nodes = node_features + self.graph_attention(batch)
        graph_attended_nodes = self.ffn(graph_attended_nodes)


        hetero_attended_nodes = graph_attended_nodes + self.heterogenous_attention(batch, q_data=q_data)
        hetero_attended_nodes = self.ffn(hetero_attended_nodes)

        batch["Coordinates"] = hetero_attended_nodes[:, :-3]

        return batch


@final
class IMPGTNO(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        graph_feature_dim: int,
        lifting_dim: int,
        norm: NormType,
        activation: FFNActivation,
        num_layers: int,
        num_heads: int,
        graph_attention_type: GraphAttentionType,
        heterogenous_attention_type: GraphHeterogenousAttentionType,
    ) -> None:
        super().__init__()

        self.lifting_layer = nn.Linear(in_features=node_feature_dim, out_features=lifting_dim)

        # One-time lifting to unified embedding space at model init. Keys must be in the batch dict
        match graph_attention_type:
            case GraphAttentionType.UNIFIED_MHA:
                self.elements_to_lift = ["concatenated_features"]
            case GraphAttentionType.SPLIT_MHA:
                self.elements_to_lift = ["x_0", "v_0", "Z"]
            case GraphAttentionType.GRIT:
                raise NotImplementedError("GRITAttention is not implemented")
            case _:
                raise ValueError(f"Invalid graph attention type: {graph_attention_type}, select from one of {GraphAttentionType.__members__.keys()}")

        self.layers = nn.Sequential(
            *[
                IMPGTNOBlock(
                    lifting_dim,
                    edge_feature_dim,
                    graph_feature_dim,
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

        self.projection_layer = nn.Linear(in_features=lifting_dim, out_features=node_feature_dim)

    @override
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Lift the elements we want to lift
        for key in self.elements_to_lift:
            batch[key] = self.lifting_layer(batch[key])

        for layer in self.layers:
            batch = layer(batch, q_data=batch["x_0"])

        out: torch.Tensor = self.projection_layer(batch["node_features"])
        return out[:, :, -3:]
