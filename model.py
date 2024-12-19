import torch
import torch.nn as nn
from enum import Enum
from modules.activations import FFNActivation, ReLU2, SwiGLU


class NormType(str, Enum):
    LAYER = "LayerNorm"
    RMS = "RMSNorm"


class GraphAttentionType(str, Enum):
    MHA = "MHA"
    GRIT = "GRIT"


class GraphHeterogenousAttentionType(str, Enum):
    GHCNA = "G-HNCA"


class GraphHeterogenousCrossAttention(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        graph_feature_dim: int,
        num_hetero_feats: int,
        lifting_dim: int,
        num_heads: int,
    ):
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
        self.keys = nn.ModuleList([nn.Linear(lifting_dim, lifting_dim) for _ in range(num_hetero_feats)])
        self.values = nn.ModuleList([nn.Linear(lifting_dim, lifting_dim) for _ in range(num_hetero_feats)])

        self.out_proj = nn.Linear(lifting_dim, lifting_dim)

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        B, N, D = batch["node_features"].shape

        # 1. Lift all features once
        lifted_nodes = self.node_lifting(batch["node_features"])  # (B, N, C)
        lifted_edges = self.edge_lifting(batch["edge_features"])  # (B, E, C)
        lifted_graph = self.graph_lifting(batch["energy"].unsqueeze(0).expand(B, -1).unsqueeze(1))  # (B, 1, C)
        assert (
            len(lifted_nodes.shape) == len(lifted_edges.shape) == len(lifted_graph.shape)
        ), f"Lifted features have inconsistent dimensionalities. Nodes: {lifted_nodes.shape}, edges: {lifted_edges.shape}, graph: {lifted_graph.shape}"
        assert (
            lifted_nodes.shape[-1] == lifted_edges.shape[-1] == lifted_graph.shape[-1]
        ), f"Last dimension must match across all lifted features.0 Nodes: {lifted_nodes.shape[-1]}, edges: {lifted_edges.shape[-1]}, graph: {lifted_graph.shape[-1]}"
        # lifted_graph = self.graph_lifting(batch["energy"]).unsqueeze(1)  # (B, 1, C)

        ### WE ARE GOOD TO HERE?

        # Put the heterogeneous embeddings into a list to loop over
        hetero_features = [lifted_nodes, lifted_edges, lifted_graph]

        # 2. Compute Q from nodes only (following your given pattern)
        q = self.query(lifted_nodes).view(B, N, self.num_heads, self.lifting_dim // self.num_heads).transpose(1, 2)
        q = q.softmax(dim=-1)

        # 3. Perform cross-attention over all heterogeneous inputs
        # Here, hetero_features = [lifted_nodes, lifted_edges, lifted_graph]
        for i in range(self.num_hetero_feats):
            h_feat = hetero_features[i]
            # Determine the sequence length for this feature type
            _, T, _ = h_feat.shape

            k = self.keys[i](h_feat).view(B, T, self.num_heads, self.lifting_dim // self.num_heads).transpose(1, 2)
            v = self.values[i](h_feat).view(B, T, self.num_heads, self.lifting_dim // self.num_heads).transpose(1, 2)

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
    ) -> None:
        super().__init__()

        self.pre_norm: nn.Module
        match norm:
            case NormType.LAYER:
                self.pre_norm = nn.LayerNorm(normalized_shape=lifting_dim)
            case NormType.RMS:
                self.pre_norm = nn.RMSNorm(normalized_shape=lifting_dim)
            case _:
                raise ValueError(f"Invalid norm type: {norm}, select from one of {NormType.__members__.keys()}")

        if lifting_dim % num_heads != 0:
            raise ValueError(f"Lifting (embedding) dim {lifting_dim} must be divisible by num_heads ({num_heads})")

        match graph_attention_type:
            case GraphAttentionType.MHA:
                self.graph_attention = nn.MultiheadAttention(
                    embed_dim=lifting_dim, num_heads=num_heads, batch_first=True
                )
            case GraphAttentionType.GRIT:
                raise NotImplementedError("GRITAttention is not implemented")
            case _:
                raise ValueError(
                    f"Invalid graph attention type: {graph_attention_type}, select from one of {GraphAttentionType.__members__.keys()}"
                )

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
                raise ValueError(
                    f"Invalid activation function: {activation}, select from one of {FFNActivation.__members__.keys()}"
                )

        self.ffn = nn.Sequential(
            nn.Linear(in_features=lifting_dim, out_features=lifting_dim),
            activation_fn,
            nn.Linear(in_features=lifting_dim, out_features=lifting_dim),
        )

        self.heterogenous_attention = GraphHeterogenousCrossAttention(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            graph_feature_dim=graph_feature_dim,
            num_hetero_feats=3,
            lifting_dim=lifting_dim,
            num_heads=num_heads,
        )

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        # Graph attention as message passing
        # MHA returns (attn_output, attn_weights)
        node_features = self.pre_norm(batch["node_features"])

        graph_attended_nodes = node_features + self.graph_attention(node_features, node_features, node_features)[0]
        graph_attended_nodes = self.ffn(graph_attended_nodes)

        hetero_attended_nodes = graph_attended_nodes + self.heterogenous_attention(batch)
        hetero_attended_nodes = self.ffn(hetero_attended_nodes)

        batch["Coordinates"] = hetero_attended_nodes[:, :-3]

        return batch


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
    ) -> None:
        super().__init__()

        self.lifting_layer = nn.Linear(in_features=node_feature_dim, out_features=lifting_dim)

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
                )
                for _ in range(num_layers)
            ]
        )

        self.projection_layer = nn.Linear(in_features=lifting_dim, out_features=node_feature_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        batch["node_features"] = self.lifting_layer(batch["node_features"])

        for layer in self.layers:
            batch = layer(batch)

        out = self.projection_layer(batch["node_features"])
        out = out[:, :-3]
        return out
