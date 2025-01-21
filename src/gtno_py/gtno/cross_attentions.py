from typing import final, override
import torch
import torch.nn as nn
from gtno_py.gtno.shape_utils import flatten_spatiotemporal, unflatten_spatiotemporal
from gtno_py.utils import get_context

@final
class QuadraticHeterogenousCrossAttention(nn.Module):

@final
class SubQuadraticHeterogenousCrossAttention(nn.Module):
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
        This is the heterogenous cross attention. We generate queries from the trunk data (nodes * timesteps) and perform cross attention by generating KV from the heterogeneous features.

        Parameters:
            batch: The batch dictionary containing the node features, edge features, graph features, etc.
            q_data: The query data for generating the query for cross attention.

        Returns:
            The updated batch dictionary with the heterogenous cross attention applied to x_0.
        """
        # Recall batch["x_0"] has shape [B*T, N, d (lifting dim)]
        B_times_T, N, d = batch["x_0"].shape
        B = B_times_T // self.num_timesteps  # Get the batch size from B*T

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
        q_data = flatten_spatiotemporal(q_data, self.num_timesteps)  # [B*T, N, d] -> [B, T*N, d]
        q_proj: torch.Tensor = self.query(q_data)  # [B, N*T, d]
        q: torch.Tensor = q_proj.view(B, self.num_heads, (N * self.num_timesteps), self.lifting_dim // self.num_heads)  # [B, num_heads, T*N, d_head]
        q: torch.Tensor = q.softmax(dim=-1)
        out = q

        # 3. Perform cross-attention over all heterogeneous inputs
        # Drop for loop, have a tensor with first dim as num_hetero_feats

        # In a way, we actually want an attention of timesteps * |V| x timesteps * |V|, then RoPE all V with the same timestep the same
        # Rope is kind of like a temporal prior, that graphs far apart in time are less relevant to eachother attentionally. This is
        # somewhat like temporal convolution in EGNO
        for i in range(self.num_hetero_feats):
            h_feat = hetero_features[i]
            # Determine the sequence length for this feature type - N_or_E = num_nodes or num_edges
            B_times_T, N_or_E, d = h_feat.shape
            T = B_times_T // B  # T = num_timesteps

            # 1) Flatten to [B, (N_or_E * T), d]
            h_feat = flatten_spatiotemporal(h_feat, T)

            # 2) Project K/V to [B, (N_or_E * T), d]
            k_proj: torch.Tensor = self.keys[i](h_feat)  # => [B, (N_or_E*T), d]
            v_proj: torch.Tensor = self.values[i](h_feat)  # => [B, (N_or_E*T), d]

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
        out = unflatten_spatiotemporal(out, self.num_timesteps)

        batch["x_0"] = out

        return batch
