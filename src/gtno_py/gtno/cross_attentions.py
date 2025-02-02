from typing import final, override
import torch
import torch.nn as nn
import torch.nn.functional as F
from gtno_py.gtno.mlps import MLP
from gtno_py.gtno.shape_utils import flatten_spatiotemporal, unflatten_spatiotemporal


@final
class TemporalRoPEWithOffset(nn.Module):
    """
    Time-only Rotary Positional Embedding (RoPE) with per-head learnable offsets.

    ### Input:
    - tensor: [B, n_heads, seq_len, d_head], where:
    - `seq_len = num_nodes * num_timesteps`
    - `d_head` must be even.

    ### Process:
    1. Generate time indices such that groups of `num_nodes` share the same timestep.
    2. Compute cos/sin embeddings for `num_timesteps`, adjusted by per-head offsets.
    3. Apply RoPE by rotating even/odd tensor components using the cos/sin values.

    ### Output:
    - Rotated tensor of the same shape [B, n_heads, seq_len, d_head].

    ### Features:
    - Per-head phase offsets allow temporal alignment for each attention head.
    - Consistent rotations across nodes within the same timestep.
    """

    def __init__(self, num_timesteps: int, d_head: int, n_heads: int, base: float = 1000.0, learnable_offset: bool = False):
        super().__init__()
        assert d_head % 2 == 0, "d_head must be even for standard RoPE."

        self.num_timesteps = num_timesteps
        self.d_head = d_head
        self.n_heads = n_heads
        self.base = base

        self.half_dim = d_head // 2

        if learnable_offset:
            # Each of n_heads gets its own offset, initialised to 0
            self.offset = nn.Parameter(torch.zeros(n_heads, device="cuda"))
        else:
            # A fixed buffer, all zeros by default
            self.register_buffer("offset", torch.zeros(n_heads, device="cuda"), persistent=False)

        self.freqs = (1.0 / (self.base ** (2 * torch.arange(0, self.half_dim, device=self.offset.device).float() / d_head))).unsqueeze(0).unsqueeze(0)  # [1, 1, half_dim]

    @override
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
          tensor: shape [B, n_heads, seq_len, d_head].
                  Must have seq_len divisible by num_timesteps.

        Returns:
          Rotated tensor of the same shape, with per-head time offsets applied.
        """
        B, H, seq_len, d_head = tensor.shape
        assert H == self.n_heads, f"Expected n_heads={self.n_heads}, got {H}"
        assert d_head == self.d_head, f"Expected d_head={self.d_head}, got {d_head}"
        assert seq_len % self.num_timesteps == 0, f"seq_len={seq_len} must be divisible by num_timesteps={self.num_timesteps}."
        num_nodes = seq_len // self.num_timesteps

        # 1) Create integer time indices for each chunk of num_nodes => shape [seq_len]
        #    e.g., times = [0,0,...,0,1,1,...,1,..., T-1, T-1,..., T-1], each repeated num_nodes times.
        times = torch.arange(self.num_timesteps, device=tensor.device).unsqueeze(1)  # [T,1]
        positions = torch.repeat_interleave(times, num_nodes, dim=1).flatten(0, 1)  # [N*T=seq_len]

        # 3) Construct angles per head: shape => [H, seq_len, half_dim].
        #    For each head i, angle_i = (positions + offset[i]) * freqs
        #    We'll broadcast offset[i] across all positions.
        #    offset: [H], positions: [seq_len]
        #    => positions + offset[i] => shape [H, seq_len], then multiply by freqs => shape [H, seq_len, half_dim].
        offset_broadcast = self.offset.unsqueeze(-1)  # [H, 1], this adds the head dim
        positions_broadcast = positions.unsqueeze(0)  # [1, seq_len]
        # shape => [H, seq_len]
        shifted_positions = positions_broadcast + offset_broadcast
        # shape => [H, seq_len, half_dim]
        angle = shifted_positions.unsqueeze(-1) * self.freqs

        # 4) cos, sin => each [1, H, seq_len, half_dim]
        cos_t = angle.cos().unsqueeze(0)
        sin_t = angle.sin().unsqueeze(0)

        # 5) Expand cos_t/sin_t to [B, H, seq_len, half_dim]
        cos_t = cos_t.expand(B, -1, seq_len, self.half_dim)
        sin_t = sin_t.expand(B, -1, seq_len, self.half_dim)

        # 6) Apply the rotation to the last dimension of 'tensor'
        #    Even indices => [0::2], odd => [1::2]
        t1 = tensor[..., 0::2]  # [B, H, seq_len, half_dim]
        t2 = tensor[..., 1::2]  # [B, H, seq_len, half_dim]

        rotated_0 = t1 * cos_t - t2 * sin_t
        rotated_1 = t1 * sin_t + t2 * cos_t

        # Re-interleave - view_as does the interleaving
        rotated = torch.stack([rotated_0, rotated_1], dim=-1).view_as(tensor)

        return rotated


@final
class QuadraticHeterogenousCrossAttention(nn.Module):
    def __init__(
        self,
        num_hetero_feats: int,
        lifting_dim: int,
        num_heads: int,
        num_timesteps: int,
        rope_on: bool = False,
        attention_dropout: float = 0.2,
    ) -> None:
        """
        Heterogenous graph cross attention. We construct separate K/V projections
        for each heterogeneous feature, then perform cross attention on queries
        generated from the q_data ("trunk").

        - The code is unchanged except for rope_on logic.
        - We do NOT alter your custom attention formula.
        - RoPE is optional; if rope_on=True, we apply it to Q and K in the last dimension.

        Args:
          num_hetero_feats: number of heterogeneous features
          lifting_dim: dimension for Q,K,V
          num_heads: number of attention heads
          num_timesteps: used for flatten_spatiotemporal
          rope_on: if True, apply RoPE to Q and K
          max_seq_len: maximum sequence length for precomputed RoPE
        """
        super().__init__()

        self.num_heads = num_heads
        self.num_hetero_feats = num_hetero_feats
        self.lifting_dim = lifting_dim
        self.num_timesteps = num_timesteps
        self.rope_on = rope_on
        self.d_head = self.lifting_dim // self.num_heads
        self.attention_denom = torch.sqrt(torch.tensor(self.d_head, dtype=torch.float32))

        torch._assert(self.d_head % 2 == 0, "d_head must be even")

        # Query projection (applied to node embeddings)
        self.query = nn.Linear(lifting_dim, lifting_dim)

        # Keys/Values for heterogeneous features
        self.kv_projs = nn.ModuleList([nn.Linear(lifting_dim, 2 * lifting_dim) for _ in range(num_hetero_feats)])
        self.out_proj = nn.Linear(lifting_dim, lifting_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)

        self.feature_weights = nn.Parameter(torch.randn(self.num_hetero_feats) * 0.1)
        self.rescale = nn.Linear(lifting_dim, lifting_dim, bias=False)

        if self.rope_on:
            self.rope = TemporalRoPEWithOffset(num_timesteps=self.num_timesteps, d_head=self.d_head, n_heads=self.num_heads, base=1000.0, learnable_offset=False)

        self.coord_mlp = nn.Sequential(nn.Linear(2 * lifting_dim + 1, 64), nn.ReLU(), nn.Linear(64, 1))  # outputs a single scalar
        self.alpha_coord = 0.1

    @override
    def forward(self, batch: dict[str, torch.Tensor], q_data: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        1. Flatten queries => [B, seq_q, d], then project to [B, heads, seq_q, d_head].
        2. For each heterogeneous feature, do:
           - Flatten => [B, seq_k, d],
           - Project K => [B, heads, seq_k, d_head] and V => same shape,
           - (optionally apply RoPE),
           - Compute Q·K^T, softmax over seq_k, multiply by V,
           - Gate and accumulate to `out_sum`.
        3. Reshape the final heads => [B, seq_q, d], project, unflatten => [B*T, N, d].
        4. Store result in batch["x_0"] and return.
        """
        # batch["x_0"] might be [B*T, N, d]

        # Flatten Q data: [B*T, N, d] -> [B, N*T (seq_q), d]
        q_data = flatten_spatiotemporal(q_data, self.num_timesteps)  # [B, N*T (seq_q), d]
        B, seq_q, d_q = q_data.shape

        # Project Q => [B, heads, seq_q, d_head]
        q_proj: torch.Tensor = self.query(q_data).view(B, seq_q, self.num_heads, self.d_head).permute(0, 2, 1, 3)  # [B, heads, seq_q, d_head]

        if self.rope_on:
            q_proj = self.rope(q_proj)

        # We'll accumulate over multiple heterogeneous features
        out_sum = torch.zeros_like(q_proj)  # same shape as q_proj

        # Collect the features
        hetero_features = [
            batch["x_0"],
            batch["v_0"],
            batch["edge_attr"],
            batch["concatenated_features"],
        ]
        assert len(hetero_features) == self.num_hetero_feats

        for i, h_feat in enumerate(hetero_features):
            # Flatten => [B, N_or_E * T, d]
            feat_flat = flatten_spatiotemporal(h_feat, self.num_timesteps)
            _, seq_k, d_k = feat_flat.shape
            assert d_k == self.lifting_dim, f"Expected {self.lifting_dim}, got {d_k}"

            # Project K and V => [B, heads, seq_k, d_head]
            kv = self.kv_projs[i](feat_flat)
            k_proj, v_proj = torch.chunk(kv, 2, dim=-1)
            k_proj = k_proj.view(B, seq_k, self.num_heads, self.d_head).permute(0, 2, 1, 3)
            v_proj = v_proj.view(B, seq_k, self.num_heads, self.d_head).permute(0, 2, 1, 3)

            if self.rope_on:
                k_proj = self.rope(k_proj)

            # 1) scores = Q·K^T / sqrt(d_head)
            scores = q_proj @ k_proj.transpose(-2, -1) / self.attention_denom
            # 2) softmax over seq_k dimension (dim=-1)
            attn_weights: torch.Tensor = self.attention_dropout(F.softmax(scores, dim=-1))
            # 3) multiply by V
            out_i = attn_weights @ v_proj

            # Gate
            gates = F.softmax(self.feature_weights, dim=0)  # ∑ gates = 1
            out_sum = out_sum + gates[i] * out_i

        out_sum = out_sum.permute(0, 2, 1, 3).reshape(B, seq_q, self.lifting_dim)
        out_sum = self.out_proj(out_sum)

        # Unflatten => [B*T, N, d]
        out_sum = unflatten_spatiotemporal(out_sum, self.num_timesteps)
        batch["x_0"] = out_sum  # stored result from attention

        # === 2) VECTORZIED COORDINATE UPDATE ===
        # We assume that the first 3 dimensions of batch["x_0"] are the actual 3D coordinates.
        # Let:
        #   - x_0_positions be of shape [B*T, N, 3]
        #   - node_feats be of shape [B*T, N, d] (all node features)
        x_0_positions = batch["x_0"][..., :3]  # [B*T, N, 3]
        node_feats = batch["x_0"]  # [B*T, N, d]

        # Compute all pairwise position differences: shape [B*T, N, N, 3]
        pos_diff = x_0_positions.unsqueeze(2) - x_0_positions.unsqueeze(1)
        # Compute Euclidean distances (with an extra dim): shape [B*T, N, N, 1]
        dist = pos_diff.norm(dim=-1, keepdim=True)

        # Prepare pairwise node features:
        # feats_i and feats_j: both of shape [B*T, N, N, d]
        feats_i = node_feats.unsqueeze(2).expand(-1, -1, node_feats.shape[1], -1)
        feats_j = node_feats.unsqueeze(1).expand(-1, node_feats.shape[1], -1, -1)

        # Concatenate features and distance along the last dimension: shape [B*T, N, N, 2*d + 1]
        mlp_input = torch.cat([feats_i, feats_j, dist], dim=-1)

        # Pass through the coordinate MLP to obtain a scalar weight per edge: [B*T, N, N, 1]
        msg = self.coord_mlp(mlp_input)

        # Compute the weighted sum of relative differences:
        # Multiply the relative differences by the message weights and sum over neighbors (axis=2)
        delta_x = (pos_diff * msg).sum(dim=2)  # [B*T, N, 3]

        # Update coordinates using the weighted sum:
        x_0_positions_updated: torch.Tensor = x_0_positions + self.alpha_coord * delta_x

        # Replace the coordinates in batch["x_0"] with the updated ones.
        # If the full node feature dimension exceeds 3, keep the remaining features unchanged.
        if batch["x_0"].shape[-1] > 3:
            batch["x_0"] = torch.cat([x_0_positions_updated, batch["x_0"][..., 3:]], dim=-1)
        else:
            batch["x_0"] = x_0_positions_updated

        return batch


class HamiltonianCrossAttentionVelPos(nn.Module):
    """
    Wraps QuadraticHeterogenousCrossAttention to treat x_0 and v_0 (each in embedding dimension)
    as canonical position q and momentum p in a Hamiltonian system.
    """

    def __init__(
        self,
        lifting_dim: int = 128,  # embedding dimension
        dt: float = 0.001,
        hidden_dim: int = 128,
    ):
        """
        Args:
          lifting_dim: dimension of each of x_0 and v_0 embeddings
          dt: time-step size for Hamiltonian integration
          hidden_dim: dimension for the hidden layers of the Hamiltonian MLP
        """
        super().__init__()
        self.cross_attention = QuadraticHeterogenousCrossAttention(
            num_hetero_feats=4,
            lifting_dim=lifting_dim,
            num_heads=4,
            num_timesteps=8,
        )
        self.lifting_dim = lifting_dim
        self.dt = dt

        # A small MLP to produce a scalar from concatenated x_0 and v_0 embeddings
        self.hamiltonian_mlp = MLP(in_features=2 * lifting_dim, out_features=1, hidden_features=hidden_dim, hidden_layers=3, activation=nn.SiLU())

    @override
    def forward(self, batch: dict[str, torch.Tensor], q_data: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Expects:
          batch["x_0"], batch["v_0"] each => [B*T, N, lifting_dim]
        Returns updated batch with x_0, v_0 replaced by their symplectic updates.
        """

        # Ensure x_0, v_0 require gradients BEFORE cross-attention
        batch["x_0"].requires_grad_(True)
        batch["v_0"].requires_grad_(True)

        # (A) CROSS-ATTENTION
        # Perform cross-attention to update batch["x_0"] (and optionally others)
        batch = self.cross_attention(batch, q_data=q_data)

        # Extract x_0 and v_0 from the updated batch
        x_0 = batch["x_0"]  # shape [B*T, N, lifting_dim]
        v_0 = batch["v_0"]  # shape [B*T, N, lifting_dim]

        # (B) HAMILTONIAN => H(x_0, v_0)
        # Concatenate along last dim => [B*T, N, 2*lifting_dim]
        cat_xv = torch.cat([x_0, v_0], dim=-1)

        # MLP -> scalar per node => [B*T, N, 1]
        H_nodes = self.hamiltonian_mlp(cat_xv)

        # Sum across nodes => [B*T], then sum over B*T => single scalar
        H_per_example = H_nodes.sum(dim=1)  # sum over nodes
        H = H_per_example.sum()  # total Hamiltonian scalar

        # (C) PARTIAL DERIVATIVES
        # Compute dH/dx_0 and dH/dv_0
        dH_dx0, dH_dv0 = torch.autograd.grad(H, (x_0, v_0), create_graph=True)

        # (D) SYMPLECTIC UPDATE
        # Update x_0 and v_0 using the symplectic equations
        dt = self.dt
        x_0_new = x_0 + dt * dH_dv0
        v_0_new = v_0 - dt * dH_dx0

        # (E) Store the updated embeddings
        batch["x_0"] = x_0_new
        batch["v_0"] = v_0_new

        return batch
