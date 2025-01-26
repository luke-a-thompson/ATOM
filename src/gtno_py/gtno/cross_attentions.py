from typing import final, override
import torch
import torch.nn as nn
import torch.nn.functional as F
from gtno_py.gtno.shape_utils import flatten_spatiotemporal, unflatten_spatiotemporal


@final
@torch.compile
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
        rope_on: bool = True,
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
        self.keys = nn.ModuleList([nn.Linear(lifting_dim, lifting_dim) for _ in range(self.num_hetero_feats)])
        self.values = nn.ModuleList([nn.Linear(lifting_dim, lifting_dim) for _ in range(self.num_hetero_feats)])
        self.out_proj = nn.Linear(lifting_dim, lifting_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)

        self.feature_weights = nn.Parameter(torch.randn(self.num_hetero_feats) * 0.1)
        self.rescale = nn.Linear(lifting_dim, lifting_dim, bias=False)

        if self.rope_on:
            self.rope = TemporalRoPEWithOffset(num_timesteps=self.num_timesteps, d_head=self.d_head, n_heads=self.num_heads, base=1000.0, learnable_offset=False)

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
            k_proj: torch.Tensor = self.keys[i](feat_flat).view(B, seq_k, self.num_heads, self.d_head).permute(0, 2, 1, 3)
            v_proj: torch.Tensor = self.values[i](feat_flat).view(B, seq_k, self.num_heads, self.d_head).permute(0, 2, 1, 3)

            if self.rope_on:
                k_proj = self.rope(k_proj)

            # 1) scores = Q·K^T / sqrt(d_head)
            scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / self.attention_denom
            # 2) softmax over seq_k dimension (dim=-1)
            attn_weights = F.softmax(scores, dim=-1)
            dropout_attn_weights: torch.Tensor = self.attention_dropout(attn_weights)
            # 3) multiply by V
            out_i = dropout_attn_weights @ v_proj

            # Gate
            gates = F.softmax(self.feature_weights, dim=0)  # ∑ gates = 1
            out_sum = out_sum + gates[i] * out_i

        # out_sum => [B, heads, seq_q, d_head]
        # Merge heads => [B, seq_q, heads*d_head]
        out_sum = out_sum.permute(0, 2, 1, 3).reshape(B, seq_q, self.lifting_dim)

        # Final linear projection
        out_sum: torch.Tensor = self.out_proj(out_sum)

        # Unflatten => [B*T, N, d]
        out_sum = unflatten_spatiotemporal(out_sum, self.num_timesteps)

        # Store result
        batch["x_0"] = out_sum
        return batch


# First: Have model output the pairwise distances, and do MSE loss on that - Don't include it for benchmark losses
# Add Brownian noise to positions, calculate pairwise distance from the noised positions
