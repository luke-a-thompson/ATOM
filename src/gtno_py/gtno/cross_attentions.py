from typing import final, override
import torch
import torch.nn as nn
import torch.nn.functional as F
from gtno_py.gtno.shape_utils import flatten_spatiotemporal, unflatten_spatiotemporal


def apply_time_only_rope(tensor: torch.Tensor, num_timesteps: int, velocity, base: float = 1000.0) -> torch.Tensor:
    """
    Applies RoPE *only* based on time index, so that all `num_nodes`
    at the same timestep T_1 share the same rotation.

    The input shape is [B, n_heads, (num_nodes * num_timesteps), d_head].
    Internally:
      1) We build a 'positions' array of length (num_nodes * num_timesteps),
         where each group of `num_nodes` shares the same timestep index.
      2) We generate cos/sin for only num_timesteps (T).
      3) We gather cos/sin by each flattened position's time index.
      4) We apply the standard RoPE rotation on the last dimension.

    Returns the rotated tensor with the same shape.
    """
    B, H, seq_len, d_head = tensor.shape
    torch._assert(d_head % 2 == 0, "d_head must be even")
    torch._assert(seq_len % num_timesteps == 0, "seq_len must be divisible by num_timesteps. This fails silently without the assert :O.")
    num_nodes = seq_len // num_timesteps

    # 1) Build positions array: shape [num_nodes * num_timesteps]
    #    e.g., for T=8, N=11: [0,0..(11 times), 1,1..(11 times), ..., 7,7..(11 times)]
    times = torch.arange(num_timesteps, device=tensor.device).unsqueeze(1)  # [T,1]
    positions = torch.repeat_interleave(times, num_nodes)  # [N*T]
    torch._assert(positions.shape == (num_nodes * num_timesteps,), f"Expected {num_nodes * num_timesteps}, got {positions.shape}")
    torch._assert(torch.all(positions[:num_nodes] == 0), f"Expected first {num_nodes} entries to be 0, got {positions[:num_nodes]}")

    # 2) Generate cos/sin for T distinct time steps
    half_dim = d_head // 2
    freqs = 1.0 / (base ** (2 * torch.arange(0, half_dim, device=tensor.device).float() / d_head))
    # shape: [T, half_dim]
    angle = times.float() * freqs.unsqueeze(0)
    cos_t = angle.cos()
    sin_t = angle.sin()

    # 3) Gather from cos_t, sin_t for each position
    #    cos_t, sin_t: [T, half_dim], positions: [N*T]
    cos_gathered = cos_t[positions]  # => [N*T, half_dim]
    sin_gathered = sin_t[positions]  # => [N*T, half_dim]

    # 4) Broadcast to [B, H, N*T, half_dim]
    cos_gathered = cos_gathered.unsqueeze(0).unsqueeze(0).expand(B, H, seq_len, half_dim)
    sin_gathered = sin_gathered.unsqueeze(0).unsqueeze(0).expand(B, H, seq_len, half_dim)

    # 5) Apply RoPE to the last dimension
    #    Split even/odd channels
    t1 = tensor[..., 0::2]  # Even channels
    t2 = tensor[..., 1::2]  # Odd channels
    rotated_0 = t1 * cos_gathered - t2 * sin_gathered
    rotated_1 = t1 * sin_gathered + t2 * cos_gathered

    return torch.stack([rotated_0, rotated_1], dim=-1).view_as(tensor)


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

        # Query projection (applied to node embeddings)
        self.query = nn.Linear(lifting_dim, lifting_dim)

        # Keys/Values for heterogeneous features
        self.keys = nn.ModuleList([nn.Linear(lifting_dim, lifting_dim) for _ in range(self.num_hetero_feats)])
        self.values = nn.ModuleList([nn.Linear(lifting_dim, lifting_dim) for _ in range(self.num_hetero_feats)])
        self.out_proj = nn.Linear(lifting_dim, lifting_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)

        self.feature_weights = nn.Parameter(torch.randn(self.num_hetero_feats) * 0.1)
        self.rescale = nn.Linear(lifting_dim, lifting_dim, bias=False)

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
        d_head = self.lifting_dim // self.num_heads
        q_proj: torch.Tensor = self.query(q_data).view(B, seq_q, self.num_heads, d_head).permute(0, 2, 1, 3)  # [B, heads, seq_q, d_head]

        if self.rope_on:
            q_proj = apply_time_only_rope(q_proj, self.num_timesteps)

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
            k_proj: torch.Tensor = self.keys[i](feat_flat).view(B, seq_k, self.num_heads, d_head).permute(0, 2, 1, 3)
            v_proj: torch.Tensor = self.values[i](feat_flat).view(B, seq_k, self.num_heads, d_head).permute(0, 2, 1, 3)

            if self.rope_on:
                k_proj = apply_time_only_rope(k_proj, self.num_timesteps)

            # 1) scores = Q·K^T / sqrt(d_head)
            scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_head, dtype=torch.float32))
            # 2) softmax over seq_k dimension (dim=-1)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attention_dropout(attn_weights)
            # 3) multiply by V
            out_i = torch.matmul(attn_weights, v_proj)

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
