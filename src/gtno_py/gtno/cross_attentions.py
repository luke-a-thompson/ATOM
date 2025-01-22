from typing import final, override
import torch
import torch.nn as nn
import torch.nn.functional as F
from gtno_py.gtno.shape_utils import flatten_spatiotemporal, unflatten_spatiotemporal
from gtno_py.utils import get_context


def apply_time_only_rope(tensor: torch.Tensor, num_nodes: int, num_timesteps: int, base: float = 10000.0) -> torch.Tensor:
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
    assert seq_len == num_nodes * num_timesteps, f"Expected seq_len={num_nodes}*{num_timesteps}={num_nodes*num_timesteps}, got {seq_len} instead."

    # 1) Build positions array: shape [num_nodes * num_timesteps]
    #    e.g., for T=8, N=11: [0,0..(11 times), 1,1..(11 times), ..., 7,7..(11 times)]
    times = torch.arange(num_timesteps, device=tensor.device).unsqueeze(1)  # [T,1]
    positions = times.expand(num_timesteps, num_nodes).reshape(-1)  # [N*T]

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
    t1 = tensor[..., 0::2]
    t2 = tensor[..., 1::2]
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
        aggregator: str = "concat",  # or "sum"
    ) -> None:
        """
        A more standard cross-attention version:

          - Q is generated from trunk (q_data).
          - Each heterogeneous feature has distinct K, V.
          - We do standard scaled dot-product attention over seq. dimension.
          - Optionally apply RoPE to Q and K before the dot-product.

        Args:
          num_hetero_feats: number of heterogeneous features
          lifting_dim: dimension for Q/K/V
          num_heads: multi-head attention count
          num_timesteps: T dimension for flatten_spatiotemporal
          rope_on: if True, apply RoPE to Q, K
          aggregator: how to combine outputs from multiple features ("sum" or "concat")
        """
        super().__init__()

        self.num_hetero_feats = num_hetero_feats
        self.lifting_dim = lifting_dim
        self.num_heads = num_heads
        self.num_timesteps = num_timesteps
        self.rope_on = rope_on
        self.aggregator = aggregator

        assert (lifting_dim % num_heads) == 0, "lifting_dim must be divisible by num_heads"
        self.d_head = lifting_dim // num_heads

        # Query projection
        self.query_proj = nn.Linear(lifting_dim, lifting_dim)
        # K,V for each heterogeneous feature
        self.keys = nn.ModuleList([nn.Linear(lifting_dim, lifting_dim) for _ in range(num_hetero_feats)])
        self.values = nn.ModuleList([nn.Linear(lifting_dim, lifting_dim) for _ in range(num_hetero_feats)])
        # Final output projection
        if aggregator == "sum":
            self.out_proj = nn.Linear(lifting_dim, lifting_dim)
        elif aggregator == "concat":
            self.out_proj = nn.Linear(num_hetero_feats * lifting_dim, lifting_dim)
        else:
            raise ValueError("Invalid aggregator choice: must be 'sum' or 'concat'")

    @override
    def forward(self, batch: dict[str, torch.Tensor], q_data: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
          batch: dictionary containing e.g. {"x_0": [B*T, N, d], "edge_attr": [B*T, E, d], ...}
          q_data: trunk data for queries, shape [B*T, trunk_len, d]

        Returns:
          The updated batch with cross-attended representation in batch["x_0"].
        """
        B_times_T, N, d = batch["x_0"].shape
        B = B_times_T // self.num_timesteps

        # Basic checks
        assert d == self.lifting_dim, "batch['x_0'] feature dimension mismatch"
        assert q_data.shape[0] == B_times_T, "q_data must match B*T in batch"
        assert q_data.shape[-1] == self.lifting_dim, "q_data last dim must match lifting_dim"

        # Example: 2 heterogeneous features: [batch["x_0"], batch["edge_attr"]]
        hetero_features = [batch["x_0"], batch["edge_attr"]]
        assert len(hetero_features) == self.num_hetero_feats, f"Expected {self.num_hetero_feats} features"

        # 1) Flatten Q: [B*T, trunk_len, d] -> [B, trunk_len*T, d]
        q_flat = flatten_spatiotemporal(q_data, self.num_timesteps)
        seq_len_q = q_flat.shape[1]  # trunk_len * T

        # 2) Project Q -> [B, num_heads, seq_len_q, d_head]
        Q = self.query_proj(q_flat)  # [B, seq_len_q, d]
        Q = Q.view(B, seq_len_q, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        # -- Optional RoPE on Q --
        if self.rope_on:
            Q = apply_time_only_rope(Q, N, self.num_timesteps)  # N/T unique rotations

        # We'll compute cross-attention over each feature and then aggregate
        outputs = []

        # 3) For each heterogeneous feature, flatten -> project -> standard cross-attention
        for i in range(self.num_hetero_feats):
            feat = hetero_features[i]  # e.g. x_0, edge_attr
            B_times_T2, N_or_E, d_feat = feat.shape
            assert d_feat == self.lifting_dim, f"Feature {i} dim mismatch"
            # Flatten: [B*T, N_or_E, d] -> [B, (N_or_E*T), d]
            feat_flat = flatten_spatiotemporal(feat, self.num_timesteps)
            seq_len_k = feat_flat.shape[1]

            # Project K, V -> [B, num_heads, seq_len_k, d_head]
            K_i: torch.Tensor = self.keys[i](feat_flat).view(B, seq_len_k, self.num_heads, self.d_head).permute(0, 2, 1, 3)
            V_i: torch.Tensor = self.values[i](feat_flat).view(B, seq_len_k, self.num_heads, self.d_head).permute(0, 2, 1, 3)

            # -- Optional RoPE on K --
            if self.rope_on:
                K_i = apply_time_only_rope(K_i, N_or_E, self.num_timesteps)

            # 4) Standard scaled dot-product attention over seq dimension
            #    attn_scores: [B, n_heads, seq_len_q, seq_len_k]
            attn_scores = torch.matmul(Q, K_i.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_head, dtype=torch.float32))
            attn_weights = F.softmax(attn_scores, dim=-1)  # sum over seq_len_k

            # out_i: [B, n_heads, seq_len_q, d_head]
            out_i = torch.matmul(attn_weights, V_i)

            # 5) Reshape to [B, seq_len_q, d]
            out_i = out_i.permute(0, 2, 1, 3).contiguous().view(B, seq_len_q, self.lifting_dim)
            outputs.append(out_i)

        # 6) Aggregate multiple features
        if self.aggregator == "sum":
            combined = torch.stack(outputs, dim=0).sum(dim=0)  # shape [B, seq_len_q, d]
            out = self.out_proj(combined)
        else:  # "concat"
            combined = torch.cat(outputs, dim=-1)  # shape [B, seq_len_q, d * num_feats]
            out = self.out_proj(combined)

        # 7) Unflatten back to [B*T, trunk_len, d]
        out = unflatten_spatiotemporal(out, self.num_timesteps)

        # 8) Store final result
        batch["x_0"] = out
        return batch


@final
class SubQuadraticHeterogenousCrossAttention(nn.Module):
    def __init__(
        self,
        num_hetero_feats: int,
        lifting_dim: int,
        num_heads: int,
        num_timesteps: int,
        rope_on: bool = False,
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

    def forward(self, batch: dict[str, torch.Tensor], q_data: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Heterogenous cross attention with your custom sub-quadratic step.

        We generate queries from q_data, flatten everything, do the loop over features,
        and optionally apply RoPE to Q and K.

        Args:
          batch: dictionary containing ["x_0"] etc. e.g. [B*T, N, lifting_dim]
          q_data: the trunk data for queries, also [B*T, something, lifting_dim]

        Returns:
          batch with updated batch["x_0"] = the final cross-attended representation.
        """
        # batch["x_0"]: [B*T, N, d]
        B_times_T, N, d = batch["x_0"].shape
        B = B_times_T // self.num_timesteps

        assert (
            batch["x_0"].shape[-1] == self.lifting_dim and batch["edge_attr"].shape[-1] == self.lifting_dim
        ), f"Lifted dims must match. Got {batch['x_0'].shape[-1]} vs {batch['edge_attr'].shape[-1]}"

        assert (
            batch["x_0"].shape[0] == B * self.num_timesteps and batch["edge_attr"].shape[0] == B * self.num_timesteps
        ), f"Batch size mismatch: {batch['x_0'].shape[0]} vs {B * self.num_timesteps}"

        # Query data
        assert q_data.shape[0] == B * self.num_timesteps and q_data.shape[-1] == self.lifting_dim, f"Query data batch size/dim mismatch. q_data: {q_data.shape}"

        # Hetero features we will attend over
        hetero_features: list[torch.Tensor] = [batch["x_0"], batch["edge_attr"]]
        assert len(hetero_features) == self.num_hetero_feats, f"Expected {self.num_hetero_feats} hetero feats, got {len(hetero_features)}"

        # 1) Flatten Q from [B*T, N, d] -> [B, N*T, d]
        q_data = flatten_spatiotemporal(q_data, self.num_timesteps)
        # 2) Project Q and reshape to multi-head
        q_proj = self.query(q_data)  # [B, N*T, d]
        q = q_proj.view(B, self.num_heads, (N * self.num_timesteps), self.lifting_dim // self.num_heads)  # [B, num_heads, T*N, d_head]

        # Softmax over the last dim (embedding), as in your custom code
        q = q.softmax(dim=-1)

        # Optionally apply RoPE to Q
        if self.rope_on:
            q = apply_time_only_rope(q, N, self.num_timesteps)

        out = q

        # 3) Loop over heterogeneous features, flatten -> project -> custom attention update
        for i in range(self.num_hetero_feats):
            h_feat = hetero_features[i]
            B_times_T2, N_or_E, d_feat = h_feat.shape

            # Flatten to [B, (N_or_E * T), d]
            h_feat = flatten_spatiotemporal(h_feat, self.num_timesteps)

            # K/V projection
            k_proj = self.keys[i](h_feat)
            v_proj = self.values[i](h_feat)

            k = k_proj.view(B, self.num_heads, (N_or_E * self.num_timesteps), self.lifting_dim // self.num_heads)
            v = v_proj.view(B, self.num_heads, (N_or_E * self.num_timesteps), self.lifting_dim // self.num_heads)

            # Optionally apply RoPE to K before softmax, matching Q
            if self.rope_on:
                k = apply_time_only_rope(k, N_or_E, self.num_timesteps)

            k = k.softmax(dim=-1)  # also across embedding dim in your code

            # Normalisation step
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1.0 / torch.clamp_min((q * k_cumsum).sum(dim=-1, keepdim=True), 1e-8)

            # out = q + (q @ (k.transpose(-2, -1) @ v)) * D_inv
            kv = k.transpose(-2, -1) @ v  # [B, num_heads, d_head, d_head]
            out = q + (q @ kv) * D_inv

        # 4) Final projection
        out = out.transpose(1, 2).contiguous().view(B, N * self.num_timesteps, self.lifting_dim)
        out = self.out_proj(out)
        out = unflatten_spatiotemporal(out, self.num_timesteps)

        batch["x_0"] = out
        return batch
