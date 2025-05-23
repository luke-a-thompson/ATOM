from typing import final, override
import torch
import torch.nn as nn
import torch.nn.functional as F
from atom.atom.mlps import MLP
from e3nn import o3


@final
class TemporalRoPEWithOffset(nn.Module):
    """
    Time-only Rotary Positional Embedding (RoPE) with per-head learnable offsets.

    Parameters
    ----------
    num_timesteps : int
        Number of timesteps T.
    d_head : int
        Dimension of each attention head. Must be even.
    n_heads : int
        Number of attention heads.
    base : float, optional
        Base value for RoPE frequency calculation, by default 1000.0.
    learnable_offset : bool, optional
        Whether to use learnable per-head offsets, by default False.

    Attributes
    ----------
    offset : nn.Parameter or torch.Tensor
        Learnable or fixed per-head offsets.
    freqs : torch.Tensor
        Precomputed RoPE frequencies.

    Raises
    ------
    AssertionError
        If `d_head` is not even.

    Notes
    -----
    Input tensor shape: `[B, n_heads, seq_len, d_head]`
        - `seq_len = num_nodes * num_timesteps`
        - `d_head` must be even.
        - `B` = batch size
        - `n_heads` = number of attention heads

    Process:
        1. Generate time indices such that groups of `num_nodes` share the same timestep.
        2. Compute cos/sin embeddings for `num_timesteps`, adjusted by per-head offsets.
        3. Apply RoPE by rotating even/odd tensor components using the cos/sin values.
        4. Handle masking for padded nodes if a mask is provided.

    Output tensor shape: `[B, n_heads, seq_len, d_head]`
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
    def forward(self, tensor: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """
        Apply RoPE to the input tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor of shape `[B, n_heads, seq_len, d_head]`.
            `seq_len = num_nodes * num_timesteps`.
        mask : torch.Tensor | None
            Optional mask of shape `[B, T*N, 1]` or `[B, seq_len, 1]` for padded nodes.
            If provided, padded nodes will not be rotated.

        Returns
        -------
        torch.Tensor
            Rotated tensor of the same shape `[B, n_heads, seq_len, d_head]`.

        Raises
        ------
        AssertionError
            If input tensor dimensions or `num_heads` do not match initialization.
        """
        B, H, seq_len, d_head = tensor.shape
        num_nodes = seq_len // self.num_timesteps
        assert H == self.n_heads, f"Expected n_heads={self.n_heads}, got {H}"
        assert d_head == self.d_head, f"Expected d_head={self.d_head}, got {d_head}"
        assert seq_len % self.num_timesteps == 0, f"seq_len={seq_len} must be divisible by num_timesteps={self.num_timesteps}."

        # 1) Create integer time indices for each chunk of num_nodes => shape [seq_len]
        #    e.g., times = [0,0,...,0,1,1,...,1,..., T-1, T-1,..., T-1], each repeated num_nodes times.
        times = torch.arange(self.num_timesteps, device=tensor.device).unsqueeze(1)  # [T,1]
        positions = torch.repeat_interleave(times, num_nodes, dim=1).flatten(0, 1)  # [N*T=seq_len]

        # 3) Construct angles per head: shape => [H, seq_len, half_dim].
        #    For each head i, angle_i = (positions + offset[i]) * freqs
        #    Broadcast offset[i] across all positions.
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

        # Avoid rotating padded nodes. Mask.shape = [B, T*N, 1]
        if mask is not None:
            cos_t = torch.where(mask, cos_t, torch.ones_like(cos_t))
            sin_t = torch.where(mask, sin_t, torch.zeros_like(sin_t))

        # 6) Apply the rotation to the last dimension of 'tensor'
        #    Even indices => [0::2], odd => [1::2]
        t1 = tensor[..., 0::2]  # [B, H, seq_len, half_dim]
        t2 = tensor[..., 1::2]  # [B, H, seq_len, half_dim]

        rotated_0 = t1 * cos_t - t2 * sin_t
        rotated_1 = t1 * sin_t + t2 * cos_t

        # Re-interleave - view_as does the interleaving
        # [B, H, seq_len, d_head]
        rotated = torch.stack([rotated_0, rotated_1], dim=-1).view_as(tensor)

        return rotated


@final
class SphericalHarmonicsAttentionBias(nn.Module):
    """
    Computes a bias for attention logits from relative node coordinates.

    For each pair of sequence elements (flattened over nodes and time), the module
    computes the relative difference, encodes it using spherical harmonics up to a
    chosen maximum degree, and then maps the concatenated coefficients through
    an MLP to produce a scalar bias per head.

    Parameters
    ----------
    num_timesteps : int
        Number of timesteps T.
    max_degree : int
        Maximum spherical harmonics degree (l) to compute.
    num_heads : int
        Number of attention heads.
    hidden_dim : int
        Hidden dimension in the intermediate MLP.

    Attributes
    ----------
    mlp : MLP
        MLP to process spherical harmonics coefficients.
    eps : float
        Small epsilon value for numerical stability.
    """

    def __init__(self, num_timesteps: int, max_degree: int, num_heads: int, hidden_dim: int):
        super().__init__()
        self.max_degree = max_degree
        self.num_heads = num_heads
        # Total number of coefficients from l=0 to max_degree.
        self.num_coeff = sum(2 * l + 1 for l in range(max_degree + 1))
        self.num_timesteps = num_timesteps
        self.mlp = MLP(
            in_dim=self.num_coeff,
            hidden_dim=hidden_dim,
            out_dim=num_heads,
            hidden_layers=2,
            activation=nn.SiLU(),
            dropout_p=0.1,
        )
        self.eps = 1e-6

    @override
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute spherical harmonics attention bias.

        Parameters
        ----------
        coords : torch.Tensor
            Coordinates tensor of shape `[B, T, N, 3]`.
            - `B` = batch size
            - `T` = number of timesteps
            - `N` = number of nodes
            - `3` = x, y, z coordinates

        Returns
        -------
        torch.Tensor
            Bias tensor of shape `[B, num_heads, S, S]` to be added to attention logits,
            where `S = N * T`.
        """
        coords = coords.clone()[..., :3]  # Ensure we only have x,y,z and clone
        B, T, N, _ = coords.shape
        S = N * T  # Total sequence length after flattening time and nodes

        # --- FIX START ---
        # Flatten the time and node dimensions correctly
        # Input coords is [B, T, N, 3], reshape to [B, N*T, 3]
        coords = coords.view(B, S, 3)
        # --- FIX END ---

        # Re-extract shape after reshape just to be safe, although S is already calculated
        B_check, S_check, _ = coords.shape
        assert B == B_check and S == S_check, f"Shape mismatch after reshape: expected ({B},{S},3), got {coords.shape}"

        # Compute pairwise relative differences: r_ij = coords_i - coords_j.
        relative_distance: torch.Tensor = coords.unsqueeze(2) - coords.unsqueeze(1)  # [B, S, S, 3]

        # Compute the norm (magnitude) and normalized direction.
        norm: torch.Tensor = relative_distance.norm(dim=-1, keepdim=True)  # [B, S, S, 1]
        unit_rel = relative_distance / (norm + self.eps)  # [B, S, S, 3]

        sh_features = []
        # For each degree l = 0, 1, ..., max_degree, compute spherical harmonics.
        for l in range(self.max_degree + 1):
            # o3.spherical_harmonics returns shape [B, S, S, 2l+1].
            Y_l = o3.spherical_harmonics(l, unit_rel, normalize=True)
            sh_features.append(Y_l)

        # Concatenate coefficients over l to form shape [B, S, S, num_coeff].
        sh_cat = torch.cat(sh_features, dim=-1)

        # Map the concatenated coefficients to a bias per head.
        bias: torch.Tensor = self.mlp(sh_cat)  # [B, S, S, num_heads]

        # Rearrange to [B, num_heads, S, S].
        bias = bias.permute(0, 3, 1, 2)

        return bias


@final
class QuadraticHeterogenousCrossAttention(nn.Module):
    def __init__(
        self,
        num_hetero_feats: int,
        lifting_dim: int,
        num_heads: int,
        num_timesteps: int,
        use_rope: bool,
        rope_base: float,
        use_spherical_harmonics: bool,
        learnable_attention_denom: bool = False,
        attention_dropout: float = 0.2,
    ) -> None:
        """
        Heterogenous graph cross attention.

        Constructs separate K/V projections for each heterogeneous feature,
        then performs cross attention on queries generated from the q_data ("trunk").

        RoPE is optional; if use_rope=True, it is applied to Q and K.

        Parameters
        ----------
        num_hetero_feats : int
            Number of heterogeneous features.
        lifting_dim : int
            Dimension for Q, K, V.
        num_heads : int
            Number of attention heads.
        num_timesteps : int
            Number of timesteps, used for RoPE and spherical harmonics.
        use_rope : bool
            If True, apply RoPE to Q and K.
        rope_base : float
            Base for RoPE calculations.
        use_spherical_harmonics : bool
            If True, add spherical harmonics bias to attention scores.
        learnable_attention_denom : bool, optional
            If True, the attention denominator (sqrt(d_head)) is learnable,
            by default False.
        attention_dropout : float, optional
            Dropout rate for attention weights, by default 0.2.

        Attributes
        ----------
        key : nn.Linear
            Linear layer for key projection.
        value : nn.Linear
            Linear layer for value projection.
        query : nn.Linear
            Linear layer for query projection.
        out_proj : nn.Linear
            Linear layer for output projection.
        attention_denom : nn.Parameter or torch.Tensor
            Attention denominator.
        feature_weights : nn.Parameter
            Learnable weights for gating heterogeneous features.
        rope : TemporalRoPEWithOffset, optional
            RoPE module.
        spherical_harmonics : SphericalHarmonicsAttentionBias, optional
            Spherical harmonics bias module.

        Raises
        ------
        AssertionError
            If `d_head` (lifting_dim / num_heads) is not even.
        """
        super().__init__()

        self.num_heads = num_heads
        self.num_hetero_feats = num_hetero_feats
        self.lifting_dim = lifting_dim
        self.num_timesteps = num_timesteps
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.use_spherical_harmonics = use_spherical_harmonics
        self.d_head = self.lifting_dim // self.num_heads

        assert self.d_head % 2 == 0, "d_head must be even"

        from e3nn import o3
        from atom.atom.atom_model import get_lifting_dim_irreps

        # lifting_dim_irreps = get_lifting_dim_irreps(self.lifting_dim)
        # self.key = o3.Linear(lifting_dim_irreps, lifting_dim_irreps)
        # self.value = o3.Linear(lifting_dim_irreps, lifting_dim_irreps)
        # self.query = o3.Linear(lifting_dim_irreps, lifting_dim_irreps)
        # self.out_proj = o3.Linear(lifting_dim_irreps, lifting_dim_irreps)

        self.key = nn.Linear(lifting_dim, lifting_dim)
        self.value = nn.Linear(lifting_dim, lifting_dim)
        self.query = nn.Linear(lifting_dim, lifting_dim)
        self.out_proj = nn.Linear(lifting_dim, lifting_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)

        denom_init = torch.full((num_heads,), float(self.d_head))
        if learnable_attention_denom:
            self.attention_denom = nn.Parameter(denom_init)
        else:
            self.register_buffer("attention_denom", denom_init, persistent=False)

        self.feature_weights = nn.Parameter(torch.randn(self.num_hetero_feats) * 0.1)

        if use_rope:
            self.rope = TemporalRoPEWithOffset(num_timesteps=self.num_timesteps, d_head=self.d_head, n_heads=self.num_heads, base=self.rope_base, learnable_offset=False)

        if use_spherical_harmonics:
            self.spherical_harmonics = SphericalHarmonicsAttentionBias(num_timesteps=self.num_timesteps, max_degree=1, num_heads=self.num_heads, hidden_dim=16)

    @override
    def forward(self, x_0: torch.Tensor, v_0: torch.Tensor | None, concatenated_features: torch.Tensor | None, q_data: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Performs heterogeneous cross-attention with multiple feature types.

        Parameters
        ----------
        x_0 : torch.Tensor
            Position features of shape `[B, T, N, d]`.
        v_0 : torch.Tensor | None
            Velocity features of shape `[B, T, N, d]` or None.
        concatenated_features : torch.Tensor | None
            Additional features of shape `[B, T, N, d]` or None.
        q_data : torch.Tensor
            Query data of shape `[B, T, N, d]`.
        mask : torch.Tensor | None, optional
            Mask of shape `[B, T, N, 1]` for padding, by default None.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `[B, T, N, d]`.

        Notes
        -----
        Process:
            1. Flatten query data from `[B, T, N, d]` to `[B, T*N, d]`.
            2. Project query to `[B, heads, T*N, d_head]`.
            3. For each heterogeneous feature (x_0, v_0, concatenated_features):
               - Project to K/V of shape `[B, heads, T*N, d_head]`.
               - Apply RoPE if enabled.
               - Compute attention scores `Q·K^T / attention_denom`.
               - Apply mask and spherical harmonics bias if enabled.
               - Compute attention weights and multiply by V.
               - Gate and accumulate to output.
            4. Reshape output to `[B, T, N, d]`.
        """
        # Flatten Q data: [B, T, N, d] -> [B, N * T (seq_q), d]
        B, T, N, d = q_data.shape
        q_data_flat = q_data.view(B, T * N, d)

        key_mask_for_scores: torch.Tensor | None = None
        rope_mask_for_rope: torch.Tensor | None = None
        if mask is not None:
            # Mask in shape: [B, T, N, 1]; need to mask attention of shape [B, heads, T*N, T*N]
            assert mask.shape == (B, T, N, 1), f"Expected mask shape (B,T,N,1) but got {mask.shape}"
            reshaped_mask = mask.reshape(B, T * N)
            key_mask_for_scores = reshaped_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T*N] for attention scores
            rope_mask_for_rope = reshaped_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, T*N, 1] for RoPE

        # Project Q => [B, heads, seq_q, d_head]
        q_proj: torch.Tensor = self.query(q_data_flat).view(B, T * N, self.num_heads, self.d_head).permute(0, 2, 1, 3)  # [B, heads, seq_q, d_head]

        if self.use_rope:
            q_proj = self.rope(q_proj, rope_mask_for_rope)

        spherical_harmonics_bias: torch.Tensor | None = None
        if self.use_spherical_harmonics:
            # Assuming x_0 is always available when spherical harmonics are used.
            spherical_harmonics_bias = self.spherical_harmonics(x_0[..., :3])

        # We'll accumulate over multiple heterogeneous features
        accumulated_out = torch.zeros_like(q_proj)

        # Collect the features of shape [B, N*T, d]
        hetero_features: list[torch.Tensor | None] = [
            x_0.view(B, T * N, d) if x_0 is not None else None,  # Flatten features if they exist
            v_0.view(B, T * N, d) if v_0 is not None else None,
            concatenated_features.view(B, T * N, d) if concatenated_features is not None else None,
        ]
        assert len(hetero_features) == self.num_hetero_feats

        gates = F.softmax(self.feature_weights, dim=0)  # Precompute gates; ∑ gates = 1
        for i, h_feat_flat in enumerate(hetero_features):
            if h_feat_flat is None:  # Skip if feature is None
                continue

            assert h_feat_flat.shape[-1] == self.lifting_dim, f"Expected {self.lifting_dim}, got {h_feat_flat.shape[-1]}"

            # Project K and V => [B, heads, seq_k, d_head]
            k_proj_i: torch.Tensor = self.key(h_feat_flat).view(B, N * T, self.num_heads, self.d_head).permute(0, 2, 1, 3)
            v_proj_i: torch.Tensor = self.value(h_feat_flat).view(B, N * T, self.num_heads, self.d_head).permute(0, 2, 1, 3)

            if self.use_rope:
                k_proj_i = self.rope(k_proj_i, rope_mask_for_rope)

            # 1) scores = Q·K^T / sqrt(d_head)
            scores = q_proj @ k_proj_i.transpose(-2, -1) / self.attention_denom.view(1, -1, 1, 1)  # Broadcasts over heads
            if key_mask_for_scores is not None:
                # scores shape is [B, heads, seq_q, seq_k] = [B, heads, T*N, T*N]
                scores = scores.masked_fill(key_mask_for_scores == 0, float("-inf"))

            if self.use_spherical_harmonics and spherical_harmonics_bias is not None:
                scores = scores + spherical_harmonics_bias

            # 2) softmax over seq_k dimension (dim=-1)
            attn_weights: torch.Tensor = self.attention_dropout(F.softmax(scores, dim=-1))
            # 3) multiply by V
            feat_i_out = attn_weights @ v_proj_i

            # Gate
            accumulated_out = accumulated_out + gates[i] * feat_i_out

        permuted_accumulated_out = accumulated_out.permute(0, 2, 1, 3).reshape(B, T * N, self.lifting_dim)
        final_out_projection: torch.Tensor = self.out_proj(permuted_accumulated_out)
        # Unflatten => [B, T, N, d]
        final_out_reshaped = final_out_projection.view(B, T, N, self.lifting_dim)

        return final_out_reshaped


@final
class QuadraticSelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_timesteps: int,
        lifting_dim: int,
        use_rope: bool,
        use_spherical_harmonics: bool,
        learnable_attention_denom: bool = False,
        attention_dropout: float = 0.2,
    ) -> None:
        """
        Quadratic self-attention mechanism.

        Parameters
        ----------
        num_heads : int
            Number of attention heads.
        num_timesteps : int
            Number of timesteps, used for RoPE and spherical harmonics.
        lifting_dim : int
            Dimension for Q, K, V.
        use_rope : bool
            If True, apply RoPE to Q and K.
        use_spherical_harmonics : bool
            If True, add spherical harmonics bias to attention scores.
        learnable_attention_denom : bool, optional
            If True, the attention denominator (sqrt(d_head)) is learnable,
            by default False.
        attention_dropout : float, optional
            Dropout rate for attention weights, by default 0.2.

        Attributes
        ----------
        kv_projs : nn.Linear
            Linear layer for combined key and value projections.
        query : nn.Linear
            Linear layer for query projection.
        out_proj : nn.Linear
            Linear layer for output projection.
        attention_denom : nn.Parameter or torch.Tensor
            Attention denominator.
        rope : TemporalRoPEWithOffset, optional
            RoPE module.
        spherical_harmonics : SphericalHarmonicsAttentionBias, optional
            Spherical harmonics bias module.

        Raises
        ------
        AssertionError
            If `d_head` (lifting_dim / num_heads) is not even.
        """
        super().__init__()
        self.num_heads = num_heads
        self.lifting_dim = lifting_dim
        self.num_timesteps = num_timesteps
        self.use_rope = use_rope
        self.use_spherical_harmonics = use_spherical_harmonics
        self.d_head = self.lifting_dim // self.num_heads

        assert self.d_head % 2 == 0, "d_head must be even"

        self.kv_projs = nn.Linear(lifting_dim, 2 * lifting_dim)
        self.query = nn.Linear(lifting_dim, lifting_dim)
        self.out_proj = nn.Linear(lifting_dim, lifting_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)

        denom_init = torch.full((num_heads,), float(self.d_head))
        if learnable_attention_denom:
            self.attention_denom = nn.Parameter(denom_init)
        else:
            self.register_buffer("attention_denom", denom_init, persistent=False)

        if use_rope:
            self.rope = TemporalRoPEWithOffset(num_timesteps=self.num_timesteps, d_head=self.d_head, n_heads=self.num_heads, base=1000.0, learnable_offset=False)

        if use_spherical_harmonics:
            self.spherical_harmonics = SphericalHarmonicsAttentionBias(num_timesteps=self.num_timesteps, max_degree=1, num_heads=self.num_heads, hidden_dim=16)

    @override
    def forward(self, tensor: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Performs self-attention on an input tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor of shape `[B, T, N, d]`.
            - `B` = batch size
            - `T` = number of timesteps
            - `N` = number of nodes
            - `d` = feature dimension
        mask : torch.Tensor | None, optional
            Mask of shape `[B, T, N, 1]` to mask attention scores, by default None.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `[B, T, N, d]`.

        Notes
        -----
        Process:
            1. Flatten input from `[B, T, N, d]` to `[B, T*N, d]`.
            2. Project to Q, K, V of shape `[B, heads, T*N, d_head]`.
            3. Apply RoPE to Q and K if enabled.
            4. Compute attention scores `Q·K^T / attention_denom`.
            5. Apply mask and spherical harmonics bias if enabled.
            6. Compute attention weights and multiply by V.
            7. Reshape output to `[B, T, N, d]`.
        """
        B, T, N, d = tensor.shape
        tensor_flat = tensor.view(B, T * N, d)

        key_mask_for_scores: torch.Tensor | None = None
        rope_mask_for_rope: torch.Tensor | None = None
        if mask is not None:
            assert mask.shape == (B, T, N, 1), f"Expected mask shape (B,T,N,1) but got {mask.shape}"
            reshaped_mask = mask.reshape(B, T * N)
            key_mask_for_scores = reshaped_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T*N] for attention scores
            rope_mask_for_rope = reshaped_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, T*N, 1] for RoPE

        q_proj: torch.Tensor = self.query(tensor_flat).view(B, T * N, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        if self.use_rope:
            q_proj = self.rope(q_proj, rope_mask_for_rope)

        spherical_harmonics_bias: torch.Tensor | None = None
        if self.use_spherical_harmonics:
            spherical_harmonics_bias = self.spherical_harmonics(tensor[..., :3])  # Use original tensor for coords

        kv: torch.Tensor = self.kv_projs(tensor_flat)
        k_proj, v_proj = torch.chunk(kv, 2, dim=-1)
        k_proj = k_proj.view(B, N * T, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        v_proj = v_proj.view(B, N * T, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        if self.use_rope:
            k_proj = self.rope(k_proj, rope_mask_for_rope)

        scores: torch.Tensor = q_proj @ k_proj.transpose(-2, -1) / self.attention_denom.view(1, -1, 1, 1)
        if key_mask_for_scores is not None:
            scores = scores.masked_fill(key_mask_for_scores == 0, float("-inf"))

        if self.use_spherical_harmonics and spherical_harmonics_bias is not None:
            scores = scores + spherical_harmonics_bias

        attn_weights: torch.Tensor = self.attention_dropout(F.softmax(scores, dim=-1))
        processed_out = attn_weights @ v_proj

        permuted_processed_out = processed_out.permute(0, 2, 1, 3).reshape(B, T * N, self.lifting_dim)
        final_out_projection: torch.Tensor = self.out_proj(permuted_processed_out).view(B, T, N, self.lifting_dim)
        return final_out_projection
