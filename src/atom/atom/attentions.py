from typing import final, override
import torch
import torch.nn as nn
import torch.nn.functional as F
from atom.atom.mlps import MLP
from e3nn import o3
from atom.training.config_options import PositionalEncodingType
from atom.atom.positional_encodings import TemporalRoPE, RoPE, SinusoidalPositionalEmbedding


@final
class QuadraticHeterogenousCrossAttention(nn.Module):
    def __init__(
        self,
        num_hetero_feats: int,
        lifting_dim: int,
        num_heads: int,
        num_timesteps: int,
        positional_encoding: PositionalEncodingType,
        rope_base: float,
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
        rope : TemporalRoPE, optional
            T-RoPE module.

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
        self.rope_base = rope_base
        self.d_head = self.lifting_dim // self.num_heads

        assert self.d_head % 2 == 0, "d_head must be even"

        self.key = nn.Linear(lifting_dim, lifting_dim)
        self.value = nn.Linear(lifting_dim, lifting_dim)
        self.query = nn.Linear(lifting_dim, lifting_dim)
        self.out_proj = nn.Linear(lifting_dim, lifting_dim)

        from e3nn import o3

        # self.key = o3.Linear(get_lifting_dim_irreps(lifting_dim), get_lifting_dim_irreps(lifting_dim))
        # self.value = o3.Linear(get_lifting_dim_irreps(lifting_dim), get_lifting_dim_irreps(lifting_dim))
        # self.query = o3.Linear(get_lifting_dim_irreps(lifting_dim), get_lifting_dim_irreps(lifting_dim))
        # self.out_proj = o3.Linear(get_lifting_dim_irreps(lifting_dim), get_lifting_dim_irreps(lifting_dim))

        self.attention_dropout = nn.Dropout(attention_dropout)

        denom_init = torch.full((num_heads,), float(self.d_head))
        if learnable_attention_denom:
            self.attention_denom = nn.Parameter(denom_init)
        else:
            self.register_buffer("attention_denom", denom_init, persistent=False)

        self.feature_weights = nn.Parameter(torch.randn(self.num_hetero_feats) * 0.1)

        self.positional_encoding_type = positional_encoding
        self.positional_encoding: nn.Module | None = None
        match positional_encoding:
            case PositionalEncodingType.TROPE:
                self.positional_encoding = TemporalRoPE(num_timesteps=self.num_timesteps, d_head=self.d_head, n_heads=self.num_heads, base=self.rope_base)
            case PositionalEncodingType.ROPE:
                self.positional_encoding = RoPE(d_head=self.d_head, n_heads=self.num_heads, base=self.rope_base, learnable_offset=False)
            case PositionalEncodingType.SINUSOIDAL:
                self.positional_encoding = SinusoidalPositionalEmbedding(d_model=self.lifting_dim)
            case PositionalEncodingType.NONE:
                self.positional_encoding = None
            case _:
                raise ValueError(f"Invalid positional encoding type: {positional_encoding}")

    @override
    def forward(
        self,
        x_0: torch.Tensor,
        v_0: torch.Tensor | None,
        concatenated_features: torch.Tensor | None,
        q_data: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
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
            1. Flatten query data from `[B, T, N, d]` to `[B, N * T (seq_q), d]`.
            2. Project query to `[B, heads, T*N, d_head]`.
            3. For each heterogeneous feature (x_0, v_0, concatenated_features):
               - Project to K/V of shape `[B, heads, T*N, d_head]`.
               - Apply RoPE if enabled.
               - Compute attention scores `Q·K^T / attention_denom`.
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

        # Apply sinusoidal PE before projection if configured
        if self.positional_encoding_type == PositionalEncodingType.SINUSOIDAL and self.positional_encoding is not None:
            q_data_flat = self.positional_encoding(q_data_flat)

        # Project Q => [B, num_heads, N*T, d_head]
        q_proj: torch.Tensor = self.query(q_data_flat).view(B, T * N, self.num_heads, self.d_head).permute(0, 2, 1, 3)  # [B, heads, seq_q, d_head]

        # Apply RoPE-like PE after projection if configured
        if self.positional_encoding_type in [PositionalEncodingType.ROPE, PositionalEncodingType.TROPE] and self.positional_encoding is not None:
            q_proj = self.positional_encoding(q_proj, rope_mask_for_rope)

        # We'll accumulate over multiple heterogeneous features
        accumulated_out = torch.zeros_like(q_proj)

        # Collect the features of shape [B, N*T, d]
        hetero_features: list[torch.Tensor | None] = [
            x_0.view(B, T * N, d) if x_0 is not None else None,  # Flatten features if they exist
            v_0.view(B, T * N, d) if v_0 is not None else None,
            concatenated_features.view(B, T * N, d) if concatenated_features is not None else None,
        ]
        assert len(hetero_features) == self.num_hetero_feats, f"Expected {self.num_hetero_feats} heterogeneous features but got {len(hetero_features)}"

        gates = F.softmax(self.feature_weights, dim=0)  # Precompute gates; ∑ gates = 1
        for i, h_feat_flat in enumerate(hetero_features):
            if h_feat_flat is None:  # Skip if feature is None
                continue

            # Apply sinusoidal PE to hetero features before projection
            if self.positional_encoding_type == PositionalEncodingType.SINUSOIDAL and self.positional_encoding is not None:
                h_feat_flat = self.positional_encoding(h_feat_flat)

            # h_feat_flat.shape should be [B, T*N, d]
            assert h_feat_flat.shape == (B, T * N, self.lifting_dim), f"Expected shape (B, T*N, d) as {B, T * N, self.lifting_dim} but got {h_feat_flat.shape}"

            # Project K and V => [B, heads, seq_k, d_head]
            k_proj_i: torch.Tensor = self.key(h_feat_flat).view(B, N * T, self.num_heads, self.d_head).permute(0, 2, 1, 3)
            v_proj_i: torch.Tensor = self.value(h_feat_flat).view(B, N * T, self.num_heads, self.d_head).permute(0, 2, 1, 3)

            # Apply RoPE-like PE after projection if configured
            if self.positional_encoding_type in [PositionalEncodingType.ROPE, PositionalEncodingType.TROPE] and self.positional_encoding is not None:
                k_proj_i = self.positional_encoding(k_proj_i, rope_mask_for_rope)

            # 1) scores = Q·K^T / sqrt(d_head)
            scores = q_proj @ k_proj_i.transpose(-2, -1) / self.attention_denom.view(1, -1, 1, 1)  # Broadcasts over heads
            if key_mask_for_scores is not None:
                # scores shape is [B, heads, seq_q, seq_k] = [B, heads, T*N, T*N]
                scores = scores.masked_fill(key_mask_for_scores == 0, float("-inf"))

            # 2) softmax over seq_k dimension (dim=-1)
            attn_weights: torch.Tensor = self.attention_dropout(F.softmax(scores, dim=-1))
            # 3) multiply by V
            feat_i_out = attn_weights @ v_proj_i

            # Gate
            accumulated_out = accumulated_out + gates[i] * feat_i_out

        permuted_accumulated_out = accumulated_out.permute(0, 2, 1, 3).reshape(B, T * N, self.lifting_dim)
        final_out_projection: torch.Tensor = self.out_proj(permuted_accumulated_out)
        assert final_out_projection.shape == (B, T * N, self.lifting_dim), f"Expected (B, T*N, d) as {B, T * N, self.lifting_dim} but got {final_out_projection.shape}"
        # Unflatten => [B, T, N, d]
        final_out_reshaped = final_out_projection.view(B, T, N, self.lifting_dim)

        return final_out_reshaped


def get_lifting_dim_irreps(lifting_dim: int) -> str:
    """
    Returns the irreps for the lifting dimension.
    """
    vector_lifting_dim_irreps: int = lifting_dim // 3
    scalar_lifting_dim_irreps: int = lifting_dim - vector_lifting_dim_irreps * 3  # Remainder

    lifting_dim_irreps: str = f"{vector_lifting_dim_irreps}x1o + {scalar_lifting_dim_irreps}x0e"
    return lifting_dim_irreps


@final
class QuadraticSelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_timesteps: int,
        lifting_dim: int,
        positional_encoding: PositionalEncodingType,
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

        Raises
        ------
        AssertionError
            If `d_head` (lifting_dim / num_heads) is not even.
        """
        super().__init__()
        self.num_heads = num_heads
        self.lifting_dim = lifting_dim
        self.num_timesteps = num_timesteps
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

        self.positional_encoding_type = positional_encoding
        self.positional_encoding: nn.Module | None = None
        match positional_encoding:
            case PositionalEncodingType.TROPE:
                self.positional_encoding = TemporalRoPE(num_timesteps=self.num_timesteps, d_head=self.d_head, n_heads=self.num_heads, base=1000.0)
            case PositionalEncodingType.ROPE:
                self.positional_encoding = RoPE(d_head=self.d_head, n_heads=self.num_heads, base=1000.0)
            case PositionalEncodingType.SINUSOIDAL:
                self.positional_encoding = SinusoidalPositionalEmbedding(d_model=self.lifting_dim)
            case PositionalEncodingType.NONE:
                self.positional_encoding = None
            case _:
                raise ValueError(f"Invalid positional encoding type: {positional_encoding}")

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

        if self.positional_encoding_type == PositionalEncodingType.SINUSOIDAL and self.positional_encoding is not None:
            tensor_flat = self.positional_encoding(tensor_flat)

        q_proj: torch.Tensor = self.query(tensor_flat).view(B, T * N, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        if self.positional_encoding_type in [PositionalEncodingType.ROPE, PositionalEncodingType.TROPE] and self.positional_encoding is not None:
            q_proj = self.positional_encoding(q_proj, rope_mask_for_rope)

        kv: torch.Tensor = self.kv_projs(tensor_flat)
        k_proj, v_proj = torch.chunk(kv, 2, dim=-1)
        k_proj = k_proj.view(B, N * T, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        v_proj = v_proj.view(B, N * T, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        if self.positional_encoding_type in [PositionalEncodingType.ROPE, PositionalEncodingType.TROPE] and self.positional_encoding is not None:
            k_proj = self.positional_encoding(k_proj, rope_mask_for_rope)

        scores: torch.Tensor = q_proj @ k_proj.transpose(-2, -1) / self.attention_denom.view(1, -1, 1, 1)
        if key_mask_for_scores is not None:
            scores = scores.masked_fill(key_mask_for_scores == 0, float("-inf"))

        attn_weights: torch.Tensor = self.attention_dropout(F.softmax(scores, dim=-1))
        processed_out = attn_weights @ v_proj

        permuted_processed_out = processed_out.permute(0, 2, 1, 3).reshape(B, T * N, self.lifting_dim)
        final_out_projection: torch.Tensor = self.out_proj(permuted_processed_out).view(B, T, N, self.lifting_dim)
        return final_out_projection
