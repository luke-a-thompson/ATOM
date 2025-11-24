from typing import final, override

import torch
import torch.nn as nn
import torch.nn.functional as F

from atom.training.config_options import PositionalEncodingType
from atom.atom.positional_encodings import TemporalRoPE, RoPE


@final
class QuadraticHeterogenousCrossAttention(nn.Module):
    def __init__(
        self,
        lifting_dim: int,
        num_heads: int,
        num_timesteps: int,
        positional_encoding: PositionalEncodingType,
        rope_base: float,
        rope_tau: float = 1000.0,
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
        attention_denom : torch.Tensor
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
        self.lifting_dim = lifting_dim
        self.num_timesteps = num_timesteps
        self.rope_base = float(rope_base)
        self.rope_tau = float(rope_tau)
        self.d_head = self.lifting_dim // self.num_heads

        assert self.d_head % 2 == 0, "d_head must be even"

        self.key = nn.Linear(lifting_dim, lifting_dim)
        self.value = nn.Linear(lifting_dim, lifting_dim)
        self.query = nn.Linear(lifting_dim, lifting_dim)
        self.out_proj = nn.Linear(lifting_dim, lifting_dim)

        self.attention_dropout = nn.Dropout(attention_dropout)
        # Fixed attention denominator sqrt(d_head)
        self.sqrt_dhead: float = float(self.d_head) ** 0.5

        self.feature_weights = nn.Parameter(torch.randn(3) * 0.1)

        self.positional_encoding_type = positional_encoding
        self.positional_encoding: nn.Module | None = None
        match positional_encoding:
            case PositionalEncodingType.TROPE:
                self.positional_encoding = TemporalRoPE(
                    num_timesteps=self.num_timesteps,
                    d_head=self.d_head,
                    n_heads=self.num_heads,
                    base=self.rope_base,
                    tau=self.rope_tau,
                )
            case PositionalEncodingType.ROPE:
                self.positional_encoding = RoPE(
                    d_head=self.d_head,
                    n_heads=self.num_heads,
                    base=self.rope_base,
                    learnable_offset=False,
                )
            case PositionalEncodingType.SINUSOIDAL:
                self.positional_encoding = None
            case PositionalEncodingType.NONE:
                self.positional_encoding = None
            case _:
                raise ValueError(
                    f"Invalid positional encoding type: {positional_encoding}"
                )

    @override
    def forward(
        self,
        x_0: torch.Tensor,
        v_0: torch.Tensor | None,
        concatenated_features: torch.Tensor | None,
        q_data: torch.Tensor,
        mask: torch.Tensor | None,
        time_increments: torch.Tensor | None = None,
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
        """
        B, T, N, d = q_data.shape
        q_data_flat = q_data.view(B, T * N, d)

        key_mask_for_scores: torch.Tensor | None = None
        rope_mask_for_rope: torch.Tensor | None = None
        if mask is not None:
            assert mask.shape == (B, T, N, 1), (
                f"Expected mask shape (B,T,N,1) but got {mask.shape}"
            )
            reshaped_mask = mask.reshape(B, T * N)
            key_mask_for_scores = reshaped_mask.unsqueeze(1).unsqueeze(1)
            rope_mask_for_rope = reshaped_mask.unsqueeze(1).unsqueeze(-1)

        # Project Q => [B, num_heads, N*T, d_head]
        q_proj: torch.Tensor = (
            self.query(q_data_flat)
            .view(B, T * N, self.num_heads, self.d_head)
            .permute(0, 2, 1, 3)
        )

        # Apply RoPE-like PE after projection if configured
        if (
            self.positional_encoding_type == PositionalEncodingType.TROPE
            and self.positional_encoding is not None
        ):
            q_proj = self.positional_encoding(
                q_proj, rope_mask_for_rope, time_increments
            )
        elif (
            self.positional_encoding_type == PositionalEncodingType.ROPE
            and self.positional_encoding is not None
        ):
            q_proj = self.positional_encoding(q_proj, rope_mask_for_rope)

        accumulated_out = torch.zeros_like(q_proj)

        hetero_features: list[torch.Tensor | None] = [
            x_0.view(B, T * N, d) if x_0 is not None else None,
            v_0.view(B, T * N, d) if v_0 is not None else None,
            concatenated_features.view(B, T * N, d)
            if concatenated_features is not None
            else None,
        ]

        gates = F.softmax(self.feature_weights, dim=0)
        for i, h_feat_flat in enumerate(hetero_features):
            if h_feat_flat is None:
                continue

            assert h_feat_flat.shape == (B, T * N, self.lifting_dim), (
                f"Expected shape (B, T*N, d) as {(B, T * N, self.lifting_dim)} but got {h_feat_flat.shape}"
            )

            k_proj_i: torch.Tensor = (
                self.key(h_feat_flat)
                .view(B, N * T, self.num_heads, self.d_head)
                .permute(0, 2, 1, 3)
            )
            v_proj_i: torch.Tensor = (
                self.value(h_feat_flat)
                .view(B, N * T, self.num_heads, self.d_head)
                .permute(0, 2, 1, 3)
            )

            if (
                self.positional_encoding_type == PositionalEncodingType.TROPE
                and self.positional_encoding is not None
            ):
                k_proj_i = self.positional_encoding(
                    k_proj_i, rope_mask_for_rope, time_increments
                )
            elif (
                self.positional_encoding_type == PositionalEncodingType.ROPE
                and self.positional_encoding is not None
            ):
                k_proj_i = self.positional_encoding(k_proj_i, rope_mask_for_rope)

            scores = (q_proj @ k_proj_i.transpose(-2, -1)) / self.sqrt_dhead
            if key_mask_for_scores is not None:
                scores = scores.masked_fill(key_mask_for_scores == 0, float("-inf"))

            attn_weights: torch.Tensor = self.attention_dropout(
                F.softmax(scores, dim=-1)
            )
            feat_i_out = attn_weights @ v_proj_i

            accumulated_out = accumulated_out + gates[i] * feat_i_out

        permuted_accumulated_out = accumulated_out.permute(0, 2, 1, 3).reshape(
            B, T * N, self.lifting_dim
        )
        final_out_projection: torch.Tensor = self.out_proj(permuted_accumulated_out)
        assert final_out_projection.shape == (B, T * N, self.lifting_dim), (
            f"Expected (B, T*N, d) as {(B, T * N, self.lifting_dim)} but got {final_out_projection.shape}"
        )
        final_out_reshaped = final_out_projection.view(B, T, N, self.lifting_dim)

        return final_out_reshaped


@final
class LinearHeterogenousCrossAttention(nn.Module):
    def __init__(
        self,
        lifting_dim: int,
        num_heads: int,
        num_timesteps: int,
        positional_encoding: PositionalEncodingType,
        rope_base: float,
        rope_tau: float,
        attention_dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.lifting_dim = lifting_dim
        self.num_timesteps = num_timesteps
        self.rope_base = float(rope_base)
        self.d_head = self.lifting_dim // self.num_heads

        assert self.d_head % 2 == 0, "d_head must be even"

        self.key = nn.Linear(lifting_dim, lifting_dim)
        self.value = nn.Linear(lifting_dim, lifting_dim)
        self.query = nn.Linear(lifting_dim, lifting_dim)
        self.out_proj = nn.Linear(lifting_dim, lifting_dim)

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.sqrt_dhead: float = float(self.d_head) ** 0.5

        self.feature_weights = nn.Parameter(torch.randn(3) * 0.1)

        self.positional_encoding_type = positional_encoding
        self.positional_encoding: nn.Module | None = None
        match positional_encoding:
            case PositionalEncodingType.TROPE:
                self.positional_encoding = TemporalRoPE(
                    num_timesteps=self.num_timesteps,
                    d_head=self.d_head,
                    n_heads=self.num_heads,
                    base=self.rope_base,
                    tau=rope_tau,
                )
            case PositionalEncodingType.ROPE:
                self.positional_encoding = RoPE(
                    d_head=self.d_head,
                    n_heads=self.num_heads,
                    base=self.rope_base,
                    learnable_offset=False,
                )
            case PositionalEncodingType.SINUSOIDAL:
                self.positional_encoding = None
            case PositionalEncodingType.NONE:
                self.positional_encoding = None
            case _:
                raise ValueError(
                    f"Invalid positional encoding type: {positional_encoding}"
                )

    @override
    def forward(
        self,
        x_0: torch.Tensor,
        v_0: torch.Tensor | None,
        concatenated_features: torch.Tensor | None,
        q_data: torch.Tensor,
        mask: torch.Tensor | None,
        time_increments: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Flatten Q data: [B, T, N, d] -> [B, N*T, d]
        B, T, N, d = q_data.shape
        q_data_flat = q_data.view(B, T * N, d)

        rope_mask_for_rope: torch.Tensor | None = None
        if mask is not None:
            assert mask.shape == (B, T, N, 1), (
                f"Expected mask shape (B,T,N,1) but got {mask.shape}"
            )
            reshaped_mask = mask.reshape(B, T * N)
            rope_mask_for_rope = reshaped_mask.unsqueeze(1).unsqueeze(
                -1
            )  # [B, 1, T*N, 1]
        else:
            reshaped_mask = None  # type: ignore

        # No additive sinusoidal PE in attention

        # Project Q => [B, heads, seq_q, d_head]
        q_proj: torch.Tensor = (
            self.query(q_data_flat)
            .view(B, T * N, self.num_heads, self.d_head)
            .permute(0, 2, 1, 3)
        )

        # Apply RoPE-like PE after projection if configured
        if (
            self.positional_encoding_type == PositionalEncodingType.TROPE
            and self.positional_encoding is not None
        ):
            q_proj = self.positional_encoding(
                q_proj, rope_mask_for_rope, time_increments
            )
        elif (
            self.positional_encoding_type == PositionalEncodingType.ROPE
            and self.positional_encoding is not None
        ):
            q_proj = self.positional_encoding(q_proj, rope_mask_for_rope)

        # Linear attention uses softmax over feature dim for q and k
        q_lin = F.softmax(q_proj, dim=-1)
        if reshaped_mask is not None:
            query_mask_expand = reshaped_mask.unsqueeze(1).unsqueeze(
                -1
            )  # [B, 1, seq, 1]
            q_lin = q_lin * query_mask_expand

        accumulated_out = torch.zeros_like(q_lin)

        hetero_features: list[torch.Tensor | None] = [
            x_0.view(B, T * N, d) if x_0 is not None else None,
            v_0.view(B, T * N, d) if v_0 is not None else None,
            concatenated_features.view(B, T * N, d)
            if concatenated_features is not None
            else None,
        ]

        gates = F.softmax(self.feature_weights, dim=0)
        for i, h_feat_flat in enumerate(hetero_features):
            if h_feat_flat is None:
                continue

            # No additive sinusoidal PE in attention

            assert h_feat_flat.shape == (B, T * N, self.lifting_dim), (
                f"Expected shape (B, T*N, d) as {B, T * N, self.lifting_dim} but got {h_feat_flat.shape}"
            )

            k_proj_i: torch.Tensor = (
                self.key(h_feat_flat)
                .view(B, N * T, self.num_heads, self.d_head)
                .permute(0, 2, 1, 3)
            )
            v_proj_i: torch.Tensor = (
                self.value(h_feat_flat)
                .view(B, N * T, self.num_heads, self.d_head)
                .permute(0, 2, 1, 3)
            )

            if (
                self.positional_encoding_type == PositionalEncodingType.TROPE
                and self.positional_encoding is not None
            ):
                k_proj_i = self.positional_encoding(
                    k_proj_i, rope_mask_for_rope, time_increments
                )
            elif (
                self.positional_encoding_type == PositionalEncodingType.ROPE
                and self.positional_encoding is not None
            ):
                k_proj_i = self.positional_encoding(k_proj_i, rope_mask_for_rope)

            k_lin = F.softmax(k_proj_i, dim=-1)
            if reshaped_mask is not None:
                key_mask_expand = reshaped_mask.unsqueeze(1).unsqueeze(
                    -1
                )  # [B,1,seq,1]
                k_lin = k_lin * key_mask_expand
                v_proj_i = v_proj_i * key_mask_expand

            # Compute normalizer D_inv as in provided linear attention: 1 / sum_j q * sum_t k
            k_cumsum = k_lin.sum(dim=-2, keepdim=True)  # sum over sequence length
            eps: float = 1e-6
            D_inv = 1.0 / ((q_lin * k_cumsum).sum(dim=-1, keepdim=True) + eps)

            # q @ (k^T @ v) with dropout on the implicit attention weights via dropout on q
            out_i = (q_lin @ (k_lin.transpose(-2, -1) @ v_proj_i)) * D_inv
            out_i = self.attention_dropout(out_i)

            accumulated_out = accumulated_out + gates[i] * out_i

        permuted_accumulated_out = accumulated_out.permute(0, 2, 1, 3).reshape(
            B, T * N, self.lifting_dim
        )
        final_out_projection: torch.Tensor = self.out_proj(permuted_accumulated_out)
        assert final_out_projection.shape == (B, T * N, self.lifting_dim), (
            f"Expected (B, T*N, d) as {B, T * N, self.lifting_dim} but got {final_out_projection.shape}"
        )
        final_out_reshaped = final_out_projection.view(B, T, N, self.lifting_dim)
        return final_out_reshaped


def get_lifting_dim_irreps(lifting_dim: int) -> str:
    """
    Returns the irreps for the lifting dimension.
    """
    vector_lifting_dim_irreps: int = lifting_dim // 3
    scalar_lifting_dim_irreps: int = (
        lifting_dim - vector_lifting_dim_irreps * 3
    )  # Remainder

    lifting_dim_irreps: str = (
        f"{vector_lifting_dim_irreps}x1o + {scalar_lifting_dim_irreps}x0e"
    )
    return lifting_dim_irreps


@final
class QuadraticSelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_timesteps: int,
        lifting_dim: int,
        positional_encoding: PositionalEncodingType,
        rope_base: float,
        rope_tau: float = 1000.0,
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
        attention_denom : torch.Tensor
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
        self.sqrt_dhead: float = float(self.d_head) ** 0.5

        self.positional_encoding_type = positional_encoding
        self.positional_encoding: nn.Module | None = None
        self.rope_base = float(rope_base)
        self.rope_tau = float(rope_tau)
        match positional_encoding:
            case PositionalEncodingType.TROPE:
                self.positional_encoding = TemporalRoPE(
                    num_timesteps=self.num_timesteps,
                    d_head=self.d_head,
                    n_heads=self.num_heads,
                    base=self.rope_base,
                    tau=self.rope_tau,
                )
            case PositionalEncodingType.ROPE:
                self.positional_encoding = RoPE(
                    d_head=self.d_head, n_heads=self.num_heads, base=self.rope_base
                )
            case PositionalEncodingType.SINUSOIDAL:
                self.positional_encoding = None
            case PositionalEncodingType.NONE:
                self.positional_encoding = None
            case _:
                raise ValueError(
                    f"Invalid positional encoding type: {positional_encoding}"
                )

    @override
    def forward(
        self,
        tensor: torch.Tensor,
        mask: torch.Tensor | None,
        time_increments: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
            assert mask.shape == (B, T, N, 1), (
                f"Expected mask shape (B,T,N,1) but got {mask.shape}"
            )
            reshaped_mask = mask.reshape(B, T * N)
            key_mask_for_scores = reshaped_mask.unsqueeze(1).unsqueeze(
                1
            )  # [B, 1, 1, T*N] for attention scores
            rope_mask_for_rope = reshaped_mask.unsqueeze(1).unsqueeze(
                -1
            )  # [B, 1, T*N, 1] for RoPE

        if (
            self.positional_encoding_type == PositionalEncodingType.SINUSOIDAL
            and self.positional_encoding is not None
        ):
            tensor_flat = self.positional_encoding(tensor_flat)

        q_proj: torch.Tensor = (
            self.query(tensor_flat)
            .view(B, T * N, self.num_heads, self.d_head)
            .permute(0, 2, 1, 3)
        )

        if (
            self.positional_encoding_type == PositionalEncodingType.TROPE
            and self.positional_encoding is not None
        ):
            q_proj = self.positional_encoding(
                q_proj, rope_mask_for_rope, time_increments
            )
        elif (
            self.positional_encoding_type == PositionalEncodingType.ROPE
            and self.positional_encoding is not None
        ):
            q_proj = self.positional_encoding(q_proj, rope_mask_for_rope)

        kv: torch.Tensor = self.kv_projs(tensor_flat)
        k_proj, v_proj = torch.chunk(kv, 2, dim=-1)
        k_proj = k_proj.view(B, N * T, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        v_proj = v_proj.view(B, N * T, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        if (
            self.positional_encoding_type == PositionalEncodingType.TROPE
            and self.positional_encoding is not None
        ):
            k_proj = self.positional_encoding(
                k_proj, rope_mask_for_rope, time_increments
            )
        elif (
            self.positional_encoding_type == PositionalEncodingType.ROPE
            and self.positional_encoding is not None
        ):
            k_proj = self.positional_encoding(k_proj, rope_mask_for_rope)

        scores: torch.Tensor = (q_proj @ k_proj.transpose(-2, -1)) / self.sqrt_dhead
        if key_mask_for_scores is not None:
            scores = scores.masked_fill(key_mask_for_scores == 0, float("-inf"))

        attn_weights: torch.Tensor = self.attention_dropout(F.softmax(scores, dim=-1))
        processed_out = attn_weights @ v_proj

        permuted_processed_out = processed_out.permute(0, 2, 1, 3).reshape(
            B, T * N, self.lifting_dim
        )
        final_out_projection: torch.Tensor = self.out_proj(permuted_processed_out).view(
            B, T, N, self.lifting_dim
        )
        return final_out_projection


@final
class GATv2GraphAttention(nn.Module):
    """
    GATv2-style graph attention that restricts attention to edges defined by `edge_index`.

    This module operates on node embeddings of shape `[B, T, N, d]` and an edge_index
    describing the one-hop graph connectivity between nodes.
    """

    def __init__(
        self,
        lifting_dim: int,
        num_heads: int,
        num_timesteps: int,
        positional_encoding: PositionalEncodingType,
        rope_base: float,
        attention_dropout: float = 0.2,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.lifting_dim = lifting_dim
        self.num_timesteps = num_timesteps
        self.rope_base = float(rope_base)
        self.d_head = self.lifting_dim // self.num_heads

        assert self.lifting_dim % self.num_heads == 0, (
            f"lifting_dim ({lifting_dim}) must be divisible by num_heads ({num_heads})"
        )
        assert self.d_head % 2 == 0, "d_head must be even"

        # Linear projection for node features
        self.node_proj = nn.Linear(lifting_dim, lifting_dim)

        # Per-head attention vector for concatenated [h_i || h_j]
        self.attn = nn.Parameter(torch.empty(num_heads, 2 * self.d_head))

        self.out_proj = nn.Linear(lifting_dim, lifting_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

        self.positional_encoding_type = positional_encoding
        self.positional_encoding: nn.Module | None = None
        match positional_encoding:
            case PositionalEncodingType.TROPE:
                self.positional_encoding = TemporalRoPE(
                    num_timesteps=self.num_timesteps,
                    d_head=self.d_head,
                    n_heads=self.num_heads,
                    base=self.rope_base,
                )
            case PositionalEncodingType.ROPE:
                self.positional_encoding = RoPE(
                    d_head=self.d_head,
                    n_heads=self.num_heads,
                    base=self.rope_base,
                    learnable_offset=False,
                )
            case PositionalEncodingType.SINUSOIDAL:
                self.positional_encoding = None
            case PositionalEncodingType.NONE:
                self.positional_encoding = None
            case _:
                raise ValueError(
                    f"Invalid positional encoding type: {positional_encoding}"
                )

        nn.init.xavier_uniform_(self.attn)

    @override
    def forward(
        self,
        tensor: torch.Tensor,
        mask: torch.Tensor | None,
        edge_index: tuple[torch.Tensor, torch.Tensor],
        edge_mask: torch.Tensor | None = None,
        time_increments: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Perform GATv2 attention on graph-structured data.

        Args:
            tensor: Node features of shape `[B, T, N, d]`.
            mask: Optional node padding mask of shape `[B, T, N, 1]`.
            edge_index: Tuple `(src, dst)` of integer tensors, each of shape `[E]` or `[B, E]`.
            edge_mask: Optional boolean mask over edges of shape `[B, E]` indicating valid edges.
            time_increments: Optional time increments for T-RoPE of shape `[B, T]`.

        Returns:
            Tensor of shape `[B, T, N, d]` with updated node features.
        """
        B, T, N, d = tensor.shape
        assert d == self.lifting_dim, (
            f"Expected feature dim {self.lifting_dim}, got {d}"
        )

        src, dst = edge_index
        assert src.shape == dst.shape, (
            f"Source and target edge indices must have same shape, got {src.shape} and {dst.shape}"
        )

        # Normalize edge_index shapes to [B, E]
        if src.dim() == 1:
            E = int(src.shape[0])
            src_b = src.unsqueeze(0).expand(B, -1)
            dst_b = dst.unsqueeze(0).expand(B, -1)
        elif src.dim() == 2 and src.shape[0] == B:
            E = int(src.shape[1])
            src_b = src
            dst_b = dst
        else:
            msg = f"edge_index tensors must have shape [E] or [B, E]; got {src.shape}"
            raise ValueError(msg)

        # Project node features with explicit [B, T, N, d] structure
        node_proj_out = self.node_proj(tensor)  # [B, T, N, d]
        h = node_proj_out.view(B, T, N, self.num_heads, self.d_head).permute(
            0, 3, 1, 2, 4
        )
        # h: [B, heads, T, N, d_head]

        # Build mask for positional encoding over flattened [T*N] dimension
        rope_mask_for_rope: torch.Tensor | None = None
        if mask is not None:
            assert mask.shape == (B, T, N, 1), (
                f"Expected mask shape (B,T,N,1) but got {mask.shape}"
            )
            mask_flat_seq = mask.view(B, T * N, 1)
            rope_mask_for_rope = mask_flat_seq.unsqueeze(1).unsqueeze(
                -1
            )  # [B, 1, T*N, 1]

        # Apply positional encoding if configured (on [B, heads, T*N, d_head])
        if (
            self.positional_encoding_type == PositionalEncodingType.TROPE
            and self.positional_encoding is not None
        ):
            h_seq = h.reshape(B, self.num_heads, T * N, self.d_head)
            h_seq = self.positional_encoding(h_seq, rope_mask_for_rope, time_increments)
            h = h_seq.view(B, self.num_heads, T, N, self.d_head)
        elif (
            self.positional_encoding_type == PositionalEncodingType.ROPE
            and self.positional_encoding is not None
        ):
            h_seq = h.reshape(B, self.num_heads, T * N, self.d_head)
            h_seq = self.positional_encoding(h_seq, rope_mask_for_rope)
            h = h_seq.view(B, self.num_heads, T, N, self.d_head)

        # Collapse (B, T) for edge-based processing: [B, heads, T, N, d_head] -> [B*T, heads, N, d_head]
        h_proj = (
            h.permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(B * T, self.num_heads, N, self.d_head)
        )

        # Expand edges over time dimension: src_b, dst_b are [B, E]; reuse same per-node indices at each timestep
        src_bte = src_b.unsqueeze(1).expand(B, T, E)  # [B, T, E]
        dst_bte = dst_b.unsqueeze(1).expand(B, T, E)  # [B, T, E]
        src_bt = src_bte.reshape(B * T, E)
        dst_bt = dst_bte.reshape(B * T, E)

        # Gather per-edge node features: [B*T, heads, E, d_head]
        batch_indices = (
            torch.arange(B * T, device=tensor.device)
            .unsqueeze(1)
            .unsqueeze(2)
            .expand(-1, self.num_heads, E)
        )
        head_indices = (
            torch.arange(self.num_heads, device=tensor.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(B * T, -1, E)
        )
        src_gather = src_bt.unsqueeze(1).expand(-1, self.num_heads, -1)
        dst_gather = dst_bt.unsqueeze(1).expand(-1, self.num_heads, -1)

        h_src = h_proj[batch_indices, head_indices, src_gather]
        h_dst = h_proj[batch_indices, head_indices, dst_gather]

        # Concatenate and score: [B*T, heads, E, 2*d_head] -> [B*T, heads, E]
        h_cat = torch.cat([h_src, h_dst], dim=-1)
        attn_scores = (
            self.attn.view(1, self.num_heads, 1, 2 * self.d_head)
            * self.leaky_relu(h_cat)
        ).sum(dim=-1)

        # Apply edge_mask (e.g. for padded edges in multitask)
        if edge_mask is not None:
            if edge_mask.dim() == 1:
                edge_mask_b = edge_mask.unsqueeze(0).expand(B, -1)
            else:
                edge_mask_b = edge_mask
            assert edge_mask_b.shape == (B, E), (
                f"edge_mask must have shape (B, E); got {edge_mask_b.shape}"
            )
            edge_mask_bt = edge_mask_b.unsqueeze(1).expand(B, T, E).reshape(B * T, E)
            attn_scores = attn_scores.masked_fill(
                edge_mask_bt.unsqueeze(1) == 0, float("-inf")
            )

        # Build dense attention matrix [B*T, heads, N, N]
        attn_matrix = torch.full(
            (B * T, self.num_heads, N, N),
            float("-inf"),
            device=tensor.device,
            dtype=attn_scores.dtype,
        )

        batch_idx = (
            torch.arange(B * T, device=tensor.device)
            .unsqueeze(1)
            .unsqueeze(2)
            .expand(-1, self.num_heads, E)
        )
        head_idx = (
            torch.arange(self.num_heads, device=tensor.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(B * T, -1, E)
        )
        dst_idx = dst_bt.unsqueeze(1).expand(-1, self.num_heads, -1)
        src_idx = src_bt.unsqueeze(1).expand(-1, self.num_heads, -1)

        attn_matrix.index_put_(
            (batch_idx, head_idx, dst_idx, src_idx), attn_scores, accumulate=False
        )

        # Apply node mask to invalidate attention where either endpoint is padded
        if mask is not None:
            mask_flat = mask.view(B * T, N, 1)
            mask_matrix = mask_flat.unsqueeze(1) * mask_flat.unsqueeze(2)
            attn_matrix = attn_matrix.masked_fill(mask_matrix == 0, float("-inf"))

        attn_weights = self.attention_dropout(F.softmax(attn_matrix, dim=-1))

        # Aggregate messages
        out = attn_weights @ h_proj  # [B*T, heads, N, d_head]
        out = out.permute(0, 2, 1, 3).contiguous().view(B * T, N, self.lifting_dim)
        out = self.out_proj(out)
        out = out.view(B, T, N, self.lifting_dim)

        return out
