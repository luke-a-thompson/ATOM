import math
from typing import final, override
import torch
import torch.nn as nn


@final
class TemporalRoPE(nn.Module):
    """
    Temporal Rotary Positional Embedding (T-RoPE).

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

    Attributes
    ----------
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
        2. Compute cos/sin embeddings for `num_timesteps`.
        3. Apply RoPE by rotating even/odd tensor components using the cos/sin values.
        4. Handle masking for padded nodes if a mask is provided.

    Output tensor shape: `[B, n_heads, seq_len, d_head]`
    """

    def __init__(
        self,
        num_timesteps: int,
        d_head: int,
        n_heads: int,
        base: float = 1000.0,
        tau: float = 1000.0,
    ):
        super().__init__()
        assert d_head % 2 == 0, "d_head must be even for standard RoPE."

        self.num_timesteps = num_timesteps
        self.d_head = d_head
        self.n_heads = n_heads
        self.base = base
        self.tau = float(tau)

        self.half_dim = d_head // 2

        self.freqs = (
            (1.0 / (self.base ** (2 * torch.arange(0, self.half_dim).float() / d_head)))
            .unsqueeze(0)
            .unsqueeze(0)
        )  # [1, 1, half_dim]

    @override
    def forward(
        self,
        tensor: torch.Tensor,
        mask: torch.Tensor | None,
        time_increments: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        assert seq_len % self.num_timesteps == 0, (
            f"seq_len={seq_len} must be divisible by num_timesteps={self.num_timesteps}."
        )

        # 1) Build cumulative times per timestep from increments, then repeat per node
        # time_increments expected shape: [B, T] of per-step increments Δt_i
        # Use exclusive cumsum to get times: [0, Δt_1, Δt_1+Δt_2, ...]
        if time_increments is None:
            # Fallback to unit increments => times = [0,1,2,...,T-1]
            B = tensor.shape[0]
            times_exclusive = (
                torch.arange(
                    self.num_timesteps, device=tensor.device, dtype=torch.float32
                )
                .unsqueeze(0)
                .expand(B, -1)
            )  # [B,T]
        else:
            # Ensure float for angle computation
            times_exclusive = (
                torch.cumsum(time_increments.to(tensor.device).float(), dim=1)
                - time_increments.to(tensor.device).float()
            )  # [B,T]

        # Repeat each timestep time across nodes, then flatten
        # times_grid: [B, T, N]
        times_grid = times_exclusive.unsqueeze(-1).expand(-1, -1, num_nodes)
        positions = times_grid.reshape(times_grid.shape[0], -1)  # [B, seq_len]

        # 2) Construct angles with scaling by tau and frequencies
        #    angle[b, pos, k] = (positions[b, pos] / tau) * freqs[k]
        angle = (positions / max(self.tau, 1e-12)).unsqueeze(-1) * self.freqs.to(
            tensor.device
        )  # [B, seq_len, half_dim]

        # 3) cos, sin => each [B, seq_len, half_dim]
        cos_t = angle.cos()
        sin_t = angle.sin()

        # 4) Expand cos_t/sin_t to [B, H, seq_len, half_dim]
        cos_t = cos_t.unsqueeze(1).expand(-1, H, -1, -1)
        sin_t = sin_t.unsqueeze(1).expand(-1, H, -1, -1)

        # Avoid rotating padded nodes. Mask.shape = [B, T*N, 1]
        if mask is not None:
            mask_bool = mask.bool()
            cos_t = torch.where(mask_bool, cos_t, torch.ones_like(cos_t))
            sin_t = torch.where(mask_bool, sin_t, torch.zeros_like(sin_t))

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
class RoPE(nn.Module):
    """
    Standard Rotary Positional Embedding (RoPE) with optional per-head learnable offsets.
    This module applies rotary embeddings to each position in the sequence independently.

    Parameters
    ----------
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
        - `d_head` must be even.
        - `B` = batch size
        - `n_heads` = number of attention heads

    Process:
        1. Generate sequential time indices from 0 to seq_len-1.
        2. Compute cos/sin embeddings for each position, adjusted by per-head offsets.
        3. Apply RoPE by rotating even/odd tensor components using the cos/sin values.
        4. Handle masking for padded nodes if a mask is provided.

    Output tensor shape: `[B, n_heads, seq_len, d_head]`
    """

    def __init__(
        self,
        d_head: int,
        n_heads: int,
        base: float = 1000.0,
        learnable_offset: bool = False,
    ):
        super().__init__()
        assert d_head % 2 == 0, "d_head must be even for standard RoPE."

        self.d_head = d_head
        self.n_heads = n_heads
        self.base = base

        self.half_dim = d_head // 2

        if learnable_offset:
            # Each of n_heads gets its own offset, initialised to 0
            self.offset = nn.Parameter(torch.zeros(n_heads))
        else:
            # A fixed buffer, all zeros by default
            self.register_buffer("offset", torch.zeros(n_heads), persistent=False)

        # Create freqs on CPU; move to the input device in forward
        self.freqs = (
            (1.0 / (self.base ** (2 * torch.arange(0, self.half_dim).float() / d_head)))
            .unsqueeze(0)
            .unsqueeze(0)
        )  # [1, 1, half_dim]

    @override
    def forward(self, tensor: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """
        Apply RoPE to the input tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor of shape `[B, n_heads, seq_len, d_head]`.
        mask : torch.Tensor | None
            Optional mask of shape `[B, 1, seq_len, 1]` for padded nodes.
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
        assert H == self.n_heads, f"Expected n_heads={self.n_heads}, got {H}"
        assert d_head == self.d_head, f"Expected d_head={self.d_head}, got {d_head}"

        # 1) Create integer time indices for each element in the sequence => shape [seq_len]
        positions = torch.arange(seq_len, device=tensor.device)

        # 2) Construct angles per head: shape => [H, seq_len, half_dim].
        offset_broadcast = self.offset.to(tensor.device).unsqueeze(
            -1
        )  # [H, 1], this adds the head dim
        positions_broadcast = positions.unsqueeze(0)  # [1, seq_len]
        shifted_positions = positions_broadcast + offset_broadcast
        angle = shifted_positions.unsqueeze(-1) * self.freqs.to(tensor.device)

        # 3) cos, sin => each [1, H, seq_len, half_dim]
        cos_t = angle.cos().unsqueeze(0)
        sin_t = angle.sin().unsqueeze(0)

        # 4) Expand cos_t/sin_t to [B, H, seq_len, half_dim]
        cos_t = cos_t.expand(B, -1, seq_len, self.half_dim)
        sin_t = sin_t.expand(B, -1, seq_len, self.half_dim)

        # Avoid rotating padded nodes. Mask.shape should be [B, 1, seq_len, 1] to broadcast
        if mask is not None:
            mask_bool = mask.bool()
            cos_t = torch.where(mask_bool, cos_t, torch.ones_like(cos_t))
            sin_t = torch.where(mask_bool, sin_t, torch.zeros_like(sin_t))

        # 5) Apply the rotation to the last dimension of 'tensor'
        t1 = tensor[..., 0::2]
        t2 = tensor[..., 1::2]

        rotated_0 = t1 * cos_t - t2 * sin_t
        rotated_1 = t1 * sin_t + t2 * cos_t

        rotated = torch.stack([rotated_0, rotated_1], dim=-1).view_as(tensor)

        return rotated


@final
class SinusoidalPositionalEmbedding(nn.Module):
    """
    Adds traditional sinusoidal positional embeddings to an input tensor.

    The embeddings are pre-computed and added to the input tensor in the forward pass.
    This implementation is based on the "Attention Is All You Need" paper.

    Parameters
    ----------
    d_model : int
        The dimension of the embedding vector.
    max_len : int, optional
        The maximum sequence length for which to pre-compute embeddings, by default 5000.

    Attributes
    ----------
    pe : torch.Tensor
        A buffer holding the pre-computed positional embeddings of shape `[1, max_len, d_model]`.

    Notes
    -----
    - Input tensor shape can be `[B, seq_len, d_model]` or `[B, T, N, d_model]`.
    - If the input is 4D `[B, T, N, d]`, it is flattened to `[B, T*N, d]` before adding PE,
      and then reshaped to its original dimensions.
    - Output tensor has the same shape as the input tensor.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.pe: torch.Tensor

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer("pe", pe, persistent=False)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional embeddings to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `[B, seq_len, d_model]` or `[B, T, N, d_model]`.

        Returns
        -------
        torch.Tensor
            Tensor with added positional embeddings, with the same shape as the input.

        Raises
        ------
        ValueError
            If the sequence length of the input tensor exceeds `max_len`.
        """
        original_shape = x.shape
        is_4d = x.dim() == 4
        if is_4d:
            B, T, N, d = x.shape
            x = x.view(B, T * N, d)

        seq_len = x.shape[1]
        if seq_len > self.pe.shape[1]:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len {self.pe.shape[1]}"
            )

        # Add positional embedding using broadcasting
        x = x + self.pe[:, :seq_len].to(x.device)

        if is_4d:
            x = x.view(original_shape)

        return x
