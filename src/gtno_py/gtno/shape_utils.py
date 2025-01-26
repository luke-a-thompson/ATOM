import torch


def flatten_spatiotemporal(x: torch.Tensor, num_timesteps: int) -> torch.Tensor:
    """
    Takes [B*T, N, d] -> reshapes to [B, N*T, d].

    Where:
        B = batch size
        N = number of nodes
        T = number of timesteps
        d = feature dimension

    Args:
        x: [B*T, N, d]
        num_timesteps: number of timesteps

    Returns:
        [B, N*T, d]
    """
    B_times_T, N_or_E, d = x.shape
    B = B_times_T // num_timesteps

    # 1) Reshape [B*T, N, d] -> [B, T, N, d]
    x = x.view(B, num_timesteps, N_or_E, d)
    # 2) Flatten to [B, N*T, d]
    x = x.view(B, N_or_E * num_timesteps, d)
    return x


def unflatten_spatiotemporal(x: torch.Tensor, num_timesteps: int) -> torch.Tensor:
    """
    Takes [B, N*T, d] -> reshapes back to [B*T, N, d].

    Where:
        B = batch size
        N = number of nodes (or edges)
        T = number of timesteps
        d = feature dimension

    Args:
        x: [B, N*T, d]
        num_timesteps: number of timesteps

    Returns:
        [B*T, N, d]
    """
    B, NT, d = x.shape
    N_or_E = NT // num_timesteps  # derive N from total length and T

    # 1) Reshape to [B, N, T, d]
    x = x.view(B, N_or_E, num_timesteps, d)
    # 2) Flatten to [B*T, N, d]
    x = x.view(B * num_timesteps, N_or_E, d)
    return x
