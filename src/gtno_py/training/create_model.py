import torch.nn as nn

from gtno_py.egno.egno_model import EGNO
from gtno_py.gtno.gtno_model import GTNO
from gtno_py.training.config_options import ModelType
from gtno_py.training.load_config import Config


def initialize_model(config: Config) -> nn.Module:
    """Initialize a model based on the configuration file.

    Args:
        config (Config): The configuration file.

    Returns:
        nn.Module: The initialized model.
    """
    match config.benchmark.model_type:
        case ModelType.GTNO:
            return GTNO(
                lifting_dim=config.gtno_config.lifting_dim,
                norm=config.gtno_config.norm,
                activation=config.gtno_config.activation,
                num_layers=config.gtno_config.num_layers,
                num_heads=config.gtno_config.num_heads,
                heterogenous_attention_type=config.gtno_config.heterogenous_attention_type,
                output_heads=config.gtno_config.output_heads,
                num_timesteps=config.dataloader.num_timesteps,
                use_rope=config.gtno_config.use_rope,
                use_spherical_harmonics=config.gtno_config.use_spherical_harmonics,
                use_equivariant_lifting=config.gtno_config.use_equivariant_lifting,
                rrwp_length=config.dataloader.rrwp_length,
                value_residual_type=config.gtno_config.value_residual_type,
                learnable_attention_denom=config.gtno_config.learnable_attention_denom,
            )
        case ModelType.EGNO:
            return EGNO(
                num_layers=config.egno_config.num_layers,
                lifting_dim=config.egno_config.lifting_dim,
                activation=config.egno_config.activation,
                use_time_conv=config.egno_config.use_time_conv,
                num_fourier_modes=config.egno_config.num_fourier_modes,
                time_embed_dim=config.egno_config.time_embed_dim,
                num_timesteps=config.dataloader.num_timesteps,
            )
        case _:
            raise ValueError(f"Invalid model type: {config.gtno_config.model_type}")
