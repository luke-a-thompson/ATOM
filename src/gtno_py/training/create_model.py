from gtno_py.gtno.gtno_model import GTNO
from gtno_py.training.config_options import ModelType
from gtno_py.training.load_config import Config
import torch.nn as nn


def initialize_model(config: Config) -> nn.Module:
    """Initialize a model based on the configuration file.

    Args:
        config (Config): The configuration file.

    Returns:
        nn.Module: The initialized model.
    """
    match config.model.model_type:
        case ModelType.GTNO:
            return GTNO(
                lifting_dim=config.model.lifting_dim,
                norm=config.model.norm,
                activation=config.model.activation,
                num_layers=config.model.num_layers,
                num_heads=config.model.num_heads,
                heterogenous_attention_type=config.model.heterogenous_attention_type,
                num_timesteps=config.model.num_timesteps,
                use_rope=config.model.use_rope,
                use_spherical_harmonics=config.model.use_spherical_harmonics,
                use_equivariant_lifting=config.model.use_equivariant_lifting,
                value_residual_type=config.model.value_residual_type,
                learnable_attention_denom=config.model.learnable_attention_denom,
            )
        case _:
            raise ValueError(f"Invalid model type: {config.model.model_type}")
