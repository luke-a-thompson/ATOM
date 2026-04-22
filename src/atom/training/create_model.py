import torch.nn as nn

from atom.egno.egno_model import EGNO
from atom.egno.egno_sequential_and_rollout import EGNNSequential, EGNNRollout
from atom.atom.atom_model import ATOM
from atom.training.config_options import Datasets, ModelType
from atom.training.create_config import Config

import torch


def initialize_model(config: Config) -> nn.Module:
    """Initialize a model based on the configuration file.

    Args:
        config (Config): The configuration file.

    Returns:
        nn.Module: The initialized model.
    """
    match config.benchmark.model_type:
        case ModelType.ATOM:
            return ATOM(
                lifting_dim=config.atom_config.lifting_dim,
                norm=config.atom_config.norm,
                activation=config.atom_config.activation,
                num_layers=config.atom_config.num_layers,
                num_heads=config.atom_config.num_heads,
                attention_type=config.atom_config.heterogenous_attention_type,
                output_heads=config.atom_config.output_heads,
                delta_update=config.atom_config.delta_update,
                num_timesteps=config.dataloader.num_timesteps,
                positional_encoding=config.atom_config.positional_encoding,
                rope_base=config.atom_config.rope_base,
                rope_tau=config.atom_config.rope_tau,
                lifting_type=config.atom_config.lifting_type,
                projection_type=config.atom_config.projection_type,
                rrwp_length=config.dataloader.rrwp_length,
                value_residual_type=config.atom_config.value_residual_type,
                output_mode=config.atom_config.output_mode,
            )
        case ModelType.EGNO:
            return EGNO(
                num_node_features=2 if config.dataloader.dataset in [Datasets.md17, Datasets.rmd17, Datasets.tg80, Datasets.md22] else 1,
                num_edge_features=5 if config.dataloader.dataset in [Datasets.md17, Datasets.rmd17, Datasets.tg80, Datasets.md22] else 2,
                num_layers=config.egno_config.num_layers,
                lifting_dim=config.egno_config.lifting_dim,
                activation=config.egno_config.activation,
                use_time_conv=config.egno_config.use_time_conv,
                num_fourier_modes=config.egno_config.num_fourier_modes,
                time_embed_dim=config.egno_config.time_embed_dim,
                num_timesteps=config.dataloader.num_timesteps,
            )
        case ModelType.EGNN_S:
            return EGNNSequential(
                num_node_features=2 if config.dataloader.dataset in [Datasets.md17, Datasets.rmd17, Datasets.tg80, Datasets.md22] else 1,
                num_edge_features=5 if config.dataloader.dataset in [Datasets.md17, Datasets.rmd17, Datasets.tg80, Datasets.md22] else 2,
                num_layers=config.egnn_config.num_layers,
                lifting_dim=config.egnn_config.lifting_dim,
                activation=config.egnn_config.activation,
                time_embed_dim=config.egnn_config.time_embed_dim,
            )
        case ModelType.EGNN_R:
            return EGNNRollout(
                num_node_features=2 if config.dataloader.dataset in [Datasets.md17, Datasets.rmd17, Datasets.tg80, Datasets.md22] else 1,
                num_edge_features=5 if config.dataloader.dataset in [Datasets.md17, Datasets.rmd17, Datasets.tg80, Datasets.md22] else 2,
                num_layers=config.egnn_config.num_layers,
                lifting_dim=config.egnn_config.lifting_dim,
                activation=config.egnn_config.activation,
                time_embed_dim=config.egnn_config.time_embed_dim,
            )
        case _:
            raise ValueError(f"Invalid model type: {config.atom_config.model_type}")
