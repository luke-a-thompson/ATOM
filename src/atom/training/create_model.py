import torch.nn as nn

from atom.egno.egno_model import EGNO
from atom.egno.egno_sequential_and_rollout import EGNNSequential, EGNNRollout
from atom.atom.atom_model import ATOM
from atom.training.config_options import Datasets, ModelType
from atom.training.create_config import Config


def initialize_model(config: Config) -> nn.Module:
    """Initialize a model based on the configuration file.

    Args:
        config (Config): The configuration file.

    Returns:
        nn.Module: The initialized model.
    """
    match config.benchmark.model_type:
        case ModelType.ATOM:
            atom_config = config.atom_config
            if atom_config is None:
                raise ValueError("ATOM model requires 'atom_config' to be set.")
            return ATOM(
                lifting_dim=atom_config.lifting_dim,
                norm=atom_config.norm,
                activation=atom_config.activation,
                num_layers=atom_config.num_layers,
                num_heads=atom_config.num_heads,
                attention_type=atom_config.heterogenous_attention_type,
                output_heads=atom_config.output_heads,
                delta_update=atom_config.delta_update,
                num_timesteps=config.dataloader.num_timesteps,
                positional_encoding=atom_config.positional_encoding,
                rope_base=atom_config.rope_base,
                rope_tau=atom_config.rope_tau,
                lifting_type=atom_config.lifting_type,
                projection_type=atom_config.projection_type,
                rrwp_length=config.dataloader.rrwp_length,
                value_residual_type=atom_config.value_residual_type,
                output_mode=atom_config.output_mode,
            )
        case ModelType.EGNO:
            egno_config = config.egno_config
            if egno_config is None:
                raise ValueError("EGNO model requires 'egno_config' to be set.")
            return EGNO(
                num_node_features=2 if config.dataloader.dataset in [Datasets.md17, Datasets.rmd17, Datasets.tg80, Datasets.md22] else 1,
                num_edge_features=5 if config.dataloader.dataset in [Datasets.md17, Datasets.rmd17, Datasets.tg80, Datasets.md22] else 2,
                num_layers=egno_config.num_layers,
                lifting_dim=egno_config.lifting_dim,
                activation=egno_config.activation,
                use_time_conv=egno_config.use_time_conv,
                num_fourier_modes=egno_config.num_fourier_modes,
                time_embed_dim=egno_config.time_embed_dim,
                num_timesteps=config.dataloader.num_timesteps,
            )
        case ModelType.EGNN_S:
            egnn_config = config.egnn_config
            if egnn_config is None:
                raise ValueError("EGNN_S model requires 'egnn_config' to be set.")
            return EGNNSequential(
                num_node_features=2 if config.dataloader.dataset in [Datasets.md17, Datasets.rmd17, Datasets.tg80, Datasets.md22] else 1,
                num_edge_features=5 if config.dataloader.dataset in [Datasets.md17, Datasets.rmd17, Datasets.tg80, Datasets.md22] else 2,
                num_layers=egnn_config.num_layers,
                lifting_dim=egnn_config.lifting_dim,
                activation=egnn_config.activation,
                time_embed_dim=egnn_config.time_embed_dim,
            )
        case ModelType.EGNN_R:
            egnn_config = config.egnn_config
            if egnn_config is None:
                raise ValueError("EGNN_R model requires 'egnn_config' to be set.")
            return EGNNRollout(
                num_node_features=2 if config.dataloader.dataset in [Datasets.md17, Datasets.rmd17, Datasets.tg80, Datasets.md22] else 1,
                num_edge_features=5 if config.dataloader.dataset in [Datasets.md17, Datasets.rmd17, Datasets.tg80, Datasets.md22] else 2,
                num_layers=egnn_config.num_layers,
                lifting_dim=egnn_config.lifting_dim,
                activation=egnn_config.activation,
                time_embed_dim=egnn_config.time_embed_dim,
            )
        case _:
            raise ValueError(f"Invalid model type: {config.atom_config.model_type}")
