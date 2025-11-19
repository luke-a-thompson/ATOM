import pytorch_optimizer as pt_optim
import torch
import torch.nn as nn
import torch.optim as optim

from atom.training.config_options import OptimizerType, SchedulerType
from atom.training.create_config import Config


def initialize_optimizer(config: Config, model: nn.Module) -> torch.optim.Optimizer:
    """Initialize an optimizer based on the configuration file.

    Args:
        config (Config): The configuration file.
        model (nn.Module): The model to optimize.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """
    match config.optimizer.type:
        case OptimizerType.SGD:
            return optim.SGD(model.parameters(), lr=config.optimizer.learning_rate, weight_decay=config.optimizer.weight_decay)
        case OptimizerType.ADAM:
            return optim.Adam(model.parameters(), lr=config.optimizer.learning_rate, weight_decay=config.optimizer.weight_decay)
        case OptimizerType.ADAMW:
            return optim.AdamW(
                model.parameters(),
                betas=config.optimizer.adam_betas,
                lr=config.optimizer.learning_rate,
                eps=config.optimizer.adam_eps,
                weight_decay=config.optimizer.weight_decay,
                amsgrad=True,
                fused=True,
            )
        case OptimizerType.ADAM_MINI:
            return pt_optim.AdamMini(model.parameters(), lr=config.optimizer.learning_rate, weight_decay=config.optimizer.weight_decay)
        case OptimizerType.MUON:
            # Muon requires explicit param groups with 'use_muon' set.
            muon_params = [p for p in model.parameters() if getattr(p, "ndim", 0) >= 2]
            non_muon_params = [p for p in model.parameters() if getattr(p, "ndim", 0) < 2]

            param_groups = []
            if len(muon_params) > 0:
                param_groups.append(
                    {
                        "params": muon_params,
                        "use_muon": True,
                        "lr": config.optimizer.learning_rate,
                        "weight_decay": config.optimizer.weight_decay,
                    }
                )

            if len(non_muon_params) > 0:
                param_groups.append(
                    {
                        "params": non_muon_params,
                        "use_muon": False,
                        # Use same LR unless specified otherwise in optimizer config
                        "lr": config.optimizer.learning_rate,
                        # Apply weight decay consistently
                        "weight_decay": config.optimizer.weight_decay,
                        # Respect user config for internal AdamW
                        "betas": tuple(config.optimizer.adam_betas),
                        "eps": config.optimizer.adam_eps,
                    }
                )

            return pt_optim.Muon(param_groups)
        case _:
            raise ValueError(f"Invalid optimizer type: {config.optimizer.type}")


def initialize_scheduler(config: Config, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler | None:
    """Initialize a scheduler based on the configuration file.

    Args:
        config (Config): The configuration file.
        optimizer (torch.optim.Optimizer): The optimizer to schedule.

    Returns:
        torch.optim.lr_scheduler._LRScheduler | None: The initialized scheduler.
    """
    match config.scheduler.type:
        case SchedulerType.NONE:
            return None
        case SchedulerType.STEP:
            return optim.lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.5)
        case SchedulerType.COS_ANNEALING:
            raise NotImplementedError("Cosine annealing scheduler not implemented")
        case _:
            raise ValueError(f"Invalid scheduler type: {config.scheduler.type}")
