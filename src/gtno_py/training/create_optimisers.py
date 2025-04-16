import pytorch_optimizer as pt_optim
import torch
import torch.nn as nn
import torch.optim as optim

from gtno_py.training.config_options import OptimizerType, SchedulerType
from gtno_py.training.load_config import Config


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
            return pt_optim.Muon(model.parameters(), lr=config.optimizer.learning_rate, weight_decay=config.optimizer.weight_decay)
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
