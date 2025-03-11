from enum import StrEnum
from gtno_py.training.load_config import Config
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_optimizer as pt_optim
from gtno_py.training.model_options import OptimizerType, SchedulerType



def initialize_optimizer(config: Config, model: nn.Module) -> torch.optim.Optimizer:
    match config.optimizer.type:
        case OptimizerType.SGD:
            return optim.SGD(model.parameters(), lr=config.optimizer.learning_rate, weight_decay=config.optimizer.weight_decay)
        case OptimizerType.ADAMW:
            return optim.AdamW(model.parameters(), lr=config.optimizer.learning_rate, weight_decay=config.optimizer.weight_decay)
        case OptimizerType.MUON:
            return pt_optim.Muon(model.parameters(), lr=config.optimizer.learning_rate, weight_decay=config.optimizer.weight_decay)
        case OptimizerType.ADAM_MINI:
            return pt_optim.AdamMini(model.parameters(), lr=config.optimizer.learning_rate, weight_decay=config.optimizer.weight_decay)
        case _:
            raise ValueError(f"Invalid optimizer type: {config.optimizer.type}")


def initialize_scheduler(config: Config, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler | None:
    match config.scheduler.type:
        case SchedulerType.NONE:
            return None
        case SchedulerType.COS_ANNEALING:
            raise NotImplementedError("Cosine annealing scheduler not implemented")
        case _:
            raise ValueError(f"Invalid scheduler type: {config.scheduler.type}")
