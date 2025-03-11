from enum import StrEnum


class DeviceType(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


class OptimizerType(StrEnum):
    SGD = "sgd"
    ADAMW = "adamw"
    MUON = "muon"
    ADAM_MINI = "adam-mini"


class SchedulerType(StrEnum):
    NONE = "none"
    COS_ANNEALING = "cosine_annealing"
