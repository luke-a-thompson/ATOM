from pydantic import BaseModel
from gtno_py.dataloaders.egno_dataloder import MD17MoleculeType, RMD17MoleculeType, MD17Version
from gtno_py.training.model_options import OptimizerType, SchedulerType, DeviceType
from gtno_py.gtno.gtno_model import NormType, GraphHeterogenousAttentionType, ValueResidualType, ModelType
from gtno_py.gtno.activations import FFNActivation
import tomllib



class WandbConfig(BaseModel):
    use_wandb: bool


class BenchmarkConfig(BaseModel):
    runs: int
    compile: bool
    molecule_type: MD17MoleculeType | RMD17MoleculeType | list[MD17MoleculeType | RMD17MoleculeType]
    max_nodes: int
    md17_version: MD17Version
    delta_T: int
    log_weights: bool


class DataloaderConfig(BaseModel):
    explicit_hydrogen: bool
    explicit_hydrogen_gradients: bool
    persistent_workers: bool
    num_workers: int
    pin_memory: bool
    force_regenerate: bool


class TrainingConfig(BaseModel):
    device: DeviceType
    seed: int
    batch_size: int
    epochs: int
    max_grad_norm: float
    learnable_noise_std: bool
    brownian_noise_std: float


class OptimizerConfig(BaseModel):
    type: OptimizerType
    learning_rate: float
    weight_decay: float
    adam_betas: tuple[float, float]
    adam_eps: float


class SchedulerConfig(BaseModel):
    type: SchedulerType


class ModelConfig(BaseModel):
    model_type: ModelType
    lifting_dim: int
    num_timesteps: int
    norm: NormType
    activation: FFNActivation
    heterogenous_attention_type: GraphHeterogenousAttentionType
    use_rope: bool
    use_spherical_harmonics: bool
    value_residual_type: ValueResidualType
    use_equivariant_lifting: bool
    num_layers: int
    num_heads: int
    learnable_attention_denom: bool


class Config(BaseModel):
    wandb: WandbConfig
    benchmark: BenchmarkConfig
    dataloader: DataloaderConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    model: ModelConfig

    @classmethod
    def from_toml(cls, path: str) -> "Config":
        """
        Load configuration from a TOML file.

        Args:
            path: Path to the TOML file

        Returns:
            Config: Validated configuration object
        """
        with open(path, "rb") as f:
            config_dict = tomllib.load(f)

        return cls(**config_dict)
