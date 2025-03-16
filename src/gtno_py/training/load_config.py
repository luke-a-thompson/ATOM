from pydantic import BaseModel, model_validator
from gtno_py.training.config_options import (
    OptimizerType,
    SchedulerType,
    DeviceType,
    NormType,
    GraphHeterogenousAttentionType,
    ValueResidualType,
    ModelType,
    FFNActivation,
    MD17MoleculeType,
    RMD17MoleculeType,
    MD17Version,
)
import tomllib
import importlib.util


class WandbConfig(BaseModel):
    use_wandb: bool


class BenchmarkConfig(BaseModel):
    compile: bool
    runs: int
    multitask: bool
    md17_version: MD17Version
    molecule_type: MD17MoleculeType | RMD17MoleculeType | list[MD17MoleculeType | RMD17MoleculeType]
    max_nodes: int
    delta_T: int
    log_weights: bool

    @model_validator(mode="after")
    def validate_multitask(self) -> "BenchmarkConfig":
        multitask = self.multitask
        molecule_type = self.molecule_type

        if multitask and not isinstance(molecule_type, list):
            raise ValueError("If 'multitask' is True, 'molecule_type' must be a list of molecules to train on.")

        return self

    @model_validator(mode="after")
    def validate_md17_version(self) -> "BenchmarkConfig":
        md17_version = self.md17_version
        molecule_type = self.molecule_type

        if md17_version == MD17Version.md17:
            if isinstance(molecule_type, list):
                for mol in molecule_type:
                    if not isinstance(mol, MD17MoleculeType):
                        raise ValueError(f"When using MD17 version, all molecules must be MD17MoleculeType, got {mol}")
            elif not isinstance(molecule_type, MD17MoleculeType):
                raise ValueError(f"When using MD17 version, molecule_type must be MD17MoleculeType, got {molecule_type}")
        elif md17_version == MD17Version.rmd17:
            if isinstance(molecule_type, list):
                for mol in molecule_type:
                    if not isinstance(mol, RMD17MoleculeType):
                        raise ValueError(f"When using RMD17 version, all molecules must be RMD17MoleculeType, got {mol}")
            elif not isinstance(molecule_type, RMD17MoleculeType):
                raise ValueError(f"When using RMD17 version, molecule_type must be RMD17MoleculeType, got {molecule_type}")

        return self

    @model_validator(mode="after")
    def validate_log_weights(self) -> "BenchmarkConfig":
        if self.log_weights:
            if importlib.util.find_spec("matplotlib") is None:
                raise ValueError("If 'log_weights' is True, matplotlib must be installed.")
        return self


class DataloaderConfig(BaseModel):
    explicit_hydrogen: bool
    explicit_hydrogen_gradients: bool
    rrwp_length: int
    persistent_workers: bool
    num_workers: int
    pin_memory: bool
    force_regenerate: bool

    @model_validator(mode="after")
    def validate_explicit_hydrogen_gradients(self) -> "DataloaderConfig":
        if self.explicit_hydrogen_gradients and not self.explicit_hydrogen:
            raise ValueError(
                "If 'explicit_hydrogen_gradients' is True, 'explicit_hydrogen' must also be True. You cannot calculate the gradients for hydrogen atoms without them being present in the graph."
            )
        return self


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
    # Architecture parameters
    num_layers: int
    num_heads: int
    lifting_dim: int
    num_timesteps: int
    # Attention parameters
    heterogenous_attention_type: GraphHeterogenousAttentionType
    use_rope: bool
    learnable_attention_denom: bool
    # Feature parameters
    use_spherical_harmonics: bool
    use_equivariant_lifting: bool
    # Layer parameters
    norm: NormType
    activation: FFNActivation
    value_residual_type: ValueResidualType


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
