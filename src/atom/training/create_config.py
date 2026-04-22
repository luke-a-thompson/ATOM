import importlib.util
import warnings
import tomllib
from pathlib import Path
from pydantic import BaseModel, model_validator, field_validator, BeforeValidator
import torch
from typing import Annotated
from pydantic import Field, PositiveInt, NonNegativeFloat, NonNegativeInt

from atom.training.config_options import (
    FFNActivation,
    AttentionType,
    LiftingType,
    Datasets,
    MD17MoleculeType,
    RMD17MoleculeType,
    MD22MoleculeType,
    TG80MoleculeType,
    ModelType,
    NormType,
    OptimizerType,
    SchedulerType,
    ValueResidualType,
    PositionalEncodingType,
    ProjectionType,
    TimeLagMode,
    OutputMode,
)


def _to_torch_dtype(value: object) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        if value == "float16":
            return torch.float16
        if value == "bfloat16":
            return torch.bfloat16
        raise ValueError(f"Invalid dtype name: {value}")
    raise ValueError(f"Invalid dtype value: {value}")


def _to_torch_device(value: object) -> torch.device:
    if isinstance(value, torch.device):
        return value
    if isinstance(value, (str, int)):
        try:
            return torch.device(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Could not convert {value} to torch.device") from e
    raise ValueError(f"Invalid device value: {value}")


class WandbConfig(BaseModel):
    use_wandb: bool


class BenchmarkConfig(BaseModel):
    benchmark_name: str
    model_type: ModelType
    compile: bool
    compile_trace: bool
    runs: Annotated[PositiveInt, Field(description="Must be greater than 0.")]
    log_weights: bool

    @model_validator(mode="before")
    @classmethod
    def validate_benchmark_name(cls, values: dict[str, object]) -> dict[str, object]:
        if values.get("benchmark_name") is None:
            user_input = input("Enter benchmark name (leave blank to use model_type): ")
            if not user_input:
                user_input = str(values.get("model_type"))
            values["benchmark_name"] = user_input
        return values

    @field_validator("compile")
    @classmethod
    def check_compile(cls, value: bool) -> bool:
        if value and torch.cuda.get_device_capability() < (7, 0):
            raise ValueError("CUDA 7.0 or higher is required to compile the model. We recommend CUDA 11.0 or higher.")
        return value

    @field_validator("log_weights")
    @classmethod
    def check_log_weights(cls, value: bool) -> bool:
        if value and importlib.util.find_spec("matplotlib") is None:
            raise ValueError("If 'log_weights' is True, matplotlib must be installed.")
        return value


class DataloaderConfig(BaseModel):
    multitask: bool
    dataset: Datasets
    # Single-task dataloader parameters
    molecule_type: MD17MoleculeType | RMD17MoleculeType | TG80MoleculeType | MD22MoleculeType | None = None

    # Multitask dataloader parameters
    train_molecules: list[MD17MoleculeType | RMD17MoleculeType | TG80MoleculeType | MD22MoleculeType] | None = None
    validation_molecules: list[MD17MoleculeType | RMD17MoleculeType | TG80MoleculeType | MD22MoleculeType] | None = None
    test_molecules: list[MD17MoleculeType | RMD17MoleculeType | TG80MoleculeType | MD22MoleculeType] | None = None

    num_timesteps: Annotated[PositiveInt, Field(description="Must be greater than 0.")]
    # Accept either a fixed integer lag or a (min, max) tuple for stochastic sampling
    delta_T: int | tuple[int, int]
    explicit_hydrogen: bool
    explicit_hydrogen_gradients: bool
    radius_graph_threshold: Annotated[NonNegativeFloat, Field(description="Must be greater than or equal to 0.0.")]
    rrwp_length: Annotated[NonNegativeInt, Field(description="Must be greater than or equal to 0.")]
    time_lag_mode: TimeLagMode
    normalize_z: bool
    persistent_workers: bool
    num_workers: Annotated[NonNegativeInt, Field(description="Must be greater than or equal to 0.")]
    pin_memory: bool
    prefetch_factor: Annotated[PositiveInt, Field(description="Must be greater than 0.")]
    force_regenerate: bool

    @field_validator("delta_T", mode="before")
    @classmethod
    def coerce_delta_T(cls, value: object) -> int | tuple[int, int]:
        # Allow TOML arrays to become tuples
        if isinstance(value, list):
            if len(value) != 2:
                raise ValueError("If 'delta_T' is a list, it must have exactly two elements.")
            try:
                return (int(value[0]), int(value[1]))
            except Exception as e:
                raise ValueError("Could not coerce 'delta_T' list elements to integers.") from e
        if isinstance(value, int):
            return value
        if isinstance(value, tuple):
            try:
                return (int(value[0]), int(value[1]))
            except Exception as e:
                raise ValueError("Could not coerce 'delta_T' tuple elements to integers.") from e
        raise ValueError("'delta_T' must be an int or a tuple/list of two ints.")

    @model_validator(mode="after")
    def validate_consistency(self) -> "DataloaderConfig":
        # explicit hydrogen gradients implies explicit hydrogen
        if self.explicit_hydrogen_gradients and not self.explicit_hydrogen:
            raise ValueError(
                "If 'explicit_hydrogen_gradients' is True, 'explicit_hydrogen' must also be True. You cannot calculate the gradients for hydrogen atoms without them being present in the graph."
            )

        # multitask presence checks
        if self.multitask:
            if not self.train_molecules or not self.validation_molecules or not self.test_molecules:
                raise ValueError("If 'multitask' is True, 'train_molecules', 'validation_molecules', and 'test_molecules' must be specified.")

            # overlap checks
            train_set = set(self.train_molecules)
            val_set = set(self.validation_molecules)
            test_set = set(self.test_molecules)
            if train_set.intersection(val_set):
                warnings.warn(f"Train and validation molecule sets overlap: {', '.join(str(mol) for mol in train_set.intersection(val_set))}")
            if train_set.intersection(test_set):
                warnings.warn(f"Train and test molecule sets overlap: {', '.join(str(mol) for mol in train_set.intersection(test_set))}.")
            if val_set.intersection(test_set):
                warnings.warn(f"Validation and test molecule sets overlap: {', '.join(str(mol) for mol in val_set.intersection(test_set))}")

        # dataset-specific enum type enforcement
        match self.dataset:
            case Datasets.md17:
                enum_type = MD17MoleculeType
            case Datasets.rmd17:
                enum_type = RMD17MoleculeType
            case Datasets.tg80:
                enum_type = TG80MoleculeType
            case Datasets.md22:
                enum_type = MD22MoleculeType
            case _:
                raise ValueError(f"Invalid dataset: {self.dataset}")

        if self.dataset in [Datasets.md17, Datasets.rmd17, Datasets.tg80, Datasets.md22]:
            if self.multitask:
                if self.train_molecules:
                    self.train_molecules = [enum_type(mol) for mol in self.train_molecules]
                if self.validation_molecules:
                    self.validation_molecules = [enum_type(mol) for mol in self.validation_molecules]
                if self.test_molecules:
                    self.test_molecules = [enum_type(mol) for mol in self.test_molecules]
            else:
                if self.molecule_type:
                    self.molecule_type = enum_type(self.molecule_type)

        # Validate delta_T semantics
        if isinstance(self.delta_T, tuple):
            if len(self.delta_T) != 2:
                raise ValueError("'delta_T' tuple must have exactly two elements.")
            dt_min, dt_max = int(self.delta_T[0]), int(self.delta_T[1])
            if dt_min <= 0 or dt_max <= 0:
                raise ValueError("Both elements of 'delta_T' must be positive integers.")
            if dt_min > dt_max:
                raise ValueError("'delta_T' lower bound must be <= upper bound.")
        else:
            if int(self.delta_T) <= 0:
                raise ValueError("'delta_T' must be a positive integer.")

        return self


class TrainingConfig(BaseModel):
    device: Annotated[torch.device, BeforeValidator(_to_torch_device)]
    use_amp: bool
    amp_dtype: Annotated[torch.dtype, BeforeValidator(_to_torch_dtype)]
    seed: int
    batch_size: Annotated[PositiveInt, Field(description="Must be greater than 0.")]
    epochs: Annotated[PositiveInt, Field(description="Must be greater than 0.")]
    max_grad_norm: Annotated[NonNegativeFloat, Field(description="Must be greater than or equal to 0.0.")]
    label_noise_std: Annotated[NonNegativeFloat, Field(description="Label noise standard deviation must be greater than or equal to 0.0.")]

    class Config:
        arbitrary_types_allowed: bool = True

    @model_validator(mode="after")
    def validate_amp_dtype(self) -> "TrainingConfig":
        if self.use_amp and self.amp_dtype not in [torch.float16, torch.bfloat16]:
            raise ValueError("'amp_dtype' must be 'float16' or 'bfloat16' if 'use_amp' is True.")
        return self


class OptimizerConfig(BaseModel):
    type: OptimizerType
    learning_rate: Annotated[NonNegativeFloat, Field(description="Must be greater than or equal to 0.0.")]
    weight_decay: Annotated[NonNegativeFloat, Field(description="Must be greater than or equal to 0.0.")]
    adam_betas: tuple[float, float]
    adam_eps: Annotated[NonNegativeFloat, Field(description="Must be greater than or equal to 0.0.")]


class SchedulerConfig(BaseModel):
    type: SchedulerType


class ATOMConfig(BaseModel):
    # Architecture parameters
    num_layers: Annotated[PositiveInt, Field(description="Must be greater than 0.")]
    num_heads: Annotated[PositiveInt, Field(description="Must be greater than 0.")]
    lifting_dim: Annotated[int, Field(strict=True, ge=2, multiple_of=2, description="Must be even and greater than 2.")]
    # Output parameters
    output_heads: Annotated[PositiveInt, Field(description="Must be greater than 0.")]
    delta_update: bool
    # Attention parameters
    heterogenous_attention_type: AttentionType
    positional_encoding: PositionalEncodingType
    rope_base: Annotated[NonNegativeFloat, Field(description="Must be greater than or equal to 0.0.")]
    rope_tau: Annotated[NonNegativeFloat, Field(description="Must be greater than or equal to 0.0.")]
    # Removed: learnable_attention_denom
    # Feature parameters
    lifting_type: LiftingType
    projection_type: ProjectionType
    # Layer parameters
    norm: NormType
    activation: FFNActivation
    value_residual_type: ValueResidualType
    output_mode: OutputMode = OutputMode.POS_ONLY

    @model_validator(mode="after")
    def validate_lifting_dim(self) -> "ATOMConfig":
        # Ensure lifting_dim divisible by num_heads and that d_head is even
        if self.lifting_dim % self.num_heads != 0:
            raise ValueError("'lifting_dim' must be divisible by 'num_heads'.")
        if (self.lifting_dim // self.num_heads) % 2 != 0:
            raise ValueError("'lifting_dim' / 'num_heads' (d_head) must be even.")
        return self

    @model_validator(mode="after")
    def validate_lifting_type_and_projection_type(self) -> "ATOMConfig":
        if self.lifting_type == LiftingType.CANONICALIZATION and self.projection_type != ProjectionType.DECANONICALIZATION:
            raise ValueError("If 'lifting_type' is 'canonicalization', 'projection_type' must be 'decanonicalization'.")
        elif self.lifting_type == LiftingType.NON_EQUIVARIANT and self.projection_type == ProjectionType.DECANONICALIZATION:
            raise ValueError("If 'lifting_type' is 'none', 'projection_type' must be 'equivariant'.")
        return self


class EGNOConfig(BaseModel):
    num_layers: Annotated[PositiveInt, Field(description="Must be greater than 0.")]
    lifting_dim: Annotated[int, Field(strict=True, ge=2, multiple_of=2, description="Must be even and greater than 2.")]
    activation: FFNActivation
    normalise_scalars: bool
    use_time_conv: bool
    num_fourier_modes: Annotated[PositiveInt, Field(description="Must be greater than 0.")]
    time_embed_dim: int


class EGNNConfig(BaseModel):
    num_layers: Annotated[PositiveInt, Field(description="Must be greater than 0.")]
    lifting_dim: Annotated[int, Field(strict=True, ge=2, multiple_of=2, description="Must be even and greater than 2.")]
    activation: FFNActivation
    time_embed_dim: int


class Config(BaseModel):
    wandb: WandbConfig
    benchmark: BenchmarkConfig
    dataloader: DataloaderConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    atom_config: ATOMConfig
    egno_config: EGNOConfig
    egnn_config: EGNNConfig | None = None

    @model_validator(mode="after")
    def validate_output_heads(self) -> "Config":
        if self.benchmark.model_type == ModelType.ATOM and self.atom_config.output_heads > 1 and self.dataloader.multitask is False:
            print("Are you sure you want to use multiple output heads for a single-task model? This is unusual, but maybe you're onto something.")
        return self

    @classmethod
    def from_toml(cls, path: Path) -> "Config":
        """
        Load configuration from a TOML file.

        Args:
            path: Path to the TOML file

        Returns:
            Config: Validated configuration object
        """
        try:
            with open(path, "rb") as f:
                config_dict = tomllib.load(f)
        except IsADirectoryError:
            raise ValueError(f"Path '{path}' is a directory, not a file. Use --config <path> if you want to run multiple configurations.")

        return cls.model_validate(config_dict)
