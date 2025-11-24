import argparse
from collections import OrderedDict
import torch
from dataclasses import dataclass
from pathlib import Path


@dataclass
class InferenceRunResult:
    """Results from a single inference run."""

    s2t_test_loss: float
    s2s_test_loss: float
    latency: float  # in seconds
    model_path: Path
    config_path: Path
    molecule_type: str
    inference_type: str


@dataclass
class MultiInferenceResults:
    """Results from multiple inference runs with statistics."""

    run_results: list[InferenceRunResult]

    @property
    def s2t_mean(self) -> float:
        return sum(result.s2t_test_loss for result in self.run_results) / len(self.run_results)

    @property
    def s2t_std(self) -> float:
        return torch.std(torch.tensor([result.s2t_test_loss for result in self.run_results])).item()

    @property
    def s2s_mean(self) -> float:
        return sum(result.s2s_test_loss for result in self.run_results) / len(self.run_results)

    @property
    def s2s_std(self) -> float:
        return torch.std(torch.tensor([result.s2s_test_loss for result in self.run_results])).item()

    @property
    def latency_mean(self) -> float:
        return sum(result.latency for result in self.run_results) / len(self.run_results)

    @property
    def latency_std(self) -> float:
        return torch.std(torch.tensor([result.latency for result in self.run_results])).item()


def parse_inference_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pretrained ATOM model")

    # Support for multiple model/config pairs
    parser.add_argument(
        "--runs",
        nargs="+",
        help="List of model,config pairs separated by commas. Example: model1.pth,config1.toml model2.pth,config2.toml",
    )

    # Legacy support for single model/config pair
    parser.add_argument(
        "--model",
        type=str,
        help="Path to a pretrained model (for single run)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a config.toml file (for single run)",
    )

    return parser.parse_args()


def parse_model_config_pairs(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Parse command line arguments into model,config pairs."""
    pairs = []

    if args.runs:
        # Multiple runs format: --runs model1.pth,config1.toml model2.pth,config2.toml
        for run_spec in args.runs:
            try:
                model_path, config_path = run_spec.split(",")
                pairs.append((model_path.strip(), config_path.strip()))
            except ValueError:
                raise ValueError(f"Invalid format for run specification: {run_spec}. Expected format: model.pth,config.toml")
    elif args.model and args.config:
        # Single run format: --model model.pth --config config.toml
        pairs.append((args.model, args.config))
    else:
        raise ValueError("Must provide either --runs for multiple runs or both --model and --config for single run")

    return pairs


def clean_state_dict_prefixes(
    state_dict: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """
    Remove the '_orig_mod.' prefix from the state_dict keys that is added by torch.compile.
    """
    new_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_key = k[len("_orig_mod.") :]
        else:
            new_key = k
        new_state_dict[new_key] = v

    return new_state_dict
