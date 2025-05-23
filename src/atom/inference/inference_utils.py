import argparse
from collections import OrderedDict
import torch


def parse_inference_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pretrained ATOM model")
    _ = parser.add_argument(
        "--model",
        type=str,
        help="Path to a pretrained model",
    )
    _ = parser.add_argument(
        "--config",
        type=str,
        help="Path to a config.toml file",
    )
    return parser.parse_args()


def clean_state_dict_prefixes(state_dict: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
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
