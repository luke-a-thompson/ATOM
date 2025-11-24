from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import torch
from usyd_colors import get_palette
from collections import OrderedDict
from matplotlib.ticker import MultipleLocator

from atom.inference.inference_utils import clean_state_dict_prefixes
from atom.training import (
    Config,
    create_dataloaders_multitask,
    create_dataloaders_single,
    eval_epoch,
    initialize_model,
)


def _parse_numeric_list(arg: str | list[int] | None) -> list[int]:
    """Parse a CLI list like "[4, 8, 12]" or space-separated values into a list of ints."""
    if arg is None:
        return []
    if isinstance(arg, list):
        return [int(x) for x in arg]
    s: str = arg.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
    else:
        parts = [p.strip() for p in s.split() if p.strip()]
    return [int(p) for p in parts]


def _resolve_config_path(config_arg: str) -> Path:
    """Allow passing a TOML file or a directory containing exactly one TOML file."""
    path = Path(config_arg)
    if path.is_file():
        return path
    tomls: list[Path] = sorted(path.glob("*.toml"))
    if len(tomls) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .toml in {path}, found {len(tomls)}"
        )
    return tomls[0]


def _resolve_model_paths(model_arg: str) -> list[Path]:
    """Return all model checkpoints from a file or recursively within a directory.

    Preference: if any files matching '*best*' exist, only use those. Otherwise use all .pt/.pth.
    """
    path = Path(model_arg)
    if path.is_file():
        return [path]
    # Search recursively
    best: list[Path] = sorted(
        list(path.rglob("*best*.pt")) + list(path.rglob("*best*.pth"))
    )
    if best:
        return best
    all_ckpts: list[Path] = sorted(list(path.rglob("*.pt")) + list(path.rglob("*.pth")))
    return all_ckpts


def _select_color_from_model_name(model_path: Path) -> str:
    """Select a USYD palette color based on model name. ATOM -> ochre (yellow)."""
    grey, red, blue, yellow, white = get_palette("primary").hex_colors()
    candidates = [p.lower() for p in model_path.parts] + [model_path.stem.lower()]
    joined = "/".join(candidates)
    # Swapped per request: ATOM -> red, EGNO -> ochre (yellow)
    if "atom" in joined:
        return red
    if "egno" in joined:
        return yellow
    if "egnn" in joined:
        return blue
    return grey


def _select_label_from_model_path(model_path: Path) -> str:
    """Infer a concise legend label from the model path (e.g., ATOM, EGNO, EGNN)."""
    candidates = [p.lower() for p in model_path.parts] + [model_path.stem.lower()]
    joined = "/".join(candidates)
    if "atom" in joined:
        return "ATOM"
    if "egno" in joined:
        return "EGNO"
    if "egnn" in joined:
        return "EGNN"
    # Fallback: parent directory name if meaningful, else stem
    parent = model_path.parent.name
    if parent and parent.lower() not in {"checkpoints", "models", "runs"}:
        return parent
    return model_path.stem


def _label_and_color_for_series(
    config_path: Path, model_anchor: Path
) -> tuple[str, str]:
    """Derive legend label and color for a series given its config and model path.

    If the config's benchmark_name == "NoPE", return ("ATOM NoPE", blue).
    Otherwise, infer from the model path using USYD palette mappings.
    """
    grey, red, blue, yellow, white = get_palette("primary").hex_colors()
    # Defaults from path
    default_label = _select_label_from_model_path(model_anchor)
    default_color = _select_color_from_model_name(model_anchor)

    try:
        cfg = Config.from_toml(config_path)
        bench_name = getattr(
            getattr(cfg, "benchmark", object()), "benchmark_name", None
        )
        if isinstance(bench_name, str) and bench_name.strip().lower() == "nope":
            return "ATOM NoPE", blue
    except Exception:
        # Fallback to defaults if any issue loading config
        pass

    return default_label, default_color


def _run_single_eval(config: Config, model_path: Path) -> tuple[float, float]:
    """Load a model checkpoint for the provided Config and evaluate on the test loader.

    Returns (s2t_test_loss, s2s_test_loss).
    """
    # Create test loader (single or multitask)
    if config.dataloader.multitask:
        test_loader = create_dataloaders_multitask(config)[2]
    else:
        test_loader = create_dataloaders_single(config)[2]

    # Initialize model and load weights
    model = initialize_model(config).to(config.training.device)
    raw_sd = cast(
        dict[str, torch.Tensor], torch.load(str(model_path), weights_only=True)
    )
    state_dict = clean_state_dict_prefixes(OrderedDict(raw_sd))
    _ = model.load_state_dict(state_dict, strict=False)
    _ = model.eval()

    s2t_loss, s2s_loss = eval_epoch(config, model, test_loader)
    return float(s2t_loss), float(s2s_loss)


def run_p_invariance(
    p_values: list[int],
    config_paths: list[str] | str,
    model_paths: list[str] | str,
    save_dir: str | None = None,
) -> dict[str, list[tuple[int, float, float]]]:
    """Evaluate MSE vs P (num_timesteps) for one or many (config, model) pairs and plot.

    Returns a dict mapping label -> list of (p, mean_s2t_mse, std_s2t_mse).
    """
    cfg_args: list[str] = (
        [config_paths] if isinstance(config_paths, str) else list(config_paths)
    )
    mdl_args: list[str] = (
        [model_paths] if isinstance(model_paths, str) else list(model_paths)
    )
    if len(cfg_args) != len(mdl_args):
        raise ValueError("--config and --model must have the same number of arguments")

    series_results: dict[str, list[tuple[int, float, float]]] = {}

    plt.figure(figsize=(6, 4))
    for cfg_arg, mdl_arg in zip(cfg_args, mdl_args):
        cfg_path = _resolve_config_path(cfg_arg)
        ckpt_paths: list[Path] = _resolve_model_paths(mdl_arg)
        if not ckpt_paths:
            raise FileNotFoundError(f"No model checkpoints found in {mdl_arg}")

        # Use config and model directory to derive label and color
        label_anchor = Path(mdl_arg)
        label, color = _label_and_color_for_series(cfg_path, label_anchor)

        results: list[tuple[int, float, float]] = []
        for p in p_values:
            s2t_vals: list[float] = []
            for ckpt in ckpt_paths:
                config = Config.from_toml(cfg_path)
                config.dataloader.num_timesteps = int(p)
                s2t_mse, _ = _run_single_eval(config, ckpt)
                s2t_vals.append(s2t_mse)
            mean = float(sum(s2t_vals) / len(s2t_vals))
            var = float(sum((v - mean) ** 2 for v in s2t_vals) / len(s2t_vals))
            std = var**0.5
            results.append((p, mean, std))

        p_sorted = sorted(results, key=lambda x: x[0])
        xs = [v for (v, _, _) in p_sorted]
        means = [m for (_, m, _) in p_sorted]
        stds = [s for (_, _, s) in p_sorted]
        means = [m * 100.0 for m in means]
        stds = [s * 100.0 for s in stds]

        plt.plot(xs, means, "-o", color=color, linewidth=2, markersize=6, label=label)
        upper = [m + 2.0 * s for m, s in zip(means, stds)]
        lower = [m - 2.0 * s for m, s in zip(means, stds)]
        plt.fill_between(xs, lower, upper, color=color, alpha=0.2)

        series_results[label] = results

    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(4))
    plt.xlabel("P")
    plt.ylabel(r"S2T MSE")
    plt.legend(loc="best")
    plt.tight_layout()

    out_dir = (
        Path(save_dir)
        if save_dir is not None
        else Path("Z_paper_content/invariance_results")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save naming depends on number of series
    out_name = (
        "p_invariance_multi.pdf"
        if len(series_results) > 1
        else f"p_invariance_{next(iter(series_results)).replace(' ', '_')}.pdf"
    )
    out_path = out_dir / out_name
    plt.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
    return series_results


def run_t_invariance(
    t_values: list[int],
    config_paths: list[str] | str,
    model_paths: list[str] | str,
    save_dir: str | None = None,
) -> dict[str, list[tuple[int, float, float]]]:
    """Evaluate MSE vs Δt (delta_T) for one or many (config, model) pairs and plot.

    Returns a dict mapping label -> list of (t, mean_s2t_mse, std_s2t_mse).
    """
    cfg_args: list[str] = (
        [config_paths] if isinstance(config_paths, str) else list(config_paths)
    )
    mdl_args: list[str] = (
        [model_paths] if isinstance(model_paths, str) else list(model_paths)
    )
    if len(cfg_args) != len(mdl_args):
        raise ValueError("--config and --model must have the same number of arguments")

    series_results: dict[str, list[tuple[int, float, float]]] = {}

    plt.figure(figsize=(6, 4))
    for cfg_arg, mdl_arg in zip(cfg_args, mdl_args):
        cfg_path = _resolve_config_path(cfg_arg)
        ckpt_paths: list[Path] = _resolve_model_paths(mdl_arg)
        if not ckpt_paths:
            raise FileNotFoundError(f"No model checkpoints found in {mdl_arg}")

        label_anchor = Path(mdl_arg)
        label, color = _label_and_color_for_series(cfg_path, label_anchor)

        results: list[tuple[int, float, float]] = []
        for t in t_values:
            s2t_vals: list[float] = []
            for ckpt in ckpt_paths:
                config = Config.from_toml(cfg_path)
                config.dataloader.delta_T = int(t)
                s2t_mse, _ = _run_single_eval(config, ckpt)
                s2t_vals.append(s2t_mse)
            mean = float(sum(s2t_vals) / len(s2t_vals))
            var = float(sum((v - mean) ** 2 for v in s2t_vals) / len(s2t_vals))
            std = var**0.5
            results.append((t, mean, std))

        t_sorted = sorted(results, key=lambda x: x[0])
        xs = [v for (v, _, _) in t_sorted]
        means = [m for (_, m, _) in t_sorted]
        stds = [s for (_, _, s) in t_sorted]
        means = [m * 100.0 for m in means]
        stds = [s * 100.0 for s in stds]

        plt.plot(xs, means, "-o", color=color, linewidth=2, markersize=6, label=label)
        upper = [m + 2.0 * s for m, s in zip(means, stds)]
        lower = [m - 2.0 * s for m, s in zip(means, stds)]
        plt.fill_between(xs, lower, upper, color=color, alpha=0.2)

        series_results[label] = results

    plt.xscale("log")
    plt.xlabel(r"$\Delta t$")
    plt.ylabel(r"S2T MSE")
    plt.legend(loc="best")
    plt.tight_layout()

    out_dir = (
        Path(save_dir)
        if save_dir is not None
        else Path("Z_paper_content/invariance_results")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = (
        "t_invariance_multi.pdf"
        if len(series_results) > 1
        else f"t_invariance_{next(iter(series_results)).replace(' ', '_')}.pdf"
    )
    out_path = out_dir / out_name
    plt.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
    return series_results


def main_p() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate P-invariance (num_timesteps) and plot MSE vs P."
    )
    parser.add_argument(
        "--p",
        dest="p",
        type=str,
        required=True,
        help="List of P values. Accepts '[4,8,12]' or space-separated values.",
    )
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        nargs="+",
        required=True,
        help="One or more paths to config .toml files or directories (same count as --model).",
    )
    parser.add_argument(
        "--model",
        dest="model",
        type=str,
        nargs="+",
        required=True,
        help="One or more paths to model checkpoints or directories (same count as --config).",
    )
    parser.add_argument(
        "--save-dir", dest="save_dir", type=str, required=False, default=None
    )
    args = parser.parse_args()

    p_values = _parse_numeric_list(args.p)
    if not p_values:
        raise ValueError("No P values provided.")

    _ = run_p_invariance(
        p_values=p_values,
        config_paths=args.config,
        model_paths=args.model,
        save_dir=args.save_dir,
    )


def main_t() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate T-invariance (delta_T) and plot MSE vs Δt."
    )
    parser.add_argument(
        "--t",
        dest="t",
        type=str,
        required=True,
        help="List of Δt values. Accepts '[1,2,4]' or space-separated values.",
    )
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        nargs="+",
        required=True,
        help="One or more paths to config .toml files or directories (same count as --model).",
    )
    parser.add_argument(
        "--model",
        dest="model",
        type=str,
        nargs="+",
        required=True,
        help="One or more paths to model checkpoints or directories (same count as --config).",
    )
    parser.add_argument(
        "--save-dir", dest="save_dir", type=str, required=False, default=None
    )
    args = parser.parse_args()

    t_values = _parse_numeric_list(args.t)
    if not t_values:
        raise ValueError("No Δt values provided.")

    _ = run_t_invariance(
        t_values=t_values,
        config_paths=args.config,
        model_paths=args.model,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    # Default to P-invariance CLI if called directly
    main_p()
