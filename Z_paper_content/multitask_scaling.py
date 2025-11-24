import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Z_paper_content.figures import red, blue, set_matplotlib_style


def _discover_results_files(benchmark_dir: Path) -> list[Path]:
    """Return all results.json files under a benchmark directory."""
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark directory does not exist: {benchmark_dir}")

    results_files: list[Path] = sorted(benchmark_dir.glob("**/results.json"))
    if not results_files:
        raise FileNotFoundError(f"No results.json files found under {benchmark_dir}")
    return results_files


def _extract_benchmark_name(results_path: Path, data: dict) -> str:
    """Extract benchmark_name from a results.json payload, with a safe fallback."""
    config_obj = data.get("config", {})
    benchmark_cfg: dict
    if isinstance(config_obj, dict):
        benchmark_cfg = (
            config_obj.get("benchmark", {})
            if isinstance(config_obj.get("benchmark", {}), dict)
            else {}
        )
    else:
        benchmark_cfg = {}

    benchmark_name_obj = benchmark_cfg.get("benchmark_name")
    if isinstance(benchmark_name_obj, str) and benchmark_name_obj.strip():
        return benchmark_name_obj.strip()

    # Fallback: use parent directory name
    return results_path.parent.name


def _to_float(value: object) -> float | None:
    """Best-effort conversion of a JSON value into float, else None."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _canonical_label_from_benchmark_name(benchmark_name: str) -> str:
    """Produce a short label for plotting from a benchmark_name string.

    For names that contain a 'fold' pattern (e.g. 'atom_multitask_muon_fold123'),
    the part starting at 'fold' is returned. Otherwise, the full benchmark_name
    is used unchanged.
    """
    lower = benchmark_name.lower()
    fold_idx = lower.find("fold")
    if fold_idx != -1:
        return benchmark_name[fold_idx:]
    return benchmark_name


def _load_scaling_metrics(results_files: list[Path]) -> dict[str, dict[str, float]]:
    """Load S2S and S2T mean/std metrics keyed by benchmark_name."""
    metrics: dict[str, dict[str, float]] = {}

    for path in results_files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        benchmark_name: str = _extract_benchmark_name(path, data)

        s2s_mean_raw: object = data.get("s2s_test_loss_mean")
        s2t_mean_raw: object = data.get("s2t_test_loss_mean")
        s2s_std_raw: object = data.get("s2s_test_loss_std")
        s2t_std_raw: object = data.get("s2t_test_loss_std")

        s2s_mean = _to_float(s2s_mean_raw)
        s2t_mean = _to_float(s2t_mean_raw)
        s2s_std = _to_float(s2s_std_raw)
        s2t_std = _to_float(s2t_std_raw)

        # Require valid means; if missing or unparseable, skip this file.
        if s2s_mean is None or s2t_mean is None:
            print(
                f"[WARN] Skipping {path} because s2s/s2t mean values are missing or invalid."
            )
            continue

        # Std values may legitimately be missing/None (e.g., single run); treat as 0.
        if s2s_std is None:
            s2s_std = 0.0
        if s2t_std is None:
            s2t_std = 0.0

        # If the same benchmark_name appears multiple times, keep the last one encountered.
        metrics[benchmark_name] = {
            "s2s_mean": s2s_mean,
            "s2s_std": s2s_std,
            "s2t_mean": s2t_mean,
            "s2t_std": s2t_std,
        }

    if not metrics:
        raise ValueError("No metrics could be loaded from the provided results files.")

    return metrics


def _sorted_benchmark_names(metrics: dict[str, dict[str, float]]) -> list[str]:
    """Sort benchmark names in a stable, slightly smarter way.

    If all labels look like they contain a 'fold' segment with trailing digits,
    sort by that numeric value; otherwise sort lexicographically.
    """
    names = list(metrics.keys())

    def _fold_key(name: str) -> int | None:
        lower = name.lower()
        idx = lower.find("fold")
        if idx == -1:
            return None
        suffix = "".join(ch for ch in name[idx + 4 :] if ch.isdigit())
        if not suffix:
            return None
        try:
            return int(suffix)
        except ValueError:
            return None

    fold_keys: list[int | None] = [_fold_key(n) for n in names]
    if all(k is not None for k in fold_keys):
        pairs = sorted(zip(names, fold_keys), key=lambda nk: nk[1])  # type: ignore[arg-type]
        return [n for n, _ in pairs]

    return sorted(names)


def plot_multitask_scaling(
    benchmark_dir: Path,
    save_path: Path | None = None,
) -> None:
    """Plot S2S and S2T test losses across benchmarks in a scaling-law directory.

    The directory is expected to contain subdirectories with a layout like:

        tg80_multitask_atom_scaling_law_20-Nov-2025_16-32-52/
            atom_multitask_muon_fold1_.../
                results.json
            atom_multitask_muon_fold12_.../
                results.json
            ...

    Each results.json is assumed to be produced by atom.training.benchmark.MultiRunResults.
    """
    results_files = _discover_results_files(benchmark_dir)
    metrics = _load_scaling_metrics(results_files)

    benchmark_names: list[str] = _sorted_benchmark_names(metrics)
    labels: list[str] = [
        _canonical_label_from_benchmark_name(name) for name in benchmark_names
    ]

    s2s_means: list[float] = [
        metrics[name]["s2s_mean"] * 100.0 for name in benchmark_names
    ]
    s2s_stds: list[float] = [
        metrics[name]["s2s_std"] * 100.0 for name in benchmark_names
    ]
    s2t_means: list[float] = [
        metrics[name]["s2t_mean"] * 100.0 for name in benchmark_names
    ]
    s2t_stds: list[float] = [
        metrics[name]["s2t_std"] * 100.0 for name in benchmark_names
    ]

    x = np.arange(len(benchmark_names), dtype=float)
    width = 0.6

    # Determine output paths for S2S and S2T plots
    if save_path is None:
        out_dir = Path("Z_paper_content") / "multitask_scaling"
        out_dir.mkdir(parents=True, exist_ok=True)
        s2s_path = out_dir / "multitask_scaling_s2s.pdf"
        s2t_path = out_dir / "multitask_scaling_s2t.pdf"
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        stem = save_path.stem
        suffix = save_path.suffix or ".pdf"
        s2s_path = save_path.with_name(f"{stem}_s2s{suffix}")
        s2t_path = save_path.with_name(f"{stem}_s2t{suffix}")

    # Plot S2S only
    fig_s2s, ax_s2s = plt.subplots(figsize=(6, 4))
    ax_s2s.bar(
        x,
        s2s_means,
        width,
        yerr=[2.0 * std for std in s2s_stds],
        label="S2S",
        color=red,
        alpha=0.8,
        capsize=4,
    )
    ax_s2s.set_xticks(x)
    ax_s2s.set_xticklabels(labels, rotation=45, ha="right")
    ax_s2s.set_ylabel("S2S Test MSE ($\\times 10^{-2}$)")
    ax_s2s.set_xlabel("Benchmark (from benchmark_name)")
    ax_s2s.set_ylim(bottom=10.0)
    fig_s2s.tight_layout()
    fig_s2s.savefig(s2s_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Saved S2S multitask scaling figure to {s2s_path}")

    # Plot S2T only
    fig_s2t, ax_s2t = plt.subplots(figsize=(6, 4))
    ax_s2t.bar(
        x,
        s2t_means,
        width,
        yerr=[2.0 * std for std in s2t_stds],
        label="S2T",
        color=blue,
        alpha=0.8,
        capsize=4,
    )
    ax_s2t.set_xticks(x)
    ax_s2t.set_xticklabels(labels, rotation=45, ha="right")
    ax_s2t.set_ylabel("S2T Test MSE ($\\times 10^{-2}$)")
    ax_s2t.set_xlabel("Benchmark (from benchmark_name)")
    ax_s2t.set_ylim(bottom=10.0)
    fig_s2t.tight_layout()
    fig_s2t.savefig(s2t_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Saved S2T multitask scaling figure to {s2t_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot S2S and S2T test losses from a multitask scaling-law benchmark directory.\n\n"
            "Example:\n"
            "  uv run python -m Z_paper_content.multitask_scaling \\\n"
            "    --benchmark-dir benchmark_runs/tg80_multitask_atom_scaling_law_20-Nov-2025_16-32-52\n"
        ),
    )

    parser.add_argument(
        "--benchmark-dir",
        type=str,
        required=True,
        help=(
            "Directory containing multitask scaling-law runs, e.g. "
            "benchmark_runs/tg80_multitask_atom_scaling_law_20-Nov-2025_16-32-52"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional path to save the PDF; defaults to "
            "Z_paper_content/multitask_scaling/multitask_scaling.pdf"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    set_matplotlib_style(font_size=18)

    benchmark_dir = Path(args.benchmark_dir)
    save_path: Path | None = Path(args.output) if args.output is not None else None

    plot_multitask_scaling(benchmark_dir=benchmark_dir, save_path=save_path)


if __name__ == "__main__":
    main()
