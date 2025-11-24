import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from atom.inference.inference_utils import clean_state_dict_prefixes
from atom.training import (
    Config,
    create_dataloaders_multitask,
    create_dataloaders_single,
    eval_epoch,
    initialize_model,
)


@dataclass
class AtomInferenceResult:
    molecule: str
    wallclock_s_mean: float
    wallclock_s_std: float
    wallclock_s_2sd: float
    wallclock_s_latex: str
    s2t_loss_mean: float
    s2t_loss_std: float
    s2s_loss_mean: float
    s2s_loss_std: float


def _sync_if_cuda(device: str) -> None:
    """Synchronize CUDA device if applicable for reliable timing."""
    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.synchronize()


def discover_atom_runs(atom_run_dir: Path) -> list[tuple[str, Path, Path]]:
    """Discover per-molecule ATOM checkpoints and configs in a benchmark directory.

    This expects a layout like:

        md17_uniform_paper_atom_25-Sep-2025_03-36-08/
            md_aspirin_25-Sep-2025_03-36-08/
                md_aspirin_25-Sep-2025_03-36-08.toml
                results.json
                run_1/best_val_model.pth
                run_2/best_val_model.pth
                run_3/best_val_model.pth
            md_ethanol_25-Sep-2025_03-36-08/
            ...

    or, for MD22-style runs:

        some_md22_benchmark_root/
            atom_md22_dha_20-Sep-2025_22-53-38/
                atom_md22_dha_20-Sep-2025_22-53-38.toml
                results.json
                run_1/best_val_model.pth
                run_2/best_val_model.pth
                run_3/best_val_model.pth
            atom_md22_nhme_20-Sep-2025_22-53-38/
            ...

    For each molecule directory we return the triple:
        (molecule_name, path_to_run_1_best_val_model, path_to_config_toml)
    """
    if not atom_run_dir.exists():
        raise FileNotFoundError(f"ATOM benchmark directory does not exist: {atom_run_dir}")

    runs: list[tuple[str, Path, Path]] = []

    for subdir in sorted(atom_run_dir.iterdir()):
        if not subdir.is_dir():
            continue

        name_lower: str = subdir.name.lower()
        molecule: str | None = None

        # MD17-style: md_<molecule>_*
        if name_lower.startswith("md_"):
            name_parts: list[str] = subdir.name.split("_")
            if len(name_parts) >= 2:
                molecule = name_parts[1].lower()

        # MD22-style: atom_md22_<molecule>_*
        elif name_lower.startswith("atom_md22_"):
            name_parts = subdir.name.split("_")
            if len(name_parts) >= 3:
                molecule = name_parts[2].lower()

        if molecule is None:
            # Skip any directories that do not follow the expected benchmark naming patterns
            continue

        toml_files: list[Path] = list(subdir.glob("*.toml"))
        if not toml_files:
            print(f"[WARN] No .toml config found in {subdir}, skipping.")
            continue
        if len(toml_files) > 1:
            print(f"[WARN] Multiple .toml files found in {subdir}, using the first one.")
        config_path: Path = toml_files[0]

        model_path: Path = subdir / "run_1" / "best_val_model.pth"
        if not model_path.exists():
            print(f"[WARN] Missing best_val_model.pth at {model_path}, skipping.")
            continue

        runs.append((molecule, model_path, config_path))

    if not runs:
        raise FileNotFoundError(
            f"No valid ATOM runs discovered in {atom_run_dir}. "
            "Expected subdirectories like md_aspirin_*/run_1/best_val_model.pth with a .toml config.",
        )

    return runs


def run_single_atom_inference(
    molecule: str,
    model_path: Path,
    config_path: Path,
    num_repeats: int,
) -> AtomInferenceResult:
    """Run timed ATOM inference on a single (model, config) pair."""
    config: Config = Config.from_toml(config_path)

    model_state = torch.load(str(model_path), map_location=config.training.device, weights_only=True)
    model_state_clean = clean_state_dict_prefixes(model_state)

    if config.dataloader.multitask:
        _, _, test_loader = create_dataloaders_multitask(config)
    else:
        _, _, test_loader = create_dataloaders_single(config)

    model = initialize_model(config).to(config.training.device)
    _ = model.load_state_dict(model_state_clean, strict=False)
    _ = model.eval()

    device_str: str = str(config.training.device)

    times: list[float] = []
    s2t_losses: list[float] = []
    s2s_losses: list[float] = []

    for _ in range(num_repeats):
        _sync_if_cuda(device_str)
        start = time.perf_counter()
        with torch.no_grad():
            s2t_loss, s2s_loss = eval_epoch(config, model, test_loader)
        _sync_if_cuda(device_str)
        wallclock: float = time.perf_counter() - start

        times.append(float(wallclock))
        s2t_losses.append(float(s2t_loss))
        s2s_losses.append(float(s2s_loss))

    times_arr: npt.NDArray[np.float64] = np.asarray(times, dtype=np.float64)
    s2t_arr: npt.NDArray[np.float64] = np.asarray(s2t_losses, dtype=np.float64)
    s2s_arr: npt.NDArray[np.float64] = np.asarray(s2s_losses, dtype=np.float64)

    time_mean: float = float(times_arr.mean())
    time_std: float = float(times_arr.std(ddof=1)) if times_arr.size > 1 else 0.0
    time_2sd: float = 2.0 * time_std
    latex_time: str = f"{time_mean:.3f}\\pm\\scriptstyle{{{time_2sd:.3f}}}"

    s2t_mean: float = float(s2t_arr.mean())
    s2t_std: float = float(s2t_arr.std(ddof=1)) if s2t_arr.size > 1 else 0.0
    s2s_mean: float = float(s2s_arr.mean())
    s2s_std: float = float(s2s_arr.std(ddof=1)) if s2s_arr.size > 1 else 0.0

    return AtomInferenceResult(
        molecule=molecule,
        wallclock_s_mean=time_mean,
        wallclock_s_std=time_std,
        wallclock_s_2sd=time_2sd,
        wallclock_s_latex=latex_time,
        s2t_loss_mean=s2t_mean,
        s2t_loss_std=s2t_std,
        s2s_loss_mean=s2s_mean,
        s2s_loss_std=s2s_std,
    )


def run_experiment(
    atom_run_dir: Path,
    num_repeats: int,
    limit_molecules: list[str] | None = None,
) -> dict[str, object]:
    """Run ATOM inference timing across all discovered molecules in a benchmark directory."""
    discovered: list[tuple[str, Path, Path]] = discover_atom_runs(atom_run_dir)

    if limit_molecules is not None:
        selected: set[str] = {m.lower() for m in limit_molecules}
        discovered = [(m, mp, cp) for (m, mp, cp) in discovered if m in selected]

    if not discovered:
        raise FileNotFoundError(
            "No ATOM runs remain after applying molecule filter. Check --molecules or the benchmark directory contents.",
        )

    atom_results: list[AtomInferenceResult] = []

    for molecule, model_path, config_path in discovered:
        print(f"Running ATOM inference timing for {molecule}...")
        result = run_single_atom_inference(
            molecule=molecule,
            model_path=model_path,
            config_path=config_path,
            num_repeats=num_repeats,
        )
        atom_results.append(result)

    stats: dict[str, object] = {
        "atom_run_dir": str(atom_run_dir),
        "num_repeats": int(num_repeats),
        "atom_results": [asdict(r) for r in atom_results],
    }
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark ATOM inference wallclock time on MD17 runs.\n\n"
            "Example:\n"
            "  uv run python -m Z_paper_content.md17_md_vs_atom_timing \\\n"
            "    --atom-run-dir benchmark_runs/md17/md17_uniform_paper_atom_25-Sep-2025_03-36-08 \\\n"
            "    --num-repeats 3"
        ),
    )

    parser.add_argument(
        "--atom-run-dir",
        type=str,
        required=True,
        help=("Directory containing MD17 ATOM benchmark runs, e.g. benchmark_runs/md17/md17_uniform_paper_atom_25-Sep-2025_03-36-08"),
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=3,
        help="Number of repeated eval_epoch calls per molecule for timing statistics (default: 3).",
    )
    parser.add_argument(
        "--molecules",
        type=str,
        nargs="*",
        help="Optional subset of MD17 molecule base names to evaluate (e.g. aspirin benzene).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="Z_paper_content/md17_md_vs_atom_timing.json",
        help="Path to JSON file where ATOM timing statistics will be saved.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    atom_run_dir: Path = Path(args.atom_run_dir)
    output_path: Path = Path(args.output_json)

    stats = run_experiment(
        atom_run_dir=atom_run_dir,
        num_repeats=args.num_repeats,
        limit_molecules=args.molecules,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved ATOM inference timing results to {output_path}")


if __name__ == "__main__":
    main()
