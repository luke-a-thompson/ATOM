import json
import glob
import os
from pathlib import Path
import numpy as np
from collections import defaultdict
import re
import argparse

## Example usage:
# uv run python -m Z_paper_content.get_runtimes /home/luke/gtno_py/benchmark_runs/tg80_egno_st /home/luke/gtno_py/benchmark_runs/tg80_atom_st --dataset tg80 --out /home/luke/gtno_py/Z_paper_content/tables/runtime_tg80.tex


def get_run_times(directory: Path) -> dict[str, dict[str, list[float]]]:
    """Get run times from all results.json files in the directory.

    Args:
        directory: Path to directory containing results.json files

    Returns:
        Dictionary mapping dataset names to dictionaries of molecule names to lists of run times
    """
    run_times: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    # Find all results.json files
    json_files = glob.glob(os.path.join(directory, "**/results.json"), recursive=True)

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        # Extract dataset name and molecule type from path and config
        path_parts = Path(json_file).parent.name.split("_")

        if "rmd17" in path_parts:
            dataset_name = "rmd17"
        elif "md17" in path_parts:
            dataset_name = "md17"
        elif "tg80" in path_parts:
            dataset_name = "tg80"
        else:
            dataset_name = path_parts[1]

        # Try to get molecule type from config, fallback to filename if key doesn't exist
        try:
            molecule = data["config"]["dataloader"]["molecule_type"]
        except (KeyError, TypeError):
            # Extract fold information from filename
            filename = Path(json_file).parent.name

            # Look for fold pattern in the filename
            fold_match = re.search(r"fold(\d+)", filename)
            if fold_match:
                molecule = f"fold{fold_match.group(1)}"
            else:
                molecule = filename

        # Get run times from single_run_results
        times = [float(run["run_time"]) for run in data["single_run_results"]]
        run_times[dataset_name][molecule].extend(times)

    return run_times


def format_latex_time(seconds: float) -> str:
    """Convert seconds to decimal minutes and format for LaTeX.

    Args:
        seconds: Time in seconds

    Returns:
        String in LaTeX format with decimal minutes
    """
    minutes = seconds / 60
    return f"{minutes:.2f}"


def calculate_total_flops(f_peak: float, minutes: float) -> float:
    """Calculate total FLOPS based on peak FLOPS and time in minutes.

    Args:
        f_peak: Peak FLOPS in TFLOPS
        minutes: Time in minutes

    Returns:
        Total FLOPS
    """
    # Convert TFLOPS to FLOPS and minutes to seconds
    f_peak_flops = f_peak * 1e12  # Convert TFLOPS to FLOPS
    seconds = minutes * 60  # Convert minutes to seconds
    return f_peak_flops * seconds


def calculate_epochs_per_minute(time_seconds: float) -> float:
    """Calculates epochs per minute, assuming a run of 1000 epochs.

    Args:
        time_seconds: Time in seconds for 1000 epochs

    Returns:
        Number of epochs processed per minute
    """
    if time_seconds <= 0:
        return 0.0
    minutes = time_seconds / 60
    return 1000 / minutes


def format_scientific(value: float) -> str:
    """Format a number in scientific notation for LaTeX.

    Args:
        value: Number to format

    Returns:
        String in LaTeX scientific notation format
    """
    if value >= 1e12:
        return f"{value/1e12:.2f}\\times 10^{{12}}"
    elif value >= 1e9:
        return f"{value/1e9:.2f}\\times 10^{{9}}"
    elif value >= 1e6:
        return f"{value/1e6:.2f}\\times 10^{{6}}"
    elif value >= 1e3:
        return f"{value/1e3:.2f}\\times 10^{{3}}"
    else:
        return f"{value:.2f}"


# Canonical MD17 molecule ordering and display names (duplicated from create_mse_tables.py to avoid import-time issues)
MD17_MOLECULE_ORDER: list[str] = [
    "aspirin",
    "benzene",
    "ethanol",
    "malonaldehyde",
    "naphthalene",
    "salicylic",
    "toluene",
    "uracil",
]

MD17_MOLECULE_DISPLAY: dict[str, str] = {
    "aspirin": "Aspirin",
    "benzene": "Benzene",
    "ethanol": "Ethanol",
    "malonaldehyde": "Malonaldehyde",
    "naphthalene": "Naphthalene",
    "salicylic": "Salicylic",
    "toluene": "Toluene",
    "uracil": "Uracil",
}


def _display_molecule_name(molecule: str) -> str:
    if molecule in MD17_MOLECULE_DISPLAY:
        return MD17_MOLECULE_DISPLAY[molecule]
    return molecule.replace("_", " ").title()


def _compute_order(molecules: list[str]) -> list[str]:
    s: set[str] = set(molecules)
    md17_set: set[str] = set(MD17_MOLECULE_ORDER)
    if s.issubset(md17_set):
        return [m for m in MD17_MOLECULE_ORDER if m in s]
    return sorted(molecules)


def _format_minutes_with_std(mean_seconds: float, std_seconds: float) -> str:
    mean_mins: float = mean_seconds / 60.0
    std_mins: float = std_seconds / 60.0
    return f"\\({mean_mins:.2f}{{\\scriptstyle \\pm{std_mins:.2f}}}\\)"


def _format_epochs_per_min(mean_seconds: float) -> str:
    epm: float = calculate_epochs_per_minute(mean_seconds)
    return f"\\({epm:.2f}\\)"


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if len(values) > 0 else float("nan")


def _safe_std(values: list[float]) -> float:
    return float(np.std(values)) if len(values) > 0 else float("nan")


def _collect_model_stats(directory: Path, dataset: str, f_peak_tflops: float) -> dict[str, dict[str, float]]:
    """Collect per-molecule runtime statistics for a single model directory.

    Returns a mapping: molecule -> { mean_seconds, std_seconds, total_flops_1e12, epochs_per_min }
    """
    rt: dict[str, dict[str, list[float]]] = get_run_times(directory)
    mol_to_stats: dict[str, dict[str, float]] = {}
    if dataset not in rt:
        return mol_to_stats
    for molecule, times in rt[dataset].items():
        mean_seconds: float = _safe_mean([float(t) for t in times])
        std_seconds: float = _safe_std([float(t) for t in times])
        mean_minutes: float = mean_seconds / 60.0
        total_flops: float = calculate_total_flops(f_peak_tflops, mean_minutes)  # in FLOPS
        total_flops_1e12: float = total_flops / 1e12
        epochs_per_min: float = calculate_epochs_per_minute(mean_seconds)
        mol_to_stats[molecule] = {
            "mean_seconds": mean_seconds,
            "std_seconds": std_seconds,
            "total_flops_1e12": total_flops_1e12,
            "epochs_per_min": epochs_per_min,
        }
    return mol_to_stats


def build_runtime_table(egno_dir: Path, atom_dir: Path, dataset: str, f_peak_tflops: float = 15.0) -> str:
    """Build LaTeX for the runtime table comparing EGNO vs ATOMS for one dataset."""
    egno_stats: dict[str, dict[str, float]] = _collect_model_stats(egno_dir, dataset, f_peak_tflops)
    atom_stats: dict[str, dict[str, float]] = _collect_model_stats(atom_dir, dataset, f_peak_tflops)

    all_molecules: list[str] = sorted(set(egno_stats.keys()) | set(atom_stats.keys()))
    if len(all_molecules) == 0:
        return "% No runtime data found"
    molecule_order: list[str] = _compute_order(all_molecules)

    # Header
    lines: list[str] = []
    lines.append("\\begin{tabular}{l" + ("c" * (len(molecule_order) + 2)) + "}")
    lines.append("        \\toprule")
    header_cells: list[str] = ["Model", "", *[_display_molecule_name(m) for m in molecule_order], "Mean \\%"]
    lines.append("        " + " & ".join(header_cells) + " \\")
    lines.append("        \\midrule")

    # EGNO rows
    def _row_for(model_gls: str, stats: dict[str, dict[str, float]]) -> None:
        # Time (mins)
        row_cells: list[str] = [model_gls, "Time (mins)"]
        for m in molecule_order:
            s = stats.get(m)
            if s is None:
                row_cells.append("")
            else:
                row_cells.append(_format_minutes_with_std(s["mean_seconds"], s["std_seconds"]))
        row_cells.append("")
        lines.append("        " + " & ".join(row_cells) + " \\")

        # Total FLOPS (x1e12)
        row_cells = ["", "Total FLOPS ($\\times10^{12}$)"]
        for m in molecule_order:
            s = stats.get(m)
            if s is None:
                row_cells.append("")
            else:
                row_cells.append(f"{s['total_flops_1e12']:.2f}")
        row_cells.append("")
        lines.append("        " + " & ".join(row_cells) + " \\")

        # Epochs/min
        row_cells = ["", "Epochs/min"]
        for m in molecule_order:
            s = stats.get(m)
            if s is None:
                row_cells.append("")
            else:
                row_cells.append(_format_epochs_per_min(s["mean_seconds"]))
        row_cells.append("")
        lines.append("        " + " & ".join(row_cells) + " \\")

    _row_for("\\gls{egno}", egno_stats)
    lines.append("        \\midrule")
    _row_for("\\gls{atoms}", atom_stats)
    lines.append("        \\midrule")

    # Reduction row (EGNO -> ATOMS) based on total FLOPS
    reductions: list[str] = []
    reduction_vals: list[float] = []
    for m in molecule_order:
        se = egno_stats.get(m)
        sa = atom_stats.get(m)
        if se is None or sa is None:
            reductions.append("")
            continue
        base: float = se["total_flops_1e12"]
        tgt: float = sa["total_flops_1e12"]
        if base == 0.0 or not (base == base) or not (tgt == tgt):
            reductions.append("")
            continue
        red: float = (base - tgt) / base * 100.0
        reduction_vals.append(red)
        reductions.append(f"\\({red:.2f}\\%\\)")
    mean_red_cell: str = ""
    if len(reduction_vals) > 0:
        mean_red: float = sum(reduction_vals) / float(len(reduction_vals))
        mean_red_cell = f"\\(\\mathbf{{{mean_red:.2f}\\%}}\\)"

    lines.append("        \\rowcolor{gray!20}")
    lines.append("        \\multicolumn{2}{c}{Total FLOPS Reduction (\\%)} " + " & " + " & ".join(reductions) + " & " + mean_red_cell + " \\")
    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}")
    return "\n".join(lines) + "\n"


def main(directory: Path, datasets: list[str], model_type: str) -> dict[str, float]:
    run_times = get_run_times(directory)

    # Hardware and model parameters
    f_peak = 15.0  # TFLOPS for Titan V

    # Calculate mean times across all molecules for each dataset
    dataset_means: dict[str, float] = {}
    dataset_stds: dict[str, float] = {}
    for dataset, molecule_times in run_times.items():
        all_times = []
        for times in molecule_times.values():
            all_times.extend(times)
        dataset_means[dataset] = float(np.mean(all_times))
        dataset_stds[dataset] = float(np.std(all_times))

    # Process each dataset
    print(f"----- {model_type.upper()} MODEL -----")

    molecule_flops: dict[str, float] = {}

    for dataset in datasets:
        if dataset not in run_times:
            print(f"\nNo data found for {dataset.upper()} in {directory}")
            continue

        print(f"\n{dataset.upper()}:")
        molecule_times = run_times[dataset]

        for molecule, times in sorted(molecule_times.items()):
            mean_time = float(np.mean(times))
            std_time = float(np.std(times))

            # Convert to minutes for calculations
            mean_minutes = mean_time / 60

            # Calculate statistics
            total_flops = calculate_total_flops(f_peak, mean_minutes)
            epochs_per_min = calculate_epochs_per_minute(mean_time)
            molecule_flops[molecule] = total_flops

            print(f"\t{molecule}")
            print(f"\t\tTime (mins): \\( {format_latex_time(mean_time)}{{\\scriptstyle \\pm{format_latex_time(std_time)}}} \\)")
            print(f"\t\tTotal FLOPS: \\( {format_scientific(total_flops)} \\)")
            print(f"\t\tEpochs/min: \\( {epochs_per_min:.2f} \\)")

        # Calculate dataset-level statistics
        if dataset in dataset_means:
            dataset_mean_time_seconds = dataset_means[dataset]
            dataset_mean_minutes = dataset_mean_time_seconds / 60
            dataset_total_flops = calculate_total_flops(f_peak, dataset_mean_minutes)
            dataset_epochs_per_min = calculate_epochs_per_minute(dataset_mean_time_seconds)

            print(f"\nDataset mean:")
            print(f"\tTime (mins): \\( {format_latex_time(dataset_means[dataset])}{{\\scriptstyle \\pm{format_latex_time(dataset_stds[dataset])}}} \\)")
            print(f"\tTotal FLOPS: \\( {format_scientific(dataset_total_flops)} \\)")
            print(f"\tEpochs/min: \\( {dataset_epochs_per_min:.2f} \\)")

    return molecule_flops


def runtime_table_cli() -> int:
    parser = argparse.ArgumentParser(description="Build runtime LaTeX table and write to Z_paper_content/tables.")
    _ = parser.add_argument("egno_dir", type=str, help="Directory with EGNO runs (contains results.json files)")
    _ = parser.add_argument("atom_dir", type=str, help="Directory with ATOMS/GTNO runs (contains results.json files)")
    _ = parser.add_argument("--dataset", type=str, default="md17", help="Dataset key to aggregate (e.g., md17, rmd17, tg80)")
    _ = parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional explicit output .tex path. Defaults to Z_paper_content/tables/runtime_<dataset>.tex",
    )
    _ = parser.add_argument("--f-peak", type=float, default=15.0, help="Peak TFLOPS of the GPU (default: 15.0)")

    args = parser.parse_args()
    egno_dir: Path = Path(args.egno_dir).expanduser().resolve()
    atom_dir: Path = Path(args.atom_dir).expanduser().resolve()
    if not egno_dir.exists():
        raise SystemExit(f"EGNO directory does not exist: {egno_dir}")
    if not atom_dir.exists():
        raise SystemExit(f"ATOMS directory does not exist: {atom_dir}")

    latex_text: str = build_runtime_table(egno_dir=egno_dir, atom_dir=atom_dir, dataset=str(args.dataset), f_peak_tflops=float(args.f_peak))

    if args.out is not None:
        out_path: Path = Path(args.out).expanduser().resolve()
    else:
        out_path = (Path(__file__).resolve().parent / "tables" / f"runtime_{args.dataset}.tex").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex_text, encoding="utf-8")
    print(latex_text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(runtime_table_cli())
