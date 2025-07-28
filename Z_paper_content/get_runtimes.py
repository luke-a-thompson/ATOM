import json
import glob
import os
from pathlib import Path
import numpy as np
from collections import defaultdict
import re


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


if __name__ == "__main__":
    # main(directory=Path("benchmark_runs/md_and_rmd"), datasets=["md17"], model_type="atom")
    # print("\n" + "=" * 50 + "\n")
    # main(directory=Path("benchmark_runs/md_and_rmd_egno"), datasets=["md17"], model_type="egno")

    # main(directory=Path("benchmark_runs/md_and_rmd"), datasets=["rmd17"], model_type="atom")
    # print("\n" + "=" * 50 + "\n")
    # main(directory=Path("benchmark_runs/md_and_rmd_egno"), datasets=["rmd17"], model_type="egno")

    atom_flops = main(directory=Path("benchmark_runs/tg80_atom_st"), datasets=["tg80"], model_type="atom")
    print("\n" + "=" * 50 + "\n")
    egno_flops = main(directory=Path("benchmark_runs/tg80_egno_st"), datasets=["tg80"], model_type="egno")

    print("\n" + "=" * 50 + "\n")
    print("FLOPS Percentage Reduction (EGNO -> ATOM):")

    common_molecules = sorted(list(set(atom_flops.keys()) & set(egno_flops.keys())))

    for molecule in common_molecules:
        egno_f = egno_flops[molecule]
        atom_f = atom_flops[molecule]

        if egno_f > 0:
            reduction = ((egno_f - atom_f) / egno_f) * 100
            print(f"\t{molecule}: {reduction:.2f}%")
        else:
            print(f"\t{molecule}: Cannot calculate reduction, EGNO FLOPS is 0.")
