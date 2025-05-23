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
        dataset_name = path_parts[1]  # Gets 'md17' or 'rmd17'

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


def main(directory: Path, datasets: list[str]) -> None:
    run_times = get_run_times(directory)

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
    for dataset in datasets:
        print(f"\n{dataset.upper()}:")
        molecule_times = run_times[dataset]

        for molecule, times in sorted(molecule_times.items()):
            mean_time = float(np.mean(times))
            std_time = float(np.std(times))

            print(f"{molecule}: \\( {format_latex_time(mean_time)}{{\\scriptstyle \\pm{format_latex_time(std_time)}}} \\)")

        print(f"\nDataset mean: \\( {format_latex_time(dataset_means[dataset])}{{\\scriptstyle \\pm{format_latex_time(dataset_stds[dataset])}}} \\)")


if __name__ == "__main__":
    # main(directory=Path("benchmark_runs/md_and_rmd"), datasets=["md17"])
    main(directory=Path("benchmark_runs/md_and_rmd_egno"), datasets=["md17"])
    # main(directory=Path("benchmark_runs/tg80_egno_st"), datasets=["tg80"])
    # main(directory=Path("benchmark_runs/tg80_atom_st"), datasets=["tg80"])

    # main(directory=Path("benchmark_runs/tg80_atom_mt"), datasets=["tg80"])
    # main(directory=Path("benchmark_runs/tg80_egno_mt"), datasets=["tg80"])
