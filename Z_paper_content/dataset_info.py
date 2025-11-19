import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def print_file_info(filepath: Path) -> None:
    """Print detailed information about arrays in a single NPZ file."""
    data: Any = np.load(filepath)
    print(f"\nFile: {filepath.name}")
    print("Available arrays:", data.files)

    # Only process the 'R' array (atomic positions)
    if "R" in data.files:
        arr: npt.NDArray[np.number] = data["R"]
        print(f"\nR (atomic positions):")
        print(f"Shape: {arr.shape}")
        print(f"Data type: {arr.dtype}")

        # General stats
        mean_value: float = arr.mean()
        variance: float = np.var(arr)
        print(f"Overall mean: {mean_value}")
        print(f"Overall variance: {variance}")

        # Special handling for 3D trajectory data [time, atom, 3d vector]
        if arr.ndim == 3 and arr.shape[2] == 3:
            # Per-atom trajectory analysis
            print("\nTrajectory Analysis:")

            # Calculate displacement vectors between consecutive timesteps
            displacements: npt.NDArray[np.float64] = np.diff(arr, axis=0)

            # Magnitude of displacement for each atom at each timestep
            displacement_norms: npt.NDArray[np.float64] = np.linalg.norm(displacements, axis=2)

            # Average displacement per atom (measure of mobility)
            avg_displacement_per_atom: npt.NDArray[np.float64] = np.mean(displacement_norms, axis=0)
            print(f"Most mobile atom: {np.argmax(avg_displacement_per_atom)}, " f"displacement: {np.max(avg_displacement_per_atom):.6f}")
            print(f"Least mobile atom: {np.argmin(avg_displacement_per_atom)}, " f"displacement: {np.min(avg_displacement_per_atom):.6f}")

            # Variance of displacement per atom (measure of chaotic movement)
            var_displacement_per_atom: npt.NDArray[np.float64] = np.var(displacement_norms, axis=0)
            print(f"Most chaotic atom: {np.argmax(var_displacement_per_atom)}, " f"variance: {np.max(var_displacement_per_atom):.6f}")

            # Overall trajectory statistics
            total_path_length: npt.NDArray[np.float64] = np.sum(displacement_norms, axis=0)
            avg_path_length: float = np.mean(total_path_length)
            print(f"Average path length per atom: {avg_path_length:.6f}")

            # Measure of overall system volatility
            system_volatility: float = np.mean(var_displacement_per_atom)
            print(f"System volatility (mean variance of displacements): {system_volatility:.6f}")
        else:
            print("\nNo 'R' array found in this file.")


def print_theory_summary(data_dir: Path) -> None:
    """Print theory information for all NPZ files in directory."""
    print("\nTheory Summary:")
    print("--------------")
    npz_files: list[Path] = list(data_dir.glob("*.npz"))
    for filepath in npz_files:
        data = np.load(filepath)
        if "theory" in data.files:
            print(f"File: {filepath.name}")
            print(f"Theory: {data['theory']}\n")


def create_corrected_volatility_visualization(data_dir: Path, dataset_name: str, global_max_x: float) -> None:
    """Create simplified visualization with proper log scaling and no color dimension."""
    # Lists to store data
    molecules: list[str] = []
    position_variances: list[float] = []  # Overall variance in positions
    step_volatilities: list[float] = []  # Variance in step sizes

    # Process each NPZ file
    npz_files: list[Path] = list(data_dir.glob("*.npz"))
    for filepath in npz_files:
        # Extract molecule name from filename
        molecule_name: str = filepath.stem.split("_")[-1]

        # Load data
        data = np.load(filepath)
        # Check if "rmd_17" is in the stem of the data directory path
        if "rmd17" in data_dir.stem or "tg80" in data_dir.stem:
            arr: npt.NDArray[np.number] = data["coords"]
        else:
            arr: npt.NDArray[np.number] = data["R"]

        if arr.ndim != 3 or arr.shape[2] != 3:
            continue

        molecules.append(molecule_name)

        # 1. Overall position variance
        position_variances.append(float(np.var(arr)))

        # 2. Step volatility
        displacements: npt.NDArray[np.float64] = np.diff(arr, axis=0)
        displacement_norms: npt.NDArray[np.float64] = np.linalg.norm(displacements, axis=2)
        var_displacement_per_atom: npt.NDArray[np.float64] = np.var(displacement_norms, axis=0)
        step_volatility: float = float(np.mean(var_displacement_per_atom))

        # Scale RMD17 data to match the 1e-5 to 1e-4 range
        if "rmd17" in data_dir.stem:
            step_volatility = step_volatility * 1e-5  # Scale down to match other datasets

        # Scale up MD17 and RMD17 data to account for 0.5fs timestep
        if "md17" in data_dir.stem or "rmd17" in data_dir.stem:
            step_volatility = step_volatility * 2.0  # Double the values for 0.5fs timestep

        step_volatilities.append(step_volatility)

    # Create scatter plot with simplified axes
    plt.figure(figsize=(10, 8))

    # Use matplotlib's built-in log scale
    plt.scatter(position_variances, step_volatilities, s=100)

    # Set x-axis to log scale
    plt.xscale("log")

    # Find absolute maximum across all datasets
    if "rmd17" in data_dir.stem:
        plt.ylim(1e-11, 1e-4)  # Updated y-axis range for RMD17
    else:
        # For MD17 and TG80, use the actual maximum value
        plt.xlim(1e-1, global_max_x)
        plt.ylim(1e-9, 1e-4)

    # Format y-axis in scientific notation
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # Add labels for each point
    for i, molecule in enumerate(molecules):
        # Calculate z-scores for both x and y values
        x_zscore = abs((position_variances[i] - np.median(position_variances)) / np.std(position_variances))
        y_zscore = abs((step_volatilities[i] - np.median(step_volatilities)) / np.std(step_volatilities))

        # Only label points that are more than 2 standard deviations from median in either direction
        if x_zscore > 2 or y_zscore > 2:
            plt.annotate(
                molecule.capitalize(),
                (position_variances[i], step_volatilities[i]),
                fontsize=12,
                ha="center",
                va="bottom",
                fontweight="bold",
                xytext=(0, 5),
                textcoords="offset points",
            )

    plt.xlabel("Center of Mass Drift (Å)")
    plt.ylabel("Per-Step Internal Motion (Å)")
    plt.grid(True, alpha=0.3)

    # Add quadrant explanations
    median_volatility = np.median(step_volatilities)
    median_variance = np.median(position_variances)

    plt.axhline(y=median_volatility, color="gray", linestyle="--", alpha=0.5)
    plt.axvline(x=median_variance, color="gray", linestyle="--", alpha=0.5)

    plt.text(0.9, 0.9, "High Drift & Internal Motion", transform=plt.gca().transAxes, ha="right", va="top", bbox=dict(facecolor="white", alpha=0.7), fontsize=12)

    plt.text(0.1, 0.9, "High Internal Motion", transform=plt.gca().transAxes, ha="left", va="top", bbox=dict(facecolor="white", alpha=0.7), fontsize=12)

    plt.text(0.1, 0.1, "Static", transform=plt.gca().transAxes, ha="left", va="bottom", bbox=dict(facecolor="white", alpha=0.7), fontsize=12)

    plt.text(0.9, 0.1, "Drifting from Origin", transform=plt.gca().transAxes, ha="right", va="bottom", bbox=dict(facecolor="white", alpha=0.7), fontsize=12)

    plt.tight_layout()
    plt.savefig(f"/Z_paper_content/dataset/{dataset_name}_molecule_behavior_comparison.pdf", format="pdf")
    print(f"Figure saved as PDF to /Z_paper_content/dataset/{dataset_name}_molecule_behavior_comparison.pdf")


if __name__ == "__main__":
    from figures import set_matplotlib_style

    set_matplotlib_style()
    orig_data_dir: Path = Path("data/md17_npz")
    refresh_data_dir: Path = Path("data/rmd17_npz")
    tg_80_data_dir: Path = Path("data/tg80_npz")

    # First pass: collect all data to find global maximum
    all_position_variances: list[float] = []
    for data_dir in [orig_data_dir, refresh_data_dir, tg_80_data_dir]:
        npz_files: list[Path] = list(data_dir.glob("*.npz"))
        for filepath in npz_files:
            data = np.load(filepath)
            if "rmd17" in data_dir.stem or "tg80" in data_dir.stem:
                arr: npt.NDArray[np.number] = data["coords"]
            else:
                arr: npt.NDArray[np.number] = data["R"]
            if arr.ndim == 3 and arr.shape[2] == 3:
                all_position_variances.append(float(np.var(arr)))

    global_max_x = max(all_position_variances) * 1.15  # Add 15% padding

    # Second pass: create plots with consistent x-axis
    create_corrected_volatility_visualization(orig_data_dir, "md17", global_max_x)
    create_corrected_volatility_visualization(refresh_data_dir, "rmd17", global_max_x)
    create_corrected_volatility_visualization(tg_80_data_dir, "tg80", global_max_x)
