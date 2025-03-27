import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Any


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


def create_corrected_volatility_visualization(data_dir: Path) -> None:
    """Create simplified visualization with proper log scaling and no color dimension."""
    import matplotlib.pyplot as plt

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
        if "R" not in data.files:
            continue

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
        step_volatilities.append(float(np.mean(var_displacement_per_atom)))

    # Create scatter plot with simplified axes
    plt.figure(figsize=(10, 8))

    # Use matplotlib's built-in log scale
    plt.scatter(position_variances, step_volatilities, s=100)

    # Set x-axis to log scale
    plt.xscale("log")

    # Add labels for each point
    for i, molecule in enumerate(molecules):
        plt.annotate(molecule.capitalize(), (position_variances[i], step_volatilities[i]), fontsize=10, ha="center", va="bottom", fontweight="bold")

    plt.xlabel("Position Variance - Flying Away Metric")
    plt.ylabel("Step Volatility - Jiggling Metric")
    plt.title("Molecule Behavior Comparison")
    plt.grid(True, alpha=0.3)

    # Add quadrant explanations
    median_volatility = np.median(step_volatilities)
    median_variance = np.median(position_variances)

    plt.axhline(y=median_volatility, color="gray", linestyle="--", alpha=0.5)
    plt.axvline(x=median_variance, color="gray", linestyle="--", alpha=0.5)

    plt.text(0.9, 0.9, "Flying Away & Jiggling", transform=plt.gca().transAxes, ha="right", va="top", bbox=dict(facecolor="white", alpha=0.7), fontsize=12)

    plt.text(0.1, 0.9, "Jiggling in Place", transform=plt.gca().transAxes, ha="left", va="top", bbox=dict(facecolor="white", alpha=0.7), fontsize=12)

    plt.text(0.1, 0.1, "Stable", transform=plt.gca().transAxes, ha="left", va="bottom", bbox=dict(facecolor="white", alpha=0.7), fontsize=12)

    plt.text(0.9, 0.1, "Flying Away Smoothly", transform=plt.gca().transAxes, ha="right", va="bottom", bbox=dict(facecolor="white", alpha=0.7), fontsize=12)

    plt.tight_layout()
    plt.savefig("/home/luke/gtno_py/Z_paper_content/dataset/molecule_behavior_comparison.pdf", format="pdf")


def analyze_trajectory_stability(data_dir: Path) -> None:
    """Directly analyze and visualize which molecules fly away vs. stay in place."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Lists to store data
    molecules: list[str] = []
    total_displacements: list[float] = []  # Start-to-end displacement
    internal_jiggling: list[float] = []  # Average internal atomic motion

    # Process each NPZ file
    npz_files: list[Path] = list(data_dir.glob("*.npz"))
    for filepath in npz_files:
        # Extract molecule name from filename
        molecule_name: str = filepath.stem.split("_")[-1]

        # Load data
        data = np.load(filepath)
        if "R" not in data.files:
            continue

        arr: npt.NDArray[np.number] = data["R"]
        if arr.ndim != 3 or arr.shape[2] != 3:
            continue

        molecules.append(molecule_name)

        # Calculate center of mass trajectory for each frame
        com_trajectory: npt.NDArray[np.float64] = np.mean(arr, axis=1)

        # 1. Total displacement (direct measure of "flying away")
        # How far the molecule's center of mass moves from start to end
        start_to_end_displacement: float = float(np.linalg.norm(com_trajectory[-1] - com_trajectory[0]))
        total_displacements.append(start_to_end_displacement)

        # 2. Internal jiggling (measure of atomic movement relative to center of mass)
        # Subtract the center of mass from each atom's position to get relative positions
        relative_positions: npt.NDArray[np.float64] = arr - com_trajectory[:, np.newaxis, :]

        # Calculate the average displacement in these relative positions
        relative_displacements: npt.NDArray[np.float64] = np.diff(relative_positions, axis=0)
        relative_displacement_norms: npt.NDArray[np.float64] = np.linalg.norm(relative_displacements, axis=2)
        avg_internal_motion: float = float(np.mean(relative_displacement_norms))
        internal_jiggling.append(avg_internal_motion)

    # Create main visualization: Displacement vs Internal Jiggling
    plt.figure(figsize=(12, 8))

    # Create bar chart of displacements (flying away metric)
    # Sort by displacement for better visualization
    sorted_indices: list[int] = np.argsort(total_displacements)[::-1]  # Descending order
    sorted_molecules: list[str] = [molecules[i] for i in sorted_indices]
    sorted_displacements: list[float] = [total_displacements[i] for i in sorted_indices]
    sorted_jiggling: list[float] = [internal_jiggling[i] for i in sorted_indices]

    # Plot both metrics side by side with different colors
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot displacement bars (flying away metric)
    bars1 = ax1.bar(np.arange(len(sorted_molecules)), sorted_displacements, width=0.4, color="red", alpha=0.7, label="Displacement (Flying Away)")
    ax1.set_ylabel("Total Displacement (Å)", color="red")
    ax1.tick_params(axis="y", labelcolor="red")

    # Create a second y-axis for jiggling
    ax2 = ax1.twinx()
    bars2 = ax2.bar(np.arange(len(sorted_molecules)) + 0.4, sorted_jiggling, width=0.4, color="blue", alpha=0.7, label="Internal Jiggling")
    ax2.set_ylabel("Internal Atomic Motion (Å)", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    # Add molecule names as x-tick labels
    plt.xticks(np.arange(len(sorted_molecules)) + 0.2, sorted_molecules, rotation=45, ha="right")
    plt.title("Molecule Behavior: Flying Away vs. Internal Jiggling")

    # Combine both legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.savefig("/home/luke/gtno_py/Z_paper_content/dataset/molecule_stability_analysis.pdf", format="pdf")

    # Print summary table for clarity
    print("\nMolecule Stability Summary:")
    print("--------------------------")
    print(f"{'Molecule':<15} {'Total Displacement':<20} {'Internal Jiggling':<20} {'Verdict':<20}")
    print("-" * 75)

    for i, mol in enumerate(sorted_molecules):
        displacement = sorted_displacements[i]
        jiggling = sorted_jiggling[i]

        # Calculate Z-scores for displacement and jiggling
        displacement_mean = np.mean(sorted_displacements)
        displacement_std = np.std(sorted_displacements)
        jiggling_mean = np.mean(sorted_jiggling)
        jiggling_std = np.std(sorted_jiggling)

        # Calculate Z-scores
        displacement_z = (displacement - displacement_mean) / displacement_std if displacement_std > 0 else 0
        jiggling_z = (jiggling - jiggling_mean) / jiggling_std if jiggling_std > 0 else 0

        # Determine verdict based on Z-scores
        if displacement_z > 2:
            verdict = "FLIES AWAY"
        elif jiggling_z > 2:
            verdict = "Jiggles in place"
        else:
            verdict = "Stable"

        print(f"{mol:<15} {displacement:<20.4f} {jiggling:<20.4f} {verdict:<20}")


if __name__ == "__main__":
    from figures import set_matplotlib_style

    set_matplotlib_style()
    orig_data_dir: Path = Path("data/md17_npz")
    refresh_data_dir: Path = Path("data/rmd17_npz")
    # print_theory_summary(orig_data_dir)

    original_benzene_file: Path = orig_data_dir / "md17_ethanol.npz"
    print_file_info(original_benzene_file)
    # refresh_benzene_file: Path = refresh_data_dir / "rmd17_benzene.npz"
    # print_file_info(refresh_benzene_file)

    # Create corrected volatility visualization
    create_corrected_volatility_visualization(orig_data_dir)

    # Analyze trajectory stability
    analyze_trajectory_stability(orig_data_dir)
