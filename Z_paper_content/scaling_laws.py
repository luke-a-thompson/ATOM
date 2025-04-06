import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from pathlib import Path
from usyd_colors import get_palette
from figures import set_matplotlib_style

# Get USYD colors
grey, red, blue, yellow, white = get_palette("primary").hex_colors()


def scaling_laws_traj_length() -> None:
    """
    Create two line charts with placeholder data using USYD colors.
    Save each chart as a separate PDF to the scaling_laws directory.

    Returns:
        None
    """
    # Set matplotlib style
    set_matplotlib_style()

    # Create placeholder data
    x_values: npt.NDArray[np.int32] = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Data for first chart - 5 different lines
    y_values1: list[npt.NDArray[np.float64]] = [
        np.array([0.65, 0.68, 0.72, 0.75, 0.78, 0.80, 0.82, 0.84, 0.86, 0.87]),
        np.array([0.60, 0.64, 0.69, 0.73, 0.77, 0.80, 0.83, 0.85, 0.87, 0.88]),
        np.array([0.55, 0.60, 0.65, 0.70, 0.75, 0.79, 0.82, 0.85, 0.87, 0.89]),
        np.array([0.50, 0.56, 0.62, 0.68, 0.74, 0.78, 0.82, 0.85, 0.87, 0.90]),
        np.array([0.45, 0.52, 0.59, 0.66, 0.73, 0.78, 0.82, 0.85, 0.87, 0.91]),
    ]

    # Data for second chart - 5 different lines
    y_values2: list[npt.NDArray[np.float64]] = [
        np.array([0.70, 0.73, 0.76, 0.78, 0.80, 0.82, 0.83, 0.84, 0.85, 0.86]),
        np.array([0.65, 0.69, 0.72, 0.75, 0.77, 0.79, 0.81, 0.82, 0.83, 0.84]),
        np.array([0.60, 0.65, 0.69, 0.72, 0.75, 0.77, 0.79, 0.80, 0.81, 0.82]),
        np.array([0.55, 0.61, 0.66, 0.70, 0.73, 0.76, 0.78, 0.79, 0.80, 0.81]),
        np.array([0.50, 0.57, 0.63, 0.67, 0.71, 0.74, 0.76, 0.77, 0.78, 0.79]),
    ]

    # Line labels
    line_labels: list[str] = ["Line A", "Line B", "Line C", "Line D", "Line E"]

    # Colors for the lines
    line_colors: list[str] = [red, blue, yellow, grey, "purple"]

    # Ensure the directory exists
    save_dir: Path = Path("Z_paper_content/scaling_laws")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create and save first chart
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    for i in range(5):
        ax1.plot(x_values, y_values1[i], color=line_colors[i], linewidth=2, label=line_labels[i], marker="o", markersize=4)

    ax1.set_xlabel("Mean fs per molecule", fontsize=14)
    ax1.set_ylabel("S2S MSE ($\\times 10^{-2}$)", fontsize=14)
    ax1.set_ylim(0.4, 1.0)
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend(loc="lower right")

    # Remove top and right spines for first chart
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Adjust layout
    plt.tight_layout()

    # Save the first figure
    save_path1: Path = save_dir / "scaling_law_1.pdf"
    plt.savefig(save_path1, format="pdf", dpi=300, bbox_inches="tight")
    print(f"First figure saved as PDF to {save_path1}")

    plt.close(fig1)

    # Create and save second chart
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    for i in range(5):
        ax2.plot(x_values, y_values2[i], color=line_colors[i], linewidth=2, label=line_labels[i], marker="o", markersize=4)

    ax2.set_xlabel("Number of Training Molecules", fontsize=14)
    ax2.set_ylabel("S2S MSE ($\\times 10^{-2}$)", fontsize=14)
    ax2.set_ylim(0.4, 1.0)
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend(loc="lower right")

    # Remove top and right spines for second chart
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Adjust layout
    plt.tight_layout()

    # Save the second figure
    save_path2: Path = save_dir / "scaling_law_2.pdf"
    plt.savefig(save_path2, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Second figure saved as PDF to {save_path2}")

    plt.close(fig2)


if __name__ == "__main__":
    scaling_laws_traj_length()
