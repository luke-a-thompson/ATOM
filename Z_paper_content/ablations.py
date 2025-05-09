import json
from pathlib import Path
from enum import Enum

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from usyd_colors import get_palette
from figures import set_matplotlib_style


class ErrorBarType(Enum):
    """Enum for different types of error bars."""

    PERCENTILE = "percentile"
    STANDARD_DEVIATION = "std_dev"


def calculate_asymmetric_variance(values: list[float]) -> tuple[float, float]:
    """
    Calculate the asymmetric variance using 5th and 95th percentiles around the mean.

    Args:
        values: List of values

    Returns:
        Tuple of (lower_bound, upper_bound) representing the asymmetric variance
    """
    mean = float(np.mean(values))
    lower_bound = float(mean - float(np.percentile(values, 5)))
    upper_bound = float(float(np.percentile(values, 95)) - mean)
    return (lower_bound, upper_bound)


def calculate_std_dev_bounds(values: list[float]) -> tuple[float, float]:
    """
    Calculate the 2-sigma standard deviation bounds around the mean.

    Args:
        values: List of values

    Returns:
        Tuple of (lower_bound, upper_bound) representing the 2-sigma bounds
    """
    mean = float(np.mean(values))
    std_dev = float(np.std(values))
    return (2.0 * std_dev, 2.0 * std_dev)  # Symmetric bounds


def plot_ablations(ablation_dir: Path, save_path: Path | None = None, error_bar_type: ErrorBarType = ErrorBarType.PERCENTILE) -> None:
    """
    Create horizontal bars using real benchmark data from benchmark_runs/Ablations.
    Calculates mean test loss and error bars from individual run results.

    Args:
        ablation_dir: Directory containing ablation results
        save_path: Optional path to save the figure
        error_bar_type: Type of error bars to display (percentile or standard deviation)

    Returns:
        None
    """
    # Get colors from the palette
    grey, red, blue, yellow, white = get_palette("primary").hex_colors()

    # Get all results.json files from the ablations directory
    results_files: list[Path] = list(ablation_dir.glob("**/results.json"))

    # Collect data from results files
    data_dict: dict[str, tuple[float, float, float]] = {}
    for results_file in results_files:
        with open(results_file, "r") as f:
            data = json.load(f)
            benchmark_name = data["config"]["benchmark"]["benchmark_name"]
            # Convert benchmark name to display name
            # Remove the "gtno_" prefix and convert to title case
            display_name = benchmark_name.replace("gtno_", "").replace("_", " ").title()

            # Keep ROPE in all caps if it exists in the display name
            if "Rope" in display_name:
                display_name = display_name.replace("Rope", "T-RoPE")

            # Extract individual run results
            s2s_test_losses = [run["s2s_test_loss"] for run in data["single_run_results"]]

            # Calculate mean and error bounds
            mean_loss = float(np.mean(s2s_test_losses))

            if error_bar_type == ErrorBarType.PERCENTILE:
                lower_bound, upper_bound = calculate_asymmetric_variance(s2s_test_losses)
            else:  # STANDARD_DEVIATION
                lower_bound, upper_bound = calculate_std_dev_bounds(s2s_test_losses)

            data_dict[display_name] = (mean_loss, lower_bound, upper_bound)

    # Sort the dictionary by values (mean loss) in descending order
    data_dict = dict(sorted(data_dict.items(), key=lambda x: x[1][0]))

    # Extract categories and values
    categories: list[str] = list(data_dict.keys())
    values: npt.NDArray[np.float64] = np.array([v[0] for v in data_dict.values()])
    lower_bounds: npt.NDArray[np.float64] = np.array([v[1] for v in data_dict.values()])
    upper_bounds: npt.NDArray[np.float64] = np.array([v[2] for v in data_dict.values()])

    # Scale values by 10^-2
    values = values * 100
    lower_bounds = lower_bounds * 100
    upper_bounds = upper_bounds * 100

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(5, 4))

    # Create horizontal bars with error bars
    bars = ax.barh(categories, values, color=red, alpha=0.8, edgecolor=grey, linewidth=0)

    # Add error bars
    _ = ax.errorbar(values, categories, xerr=[lower_bounds, upper_bounds], fmt="none", color=grey, capsize=3, capthick=1)
    # Add value label for the highest bar (last item since sorted in ascending order)
    highest_bar_index = len(categories) - 1
    highest_value = values[highest_bar_index]
    highest_category = categories[highest_bar_index]

    # Format the value with 2 decimal places
    value_text = f"{highest_value:.2f}"

    # Position the text slightly to the right of the bar end
    text_x_position = 10.25

    # Add the text annotation
    _ = ax.text(x=text_x_position, y=highest_category, s=value_text, verticalalignment="center", ha="left", color=grey, fontsize=10, fontweight="bold")

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Customize axis
    _ = ax.set_xlabel("Mean S2S MSE ($\\times 10^{-2}$)")
    _ = ax.set_ylabel("")
    _ = ax.set_xlim(0, right=12)

    # Hide y-axis labels (model names)
    # ax.set_yticks([])

    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="x", style="plain", scilimits=(-2, -2))

    # Adjust layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
        print(f"Figure saved as PDF to {save_path}")


if __name__ == "__main__":
    ablation_dir = Path("benchmark_runs/MD17_ablations")
    # Use percentile error bars by default
    set_matplotlib_style()
    plot_ablations(ablation_dir=ablation_dir, save_path=Path("Z_paper_content/ablations/ablation_MD17.pdf"), error_bar_type=ErrorBarType.PERCENTILE)

    # Uncomment to use standard deviation error bars instead
    # plot_ablations(
    #     ablation_dir=ablation_dir,
    #     save_path=Path("Z_paper_content/ablations/ablation_MD17_std_dev.pdf"),
    #     error_bar_type=ErrorBarType.STANDARD_DEVIATION
    # )
