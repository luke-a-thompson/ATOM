import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from usyd_colors import get_palette
import json
from typing import Any
import re
from matplotlib.ticker import ScalarFormatter

grey, red, blue, yellow, white = get_palette("primary").hex_colors()


def set_matplotlib_style(font_size: int = 14) -> None:
    """
    Set the matplotlib style to use Times New Roman font with size 17.

    Returns:
        None
    """
    font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
    fm.fontManager.addfont(font_path)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",  # Directly specify the font family
            "font.size": font_size,
            "mathtext.fontset": "stix",  # Use STIX fonts for math expressions
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
        }
    )


def plot_lambda_value_residuals(weights_dir: Path, figure_file_name: str, figure_dir: Path = Path("Z_paper_content/lambda_value_residuals")) -> None:
    """
    Plot the lambda values as a line chart showing their evolution over time.

    Args:
        weights_dir: Directory containing the weights file
        figure_file_name: Name of the figure file
        figure_dir: Directory to save the figure as PDF

    Returns:
        None
    """
    # Load the lambda values
    weights = np.load(weights_dir / "lambda_v_residual.npz", allow_pickle=True)
    lambda_values = weights["lambda_v_residual"]

    print(f"Lambda values shape: {lambda_values.shape}")

    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get the number of timesteps and lambda values
    n_timesteps: int = lambda_values.shape[0]
    n_lambda_values: int = lambda_values.shape[1]

    # Create 10 evenly spaced indices for every 100 timesteps
    selected_indices: list[int] = []
    for i in range(10):
        selected_indices.append(i * (n_timesteps // 10))
    if n_timesteps - 1 not in selected_indices:
        selected_indices.append(n_timesteps - 1)  # Always include the last element

    # Create x-axis values (timesteps)
    x: npt.NDArray[np.int32] = np.arange(n_timesteps)

    # Define colors for the lambda values
    colors: list[str] = [red, blue, yellow, grey, "purple"]

    # Plot each lambda value as a separate line
    for i in range(n_lambda_values):
        ax.plot(x, lambda_values[:, i], label=f"λ{i+1}", color=colors[i % len(colors)], linewidth=2)

    # Add vertical lines at selected timesteps
    for idx in selected_indices:
        ax.axvline(x=idx, color="gray", linestyle="--", alpha=0.5)

    # Remove top and right spines (borders)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add title and labels
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Lambda Values")

    # Add legend
    ax.legend(loc="best")

    # Ensure the directory exists
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Create the save path
    save_path = figure_dir / f"lambda_values_{Path(figure_file_name).stem}.pdf"

    # Save as PDF with high quality
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Figure saved as PDF to {save_path}")

    plt.show()


def plot_learnable_attention_weights(
    weights_dir: Path,
    figure_file_name: str,
    data_file_name: str = "attention_denom.npz",
    figure_dir: Path = Path("Z_paper_content/attention_weights"),
    step_size: int = 50,
) -> None:
    """
    Plot the learnable attention weights for each layer as a time-series box plot.

    Args:
        weights_dir: Directory containing the weights file
        figure_file_name: Name of the figure file
        data_file_name: Name of the data file containing weights
        figure_dir: Directory to save the figure as PDF
        step_size: Plot every step_size-th element (default: 25)

    Returns:
        None
    """

    weights: npt.NDArray[np.float32] = np.load(weights_dir / data_file_name, allow_pickle=True)
    attention_denom: npt.NDArray[np.float32] = weights["attention_denom"]
    print(f"Attention denom shape: {attention_denom.shape}")

    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Generate positions for the boxes
    n_timesteps: int = attention_denom.shape[0]
    # Flatten the last two dimensions (layers and heads)
    # Original shape: [timesteps, layers, heads]
    # New shape: [timesteps, layers*heads]
    n_layers: int = attention_denom.shape[1]
    n_heads: int = attention_denom.shape[2]
    attention_denom = attention_denom.reshape(n_timesteps, n_layers * n_heads)
    print(f"Reshaped attention denom: {attention_denom.shape} (timesteps, layers*heads)")

    # Select indices at regular intervals based on step_size
    selected_indices: list[int] = list(range(0, n_timesteps, step_size))
    if n_timesteps - 1 not in selected_indices:
        selected_indices.append(n_timesteps - 1)  # Always include the last element

    positions: npt.NDArray[np.int32] = np.array(selected_indices) + 1

    # Create the box plot directly from the numpy array, but only for selected timesteps
    box_plot = ax.boxplot(
        [attention_denom[t] for t in selected_indices],
        positions=positions,
        patch_artist=True,
        widths=step_size * 0.75,
        showfliers=False,  # Hide outliers
    )

    # Customize the box plot appearance
    for box in box_plot["boxes"]:
        box.set(facecolor=blue, alpha=0.8)

    for median in box_plot["medians"]:
        median.set_color(red)  # Set median line color
        median.set_linewidth(2)  # Optional: Adjust line width for better visibility

    # Remove top and right spines (borders)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add title and labels
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Attention Denominator")

    # Set the x-tick positions and labels
    ax.set_xticks(positions)
    # Only add 1 to the first and last indices
    x_labels = []
    for i, idx in enumerate(selected_indices):
        if i == 0 or i == len(selected_indices) - 1:  # First or last element
            x_labels.append(str(idx + 1))
        else:
            x_labels.append(str(idx))
    ax.set_xticklabels(x_labels)

    plt.tight_layout()
    plt.ylim(15, 17)

    # Save the figure
    # Ensure the directory exists
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Create the save path with the filename derived from the weights file
    save_path = figure_dir / f"attention_weights_{Path(figure_file_name).stem}.pdf"

    # Save as PDF with high quality
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Figure saved as PDF to {save_path}")

    plt.show()


from typing import Literal


def plot_invariance_results(invariance_to_plot: Literal["t", "p"]) -> None:
    """
    Plot the results of the invariance ablation study.
    Creates a line chart with shaded 2SD around the line using mean S2T loss and standard deviation.
    Also plots the mean test all loss for both EGNO models from separate directories.

    Args:
        invariance_to_plot: Either "t" for T-invariance or "p" for P-invariance (num_timesteps)
    """
    # Get all results.json files
    figure_dir_name = "invariance_results"
    if invariance_to_plot == "t":
        invariance_dir = Path("benchmark_runs/t_invariance")
        egno_dir = Path("benchmark_runs/t_invariance_egno")
        egno_dir_no_temporal = Path("benchmark_runs/t_invariance_no_temporal")
        x_label = "$ \\Delta t $ (fs)"
        figure_file_name = "t_invariance_results.pdf"
    else:  # p invariance
        invariance_dir = Path("benchmark_runs/p_invariance")
        egno_dir = Path("benchmark_runs/p_invariance_egno_fixedlr")
        egno_dir_no_temporal = None  # Not used for p invariance
        x_label = "Number of Timesteps (P)"
        figure_file_name = "p_invariance_results.pdf"

    # Get results files for all models
    s2t_results_files: list[Path] = list(invariance_dir.glob("**/results.json"))
    egno_results_files: list[Path] = list(egno_dir.glob("*.json"))
    egno_no_temporal_files: list[Path] = list(egno_dir_no_temporal.glob("*.json")) if egno_dir_no_temporal else []

    print(f"Found {len(s2t_results_files)} S2T results files, {len(egno_results_files)} EGNO results files, and {len(egno_no_temporal_files)} EGNO no temporal files")

    # Extract parameter values and corresponding metrics for S2T model
    s2t_param_values: list[int] = []
    s2t_means: list[float] = []
    s2t_stds: list[float] = []

    for results_file in s2t_results_files:
        with open(results_file, "r") as f:
            s2t_data: dict[str, Any] = json.load(f)

            # Extract parameter from the configuration
            s2t_param_value: int = 0  # Default value

            if invariance_to_plot == "t":
                # For T-invariance, extract from benchmark name
                benchmark_name = s2t_data["config"]["benchmark"]["benchmark_name"]
                match = re.search(r"delta_t_(\d+)", benchmark_name)
                if match:
                    s2t_param_value = int(match.group(1))
            else:  # p invariance (num_timesteps)
                # For P-invariance, extract from dataloader configuration
                s2t_param_value = s2t_data["config"]["dataloader"]["num_timesteps"]

            # Add the parameter value and metrics
            s2t_param_values.append(s2t_param_value)
            s2t_means.append(s2t_data["s2t_test_loss_mean"])
            s2t_stds.append(s2t_data["s2t_test_loss_std"])

    # Function to process EGNO results
    def process_egno_files(results_files: list[Path]) -> tuple[list[int], list[float], list[float]]:
        param_values: list[int] = []
        means: list[float] = []
        stds: list[float] = []

        for results_file in results_files:
            with open(results_file, "r") as f:
                data: dict[str, Any] = json.load(f)
                param_value: int = 0

                # Extract from file name
                file_name = results_file.name
                if invariance_to_plot == "t":
                    # Check if this is a no temporal file
                    if "notempconv" in file_name:
                        match = re.search(r"delta_frame_(\d+)_notempconv", file_name)
                    else:
                        match = re.search(r"bigDelta_(\d+)", file_name)
                    if match:
                        param_value = int(match.group(1))
                else:  # p invariance
                    match = re.search(r"num_timesteps_(\d+)", file_name)
                    if match:
                        param_value = int(match.group(1))

                if "runs" in data:
                    all_losses = [run["test_all_loss"] for run in data["runs"]]
                    mean = sum(all_losses) / len(all_losses)
                    variance = sum((x - mean) ** 2 for x in all_losses) / len(all_losses)
                    std = variance**0.5

                    param_values.append(param_value)
                    means.append(mean)
                    stds.append(std)

        return param_values, means, stds

    # Process EGNO results
    egno_param_values, egno_means, egno_stds = process_egno_files(egno_results_files)

    # Initialize no temporal variables
    egno_no_temporal_param_values: list[int] = []
    egno_no_temporal_means: list[float] = []
    egno_no_temporal_stds: list[float] = []

    if invariance_to_plot == "t" and egno_no_temporal_files:
        egno_no_temporal_param_values, egno_no_temporal_means, egno_no_temporal_stds = process_egno_files(egno_no_temporal_files)

    # Sort all results by parameter values
    def sort_results(param_values: list[int], means: list[float], stds: list[float]) -> tuple[list[int], list[float], list[float]]:
        sorted_indices = np.argsort(param_values)
        return ([param_values[i] for i in sorted_indices], [means[i] for i in sorted_indices], [stds[i] for i in sorted_indices])

    s2t_param_values, s2t_means, s2t_stds = sort_results(s2t_param_values, s2t_means, s2t_stds)
    egno_param_values, egno_means, egno_stds = sort_results(egno_param_values, egno_means, egno_stds)
    if invariance_to_plot == "t" and egno_no_temporal_param_values:
        egno_no_temporal_param_values, egno_no_temporal_means, egno_no_temporal_stds = sort_results(egno_no_temporal_param_values, egno_no_temporal_means, egno_no_temporal_stds)

    print(f"S2T param values: {s2t_param_values}")
    print(f"EGNO param values: {egno_param_values}")
    if invariance_to_plot == "t":
        print(f"EGNO no temporal param values: {egno_no_temporal_param_values}")

    # Scale values by 10^2 for display
    def scale_values(means: list[float], stds: list[float]) -> tuple[list[float], list[float]]:
        return [mean * 100 for mean in means], [std * 100 for std in stds]

    s2t_means_scaled, s2t_stds_scaled = scale_values(s2t_means, s2t_stds)
    egno_means_scaled, egno_stds_scaled = scale_values(egno_means, egno_stds)

    # Initialize scaled variables for no temporal case
    egno_no_temporal_means_scaled: list[float] = []
    egno_no_temporal_stds_scaled: list[float] = []

    if invariance_to_plot == "t" and egno_no_temporal_param_values:
        egno_no_temporal_means_scaled, egno_no_temporal_stds_scaled = scale_values(egno_no_temporal_means, egno_no_temporal_stds)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot the S2T mean line with circle markers
    ax.plot(s2t_param_values, s2t_means_scaled, "-o", color=blue, linewidth=2, label="ATOM", markersize=6)

    # Calculate and plot 2SD range for S2T
    s2t_upper_bound = [mean + 2 * std for mean, std in zip(s2t_means_scaled, s2t_stds_scaled)]
    s2t_lower_bound = [mean - 2 * std for mean, std in zip(s2t_means_scaled, s2t_stds_scaled)]
    ax.fill_between(s2t_param_values, s2t_lower_bound, s2t_upper_bound, color=blue, alpha=0.2)

    # Plot EGNO results if available
    if egno_param_values:
        ax.plot(egno_param_values, egno_means_scaled, "-s", color=red, linewidth=2, label="EGNO", markersize=6)
        egno_upper_bound = [mean + 2 * std for mean, std in zip(egno_means_scaled, egno_stds_scaled)]
        egno_lower_bound = [mean - 2 * std for mean, std in zip(egno_means_scaled, egno_stds_scaled)]
        ax.fill_between(egno_param_values, egno_lower_bound, egno_upper_bound, color=red, alpha=0.2)

    # Plot EGNO no temporal results if available
    if invariance_to_plot == "t" and egno_no_temporal_param_values:
        ax.plot(egno_no_temporal_param_values, egno_no_temporal_means_scaled, "-^", color=yellow, linewidth=2, label="EGNN", markersize=6)
        egno_no_temporal_upper_bound = [mean + 2 * std for mean, std in zip(egno_no_temporal_means_scaled, egno_no_temporal_stds_scaled)]
        egno_no_temporal_lower_bound = [mean - 2 * std for mean, std in zip(egno_no_temporal_means_scaled, egno_no_temporal_stds_scaled)]
        ax.fill_between(egno_no_temporal_param_values, egno_no_temporal_lower_bound, egno_no_temporal_upper_bound, color=yellow, alpha=0.2)

    # Set x-axis to log scale for t-invariance
    if invariance_to_plot == "t":
        ax.set_xscale("log")

    # Add labels and title
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel("Mean S2T MSE ($\\times 10^{-2}$)", fontsize=14)

    # Add legend
    ax.legend(loc="best", fontsize=12)

    # Ensure the directory exists
    figure_dir = Path(f"Z_paper_content/{figure_dir_name}")
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Save the figure
    plt.tight_layout()
    save_path = figure_dir / figure_file_name
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Figure saved as PDF to {save_path}")


if __name__ == "__main__":
    set_matplotlib_style(font_size=18)
    plot_learnable_attention_weights(Path("benchmark_runs/denom_aspirin_md17_22-May-2025_17-09-23/run_1"), "aspirin")
    plot_lambda_value_residuals(Path("benchmark_runs/denom_aspirin_md17_22-May-2025_17-09-23/run_1"), "aspirin")
    plot_learnable_attention_weights(Path("benchmark_runs/denom_uracil_md17_22-May-2025_17-24-37/run_1"), "uracil")
    plot_lambda_value_residuals(Path("benchmark_runs/denom_uracil_md17_22-May-2025_17-24-37/run_1"), "uracil")
    # print_ablation_results()
    # plot_invariance_results("t")  # or "p" for P-invariance (num_timesteps)
