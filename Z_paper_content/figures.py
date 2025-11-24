import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from usyd_colors import get_palette
import json
from typing import Any
import re
from typing import Literal

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
        invariance_dir = Path("benchmark_runs/t_invariance_atom")
        invariance_dir_no_trope = Path("benchmark_runs/t_invariance_atom_notrope")
        egno_dir = Path("benchmark_runs/t_invariance_egno")
        egnn_dir = Path("benchmark_runs/t_invariance_egnn")
        x_label = "$ \\Delta t $ (fs)"
        figure_file_name = "t_invariance_results.pdf"
    else:  # p invariance
        invariance_dir = Path("benchmark_runs/p_invariance_atom")
        invariance_dir_no_trope = None  # Not used for p invariance
        egno_dir = Path("benchmark_runs/p_invariance_egno_fixedlr")
        egnn_dir = None  # Not used for p invariance
        x_label = "Number of Timesteps (P)"
        figure_file_name = "p_invariance_results.pdf"

    # Get results files for all models
    s2t_results_files: list[Path] = list(invariance_dir.glob("**/results.json"))
    s2t_results_files_no_trope: list[Path] = (
        list(invariance_dir_no_trope.glob("**/results.json"))
        if invariance_dir_no_trope
        else []
    )
    egno_results_files: list[Path] = list(egno_dir.glob("*.json"))
    egnn_results_files: list[Path] = list(egnn_dir.glob("*.json")) if egnn_dir else []

    print(
        f"Found {len(s2t_results_files)} S2T results files, {len(s2t_results_files_no_trope)} S2T results files no trope, {len(egno_results_files)} EGNO results files, {len(egnn_results_files)} EGNN results files"
    )

    # Extract parameter values and corresponding metrics for S2T model (with trope)
    s2t_param_values: list[int] = []
    s2t_means: list[float] = []
    s2t_stds: list[float] = []

    # Process ATOM with trope results
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

    # Extract parameter values and corresponding metrics for S2T model (without trope)
    s2t_no_trope_param_values: list[int] = []
    s2t_no_trope_means: list[float] = []
    s2t_no_trope_stds: list[float] = []

    # Process ATOM without trope results
    for results_file in s2t_results_files_no_trope:
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
            s2t_no_trope_param_values.append(s2t_param_value)
            s2t_no_trope_means.append(s2t_data["s2t_test_loss_mean"])
            s2t_no_trope_stds.append(s2t_data["s2t_test_loss_std"])

    # Function to process EGNO results
    def process_egno_files(
        results_files: list[Path],
    ) -> tuple[list[int], list[float], list[float]]:
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
                    variance = sum((x - mean) ** 2 for x in all_losses) / len(
                        all_losses
                    )
                    std = variance**0.5

                    param_values.append(param_value)
                    means.append(mean)
                    stds.append(std)

        return param_values, means, stds

    # Process EGNO results
    egno_param_values, egno_means, egno_stds = process_egno_files(egno_results_files)

    # Process EGNN results
    egnn_param_values, egnn_means, egnn_stds = process_egno_files(egnn_results_files)

    # Sort all results by parameter values
    def sort_results(
        param_values: list[int], means: list[float], stds: list[float]
    ) -> tuple[list[int], list[float], list[float]]:
        sorted_indices = np.argsort(param_values)
        return (
            [param_values[i] for i in sorted_indices],
            [means[i] for i in sorted_indices],
            [stds[i] for i in sorted_indices],
        )

    s2t_param_values, s2t_means, s2t_stds = sort_results(
        s2t_param_values, s2t_means, s2t_stds
    )
    egno_param_values, egno_means, egno_stds = sort_results(
        egno_param_values, egno_means, egno_stds
    )
    egnn_param_values, egnn_means, egnn_stds = sort_results(
        egnn_param_values, egnn_means, egnn_stds
    )

    # Sort ATOM no trope results if available
    if s2t_no_trope_param_values:
        s2t_no_trope_param_values, s2t_no_trope_means, s2t_no_trope_stds = sort_results(
            s2t_no_trope_param_values, s2t_no_trope_means, s2t_no_trope_stds
        )

    print(f"S2T param values: {s2t_param_values}")
    print(f"EGNO param values: {egno_param_values}")
    print(f"EGNN param values: {egnn_param_values}")
    if s2t_no_trope_param_values:
        print(f"S2T no trope param values: {s2t_no_trope_param_values}")

    # Scale values by 10^2 for display
    def scale_values(
        means: list[float], stds: list[float]
    ) -> tuple[list[float], list[float]]:
        return [mean * 100 for mean in means], [std * 100 for std in stds]

    s2t_means_scaled, s2t_stds_scaled = scale_values(s2t_means, s2t_stds)
    egno_means_scaled, egno_stds_scaled = scale_values(egno_means, egno_stds)
    egnn_means_scaled, egnn_stds_scaled = scale_values(egnn_means, egnn_stds)

    # Scale ATOM no trope results if available
    s2t_no_trope_means_scaled: list[float] = []
    s2t_no_trope_stds_scaled: list[float] = []
    if s2t_no_trope_param_values:
        s2t_no_trope_means_scaled, s2t_no_trope_stds_scaled = scale_values(
            s2t_no_trope_means, s2t_no_trope_stds
        )

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot the S2T mean line with circle markers (with trope)
    ax.plot(
        s2t_param_values,
        s2t_means_scaled,
        "-o",
        color=blue,
        linewidth=2,
        label="ATOM",
        markersize=6,
    )

    # Calculate and plot 2SD range for S2T (with trope)
    s2t_upper_bound = [
        mean + 2 * std for mean, std in zip(s2t_means_scaled, s2t_stds_scaled)
    ]
    s2t_lower_bound = [
        mean - 2 * std for mean, std in zip(s2t_means_scaled, s2t_stds_scaled)
    ]
    ax.fill_between(
        s2t_param_values, s2t_lower_bound, s2t_upper_bound, color=blue, alpha=0.2
    )

    # Plot the S2T mean line with triangle markers (without trope) if available
    if s2t_no_trope_param_values:
        ax.plot(
            s2t_no_trope_param_values,
            s2t_no_trope_means_scaled,
            "-^",
            color=blue,
            linewidth=2,
            label="ATOM (No T-RoPE)",
            markersize=6,
            alpha=0.7,
        )

        # Calculate and plot 2SD range for S2T (without trope)
        s2t_no_trope_upper_bound = [
            mean + 2 * std
            for mean, std in zip(s2t_no_trope_means_scaled, s2t_no_trope_stds_scaled)
        ]
        s2t_no_trope_lower_bound = [
            mean - 2 * std
            for mean, std in zip(s2t_no_trope_means_scaled, s2t_no_trope_stds_scaled)
        ]
        ax.fill_between(
            s2t_no_trope_param_values,
            s2t_no_trope_lower_bound,
            s2t_no_trope_upper_bound,
            color=blue,
            alpha=0.1,
        )

    # Plot EGNO results if available
    if egno_param_values:
        ax.plot(
            egno_param_values,
            egno_means_scaled,
            "-s",
            color=red,
            linewidth=2,
            label="EGNO",
            markersize=6,
        )
        egno_upper_bound = [
            mean + 2 * std for mean, std in zip(egno_means_scaled, egno_stds_scaled)
        ]
        egno_lower_bound = [
            mean - 2 * std for mean, std in zip(egno_means_scaled, egno_stds_scaled)
        ]
        ax.fill_between(
            egno_param_values, egno_lower_bound, egno_upper_bound, color=red, alpha=0.2
        )

    # Plot EGNN results if available
    if egnn_param_values:
        ax.plot(
            egnn_param_values,
            egnn_means_scaled,
            "-d",
            color=yellow,
            linewidth=2,
            label="EGNN",
            markersize=6,
        )
        egnn_upper_bound = [
            mean + 2 * std for mean, std in zip(egnn_means_scaled, egnn_stds_scaled)
        ]
        egnn_lower_bound = [
            mean - 2 * std for mean, std in zip(egnn_means_scaled, egnn_stds_scaled)
        ]
        ax.fill_between(
            egnn_param_values,
            egnn_lower_bound,
            egnn_upper_bound,
            color=yellow,
            alpha=0.2,
        )

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
    plot_invariance_results("p")  # or "p" for P-invariance (num_timesteps)
