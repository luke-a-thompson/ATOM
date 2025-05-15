import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from typing import Literal
from pathlib import Path
from figures import set_matplotlib_style

TIMESTEPS = 200

STARTING_NODE_ALPHA = 1.0
STARTING_EDGE_ALPHA = 0.7

ADD_GHOST = False
ENDING_NODE_ALPHA = 0.3
ENDING_EDGE_ALPHA = 0.1

# Load the MD17 uracil dataset


def plot_trajectory(ax: plt.Axes, filename: Path, md_17_version: Literal["md17", "rmd17", "tg80"]) -> set[tuple[int, str]]:
    data: dict[str, npt.NDArray[np.number]] = np.load(filename)
    # Get only non-hydrogen atoms
    if md_17_version == "md17":
        mask: npt.NDArray[np.bool_] = data["z"] > 1
        filtered_R: npt.NDArray[np.float64] = data["R"][:, mask, :]
        filtered_z: npt.NDArray[np.int_] = data["z"][mask]  # Get the atomic numbers of filtered atoms
    elif md_17_version == "rmd17" or md_17_version == "tg80":
        mask: npt.NDArray[np.bool_] = data["nuclear_charges"] > 1
        filtered_R: npt.NDArray[np.float64] = data["coords"][:, mask, :]
        filtered_z: npt.NDArray[np.int_] = data["nuclear_charges"][mask]  # Get the atomic numbers of filtered atoms

    # Get number of non-hydrogen atoms
    num_atoms = filtered_R.shape[1]

    # Dictionary to map atomic numbers to element names
    element_map = {6: "C", 7: "N", 8: "O", 9: "F", 16: "S"}

    # Dictionary to map atomic numbers to colors
    color_map = {
        6: "gray",  # Carbon - gray
        7: "blue",  # Nitrogen - blue
        8: "red",  # Oxygen - red
        9: "green",  # Fluorine - green
        16: "yellow",  # Sulfur - yellow
    }

    # Keep track of unique atom types we've seen
    unique_atom_types = set()

    # Plot trajectory for each atom
    for atom_idx in range(num_atoms):
        # Get x, y, z coordinates for this atom over time
        x = filtered_R[:TIMESTEPS, atom_idx, 0]
        y = filtered_R[:TIMESTEPS, atom_idx, 1]
        z = filtered_R[:TIMESTEPS, atom_idx, 2]

        # Get element name or atomic number
        z_num = filtered_z[atom_idx]
        element = element_map.get(z_num, str(int(z_num)))

        # Add to set of unique atom types
        unique_atom_types.add((z_num, element))

        # Get color for this atom type
        atom_color = color_map.get(z_num, "purple")  # Default to purple for unknown elements

        # Plot the trajectory line (path from start to current position)
        ax.plot(x, y, z, color=atom_color, alpha=0.4, linewidth=1)

        # Mark the starting position with a solid marker
        ax.scatter(x[0], y[0], z[0], color=atom_color, s=80, edgecolor="black", alpha=STARTING_NODE_ALPHA)

        # Mark the ending position with a ghosted (transparent) marker
        if ADD_GHOST:
            ax.scatter(x[-1], y[-1], z[-1], color=atom_color, s=80, edgecolor="black", alpha=ENDING_NODE_ALPHA)

    # Define bond distance threshold (adjust as needed for your molecule)
    bond_threshold = 1.8  # Angstroms

    # Add bonds between atoms for the starting frame (solid)
    start_positions = filtered_R[0]  # Get positions at first timestep
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = np.linalg.norm(start_positions[i] - start_positions[j])
            if dist < bond_threshold:
                # Draw a line between bonded atoms
                ax.plot(
                    [start_positions[i, 0], start_positions[j, 0]],
                    [start_positions[i, 1], start_positions[j, 1]],
                    [start_positions[i, 2], start_positions[j, 2]],
                    "k-",
                    alpha=STARTING_EDGE_ALPHA,
                    linewidth=1.5,
                )

    if ADD_GHOST:
        # Add bonds between atoms for the ending frame (ghosted)
        end_positions = filtered_R[TIMESTEPS - 1]  # Get positions at last timestep
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                dist: float = np.linalg.norm(end_positions[i] - end_positions[j])
                if dist < bond_threshold:
                    # Draw a ghosted line between bonded atoms
                    ax.plot(
                        [end_positions[i, 0], end_positions[j, 0]],
                        [end_positions[i, 1], end_positions[j, 1]],
                        [end_positions[i, 2], end_positions[j, 2]],
                        "k-",
                        alpha=ENDING_EDGE_ALPHA,
                        linewidth=1.5,
                    )

    # Set labels (removed title)
    # ax.set_xlabel("X position")
    # ax.set_ylabel("Y position")
    # ax.set_zlabel("Z position")

    # Set axis limits
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)

    # Set fixed viewing angle and position
    ax.view_init(elev=30, azim=45)  # Set elevation and azimuth angles
    ax.dist = 10  # Set distance from the plot

    # Instead of creating and adding the legend here, return the unique_atom_types
    return unique_atom_types


def create_tiled_figure(data_dir: Path, md_17_version: Literal["md17", "rmd17", "tg80"], n_cols: int | None = None, n_rows: int | None = None) -> None:
    # Get all NPZ files
    files: list[Path] = sorted(list(data_dir.glob("*.npz")))
    n_files: int = len(files)

    # For tg80, use 6x4 grid per figure
    if md_17_version == "tg80":
        n_cols = 4
        n_rows = 6
        plots_per_figure = n_cols * n_rows
        n_figures = (n_files + plots_per_figure - 1) // plots_per_figure  # Ceiling division
    # Calculate grid dimensions for other datasets
    elif n_cols is None or n_rows is None:
        n_cols = int(np.ceil(np.sqrt(n_files)))
        n_rows = int(np.ceil(n_files / n_cols))
        plots_per_figure = n_cols * n_rows
        n_figures = 1
    else:
        # Ensure we have enough space for all files
        while n_cols * n_rows < n_files:
            n_rows += 1
        plots_per_figure = n_cols * n_rows
        n_figures = 1

    # Keep track of all unique atom types across all plots
    all_atom_types: set[tuple[int, str]] = set()

    # Process files in batches for each figure
    for fig_idx in range(n_figures):
        start_idx = fig_idx * plots_per_figure
        end_idx = min((fig_idx + 1) * plots_per_figure, n_files)
        current_files = files[start_idx:end_idx]

        # For the last figure, crop rows if not full (tg80 only)
        if md_17_version == "tg80" and fig_idx == n_figures - 1:
            n_rows_this_fig = int(np.ceil(len(current_files) / n_cols))
        else:
            n_rows_this_fig = n_rows

        # Create a figure for this batch
        fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows_this_fig))

        # Plot each trajectory in its own subplot
        for idx, file in enumerate(current_files):
            ax = fig.add_subplot(n_rows_this_fig, n_cols, idx + 1, projection="3d")
            unique_atoms = plot_trajectory(ax, file, md_17_version)
            all_atom_types.update(unique_atoms)
            molecule_name: str = file.stem.strip(f"{md_17_version}_").title()  # Capitalize molecule name
            # Move title below plot and make it larger
            ax.set_title(f"{molecule_name}", pad=-15, y=-0.1, fontsize=18)

        # Create the common legend
        color_map = {
            6: "gray",  # Carbon - gray
            7: "blue",  # Nitrogen - blue
            8: "red",  # Oxygen - red
            9: "green",  # Fluorine - green
            16: "yellow",  # Sulfur - yellow
        }

        legend_elements: list[Line2D] = []
        for z_num, element in sorted(all_atom_types):
            atom_color = color_map.get(z_num, "purple")
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=atom_color,
                    markeredgecolor="black",
                    markersize=10,
                    label=f"{element} (Z={int(z_num)})",
                )
            )

        # Add a single legend at the bottom
        _ = fig.legend(
            handles=legend_elements,
            loc="center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=len(legend_elements),
            frameon=True,
            fancybox=True,
        )

        # Adjust the layout with much tighter horizontal spacing
        plt.tight_layout(h_pad=0.5, w_pad=0.05)  # Much less horizontal and vertical padding between subplots
        # Add just enough space at the bottom for the legend and reduce horizontal spacing
        plt.subplots_adjust(bottom=0.08, wspace=0.05)  # Reduced bottom margin and horizontal spacing

        # Save each figure with a unique name
        suffix = f"_{fig_idx + 1}" if n_figures > 1 else ""
        plt.savefig(f"Z_paper_content/trajectories/{md_17_version}_combined_trajectories{suffix}.pdf", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    set_matplotlib_style()
    md17_dir: Path = Path("data/md17_npz")
    rmd17_dir: Path = Path("data/rmd17_npz")
    tg80_dir: Path = Path("data/tg80_npz")

    create_tiled_figure(md17_dir, "md17", 2, 4)
    create_tiled_figure(tg80_dir, "tg80", 2, 4)
    create_tiled_figure(rmd17_dir, "rmd17", 2, 4)
