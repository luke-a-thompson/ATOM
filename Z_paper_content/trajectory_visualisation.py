import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from typing import Literal
from pathlib import Path
from Z_paper_content.figures import set_matplotlib_style

TIMESTEPS = 200

STARTING_NODE_ALPHA = 1.0
STARTING_EDGE_ALPHA = 0.7

ADD_GHOST = False
ENDING_NODE_ALPHA = 0.3
ENDING_EDGE_ALPHA = 0.1


def _compute_even_frames(start_t: int, delta_t: int, P: int) -> list[int]:
    """Compute P+1 evenly spaced frame indices from start_t to start_t+delta_t inclusive.

    Rounds to nearest integer and removes duplicates while preserving order.
    """
    if P <= 0:
        return [start_t]
    grid = np.linspace(float(start_t), float(start_t + delta_t), P + 1)
    rounded = [int(round(x)) for x in grid]
    # Deduplicate preserving order
    seen: set[int] = set()
    unique: list[int] = []
    for t in rounded:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def _compute_principal_axes(positions: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return principal axes (3x3, columns are eigenvectors) from centered positions [N,3]."""
    pos64: npt.NDArray[np.float64] = positions.astype(np.float64, copy=False)
    cov: npt.NDArray[np.float64] = ((pos64.T @ pos64) / max(len(pos64), 1)).astype(np.float64, copy=False)
    evals, evecs = np.linalg.eigh(cov)
    order: npt.NDArray[np.int_] = np.argsort(evals)[::-1]
    evecs_sorted: npt.NDArray[np.float64] = evecs[:, order].astype(np.float64, copy=False)
    return evecs_sorted


def _resolve_axis_signs(evecs: npt.NDArray[np.float64], positions: npt.NDArray[np.float64], atomic_numbers: npt.NDArray[np.int_]) -> npt.NDArray[np.float64]:
    """Resolve eigenvector sign ambiguity deterministically using geometry and atom identity.

    - e1 sign: align with net direction of the structure (sum of coordinates).
    - e2 sign: align with vector to highest-Z atom from centroid.
    Ensures right-handed frame with e3 = e1 x e2.
    """
    positions64: npt.NDArray[np.float64] = positions.astype(np.float64, copy=False)
    evecs64: npt.NDArray[np.float64] = evecs.astype(np.float64, copy=False)
    centroid: npt.NDArray[np.float64] = positions64.mean(axis=0)
    centered: npt.NDArray[np.float64] = positions64 - centroid

    e1: npt.NDArray[np.float64] = evecs64[:, 0]
    e2: npt.NDArray[np.float64] = evecs64[:, 1]

    net_dir: npt.NDArray[np.float64] = centered.sum(axis=0)
    if float(e1 @ net_dir) < 0.0:
        e1 = -e1

    max_z_idx: int = int(np.argmax(atomic_numbers))
    anchor_vec: npt.NDArray[np.float64] = centered[max_z_idx]
    if float(e2 @ anchor_vec) < 0.0:
        e2 = -e2

    e3: npt.NDArray[np.float64] = np.cross(e1, e2).astype(np.float64, copy=False)
    # Re-orthonormalize e2 against e1 to guard numerical drift, and recompute e3
    e1 = e1 / (np.linalg.norm(e1) + 1e-12)
    e2 = e2 - (e2 @ e1) * e1
    e2 = e2 / (np.linalg.norm(e2) + 1e-12)
    e3 = np.cross(e1, e2).astype(np.float64, copy=False)
    e3 = e3 / (np.linalg.norm(e3) + 1e-12)

    Q: npt.NDArray[np.float64] = np.stack([e1, e2, e3], axis=1).astype(np.float64, copy=False)
    return Q


def _canonicalize_coordinates(R: npt.NDArray[np.float64], atomic_numbers: npt.NDArray[np.int_]) -> npt.NDArray[np.float64]:
    """Center at t=0 centroid and rotate all frames into a canonical PCA frame.

    Args:
        R: array of shape [T, N, 3]
        atomic_numbers: array of shape [N]

    Returns:
        Rotated coordinates of shape [T, N, 3].
    """
    R64: npt.NDArray[np.float64] = R.astype(np.float64, copy=False)
    start_positions: npt.NDArray[np.float64] = R64[0]
    c0: npt.NDArray[np.float64] = start_positions.mean(axis=0)
    centered0: npt.NDArray[np.float64] = start_positions - c0
    evecs: npt.NDArray[np.float64] = _compute_principal_axes(centered0)
    Q: npt.NDArray[np.float64] = _resolve_axis_signs(evecs, start_positions, atomic_numbers)
    centered_all: npt.NDArray[np.float64] = R64 - c0[None, None, :]
    rotated: npt.NDArray[np.float64] = (centered_all @ Q).astype(np.float64, copy=False)
    return rotated


# Load the MD17 uracil dataset


def plot_trajectory(ax: Axes3D, filename: Path, md_17_version: Literal["md17", "rmd17", "tg80"]) -> set[tuple[int, str]]:
    data = np.load(filename, allow_pickle=False)
    # Get only non-hydrogen atoms
    all_atomic_numbers: npt.NDArray[np.int_]
    all_coords: npt.NDArray[np.float64]
    if md_17_version == "md17":
        all_atomic_numbers = data["z"].astype(np.int_, copy=False)
        all_coords = data["R"].astype(np.float64, copy=False)
    else:
        all_atomic_numbers = data["nuclear_charges"].astype(np.int_, copy=False)
        all_coords = data["coords"].astype(np.float64, copy=False)

    atom_mask: npt.NDArray[np.bool_] = all_atomic_numbers > 1
    filtered_R: npt.NDArray[np.float64] = all_coords[:, atom_mask, :]
    filtered_z: npt.NDArray[np.int_] = all_atomic_numbers[atom_mask]

    # Canonicalize orientation using first frame PCA and deterministic sign resolution
    filtered_R = _canonicalize_coordinates(filtered_R, filtered_z)

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
                dist = np.linalg.norm(end_positions[i] - end_positions[j])
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

    # Set axis limits: default ±2.5, expand symmetrically if exceeded (by up to +2)
    default_limit: float = 2.5
    max_abs_extent: float = float(np.max(np.abs(filtered_R[:TIMESTEPS])))
    if max_abs_extent <= default_limit:
        limit: float = default_limit
    else:
        # Increase by ~1 if exceeded, but cap extra margin to 2.0 overall
        expanded: float = float(np.ceil(max_abs_extent + 1.0))
        limit = min(default_limit + 2.0, expanded)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)

    # Set fixed viewing angle
    ax.view_init(elev=30, azim=45)

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
            bbox_to_anchor=(0.5, -0.05),
            ncol=len(legend_elements),
            frameon=True,
            fancybox=True,
        )

        # Adjust the layout with much tighter horizontal spacing
        plt.tight_layout(h_pad=0.5, w_pad=0.05)
        plt.subplots_adjust(bottom=0.15)

        # Save each figure with a unique name
        suffix = f"_{fig_idx + 1}" if n_figures > 1 else ""
        plt.savefig(f"Z_paper_content/trajectories/{md_17_version}_combined_trajectories{suffix}.pdf", bbox_inches="tight")
        plt.close()


def save_single_trajectory_png(
    filename: Path,
    md_17_version: Literal["md17", "rmd17", "tg80"],
    out_path: Path,
    frame_index: int,
    dpi: int = 600,
    show_bonds: bool = True,
) -> None:
    """Render a single frame as a high-resolution PNG with no axes/background.

    Args:
        filename: Path to the .npz file to visualize.
        md_17_version: Which dataset schema the file conforms to.
        out_path: Destination PNG path.
        frame_index: Index of the frame (t) to render.
        dpi: PNG resolution in dots-per-inch.
        show_bonds: Whether to draw bonds for the selected frame.
    """
    # Load data
    data = np.load(filename, allow_pickle=False)
    if md_17_version == "md17":
        all_atomic_numbers: npt.NDArray[np.int_] = data["z"].astype(np.int_, copy=False)
        all_coords: npt.NDArray[np.float64] = data["R"].astype(np.float64, copy=False)
    else:
        all_atomic_numbers = data["nuclear_charges"].astype(np.int_, copy=False)
        all_coords = data["coords"].astype(np.float64, copy=False)

    # Non-hydrogen mask and canonicalization
    atom_mask: npt.NDArray[np.bool_] = all_atomic_numbers > 1
    filtered_R: npt.NDArray[np.float64] = all_coords[:, atom_mask, :]
    filtered_z: npt.NDArray[np.int_] = all_atomic_numbers[atom_mask]
    filtered_R = _canonicalize_coordinates(filtered_R, filtered_z)

    n_frames: int = int(filtered_R.shape[0])
    if frame_index < 0 or frame_index >= n_frames:
        print(f"Warning: frame_index {frame_index} out of range [0, {n_frames - 1}] for {filename}")
        return

    frame_positions: npt.NDArray[np.float64] = filtered_R[frame_index]

    # Color map
    color_map: dict[int, str] = {6: "gray", 7: "blue", 8: "red", 9: "green", 16: "yellow"}

    # Create figure
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Plot atoms for the single frame
    for atom_idx in range(frame_positions.shape[0]):
        x, y, z = frame_positions[atom_idx]
        z_num = int(filtered_z[atom_idx])
        atom_color = color_map.get(z_num, "purple")
        _ = ax.scatter(x, y, z, color=atom_color, s=80, edgecolor="black", alpha=1.0)

    # Optional bonds
    if show_bonds:
        bond_threshold: float = 1.8
        num_atoms: int = frame_positions.shape[0]
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                if float(np.linalg.norm(frame_positions[i] - frame_positions[j])) < bond_threshold:
                    _ = ax.plot(
                        [frame_positions[i, 0], frame_positions[j, 0]],
                        [frame_positions[i, 1], frame_positions[j, 1]],
                        [frame_positions[i, 2], frame_positions[j, 2]],
                        "k-",
                        alpha=0.9,
                        linewidth=1.5,
                    )

    # Tight zoom around the selected frame
    mins: npt.NDArray[np.float64] = frame_positions.min(axis=0)
    maxs: npt.NDArray[np.float64] = frame_positions.max(axis=0)
    center: npt.NDArray[np.float64] = (mins + maxs) / 2.0
    half_ranges: npt.NDArray[np.float64] = (maxs - mins) / 2.0
    radius: float = float(np.max(half_ranges))
    radius = max(radius, 1e-3) * 1.01
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass

    # Hide axes and background
    ax.set_axis_off()
    axis_candidates = [
        getattr(ax, "xaxis", None),
        getattr(ax, "yaxis", None),
        getattr(ax, "zaxis", None),
        getattr(ax, "w_xaxis", None),
        getattr(ax, "w_yaxis", None),
        getattr(ax, "w_zaxis", None),
    ]
    for axis in axis_candidates:
        if axis is None:
            continue
        set_pane = getattr(axis, "set_pane_color", None)
        if callable(set_pane):
            try:
                set_pane((1.0, 1.0, 1.0, 0.0))
            except Exception:
                pass
        line_obj = getattr(axis, "line", None)
        if line_obj is not None:
            set_color = getattr(line_obj, "set_color", None)
            if callable(set_color):
                try:
                    set_color((1.0, 1.0, 1.0, 0.0))
                except Exception:
                    pass
        set_ticks = getattr(axis, "set_ticks", None)
        if callable(set_ticks):
            try:
                set_ticks([])
            except Exception:
                pass
    ax.patch.set_alpha(0.0)
    fig.patch.set_alpha(0.0)

    # Fill canvas
    try:
        ax.set_position((0.0, 0.0, 1.0, 1.0))
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    except Exception:
        pass
    plt.tight_layout(pad=0.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=dpi, bbox_inches="tight", pad_inches=0.0, transparent=True)
    plt.close(fig)


def save_tg80_molecule_pngs(
    data_dir: Path,
    molecule_names: list[str],
    out_dir: Path,
    dpi: int = 600,
    default_frame_index: int = 3000,
    extra_frames: dict[str, list[int]] | None = None,
) -> None:
    """Save high-resolution PNGs for selected TG80 molecules.

    Args:
        data_dir: Directory containing tg80_*.npz files.
        molecule_names: Names like "Aspirin_lowest"; "_lowest" will be stripped.
        out_dir: Directory to write PNGs into.
        dpi: PNG resolution.
        default_frame_index: Default frame index for the primary PNG per molecule.
        extra_frames: Optional mapping from base molecule name to extra frame indices.
    """
    for raw_name in molecule_names:
        base: str = raw_name.replace("_lowest", "").lower()
        npz_path: Path = data_dir / f"tg80_{base}.npz"
        png_path: Path = out_dir / f"tg80_{base}.png"
        if npz_path.exists():
            save_single_trajectory_png(
                npz_path,
                "tg80",
                png_path,
                frame_index=default_frame_index,
                dpi=dpi,
                show_bonds=True,
            )
            if extra_frames is not None and base in extra_frames:
                for t in extra_frames[base]:
                    extra_png: Path = out_dir / f"tg80_{base}_t{int(t)}.png"
                    save_single_trajectory_png(
                        npz_path,
                        "tg80",
                        extra_png,
                        frame_index=int(t),
                        dpi=dpi,
                        show_bonds=True,
                    )
        else:
            print(f"Warning: file not found {npz_path}")


def create_uracil_comparison() -> None:
    """Create a 1x3 visualization of uracil from each dataset."""
    # Create figure with 1 row and 3 columns
    fig = plt.figure(figsize=(15, 5))

    # Define the directories and versions
    dirs_and_versions: list[tuple[Path, Literal["md17", "rmd17", "tg80"]]] = [(Path("data/md17_npz"), "md17"), (Path("data/rmd17_npz"), "rmd17"), (Path("data/tg80_npz"), "tg80")]

    # Plot uracil from each dataset
    unique_atoms: set[tuple[int, str]] = set()
    for idx, (data_dir, version) in enumerate(dirs_and_versions):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        file = data_dir / f"{version}_uracil.npz"
        atoms_here = plot_trajectory(ax, file, version)
        unique_atoms.update(atoms_here)

        # Add title
        ax.set_title(f"{version.upper()} Uracil", pad=-15, y=-0.1, fontsize=18)

    # Create the common legend
    color_map = {
        6: "gray",  # Carbon - gray
        7: "blue",  # Nitrogen - blue
        8: "red",  # Oxygen - red
        9: "green",  # Fluorine - green
        16: "yellow",  # Sulfur - yellow
    }

    legend_elements: list[Line2D] = []
    for z_num, element in sorted(unique_atoms):
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
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(legend_elements),
        frameon=True,
        fancybox=True,
    )

    # Adjust the layout
    plt.tight_layout(h_pad=0.5, w_pad=0.05)
    plt.subplots_adjust(bottom=0.15)

    # Save the figure
    plt.savefig("Z_paper_content/trajectories/uracil_comparison.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    set_matplotlib_style()
    md17_dir: Path = Path("data/md17_npz")
    rmd17_dir: Path = Path("data/rmd17_npz")
    tg80_dir: Path = Path("data/tg80_npz")

    create_tiled_figure(md17_dir, "md17", 2, 4)
    create_tiled_figure(tg80_dir, "tg80", 2, 4)
    create_tiled_figure(rmd17_dir, "rmd17", 2, 4)

    # Create uracil comparison
    create_uracil_comparison()

    # Save high-res PNGs for selected TG80 molecules
    png_out_dir: Path = Path("Z_paper_content/trajectories/pngs")
    molecules_to_save: list[str] = ["Aspirin_lowest", "Isoquinoline_lowest", "Succinicacid_lowest", "Pyrimidine_lowest", "Propylene_lowest"]
    # Evenly spaced aspirin frames from t0 over delta_t using P segments
    aspirin_start_t: int = 6000
    delta_t: int = 10000
    P: int = 8
    aspirin_frames: list[int] = _compute_even_frames(aspirin_start_t, delta_t, P)
    aspirin_extras: dict[str, list[int]] = {"aspirin": aspirin_frames}
    save_tg80_molecule_pngs(
        tg80_dir,
        molecules_to_save,
        png_out_dir,
        dpi=600,
        default_frame_index=3000,
        extra_frames=aspirin_extras,
    )
