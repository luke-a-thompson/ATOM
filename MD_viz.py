import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

TIMESTEPS = 200

STARTING_NODE_ALPHA = 1.0
STARTING_EDGE_ALPHA = 0.7

ADD_GHOST = False
ENDING_NODE_ALPHA = 0.3
ENDING_EDGE_ALPHA = 0.1

# Load the MD17 uracil dataset
data = np.load("project/md17_uracil.npz")

# Get only non-hydrogen atoms
mask = data["z"] > 1
filtered_R = data["R"][:, mask, :]
filtered_z = data["z"][mask]  # Get the atomic numbers of filtered atoms

# Create figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

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


# Define a function to calculate distance between atoms
def calculate_distance(pos1, pos2):
    return np.sqrt(np.sum((pos1 - pos2) ** 2))


# Define bond distance threshold (adjust as needed for your molecule)
bond_threshold = 1.8  # Angstroms

# Add bonds between atoms for the starting frame (solid)
start_positions = filtered_R[0]  # Get positions at first timestep
for i in range(num_atoms):
    for j in range(i + 1, num_atoms):
        dist = calculate_distance(start_positions[i], start_positions[j])
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
            dist = calculate_distance(end_positions[i], end_positions[j])
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

# Set labels and title
ax.set_xlabel("X position")
ax.set_ylabel("Y position")
ax.set_zlabel("Z position")
ax.set_title(f"Start (solid) and end (ghosted) positions with trajectories over {TIMESTEPS} timesteps")

# Set axis limits
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)

# Set fixed viewing angle and position
ax.view_init(elev=30, azim=45)  # Set elevation and azimuth angles
ax.dist = 10  # Set distance from the plot

# Create custom legend with atom types and position indicators
legend_elements = []
for z_num, element in sorted(unique_atom_types):
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

# Add a legend with atom types and position indicators
ax.legend(handles=legend_elements, loc="best")

plt.tight_layout()
plt.savefig("project/trajectory.pdf")
