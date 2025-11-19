import numpy as np
import os
import re
from numpy.typing import NDArray


def read_xyz_file(file_path: str) -> tuple[NDArray[np.float64], list[str], NDArray[np.float64]]:
    """
    Read a multi-frame XYZ file and return coordinates, atomic symbols, and energies.

    Returns:
        tuple containing:
        - coords: np.ndarray of shape (nframes, natoms, 3)
        - symbols: list[str] of atomic symbols
        - energies: np.ndarray of shape (nframes,)
    """
    all_coords: list[list[list[float]]] = []
    energies: list[float] = []
    symbols: list[str] | None = None  # We'll get this from the first frame

    with open(file_path, "r") as f:
        while True:
            # Read number of atoms
            natoms_line = f.readline()
            if not natoms_line:  # End of file
                break

            natoms = int(natoms_line)

            # Read comment line and extract energy
            comment = f.readline()
            energy_match = re.search(r"E_Pot=([-\d.]+)", comment)
            if energy_match:
                energy = float(energy_match.group(1))
                energies.append(energy)

            # Read coordinates
            frame_coords: list[list[float]] = []
            frame_symbols: list[str] = []
            for _ in range(natoms):
                line = f.readline()
                if not line:
                    break
                parts = line.split()
                if len(parts) == 4:
                    symbol, x, y, z = parts
                    frame_symbols.append(symbol)
                    frame_coords.append([float(x), float(y), float(z)])

            if symbols is None:  # Store symbols from first frame
                symbols = frame_symbols

            all_coords.append(frame_coords)

    if symbols is None:
        raise ValueError("No valid frames found in XYZ file")

    # Convert to numpy arrays
    coords = np.array(all_coords)  # Shape: (nframes, natoms, 3)
    energies_array = np.array(energies)

    return coords, symbols, energies_array


def process_tg80_file(input_file: str, output_dir: str, molecule_name: str) -> None:
    """
    Process a TG80 XYZ file and save it as NPZ format.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the XYZ file
    coords, symbols, energies = read_xyz_file(input_file)

    # Create output filename
    output_file = os.path.join(output_dir, f"tg80_{molecule_name.lower()}.npz")

    # Convert symbols to atomic numbers
    nuclear_charges = np.array([atomic_number(s) for s in symbols])

    # Create placeholder forces array
    nframes, natoms, _ = coords.shape
    forces = np.zeros((nframes, natoms, 3))

    # Save as NPZ with new column names
    np.savez(
        output_file,
        coords=coords,  # Atomic coordinates (nframes, natoms, 3)
        nuclear_charges=nuclear_charges,  # Atomic numbers (natoms,)
        energy=energies,  # Energies (nframes,)
        forces=forces,  # Forces (nframes, natoms, 3)
    )
    print(f"Processed {nframes} frames")
    print(f"Output saved to {output_file}")


def atomic_number(symbol: str) -> int:
    """
    Convert atomic symbol to atomic number.
    """
    atomic_numbers = {"H": 1, "C": 6, "N": 7, "O": 8}
    return atomic_numbers.get(symbol, 0)


def process_all_molecules(data_raw_dir: str, output_dir: str) -> None:
    """
    Process all molecule folders in data_raw directory.

    Args:
        data_raw_dir: Path to the data_raw directory containing molecule folders
        output_dir: Path to save the processed NPZ files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all molecule folders
    molecule_folders = []
    for directory in os.listdir(data_raw_dir):
        folder_path = os.path.join(data_raw_dir, directory)
        # Skip if not a directory or matches exclusion criteria
        if not os.path.isdir(folder_path):
            continue
        if directory.startswith("."):
            continue
        if directory.endswith(".zip"):
            continue
        if directory == "zips":
            continue

        molecule_folders.append(directory)

    for folder in molecule_folders:
        # Extract molecule name from folder (remove _lowest suffix)
        molecule_name = str(folder.replace("_lowest", ""))

        # Path to the trajectory.xyz file
        xyz_file = os.path.join(data_raw_dir, folder, "trajectory.xyz")

        if os.path.exists(xyz_file):
            print(f"Processing {molecule_name}...")
            process_tg80_file(xyz_file, output_dir, molecule_name)
        else:
            print(f"Warning: No trajectory.xyz found in {folder}")


if __name__ == "__main__":
    data_raw_dir = "data_raw"
    output_dir = "data/tg80_npz"
    process_all_molecules(data_raw_dir, output_dir)
