import numpy as np
from pathlib import Path
from typing import Any


def print_file_info(filepath: Path) -> None:
    """Print detailed information about arrays in a single NPZ file."""
    data: Any = np.load(filepath)
    print(f"\nFile: {filepath.name}")
    print("Available arrays:", data.files)

    # Print shapes and info for each array
    for key in data.files:
        arr: np.ndarray = data[key]
        print(f"\n{key}:")
        print(f"Shape: {arr.shape}")
        print(f"Data type: {arr.dtype}")

        if np.issubdtype(arr.dtype, np.number):
            print(f"Min value: {arr.min()}")
            print(f"Max value: {arr.max()}")
            print(f"Mean value: {arr.mean()}")
        elif np.issubdtype(arr.dtype, np.character):
            print(f"Content: {arr}")
        else:
            print("This array contains data that cannot be processed for min, max, or mean values.")


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


def main() -> None:
    orig_data_dir: Path = Path("data/md17_npz")
    refresh_data_dir: Path = Path("data/rmd17_npz")

    # Print theory summary for all files
    print_theory_summary(orig_data_dir)

    # Print detailed info for benzene file as an example
    benzene_file: Path = orig_data_dir / "md17_benzene.npz"
    print_file_info(benzene_file)

    # Print detailed info for benzene file as an example
    benzene_file: Path = refresh_data_dir / "rmd17_benzene.npz"
    print_file_info(benzene_file)


if __name__ == "__main__":
    main()
