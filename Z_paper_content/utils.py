"""Shared utilities for paper content generation scripts."""

from pathlib import Path
from typing import Any
import json
import numpy as np


# Dataset constants
MD17_MOLECULE_ORDER: list[str] = [
    "aspirin",
    "benzene",
    "ethanol",
    "malonaldehyde",
    "naphthalene",
    "salicylic",
    "toluene",
    "uracil",
]

MD17_MOLECULE_DISPLAY: dict[str, str] = {
    "aspirin": "Aspirin",
    "benzene": "Benzene",
    "ethanol": "Ethanol",
    "malonaldehyde": "Malonaldehyde",
    "naphthalene": "Naphthalene",
    "salicylic": "Salicylic",
    "toluene": "Toluene",
    "uracil": "Uracil",
}

MD22_MOLECULE_DISPLAY: dict[str, str] = {
    "nhme": "Ac-Ala3-NHME",
    "dha": "Docosahexaenoic acid",
    "stachyose": "Stachyose",
}

MD22_MOLECULE_ORDER: list[str] = [
    "nhme",
    "dha",
    "stachyose",
]

MODEL_DISPLAY: dict[str, str] = {
    "GTNO": "\\gls{atoms}",
    "EGNO": "\\gls{egno}",
    "EGNN_S": "EGNN-S",
    "EGNN_R": "EGNN-R",
}


def find_results_files(root: Path) -> list[Path]:
    """Find all results.json files under the provided root.

    If `root` itself contains a `results.json`, include it; otherwise, include
    all immediate children that contain a `results.json`.
    """
    results_files: list[Path] = []
    if (root / "results.json").is_file():
        results_files.append(root / "results.json")
        return results_files

    # Scan one level deep for experiment folders
    for child in sorted(root.iterdir()):
        if child.is_dir():
            results_path = child / "results.json"
            if results_path.is_file():
                results_files.append(results_path)

    return results_files


def load_results_json(results_path: Path) -> dict[str, Any] | None:
    """Load a results.json file, returning None on error."""
    try:
        with results_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def canonicalize_model_type(name: str) -> str:
    """Canonicalize model type name to standard form."""
    n: str = name.strip().lower()
    if n in {"gtno", "atom", "atoms"}:
        return "GTNO"
    if n in {"egno"}:
        return "EGNO"
    if n in {"egnn_s"}:
        return "EGNN_S"
    if n in {"egnn_r"}:
        return "EGNN_R"
    return name


def canonicalize_molecule_name(molecule: str) -> str:
    """Canonicalize molecule name to lowercase."""
    return molecule.strip().lower()


def format_mean_std_latex(
    mean_value: float,
    std_value: float,
    decimals: int = 2,
    scale_multiplier: float = 1.0,
    suffix: str = "",
) -> str:
    """Format a LaTeX cell as mean±std.

    Args:
        mean_value: Mean value
        std_value: Standard deviation
        decimals: Number of decimal places for both mean and std (default 2)
        scale_multiplier: Multiply mean and std by this factor before formatting (e.g., 100.0 for 1e-2 scaling)
        suffix: Optional LaTeX text appended after the value (e.g., ``\\times 10^{-2}``). Leave empty to omit.

    Returns:
        LaTeX formatted string
    """
    if not (mean_value == mean_value) or not (std_value == std_value):
        return "-"
    mv: float = mean_value * scale_multiplier
    sv: float = std_value * scale_multiplier
    fmt: str = f"{{:.{decimals}f}}"
    base: str = f"\\({fmt.format(mv)}{{\\scriptstyle \\pm{fmt.format(sv)}}}\\)"
    return base + (suffix if suffix else "")


def compute_statistics(values: list[float]) -> tuple[float, float]:
    """Compute mean and standard deviation of values.

    Args:
        values: List of numeric values

    Returns:
        Tuple of (mean, std)
    """
    if not values:
        return float("nan"), float("nan")
    vals: list[float] = [v for v in values if v == v]  # Filter NaN
    if not vals:
        return float("nan"), float("nan")
    mean_val: float = float(np.mean(vals))
    std_val: float = float(np.std(vals)) if len(vals) > 1 else 0.0
    return mean_val, std_val


def resolve_path(path_str: str) -> Path:
    """Resolve and expand a path string."""
    return Path(path_str).expanduser().resolve()


def detect_dataset_from_path(path: Path) -> str | None:
    """Detect dataset name from a path by checking directory names.

    Returns one of: 'md17', 'md22', 'rmd17', 'tg80', or None
    """
    tokens: list[str] = ["md17", "md22", "rmd17", "tg80"]
    path_parts: list[str] = [p.lower() for p in path.parts]
    for token in tokens:
        if any(token in part for part in path_parts):
            return token
    return None

