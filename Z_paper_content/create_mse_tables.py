import argparse
import json
import re
from pathlib import Path
from dataclasses import dataclass


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


@dataclass
class ExperimentResult:
    model_type: str
    molecule: str
    latex_s2s: str
    latex_s2t: str
    s2s_mean: float
    s2t_mean: float
    time_lag_mode: str


def _format_mean_std_latex(
    mean_value: float,
    std_value: float,
    decimals: int = 2,
    scale_multiplier: float = 1.0,
    suffix: str = "",
) -> str:
    """Format a LaTeX cell as mean±std.

    - decimals: number of decimal places for both mean and std (default 2)
    - scale_multiplier: multiply mean and std by this factor before formatting (e.g., 100.0 for 1e-2 scaling)
    - suffix: optional LaTeX text appended after the value (e.g., ``\\times 10^{-2}``). Leave empty to omit.
    """
    if not (mean_value == mean_value) or not (std_value == std_value):
        return "-"
    mv: float = mean_value * scale_multiplier
    sv: float = std_value * scale_multiplier
    fmt: str = f"{{:.{decimals}f}}"
    base: str = f"\\({fmt.format(mv)}{{\\scriptstyle \\pm{fmt.format(sv)}}}\\)"
    return base + (suffix if suffix else "")


def _aggregate_folds(results: list[ExperimentResult]) -> list[ExperimentResult]:
    """Aggregate multiple folds per (mode, model, molecule).

    - Mean across folds for ``s2s_mean`` and ``s2t_mean``
    - ``latex_*`` generated as ``mean±std``
    """
    from collections import defaultdict

    def mean(values: list[float]) -> float:
        vals: list[float] = [v for v in values if v == v]
        if len(vals) == 0:
            return float("nan")
        return sum(vals) / float(len(vals))

    def std(values: list[float], m: float) -> float:
        vals: list[float] = [v for v in values if v == v]
        n: int = len(vals)
        if n <= 1:
            return 0.0
        var: float = sum((v - m) * (v - m) for v in vals) / float(n - 1)
        return var**0.5

    grouped: dict[tuple[str, str, str], list[ExperimentResult]] = defaultdict(list)
    all_molecules: list[str] = []
    for r in results:
        key: tuple[str, str, str] = (
            r.time_lag_mode,
            _canonicalize_model_type(r.model_type),
            r.molecule,
        )
        grouped[key].append(r)
        all_molecules.append(r.molecule)

    is_multitask_folds: bool = len(all_molecules) > 0 and all(m.startswith("fold") and m[4:].isdigit() for m in all_molecules)

    aggregated: list[ExperimentResult] = []
    for (mode_key, model_key, molecule), bucket in grouped.items():
        # If there is only a single results.json for this (mode, model, molecule),
        # preserve its precomputed latex (which already encodes mean±std, scaled by 1e2 in saving code).
        if len(bucket) == 1:
            single: ExperimentResult = bucket[0]
            s2s_m: float = single.s2s_mean
            s2t_m: float = single.s2t_mean
            latex_s2s: str = single.latex_s2s
            latex_s2t: str = single.latex_s2t
        else:
            s2s_vals: list[float] = [b.s2s_mean for b in bucket]
            s2t_vals: list[float] = [b.s2t_mean for b in bucket]

            s2s_m: float = mean(s2s_vals)
            s2t_m: float = mean(s2t_vals)
            s2s_s: float = std(s2s_vals, s2s_m)
            s2t_s: float = std(s2t_vals, s2t_m)

            # For multitask folds, scale values by 1e2 but do NOT append any suffix; caption will note ×10^{-2}
            scale: float = 100.0 if is_multitask_folds else 1.0
            latex_s2s: str = _format_mean_std_latex(s2s_m, s2s_s, decimals=2, scale_multiplier=scale, suffix="")
            latex_s2t: str = _format_mean_std_latex(s2t_m, s2t_s, decimals=2, scale_multiplier=scale, suffix="")

        aggregated.append(
            ExperimentResult(
                model_type=model_key,
                molecule=molecule,
                latex_s2s=latex_s2s,
                latex_s2t=latex_s2t,
                s2s_mean=s2s_m,
                s2t_mean=s2t_m,
                time_lag_mode=mode_key,
            )
        )

    return aggregated


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


def load_experiment_result(results_path: Path) -> ExperimentResult | None:
    """Load a single ExperimentResult from a results.json file, or None if invalid."""
    try:
        with results_path.open("r", encoding="utf-8") as f:
            obj: object = json.load(f)
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None

    data: dict[str, object] = obj

    try:
        tmp_config: object = data.get("config", {})
        if isinstance(tmp_config, dict):
            config: dict[str, object] = tmp_config
        else:
            config = {}

        tmp_bench: object = config.get("benchmark", {})
        if isinstance(tmp_bench, dict):
            benchmark_cfg: dict[str, object] = tmp_bench
        else:
            benchmark_cfg = {}

        tmp_loader: object = config.get("dataloader", {})
        if isinstance(tmp_loader, dict):
            dataloader_cfg: dict[str, object] = tmp_loader
        else:
            dataloader_cfg = {}

        model_type: str = str(benchmark_cfg.get("model_type", "UNKNOWN")).strip()
        molecule: str = str(dataloader_cfg.get("molecule_type", "unknown")).strip().lower()

        # If directory name encodes a fold (e.g., "..._fold3_...") treat the fold as the column key
        # so that multitask runs render each fold as its own column.
        parent_name: str = results_path.parent.name.lower()
        fold_match = re.search(r"fold(\d+)", parent_name)
        if fold_match is not None:
            fold_idx: int = int(fold_match.group(1))
            molecule = f"fold{fold_idx}"

        s2s_latex_obj: object = data.get("latex_s2s", "-")
        latex_s2s: str = s2s_latex_obj if isinstance(s2s_latex_obj, str) else "-"
        s2t_latex_obj: object = data.get("latex_s2t", "-")
        latex_s2t: str = s2t_latex_obj if isinstance(s2t_latex_obj, str) else "-"

        s2s_mean_obj: object = data.get("s2s_test_loss_mean", float("nan"))
        if isinstance(s2s_mean_obj, (int, float)):
            s2s_mean: float = float(s2s_mean_obj)
        else:
            s2s_mean = float("nan")

        s2t_mean_obj: object = data.get("s2t_test_loss_mean", float("nan"))
        if isinstance(s2t_mean_obj, (int, float)):
            s2t_mean: float = float(s2t_mean_obj)
        else:
            s2t_mean = float("nan")

        # Extract time lag mode from config (e.g., "uniform" or "last")
        tlm_obj: object = dataloader_cfg.get("time_lag_mode", "last")
        time_lag_mode: str = str(tlm_obj).strip().lower()

        return ExperimentResult(
            model_type=model_type,
            molecule=molecule,
            latex_s2s=latex_s2s,
            latex_s2t=latex_s2t,
            s2s_mean=s2s_mean,
            s2t_mean=s2t_mean,
            time_lag_mode=time_lag_mode,
        )
    except Exception:
        return None


def group_results_by_model(
    results: list[ExperimentResult],
) -> dict[str, dict[str, ExperimentResult]]:
    """Group results by model -> molecule -> {s2s, s2t}.

    Returns a nested dict: {model_type: {molecule: ExperimentResult}}
    """
    grouped: dict[str, dict[str, ExperimentResult]] = {}
    for r in results:
        canonical_model: str = _canonicalize_model_type(r.model_type)
        model_group: dict[str, ExperimentResult] = grouped.setdefault(canonical_model, {})
        # If duplicates for the same molecule/model exist, keep the latest by mtime
        model_group[r.molecule] = r
    return grouped


def _bold_latex_value(latex_value: str) -> str:
    """Bold the main numeric value inside a latex cell like "\\(X{\\scriptstyle \\pmY}\\)".

    If it already contains \\mathbf, returns unchanged. If value is '-', returns as is.
    """
    if latex_value == "-" or "\\mathbf" in latex_value:
        return latex_value
    if latex_value.startswith("\\(") and latex_value.endswith("\\)"):
        inner: str = latex_value[2:-2]
        split_token: str = "{\\scriptstyle"
        if split_token in inner:
            left: str = inner.split(split_token, 1)[0].strip()
            right: str = inner[len(left) :]
            return f"\\(\\mathbf{{{left}}}{right}\\)"
        return f"\\(\\mathbf{{{inner}}}\\)"
    return f"\\(\\mathbf{{{latex_value}}}\\)"


def _compute_molecule_order(
    grouped: dict[str, dict[str, ExperimentResult]],
) -> list[str]:
    molecules: set[str] = set()
    for model_map in grouped.values():
        molecules.update(model_map.keys())
    # Prefer canonical orders when we detect known benchmarks
    # Special-case: when columns are folds (fold1, fold2, ...), sort by numeric index
    if all(m.startswith("fold") and m[4:].isdigit() for m in molecules):
        return [f"fold{i}" for i in sorted(int(m[4:]) for m in molecules)]
    if molecules.issubset(set(MD17_MOLECULE_ORDER)):
        return [m for m in MD17_MOLECULE_ORDER if m in molecules]
    if molecules.issubset(set(MD22_MOLECULE_ORDER)):
        return [m for m in MD22_MOLECULE_ORDER if m in molecules]
    return sorted(molecules)


def _display_molecule_name(molecule: str) -> str:
    # Prefer explicit mapping if available, else title-case the token
    if molecule.startswith("fold") and molecule[4:].isdigit():
        return f"Fold {int(molecule[4:])}"
    if molecule in MD17_MOLECULE_DISPLAY:
        return MD17_MOLECULE_DISPLAY[molecule]
    if molecule in MD22_MOLECULE_DISPLAY:
        return MD22_MOLECULE_DISPLAY[molecule]
    return molecule.replace("_", " ").title()


def _best_model_for_each_molecule(
    grouped: dict[str, dict[str, ExperimentResult]],
    molecule_order: list[str],
    metric: str,
) -> dict[str, str]:
    best_model_for_molecule: dict[str, str] = {}
    for molecule in molecule_order:
        best_model: str | None = None
        best_val: float | None = None
        for model_key, mol_map in grouped.items():
            r: ExperimentResult | None = mol_map.get(molecule)
            if r is None:
                continue
            val: float = r.s2s_mean if metric == "s2s" else r.s2t_mean
            if best_val is None or val < best_val:
                best_val = val
                best_model = model_key
        if best_model is not None:
            best_model_for_molecule[molecule] = best_model
    return best_model_for_molecule


def build_tables(egno_dir: Path, atom_dir: Path, bold_best: bool = True) -> dict[str, str]:
    """Build LaTeX tables per time_lag_mode and return {mode: latex_str}.

    Two directories are required: one for EGNO runs and one for ATOMS (GTNO) runs.
    Results are split by `time_lag_mode` (e.g., 'uniform' or 'last') so each mode
    produces a separate table.
    """
    all_results: list[ExperimentResult] = []
    for d in [egno_dir, atom_dir]:
        results_files: list[Path] = find_results_files(d)
        results_maybe: list[ExperimentResult | None] = [load_experiment_result(p) for p in results_files]
        all_results.extend([r for r in results_maybe if r is not None])

    # Aggregate across folds for the same (mode, model, molecule)
    all_results = _aggregate_folds(all_results)

    outputs: dict[str, str] = {}
    if len(all_results) == 0:
        return outputs

    mode_to_grouped: dict[str, dict[str, dict[str, ExperimentResult]]] = {}
    for r in all_results:
        mode_key: str = r.time_lag_mode if r.time_lag_mode in {"uniform", "last"} else str(r.time_lag_mode)
        mode_map: dict[str, dict[str, ExperimentResult]] = mode_to_grouped.setdefault(mode_key, {})
        canonical_model: str = _canonicalize_model_type(r.model_type)
        model_map: dict[str, ExperimentResult] = mode_map.setdefault(canonical_model, {})
        model_map[r.molecule] = r

    for mode_key, grouped in mode_to_grouped.items():
        if len(grouped) == 0:
            continue
        outputs[mode_key] = build_combined_table_with_two_sections(grouped, bold_best=bold_best)

    return outputs


def _collect_results_from_dirs(directories: list[Path]) -> list[ExperimentResult]:
    """Collect all parsed results from an arbitrary list of directories.

    Each directory is expected to contain one or more immediate children with a
    ``results.json`` file (or a ``results.json`` at its root).
    """
    all_results: list[ExperimentResult] = []
    for d in directories:
        results_files: list[Path] = find_results_files(d)
        results_maybe: list[ExperimentResult | None] = [load_experiment_result(p) for p in results_files]
        all_results.extend([r for r in results_maybe if r is not None])
    return all_results


def build_tables_from_dirs(model_dirs: list[Path], bold_best: bool = True) -> dict[str, str]:
    """Build LaTeX tables per time_lag_mode from a variable number of model directories.

    - Accepts one or more directories. Each directory can correspond to any model type
      (e.g., EGNO, EGNN-S, EGNN-R, GTNO), determined from the saved ``results.json``.
    - Returns a mapping from ``time_lag_mode`` (e.g., "uniform", "last") to LaTeX table text.
    """
    all_results: list[ExperimentResult] = _collect_results_from_dirs(model_dirs)
    all_results = _aggregate_folds(all_results)

    outputs: dict[str, str] = {}
    if len(all_results) == 0:
        return outputs

    mode_to_grouped: dict[str, dict[str, dict[str, ExperimentResult]]] = {}
    for r in all_results:
        mode_key: str = r.time_lag_mode if r.time_lag_mode in {"uniform", "last"} else str(r.time_lag_mode)
        mode_map: dict[str, dict[str, ExperimentResult]] = mode_to_grouped.setdefault(mode_key, {})
        canonical_model: str = _canonicalize_model_type(r.model_type)
        model_map: dict[str, ExperimentResult] = mode_map.setdefault(canonical_model, {})
        model_map[r.molecule] = r

    for mode_key, grouped in mode_to_grouped.items():
        if len(grouped) == 0:
            continue
        outputs[mode_key] = build_combined_table_with_two_sections(grouped, bold_best=bold_best)

    return outputs


def _canonicalize_model_type(name: str) -> str:
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


def _format_percent_value(value: float) -> str:
    return f"\\({value:+.2f}\\%\\)"


def _build_rows_for_metric(
    grouped: dict[str, dict[str, ExperimentResult]],
    molecule_order: list[str],
    metric: str,
    bold_best: bool,
) -> list[str]:
    lines: list[str] = []
    # Preferred row order (top to bottom): EGNN-R, EGNN-S, EGNO, ATOM (GTNO)
    preferred_order: list[str] = ["EGNN_R", "EGNN_S", "EGNO", "GTNO"]
    model_rows: list[str] = [m for m in preferred_order if m in grouped]
    # Append any unexpected models at the end in sorted order
    remaining: list[str] = sorted([m for m in grouped.keys() if m not in model_rows])
    model_rows.extend(remaining)
    best_per_molecule: dict[str, str] = _best_model_for_each_molecule(grouped, molecule_order, metric)
    for model in model_rows:
        display_name: str = MODEL_DISPLAY.get(model, model)
        row_cells: list[str] = [display_name]
        for molecule in molecule_order:
            r: ExperimentResult | None = grouped[model].get(molecule)
            if r is None:
                row_cells.append("-")
            else:
                cell: str = r.latex_s2s if metric == "s2s" else r.latex_s2t
                if bold_best and best_per_molecule.get(molecule) == model:
                    cell = _bold_latex_value(cell)
                row_cells.append(cell)
        lines.append("    " + " & ".join(row_cells) + " \\")
    return lines


def _build_improvement_row(
    grouped: dict[str, dict[str, ExperimentResult]],
    molecule_order: list[str],
    metric: str,
) -> str:
    # Gap is ATOM (GTNO) vs best non-ATOM model: (best_other - atom) / best_other * 100
    if "GTNO" not in grouped:
        return ""
    improvements: list[str] = []
    values: list[float] = []
    for molecule in molecule_order:
        atom_res: ExperimentResult | None = grouped["GTNO"].get(molecule)
        if atom_res is None:
            improvements.append("-")
            continue
        atom_mean: float = atom_res.s2s_mean if metric == "s2s" else atom_res.s2t_mean

        # Find best (minimum) among all models except GTNO
        best_other: float | None = None
        for model_key, mol_map in grouped.items():
            if model_key == "GTNO":
                continue
            other_res: ExperimentResult | None = mol_map.get(molecule)
            if other_res is None:
                continue
            val: float = other_res.s2s_mean if metric == "s2s" else other_res.s2t_mean
            if not (val == val):
                continue
            if best_other is None or val < best_other:
                best_other = val

        if best_other is None or not (atom_mean == atom_mean) or best_other == 0.0:
            improvements.append("-")
            continue

        imp: float = (best_other - atom_mean) / best_other * 100.0
        improvements.append(_format_percent_value(imp))
        values.append(imp)

    if len(values) > 0:
        mean_val: float = sum(values) / float(len(values))
        print(f"Mean {metric.upper()} Gap: {mean_val:+.2f}%")
    return "    " + "\\rowcolor{gray!20} Gap" + " & " + " & ".join(improvements) + " \\"


def build_combined_table_with_two_sections(grouped: dict[str, dict[str, ExperimentResult]], bold_best: bool = True) -> str:
    molecule_order: list[str] = _compute_molecule_order(grouped)
    headers: list[str] = ["", *[_display_molecule_name(m) for m in molecule_order]]
    lines: list[str] = []
    lines.append("\\begin{tabular}{l" + ("c" * len(molecule_order)) + "}")
    lines.append("    \\toprule")
    lines.append("    " + " & ".join(headers) + " \\")
    lines.append("    \\midrule")
    # Top section: S2S
    lines.extend(_build_rows_for_metric(grouped, molecule_order, metric="s2s", bold_best=bold_best))
    lines.append("    \\midrule")
    lines.append(_build_improvement_row(grouped, molecule_order, metric="s2s"))
    lines.append("    \\midrule")
    # Bottom section: S2T
    lines.extend(_build_rows_for_metric(grouped, molecule_order, metric="s2t", bold_best=bold_best))
    lines.append("    \\midrule")
    lines.append(_build_improvement_row(grouped, molecule_order, metric="s2t"))
    lines.append("    \\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def _collect_results(egno_dir: Path, atom_dir: Path) -> list[ExperimentResult]:
    """Collect all parsed results from EGNO and ATOMS directories."""
    all_results: list[ExperimentResult] = []
    for d in [egno_dir, atom_dir]:
        results_files: list[Path] = find_results_files(d)
        results_maybe: list[ExperimentResult | None] = [load_experiment_result(p) for p in results_files]
        all_results.extend([r for r in results_maybe if r is not None])
    return all_results


def _infer_dataset_token_from_results(results: list[ExperimentResult]) -> str:
    """Infer dataset token (e.g., 'md17' or 'md22') from molecules present."""
    molecules: set[str] = {r.molecule for r in results}
    if len(molecules) == 0:
        return "results"
    if molecules.issubset(set(MD17_MOLECULE_ORDER)):
        return "md17"
    if molecules.issubset(set(MD22_MOLECULE_ORDER)):
        return "md22"
    return "results"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build LaTeX tables from MD17 results.json files.")
    _ = parser.add_argument(
        "egno_dir",
        type=str,
        help="Directory containing EGNO experiment folders (each with results.json)",
    )
    _ = parser.add_argument(
        "atom_dir",
        type=str,
        help="Directory containing ATOMS/GTNO experiment folders (each with results.json)",
    )

    args = parser.parse_args()

    egno_dir: Path = Path(args.egno_dir).expanduser().resolve()
    atom_dir: Path = Path(args.atom_dir).expanduser().resolve()
    if not egno_dir.exists():
        raise SystemExit(f"EGNO directory does not exist: {egno_dir}")
    if not atom_dir.exists():
        raise SystemExit(f"ATOMS directory does not exist: {atom_dir}")

    tables_by_mode: dict[str, str] = build_tables(egno_dir, atom_dir)

    # Determine output path base (always under Z_paper_content/tables/)
    results: list[ExperimentResult] = _collect_results(egno_dir, atom_dir)
    dataset_token: str = _infer_dataset_token_from_results(results)
    base_dir: Path = (Path(__file__).resolve().parent / "tables").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    # Write a file per mode (e.g., md17_uniform_tables.tex, md17_last_tables.tex)
    for mode_key, latex_text in tables_by_mode.items():
        suffix: str = f"{dataset_token}_{mode_key}_tables.tex" if dataset_token in {"md17", "md22"} else f"tables_{mode_key}.tex"
        out_path: Path = (base_dir / suffix).resolve()
        _ = out_path.write_text(latex_text, encoding="utf-8")
        # Do not print LaTeX table to stdout; means are printed during construction

    return 0
