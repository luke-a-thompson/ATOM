import argparse
import json
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
}


@dataclass
class ExperimentResult:
    model_type: str
    molecule: str
    latex_s2s: str
    latex_s2t: str
    s2s_mean: float
    s2t_mean: float


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

    # Also include aggregated EGNO-style files: *MoleculeType.*_results.json at root
    for child in sorted(root.iterdir()):
        if child.is_file() and child.suffix == ".json" and child.name.endswith("_results.json") and "MoleculeType." in child.name:
            results_files.append(child)
    return results_files


def load_experiment_result(results_path: Path) -> ExperimentResult | None:
    """Load a single ExperimentResult from a results.json file, or None if invalid."""
    # Special-case aggregated EGNO-style files
    if results_path.name != "results.json" and results_path.name.endswith("_results.json") and "MoleculeType." in results_path.name:
        agg = _load_aggregated_results(results_path)
        if agg is not None:
            return agg
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

        return ExperimentResult(
            model_type=model_type,
            molecule=molecule,
            latex_s2s=latex_s2s,
            latex_s2t=latex_s2t,
            s2s_mean=s2s_mean,
            s2t_mean=s2t_mean,
        )
    except Exception:
        return None


def _parse_molecule_from_aggregated_filename(path: Path) -> str | None:
    name: str = path.name
    try:
        after_tag: str = name.split("MoleculeType.", 1)[1]
        molecule_part: str = after_tag.split("_num_timesteps", 1)[0]
        return molecule_part.strip().lower()
    except Exception:
        return None


def _format_latex_from_mean_std(mean_value: float, std_value: float) -> str:
    # Scale by 100 to match existing latex formatting in MD17 results.json
    mean_scaled: float = mean_value * 100.0
    std_scaled: float = std_value * 100.0
    return f"\\({mean_scaled:.2f}{{\\scriptstyle \\pm{std_scaled:.2f}}}\\)"


def _load_aggregated_results(results_path: Path) -> ExperimentResult | None:
    try:
        with results_path.open("r", encoding="utf-8") as f:
            obj: object = json.load(f)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    data: dict[str, object] = obj

    molecule: str | None = _parse_molecule_from_aggregated_filename(results_path)
    if molecule is None:
        return None

    model_type: str = "EGNO"

    s2t_mean_obj: object = data.get("mean_test_final_loss", float("nan"))
    s2s_mean_obj: object = data.get("mean_test_all_loss", float("nan"))
    # Prefer std_test_* if present, else fall back to test_loss_std
    s2t_std_obj: object = data.get("std_test_final_loss", data.get("test_loss_std", float("nan")))
    s2s_std_obj: object = data.get("std_test_all_loss", data.get("test_loss_std", float("nan")))

    s2t_mean: float = float(s2t_mean_obj) if isinstance(s2t_mean_obj, (int, float)) else float("nan")
    s2s_mean: float = float(s2s_mean_obj) if isinstance(s2s_mean_obj, (int, float)) else float("nan")
    s2t_std: float = float(s2t_std_obj) if isinstance(s2t_std_obj, (int, float)) else float("nan")
    s2s_std: float = float(s2s_std_obj) if isinstance(s2s_std_obj, (int, float)) else float("nan")

    latex_s2t: str = _format_latex_from_mean_std(s2t_mean, s2t_std) if s2t_mean == s2t_mean and s2t_std == s2t_std else "-"
    latex_s2s: str = _format_latex_from_mean_std(s2s_mean, s2s_std) if s2s_mean == s2s_mean and s2s_std == s2s_std else "-"

    return ExperimentResult(
        model_type=model_type,
        molecule=molecule,
        latex_s2s=latex_s2s,
        latex_s2t=latex_s2t,
        s2s_mean=s2s_mean,
        s2t_mean=s2t_mean,
    )


def group_results_by_model(results: list[ExperimentResult]) -> dict[str, dict[str, ExperimentResult]]:
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


def _compute_molecule_order(grouped: dict[str, dict[str, ExperimentResult]]) -> list[str]:
    molecules: set[str] = set()
    for model_map in grouped.values():
        molecules.update(model_map.keys())
    # Prefer canonical orders when we detect known benchmarks
    if molecules.issubset(set(MD17_MOLECULE_ORDER)):
        return [m for m in MD17_MOLECULE_ORDER if m in molecules]
    if molecules.issubset(set(MD22_MOLECULE_ORDER)):
        return [m for m in MD22_MOLECULE_ORDER if m in molecules]
    return sorted(molecules)


def _display_molecule_name(molecule: str) -> str:
    # Prefer explicit mapping if available, else title-case the token
    if molecule in MD17_MOLECULE_DISPLAY:
        return MD17_MOLECULE_DISPLAY[molecule]
    if molecule in MD22_MOLECULE_DISPLAY:
        return MD22_MOLECULE_DISPLAY[molecule]
    return molecule.replace("_", " ").title()


def _best_model_for_each_molecule(grouped: dict[str, dict[str, ExperimentResult]], molecule_order: list[str], metric: str) -> dict[str, str]:
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


def build_tables(egno_dir: Path, atom_dir: Path) -> str:
    """Build LaTeX text containing two tables: top S2S and lower S2T.

    Two directories are required: one for EGNO runs and one for ATOMS (GTNO) runs.
    The rows will include both models if present.
    """
    all_results: list[ExperimentResult] = []
    for d in [egno_dir, atom_dir]:
        results_files: list[Path] = find_results_files(d)
        results_maybe: list[ExperimentResult | None] = [load_experiment_result(p) for p in results_files]
        all_results.extend([r for r in results_maybe if r is not None])

    grouped: dict[str, dict[str, ExperimentResult]] = group_results_by_model(all_results)

    sections: list[str] = []
    if len(grouped) == 0:
        return "% No results found under: " + str(egno_dir) + ", " + str(atom_dir)

    sections.append(build_combined_table_with_two_sections(grouped))
    return "\n".join(sections) + "\n"


def _canonicalize_model_type(name: str) -> str:
    n: str = name.strip().lower()
    if n in {"gtno", "atom", "atoms"}:
        return "GTNO"
    if n in {"egno"}:
        return "EGNO"
    return name


def _format_percent_value(value: float) -> str:
    return f"\\({value:.2f}\\%\\)"


def _build_rows_for_metric(grouped: dict[str, dict[str, ExperimentResult]], molecule_order: list[str], metric: str) -> list[str]:
    lines: list[str] = []
    # Row order: EGNO then GTNO if present
    model_rows: list[str] = [m for m in ["EGNO", "GTNO"] if m in grouped]
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
                if best_per_molecule.get(molecule) == model:
                    cell = _bold_latex_value(cell)
                row_cells.append(cell)
        row_cells.append("")
        lines.append("    " + " & ".join(row_cells) + " \\")
    return lines


def _build_improvement_row(grouped: dict[str, dict[str, ExperimentResult]], molecule_order: list[str], metric: str) -> str:
    # Improvement defined as percent reduction vs EGNO baseline when GTNO is target
    if "GTNO" not in grouped or "EGNO" not in grouped:
        return ""
    improvements: list[str] = []
    values: list[float] = []
    for molecule in molecule_order:
        tgt: ExperimentResult | None = grouped["GTNO"].get(molecule)
        base: ExperimentResult | None = grouped["EGNO"].get(molecule)
        if tgt is None or base is None:
            improvements.append("-")
            continue
        tgt_mean: float = tgt.s2s_mean if metric == "s2s" else tgt.s2t_mean
        base_mean: float = base.s2s_mean if metric == "s2s" else base.s2t_mean
        if not (base_mean == base_mean) or base_mean == 0.0 or not (tgt_mean == tgt_mean):
            improvements.append("-")
            continue
        imp: float = (base_mean - tgt_mean) / base_mean * 100.0
        improvements.append(_format_percent_value(imp))
        values.append(imp)
    mean_cell: str = "-"
    if len(values) > 0:
        mean_val: float = sum(values) / float(len(values))
        mean_cell = _format_percent_value(mean_val)
    return "    " + "\\rowcolor{gray!20} Improvement (\\%)" + " & " + " & ".join(improvements) + " & " + mean_cell + " \\"


def build_combined_table_with_two_sections(grouped: dict[str, dict[str, ExperimentResult]]) -> str:
    molecule_order: list[str] = _compute_molecule_order(grouped)
    headers: list[str] = ["", *[_display_molecule_name(m) for m in molecule_order], "Mean %"]
    lines: list[str] = []
    lines.append("\\begin{tabular}{l" + ("c" * len(molecule_order)) + " c}")
    lines.append("    \\toprule")
    lines.append("    " + " & ".join(headers) + " \\")
    lines.append("    \\midrule")
    # Top section: S2S
    lines.extend(_build_rows_for_metric(grouped, molecule_order, metric="s2s"))
    lines.append("    \\midrule")
    lines.append(_build_improvement_row(grouped, molecule_order, metric="s2s"))
    lines.append("    \\midrule")
    # Bottom section: S2T
    lines.extend(_build_rows_for_metric(grouped, molecule_order, metric="s2t"))
    lines.append("    \\midrule")
    lines.append(_build_improvement_row(grouped, molecule_order, metric="s2t"))
    lines.append("    \\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


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
    _ = parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to write the LaTeX output (e.g., Z_paper_content/md17_tables.tex). If omitted, writes alongside input_dir as tables.tex.",
    )

    args = parser.parse_args()

    egno_dir: Path = Path(args.egno_dir).expanduser().resolve()
    atom_dir: Path = Path(args.atom_dir).expanduser().resolve()
    if not egno_dir.exists():
        raise SystemExit(f"EGNO directory does not exist: {egno_dir}")
    if not atom_dir.exists():
        raise SystemExit(f"ATOMS directory does not exist: {atom_dir}")

    latex_text: str = build_tables(egno_dir, atom_dir)

    # Determine output path
    if args.out is not None:
        out_path: Path = Path(args.out).expanduser().resolve()
    else:
        out_path = Path.cwd() / "tables.tex"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex_text, encoding="utf-8")

    print(latex_text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
