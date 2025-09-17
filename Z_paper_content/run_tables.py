import argparse
from pathlib import Path

from Z_paper_content.create_mse_tables import build_tables as build_mse_tables
from Z_paper_content.create_mse_tables import _collect_results as _collect_mse_results
from Z_paper_content.create_mse_tables import _infer_dataset_token_from_results as _infer_dataset_token
from Z_paper_content.create_runtime_tables import build_runtime_table, get_run_times


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate MSE and runtime tables into Z_paper_content/tables/.")
    _ = parser.add_argument("egno_dir", type=str, help="Directory with EGNO runs (contains results.json files)")
    _ = parser.add_argument("atom_dir", type=str, help="Directory with ATOMS/GTNO runs (contains results.json files)")
    args: argparse.Namespace = parser.parse_args()
    return run_tables(str(args.egno_dir), str(args.atom_dir))


def run_tables(egno_dir: str, atom_dir: str) -> int:
    egno_dir_path: Path = Path(egno_dir).expanduser().resolve()
    atom_dir_path: Path = Path(atom_dir).expanduser().resolve()
    if not egno_dir_path.exists():
        raise SystemExit(f"EGNO directory does not exist: {egno_dir_path}")
    if not atom_dir_path.exists():
        raise SystemExit(f"ATOMS directory does not exist: {atom_dir_path}")

    tables_dir: Path = (Path(__file__).resolve().parent / "tables").resolve()
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Build MSE table(s) and write with dataset name appended, split per time_lag_mode
    mse_by_mode: dict[str, str] = build_mse_tables(egno_dir=egno_dir_path, atom_dir=atom_dir_path)
    results = _collect_mse_results(egno_dir=egno_dir_path, atom_dir=atom_dir_path)
    dataset_token: str = _infer_dataset_token(results)

    for mode_key, mse_tex in mse_by_mode.items():
        mse_filename: str = f"{dataset_token}_{mode_key}_tables.tex" if dataset_token in {"md17", "md22"} else f"tables_{mode_key}.tex"
        _ = (tables_dir / mse_filename).write_text(mse_tex, encoding="utf-8")

    # Build a single runtime table using a sensible dataset token
    egno_rt: dict[str, dict[str, list[float]]] = get_run_times(egno_dir_path)
    atom_rt: dict[str, dict[str, list[float]]] = get_run_times(atom_dir_path)
    detected_keys: set[str] = set(egno_rt.keys()) | set(atom_rt.keys())
    preferred_order: list[str] = ["md17", "md22", "rmd17", "tg80"]
    chosen_dataset: str | None = None
    for key in preferred_order:
        if key in detected_keys:
            chosen_dataset = key
            break
    if chosen_dataset is None:
        # fall back to MSE-inferred token if it looks like a known dataset
        if dataset_token in preferred_order:
            chosen_dataset = dataset_token
        elif len(detected_keys) == 1:
            # as a last resort, pick the single detected key
            chosen_dataset = next(iter(detected_keys))

    if chosen_dataset is not None:
        runtime_tex: str = build_runtime_table(egno_dir=egno_dir_path, atom_dir=atom_dir_path, dataset=chosen_dataset, f_peak_tflops=15.0)
        _ = (tables_dir / f"runtime_{chosen_dataset}.tex").write_text(runtime_tex, encoding="utf-8")

    # Also print to stdout for convenience
    # Do not print any LaTeX tables to stdout

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
