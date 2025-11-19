import argparse
from pathlib import Path

from Z_paper_content.create_mse_tables import build_tables as build_mse_tables
from Z_paper_content.create_mse_tables import build_tables_from_dirs as build_mse_tables_from_dirs
from Z_paper_content.create_mse_tables import _collect_results as _collect_mse_results
from Z_paper_content.create_mse_tables import _infer_dataset_token_from_results as _infer_dataset_token
from Z_paper_content.create_runtime_tables import build_runtime_table, get_run_times


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate MSE and runtime tables into Z_paper_content/tables/.")
    _ = parser.add_argument("dirs", nargs="+", type=str, help="One or more directories with runs (contain results.json files)")
    _ = parser.add_argument("--no-bold", action="store_true", help="Do not bold best values in tables")
    args: argparse.Namespace = parser.parse_args()
    return run_tables_variadic([str(d) for d in args.dirs], bold_best=(not bool(args.no_bold)))


def run_tables(egno_dir: str, atom_dir: str) -> int:
    # Backward-compatible 2-arg entrypoint
    return run_tables_variadic([egno_dir, atom_dir])


def run_tables_variadic(dirs: list[str], bold_best: bool = True) -> int:
    if len(dirs) < 1:
        raise SystemExit("At least one directory path must be provided")
    dir_paths: list[Path] = [Path(d).expanduser().resolve() for d in dirs]
    for p in dir_paths:
        if not p.exists():
            raise SystemExit(f"Directory does not exist: {p}")

    tables_dir: Path = (Path(__file__).resolve().parent / "tables").resolve()
    tables_dir.mkdir(parents=True, exist_ok=True)

    def _detect_dataset_from_dirnames() -> str | None:
        tokens: list[str] = ["md17", "md22", "rmd17", "tg80"]
        names: list[str] = [p.name.lower() for p in dir_paths]
        for t in tokens:
            if any(t in n for n in names):
                return t
        return None

    # For pre-inference, if we have exactly two dirs and they are EGNO/ATOMS style, use the 2-dir collector,
    # otherwise collect across all provided dirs
    if len(dir_paths) == 2:
        pre_inferred_token: str = _infer_dataset_token(_collect_mse_results(egno_dir=dir_paths[0], atom_dir=dir_paths[1]))
    else:
        from Z_paper_content.create_mse_tables import _collect_results_from_dirs

        pre_inferred_token = _infer_dataset_token(_collect_results_from_dirs(dir_paths))

    # Build a single runtime table and detect dataset key robustly (supports md17/md22/rmd17/tg80)
    detected_keys: set[str] = set()
    if len(dir_paths) == 2:
        egno_rt: dict[str, dict[str, list[float]]] = get_run_times(dir_paths[0])
        atom_rt: dict[str, dict[str, list[float]]] = get_run_times(dir_paths[1])
        detected_keys = set(egno_rt.keys()) | set(atom_rt.keys())
    preferred_order: list[str] = ["md17", "md22", "rmd17", "tg80"]
    chosen_dataset: str | None = _detect_dataset_from_dirnames()
    for key in preferred_order:
        if key in detected_keys:
            chosen_dataset = key
            break
    if chosen_dataset is None:
        # fall back to MSE-inferred token if it looks like a known dataset
        if pre_inferred_token in preferred_order:
            chosen_dataset = pre_inferred_token
        elif len(detected_keys) == 1:
            # as a last resort, pick the single detected key
            chosen_dataset = next(iter(detected_keys))

    if chosen_dataset is not None and len(dir_paths) == 2:
        runtime_tex: str = build_runtime_table(egno_dir=dir_paths[0], atom_dir=dir_paths[1], dataset=chosen_dataset, f_peak_tflops=15.0)
        _ = (tables_dir / f"runtime_{chosen_dataset}.tex").write_text(runtime_tex, encoding="utf-8")

    # Build MSE table(s) and write with dataset name appended, split per time_lag_mode
    if len(dir_paths) == 2:
        mse_by_mode: dict[str, str] = build_mse_tables(egno_dir=dir_paths[0], atom_dir=dir_paths[1], bold_best=bold_best)
        results = _collect_mse_results(egno_dir=dir_paths[0], atom_dir=dir_paths[1])
    else:
        mse_by_mode = build_mse_tables_from_dirs(dir_paths, bold_best=bold_best)
        from Z_paper_content.create_mse_tables import _collect_results_from_dirs

        results = _collect_results_from_dirs(dir_paths)
    inferred_from_molecules: str = _infer_dataset_token(results)
    mse_dataset_token: str = chosen_dataset if chosen_dataset is not None else inferred_from_molecules

    for mode_key, mse_tex in mse_by_mode.items():
        if mse_dataset_token in {"md17", "md22", "rmd17", "tg80"}:
            mse_filename: str = f"{mse_dataset_token}_{mode_key}_tables.tex"
        else:
            mse_filename = f"tables_{mode_key}.tex"
        _ = (tables_dir / mse_filename).write_text(mse_tex, encoding="utf-8")

    # If only a single mode was produced, also create a simplified alias like `tg80_tables.tex`
    if len(mse_by_mode) == 1 and mse_dataset_token in {"md17", "md22", "rmd17", "tg80"}:
        only_mode: str = next(iter(mse_by_mode.keys()))
        alias_path: Path = tables_dir / f"{mse_dataset_token}_tables.tex"
        _ = alias_path.write_text(mse_by_mode[only_mode], encoding="utf-8")

    # Also print to stdout for convenience
    # Do not print any LaTeX tables to stdout

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
