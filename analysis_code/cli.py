"""Unified CLI for paper analysis, figures, and tables.

This module lives at the repository root (``analysis_code``) but delegates all
heavy lifting to the existing helpers in ``Z_paper_content``. All outputs are
written under the ``Z_paper_content`` directory (e.g. ``Z_paper_content/tables``).
"""

from __future__ import annotations

import argparse
from pathlib import Path


PAPER_ROOT: Path = Path("Z_paper_content")


def _ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def cmd_tables(args: argparse.Namespace) -> int:
    """Generate LaTeX tables."""
    if args.subcommand == "mse":
        from Z_paper_content.create_mse_tables import build_tables_from_dirs
        from Z_paper_content.create_mse_tables import (
            _collect_results_from_dirs,
            _infer_dataset_token_from_results,
        )

        dirs: list[Path] = [Path(d).expanduser().resolve() for d in args.dirs]
        for d in dirs:
            if not d.exists():
                raise SystemExit(f"Directory does not exist: {d}")

        tables_by_mode = build_tables_from_dirs(dirs, bold_best=not args.no_bold)

        # All tables go under Z_paper_content/tables
        tables_dir: Path = _ensure_dir(PAPER_ROOT / "tables")

        # Infer dataset token for filenames (e.g. md17_uniform_tables.tex)
        results = _collect_results_from_dirs(dirs)
        dataset_token: str = _infer_dataset_token_from_results(results)

        for mode_key, mse_tex in tables_by_mode.items():
            if dataset_token in {"md17", "md22", "rmd17", "tg80"}:
                mse_filename: str = f"{dataset_token}_{mode_key}_tables.tex"
            else:
                mse_filename = f"tables_{mode_key}.tex"
            (tables_dir / mse_filename).write_text(mse_tex, encoding="utf-8")

        if len(tables_by_mode) == 1 and dataset_token in {"md17", "md22", "rmd17", "tg80"}:
            only_mode: str = next(iter(tables_by_mode.keys()))
            alias_path: Path = tables_dir / f"{dataset_token}_tables.tex"
            alias_path.write_text(tables_by_mode[only_mode], encoding="utf-8")

        return 0

    if args.subcommand == "runtime":
        from Z_paper_content.create_runtime_tables import build_runtime_table

        egno_dir: Path = Path(args.egno_dir).expanduser().resolve()
        atom_dir: Path = Path(args.atom_dir).expanduser().resolve()
        if not egno_dir.exists():
            raise SystemExit(f"EGNO directory does not exist: {egno_dir}")
        if not atom_dir.exists():
            raise SystemExit(f"ATOM directory does not exist: {atom_dir}")

        runtime_tex: str = build_runtime_table(
            egno_dir=egno_dir,
            atom_dir=atom_dir,
            dataset=args.dataset,
            f_peak_tflops=args.f_peak,
        )

        tables_dir = _ensure_dir(PAPER_ROOT / "tables")
        (tables_dir / f"runtime_{args.dataset}.tex").write_text(runtime_tex, encoding="utf-8")
        return 0

    if args.subcommand == "all":
        # Backward-compatible wrapper for the existing helper
        from Z_paper_content import run_tables

        return run_tables.run_tables_variadic(
            [str(d) for d in args.dirs],
            bold_best=not args.no_bold,
        )

    raise SystemExit(f"Unknown tables subcommand: {args.subcommand}")


def cmd_figures(args: argparse.Namespace) -> int:
    """Generate figures."""
    if args.subcommand == "ablations":
        from Z_paper_content.figures import set_matplotlib_style
        from Z_paper_content import ablations

        set_matplotlib_style()
        ablation_dir: Path = Path(args.ablation_dir).expanduser().resolve()
        if not ablation_dir.exists():
            raise SystemExit(f"Ablation directory does not exist: {ablation_dir}")

        default_out: Path = PAPER_ROOT / "ablations" / "ablation_MD17_ST.pdf"
        ablation_out: Path = Path(args.output).expanduser().resolve() if args.output is not None else default_out
        _ensure_dir(ablation_out.parent)

        ablations.plot_ablations(
            ablation_dir=ablation_dir,
            save_path=ablation_out,
            error_bar_type=ablations.ErrorBarType.PERCENTILE if args.error_bar_type == "percentile" else ablations.ErrorBarType.STANDARD_DEVIATION,
            add_text=args.add_text,
        )
        return 0

    if args.subcommand == "invariance-p":
        from Z_paper_content import invariances

        p_values: list[int] = invariances._parse_numeric_list(args.p)
        if not p_values:
            raise SystemExit("No P values provided.")

        if len(args.config) != len(args.model):
            raise SystemExit("--config and --model must have the same number of arguments")

        invariances.run_p_invariance(
            p_values=p_values,
            config_paths=args.config,
            model_paths=args.model,
            save_dir=args.save_dir,
        )
        return 0

    if args.subcommand == "invariance-t":
        from Z_paper_content import invariances

        t_values: list[int] = invariances._parse_numeric_list(args.t)
        if not t_values:
            raise SystemExit("No Δt values provided.")

        if len(args.config) != len(args.model):
            raise SystemExit("--config and --model must have the same number of arguments")

        invariances.run_t_invariance(
            t_values=t_values,
            config_paths=args.config,
            model_paths=args.model,
            save_dir=args.save_dir,
        )
        return 0

    if args.subcommand == "multitask-scaling":
        from Z_paper_content.figures import set_matplotlib_style
        from Z_paper_content import multitask_scaling

        set_matplotlib_style(font_size=18)
        benchmark_dir: Path = Path(args.benchmark_dir).expanduser().resolve()
        if not benchmark_dir.exists():
            raise SystemExit(f"Benchmark directory does not exist: {benchmark_dir}")

        save_path: Path | None = Path(args.output).expanduser().resolve() if args.output is not None else None
        if save_path is not None:
            _ensure_dir(save_path.parent)
        multitask_scaling.plot_multitask_scaling(benchmark_dir=benchmark_dir, save_path=save_path)
        return 0

    if args.subcommand == "trajectories":
        from Z_paper_content.figures import set_matplotlib_style
        from Z_paper_content import trajectory_visualisation

        set_matplotlib_style()
        data_dir: Path = Path(args.data_dir).expanduser().resolve()
        if not data_dir.exists():
            raise SystemExit(f"Data directory does not exist: {data_dir}")

        if args.dataset == "md17":
            trajectory_visualisation.create_tiled_figure(data_dir, "md17", 2, 4)
        elif args.dataset == "rmd17":
            trajectory_visualisation.create_tiled_figure(data_dir, "rmd17", 2, 4)
        elif args.dataset == "tg80":
            trajectory_visualisation.create_tiled_figure(data_dir, "tg80", 2, 4)
        else:
            raise SystemExit(f"Unknown dataset: {args.dataset}")

        return 0

    raise SystemExit(f"Unknown figures subcommand: {args.subcommand}")


def cmd_analyze(args: argparse.Namespace) -> int:
    """Run analysis scripts."""
    if args.subcommand == "folds":
        from Z_paper_content import analyze_folds
        import numpy as np

        egno_dir: Path = Path(args.egno_dir).expanduser().resolve()
        atom_dir: Path = Path(args.atom_dir).expanduser().resolve()
        if not egno_dir.exists():
            raise SystemExit(f"EGNO directory does not exist: {egno_dir}")
        if not atom_dir.exists():
            raise SystemExit(f"ATOM directory does not exist: {atom_dir}")

        egno_stats: dict[int, dict[str, dict[str, float | int]]] = analyze_folds.analyze_folds(str(egno_dir))
        atom_stats: dict[int, dict[str, dict[str, float | int]]] = analyze_folds.analyze_folds(str(atom_dir))

        print("\nResults by fold (EGNO above ATOM, then % diff):")
        print("-" * 70)
        common_folds: list[int] = sorted(set(egno_stats.keys()) & set(atom_stats.keys()))

        total_s2s_improvement: float = 0.0
        total_s2t_improvement: float = 0.0
        s2s_fold_count_for_average: int = 0
        s2t_fold_count_for_average: int = 0

        for fold_num in common_folds:
            print(f"Fold {fold_num}:")
            # EGNO
            print("  EGNO S2S Test Loss (x10^-2):")
            print(f"    Mean: {egno_stats[fold_num]['s2s']['mean'] * 100:.2f}")
            print(f"    2 Std:  {egno_stats[fold_num]['s2s']['std'] * 2 * 100:.2f}")
            print(f"    N runs: {egno_stats[fold_num]['s2s']['n_runs']}")
            print("  EGNO S2T Test Loss (x10^-2):")
            print(f"    Mean: {egno_stats[fold_num]['s2t']['mean'] * 100:.2f}")
            print(f"    2 Std:  {egno_stats[fold_num]['s2t']['std'] * 2 * 100:.2f}")
            print(f"    N runs: {egno_stats[fold_num]['s2t']['n_runs']}")
            # ATOM
            print("  ATOM S2S Test Loss (x10^-2):")
            print(f"    Mean: {atom_stats[fold_num]['s2s']['mean'] * 100:.2f}")
            print(f"    2 Std:  {atom_stats[fold_num]['s2s']['std'] * 2 * 100:.2f}")
            print(f"    N runs: {atom_stats[fold_num]['s2s']['n_runs']}")
            print("  ATOM S2T Test Loss (x10^-2):")
            print(f"    Mean: {atom_stats[fold_num]['s2t']['mean'] * 100:.2f}")
            print(f"    2 Std:  {atom_stats[fold_num]['s2t']['std'] * 2 * 100:.2f}")
            print(f"    N runs: {atom_stats[fold_num]['s2t']['n_runs']}")
            # % diff
            s2s_atom: float = float(
                atom_stats[fold_num]["s2s"]["mean"]  # type: ignore[assignment]
            )
            s2s_egno: float = float(
                egno_stats[fold_num]["s2s"]["mean"]  # type: ignore[assignment]
            )
            s2t_atom: float = float(
                atom_stats[fold_num]["s2t"]["mean"]  # type: ignore[assignment]
            )
            s2t_egno: float = float(
                egno_stats[fold_num]["s2t"]["mean"]  # type: ignore[assignment]
            )

            # For overall average improvement (ATOM vs EGNO baseline)
            s2s_improvement_current_fold: float = (
                ((s2s_egno - s2s_atom) / s2s_egno * 100.0) if s2s_egno != 0.0 and not (np.isnan(s2s_atom) or np.isnan(s2s_egno)) else float("nan")
            )
            s2t_improvement_current_fold: float = (
                ((s2t_egno - s2t_atom) / s2t_egno * 100.0) if s2t_egno != 0.0 and not (np.isnan(s2t_atom) or np.isnan(s2t_egno)) else float("nan")
            )

            if not np.isnan(s2s_improvement_current_fold):
                total_s2s_improvement += s2s_improvement_current_fold
                s2s_fold_count_for_average += 1

            if not np.isnan(s2t_improvement_current_fold):
                total_s2t_improvement += s2t_improvement_current_fold
                s2t_fold_count_for_average += 1

            # For per-fold absolute % diff display
            s2s_pct_abs: float = (
                abs(((s2s_egno - s2s_atom) / s2s_egno) * 100.0)
                if s2s_egno != 0.0 and not (np.isnan(s2s_atom) or np.isnan(s2s_egno))
                else float("nan")
            )
            s2t_pct_abs: float = (
                abs(((s2t_egno - s2t_atom) / s2t_egno) * 100.0)
                if s2t_egno != 0.0 and not (np.isnan(s2t_atom) or np.isnan(s2t_egno))
                else float("nan")
            )

            print(f"  % Diff S2S Mean (Abs): {s2s_pct_abs:.2f}%")
            print(f"  % Diff S2T Mean (Abs): {s2t_pct_abs:.2f}%")
            print("-" * 70)

        mean_s2s_improvement: float = total_s2s_improvement / float(s2s_fold_count_for_average) if s2s_fold_count_for_average > 0 else float("nan")
        mean_s2t_improvement: float = total_s2t_improvement / float(s2t_fold_count_for_average) if s2t_fold_count_for_average > 0 else float("nan")

        print("\nOverall Mean Improvement (ATOM vs EGNO baseline):")
        print("-" * 70)
        print(f"Mean S2S Improvement: {mean_s2s_improvement:.2f}% (over {s2s_fold_count_for_average} folds)")
        print(f"Mean S2T Improvement: {mean_s2t_improvement:.2f}% (over {s2t_fold_count_for_average} folds)")
        print("-" * 70)
        return 0

    if args.subcommand == "dataset":
        from Z_paper_content.figures import set_matplotlib_style
        from Z_paper_content import dataset_info
        import numpy as np

        set_matplotlib_style()
        data_dir: Path = Path(args.data_dir).expanduser().resolve()
        if not data_dir.exists():
            raise SystemExit(f"Data directory does not exist: {data_dir}")

        # First pass: collect all data to find global maximum
        all_position_variances: list[float] = []
        npz_files: list[Path] = list(data_dir.glob("*.npz"))
        for filepath in npz_files:
            data = np.load(filepath)
            if args.dataset in {"rmd17", "tg80"}:
                arr = data["coords"]
            else:
                arr = data["R"]
            if arr.ndim == 3 and arr.shape[2] == 3:
                all_position_variances.append(float(np.var(arr)))

        global_max_x: float = max(all_position_variances) * 1.15 if all_position_variances else 1.0

        # Second pass: create plot
        dataset_info.create_corrected_volatility_visualization(data_dir, args.dataset, global_max_x)
        return 0

    if args.subcommand == "timing":
        # Call into the library API directly rather than re-parsing CLI flags
        from Z_paper_content.md17_md_vs_atom_timing import run_experiment
        import json

        atom_run_dir: Path = Path(args.atom_run_dir).expanduser().resolve()
        if not atom_run_dir.exists():
            raise SystemExit(f"ATOM benchmark directory does not exist: {atom_run_dir}")

        limit_molecules: list[str] | None = list(args.molecules) if args.molecules is not None else None
        stats: dict[str, object] = run_experiment(
            atom_run_dir=atom_run_dir,
            num_repeats=args.num_repeats,
            limit_molecules=limit_molecules,
        )

        # Ensure we always write under Z_paper_content by default
        if args.output_json:
            out_path: Path = Path(args.output_json).expanduser().resolve()
        else:
            out_path = PAPER_ROOT / "md17_md_vs_atom_timing.json"
        _ensure_dir(out_path.parent)
        out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"Saved ATOM inference timing results to {out_path}")
        return 0

    raise SystemExit(f"Unknown analyze subcommand: {args.subcommand}")


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Unified CLI for paper analysis, figures, and tables.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")

    # Tables subcommand
    tables_parser = subparsers.add_parser("tables", help="Generate LaTeX tables")
    tables_subparsers = tables_parser.add_subparsers(dest="subcommand", required=True)
    mse_parser = tables_subparsers.add_parser("mse", help="Generate MSE tables")
    mse_parser.add_argument("dirs", nargs="+", help="Directories with results.json files")
    mse_parser.add_argument("--no-bold", action="store_true", help="Do not bold best values")
    runtime_parser = tables_subparsers.add_parser("runtime", help="Generate runtime tables")
    runtime_parser.add_argument("egno_dir", help="Directory with EGNO runs")
    runtime_parser.add_argument("atom_dir", help="Directory with ATOM runs")
    runtime_parser.add_argument("--dataset", default="md17", choices=["md17", "md22", "rmd17", "tg80"], help="Dataset name")
    runtime_parser.add_argument("--f-peak", type=float, default=15.0, help="Peak TFLOPS")
    all_parser = tables_subparsers.add_parser("all", help="Generate all tables")
    all_parser.add_argument("dirs", nargs="+", help="Directories with results.json files")
    all_parser.add_argument("--no-bold", action="store_true", help="Do not bold best values")

    # Figures subcommand
    figures_parser = subparsers.add_parser("figures", help="Generate figures")
    figures_subparsers = figures_parser.add_subparsers(dest="subcommand", required=True)
    ablations_parser = figures_subparsers.add_parser("ablations", help="Generate ablation plots")
    ablations_parser.add_argument("ablation_dir", help="Directory with ablation results")
    ablations_parser.add_argument("--output", help="Output path (default: Z_paper_content/ablations/ablation_MD17_ST.pdf)")
    ablations_parser.add_argument("--error-bar-type", choices=["percentile", "std_dev"], default="percentile", help="Error bar type")
    ablations_parser.add_argument("--add-text", action="store_true", default=True, help="Add text labels")
    invariance_p_parser = figures_subparsers.add_parser("invariance-p", help="Generate P-invariance plots")
    invariance_p_parser.add_argument("--p", required=True, help="List of P values. Accepts '[4,8,12]' or space-separated values.")
    invariance_p_parser.add_argument(
        "--config", required=True, nargs="+", help="One or more paths to config .toml files or directories (same count as --model)."
    )
    invariance_p_parser.add_argument(
        "--model", required=True, nargs="+", help="One or more paths to model checkpoints or directories (same count as --config)."
    )
    invariance_p_parser.add_argument("--save-dir", help="Directory to save output (default: Z_paper_content/invariance_results)")
    invariance_t_parser = figures_subparsers.add_parser("invariance-t", help="Generate T-invariance plots")
    invariance_t_parser.add_argument("--t", required=True, help="List of Δt values. Accepts '[1,2,4]' or space-separated values.")
    invariance_t_parser.add_argument(
        "--config", required=True, nargs="+", help="One or more paths to config .toml files or directories (same count as --model)."
    )
    invariance_t_parser.add_argument(
        "--model", required=True, nargs="+", help="One or more paths to model checkpoints or directories (same count as --config)."
    )
    invariance_t_parser.add_argument("--save-dir", help="Directory to save output (default: Z_paper_content/invariance_results)")
    multitask_parser = figures_subparsers.add_parser("multitask-scaling", help="Generate multitask scaling plots")
    multitask_parser.add_argument("benchmark_dir", help="Directory with multitask scaling results")
    multitask_parser.add_argument("--output", help="Output path")
    trajectories_parser = figures_subparsers.add_parser("trajectories", help="Generate trajectory visualizations")
    trajectories_parser.add_argument("data_dir", help="Directory with .npz trajectory files")
    trajectories_parser.add_argument("--dataset", required=True, choices=["md17", "rmd17", "tg80"], help="Dataset name")

    # Analyze subcommand
    analyze_parser = subparsers.add_parser("analyze", help="Run analysis scripts")
    analyze_subparsers = analyze_parser.add_subparsers(dest="subcommand", required=True)
    folds_parser = analyze_subparsers.add_parser("folds", help="Analyze fold results")
    folds_parser.add_argument("egno_dir", help="Directory with EGNO fold results")
    folds_parser.add_argument("atom_dir", help="Directory with ATOM fold results")
    dataset_parser = analyze_subparsers.add_parser("dataset", help="Analyze dataset trajectories")
    dataset_parser.add_argument("data_dir", help="Directory with .npz trajectory files")
    dataset_parser.add_argument("--dataset", required=True, choices=["md17", "rmd17", "tg80"], help="Dataset name")
    timing_parser = analyze_subparsers.add_parser("timing", help="Analyze inference timing")
    timing_parser.add_argument("--atom-run-dir", required=True, help="Directory containing ATOM benchmark runs")
    timing_parser.add_argument("--num-repeats", type=int, default=3, help="Number of repeated eval_epoch calls per molecule (default: 3)")
    timing_parser.add_argument("--molecules", nargs="*", help="Optional subset of MD17 molecule base names to evaluate (e.g. aspirin benzene).")
    timing_parser.add_argument(
        "--output-json",
        help="Path to JSON file where timing statistics will be saved (default: Z_paper_content/md17_md_vs_atom_timing.json)",
    )

    args: argparse.Namespace = parser.parse_args()

    if args.command == "tables":
        return cmd_tables(args)
    if args.command == "figures":
        return cmd_figures(args)
    if args.command == "analyze":
        return cmd_analyze(args)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
