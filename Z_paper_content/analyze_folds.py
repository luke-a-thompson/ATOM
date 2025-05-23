import json
import os
import numpy as np
from collections import defaultdict


def get_fold_number(path: str) -> int:
    """Extract fold number from directory name."""
    return int(path.split("fold")[1].split("_")[0])


def analyze_folds(base_dir: str) -> dict[int, dict[str, dict[str, float | int]]]:
    """Analyze results across folds and return statistics from all individual runs."""
    results_by_fold: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"s2s": [], "s2t": []})

    # Walk through all directories
    for dir_name in os.listdir(base_dir):
        if not (dir_name.startswith("atom_tg80_multitask_muon_fold") or dir_name.startswith("egno_tg80_multitask_muon_fold")):
            continue

        results_path = os.path.join(base_dir, dir_name, "results.json")
        if not os.path.exists(results_path):
            continue

        fold_num = get_fold_number(dir_name)

        with open(results_path, "r") as f:
            results = json.load(f)
            # Extract all individual run losses from 'single_run_results'
            if "single_run_results" in results:
                for run in results["single_run_results"]:
                    if "s2s_test_loss" in run:
                        results_by_fold[fold_num]["s2s"].append(run["s2s_test_loss"])
                    if "s2t_test_loss" in run:
                        results_by_fold[fold_num]["s2t"].append(run["s2t_test_loss"])

    # Calculate statistics
    stats: dict[int, dict[str, dict[str, float | int]]] = {}
    for fold_num, metrics in results_by_fold.items():
        stats[fold_num] = {
            "s2s": {
                "mean": float(np.mean(metrics["s2s"])) if metrics["s2s"] else float("nan"),
                "std": float(np.std(metrics["s2s"])) if metrics["s2s"] else float("nan"),
                "n_runs": len(metrics["s2s"]),
            },
            "s2t": {
                "mean": float(np.mean(metrics["s2t"])) if metrics["s2t"] else float("nan"),
                "std": float(np.std(metrics["s2t"])) if metrics["s2t"] else float("nan"),
                "n_runs": len(metrics["s2t"]),
            },
        }

    return stats


if __name__ == "__main__":
    egno_dir = "benchmark_runs/tg80_egno_mt"
    atom_dir = "benchmark_runs/tg80_atom_mt"
    egno_stats = analyze_folds(egno_dir)
    atom_stats = analyze_folds(atom_dir)

    print("\nResults by fold (EGNO above ATOM, then % diff):")
    print("-" * 70)
    common_folds = sorted(set(egno_stats.keys()) & set(atom_stats.keys()))

    total_s2s_improvement: float = 0.0
    total_s2t_improvement: float = 0.0
    s2s_fold_count_for_average: int = 0
    s2t_fold_count_for_average: int = 0

    for fold_num in common_folds:
        print(f"Fold {fold_num}:")
        # EGNO
        print(f"  EGNO S2S Test Loss (x10^-2):")
        print(f"    Mean: {egno_stats[fold_num]['s2s']['mean'] * 100:.2f}")
        print(f"    2 Std:  {egno_stats[fold_num]['s2s']['std'] * 2 * 100:.2f}")
        print(f"    N runs: {egno_stats[fold_num]['s2s']['n_runs']}")
        print(f"  EGNO S2T Test Loss (x10^-2):")
        print(f"    Mean: {egno_stats[fold_num]['s2t']['mean'] * 100:.2f}")
        print(f"    2 Std:  {egno_stats[fold_num]['s2t']['std'] * 2 * 100:.2f}")
        print(f"    N runs: {egno_stats[fold_num]['s2t']['n_runs']}")
        # ATOM
        print(f"  ATOM S2S Test Loss (x10^-2):")
        print(f"    Mean: {atom_stats[fold_num]['s2s']['mean'] * 100:.2f}")
        print(f"    2 Std:  {atom_stats[fold_num]['s2s']['std'] * 2 * 100:.2f}")
        print(f"    N runs: {atom_stats[fold_num]['s2s']['n_runs']}")
        print(f"  ATOM S2T Test Loss (x10^-2):")
        print(f"    Mean: {atom_stats[fold_num]['s2t']['mean'] * 100:.2f}")
        print(f"    2 Std:  {atom_stats[fold_num]['s2t']['std'] * 2 * 100:.2f}")
        print(f"    N runs: {atom_stats[fold_num]['s2t']['n_runs']}")
        # % diff
        s2s_atom = atom_stats[fold_num]["s2s"]["mean"]
        s2s_egno = egno_stats[fold_num]["s2s"]["mean"]
        s2t_atom = atom_stats[fold_num]["s2t"]["mean"]
        s2t_egno = egno_stats[fold_num]["s2t"]["mean"]

        # For overall average improvement (ATOM vs EGNO baseline)
        s2s_improvement_current_fold = ((s2s_egno - s2s_atom) / s2s_egno * 100) if s2s_egno != 0 and not (np.isnan(s2s_atom) or np.isnan(s2s_egno)) else float("nan")
        s2t_improvement_current_fold = ((s2t_egno - s2t_atom) / s2t_egno * 100) if s2t_egno != 0 and not (np.isnan(s2t_atom) or np.isnan(s2t_egno)) else float("nan")

        if not np.isnan(s2s_improvement_current_fold):
            total_s2s_improvement += s2s_improvement_current_fold
            s2s_fold_count_for_average += 1

        if not np.isnan(s2t_improvement_current_fold):
            total_s2t_improvement += s2t_improvement_current_fold
            s2t_fold_count_for_average += 1

        # For per-fold absolute % diff display
        s2s_pct_abs = abs(((s2s_egno - s2s_atom) / s2s_egno) * 100) if s2s_egno != 0 and not (np.isnan(s2s_atom) or np.isnan(s2s_egno)) else float("nan")
        s2t_pct_abs = abs(((s2t_egno - s2t_atom) / s2t_egno) * 100) if s2t_egno != 0 and not (np.isnan(s2t_atom) or np.isnan(s2t_egno)) else float("nan")

        print(f"  % Diff S2S Mean (Abs): {s2s_pct_abs:.2f}%")
        print(f"  % Diff S2T Mean (Abs): {s2t_pct_abs:.2f}%")
        print("-" * 70)

    # Calculate and print mean improvements across folds
    mean_s2s_improvement = (total_s2s_improvement / s2s_fold_count_for_average) if s2s_fold_count_for_average > 0 else float("nan")
    mean_s2t_improvement = (total_s2t_improvement / s2t_fold_count_for_average) if s2t_fold_count_for_average > 0 else float("nan")

    print("\nOverall Mean Improvement (ATOM vs EGNO baseline):")
    print("-" * 70)
    print(f"Mean S2S Improvement: {mean_s2s_improvement:.2f}% (over {s2s_fold_count_for_average} folds)")
    print(f"Mean S2T Improvement: {mean_s2t_improvement:.2f}% (over {s2t_fold_count_for_average} folds)")
    print("-" * 70)
