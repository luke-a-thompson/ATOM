import json
import os
from pathlib import Path
import numpy as np
from collections import defaultdict


def get_fold_number(path: str) -> int:
    """Extract fold number from directory name."""
    return int(path.split("fold")[1].split("_")[0])


def analyze_folds(base_dir: str) -> dict[int, dict[str, dict[str, float | int]]]:
    """Analyze results across folds and return statistics."""
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
            # Get both s2s and s2t test losses
            if "s2s_test_loss_mean" in results:
                results_by_fold[fold_num]["s2s"].append(results["s2s_test_loss_mean"])
            if "s2t_test_loss_mean" in results:
                results_by_fold[fold_num]["s2t"].append(results["s2t_test_loss_mean"])

    # Calculate statistics
    stats = {}
    for fold_num, metrics in results_by_fold.items():
        stats[fold_num] = {
            "s2s": {"mean": np.mean(metrics["s2s"]), "std": np.std(metrics["s2s"]), "n_runs": len(metrics["s2s"])},
            "s2t": {"mean": np.mean(metrics["s2t"]), "std": np.std(metrics["s2t"]), "n_runs": len(metrics["s2t"])},
        }

    return stats


if __name__ == "__main__":
    base_dir = "benchmark_runs/tg80_egno_mt"
    stats = analyze_folds(base_dir)

    print("\nResults by fold:")
    print("-" * 70)
    for fold_num in sorted(stats.keys()):
        print(f"Fold {fold_num}:")
        print(f"  S2S Test Loss:")
        print(f"    Mean: {stats[fold_num]['s2s']['mean']:.4f}")
        print(f"    Std:  {stats[fold_num]['s2s']['std']:.4f}")
        print(f"    N runs: {stats[fold_num]['s2s']['n_runs']}")
        print(f"  S2T Test Loss:")
        print(f"    Mean: {stats[fold_num]['s2t']['mean']:.4f}")
        print(f"    Std:  {stats[fold_num]['s2t']['std']:.4f}")
        print(f"    N runs: {stats[fold_num]['s2t']['n_runs']}")
        print("-" * 70)
