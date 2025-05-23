import subprocess
import re
from statistics import mean, stdev
import torch


def run_inference_command(model_path: str, config_path: str) -> float:
    """
    Run a single inference command using poetry and return the Test S2T loss.

    Args:
        model_path: Path to the model file
        config_path: Path to the config file

    Returns:
        float: The Test S2T loss value
    """
    cmd = f"poetry run inference --model {model_path} --config {config_path}"
    print(f"Running: {cmd}")

    # Run the command and capture output
    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

    # Extract Test S2T loss using regex
    match = re.search(r"Test S2T loss: (\d+\.\d+)", result.stdout)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Could not find Test S2T loss in output for {model_path}")


def compute_statistics(losses: list[float]) -> tuple[float, float]:
    """
    Compute mean and 2-sigma standard deviation of losses.

    Args:
        losses: List of loss values

    Returns:
        tuple: (mean, 2*std_dev)
    """
    if not losses:
        return 0.0, 0.0
    return mean(losses), 2 * stdev(losses)


def run_all_inferences() -> None:
    """
    Run all inference commands for different molecules and compute statistics.
    """
    # Define the base paths
    base_model_path = "benchmark_runs/tg80_atom_mt"
    base_config_path = "configs/tg80_multitask_for_inference"

    # Define all inference runs
    inference_runs: dict[str, list[dict[str, str]]] = {
        "Uracil": [
            {
                "model": f"{base_model_path}/atom_tg80_multitask_muon_fold1_multitask_15-May-2025_09-36-35/run_1/best_val_model.pth",
                "config": f"{base_config_path}/atom_multitask_muon_fold1.toml",
            },
            {
                "model": f"{base_model_path}/atom_tg80_multitask_muon_fold1_multitask_15-May-2025_12-39-07/run_1/best_val_model.pth",
                "config": f"{base_config_path}/atom_multitask_muon_fold1.toml",
            },
            {
                "model": f"{base_model_path}/atom_tg80_multitask_muon_fold1_multitask_15-May-2025_12-39-07/run_2/best_val_model.pth",
                "config": f"{base_config_path}/atom_multitask_muon_fold1.toml",
            },
        ],
        "Nitrobenzene": [
            {
                "model": f"{base_model_path}/atom_tg80_multitask_muon_fold2_multitask_15-May-2025_10-31-21/run_1/best_val_model.pth",
                "config": f"{base_config_path}/atom_multitask_muon_fold2.toml",
            },
            {
                "model": f"{base_model_path}/atom_tg80_multitask_muon_fold2_multitask_15-May-2025_13-18-52/run_1/best_val_model.pth",
                "config": f"{base_config_path}/atom_multitask_muon_fold2.toml",
            },
            {
                "model": f"{base_model_path}/atom_tg80_multitask_muon_fold2_multitask_15-May-2025_13-18-52/run_2/best_val_model.pth",
                "config": f"{base_config_path}/atom_multitask_muon_fold2.toml",
            },
        ],
        "Indole": [
            {
                "model": f"{base_model_path}/atom_tg80_multitask_muon_fold3_multitask_15-May-2025_11-20-46/run_1/best_val_model.pth",
                "config": f"{base_config_path}/atom_multitask_muon_fold3.toml",
            },
            {
                "model": f"{base_model_path}/atom_tg80_multitask_muon_fold3_multitask_15-May-2025_14-10-00/run_1/best_val_model.pth",
                "config": f"{base_config_path}/atom_multitask_muon_fold3.toml",
            },
            {
                "model": f"{base_model_path}/atom_tg80_multitask_muon_fold3_multitask_15-May-2025_14-10-00/run_2/best_val_model.pth",
                "config": f"{base_config_path}/atom_multitask_muon_fold3.toml",
            },
        ],
        "Napthalene": [
            {
                "model": f"{base_model_path}/atom_tg80_multitask_muon_fold4_multitask_15-May-2025_12-08-25/run_1/best_val_model.pth",
                "config": f"{base_config_path}/atom_multitask_muon_fold4.toml",
            },
            {
                "model": f"{base_model_path}/atom_tg80_multitask_muon_fold4_multitask_15-May-2025_12-39-39/run_1/best_val_model.pth",
                "config": f"{base_config_path}/atom_multitask_muon_fold4.toml",
            },
            {
                "model": f"{base_model_path}/atom_tg80_multitask_muon_fold4_multitask_15-May-2025_12-39-39/run_2/best_val_model.pth",
                "config": f"{base_config_path}/atom_multitask_muon_fold4.toml",
            },
        ],
        "Butanol": [
            {
                "model": f"{base_model_path}/atom_tg80_multitask_muon_fold5_multitask_15-May-2025_12-40-10/run_1/best_val_model.pth",
                "config": f"{base_config_path}/atom_multitask_muon_fold5.toml",
            },
            {
                "model": f"{base_model_path}/atom_tg80_multitask_muon_fold5_multitask_15-May-2025_12-40-10/run_2/best_val_model.pth",
                "config": f"{base_config_path}/atom_multitask_muon_fold5.toml",
            },
        ],
    }

    # Store results for each molecule
    results: dict[str, list[float]] = {}

    # Run all inferences
    for molecule, runs in inference_runs.items():
        print(f"\nRunning inferences for {molecule}")
        print("=" * 50)

        molecule_losses: list[float] = []
        for i, run in enumerate(runs, 1):
            torch.manual_seed(i + 50)
            print(f"\nRun {i} for {molecule}")
            print("-" * 30)
            try:
                loss = run_inference_command(run["model"], run["config"])
                molecule_losses.append(loss)
                print(f"Test S2T loss: {loss:.2f}")
            except Exception as e:
                print(f"Error in run {i} for {molecule}: {str(e)}")

        results[molecule] = molecule_losses

    # Print statistics for each molecule
    print("\n\nStatistics Summary")
    print("=" * 50)
    for molecule, losses in results.items():
        if losses:
            mean_loss, two_sigma = compute_statistics(losses)
            print(f"\n{molecule}:")
            print(f"Mean Test S2T loss: {mean_loss:.2f}")
            print(f"2-sigma std dev: {two_sigma:.2f}")
            print(f"Individual losses: {[f'{loss:.2f}' for loss in losses]}")


if __name__ == "__main__":
    run_all_inferences()
