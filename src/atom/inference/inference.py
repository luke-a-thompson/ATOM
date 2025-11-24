from atom.inference.inference_utils import (
    parse_inference_args,
    clean_state_dict_prefixes,
    parse_model_config_pairs,
    InferenceRunResult,
    MultiInferenceResults,
)
from atom.training import (
    Config,
    eval_epoch,
    create_dataloaders_single,
    create_dataloaders_multitask,
)
import torch
from atom.training import initialize_model
from collections import OrderedDict
from pathlib import Path
import time


def run_single_inference(model_path: str, config_path: str) -> InferenceRunResult:
    """Run inference on a single model/config pair."""
    start_time = time.time()

    try:
        config = Config.from_toml(Path(config_path))
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {config_path} not found")

    try:
        model_state_dict: OrderedDict[str, torch.Tensor] = torch.load(
            str(model_path), weights_only=True
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file {model_path} not found")

    if config.dataloader.multitask:
        test_loader = create_dataloaders_multitask(config)[2]
        molecule_type = str(config.dataloader.test_molecules)
        inference_type = "multitask"
    else:
        test_loader = create_dataloaders_single(config)[2]
        molecule_type = str(config.dataloader.molecule_type)
        inference_type = "single task"

    model = initialize_model(config).to(config.training.device)
    clean_model_state_dict = clean_state_dict_prefixes(model_state_dict)
    _ = model.load_state_dict(clean_model_state_dict)
    _ = model.eval()

    test_s2t_loss, test_s2s_loss = eval_epoch(config, model, test_loader)

    latency = time.time() - start_time

    return InferenceRunResult(
        s2t_test_loss=test_s2t_loss,
        s2s_test_loss=test_s2s_loss,
        latency=latency,
        model_path=Path(model_path),
        config_path=Path(config_path),
        molecule_type=molecule_type,
        inference_type=inference_type,
    )


def main() -> None:
    args = parse_inference_args()
    model_config_pairs = parse_model_config_pairs(args)

    print(f"Running inference on {len(model_config_pairs)} model/config pair(s)...")
    print("=" * 80)

    results = []
    for i, (model_path, config_path) in enumerate(model_config_pairs, 1):
        print(f"Processing run {i}/{len(model_config_pairs)}: {Path(model_path).name}")

        result = run_single_inference(model_path, config_path)
        results.append(result)

    # Display results
    print("\n" + "=" * 80)
    if len(results) > 1:
        multi_results = MultiInferenceResults(run_results=results)

        print("SUMMARY STATISTICS:")
        print("=" * 80)
        print(f"Number of runs: {len(results)}")
        print(
            f"Molecule type: {results[0].molecule_type} ({results[0].inference_type})"
        )
        print(
            f"S2S Test Loss: {multi_results.s2s_mean * 100:.2f}x10^-2 ± {multi_results.s2s_std * 100:.2f}x10^-2"
        )
        print(
            f"S2T Test Loss: {multi_results.s2t_mean * 100:.2f}x10^-2 ± {multi_results.s2t_std * 100:.2f}x10^-2"
        )
        print(
            f"Latency: {multi_results.latency_mean:.2f}s ± {multi_results.latency_std:.2f}s"
        )

        print("\nIndividual results:")
        for i, result in enumerate(results, 1):
            print(
                f"  Run {i} ({result.model_path.name}): S2T={result.s2t_test_loss * 100:.2f}x10^-2, S2S={result.s2s_test_loss * 100:.2f}x10^-2, Latency={result.latency:.2f}s"
            )
    else:
        result = results[0]
        print("RESULTS:")
        print("=" * 80)
        print(f"Model: {result.model_path.name}")
        print(f"Molecule type: {result.molecule_type} ({result.inference_type})")
        print(f"S2S Test Loss: {result.s2s_test_loss * 100:.2f}x10^-2")
        print(f"S2T Test Loss: {result.s2t_test_loss * 100:.2f}x10^-2")
        print(f"Latency: {result.latency:.2f}s")


if __name__ == "__main__":
    main()
