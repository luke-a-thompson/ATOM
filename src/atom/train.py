from pathlib import Path
from datetime import datetime
import wandb

from atom.training import (
    Config,
    parse_train_args,
    set_environment_variables,
    singletask_benchmark,
    multitask_benchmark,
    get_config_files,
)


def main() -> None:
    args = parse_train_args()
    invocation_timestamp: str = datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
    if args.config:
        single_config_path: Path = Path(args.config).expanduser().resolve()
        config = Config.from_toml(single_config_path)
        _ = wandb.init(project="ATOM", name=config.benchmark.benchmark_name, config=dict(config), mode="disabled" if not config.wandb.use_wandb else "online")
        set_environment_variables(config)

        # Preserve directory structure under benchmark_runs
        base_dir_name: str = single_config_path.parent.name
        config_stem: str = single_config_path.stem
        # Base directory for this invocation
        base_benchmark_dir: Path = Path("benchmark_runs") / f"{base_dir_name}_{invocation_timestamp}"
        experiment_dir: Path = base_benchmark_dir / f"{config_stem}_{invocation_timestamp}"

        if config.dataloader.multitask:
            multitask_benchmark(config, single_config_path, experiment_dir)
        else:
            singletask_benchmark(config, single_config_path, experiment_dir)
    elif args.configs:
        base_configs_dir: Path = Path(args.configs).expanduser().resolve()
        for config_path in get_config_files(args.configs):
            try:
                config = Config.from_toml(config_path)
            except Exception as e:
                raise ValueError(f"Error loading config from {config_path}: {e}")
            _ = wandb.init(project="ATOM", name=config.benchmark.benchmark_name, config=dict(config), mode="disabled" if not config.wandb.use_wandb else "online")
            set_environment_variables(config)

            # Compute relative path to preserve structure under the provided base directory
            try:
                relative_path: Path = Path(config_path).resolve().relative_to(base_configs_dir)
            except Exception:
                # fallback: filename only, but keep Path type
                relative_path = Path(Path(config_path).name)

            # Build directories and copy config
            base_benchmark_root: Path = Path("benchmark_runs") / f"{base_configs_dir.name}_{invocation_timestamp}"
            if relative_path.suffix:
                rel_config_stem: str = relative_path.stem
                rel_parent_dir: Path = relative_path.parent
            else:
                rel_config_stem = Path(relative_path).stem
                rel_parent_dir = Path("")

            rel_experiment_dir: Path = (base_benchmark_root / rel_parent_dir / f"{rel_config_stem}_{invocation_timestamp}").resolve()

            if config.dataloader.multitask:
                multitask_benchmark(config, config_path, rel_experiment_dir)
            else:
                singletask_benchmark(config, config_path, rel_experiment_dir)
    else:
        raise ValueError("No config file or directory provided")


if __name__ == "__main__":
    main()
