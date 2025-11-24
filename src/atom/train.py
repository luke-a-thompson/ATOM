from pathlib import Path
from datetime import datetime

from atom.training import (
    Config,
    parse_train_args,
    set_environment_variables,
    singletask_benchmark,
    multitask_benchmark,
    get_config_files,
    show_connectivity,
)


def main() -> None:
    args = parse_train_args()
    invocation_timestamp: str = datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
    if args.config:
        single_config_path: Path = Path(args.config).expanduser().resolve()
        config = Config.from_toml(single_config_path)

        if getattr(args, "show_connectivity", False):
            show_connectivity(config)
            return

        set_environment_variables(config)

        # Preserve directory structure under benchmark_runs
        base_dir_name: str = single_config_path.parent.name
        config_stem: str = single_config_path.stem
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

            if getattr(args, "show_connectivity", False):
                show_connectivity(config)
                continue

            set_environment_variables(config)

            try:
                relative_path: Path = Path(config_path).resolve().relative_to(base_configs_dir)
            except Exception:
                relative_path = Path(Path(config_path).name)

            base_benchmark_root: Path = Path("benchmark_runs") / f"{base_configs_dir.name}_{invocation_timestamp}"
            if relative_path.suffix:
                rel_config_stem: str = relative_path.stem
                rel_parent_dir: Path = relative_path.parent
            else:
                rel_config_stem = Path(relative_path).stem
                rel_parent_dir = Path("")

            rel_experiment_dir: Path = (base_benchmark_root / rel_parent_dir / f"{rel_config_stem}_{invocation_timestamp}").resolve()

            try:
                if config.dataloader.multitask:
                    multitask_benchmark(config, config_path, rel_experiment_dir)
                else:
                    singletask_benchmark(config, config_path, rel_experiment_dir)
            except Exception as e:
                print(f"Error running config {config_path}: {e}")
    else:
        raise ValueError("No config file or directory provided")


if __name__ == "__main__":
    main()
