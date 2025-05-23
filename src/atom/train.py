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
    if args.config:
        config = Config.from_toml(args.config)
        _ = wandb.init(project="ATOM", name=config.benchmark.benchmark_name, config=dict(config), mode="disabled" if not config.wandb.use_wandb else "online")
        set_environment_variables(config)

        if config.dataloader.multitask:
            multitask_benchmark(config)
        else:
            singletask_benchmark(config)
    elif args.configs:
        for config_path in get_config_files(args.configs):
            try:
                config = Config.from_toml(config_path)
            except Exception as e:
                raise ValueError(f"Error loading config from {config_path}: {e}")
            _ = wandb.init(project="ATOM", name=config.benchmark.benchmark_name, config=dict(config), mode="disabled" if not config.wandb.use_wandb else "online")
            set_environment_variables(config)

            if config.dataloader.multitask:
                multitask_benchmark(config)
            else:
                singletask_benchmark(config)
    else:
        raise ValueError("No config file or directory provided")


if __name__ == "__main__":
    main()
