import wandb

from gtno_py.training import (
    Config,
    set_seeds,
    parse_args,
    set_environment_variables,
    singletask_benchmark,
    multitask_benchmark,
)


def main() -> None:
    args = parse_args()
    config = Config.from_toml(args.config_path)
    project_name = input("Enter project name: ")
    _ = wandb.init(project="GTNO", name=project_name, config=dict(config), mode="disabled" if not config.wandb.use_wandb else "online")
    set_seeds(config.training.seed)
    set_environment_variables(config)

    if config.dataloader.multitask:
        multitask_benchmark(config)
    else:
        singletask_benchmark(config)


if __name__ == "__main__":
    main()
