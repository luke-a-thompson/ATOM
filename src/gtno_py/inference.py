from gtno_py.inference import parse_inference_args
from gtno_py.training import Config, eval_epoch
import torch


def main() -> None:
    args = parse_inference_args()
    try:
        config = Config.from_toml(args.config)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {args.config} not found")
    try:
        model = torch.load(args.model)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file {args.model} not found")

    test_loader = create_dataloaders_single(config)

    test_s2t_loss, test_s2s_loss = eval_epoch(config, model, test_loader)


if __name__ == "__main__":
    main()
