from atom.inference.inference_utils import parse_inference_args, clean_state_dict_prefixes
from atom.training import Config, eval_epoch, create_dataloaders_single, create_dataloaders_multitask
import torch
from atom.training import initialize_model
from collections import OrderedDict


def main() -> None:
    args = parse_inference_args()
    try:
        config = Config.from_toml(args.config)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {args.config} not found")

    try:
        model_state_dict: OrderedDict[str, torch.Tensor] = torch.load(str(args.model), weights_only=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file {args.model} not found")

    if config.dataloader.multitask:
        test_loader = create_dataloaders_multitask(config)[2]
    else:
        test_loader = create_dataloaders_single(config)[2]

    model = initialize_model(config).to(config.training.device)
    clean_model_state_dict = clean_state_dict_prefixes(model_state_dict)
    _ = model.load_state_dict(clean_model_state_dict)
    _ = model.eval()

    test_s2t_loss, test_s2s_loss = eval_epoch(config, model, test_loader)

    print(f"Test S2T loss: {test_s2t_loss*100:.2f}x10^-2")
    print(f"Test S2S loss: {test_s2s_loss*100:.2f}x10^-2")


if __name__ == "__main__":
    main()
