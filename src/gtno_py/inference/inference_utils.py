import argparse


def parse_inference_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pretrained GTNO model")
    _ = parser.add_argument(
        "--model",
        type=str,
        help="Path to a pretrained model",
    )
    _ = parser.add_argument(
        "--config",
        type=str,
        help="Path to a config.toml file",
    )
    return parser.parse_args()