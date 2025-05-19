from argparse import ArgumentParser
from pathlib import Path

from src.config_loader import load_yaml_file
from src.training.train import train_model
from src.training.training_config import TrainingConfig


def main() -> None:
    parser = ArgumentParser()

    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to the model training config file",
    )

    args = parser.parse_args()

    training_config_path = Path(args.config_file)

    if not training_config_path.exists() or training_config_path.suffix != ".yaml":
        raise RuntimeError(
            f"Did not find config yaml file in path: {training_config_path}"
        )

    training_config = load_yaml_file(training_config_path)
    training_config = TrainingConfig(**training_config["train_config"])
    train_model(training_config)


if __name__ == "__main__":
    main()
