"""Starts the model training based on the given config file.

Example usage:
- uv run -m scripts.train_model --config-file path/to/config
"""

from argparse import ArgumentParser
from pathlib import Path

from rtmdet_object_detection_dev.config_loader import load_yaml_file
from rtmdet_object_detection_dev.training.train import train_model
from rtmdet_object_detection_dev.training.training_config import TrainingConfig


def main() -> None:
    """Run training."""
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
        msg = f"Did not find config yaml file in path: {training_config_path}"
        raise RuntimeError(msg)

    training_config = load_yaml_file(training_config_path)
    training_config = TrainingConfig(**training_config["train_config"])
    train_model(training_config)


if __name__ == "__main__":
    main()
