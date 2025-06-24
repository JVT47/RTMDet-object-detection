"""A script that is used to convert trained RTMDet models to ONNX format.

Example usage:
uv run -m scripts.convert_to_onnx --model-name RTMDetM --num_classes 80 --model-weights model_weights/RTMDetM-coco.pth
--output-name RTMDetM-coco
"""

from argparse import ArgumentParser
from pathlib import Path

import torch

from rtmdet_object_detection_dev.model.model import make_model


def main() -> None:
    """Run script."""
    parser = ArgumentParser(description="Script that converts a given RTMDet model to onnx.")

    parser.add_argument(
        "--model-name",
        required=True,
        type=str,
        help="Name of the model. E.g., RTMDetTiny",
    )
    parser.add_argument(
        "--num_classes",
        required=True,
        type=int,
        help="The number of classes the model predicts",
    )
    parser.add_argument(
        "--model-weights",
        required=True,
        type=str,
        help="Path to the model weights file",
    )
    parser.add_argument("--output-name", required=True, type=str, help="Name of the produced onnx file")

    args = parser.parse_args()

    model = make_model(
        args.model_name,
        args.num_classes,
        args.model_weights,
        eval_mode=True,
        raw_output=True,
    )

    dummy_input = torch.zeros((1, 3, 640, 640))

    save_dir_path = Path("model_weights", "onnx")
    save_dir_path.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_input,),
        f=save_dir_path.joinpath(f"{args.output_name}.onnx"),
        dynamo=False,
        optimize=True,
        input_names=["x"],
        dynamic_axes={"x": [0, 2, 3]},
    )


if __name__ == "__main__":
    main()
