"""A script that reads images from images directory, draws model detections to them, and saves the results to images/results directory.

Uses the python implementation for inference with the given pytorch model weights.

Example usage:
    uv run -m scripts.inference_python --model-name RTMDetM --model-weights model_weights/RTMDetM-coco.pth
"""  # noqa: E501

from argparse import ArgumentParser, Namespace
from collections.abc import Iterable
from itertools import chain
from pathlib import Path

import torch
import torchvision

from rtmdet_object_detection_dev.dataclasses.detection_result import DetectionResult
from rtmdet_object_detection_dev.inference.utils import batch_image_files
from rtmdet_object_detection_dev.model.model import make_model
from rtmdet_object_detection_dev.processors.rtmdet_postprocessor import (
    RTMDetPostprocessor,
)
from rtmdet_object_detection_dev.processors.rtmdet_preprocessor import (
    RTMDetPreprocessor,
)

INFERENCE_SHAPE = 640, 640
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.3
PADDING_COLOR = 114, 114, 114
COLOR_MEAN = 103.53, 116.28, 123.675
COLOR_STD = 57.375, 57.12, 58.395


def preprocess_images_to_batch(images: Iterable[torch.Tensor], preprocessor: RTMDetPreprocessor) -> torch.Tensor:
    """Preprocess all the given input images and combines them into one tensor."""
    processed_images = (preprocessor.process_image(image) for image in images)

    return torch.stack(list(processed_images), dim=0)


def draw_bboxes_to_original_image(
    image: torch.Tensor,
    det_result: DetectionResult,
    input_size: torch.Size,
) -> torch.Tensor:
    """Draw the detected bboxes to the original image.

    Transforms the bbox coordinates relative to the original dimensions if needed.
    """
    bboxes = RTMDetPostprocessor.bbox_to_original_image(det_result.bboxes, input_size, image.shape)
    labels = [
        f"{label.item()}, {score.item():.2f}"
        for label, score in zip(det_result.classes, det_result.scores, strict=True)
    ]

    return torchvision.utils.draw_bounding_boxes(image, bboxes, labels, colors=(255, 255, 0), width=2)


def get_arguments() -> Namespace:
    """Get arguments passed to the command line."""
    parser = ArgumentParser()

    parser.add_argument(
        "--model-name",
        type=str,
        required=False,
        default="RTMDetM",
        help="Name of the model used. Should match with the given weights if provided. Default: RTMDetM.",
    )

    parser.add_argument(
        "--model-weights",
        type=str,
        required=False,
        default="model_weights/RTMDetM-coco.pth",
        help=(
            "Path to the model pytorch weights. If not present model_weights/RTMDetM-coco.pth will be used as default."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        default=4,
        help="Batch size used in inference. Default: 4.",
    )

    args = parser.parse_args()

    if not Path(args.model_weights).exists():
        msg = f"Did not find the given model weights: {args.model_weights}"
        raise ValueError(msg)

    return args


def main() -> None:
    """Run inference."""
    input_dir = Path("images")
    output_dir = input_dir.joinpath("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    args = get_arguments()

    model = make_model(
        model_name=args.model_name,
        num_classes=80,
        model_weights=args.model_weights,
        eval_mode=True,
    )
    preprocessor = RTMDetPreprocessor(
        dest_size=INFERENCE_SHAPE,
        pad_color=PADDING_COLOR,
        mean=COLOR_MEAN,
        std=COLOR_STD,
    )
    postprocessor = RTMDetPostprocessor(SCORE_THRESHOLD, IOU_THRESHOLD)

    image_file_generator = chain(input_dir.glob("*.jpg"), input_dir.glob("*.png"))

    with torch.inference_mode():
        for image_file_batch in batch_image_files(image_file_generator, args.batch_size):
            images = [torchvision.io.read_image(str(file_path)).float() for file_path in image_file_batch]

            input_batch = preprocess_images_to_batch(images, preprocessor)
            model_output = model(input_batch)
            det_results = postprocessor.process_batch(model_output)

            image_filenames = [file.name for file in image_file_batch]
            for filename, img, det_result in zip(image_filenames, images, det_results, strict=True):
                image = img / 255
                image = draw_bboxes_to_original_image(image, det_result, input_size=preprocessor.dest_size)
                torchvision.utils.save_image(image, output_dir.joinpath(filename))


if __name__ == "__main__":
    main()
