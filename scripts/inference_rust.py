"""
A script that reads images from the images directory, draws model detections to them, and saves
the results to images/results directory. Uses the rust based rtmdet-object-detection package and
a given onnx model to do the inference.

Example usage:
    uv run -m scripts.inference_rust --model-path path/to/model/onnx
"""

from argparse import ArgumentParser, Namespace
from itertools import chain
from pathlib import Path
import torch
import torchvision

from rtmdet_object_detection_dev.inference.utils import batch_image_files

try:
    import rtmdet_object_detection
except ImportError:
    raise ImportError(
        "The Rust extension 'rtmdet_object_detection' is not built."
        "Please run 'maturin develop -r' in src/rtmdet-object-detection before running this script"
    )

INFERENCE_SHAPE = 640, 640
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.3
PADDING_COLOR = 114, 114, 114
COLOR_MEAN = 103.53, 116.28, 123.675
COLOR_STD = 57.375, 57.12, 58.395


def draw_bboxes_to_original_image(
    image: torch.Tensor, det_result: rtmdet_object_detection.DetectionOutput
):
    """
    Draws each bbox in det_result to the given image in yellow. Labels each bbox with its label and detection score.
    """
    bboxes = torch.zeros(len(det_result.bboxes), 4)
    labels = []
    for i, bbox in enumerate(det_result.bboxes):
        bboxes[i] = torch.tensor([*bbox.top_left, *bbox.bottom_right])
        labels.append(f"{bbox.class_num}, {bbox.score:.2f}")

    image = torchvision.utils.draw_bounding_boxes(
        image, bboxes, labels, colors=(255, 255, 0), width=2
    )

    return image


def get_arguments() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        default="model_weights/onnx/RTMDetM-coco.onnx",
        help="Path to the onnx model file. If not present model_weights/onnx/RTMDetM-coco.onnx will be used as default.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        default=4,
        help="Batch size used in inference. Default: 4.",
    )

    args = parser.parse_args()

    if not Path(args.model_path).exists():
        raise ValueError(f"Did not find the given model path: {args.model_path}")

    return args


def main() -> None:
    input_dir = Path("images")
    output_dir = input_dir.joinpath("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    args = get_arguments()

    detector = rtmdet_object_detection.RTMDetDetector(
        args.model_path,
        INFERENCE_SHAPE,
        args.batch_size,
        SCORE_THRESHOLD,
        IOU_THRESHOLD,
        PADDING_COLOR,
        COLOR_MEAN,
        COLOR_STD,
    )

    image_file_generator = chain(input_dir.glob("*.jpg"), input_dir.glob("*.png"))

    for image_file_batch in batch_image_files(image_file_generator, args.batch_size):
        images = list(
            map(
                lambda file_path: torchvision.io.read_image(str(file_path)),
                image_file_batch,
            )
        )
        images_np = [image.permute(1, 2, 0).numpy() for image in images]

        detection_results = detector.detect_from_numpy(images_np)

        image_filenames = [file.name for file in image_file_batch]
        for filename, image, det_result in zip(
            image_filenames, images, detection_results
        ):
            image = image / 255
            image = draw_bboxes_to_original_image(image, det_result)
            torchvision.utils.save_image(image, output_dir.joinpath(filename))


if __name__ == "__main__":
    main()
