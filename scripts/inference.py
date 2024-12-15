"""
A script that reads images from a directory, draws model detections to them, and saves them to a directory.
"""
from itertools import islice
import os
from pathlib import Path
import torch
import torchvision
from typing import Generator, Iterable

from src.dataclasses.detection_result import DetectionResult
from src.model.model import make_model
from src.processors.rtmdet_preprocessor import RTMDetPreprocessor
from src.processors.rtmdet_postprocessor import RTMDetPostprocessor


def batch_image_files(image_files: Iterable[str], batch_size: int) -> Generator[list[str], None, None]:
    """
    Generator that yields a batch of image file paths. 
    """
    iterator = iter(image_files)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def preprocess_images_to_batch(images: Iterable[torch.Tensor], preprocessor: RTMDetPreprocessor) -> torch.Tensor:
    """
    Preprocesses all the given input images and combines them into one tensor.
    """
    processed_images = map(lambda image: preprocessor.process_image(image), images)

    return torch.stack(list(processed_images), dim=0)


def draw_bboxes_to_original_image(image: torch.Tensor, det_result: DetectionResult, input_size: torch.Size) -> torch.Tensor:
    """
    Draws the detected bboxes to the original image. Transforms the bbox coordinates relative to the original
    dimensions if needed. 
    """
    bboxes = RTMDetPostprocessor.bbox_to_original_image(det_result.bboxes, input_size, image.shape)
    labels = [f"{label.item()}, {score.item():.2f}" for label, score in zip(det_result.classes, det_result.scores)]
    
    image = torchvision.utils.draw_bounding_boxes(image, bboxes, labels, colors=(255, 255, 0), width=2)

    return image


def main() -> None:
    input_dir = "images"
    output_dir = f"{input_dir}/results"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    batch_size = 4

    model_weights = "model_weights/RTMDetL-original-weights.pth"
    model = make_model(model_name="RTMDetL", num_classes=80, model_weights=model_weights, eval=True)
    preprocessor = RTMDetPreprocessor(dest_size=(448, 800))
    postprocessor = RTMDetPostprocessor()

    image_files = (f"{input_dir}/{file}" for file in os.listdir(input_dir) if file.endswith(".jpg") or file.endswith(".png"))

    with torch.inference_mode():
        for image_file_batch in batch_image_files(image_files, batch_size):
            images = list(map(lambda file_path: torchvision.io.read_image(file_path).float(), image_file_batch))

            input_batch = preprocess_images_to_batch(images, preprocessor)
            model_output = model(input_batch)
            det_results = postprocessor.process_batch(model_output)

            image_file_names = [file.rsplit("/", maxsplit=1)[-1] for file in image_file_batch]
            for file_name, image, det_result in zip(image_file_names, images, det_results):
                image = image / 255
                image = draw_bboxes_to_original_image(image, det_result, input_size=preprocessor.dest_size)
                torchvision.utils.save_image(image, f"{output_dir}/{file_name}")


if __name__ == "__main__":
    main()