from pathlib import Path

import torch
import torchvision
import torchvision.tv_tensors
import yaml
from torch.utils.data import Dataset
from torchvision.transforms import v2

from rtmdet_object_detection_dev.dataclasses.bbox_label_container import (
    BBoxLabelContainer,
)
from rtmdet_object_detection_dev.processors.rtmdet_preprocessor import (
    RTMDetPreprocessor,
)


class OxfordPetDataset(Dataset):
    """Dataset implementation for the Oxford Pet Data."""

    def __init__(
        self,
        annotations_file_path: Path,
        image_dir_path: Path,
        preprocessor_config: dict,
        num_classes: int = 37,
        *,
        augment: bool = True,
    ) -> None:
        """Initialize the dataset.

        ## Args
        - annotation_file_path: Path to the yaml file that holds the image annotations.
        - image_dir_path: Path to the dir that contains the dataset images.
        - preprocessor_config: dict that is unpacked when initializing the RTMDetPreprocessor.
        - num_classes: number of classes in the dataset.
        - augment: augment images and bboxes with large scale jitter and random cropping.
        """
        super().__init__()

        with annotations_file_path.open() as f:
            self.annotations = yaml.safe_load(f)["annotations"]

        self.image_dir_path = image_dir_path
        self.preprocessor = RTMDetPreprocessor(**preprocessor_config)

        self.num_classes = num_classes
        self.augment = augment
        dest_size: tuple[int, int] = tuple(self.preprocessor.dest_size[-2:])  # type: ignore This produces the correct type

        self.transforms = v2.Compose(
            [
                v2.RandomHorizontalFlip(),
                v2.ScaleJitter(dest_size),
                v2.RandomCrop(dest_size, pad_if_needed=True, fill=self.preprocessor.pad_color),
            ],
        )

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.annotations)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, BBoxLabelContainer]:
        """Get an image and its bounding boxes."""
        annotation = self.annotations[index]

        image_path = self.image_dir_path.joinpath(annotation["filename"])
        image = torchvision.io.image.read_image(str(image_path), mode=torchvision.io.ImageReadMode.RGB).float()

        gt = self._get_bbox_label_container(annotation)
        bboxes = self.preprocessor.process_bboxes(gt.bboxes, image.shape)

        image = self.preprocessor.process_image(image)

        if not self.augment:
            return image, gt

        bboxes = torchvision.tv_tensors.BoundingBoxes(bboxes, format="XYXY", canvas_size=image.shape[-2:])  # type: ignore This works
        image, bboxes = self.transforms(image, bboxes)
        gt.bboxes = bboxes.data

        return image, gt

    def _get_bbox_label_container(self, annotation: dict) -> BBoxLabelContainer:
        bboxes = []
        labels = []
        for object_annotation in annotation["objects"]:
            bboxes.append(
                torch.tensor(
                    [
                        object_annotation["bbox"]["xmin"],
                        object_annotation["bbox"]["ymin"],
                        object_annotation["bbox"]["xmax"],
                        object_annotation["bbox"]["ymax"],
                    ],
                ),
            )
            label_id = object_annotation["breed_id"]
            labels.append(torch.nn.functional.one_hot(torch.tensor(label_id), self.num_classes))

        return BBoxLabelContainer(torch.stack(bboxes), torch.stack(labels))


if __name__ == "__main__":
    """Show random outputs of the dataset to test that it works."""
    import random

    import cv2
    from torchvision.utils import draw_bounding_boxes

    dataset = OxfordPetDataset(
        Path("data", "annotations", "train.yaml"),
        Path("data", "images"),
        preprocessor_config={
            "dest_size": (480, 480),
            "mean": (0, 0, 0),
            "std": (1, 1, 1),
        },
        augment=False,
    )

    indices = random.sample(range(len(dataset)), 20)

    for i in indices:
        image, gt = dataset.__getitem__(i)

        image = image.to(torch.uint8)
        image = draw_bounding_boxes(
            image,
            gt.bboxes,
            [str(label.argmax().item()) for label in gt.labels],
            colors=(0, 255, 255),
        )

        image = image.permute(1, 2, 0).numpy()

        cv2.imshow("", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
