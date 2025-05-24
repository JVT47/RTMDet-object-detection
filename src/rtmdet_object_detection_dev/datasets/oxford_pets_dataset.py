from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.io.image import decode_image
import yaml

from rtmdet_object_detection_dev.dataclasses.bbox_label_container import (
    BBoxLabelContainer,
)
from rtmdet_object_detection_dev.processors.rtmdet_preprocessor import (
    RTMDetPreprocessor,
)


class OxfordPetDataset(Dataset):
    def __init__(
        self,
        annotations_file_path: Path,
        image_dir_path: Path,
        preprocessor_config: dict,
        num_classes: int = 37,
    ) -> None:
        """
        ## Args
        - annotation_file_path: Path to the yaml file that holds the image annotations.
        - image_dir_path: Path to the dir that contains the dataset images.
        - preprocessor_config: dict that is unpacked when initializing the RTMDetPreprocessor.
        - num_classes: number of classes in the dataset.
        """
        super().__init__()

        with open(annotations_file_path, "r") as f:
            self.annotations = yaml.safe_load(f)["annotations"]

        self.image_dir_path = image_dir_path
        self.preprocessor = RTMDetPreprocessor(**preprocessor_config)
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, BBoxLabelContainer]:
        annotation = self.annotations[index]

        image_path = self.image_dir_path.joinpath(annotation["filename"])
        image = decode_image(str(image_path)).float()

        gt = self._get_bbox_label_container(annotation)
        gt.bboxes = self.preprocessor.process_bboxes(gt.bboxes, image.shape)

        image = self.preprocessor.process_image(image)

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
                    ]
                )
            )
            label_id = object_annotation["breed_id"]
            labels.append(
                torch.nn.functional.one_hot(torch.tensor(label_id), self.num_classes)
            )

        return BBoxLabelContainer(torch.stack(bboxes), torch.stack(labels))
