import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from rtmdet_object_detection_dev.dataclasses.bbox_label_container import (
    BBoxLabelContainer,
)
from rtmdet_object_detection_dev.datasets.oxford_pets_dataset import OxfordPetDataset


def get_dataset(name: str, config: dict) -> Dataset:
    """Return the chosen dataset.

    ## Args
    - name: name of the dataset
    - config: dict that is unpacked when initializing the dataset.
    """
    if name == "OxfordPetDataset":
        return OxfordPetDataset(**config)

    msg = f"No Implementation for dataset '{name}'"
    raise ValueError(msg)


def _custom_collate(
    batch: list[tuple[torch.Tensor, BBoxLabelContainer]],
) -> tuple[torch.Tensor, list[BBoxLabelContainer]]:
    images, bbox_label_container = zip(*batch, strict=True)
    images = torch.stack(images)
    return images, list(bbox_label_container)


def get_dataloader(dataset_configs: list[dict], batch_size: int = 8, *, shuffle: bool = True) -> DataLoader:
    """Return a dataloader built with the config.

    Factory method to construct a dataloader that loads data from potentially multiple
    datasets. All of the datasets must return items in form tensor, BBoxLabelContainer.

    # Args:
    - dataset_configs: list of configs for the chosen datasets.
    - batch_size: size of batches for the dataloader.
    - shuffle: if true data comes in random order.
    """
    datasets = [get_dataset(**config) for config in dataset_configs]
    dataset = ConcatDataset(datasets)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_custom_collate)
