import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.dataclasses.bbox_label_container import BBoxLabelContainer
from src.datasets.oxford_pets_dataset import OxfordPetDataset


def get_dataset(name: str, config: dict) -> Dataset:
    """
    ## Args
    - name: name of the dataset
    - config: dict that is unpacked when initializing the dataset.
    """
    if name == "OxfordPetDataset":
        return OxfordPetDataset(**config)

    raise ValueError(f"No implementation for dataset '{name}'")


def _custom_collate(
    batch: list[tuple[torch.Tensor, BBoxLabelContainer]],
) -> tuple[torch.Tensor, list[BBoxLabelContainer]]:
    images, bbox_label_container = zip(*batch)
    images = torch.stack(images)
    return images, list(bbox_label_container)


def get_dataloader(
    dataset_configs: list[dict], batch_size: int = 8, shuffle: bool = True
) -> DataLoader:
    """
    Factory method to construct a dataloader that loads data from potentially multiple
    datasets. All of the datasets must return items in form tensor, BBoxLabelContainer.

    # Args:
    - dataset_configs: list of configs for the chosen datasets.
    - batch_size: size of batches for the dataloader.
    - shuffle: if true data comes in random order.
    """
    datasets = []
    for config in dataset_configs:
        datasets.append(get_dataset(**config))

    dataset = ConcatDataset(datasets)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_custom_collate
    )

    return loader
