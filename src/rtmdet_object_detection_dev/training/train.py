import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from rtmdet_object_detection_dev.model.model import make_model
from rtmdet_object_detection_dev.dataclasses.bbox_label_container import (
    BBoxLabelContainer,
)
from rtmdet_object_detection_dev.datasets.dataset_factory import get_dataloader
from rtmdet_object_detection_dev.losses.loss_fn_factory import get_loss_fn
from rtmdet_object_detection_dev.training.optimizer_factory import get_optimizer
from rtmdet_object_detection_dev.training.training_config import TrainingConfig


def gts_to_device(
    gts: list[BBoxLabelContainer], device: torch.device
) -> list[BBoxLabelContainer]:
    return [
        BBoxLabelContainer(gt.bboxes.to(device), gt.labels.to(device)) for gt in gts
    ]


def train_one_epoch(
    model: nn.Module,
    training_dataloader: DataLoader,
    optimizer: Optimizer,
    rtmdet_loss: nn.Module,
    device: torch.device,
) -> float:
    """
    Trains the model with the training data and returns the mean training loss.
    """
    model.train()
    running_loss = 0.0

    for data in training_dataloader:
        optimizer.zero_grad()

        images, gts = data[0].to(device), gts_to_device(data[1], device)

        rtmdet_output = model(images)

        loss: torch.Tensor = rtmdet_loss(rtmdet_output, gts)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(training_dataloader)


def validate(
    model: nn.Module,
    validation_dataloader: DataLoader,
    rtmdet_loss: nn.Module,
    device: torch.device,
) -> float:
    """
    Calculates the mean loss for the validation set.
    """
    with torch.inference_mode():
        model.eval()
        running_loss = 0.0

        for data in validation_dataloader:
            images, gts = data[0].to(device), gts_to_device(data[1], device)

            rtmdet_output = model(images)

            loss: torch.Tensor = rtmdet_loss(rtmdet_output, gts)
            running_loss += loss.item()

        return running_loss / len(validation_dataloader)


def train_model(training_config: TrainingConfig) -> None:
    """
    ## Args
        - training_config: dict containing the required fields to configure model training
    """
    model = make_model(**training_config.model_cfg)
    device = torch.device(training_config.device)
    model.to(device)

    training_dataloader = get_dataloader(**training_config.training_dataloader_config)
    validation_dataloader = get_dataloader(
        **training_config.validation_dataloader_config
    )

    loss = get_loss_fn(**training_config.loss_fn_config)
    optimizer = get_optimizer(model, **training_config.optimizer_config)

    best_validation_loss = float("inf")
    for i in range(training_config.epochs):
        print(f"Epoch {i + 1} / {training_config.epochs}")

        training_mean_loss = train_one_epoch(
            model, training_dataloader, optimizer, loss, device
        )
        validation_mean_loss = validate(model, validation_dataloader, loss, device)

        print(
            f"Training loss: {training_mean_loss}, validation loss: {validation_mean_loss}"
        )

        if validation_mean_loss < best_validation_loss:
            best_validation_loss = validation_mean_loss
            weights_path = training_config.weights_save_path.joinpath(
                f"{training_config.session_name}.pth"
            )
            torch.save(model.state_dict(), weights_path)
