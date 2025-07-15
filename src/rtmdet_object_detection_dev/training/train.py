import logging

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from rtmdet_object_detection_dev.dataclasses.bbox_label_container import (
    BBoxLabelContainer,
)
from rtmdet_object_detection_dev.datasets.dataset_factory import get_dataloader
from rtmdet_object_detection_dev.losses.loss_fn_factory import get_loss_fn
from rtmdet_object_detection_dev.model.model import make_model
from rtmdet_object_detection_dev.training.lr_scheduler_factory import get_lr_scheduler
from rtmdet_object_detection_dev.training.optimizer_factory import get_optimizer
from rtmdet_object_detection_dev.training.training_config import TrainingConfig

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def gts_to_device(gts: list[BBoxLabelContainer], device: torch.device) -> list[BBoxLabelContainer]:
    """Move gts to the given device."""
    return [BBoxLabelContainer(gt.bboxes.to(device), gt.labels.to(device)) for gt in gts]


def train_one_epoch(  # noqa: PLR0913
    model: nn.Module,
    ema_model: torch.optim.swa_utils.AveragedModel,
    training_dataloader: DataLoader,
    optimizer: Optimizer,
    rtmdet_loss: nn.Module,
    device: torch.device,
) -> float:
    """Train the model with the training data and returns the mean training loss."""
    model.train()
    running_loss = 0.0

    for data in training_dataloader:
        optimizer.zero_grad()

        images, gts = data[0].to(device), gts_to_device(data[1], device)

        rtmdet_output = model(images)

        loss: torch.Tensor = rtmdet_loss(rtmdet_output, gts)
        loss.backward()

        optimizer.step()
        ema_model.update_parameters(model)

        running_loss += loss.item()

    return running_loss / len(training_dataloader)


def validate(
    model: nn.Module,
    validation_dataloader: DataLoader,
    rtmdet_loss: nn.Module,
    device: torch.device,
) -> float:
    """Calculate the mean loss for the validation set."""
    with torch.inference_mode():
        model.eval()
        running_loss = 0.0

        for data in validation_dataloader:
            images, gts = data[0].to(device), gts_to_device(data[1], device)

            rtmdet_output = model(images)

            loss: torch.Tensor = rtmdet_loss(rtmdet_output, gts)
            running_loss += loss.item()

        return running_loss / len(validation_dataloader)


def save_best_model(model: nn.Module, training_config: TrainingConfig) -> None:
    """Save the model to disk."""
    training_config.weights_save_path.mkdir(parents=True, exist_ok=True)
    weights_path = training_config.weights_save_path.joinpath(f"{training_config.session_name}.pth")
    torch.save(model.state_dict(), weights_path)


def is_substantial_increase(validation_loss: float, best_validation_loss: float, threshold: float) -> bool:
    """Measure if the validation loss has risen substantially from the best loss."""
    if best_validation_loss == float("inf"):
        return False

    eps = 1e-10
    if best_validation_loss < eps:
        return True

    relative_increase = (validation_loss - best_validation_loss) / best_validation_loss
    return relative_increase > threshold


def train_model(training_config: TrainingConfig) -> None:
    """Train the model according to the given config.

    ## Args
        - training_config: TrainingConfig containing the required fields to configure model training
    """
    model = make_model(**training_config.model_cfg)
    device = torch.device(training_config.device)
    model.to(device)

    ema_model = torch.optim.swa_utils.AveragedModel(
        model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(training_config.ema_decay),
    )

    training_dataloader = get_dataloader(**training_config.training_dataloader_config)
    validation_dataloader = get_dataloader(**training_config.validation_dataloader_config)

    loss = get_loss_fn(**training_config.loss_fn_config)
    optimizer = get_optimizer(model, **training_config.optimizer_config)

    lr_scheduler = get_lr_scheduler(optimizer, **training_config.lr_scheduler_config)

    best_validation_loss = float("inf")
    epochs_since_improvement = 0
    for i in range(training_config.epochs):
        logger.info("Epoch %s / %s", i + 1, training_config.epochs)

        training_mean_loss = train_one_epoch(model, ema_model, training_dataloader, optimizer, loss, device)
        validation_mean_loss = validate(ema_model, validation_dataloader, loss, device)
        lr_scheduler.step()

        logger.info("Training loss: %s, validation loss: %s", training_mean_loss, validation_mean_loss)

        is_best_loss = validation_mean_loss < best_validation_loss
        if is_best_loss:
            best_validation_loss = validation_mean_loss
            epochs_since_improvement = 0
            save_best_model(ema_model.module, training_config)
            continue

        if is_substantial_increase(
            validation_mean_loss,
            best_validation_loss,
            training_config.early_stopping_threshold,
        ):
            epochs_since_improvement += 1
            logger.info("Validation loss increased substantially for %s epoch(s).", epochs_since_improvement)
        else:
            epochs_since_improvement = 0

        if epochs_since_improvement >= training_config.early_stopping_patience:
            logger.info(
                "Early stopping triggered after %s substantial increases in validation loss.",
                training_config.early_stopping_patience,
            )
            break
