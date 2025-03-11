from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.model.model import make_model
from src.datasets.dataset_factory import get_dataloader
from src.losses.loss_fn_factory import get_loss_fn
from src.training.optimizer_factory import get_optimizer


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
    n_samples = 0

    for data in training_dataloader:
        optimizer.zero_grad()

        images, gts = data[0].to(device), data[1].to(device)

        rtmdet_output = model(images)

        loss: torch.Tensor = rtmdet_loss(rtmdet_output, gts)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        n_samples += images.shape[0]

    return running_loss / n_samples


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
        n_samples = 0

        for data in validation_dataloader:
            images, gts = data[0].to(device), data[1].to(device)

            rtmdet_output = model(images)

            loss: torch.Tensor = rtmdet_loss(rtmdet_output, gts)
            running_loss += loss.item()
            n_samples += images.shape[0]

        return running_loss / n_samples


def train_model(
    model_config: dict,
    training_dataloader_config: dict,
    validation_dataloader_config: dict,
    loss_fn_config: dict,
    optimizer_config: dict,
    weights_save_path: Path,
    session_name: str,
    epochs: int,
    device: torch.device,
) -> None:
    """
    ## Args
        - model_config: contains arguments for the model factory
        - training_config: contains arguments for the dataloader factory. Used for optimization
        - validation_config: contains arguments for the dataloader factory. Used in validation
        - loss_fn_config: contains arguments for the loss_fn factory
        - optimizer_config: contains arguments for the optimizer factory. Should not contain model parameters
        - weights_save_dir_path: dir where the model weights file is saved
        - session_name: name for the training session. Used as the filename for model weights. .pth suffix added automatically
        - epoch: number of epoch to train for
        - device: device used for training
    """
    model = make_model(**model_config)
    model.to(device)

    training_dataloader = get_dataloader(**training_dataloader_config)
    validation_dataloader = get_dataloader(**validation_dataloader_config)

    loss = get_loss_fn(**loss_fn_config)
    optimizer = get_optimizer(model, **optimizer_config)

    best_validation_loss = float("inf")
    for i in range(epochs):
        print(f"Epoch {i + 1} / {epochs}")

        training_mean_loss = train_one_epoch(
            model, training_dataloader, optimizer, loss, device
        )
        validation_mean_loss = validate(model, validation_dataloader, loss, device)

        print(
            f"Training loss: {training_mean_loss}, validation loss: {validation_mean_loss}"
        )

        if validation_mean_loss < best_validation_loss:
            best_validation_loss = validation_mean_loss
            weights_path = weights_save_path.joinpath(f"{session_name}.pth")
            torch.save(model.state_dict(), weights_path)
