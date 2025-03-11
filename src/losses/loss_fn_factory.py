import torch.nn as nn

from src.losses.rtmdet_loss import RTMDetLoss


def get_loss_fn(name: str, config: dict) -> nn.Module:
    if name == "RTMDetLoss":
        return RTMDetLoss(**config)

    raise ValueError(f"No '{name}' loss function implemented")
