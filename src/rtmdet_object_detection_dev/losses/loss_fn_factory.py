from torch import nn

from rtmdet_object_detection_dev.losses.rtmdet_loss import RTMDetLoss


def get_loss_fn(name: str, config: dict) -> nn.Module:
    """Return the given loss module."""
    if name == "RTMDetLoss":
        return RTMDetLoss(**config)

    msg = f"No '{name}' loss function implemented."
    raise ValueError(msg)
