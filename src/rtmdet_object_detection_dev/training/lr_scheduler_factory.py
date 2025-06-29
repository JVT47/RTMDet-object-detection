import torch


def get_lr_scheduler(optimizer: torch.optim.Optimizer, name: str, config: dict) -> torch.optim.lr_scheduler.LRScheduler:
    """Return the configured lr scheduler."""
    if name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **config)

    msg = f"No '{name}' lr scheduler implemented."
    raise ValueError(msg)
