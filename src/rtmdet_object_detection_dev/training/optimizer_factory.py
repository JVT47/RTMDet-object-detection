from torch.optim import AdamW, Optimizer
import torch.nn as nn


def get_optimizer(model: nn.Module, name: str, config: dict) -> Optimizer:
    if name == "AdamW":
        return AdamW(model.parameters(), **config)

    raise ValueError(f"No '{name}' optimizer implemented")
