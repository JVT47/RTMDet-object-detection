from torch import nn
from torch.optim import AdamW, Optimizer


def get_optimizer(model: nn.Module, name: str, config: dict) -> Optimizer:
    """Get the configured optimizer."""
    if name == "AdamW":
        return AdamW(model.parameters(), **config)

    msg = f"No '{name}' optimizer implemented"
    raise ValueError(msg)
