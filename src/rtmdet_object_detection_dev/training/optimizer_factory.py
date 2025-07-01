from torch import nn
from torch.optim import AdamW, Optimizer


def get_optimizer_param_groups(model: nn.Module) -> list[dict]:
    """Get model param groups for the optimizer.

    Makes sure that weight decay is not applied to batch norm and bias variables.
    """
    decay = []
    no_decay = []
    for param in model.parameters():
        if not param.requires_grad:
            continue

        batch_norm_and_bias_shape = 1
        if len(param.shape) == batch_norm_and_bias_shape:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def get_optimizer(model: nn.Module, name: str, config: dict) -> Optimizer:
    """Get the configured optimizer."""
    if name == "AdamW":
        return AdamW(get_optimizer_param_groups(model), **config)

    msg = f"No '{name}' optimizer implemented"
    raise ValueError(msg)
