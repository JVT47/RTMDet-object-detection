import torch
import torch.nn as nn
from collections import OrderedDict

from .rtmdet_models import RTMDet


def load_state_dict_with_resizing(
    model: nn.Module, weights_state_dict: dict[str, torch.Tensor]
) -> None:
    """
    Load the matching weights in the state dict to the model. If the shape does not match
    resizes the weights to match to the model's shape. Ignores missing or extra weights.
    """
    model_state_dict: dict[str, torch.Tensor] = model.state_dict()

    resized_dict = OrderedDict()
    for k, v in weights_state_dict.items():
        if k not in model_state_dict:
            continue

        if model_state_dict[k].shape == v.shape:
            resized_dict[k] = v
            continue

        old_shape = v.shape
        new_shape = model_state_dict[k].shape

        if len(v.shape) == 4:
            out_channels = min(old_shape[0], new_shape[0])
            resized = v[:out_channels, : new_shape[1], : new_shape[2], : new_shape[3]]
            padded = torch.zeros(new_shape)
            padded[: resized.shape[0]] = resized
            resized_dict[k] = padded
            continue

        if len(v.shape) == 1:
            resized = v[: new_shape[0]]
            padded = torch.zeros(new_shape)
            padded[: resized.shape[0]] = resized
            resized_dict[k] = padded
            continue

        raise RuntimeError(f"Resizing of key {k} with shape {v.shape} not supported.")

    model.load_state_dict(resized_dict, strict=False)


def make_model(
    model_name: str,
    num_classes: int,
    model_weights: str | None,
    strict: bool = True,
    eval: bool = True,
    raw_output: bool = False,
) -> nn.Module:
    """
    ## Args
    - model_name: name of the model, e.g., 'RTMDetTiny'
    - num_classes: the number of classes that the model should predict.
    - model_weights: path to the model weight file. If None, no weights are loaded.
    - strict: bool that tells if weights should be loaded strictly, i.e., with perfect compatibility with the model.
    - eval: bool that tells if the model should be set to eval mode.
    - raw_output: if true returns outputs as a tuple of classification output tuple and regression output tuple instead of the
                    RTMDetOutput class. Used when converting to onnx.
    """
    model = None
    if model_name == "RTMDetTiny":
        model = RTMDet(
            widen_factor=0.375,
            deepen_factor=0.167,
            num_classes=num_classes,
            exp_on_reg=False,
            raw_output=raw_output,
        )
    elif model_name == "RTMDetS":
        model = RTMDet(
            widen_factor=0.5,
            deepen_factor=0.33,
            num_classes=num_classes,
            exp_on_reg=False,
            raw_output=raw_output,
        )
    elif model_name == "RTMDetM":
        model = RTMDet(
            widen_factor=0.75,
            deepen_factor=0.67,
            num_classes=num_classes,
            exp_on_reg=True,
            raw_output=raw_output,
        )
    elif model_name == "RTMDetL":
        model = RTMDet(
            widen_factor=1.0,
            deepen_factor=1.0,
            num_classes=num_classes,
            exp_on_reg=True,
            raw_output=raw_output,
        )
    elif model_name == "RTMDetX":
        model = RTMDet(
            widen_factor=1.25,
            deepen_factor=1.33,
            num_classes=num_classes,
            exp_on_reg=True,
            raw_output=raw_output,
        )
    else:
        raise ValueError(f"Invalid model name '{model_name}'.")

    if model_weights is not None:
        weights_state_dict = torch.load(model_weights, weights_only=True)
        if strict:
            model.load_state_dict(weights_state_dict, strict=True)
        else:
            load_state_dict_with_resizing(model, weights_state_dict)

    if eval:
        model.eval()

    return model
