import torch
import torch.nn as nn

from .rtmdet_models import RTMDet


def make_model(
    model_name: str,
    num_classes: int,
    model_weights: str | None,
    eval: bool = True,
    raw_output: bool = False,
) -> nn.Module:
    """
    ## Args
    - model_name: name of the model, e.g., 'RTMDetTiny'
    - num_classes: the number of classes that the model should predict.
    - model_weights: path to the model weight file. If None, no weights are loaded.
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
        model.load_state_dict(torch.load(model_weights, weights_only=True), strict=True)

    if eval:
        model.eval()

    return model
