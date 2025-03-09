import torch
import torch.nn as nn

from .rtmdet_models import RTMDetTiny, RTMDetS, RTMDetM, RTMDetL, RTMDetX


def make_model(
    model_name: str, num_classes: int, model_weights: str | None, eval: bool = True
) -> nn.Module:
    """
    ## Args
    - model_name: name of the model, e.g., 'RTMDetTiny'
    - num_classes: the number of classes that the model should predict.
    - model_weights: path to the model weight file. If None, no weights are loaded.
    - eval: bool that tells if the model should be set to eval mode.
    """
    model = None
    if model_name == "RTMDetTiny":
        model = RTMDetTiny(num_classes)
    elif model_name == "RTMDetS":
        model = RTMDetS(num_classes)
    elif model_name == "RTMDetM":
        model = RTMDetM(num_classes)
    elif model_name == "RTMDetL":
        model = RTMDetL(num_classes)
    elif model_name == "RTMDetX":
        model = RTMDetX(num_classes)
    else:
        raise ValueError(f"Invalid model name '{model_name}'.")

    if model_weights is not None:
        model.load_state_dict(torch.load(model_weights, weights_only=True), strict=True)

    if eval:
        model.eval()

    return model
