from pathlib import Path
from typing import Any

from pydantic import BaseModel


class TrainingConfig(BaseModel):
    """Pydantic model for the training config."""

    model_cfg: dict[str, Any]
    training_dataloader_config: dict[str, Any]
    validation_dataloader_config: dict[str, Any]
    loss_fn_config: dict[str, Any]
    optimizer_config: dict[str, Any]
    weights_save_path: Path
    session_name: str
    epochs: int
    device: str
