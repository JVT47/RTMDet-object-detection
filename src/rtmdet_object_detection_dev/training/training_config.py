from pathlib import Path
from pydantic import BaseModel
from typing import Any


class TrainingConfig(BaseModel):
    model_cfg: dict[str, Any]
    training_dataloader_config: dict[str, Any]
    validation_dataloader_config: dict[str, Any]
    loss_fn_config: dict[str, Any]
    optimizer_config: dict[str, Any]
    weights_save_path: Path
    session_name: str
    epochs: int
    device: str
