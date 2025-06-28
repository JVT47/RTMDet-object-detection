import torch
from torch import nn

from rtmdet_object_detection_dev.dataclasses.bbox_label_container import (
    BBoxLabelContainer,
)
from rtmdet_object_detection_dev.dataclasses.rtmdet_output import RTMDetOutput

from .giou_loss import GIoULoss
from .quality_focal_loss import QualityFocalLoss
from .rtmdet_label_assigner import RTMDetLabelAssigner


class RTMDetLoss(nn.Module):
    """The loss function for RTMDet models. Takes care of label assignment."""

    def __init__(self, reg_loss_weight: float = 2, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Initialize the loss.

        ## Args
        - reg_loss_weight: the weight of the regression loss. The weight for classification is 1.
        """
        super().__init__(*args, **kwargs)

        self.reg_loss_weight = reg_loss_weight
        self.cls_loss = QualityFocalLoss()
        self.reg_loss = GIoULoss()
        self.label_assigner = RTMDetLabelAssigner()

    def forward(self, model_output: RTMDetOutput, ground_truths: list[BBoxLabelContainer]) -> torch.Tensor:
        """Calculate the loss between the model outputs and ground truths.

        ## Args
        - model_output: predictions made by the model
        - ground_truths: list of ground truth bboxes and labels for the batch.
                         Should be in the same order as the model inputs.
        """
        targets = self.label_assigner.assign_targets(ground_truths, model_output)
        preds = model_output.process_and_combine_layers()

        batch_size, n_preds = preds.bboxes.shape[0:2]

        mask = targets.bboxes.sum(dim=-1) != 0
        n_positives = (mask).sum(dim=-1)  # shape (B)
        n_positives[n_positives == 0] = 1

        loss = torch.zeros((batch_size, n_preds), device=model_output.get_device())
        loss[mask] = self.reg_loss_weight * self.reg_loss(preds.bboxes[mask], targets.bboxes[mask])

        IoU, _ = self.reg_loss.calc_IoU_and_union(preds.bboxes, targets.bboxes)  # noqa: N806
        loss += self.cls_loss(preds.labels, targets.labels, IoU)

        loss = loss.sum(dim=1) * 1 / n_positives

        return loss.sum() / batch_size
