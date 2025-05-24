import torch

from rtmdet_object_detection_dev.losses.quality_focal_loss import QualityFocalLoss


class TestQualityFocalLoss:
    quality_focal_loss = QualityFocalLoss(beta=2)

    def test_forward(self) -> None:
        pred_label = torch.tensor([[0.5, 0.8, 0.2], [0.4, 0.9, 0.1]])
        target_label = torch.tensor([[0, 1, 0], [1, 0, 0]])
        iou = torch.tensor([0.5, 1.0])

        loss = self.quality_focal_loss(pred_label, target_label, iou)
        target = torch.tensor([0.265, 2.200])

        torch.testing.assert_close(loss, target, atol=1e-2, rtol=0.0)

    def test_forward_batch(self) -> None:
        pred_label = torch.tensor([[0.5, 0.8, 0.2], [0.4, 0.9, 0.1]])
        pred_label = torch.stack([pred_label, pred_label])
        target_label = torch.tensor([[0, 1, 0], [1, 0, 0]])
        target_label = torch.stack([target_label, target_label])
        iou = torch.tensor([0.5, 1.0])
        iou = torch.stack([iou, iou])

        loss = self.quality_focal_loss(pred_label, target_label, iou)
        target = torch.tensor([0.265, 2.200])
        target = torch.stack([target, target])

        torch.testing.assert_close(loss, target, atol=1e-2, rtol=0.0)
