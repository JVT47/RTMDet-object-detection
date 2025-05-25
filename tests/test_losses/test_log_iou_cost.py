import torch

from rtmdet_object_detection_dev.losses.log_iou_cost import LogIoUCost


class TestLogIoUCost:
    log_iou_cost = LogIoUCost()

    def test_forward_returns_correct_loss(self) -> None:
        iou = torch.tensor([0.5, 1.0, 0.1])

        cost = self.log_iou_cost(iou)
        target = torch.tensor([0.693, 0.0, 2.30])

        torch.testing.assert_close(cost, target, atol=1e-2, rtol=0.0)
