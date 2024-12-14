import torch

from src.dataclasses.bbox_label_container import BBoxLabelContainer
from src.losses.rtmdet_loss import RTMDetLoss
from tests.utils import create_model_output_1


class TestRTMDetLoss:
    loss = RTMDetLoss(reg_loss_weight=2)

    def test_forward_1(self) -> None:
        model_output = create_model_output_1()
        gt_1 = BBoxLabelContainer(torch.tensor([[7.5, 7, 8.5, 8]]), torch.tensor([[0, 0, 1]]))
        gt_2 = BBoxLabelContainer(torch.tensor([[16, 15.5, 17, 17]]), torch.tensor([[1, 0, 0]]))

        loss = self.loss(model_output, [gt_1, gt_2])
        target = 12.986 + 2 * 0.5 / 1.5
        target += 1.903 + 2 * 1 / 1.5
        target = torch.tensor(target)

        torch.testing.assert_close(loss, target, atol=1e-2, rtol=0)