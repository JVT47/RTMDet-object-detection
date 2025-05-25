import torch

from rtmdet_object_detection_dev.losses.center_cost import CenterCost


class TestCenterCost:
    center_cost = CenterCost(alpha=10, beta=1)

    def test_forward_returns_correct_loss(self) -> None:
        gt_bbox = torch.tensor([0, 0, 2, 2])
        grid_points = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]])

        cost = self.center_cost(gt_bbox, grid_points)
        target = torch.tensor([2.595, 1, 1, 0.1])

        torch.testing.assert_close(cost, target, atol=1e-2, rtol=0.0)
