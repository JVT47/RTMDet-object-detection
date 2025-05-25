import torch

from tests.utils import create_model_output_1


class TestRTMDetOutput:
    model_output = create_model_output_1()

    def test_create_model_grid_points_returns_correct_points(self) -> None:
        grid_points = self.model_output.create_model_grid_points()

        target = torch.tensor(
            [
                [0.0, 0],
                [8, 0],
                [16, 0],
                [24, 0],
                [0, 8],
                [8, 8],
                [16, 8],
                [24, 8],
                [0, 16],
                [8, 16],
                [16, 16],
                [24, 16],
                [0, 24],
                [8, 24],
                [16, 24],
                [24, 24],
                [0, 0],
                [16, 0],
                [0, 16],
                [16, 16],
                [0, 0],
            ]
        )

        torch.testing.assert_close(grid_points, target)

    def test_transform_reg_pred_to_bbox_pred_returns_correct_bboxes(self) -> None:
        reg_pred = torch.tensor(
            [[[1, 1, 0, 0], [0, 0, 2, 2]], [[1, 1, 1, 1], [2, 2, 2, 2]]]
        )
        grid_points = torch.tensor([[4, 4], [8, 8]])

        bboxes = self.model_output.transform_reg_pred_to_bbox_pred(
            reg_pred, grid_points
        )
        targets = torch.tensor(
            [[[3, 3, 4, 4], [8, 8, 10, 10]], [[3, 3, 5, 5], [6, 6, 10, 10]]]
        )

        torch.testing.assert_close(bboxes, targets)
