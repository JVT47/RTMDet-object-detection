import torch

from rtmdet_object_detection_dev.dataclasses.bbox_label_container import (
    BBoxLabelContainer,
)
from rtmdet_object_detection_dev.losses.rtmdet_label_assigner import RTMDetLabelAssigner
from tests.utils import create_model_output_1


class TestRTMDetLabelAssigner:
    label_assigner = RTMDetLabelAssigner(q=20)

    def test_make_assigner_matrix_feasible_removes_duplicate_assignments(self) -> None:
        assigner_matrix = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.bool)
        cost_matrix = torch.tensor([[1, 10, 3], [10, 1, 2], [3, 1.1, 0.5]])

        self.label_assigner._make_assigner_matrix_feasible(assigner_matrix, cost_matrix)  # noqa: SLF001
        target = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.bool)

        torch.testing.assert_close(assigner_matrix, target)

    def test_simOTA_returns_correct_assigner_matrix_1(self) -> None:  # noqa: N802
        ground_truths = BBoxLabelContainer(
            torch.tensor([[0.0, 0, 1, 1], [7, 7, 9, 9]]),
            torch.tensor([[0, 0, 1], [1, 0, 0]]),
        )
        bbox_pred = torch.tensor(
            [
                [0.0, 0.0, 1, 1],
                [8, 0, 9, 1],
                [15, -1, 16, 2],
                [0, 8, 1, 9],
                [7.5, 7, 9, 9],
                [16, 8, 17, 9],
            ],
        )
        label_pred = torch.tensor(
            [
                [0.1, 0.6, 0.9],
                [0.2, 0.3, 0.4],
                [0.5, 0.3, 0.2],
                [0.1, 0.2, 0.3],
                [0.7, 0.2, 0.2],
                [0.05, 0.4, 0.3],
            ],
        )
        predictions = BBoxLabelContainer(bbox_pred, label_pred)
        grid_points = torch.tensor([[0.0, 0], [8, 0], [16, 0], [0, 8], [8, 8], [16, 8]])

        assigner_matrix = self.label_assigner.simOTA(ground_truths, predictions, grid_points)
        target = torch.tensor([[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]], dtype=torch.bool)

        torch.testing.assert_close(assigner_matrix, target)

    def test_simOTA_returns_correct_assigner_matrix_2(self) -> None:  # noqa: N802
        ground_truths = BBoxLabelContainer(
            torch.tensor([[0.0, 0, 1, 1], [7, 7, 9, 9]]),
            torch.tensor([[0, 0, 1], [1, 0, 0]]),
        )
        bbox_pred = torch.tensor(
            [
                [0.0, 0.0, 1, 1],
                [0.5, 0.0, 1, 1],
                [15, -1, 16, 2],
                [0, 8, 1, 9],
                [7.5, 7, 9, 9],
                [16, 8, 17, 9],
            ],
        )
        label_pred = torch.tensor(
            [
                [0.1, 0.6, 0.9],
                [0.2, 0.3, 0.8],
                [0.5, 0.3, 0.2],
                [0.1, 0.2, 0.3],
                [0.7, 0.2, 0.2],
                [0.05, 0.4, 0.3],
            ],
        )
        predictions = BBoxLabelContainer(bbox_pred, label_pred)
        grid_points = torch.tensor([[0.0, 0], [8, 0], [16, 0], [0, 8], [8, 8], [16, 8]])

        assigner_matrix = self.label_assigner.simOTA(ground_truths, predictions, grid_points)
        target = torch.tensor([[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]], dtype=torch.bool)

        torch.testing.assert_close(assigner_matrix, target)

    def test_assign_targets_to_batch_element_assigns_correct_bboxes_and_labels(
        self,
    ) -> None:
        element_gts = BBoxLabelContainer(
            torch.tensor([[0.0, 0, 1, 1], [7, 7, 9, 9]]),
            torch.tensor([[0, 0, 1], [1, 0, 0]]),
        )
        bbox_pred = torch.tensor(
            [
                [0.0, 0.0, 1, 1],
                [0.5, 0.0, 1, 1],
                [15, -1, 16, 2],
                [0, 8, 1, 9],
                [7.5, 7, 9, 9],
                [16, 8, 17, 9],
            ],
        )
        label_pred = torch.tensor(
            [
                [0.1, 0.6, 0.9],
                [0.2, 0.3, 0.8],
                [0.5, 0.3, 0.2],
                [0.1, 0.2, 0.3],
                [0.7, 0.2, 0.2],
                [0.05, 0.4, 0.3],
            ],
        )
        element_preds = BBoxLabelContainer(bbox_pred, label_pred)
        grid_points = torch.tensor([[0.0, 0], [8, 0], [16, 0], [0, 8], [8, 8], [16, 8]])

        targets = self.label_assigner.assign_targets_for_batch_element(element_gts, element_preds, grid_points)
        target_bboxes = torch.tensor(
            [
                [0.0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [7, 7, 9, 9],
                [0, 0, 0, 0],
            ],
        )
        target_labels = torch.tensor([[0.0, 0, 1], [0, 0, 1], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0]])

        torch.testing.assert_close(targets.bboxes, target_bboxes)
        torch.testing.assert_close(targets.labels, target_labels)

    def test_assign_targets_assigns_correct_bboxes_and_labels(self) -> None:
        model_output = create_model_output_1()
        gt_1 = BBoxLabelContainer(torch.tensor([[7.5, 7, 8.5, 8]]), torch.tensor([[0, 0, 1]]))
        gt_2 = BBoxLabelContainer(torch.tensor([[16, 15.5, 17, 17]]), torch.tensor([[1, 0, 0]]))

        targets = self.label_assigner.assign_targets([gt_1, gt_2], model_output)
        target_bboxes = torch.zeros((2, 21, 4))
        target_bboxes[0, 5] = torch.tensor([7.5, 7, 8.5, 8])
        target_bboxes[1, 19] = torch.tensor([16, 15.5, 17, 17])
        target_labels = torch.zeros((2, 21, 3))
        target_labels[0, 5] = torch.tensor([0, 0, 1])
        target_labels[1, 19] = torch.tensor([1, 0, 0])

        torch.testing.assert_close(targets.bboxes, target_bboxes)
        torch.testing.assert_close(targets.labels, target_labels)
