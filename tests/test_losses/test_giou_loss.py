import torch

from rtmdet_object_detection_dev.losses.giou_loss import GIoULoss


class TestGIoULoss:
    giou_loss = GIoULoss()

    def test_calc_iou_and_union_returns_correct_iou_and_union_scores_1(self) -> None:
        bboxes_1 = torch.tensor(
            [[[0.0, 0, 1, 1], [1, 1, 2, 2]], [[0, 0, 1, 2], [1, 1, 2, 2]]]
        )
        bboxes_2 = torch.tensor(
            [[[0.0, 0, 1, 2], [1, 1, 2, 2]], [[0, 0, 1, 1], [1, 1, 3, 3]]]
        )

        iou, union = self.giou_loss.calc_IoU_and_union(bboxes_1, bboxes_2)
        target_iou = torch.tensor([[0.5, 1.0], [0.5, 0.25]])
        target_union = torch.tensor([[2.0, 1.0], [2.0, 4.0]])

        torch.testing.assert_close(iou, target_iou)
        torch.testing.assert_close(union, target_union)

    def test_calc_iou_and_union_returns_correct_iou_and_union_scores(self) -> None:
        bboxes_1 = torch.tensor([[12.0, 12, 12, 12], [0, 0, 1, 1]])
        bboxes_2 = torch.tensor([[13.0, 13, 13, 13], [14, 14, 14, 14]])

        iou, union = self.giou_loss.calc_IoU_and_union(bboxes_1, bboxes_2)
        target_iou = torch.tensor([0.0, 0])
        target_union = torch.tensor([0.0, 1.0])

        torch.testing.assert_close(iou, target_iou)
        torch.testing.assert_close(union, target_union)

    def test_forward_returns_correct_loss(self) -> None:
        bboxes_1 = torch.tensor(
            [[[0.0, 0, 2, 1], [1, 1, 2, 2]], [[0, 0, 1, 2], [1, 1, 2, 2]]]
        )
        bboxes_2 = torch.tensor(
            [[[0.0, 0, 1, 2], [1, 1, 2, 2]], [[0, 0, 2, 1], [1, 1, 3, 3]]]
        )

        loss = self.giou_loss(bboxes_1, bboxes_2)
        target = torch.tensor([[0.917, 0.0], [0.917, 0.75]])

        torch.testing.assert_close(loss, target, atol=1e-2, rtol=0.0)
