import torch

from src.dataclasses.bbox_label_container import BBoxLabelContainer
from src.dataclasses.rtmdet_output import RTMDetOutput
from src.processors.rtmdet_postprocessor import RTMDetPosprocessor


class TestRTMDetPostprocesor:
    postprocessor = RTMDetPosprocessor(
        score_threshold=0.5,
        iou_threshold=0.5,
    )

    def test_perform_nms(self) -> None:
        bboxes = torch.tensor([[0.0, 0, 3, 2], [1.0, 0, 3, 2], [0.0, 3, 1, 4]])
        classes = torch.tensor([1, 1, 2])
        scores = torch.tensor([0.6, 0.55, 0.8])

        detection_result = self.postprocessor._perform_nms(bboxes, classes, scores)
        
        torch.testing.assert_close(detection_result.bboxes, torch.tensor([[0.0, 0, 3, 2], [0.0, 3, 1, 4]]))
        torch.testing.assert_close(detection_result.classes, torch.tensor([1, 2]))
        torch.testing.assert_close(detection_result.scores, torch.tensor([0.6, 0.8]))
    
    def test_process_single_batch_element(self) -> None:
        bboxes = torch.tensor([[0.0, 0, 2, 2], [0.0, 0, 1, 1], [0.0, 2, 1, 4]])
        labels = torch.tensor([[0.8, 0.1, 0.2], [0.4, 0.45, 0.3], [0.5, 0.1, 0.9]])
        preds = BBoxLabelContainer(bboxes, labels)

        detection_result = self.postprocessor.process_single_batch_element(preds)

        torch.testing.assert_close(detection_result.bboxes, torch.tensor([[0.0, 0, 2, 2], [0.0, 2, 1, 4]]))
        torch.testing.assert_close(detection_result.classes, torch.tensor([0, 2]))
        torch.testing.assert_close(detection_result.scores, torch.tensor([0.8, 0.9]))
    
    def test_process_batch(self) -> None:
        cls_1 = torch.ones((2, 4, 4, 3)) * -1
        cls_1[0, 1, 1] = torch.tensor([0, 10, 5])
        cls_1 = cls_1.permute(0, 3, 1, 2)
        cls_2 = torch.ones((2, 2, 2, 3)) * -1
        cls_2[1, 1, 1] = torch.tensor([10, 0, -10])
        cls_2 = cls_2.permute(0, 3, 1, 2)
        cls_3 = torch.ones((2, 3, 1, 1)) * -1

        reg_1 = torch.zeros((2, 4, 4, 4))
        reg_1[0, 1, 1] = torch.tensor([1, 1, 0, 0])
        reg_1 = reg_1.permute(0, 3, 1, 2)
        reg_2 = torch.zeros((2, 2, 2, 4))
        reg_2[1, 1, 1] = torch.tensor([0, 0, 1, 1])
        reg_2 = reg_2.permute(0, 3, 1, 2)
        reg_3 = torch.zeros((2, 4, 1, 1))

        model_output = RTMDetOutput((cls_1, cls_2, cls_3), (reg_1, reg_2, reg_3))

        detection_results = self.postprocessor.process_batch(model_output)

        torch.testing.assert_close(detection_results[0].bboxes, torch.tensor([[7.0, 7, 8, 8]]))
        torch.testing.assert_close(detection_results[0].classes, torch.tensor([1]))
        torch.testing.assert_close(detection_results[0].scores, torch.tensor([1.0]), atol=1e-3, rtol=0.0)

        torch.testing.assert_close(detection_results[1].bboxes, torch.tensor([[16.0, 16, 17, 17]]))
        torch.testing.assert_close(detection_results[1].classes, torch.tensor([0]))
        torch.testing.assert_close(detection_results[1].scores, torch.tensor([1.0]), atol=1e-3, rtol=0.0)
