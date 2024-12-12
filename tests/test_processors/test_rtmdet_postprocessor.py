import torch

from tests.utils import create_model_output_1
from src.dataclasses.bbox_label_container import BBoxLabelContainer
from src.dataclasses.rtmdet_output import RTMDetOutput
from src.processors.rtmdet_postprocessor import RTMDetPostprocessor


class TestRTMDetPostprocessor:
    postprocessor = RTMDetPostprocessor(
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
        model_output = create_model_output_1()

        detection_results = self.postprocessor.process_batch(model_output)

        torch.testing.assert_close(detection_results[0].bboxes, torch.tensor([[7.0, 7, 8, 8]]))
        torch.testing.assert_close(detection_results[0].classes, torch.tensor([1]))
        torch.testing.assert_close(detection_results[0].scores, torch.tensor([1.0]), atol=1e-3, rtol=0.0)

        torch.testing.assert_close(detection_results[1].bboxes, torch.tensor([[16.0, 16, 17, 17]]))
        torch.testing.assert_close(detection_results[1].classes, torch.tensor([0]))
        torch.testing.assert_close(detection_results[1].scores, torch.tensor([1.0]), atol=1e-3, rtol=0.0)
    
    def test_bbox_to_original_image(self) -> None:
        bboxes = torch.tensor([[0.0, 0, 2, 2], [20, 20, 28, 28], [30, 40, 32, 48]])
        preprocess_shape = torch.Size((32, 64))
        orig_img_shape = torch.Size((16, 24))

        bboxes = self.postprocessor.bbox_to_original_image(bboxes, preprocess_shape, orig_img_shape)
        target = torch.tensor([[0.0, 0, 1, 1], [10, 10, 14, 14], [15, 20, 16, 24]])

        torch.testing.assert_close(bboxes, target)
