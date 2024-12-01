from dataclasses import dataclass
import torch
import torchvision

from .utils import BBoxLabelContainer, make_RTMDet_grid_points, transform_reg_pred_to_bbox_pred
from src.model.types import RTMDetOutput

@dataclass
class DetectionResult:
    """
    A class to hold the bounding boxes, classes and confidence scores for an image. 
    """
    bboxes: torch.Tensor # (N, 4) N = number of positive detections
    classes: torch.Tensor # (N)
    scores: torch.Tensor # (N)


class RTMDetPosprocessor:
    """
    Potprocessor that takes RTMDet model outputs and converts
    them to bounding boxes.
    """
    def __init__(self, score_threshold: float = 0.5, iou_threshold: float = 0.5, *args, **kwargs) -> None:
        """
        score_threshold: the minimum confidence score that a detection should have. All predictions with a lower score are 
                         classified as background.
        iou_threshold: the maximum IoU that two detections from the same class are allowed to have. Used in NMS to 
                       filter out duplicates.
        """
        super().__init__(*args, **kwargs)

        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
    
    def process_batch(self, model_output: RTMDetOutput) -> list[DetectionResult]:
        """
        Transforms each model output batch element to a detection result. Returns the detecton results in a list
        in the same order as they are in the batch. 
        """
        _, _, first_layer_height, first_layer_width = model_output.cls_preds[0].shape
        grid_points = make_RTMDet_grid_points(first_layer_height, first_layer_width)
        batch_preds = model_output.flatten()

        detection_results = []
        for reg_pred, cls_pred in zip(batch_preds.bboxes, batch_preds.labels):
            detection_result = self.process_single_batch_element(BBoxLabelContainer(reg_pred, cls_pred), grid_points)
            detection_results.append(detection_result)
        
        return detection_results
    
    def process_single_batch_element(self, bbox_and_label_preds: BBoxLabelContainer, grid_points: torch.Tensor) -> DetectionResult:
        """
        Processes the model outputs for a singel image, i.e., the input tensors should not have the batch dimension.
        """
        reg_pred, cls_pred = bbox_and_label_preds.bboxes, bbox_and_label_preds.labels
        bboxes = transform_reg_pred_to_bbox_pred(reg_pred, grid_points)

        scores, classes = cls_pred.sigmoid().max(dim=-1)

        score_mask = scores > self.score_threshold

        bboxes = bboxes[score_mask]
        classes = classes[score_mask]
        scores = scores[score_mask]

        detection_result = self._perform_nms(bboxes, classes, scores)

        return detection_result

    def _perform_nms(self, bboxes, classes, scores) -> DetectionResult:
        """
        Perform non maximum supression for each individual class and returns the kept detections. NMS is done
        for each individual class to allow different classes to overlap. All input tensors should not have the batch 
        dimension.
        """
        kept_bboxes = []
        kept_classes = []
        kept_scores = []
        for class_label in classes.unique():
            mask = classes == class_label
            indices = torchvision.ops.nms(bboxes[mask], scores[mask], self.iou_threshold)

            kept_bboxes.append(bboxes[mask][indices])
            kept_classes.append(classes[mask][indices])
            kept_scores.append(scores[mask][indices])
        
        return DetectionResult(torch.cat(kept_bboxes, dim=0), torch.cat(kept_classes, dim=0), torch.cat(kept_scores, dim=0))