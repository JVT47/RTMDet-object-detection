from dataclasses import dataclass
import torch
import torchvision

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
        batch_size, _, _, _ = model_output.cls_preds[0].shape

        detection_results = []
        for i in range(batch_size):
            batch_element_cls = model_output.cls_preds[0][i], model_output.cls_preds[1][i], model_output.cls_preds[2][i]
            batch_element_reg = model_output.reg_preds[0][i], model_output.reg_preds[1][i], model_output.reg_preds[2][i]
            detection_result = self.process_single_batch_element(RTMDetOutput(batch_element_cls, batch_element_reg))
            detection_results.append(detection_result)
        
        return detection_results
    
    def process_single_batch_element(self, model_output: RTMDetOutput) -> DetectionResult:
        """
        Processes the model outputs for a singel image, i.e., the input tensors should not have the batch dimension.
        """
        bboxes = []
        classes = []
        scores = []
        scales = [8, 16, 32]

        for i in range(3):
            cls_pred, reg_pred = model_output.cls_preds[i], model_output.reg_preds[i]
            _, height, width = cls_pred.shape

            cls_pred = cls_pred.sigmoid()
            conf_scores, class_pred = cls_pred.max(dim=0)

            reg_pred = reg_pred.permute(1, 2, 0) # Change from (4, H, W) to (H, W, 4)
            reg_pred = reg_pred.reshape((-1, 4)) # (H * W, 4)
            conf_scores = conf_scores.reshape((-1))
            class_pred = class_pred.reshape((-1))
            
            scale = scales[i]
            points_xx, points_yy = torch.meshgrid(torch.arange(0, height, dtype=torch.float), torch.arange(0, width, dtype=torch.float), indexing="xy")
            points = torch.stack([points_xx.flatten(), points_yy.flatten()], dim=-1) * scale # shape (H * W, 2) in (x, y) format
    
            bbox_pred = torch.cat([points, points], dim=-1)
            bbox_pred[:, :2] -= reg_pred[:, :2]
            bbox_pred[:, 2:] += reg_pred[:, 2:]

            score_mask = conf_scores > self.score_threshold

            bboxes.append(bbox_pred[score_mask])
            classes.append(class_pred[score_mask])
            scores.append(conf_scores[score_mask])
        
        bboxes = torch.cat(bboxes, dim=0)
        classes = torch.cat(classes, dim=0)
        scores = torch.cat(scores, dim=0)

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