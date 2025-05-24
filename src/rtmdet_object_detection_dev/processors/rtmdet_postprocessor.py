import torch
import torchvision

from rtmdet_object_detection_dev.dataclasses.bbox_label_container import (
    BBoxLabelContainer,
)
from rtmdet_object_detection_dev.dataclasses.detection_result import DetectionResult
from rtmdet_object_detection_dev.dataclasses.rtmdet_output import RTMDetOutput


class RTMDetPostprocessor:
    """
    Postprocessor that takes RTMDet model outputs and converts
    them to bounding boxes. The bounding box locations are given as
    coordinates in the model input images that are likely to be
    padded so they do not directly convert to the dimensions of the
    original input image.
    """

    def __init__(
        self, score_threshold: float = 0.5, iou_threshold: float = 0.5, *args, **kwargs
    ) -> None:
        """
        ## Args
        - score_threshold: the minimum confidence score that a detection should have. All predictions with a lower score are
                           classified as background.
        - iou_threshold: the maximum IoU that two detections from the same class are allowed to have. Used in NMS to
                         filter out duplicates.
        """
        super().__init__(*args, **kwargs)

        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

    def process_batch(self, model_output: RTMDetOutput) -> list[DetectionResult]:
        """
        Transforms each model output batch element to a detection result. Returns the detection results in a list
        in the same order as they are in the batch.
        """
        batch_preds = model_output.process_and_combine_layers()

        detection_results = []
        for bbox_pred, cls_pred in zip(batch_preds.bboxes, batch_preds.labels):
            detection_result = self.process_single_batch_element(
                BBoxLabelContainer(bbox_pred, cls_pred)
            )
            detection_results.append(detection_result)

        return detection_results

    def process_single_batch_element(
        self, bbox_and_label_preds: BBoxLabelContainer
    ) -> DetectionResult:
        """
        Processes the model outputs for a single image, i.e., the input tensors should not have the batch dimension.
        """
        bboxes, cls_pred = bbox_and_label_preds.bboxes, bbox_and_label_preds.labels

        scores, classes = cls_pred.max(dim=-1)
        score_mask = scores > self.score_threshold

        bboxes = bboxes[score_mask]
        classes = classes[score_mask]
        scores = scores[score_mask]

        detection_result = self._perform_nms(bboxes, classes, scores)

        return detection_result

    @staticmethod
    def bbox_to_original_image(
        bboxes: torch.Tensor, preprocess_shape: torch.Size, orig_img_shape: torch.Size
    ) -> torch.Tensor:
        """
        Transforms bboxes in the preprocessed image shape back to the original image dimensions.
        ## Args
        - bboxes: tensor of shape (n, 4)
        - preprocess_shape: Size of form (..., H_1, W_1). Shape of preprocessed images
        - orig_img_shape: Size of form (..., H_2, W_2). Shape of the original image.
        """
        orig_height, orig_width = orig_img_shape[-2:]
        pre_height, pre_width = preprocess_shape[-2:]
        scale_factor = min(pre_height / orig_height, pre_width / orig_width)
        rescale_height, rescale_width = (
            int(scale_factor * orig_height),
            int(scale_factor * orig_width),
        )

        bboxes /= torch.tensor(
            [rescale_width, rescale_height, rescale_width, rescale_height]
        )
        bboxes *= torch.tensor([orig_width, orig_height, orig_width, orig_height])

        return bboxes

    def _perform_nms(
        self, bboxes: torch.Tensor, classes: torch.Tensor, scores: torch.Tensor
    ) -> DetectionResult:
        """
        Perform non maximum suppression for each individual class and returns the kept detections. NMS is done
        for each individual class to allow different classes to overlap. All input tensors should not have the batch
        dimension.
        """
        kept_bboxes = []
        kept_classes = []
        kept_scores = []
        for class_label in classes.unique():
            mask = classes == class_label
            indices = torchvision.ops.nms(
                bboxes[mask], scores[mask], self.iou_threshold
            )

            kept_bboxes.append(bboxes[mask][indices])
            kept_classes.append(classes[mask][indices])
            kept_scores.append(scores[mask][indices])

        return DetectionResult(
            torch.cat(kept_bboxes, dim=0),
            torch.cat(kept_classes, dim=0),
            torch.cat(kept_scores, dim=0),
        )
