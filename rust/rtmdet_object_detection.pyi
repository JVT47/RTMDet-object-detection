import numpy as np
from typing import Any

class BBox:
    """
    A class containing the properties of a bound box.
    """

    @property
    def top_left(self) -> tuple[float, float]:
        """
        The top left (x, y) coordinates of the bounding box.
        """

    @property
    def bottom_right(self) -> tuple[float, float]:
        """
        The bottom right (x, y) coordinates of the bounding box.
        """

    @property
    def class_num(self) -> int:
        """
        The index of the detected class. Starts from 0.
        """

    @property
    def score(self) -> float:
        """
        The probability confidence of the detected class.
        """

class DetectionResult:
    """
    A class for holding the bounding box detections for an image.
    """

    @property
    def bboxes(self) -> list[BBox]:
        """
        List containing the detected bounding boxes for the image.
        """

class RTMDetDetector:
    """
    A class for the RTMDet object detector.
    """

    def __init__(
        self,
        model_path: str,
        inference_shape: tuple[int, int],
        batch_size: int,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.3,
    ) -> None:
        """
        ## Args:
            - model_path: path to the RTMDet model onnx file.
            - inference_shape: (width, height) to which images are preprocessed before the model.
            - batch_size: The number of images inserted to the model at the same time.
            - score_threshold: The minimum score required for a valid detection
            - iou_threshold: the maximum iou score allowed in nms for detections of the same class.
        """

    def detect_from_numpy(
        self, arrays: list[np.ndarray[Any, np.dtype[np.uint8]]]
    ) -> list[DetectionResult]:
        """
        Performs object detection for the given numpy images.

        # Args:
            - arrays: list of images
        """
