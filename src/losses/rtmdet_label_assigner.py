import torch
import torchvision

from src.losses.quality_focal_loss import QualityFocalLoss
from src.losses.log_iou_cost import LogIoUCost
from src.losses.center_cost import CenterCost
from src.dataclasses.bbox_label_container import BBoxLabelContainer
from src.dataclasses.rtmdet_output import RTMDetOutput


class RTMDetLabelAssigner:
    """
    A class that turns ground truth labels to RTMDet target tensors 
    so that RTMDet can be trained. 
    """
    def __init__(self, q: int = 20) -> None:
        """
        ## Args
        - q: hyperparameter that determines how many of the biggest IoU scores should be
             added together to achieve the top k number in simOTA.
        """
        self.q = q
        self.cls_cost = QualityFocalLoss()
        self.reg_cost = LogIoUCost()
        self.center_cost = CenterCost()

    def assign_targets(self, ground_truths: list[BBoxLabelContainer], model_output: RTMDetOutput) -> BBoxLabelContainer:
        """
        Creates target tensors to the ground truths based on the model predictions with simOTA label assignment.
        """
        grid_points = model_output.create_model_grid_points()
        preds = model_output.process_and_combine_layers()

        targets: list[BBoxLabelContainer] = []
        for element_gts, pred_bboxes, pred_labels in zip(ground_truths, preds.bboxes, preds.labels):
            element_preds = BBoxLabelContainer(pred_bboxes, pred_labels)
            element_targets = self.assign_targets_for_batch_element(element_gts, element_preds, grid_points)
            targets.append(element_targets)
        
        target_bboxes = torch.stack([target.bboxes for target in targets], dim=0)
        target_labels = torch.stack([target.labels for target in targets], dim=0)
        
        return BBoxLabelContainer(target_bboxes, target_labels)
    
    def assign_targets_for_batch_element(self, element_gts: BBoxLabelContainer, element_preds: BBoxLabelContainer, grid_points: torch.Tensor) -> BBoxLabelContainer:
        n, num_classes = element_preds.labels.shape
        target = BBoxLabelContainer(torch.zeros((n, 4)), torch.zeros((n, num_classes)))
        element_gts = BBoxLabelContainer(element_gts.bboxes.float(), element_gts.labels.float())

        assigner_matrix = self.simOTA(element_gts, element_preds, grid_points)
        for i, mask in enumerate(assigner_matrix):
            target.bboxes[mask, :] = element_gts.bboxes[i, :]
            target.labels[mask, :] = element_gts.labels[i, :]

        return target

    def simOTA(self, ground_truths: BBoxLabelContainer, predictions: BBoxLabelContainer, grid_points: torch.Tensor) -> torch.Tensor:
        """
        ## Args
        - gt_bboxes: tensor of shape (m, 4) where m is the number of labeled bounding boxes. BBoxes should
                     be in form (x_min, y_min, x_max, y_max).
        - gt_labels: tensor of shape (m, c) where c is the number of classes that the model should predict.
                     The labels should be one-hot encoded.
        - grid_points: tensor of shape (n, 2) where n is the number of predictions made by the model
        - pred_labels: tensor of shape (n, c). Predictions should be in the probability form, i.e., each element in [0, 1]
        - pred_bboxes: tensor of shape (n, 4). BBoxes should be in form (x_min, y_min, x_max, y_max)
        ## Returns
        - a tensor of shape (m, n)
        """
        m, _ = ground_truths.bboxes.shape
        n, _ = predictions.bboxes.shape
        assigner_matrix = torch.zeros((m, n), dtype=torch.bool)
        cost_matrix = torch.zeros((m, n))
        IoU_scores = torchvision.ops.box_iou(ground_truths.bboxes, predictions.bboxes) # Shape (m, n)

        for i, (gt_bbox, gt_label, IoU_score) in enumerate(zip(ground_truths.bboxes, ground_truths.labels, IoU_scores)):
            sorted_IoUs, _ = IoU_score.sort(descending=True)
            top_k = max(1, int(sorted_IoUs[:self.q].sum().round())) # Note that q <= n. In all practical applications this should hold.
            
            cost: torch.Tensor = self.cls_cost(predictions.labels, gt_label, IoU=IoU_score)
            cost += 3 * self.reg_cost(IoU_score)
            cost += self.center_cost(gt_bbox, grid_points)

            lowest_indices = cost.argsort()[:top_k] # q <= n implies k <= n
            assigner_matrix[i, lowest_indices] = 1
            cost_matrix[i] = cost
        
        self._make_assigner_matrix_feasible(assigner_matrix, cost_matrix)

        return assigner_matrix

    def _make_assigner_matrix_feasible(self, assigner_matrix: torch.Tensor, cost_matrix: torch.Tensor) -> None:
        """
        Makes sure that the assigner matrix is feasible, i.e., each column sum is at most one. Note that this implementation 
        guarantees that each grid point predicts at most one ground truth but it is possible that some ground truth gets no 
        grid points assigned to it. However, in practice this should be extremely rare.
        ## Args
        - assigner_matrix: tensor of shape (m, n). m = number of ground truths, n = number of predictions
        - cost_matrix: tensor of shape (m, n) with the calculated cost values.
        """
        column_mask = assigner_matrix.sum(dim=0) > 1
        row_mask = cost_matrix.argmin(dim=0)[column_mask]

        assigner_matrix[:, column_mask] = 0 # Assign one to the column element with lowest cost. All other elements to zero.
        assigner_matrix[row_mask, column_mask] = 1