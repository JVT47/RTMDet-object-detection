from dataclasses import dataclass
import torch

@dataclass
class BBoxLabelContainer:
    """
    A class that holds the bbox and label tensors.
    """
    bboxes: torch.Tensor
    labels: torch.Tensor


def make_RTMDet_grid_points(first_layer_height: int, first_layer_width: int) -> torch.Tensor:
    """
    Constructs a tensor of shape (n, 2) with the RTMDet grid point locations. 

    first_scale_height: the height of the largest model output tensor  
    fist_scale_width: the width of the largest model output tensor  
    """
    scales = [8, 16, 32]
    grid_points = []
    height = first_layer_height
    width = first_layer_width

    for scale in scales:
        points_xx, points_yy = torch.meshgrid(torch.arange(0, height, dtype=torch.float),
                                              torch.arange(0, width, dtype=torch.float), indexing="xy")
        points = torch.stack([points_xx.flatten(), points_yy.flatten()], dim=-1) * scale # shape (H * W, 2) in (x, y) format
        grid_points.append(points)

        height /= 2
        width /= 2
    
    return torch.cat(grid_points, dim=0)


def transform_reg_pred_to_bbox_pred(reg_pred: torch.Tensor, grid_points: torch.Tensor) -> torch.Tensor:
    """
    Transforms reg_pred from the model into bbox coordinates of the form (x_min, y_min, x_max, y_max)

    reg_pred: tensor of shape (n, 4)
    grid_points: tensor of shape (n, 2)
    """
    bbox_pred = torch.cat([grid_points, grid_points], dim=-1)
    bbox_pred[:, :2] -= reg_pred[:, :2]
    bbox_pred[:, 2:] += reg_pred[:, 2:]

    return bbox_pred