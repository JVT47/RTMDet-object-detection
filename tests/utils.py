import torch

from rtmdet_object_detection_dev.dataclasses.rtmdet_output import RTMDetOutput


def create_model_output_1() -> RTMDetOutput:
    """
    Creates a mock output of a RTMDet model.
    """
    cls_1 = torch.ones((2, 4, 4, 3)) * -1
    cls_1[0, 1, 1] = torch.tensor([0, 10, 5])
    cls_1 = cls_1.permute(0, 3, 1, 2)
    cls_2 = torch.ones((2, 2, 2, 3)) * -1
    cls_2[1, 1, 1] = torch.tensor([10, 0, -10])
    cls_2 = cls_2.permute(0, 3, 1, 2)
    cls_3 = torch.ones((2, 3, 1, 1)) * -1

    reg_1 = torch.zeros((2, 4, 4, 4))
    reg_1[0, 1, 1] = torch.tensor([1, 1, 0, 0])  # bbox [7, 7, 8, 8]
    reg_1 = reg_1.permute(0, 3, 1, 2)
    reg_2 = torch.zeros((2, 2, 2, 4))
    reg_2[1, 1, 1] = torch.tensor([0, 0, 1, 1])  # bbox [16, 16, 17, 17]
    reg_2 = reg_2.permute(0, 3, 1, 2)
    reg_3 = torch.zeros((2, 4, 1, 1))

    return RTMDetOutput((cls_1, cls_2, cls_3), (reg_1, reg_2, reg_3))


def create_model_output_2() -> RTMDetOutput:
    """
    Creates a mock output of a RTMDet model.
    """
    cls_1 = torch.ones((2, 8, 8, 2)) * -1
    cls_1[0, 0, 1] = torch.tensor([2, -1])
    cls_1[1, 1, 1] = torch.tensor([1, -1])
    cls_1 = cls_1.permute(0, 3, 2, 1)
    cls_2 = torch.ones((2, 4, 4, 2)) * -1
    cls_2[1, 2, 2] = torch.tensor([-1, 2])
    cls_2 = cls_2.permute(0, 3, 2, 1)
    cls_3 = torch.ones((2, 2, 2, 2)) * -1
    cls_3[1, 0, 0] = torch.tensor([-1, 1])
    cls_3 = cls_3.permute(0, 3, 2, 1)

    reg_1 = torch.zeros((2, 8, 8, 4))
    reg_1[0, 0, 1] = torch.tensor([0, 1, 1, 0])  # bbox [0, 7, 1, 8]
    reg_1[1, 1, 1] = torch.tensor([1, 1, 1, 1])  # bbox [7, 7, 9, 9]
    reg_1[1, 4, 4] = torch.tensor([1, 1, 1, 1])  # bbox [31, 31, 33, 33]
    reg_1 = reg_1.permute(0, 3, 2, 1)
    reg_2 = torch.zeros((2, 4, 4, 4))
    reg_2[1, 2, 2] = torch.tensor([2, 2, 2, 2])  # bbox [30, 30, 34, 34]
    reg_2 = reg_2.permute(0, 3, 2, 1)
    reg_3 = torch.zeros((2, 2, 2, 4))
    reg_3[1, 0, 0] = torch.tensor([0, 0, 1, 1])  # bbox [0, 0, 1, 1]
    reg_3 = reg_3.permute(0, 3, 2, 1)

    return RTMDetOutput((cls_1, cls_2, cls_3), (reg_1, reg_2, reg_3))
