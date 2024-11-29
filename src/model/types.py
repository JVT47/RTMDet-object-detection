import torch


type RTMDetCls = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
type RTMDetReg = tuple[torch.Tensor, torch.Tensor, torch.Tensor]