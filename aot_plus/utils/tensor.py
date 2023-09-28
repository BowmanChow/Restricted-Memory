import torch

def bchw_2_lbc(tensor: torch.Tensor):
    n, c, h, w = tensor.size()
    tensor_new = tensor.view(n, c, h * w).permute(2, 0, 1)
    return tensor_new.contiguous()