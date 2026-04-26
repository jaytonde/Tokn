import torch
from torch import nn
import torch.nn.functional as F

class SiluAndMul(nn.Module):
    """
    split into two halves
    nonlinearly activate one half
    use the other half as a gate
    multiply them elementwise
    """

    @torch.compile #asks PyTorch to compile/optimize this function
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1) #splits it into two equal parts along the last dimension
        return F.silu(x) * y #One half is the “activation” branch, the other is the “gate” branch.