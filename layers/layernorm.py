import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def rms_scale_kernel(
    x_ptr,        # fp32 (rows, n)
    inv_ptr,      # fp32 (rows,)
    weight_ptr,   # bf16 (n,)
    y_ptr,        # bf16 (rows, n)
    n_cols: tl.constexpr,
):
    """Fused tail of RMSNorm: y = round_bf16(x * inv) * weight (bf16 mult)."""
    row = tl.program_id(0)
    offs = row * n_cols + tl.arange(0, n_cols)
    x = tl.load(x_ptr + offs)
    inv = tl.load(inv_ptr + row)
    w = tl.load(weight_ptr + tl.arange(0, n_cols))
    x_bf = (x * inv).to(tl.bfloat16)
    tl.store(y_ptr + offs, x_bf * w)


class RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps    = eps
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Compute the RMS statistics (var, inv) in torch to stay bit-identical
        # with the previous implementation (reduction order matters in bf16);
        # only the post-stat elementwise tail is fused into Triton.
        xf = x.float()
        res = None
        if residual is not None:
            xf = xf + residual.float()
            res = xf.to(x.dtype)
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        inv = torch.rsqrt(var + self.eps)

        n_rows = xf.numel() // self.hidden_size
        y = torch.empty_like(x)
        rms_scale_kernel[(n_rows,)](
            xf.view(n_rows, self.hidden_size),
            inv.view(n_rows),
            self.weight,
            y.view(n_rows, self.hidden_size),
            self.hidden_size,
        )
        if residual is None:
            return y
        return y, res
