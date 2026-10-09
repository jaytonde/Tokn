import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def convert_add_kernel(
    x_ptr,        # bf16 (rows, n)
    r_ptr,        # bf16 (rows, n), ignored if HAS_RES==False
    xf_ptr,       # fp32 out (rows, n)
    res_ptr,      # bf16 out (rows, n), ignored if HAS_RES==False
    n_cols: tl.constexpr,
    HAS_RES: tl.constexpr,
):
    """xf = x.to(f32) (+ r.to(f32)); res = round_bf16(xf) when HAS_RES."""
    row = tl.program_id(0)
    offs = row * n_cols + tl.arange(0, n_cols)
    x = tl.load(x_ptr + offs).to(tl.float32)
    if HAS_RES:
        r = tl.load(r_ptr + offs).to(tl.float32)
        x = x + r
        tl.store(res_ptr + offs, x.to(x_ptr.dtype.element_ty))
    tl.store(xf_ptr + offs, x)


@triton.jit
def rms_scale_kernel(
    x_ptr,        # fp32 (rows, n)
    var_ptr,      # fp32 (rows,)
    weight_ptr,   # bf16 (n,)
    y_ptr,        # bf16 (rows, n)
    n_cols: tl.constexpr,
    eps,
):
    """y = round_bf16(x * rsqrt(var + eps)) * weight  (bf16 weight mult)."""
    row = tl.program_id(0)
    offs = row * n_cols + tl.arange(0, n_cols)
    x = tl.load(x_ptr + offs)
    var = tl.load(var_ptr + row)
    inv = tl.rsqrt(var + eps)          # bit-identical to torch.rsqrt
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
        # The RMS statistics (var) are computed in torch so the reduction stays
        # bit-identical with the previous implementation (reduction order matters
        # in bf16). Only the elementwise parts are fused into Triton:
        #   convert (+ residual add)          -> convert_add_kernel
        #   var = mean(xf^2)                  -> torch
        #   y = round_bf16(xf*rsqrt(var+eps))*w -> rms_scale_kernel
        x = x.contiguous()
        n_rows = x.numel() // self.hidden_size
        xf = torch.empty(n_rows, self.hidden_size, dtype=torch.float32, device=x.device)
        res = torch.empty_like(x) if residual is not None else None
        convert_add_kernel[(n_rows,)](
            x, residual if residual is not None else x, xf, res,
            self.hidden_size, residual is not None,
        )
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        y = torch.empty_like(x)
        rms_scale_kernel[(n_rows,)](
            xf, var.view(n_rows), self.weight, y, self.hidden_size, self.eps,
        )
        if residual is None:
            return y
        return y, res
