"""Numeric check: is new RMSNorm var/y bit-identical to old on real model data?"""
import sys
sys.path.insert(0, "/home/Tokn")
import torch
from layers.layernorm import RMSNorm

torch.manual_seed(0)
dev = "cuda"

def old_rmsnorm(x, residual, hidden, eps, weight):
    x = x.contiguous()
    n_rows = x.numel() // hidden
    xf = torch.empty(n_rows, hidden, dtype=torch.float32, device=x.device)
    res = torch.empty_like(x) if residual is not None else None
    if residual is not None:
        xf.copy_((x.float() + residual.float()))
        res.copy_((x.float() + residual.float()).to(x.dtype))
    else:
        xf.copy_(x.float())
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    y = torch.empty_like(x)
    y.copy_(((xf * torch.rsqrt(var + eps)).to(x.dtype)) * weight)
    if residual is None:
        return y
    return y, res

for hidden in (1024, 128):
    for has_res in (False, True):
        x = (torch.randn(5, hidden, dtype=torch.bfloat16, device=dev) * 3).to(torch.bfloat16)
        residual = (torch.randn(5, hidden, dtype=torch.bfloat16, device=dev) * 2).to(torch.bfloat16) if has_res else None
        weight = torch.randn(hidden, dtype=torch.bfloat16, device=dev)
        mod = RMSNorm(hidden, eps=1e-6)
        mod.weight.data.copy_(weight)
        mod = mod.to(dev)
        with torch.inference_mode():
            new_y = mod(x, residual.clone() if residual is not None else None)
        old = old_rmsnorm(x, residual.clone() if residual is not None else None, hidden, 1e-6, weight)
        old_y = old[0] if has_res else old
        if has_res:
            # compare res too
            print(f"hidden={hidden} res={has_res}: y bit_eq={torch.equal(new_y[0], old_y)} max|d|={(new_y[0].float()-old_y.float()).abs().max().item():.3e} "
                  f"res bit_eq={torch.equal(new_y[1], old[1])}")
        else:
            print(f"hidden={hidden} res={has_res}: y bit_eq={torch.equal(new_y, old_y)} max|d|={(new_y.float()-old_y.float()).abs().max().item():.3e}")
