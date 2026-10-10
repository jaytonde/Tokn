import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def convert_add_kernel(
    x_ptr,        # bf16 (rows, n)
    r_ptr,        # bf16 (rows, n), ignored if HAS_RES==False
    xf_ptr,       # fp32 out (rows, n)
    x2_ptr,       # fp32 out (rows, n): x*x (== torch.pow(x,2) bit-identically)
    res_ptr,      # bf16 out (rows, n), ignored if HAS_RES==False
    n_cols: tl.constexpr,
    HAS_RES: tl.constexpr,
):
    """xf = x.to(f32) (+ r.to(f32)); res = round_bf16(xf) when HAS_RES;
    x2 = xf*xf so the var reduction can skip a separate pow kernel.

    xf*xf is bit-identical to torch.pow(xf, 2.0) (verified on this GPU), so
    torch's mean reduction over x2 produces the identical var as the old
    xf.pow(2).mean(-1)."""
    row = tl.program_id(0)
    offs = row * n_cols + tl.arange(0, n_cols)
    x = tl.load(x_ptr + offs).to(tl.float32)
    if HAS_RES:
        r = tl.load(r_ptr + offs).to(tl.float32)
        x = x + r
        tl.store(res_ptr + offs, x.to(x_ptr.dtype.element_ty))
    tl.store(xf_ptr + offs, x)
    tl.store(x2_ptr + offs, x * x)


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


@triton.jit
def qk_convert_kernel(
    q_ptr, k_ptr,      # bf16 q (bs,nqh,D) / k (bs,nkvh,D), row-strided
    xf_ptr,            # fp32 out (total, D)
    x2_ptr,            # fp32 out (total, D): x*x
    n_q_rows,
    q_heads,
    q_stride,          # stride(0) of q
    k_heads,
    k_stride,          # stride(0) of k
    n_cols: tl.constexpr,
):
    """xf = row.to(f32); x2 = xf*xf, for q rows [0,nq) and k rows [nq,total).

    Reads q/k directly from their strided views (they are slices of the qkv
    GEMM output, so dim-0 stride is 4096, not 2048); this avoids the
    per-step reshape copy that makes them contiguous. Per-row identical to the
    separate RMSNorm convert (same values, same per-row reduction inputs)."""
    row = tl.program_id(0)
    offs = tl.arange(0, n_cols)
    if row < n_q_rows:
        b = row // q_heads
        h = row % q_heads
        x = tl.load(q_ptr + b * q_stride + h * n_cols + offs).to(tl.float32)
    else:
        rr = row - n_q_rows
        b = rr // k_heads
        h = rr % k_heads
        x = tl.load(k_ptr + b * k_stride + h * n_cols + offs).to(tl.float32)
    tl.store(xf_ptr + row * n_cols + offs, x)
    tl.store(x2_ptr + row * n_cols + offs, x * x)


@triton.jit
def qk_scale_kernel(
    xf_ptr,            # fp32 (total, D)
    var_ptr,           # fp32 (total,)
    wq_ptr, wk_ptr,    # bf16 (D,)
    y_ptr,             # bf16 (total, D)
    n_q_rows,
    n_cols: tl.constexpr,
    eps,
):
    """y = round_bf16(x*rsqrt(var+eps))*w, w = wq for q rows else wk."""
    row = tl.program_id(0)
    offs = tl.arange(0, n_cols)
    x = tl.load(xf_ptr + row * n_cols + offs)
    var = tl.load(var_ptr + row)
    inv = tl.rsqrt(var + eps)
    if row < n_q_rows:
        w = tl.load(wq_ptr + offs)
    else:
        w = tl.load(wk_ptr + offs)
    x_bf = (x * inv).to(tl.bfloat16)
    tl.store(y_ptr + row * n_cols + offs, x_bf * w)


class QKNorm(nn.Module):
    """RMSNorm applied to q and k with per-row-group weights, in one set of
    kernel launches (3 kernels for both tensors instead of 6).

    The mean reduction is still torch's over each row independently, so every
    row is bit-identical to the old separate q_norm/k_norm calls."""

    def __init__(self, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.head_dim = head_dim
        self.w_q = nn.Parameter(torch.ones(head_dim))
        self.w_k = nn.Parameter(torch.ones(head_dim))

    def prepare(self, q: torch.Tensor, k: torch.Tensor):
        """Return (xf, var) fp32 for q+k rows (convert + mean only).

        Used by the fused scale+rotary kernel (Qwen3Attention bs>=2 path):
        the scale step moves into the rotary kernel, avoiding a separate
        qk_scale launch and a memory round-trip. Per-row values identical to
        the split path (same convert, same torch mean).
        """
        hd = q.shape[-1]
        nq = q.numel() // hd
        nk = k.numel() // hd
        total = nq + nk
        xf = torch.empty(total, hd, dtype=torch.float32, device=q.device)
        x2 = torch.empty(total, hd, dtype=torch.float32, device=q.device)
        qk_convert_kernel[(total,)](q, k, xf, x2, nq, q.shape[1], q.stride(0),
                                    k.shape[1], k.stride(0), hd)
        var = x2.mean(dim=-1, keepdim=True).view(total)
        return xf, var

    def prepare_bs1(self, q: torch.Tensor, k: torch.Tensor):
        """bs=1 fused convert + SEPARATE q/k means -> (xf, var_q, var_k).

        At bs=1 the merged (24,128) mean changes the k-reduce config vs a
        standalone (8,128) tensor, so a single x2.mean(-1) over all rows would
        break bit-identity (see the forward() fallback note). The convert is
        elementwise per-row and IS identical when merged, so one qk_convert
        kernel covers q+k, then the q and k means are torch reductions over
        their own contiguous row slices (same shapes as the old standalone
        q_norm/k_norm calls -> same reduce config -> bit-identical)."""
        hd = q.shape[-1]
        nq = q.numel() // hd
        nk = k.numel() // hd
        total = nq + nk
        xf = torch.empty(total, hd, dtype=torch.float32, device=q.device)
        x2 = torch.empty(total, hd, dtype=torch.float32, device=q.device)
        qk_convert_kernel[(total,)](q, k, xf, x2, nq, q.shape[1], q.stride(0),
                                    k.shape[1], k.stride(0), hd)
        var_q = x2[:nq].mean(dim=-1)   # (nq,), same reduce as standalone q norm
        var_k = x2[nq:].mean(dim=-1)   # (nk,), same reduce as standalone k norm
        return xf, var_q, var_k

    def _rms_one(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Standard no-residual RMSNorm for one tensor (bit-identical to the
        old RMSNorm(x) path: torch mean reduction, Triton convert+scale)."""
        x = x.contiguous()
        n_rows = x.numel() // self.head_dim
        xf = torch.empty(n_rows, self.head_dim, dtype=torch.float32, device=x.device)
        x2 = torch.empty(n_rows, self.head_dim, dtype=torch.float32, device=x.device)
        convert_add_kernel[(n_rows,)](
            x, x, xf, x2, x, self.head_dim, False,
        )
        var = x2.mean(dim=-1, keepdim=True)
        y = torch.empty_like(x)
        rms_scale_kernel[(n_rows,)](xf, var.view(n_rows), w, y, self.head_dim, self.eps)
        return y

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hd = q.shape[-1]
        nq = q.numel() // hd
        nk = k.numel() // hd
        if nk < 16:
            # bs=1 for Qwen3-0.6B: torch's reduce kernel uses a different
            # config for the standalone (8,128) k-tensor than for a merged
            # (24,128) tensor, so merging would break bs=1 bit-identity.
            # Fall back to the separate-norm path (still bit-identical to the
            # old q_norm/k_norm calls).
            return (
                self._rms_one(q, self.w_q).view(q.shape),
                self._rms_one(k, self.w_k).view(k.shape),
            )
        total = nq + nk
        xf = torch.empty(total, hd, dtype=torch.float32, device=q.device)
        x2 = torch.empty(total, hd, dtype=torch.float32, device=q.device)
        qk_convert_kernel[(total,)](q, k, xf, x2, nq, q.shape[1], q.stride(0),
                                    k.shape[1], k.stride(0), hd)
        var = x2.mean(dim=-1, keepdim=True)
        y = torch.empty(total, hd, dtype=q.dtype, device=q.device)
        qk_scale_kernel[(total,)](xf, var.view(total), self.w_q, self.w_k, y, nq, hd, self.eps)
        return y[:nq].view(q.shape), y[nq:].view(k.shape)


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
        x2 = torch.empty(n_rows, self.hidden_size, dtype=torch.float32, device=x.device)
        res = torch.empty_like(x) if residual is not None else None
        convert_add_kernel[(n_rows,)](
            x, residual if residual is not None else x, xf, x2, res,
            self.hidden_size, residual is not None,
        )
        var = x2.mean(dim=-1, keepdim=True)
        y = torch.empty_like(x)
        rms_scale_kernel[(n_rows,)](
            xf, var.view(n_rows), self.weight, y, self.hidden_size, self.eps,
        )
        if residual is None:
            return y
        return y, res
