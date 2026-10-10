from functools import lru_cache
import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def scale_rotate_kernel(
    xf_ptr,            # fp32 (total_rows, 2*half_dim)
    var_ptr,           # fp32 (total_rows,)
    wq_ptr, wk_ptr,    # bf16 (2*half_dim,)
    cache_ptr,         # cos_sin_cache (max_pos, 1, 2*half_dim)
    pos_ptr,           # (num_tokens,) long
    outq_ptr, outk_ptr,  # bf16 outputs
    num_q_heads,
    num_k_heads,
    n_q_rows,          # num_tokens * num_q_heads
    cache_stride,
    half_dim: tl.constexpr,
    eps,
):
    """Fused RMSNorm-scale + RoPE for q and k in ONE launch.

    For each (token, head): scale = round_bf16(xf*rsqrt(var+eps)) * w (per
    head_dim), then rotate with cos/sin from the position cache. Bit-identical
    to the old (qk_scale_kernel then rotary_emb_merged_kernel): same bf16
    roundings, same fp32 rotary math."""
    token = tl.program_id(0)
    h = tl.program_id(1)
    lo = tl.arange(0, half_dim)
    pos = tl.load(pos_ptr + token)
    cb = pos * cache_stride + lo
    cos = tl.load(cache_ptr + cb).to(tl.float32)
    sin = tl.load(cache_ptr + cb + half_dim).to(tl.float32)

    if h < num_q_heads:
        row = token * num_q_heads + h
        w_ptr = wq_ptr
        out_ptr = outq_ptr
        out_base = row * (2 * half_dim)
    else:
        hh = h - num_q_heads
        row = n_q_rows + token * num_k_heads + hh
        w_ptr = wk_ptr
        out_ptr = outk_ptr
        out_base = (token * num_k_heads + hh) * (2 * half_dim)

    inv = tl.rsqrt(tl.load(var_ptr + row) + eps)
    x1 = tl.load(xf_ptr + row * (2 * half_dim) + lo)
    x2 = tl.load(xf_ptr + row * (2 * half_dim) + half_dim + lo)
    w1 = tl.load(w_ptr + lo)
    w2 = tl.load(w_ptr + half_dim + lo)
    # match qk_scale_kernel: round_bf16(x*inv) then * w, stored rounded to bf16
    s1 = ((x1 * inv).to(tl.bfloat16) * w1).to(tl.bfloat16)
    s2 = ((x2 * inv).to(tl.bfloat16) * w2).to(tl.bfloat16)

    y1 = s1.to(tl.float32) * cos - s2.to(tl.float32) * sin
    y2 = s2.to(tl.float32) * cos + s1.to(tl.float32) * sin
    tl.store(out_ptr + out_base + lo, y1.to(out_ptr.dtype.element_ty))
    tl.store(out_ptr + out_base + half_dim + lo, y2.to(out_ptr.dtype.element_ty))


@triton.jit
def rotary_emb_merged_kernel(
    q_ptr, k_ptr,
    cache_ptr,      # cos_sin_cache (max_pos, 1, 2*half_dim)
    pos_ptr,        # (num_tokens,) long positions
    outq_ptr, outk_ptr,
    num_q_heads,
    num_k_heads,
    cache_stride,   # row stride of the cache (2*half_dim)
    half_dim: tl.constexpr,
):
    """Fused RoPE for both q and k in ONE launch (bit-identical to two calls).

    program (token, h) with h in [0, num_q_heads+num_k_heads); h < num_q_heads
    applies to q else k. Same cos/sin cache lookup by position, same fp32 math.
    """
    token = tl.program_id(0)
    h = tl.program_id(1)
    offs = tl.arange(0, half_dim)
    pos = tl.load(pos_ptr + token)
    cb = pos * cache_stride + offs
    cos = tl.load(cache_ptr + cb).to(tl.float32)
    sin = tl.load(cache_ptr + cb + half_dim).to(tl.float32)
    if h < num_q_heads:
        base = (token * num_q_heads + h) * (2 * half_dim)
        x1 = tl.load(q_ptr + base + offs).to(tl.float32)
        x2 = tl.load(q_ptr + base + half_dim + offs).to(tl.float32)
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        tl.store(outq_ptr + base + offs, y1.to(q_ptr.dtype.element_ty))
        tl.store(outq_ptr + base + half_dim + offs, y2.to(q_ptr.dtype.element_ty))
    else:
        hh = h - num_q_heads
        base = (token * num_k_heads + hh) * (2 * half_dim)
        x1 = tl.load(k_ptr + base + offs).to(tl.float32)
        x2 = tl.load(k_ptr + base + half_dim + offs).to(tl.float32)
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        tl.store(outk_ptr + base + offs, y1.to(k_ptr.dtype.element_ty))
        tl.store(outk_ptr + base + half_dim + offs, y2.to(k_ptr.dtype.element_ty))


def apply_rotary_emb_merged(
        query: torch.Tensor,
        key: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused RoPE for q+k in one kernel launch (bit-identical to torch)."""
    num_tokens = query.shape[0]
    num_q_heads = query.shape[1]
    num_k_heads = key.shape[1]
    half_dim = query.shape[2] // 2
    outq = torch.empty_like(query)
    outk = torch.empty_like(key)
    rotary_emb_merged_kernel[(num_tokens, num_q_heads + num_k_heads)](
        query, key, cos_sin_cache, positions, outq, outk,
        num_q_heads, num_k_heads, cos_sin_cache.stride(0), half_dim,
    )
    return outq, outk


@triton.jit
def rotary_emb_fused_kernel(
    x_ptr,
    cache_ptr,      # cos_sin_cache (max_pos, 1, 2*half_dim)
    pos_ptr,        # (num_tokens,) long positions
    out_ptr,
    num_heads,
    cache_stride,   # row stride of the cache (2*half_dim)
    half_dim: tl.constexpr,
):
    """Fused RoPE that also does the cos/sin cache lookup by position.

    Reads cos = cache[pos, 0:half_dim], sin = cache[pos, half_dim:2*half_dim]
    directly, eliminating the per-step indexSelect gather (1 kernel/layer).
    Bit-identical to the old (gather then apply_rotary_emb): same bf16 cache
    values, same fp32 upcast, same math."""
    token = tl.program_id(0)
    head = tl.program_id(1)
    row = token * num_heads + head
    base = row * (2 * half_dim)
    h = tl.arange(0, half_dim)

    x1 = tl.load(x_ptr + base + h).to(tl.float32)
    x2 = tl.load(x_ptr + base + half_dim + h).to(tl.float32)
    pos = tl.load(pos_ptr + token)
    cb = pos * cache_stride + h
    cos = tl.load(cache_ptr + cb).to(tl.float32)
    sin = tl.load(cache_ptr + cb + half_dim).to(tl.float32)

    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    tl.store(out_ptr + base + h, y1.to(x_ptr.dtype.element_ty))
    tl.store(out_ptr + base + half_dim + h, y2.to(x_ptr.dtype.element_ty))


def apply_rotary_emb_fused(
        x: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    """Fused RoPE with in-kernel cos/sin cache lookup (bit-identical to torch)."""
    x = x.contiguous()
    num_tokens = x.shape[0]
    num_heads = x.shape[1]
    half_dim = x.shape[2] // 2
    out = torch.empty_like(x)
    rotary_emb_fused_kernel[(num_tokens, num_heads)](
        x, cos_sin_cache, positions, out, num_heads, cos_sin_cache.stride(0), half_dim,
    )
    return out


@triton.jit
def rotary_emb_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    num_heads,
    cos_row_stride,
    half_dim: tl.constexpr,
):
    """Fused RoPE: y1 = x1*cos - x2*sin ; y2 = x2*cos + x1*sin, fp32 then bf16.

    x layout: (num_tokens, num_heads, 2*half_dim) contiguous.
    cos/sin layout: (num_tokens, 1, half_dim), stride 0 = cos_row_stride.
    """
    token = tl.program_id(0)
    head = tl.program_id(1)
    row = token * num_heads + head
    base = row * (2 * half_dim)
    h = tl.arange(0, half_dim)

    x1 = tl.load(x_ptr + base + h).to(tl.float32)
    x2 = tl.load(x_ptr + base + half_dim + h).to(tl.float32)
    cb = token * cos_row_stride + h
    cos = tl.load(cos_ptr + cb).to(tl.float32)
    sin = tl.load(sin_ptr + cb).to(tl.float32)

    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    tl.store(out_ptr + base + h, y1.to(x_ptr.dtype.element_ty))
    tl.store(out_ptr + base + half_dim + h, y2.to(x_ptr.dtype.element_ty))


def apply_rotary_emb(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
) -> torch.Tensor:
    """RoPE matching torch's apply_rotary_emb bit-for-bit (fp32, single round)."""
    x = x.contiguous()
    num_tokens = x.shape[0]
    num_heads = x.shape[1]
    half_dim = x.shape[2] // 2
    out = torch.empty_like(x)
    rotary_emb_kernel[(num_tokens, num_heads)](
        x, cos, sin, out, num_heads, cos.stride(0), half_dim,
    )
    return out


class RotaryEmbedding(nn.Module):
    def __init__(self,
                 head_size: int,
                 rotary_dim: int,
                 max_postiona_embeddings: int,
                 base: float) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float)/ rotary_dim))
        t = torch.arange(max_postiona_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(self,
                positions: torch.Tensor,
                query: torch.Tensor,
                key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_rotary_emb_merged(query, key, positions, self.cos_sin_cache)

    def scale_rotate(self,
                     positions: torch.Tensor,
                     xf: torch.Tensor,
                     var: torch.Tensor,
                     w_q: torch.Tensor,
                     w_k: torch.Tensor,
                     q_shape,
                     k_shape,
                     eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused RMSNorm-scale + RoPE for q and k (from QKNorm.prepare xf/var).

        Bit-identical to qk_scale then rotary, in one kernel launch."""
        num_tokens = q_shape[0]
        num_q_heads = q_shape[1]
        num_k_heads = k_shape[1]
        half_dim = q_shape[2] // 2
        n_q_rows = num_tokens * num_q_heads
        outq = torch.empty(q_shape, dtype=torch.bfloat16, device=xf.device)
        outk = torch.empty(k_shape, dtype=torch.bfloat16, device=xf.device)
        scale_rotate_kernel[(num_tokens, num_q_heads + num_k_heads)](
            xf, var, w_q, w_k, self.cos_sin_cache, positions, outq, outk,
            num_q_heads, num_k_heads, n_q_rows, self.cos_sin_cache.stride(0),
            half_dim, eps,
        )
        return outq, outk


def get_rope(
        head_size: int,
        rotary_dim: int,
        max_postion: int,
        base: float):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_postion, base)
    return rotary_emb
