from functools import lru_cache
import torch
from torch import nn
import triton
import triton.language as tl


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


def get_rope(
        head_size: int,
        rotary_dim: int,
        max_postion: int,
        base: float):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_postion, base)
    return rotary_emb
