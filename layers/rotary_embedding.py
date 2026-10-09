from functools import lru_cache
import torch
from torch import nn
import triton
import triton.language as tl


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
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


def get_rope(
        head_size: int,
        rotary_dim: int,
        max_postion: int,
        base: float):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_postion, base)
    return rotary_emb
