from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
) -> torch.Tensor:
    """
    This code implements RoPE, or rotary positional embeddings, 
    for transformer attention.This is how position information 
    gets mixed into query/key vectors without adding a separate 
    position embedding tensor
    """
    x1, x2 = torch.chunk(x.float(), 2, dim=-1) #splits the last dimension of x into two halves
    y1 = x1 * cos - x2 * sin #applies a 2D rotation using cos and sin
    y2 = x2 * cos + x1 * sin #applies a 2D rotation using cos and sin

    return torch.cat((y1, y2), dim=-1).to(x.dtype)

class RotaryEmbedding(nn.Module):
    def __init__(self,
                 head_size: int,
                 rotary_dim: int,
                 max_postiona_embeddings: int,
                 base: float) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float)/ rotary_dim)) #torch.arange(0, rotary_dim, 2, dtype=torch.float) creates 0, 2, 4, ... up to rotary_dim - 2.It steps by 2 because RoPE works on pairs of dimensions.
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
