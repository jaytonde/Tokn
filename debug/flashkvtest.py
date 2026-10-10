"""Test: can we fold store_kvcache into flash_attn_with_kvcache via k/v args,
and is it bit-identical to the current (store then attn-without-k/v) path?
"""
import torch

torch.manual_seed(1)
dev = "cuda"
from flash_attn import flash_attn_with_kvcache

nheads, kv_heads, hd = 16, 8, 128
block_size = 256
scale = hd ** -0.5

def current_path(bs, seqlen):
    # emulate the current tokn decode: store k,v into cache then attn
    nblocks = max(1, (seqlen + block_size - 1) // block_size)
    q = torch.randn(bs, 1, nheads, hd, dtype=torch.bfloat16, device=dev)
    k = torch.randn(bs, 1, kv_heads, hd, dtype=torch.bfloat16, device=dev)
    v = torch.randn(bs, 1, kv_heads, hd, dtype=torch.bfloat16, device=dev)
    k_cache = torch.zeros(bs, block_size, kv_heads, hd, dtype=torch.bfloat16, device=dev)
    v_cache = torch.zeros(bs, block_size, kv_heads, hd, dtype=torch.bfloat16, device=dev)
    # block_table: seq occupies blocks 0..nblocks-1 contiguously; new token at slot seqlen
    bt = torch.zeros(bs, nblocks + 1, dtype=torch.int32, device=dev)
    for b in range(bs):
        bt[b, :nblocks] = torch.arange(nblocks, device=dev)
    seqlens = torch.full((bs,), seqlen, dtype=torch.int32, device=dev)
    # emulate store_kvcache: write k,v at slot (block seqlen//256, offset seqlen%256)
    block_idx = seqlen // block_size
    offset = seqlen % block_size
    k_cache[:, block_idx, :, :] = k[:, 0, :, :]
    v_cache[:, block_idx, :, :] = v[:, 0, :, :]
    out = flash_attn_with_kvcache(
        q, k_cache, v_cache, cache_seqlens=seqlens, block_table=bt,
        softmax_scale=scale, causal=True, num_splits=0,
    )
    return out.squeeze(1)

def fused_path(bs, seqlen):
    nblocks = max(1, (seqlen + block_size - 1) // block_size)
    q = torch.randn(bs, 1, nheads, hd, dtype=torch.bfloat16, device=dev)
    k = torch.randn(bs, 1, kv_heads, hd, dtype=torch.bfloat16, device=dev)
    v = torch.randn(bs, 1, kv_heads, hd, dtype=torch.bfloat16, device=dev)
    k_cache = torch.zeros(bs, block_size, kv_heads, hd, dtype=torch.bfloat16, device=dev)
    v_cache = torch.zeros(bs, block_size, kv_heads, hd, dtype=torch.bfloat16, device=dev)
    bt = torch.zeros(bs, nblocks + 1, dtype=torch.int32, device=dev)
    for b in range(bs):
        bt[b, :nblocks] = torch.arange(nblocks, device=dev)
    seqlens = torch.full((bs,), seqlen, dtype=torch.int32, device=dev)
    # pass k,v directly: flash writes them into cache at position cache_seqlens
    out = flash_attn_with_kvcache(
        q, k_cache, v_cache, k=k, v=v, cache_seqlens=seqlens, block_table=bt,
        softmax_scale=scale, causal=True, num_splits=0,
    )
    return out.squeeze(1), k_cache[:, block_idx, :, :] if False else None

import itertools
for bs, seqlen in itertools.product((1, 5), (50, 128, 256, 300, 500)):
    torch.manual_seed(1)
    o_current = current_path(bs, seqlen)
    torch.manual_seed(1)
    o_fused, _ = fused_path(bs, seqlen)
    eq = torch.equal(o_current, o_fused)
    md = (o_current.float() - o_fused.float()).abs().max().item()
    print(f"bs={bs} seqlen={seqlen}: store-then-attn vs fused k/v bit_eq={eq} max|d|={md:.3e}")
