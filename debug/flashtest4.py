"""Does eager flash_attn_with_kvcache with a (1,8) padded block_table differ
from eager with the (1,2) unpadded block_table for the same cache_seqlen?

This is the model's exact situation: eager uses context.block_tables (1,nblocks)
while the CUDA graph uses gv["block_tables"] (1,8) with zero padding.

Usage: PYTHONPATH=/home/Tokn .venv/bin/python debug/flashtest4.py
"""

import sys

sys.path.insert(0, "/home/Tokn")

import torch
from flash_attn import flash_attn_with_kvcache

torch.manual_seed(0)
dev = "cuda"
dtype = torch.bfloat16

num_blocks = 8
block_size = 256
num_kv_heads = 8
head_dim = 128
cache_shape = (2, 1, num_blocks, block_size, num_kv_heads, head_dim)
kvcache = torch.zeros(cache_shape, device=dev, dtype=dtype)
kvcache[0, 0].normal_()
kvcache[1, 0].normal_()
scale = head_dim ** -0.5

q = torch.randn(1, 16, head_dim, device=dev, dtype=dtype) * 0.1


def run(cache_seqlen, bt):
    cache_seqlens = torch.tensor([cache_seqlen], device=dev, dtype=torch.int32)
    return flash_attn_with_kvcache(
        q.unsqueeze(1), kvcache[0, 0], kvcache[1, 0],
        cache_seqlens=cache_seqlens, block_table=bt,
        softmax_scale=scale, causal=True,
    ).squeeze(1)


def main():
    for seqlen in [100, 256, 300, 494, 512]:
        nblocks = (seqlen + 255) // 256
        bt_short = torch.arange(nblocks, device=dev, dtype=torch.int32).unsqueeze(0)
        bt_pad0 = torch.zeros(1, 8, device=dev, dtype=torch.int32)
        bt_pad0[:1, :nblocks] = bt_short
        bt_padm1 = torch.full((1, 8), -1, device=dev, dtype=torch.int32)
        bt_padm1[:1, :nblocks] = bt_short
        a = run(seqlen, bt_short)
        b = run(seqlen, bt_pad0)
        md = (a - b).abs().max().item()
        print(f"seqlen={seqlen} nblocks={nblocks}: short-vs-pad0 max|d|= {md:.5g} "
              f"bit_eq={torch.equal(a, b)}", flush=True)
        c = run(seqlen, bt_padm1)
        md2 = (a - c).abs().max().item()
        print(f"  short-vs-padm1 max|d|= {md2:.5g} bit_eq={torch.equal(a, c)}", flush=True)


if __name__ == "__main__":
    main()
