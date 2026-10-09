"""Test whether flash_attn_with_kvcache, captured in a CUDA graph with a
degenerate cache_seqlen, reproduces an eager call with a real cache_seqlen.

Usage: PYTHONPATH=/home/Tokn .venv/bin/python debug/flashtest.py
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

scale = head_dim ** -0.5

block_table = torch.tensor([[0, 1, 2, 3]], device=dev, dtype=torch.int32)


def run_eager(q, cache_seqlen):
    cache_seqlens = torch.tensor([cache_seqlen], device=dev, dtype=torch.int32)
    return flash_attn_with_kvcache(
        q.unsqueeze(1), kvcache[0, 0], kvcache[1, 0],
        cache_seqlens=cache_seqlens, block_table=block_table,
        softmax_scale=scale, causal=True,
    ).squeeze(1)


def run_graph(q, capture_seqlen, run_seqlen):
    lens_buf = torch.zeros(1, device=dev, dtype=torch.int32)
    lens_buf[0] = capture_seqlen
    out = torch.zeros(1, 16, head_dim, device=dev, dtype=dtype)
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode():
        out[:] = flash_attn_with_kvcache(
            q.unsqueeze(1), kvcache[0, 0], kvcache[1, 0],
            cache_seqlens=lens_buf, block_table=block_table,
            softmax_scale=scale, causal=True,
        ).squeeze(1)
        with torch.cuda.graph(g):
            out[:] = flash_attn_with_kvcache(
                q.unsqueeze(1), kvcache[0, 0], kvcache[1, 0],
                cache_seqlens=lens_buf, block_table=block_table,
                softmax_scale=scale, causal=True,
            ).squeeze(1)
        lens_buf[0] = run_seqlen
        g.replay()
    torch.cuda.synchronize()
    return out.detach().clone()


def main():
    kvcache[0, 0].normal_()
    kvcache[1, 0].normal_()

    q = torch.randn(1, 16, head_dim, device=dev, dtype=dtype) * 0.1

    for cap, run in [(0, 100), (0, 300), (300, 100), (100, 100)]:
        ref = run_eager(q, run)
        got = run_graph(q, cap, run)
        md = (ref - got).abs().max().item()
        print(f"capture_seqlen={cap} run_seqlen={run}: max|d|= {md:.5g} "
              f"bit_eq={torch.equal(ref, got)}", flush=True)


if __name__ == "__main__":
    main()
