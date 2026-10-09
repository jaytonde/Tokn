"""flash_attn_with_kvcache graph capture: does a ZERO block_table at capture and
a REAL block_table at replay change the result vs eager with the real table?

Usage: PYTHONPATH=/home/Tokn .venv/bin/python debug/flashtest2.py
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
real_bt = torch.tensor([[0, 1, 2, 3]], device=dev, dtype=torch.int32)


def run_eager(cache_seqlen):
    cache_seqlens = torch.tensor([cache_seqlen], device=dev, dtype=torch.int32)
    return flash_attn_with_kvcache(
        q.unsqueeze(1), kvcache[0, 0], kvcache[1, 0],
        cache_seqlens=cache_seqlens, block_table=real_bt,
        softmax_scale=scale, causal=True,
    ).squeeze(1)


def run_graph(capture_bt, run_seqlen):
    bt_buf = torch.zeros(1, 8, device=dev, dtype=torch.int32)
    bt_buf[:1, : capture_bt.size(1)] = capture_bt
    lens_buf = torch.zeros(1, device=dev, dtype=torch.int32)
    lens_buf[0] = run_seqlen  # capture and run same seqlen
    out = torch.zeros(1, 16, head_dim, device=dev, dtype=dtype)
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode():
        out[:] = flash_attn_with_kvcache(
            q.unsqueeze(1), kvcache[0, 0], kvcache[1, 0],
            cache_seqlens=lens_buf, block_table=bt_buf,
            softmax_scale=scale, causal=True,
        ).squeeze(1)
        with torch.cuda.graph(g):
            out[:] = flash_attn_with_kvcache(
                q.unsqueeze(1), kvcache[0, 0], kvcache[1, 0],
                cache_seqlens=lens_buf, block_table=bt_buf,
                softmax_scale=scale, causal=True,
            ).squeeze(1)
        g.replay()
    torch.cuda.synchronize()
    return out.detach().clone()


def main():
    for label, capture_bt in [("real bt", real_bt), ("zero bt", torch.zeros(1, 8, device=dev, dtype=torch.int32))]:
        for run in [100, 300]:
            ref = run_eager(run)
            got = run_graph(capture_bt, run)
            md = (ref - got).abs().max().item()
            print(f"{label} run_seqlen={run}: max|d|= {md:.5g} bit_eq={torch.equal(ref, got)}", flush=True)


if __name__ == "__main__":
    main()
