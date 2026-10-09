"""Does a prior REAL eager flash_attn_with_kvcache call before graph capture
make the captured graph bit-identical to eager?

Usage: PYTHONPATH=/home/Tokn .venv/bin/python debug/flashtest3.py
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


def run_graph(cache_seqlen, warmup_real=False):
    lens_buf = torch.zeros(1, device=dev, dtype=torch.int32)
    bt_buf = real_bt.clone()
    lens_buf[0] = cache_seqlen
    out = torch.zeros(1, 16, head_dim, device=dev, dtype=dtype)
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode():
        if warmup_real:
            # a real eager flash call before capture
            run_eager(cache_seqlen)
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
    for warm in [False, True]:
        for run in [100, 300, 500]:
            ref = run_eager(run)
            got = run_graph(run, warmup_real=warm)
            md = (ref - got).abs().max().item()
            print(f"warmup_real={warm} run_seqlen={run}: max|d|= {md:.5g} bit_eq={torch.equal(ref, got)}", flush=True)


if __name__ == "__main__":
    main()
