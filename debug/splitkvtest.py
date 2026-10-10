"""Test whether forcing num_splits=1 in flash_attn_with_kvcache avoids the
splitkv+combine kernels and stays bit-identical for our decode shapes.

Usage: PYTHONPATH=/home/Tokn .venv/bin/python debug/splitkvtest.py
"""

import torch
from flash_attn import flash_attn_with_kvcache

torch.manual_seed(0)
dev = "cuda"
dtype = torch.bfloat16

bs = 5
num_heads = 16
num_kv_heads = 8
head_dim = 128
block_size = 256
max_num_blocks = 8
num_blocks = 56
scale = head_dim ** -0.5

k_cache = torch.randn(2, 28, num_blocks, block_size, num_kv_heads, head_dim, device=dev, dtype=dtype) * 0.1
v_cache = torch.randn(2, 28, num_blocks, block_size, num_kv_heads, head_dim, device=dev, dtype=dtype) * 0.1

# block table: row i uses blocks [i*2, i*2+1, ...] (distinct per row). Pad with -1.
block_table = torch.full((bs, max_num_blocks), -1, dtype=torch.int32, device=dev)
for i in range(bs):
    nblocks = (i + 1) % 8 + 1
    block_table[i, :nblocks] = torch.arange(i * 8, i * 8 + nblocks, device=dev)

def make_contexts(seqlen):
    # context length per sequence, spanning 1-2 blocks
    context_lens = torch.tensor([seqlen - i * 7 for i in range(bs)], dtype=torch.int32, device=dev)
    return context_lens

def run(q, kc, vc, ctx, ns):
    return flash_attn_with_kvcache(q, kc, vc,
                                   cache_seqlens=ctx, block_table=block_table,
                                   softmax_scale=scale, causal=True, num_splits=ns)

for seqlen in (50, 120, 200, 256, 300, 400):
    ctx = make_contexts(seqlen)
    q = torch.randn(bs, 1, num_heads, head_dim, device=dev, dtype=dtype) * 0.05
    # paged KV cache: (num_blocks, page_block_size, nheads_k, head_dim)
    kc = k_cache[0, 0].clone()
    vc = v_cache[0, 0].clone()

    with torch.inference_mode():
        o0 = run(q, kc, vc, ctx, 0)
        o1 = run(q, kc, vc, ctx, 1)
    torch.cuda.synchronize()

    eq = torch.equal(o0, o1)
    maxdiff = (o0.float() - o1.float()).abs().max().item()
    print(f"seqlen~{seqlen}: num_splits=0 vs 1 bit_eq={eq} maxdiff={maxdiff:.3e}")

    # timing: warm up both then measure
    def bench(fn, iters=50):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    with torch.inference_mode():
        t0 = bench(lambda: run(q, kc, vc, ctx, 0))
        t1 = bench(lambda: run(q, kc, vc, ctx, 1))
    print(f"   ns=0: {t0*1000:.2f} us   ns=1: {t1*1000:.2f} us   saved {1000*(t0-t1):.2f} us")
