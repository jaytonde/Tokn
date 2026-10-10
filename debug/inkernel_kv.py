"""Test whether flash_attn_with_kvcache(k, v) in-kernel store is bit-identical
to (store_kvcache then flash_attn_with_kvcache) and faster.

Usage: PYTHONPATH=/home/Tokn .venv/bin/python debug/inkernel_kv.py
"""

import torch
from flash_attn import flash_attn_with_kvcache
from layers.attention import store_kvcache

torch.manual_seed(0)
dev = "cuda"
dtype = torch.bfloat16
bs = 5
num_heads = 16
num_kv_heads = 8
head_dim = 128
block_size = 256
num_blocks = 56
scale = head_dim ** -0.5

def timeit(fn, n=200):
    for _ in range(20): fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(n): fn()
    t1.record(); torch.cuda.synchronize()
    return t0.elapsed_time(t1) / n

def make_state(seqlen):
    k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device=dev, dtype=dtype) * 0.1
    v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device=dev, dtype=dtype) * 0.1
    block_table = torch.full((bs, 8), -1, dtype=torch.int32, device=dev)
    for i in range(bs):
        nblocks = (seqlen + block_size - 1) // block_size
        block_table[i, :nblocks] = torch.arange(i * 8, i * 8 + nblocks, device=dev)
    ctx = torch.full((bs,), seqlen, dtype=torch.int32, device=dev)
    return k_cache, v_cache, block_table, ctx

# slot for the new token = block 0 offset seqlen (per row: block i*8, offset within)
def slots(seqlen):
    b = seqlen // block_size
    off = seqlen % block_size
    return torch.tensor([i * 8 * block_size + b * block_size + off for i in range(bs)],
                        dtype=torch.int32, device=dev)

for seqlen in (50, 120, 200, 256, 300, 400):
    k_cache, v_cache, block_table, ctx = make_state(seqlen)
    slot = slots(seqlen)
    q = torch.randn(bs, 1, num_heads, head_dim, device=dev, dtype=dtype) * 0.05
    k = torch.randn(bs, 1, num_kv_heads, head_dim, device=dev, dtype=dtype) * 0.05
    v = torch.randn(bs, 1, num_kv_heads, head_dim, device=dev, dtype=dtype) * 0.05

    with torch.inference_mode():
        # Path A: separate store + flash
        kcA = k_cache.clone(); vcA = v_cache.clone()
        store_kvcache(k.squeeze(1), v.squeeze(1), kcA, vcA, slot)
        oA = flash_attn_with_kvcache(q, kcA, vcA, cache_seqlens=ctx, block_table=block_table,
                                     softmax_scale=scale, causal=True, num_splits=1)
        # Path B: in-kernel store
        kcB = k_cache.clone(); vcB = v_cache.clone()
        oB = flash_attn_with_kvcache(q, kcB, vcB, k=k, v=v,
                                     cache_seqlens=ctx, block_table=block_table,
                                     softmax_scale=scale, causal=True, num_splits=1)
    torch.cuda.synchronize()
    eq_out = torch.equal(oA, oB)
    eq_kc = torch.equal(kcA, kcB)
    eq_vc = torch.equal(vcA, vcB)
    md = (oA.float() - oB.float()).abs().max().item()
    print(f"seqlen~{seqlen}: out_eq={eq_out} kc_eq={eq_kc} vc_eq={eq_vc} maxdiff={md:.3e}")

    with torch.inference_mode():
        ta = timeit(lambda: (store_kvcache(k.squeeze(1), v.squeeze(1), kcA, vcA, slot),
                             flash_attn_with_kvcache(q, kcA, vcA, cache_seqlens=ctx,
                                                     block_table=block_table, softmax_scale=scale,
                                                     causal=True, num_splits=1)))
        tb = timeit(lambda: flash_attn_with_kvcache(q, kcB, vcB, k=k, v=v,
                                                    cache_seqlens=ctx, block_table=block_table,
                                                    softmax_scale=scale, causal=True, num_splits=1))
    print(f"   A(store+flash)={ta*1000:.2f} us  B(in-kernel)={tb*1000:.2f} us  saved={1000*(ta-tb):.2f} us")
