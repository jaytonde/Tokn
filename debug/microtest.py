"""GPU micro-tests for session-3 candidate optimizations.

1. Is torch.pow(x, 2) bit-identical to x*x for fp32 on this GPU?
2. Does flash_attn_with_kvcache(num_splits=1) differ from num_splits=0
   for benchmark-like decode shapes?
"""
import torch

torch.manual_seed(0)
dev = "cuda"

# --- Test 1: pow vs square ---
x = torch.randn(100000, dtype=torch.float32, device=dev) * 3 - 1
a = torch.pow(x, 2)
b = x * x
print(f"pow(x,2) vs x*x: bit_eq={torch.equal(a, b)} max|d|={(a-b).abs().max().item():.3e}")
# also test the exact shape used by RMSNorm: (rows, 1024) fp32
x2 = torch.randn(5, 1024, dtype=torch.float32, device=dev) * 3 - 1
print(f"rms-shape pow vs sq: bit_eq={torch.equal(torch.pow(x2,2), x2*x2)}")
# negative values too
xn = (torch.randn(5, 1024, dtype=torch.float32, device=dev) * 5) - 10
print(f"neg-heavy pow vs sq: bit_eq={torch.equal(torch.pow(xn,2), xn*xn)}")
# pow with scalar int 2 (as in xf.pow(2)) -- same thing
print(f"xf.pow(2) vs x*x: bit_eq={torch.equal(x2.pow(2), x2*x2)}")

# --- Test 2: flash splitkv ---
from flash_attn import flash_attn_with_kvcache

def run_flash(bs, seqlen, num_splits):
    nheads, kv_heads, hd = 16, 8, 128
    block_size = 256
    nblocks = max(1, (seqlen + block_size - 1) // block_size)
    q = torch.randn(bs, 1, nheads, hd, dtype=torch.bfloat16, device=dev)
    k_cache = torch.randn(bs, block_size, kv_heads, hd, dtype=torch.bfloat16, device=dev)
    v_cache = torch.randn(bs, block_size, kv_heads, hd, dtype=torch.bfloat16, device=dev)
    block_table = torch.zeros(bs, nblocks, dtype=torch.int32, device=dev)
    cache_seqlens = torch.full((bs,), seqlen, dtype=torch.int32, device=dev)
    return flash_attn_with_kvcache(
        q, k_cache, v_cache, cache_seqlens=cache_seqlens,
        block_table=block_table, softmax_scale=hd ** -0.5, causal=True,
        num_splits=num_splits,
    )

for seqlen in (50, 128, 256, 300, 500):
    for bs in (1, 5):
        o0 = run_flash(bs, seqlen, 0)
        o1 = run_flash(bs, seqlen, 1)
        eq = torch.equal(o0, o1)
        md = (o0.float() - o1.float()).abs().max().item()
        print(f"bs={bs} seqlen={seqlen}: num_splits 0 vs 1 bit_eq={eq} max|d|={md:.3e}")
