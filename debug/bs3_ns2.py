"""bs=3 ctx~280: ns=0 heuristic vs explicit ns, timing + bit-identity, using the engine's exact flash call."""
import sys
sys.path.insert(0, "/home/Tokn")
import torch as T
from utils.configs import ServerConfigs
from src.llm import LLM

cfg = ServerConfigs(
    model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda",
    max_length=2048, max_model_len=2048, tensor_parallel_size=1,
    enforce_eager=False, max_num_seqs=5, max_num_batched_tokens=1024,
)
llm = LLM(model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda", max_model_len=2048, config=cfg)
RUN = llm.RUN

from flash_attn import flash_attn_with_kvcache

bs = 3
head_dim = 128
nheads = 16
nkv = 8
block_size = 256
num_blocks = RUN.num_blocks
k_cache = RUN.kv_cache[0, 0].view(-1, nkv, head_dim)   # (num_blocks*bs, nkv, hd) flat
v_cache = RUN.kv_cache[1, 0].view(-1, nkv, head_dim)

T.manual_seed(0)
seqlens = [280, 290, 300]
# block tables: give each seq 2 blocks at distinct block ids
block_tables = T.zeros(bs, 2, dtype=T.int32, device="cuda")
for i, sl in enumerate(seqlens):
    b0 = 2*i
    block_tables[i,0] = b0; block_tables[i,1] = b0+1
    for p in range(sl):
        slot = (b0 if p < 256 else b0+1)*block_size + (p % 256)
        if p < 256:
            k_cache[slot] = T.randn(nkv, head_dim, dtype=T.bfloat16, device="cuda")
        else:
            k_cache[slot] = T.randn(nkv, head_dim, dtype=T.bfloat16, device="cuda")
    # same for v
    for p in range(sl):
        slot = (b0 if p < 256 else b0+1)*block_size + (p % 256)
        v_cache[slot] = T.randn(nkv, head_dim, dtype=T.bfloat16, device="cuda")

context_lens = T.tensor(seqlens, dtype=T.int32, device="cuda")
q = T.randn(bs, nheads, head_dim, dtype=T.bfloat16, device="cuda")

def run(ns):
    return flash_attn_with_kvcache(
        q.unsqueeze(1), k_cache.view(num_blocks, block_size, nkv, head_dim),
        v_cache.view(num_blocks, block_size, nkv, head_dim),
        cache_seqlens=context_lens, block_table=block_tables,
        softmax_scale=head_dim**-0.5, causal=True, num_splits=ns)

with T.inference_mode():
    out0 = run(0)
    T.cuda.synchronize()
    for ns in (1, 2, 3, 4, 8):
        out = run(ns)
        T.cuda.synchronize()
        eq = T.equal(out0, out)
        md = (out0.float()-out.float()).abs().max().item() if not eq else 0.0
        ev = T.cuda.Event(enable_timing=True); ev2 = T.cuda.Event(enable_timing=True)
        ev.record()
        for _ in range(300): run(ns)
        ev2.record(); T.cuda.synchronize()
        t = ev.elapsed_time(ev2)/300*1000
        ev.record()
        for _ in range(300): run(0)
        ev2.record(); T.cuda.synchronize()
        t0 = ev.elapsed_time(ev2)/300*1000
        print(f"ns={ns:2d}: time={t:7.2f}us (ns0={t0:7.2f}us, d={t-t0:+7.2f})  bit_eq={eq}  maxdiff={md:.2e}")
