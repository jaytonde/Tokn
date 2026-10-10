"""Microbenchmark non-GEMM decode ops at bs=5 to build a real kernel budget."""
import sys
sys.path.insert(0, "/home/Tokn")
import torch
from utils.configs import ServerConfigs
from src.engine import Engine
from utils.sampling_params import SamplingParams
from layers.layernorm import RMSNorm
from layers.rotary_embedding import RotaryEmbedding
from utils.context import set_context, get_context

torch.manual_seed(0)
dev = "cuda"; dtype = torch.bfloat16

def timeit(fn, n=300):
    for _ in range(30): fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(n): fn()
    t1.record(); torch.cuda.synchronize()
    return t0.elapsed_time(t1) / n

bs = 5
H = 1024; HD = 128
x = torch.randn(bs, H, device=dev, dtype=dtype)
r = torch.randn(bs, H, device=dev, dtype=dtype)
w = torch.randn(H, device=dev, dtype=dtype)

rn = RMSNorm(H).to(device=dev, dtype=dtype)
t = timeit(lambda: rn(x, r))
print(f"RMSNorm hidden (convert_add+reduce+rms_scale): {t*1000:.2f} us/call  x113 = {t*113*1000:.0f} us/step")

rn_small = RMSNorm(HD).to(device=dev, dtype=dtype)
xs = torch.randn(bs, HD, device=dev, dtype=dtype)
rs = torch.randn(bs, HD, device=dev, dtype=dtype)
t = timeit(lambda: rn_small(xs))
print(f"RMSNorm head_dim (no residual): {t*1000:.2f} us/call  x56 = {t*56*1000:.0f} us/step")

# rotary
rope = RotaryEmbedding(HD, HD, 4096*32, 10000.0).to(device=dev, dtype=dtype)
pos = torch.randint(0, 2000, (bs,), device=dev, dtype=torch.long)
q = torch.randn(bs, 16, HD, device=dev, dtype=dtype)
k = torch.randn(bs, 8, HD, device=dev, dtype=dtype)
t = timeit(lambda: rope(pos, q, k))
print(f"rotary q+k: {t*1000:.2f} us/call  x28 = {t*28*1000:.0f} us/step")

# embed lookup (gather)
ids = torch.randint(0, 151936, (bs,), device=dev, dtype=torch.long)
emb = torch.randn(151936, H, device=dev, dtype=dtype)
t = timeit(lambda: emb[ids])
print(f"embed gather: {t*1000:.2f} us/call")

# silu * mul
import torch.nn.functional as F
gu = torch.randn(bs, 6144, device=dev, dtype=dtype)
def silu_and_mul(g):
    a, b = g.chunk(2, -1)
    return F.silu(a) * b
t = timeit(lambda: silu_and_mul(gu))
print(f"silu*mul eager: {t*1000:.2f} us/call")

# torch.compile version (as used in production)
from layers.activation import SiluAndMul
sam = SiluAndMul().to(device=dev, dtype=dtype)
_ = sam(gu); torch.cuda.synchronize()
t = timeit(lambda: sam(gu))
print(f"silu*mul torch.compile: {t*1000:.2f} us/call")

# argmax over vocab (sampler)
logits = torch.randn(bs, 151936, device=dev, dtype=dtype)
t = timeit(lambda: torch.argmax(logits, dim=-1))
print(f"argmax vocab: {t*1000:.2f} us/call")
