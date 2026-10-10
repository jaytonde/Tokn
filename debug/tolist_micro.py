import time, torch
import numpy as np
torch.cuda.init()
x = torch.zeros(5, dtype=torch.int32, device="cuda")

# warmup
for _ in range(50):
    x[:2].tolist()
torch.cuda.synchronize()

N = 2000
# 1: plain tolist
t0 = time.perf_counter()
for _ in range(N):
    r = x[:2].tolist()
t1 = time.perf_counter()
print(f"tolist slice      : {(t1-t0)/N*1e6:6.2f} us")

# 2: dict access + slice + tolist (engine pattern)
gv = {"tbuf_tok": x}
t0 = time.perf_counter()
for _ in range(N):
    r = gv["tbuf_tok"][:2].tolist()
t1 = time.perf_counter()
print(f"dict+slice+list   : {(t1-t0)/N*1e6:6.2f} us")

# 3: numpy read from pinned mirror (no D2H)
pin = torch.zeros(5, dtype=torch.int32, pin_memory=True)
pin_np = pin.numpy()
# sync the values once
pin[:2].copy_(x[:2], non_blocking=True)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    r = pin_np[:2].tolist()
t1 = time.perf_counter()
print(f"pinned np tolist  : {(t1-t0)/N*1e6:6.2f} us")

# 4: pinned np .copy() into a python list
t0 = time.perf_counter()
for _ in range(N):
    r = [int(v) for v in pin_np[:2]]
t1 = time.perf_counter()
print(f"pinned np listcomp: {(t1-t0)/N*1e6:6.2f} us")

# 5: tolist on full (no slice)
t0 = time.perf_counter()
for _ in range(N):
    r = x.tolist()
t1 = time.perf_counter()
print(f"tolist full       : {(t1-t0)/N*1e6:6.2f} us")

# 6: in-graph-style D2H into pinned (the GPU-side cost)
y = torch.zeros(2, dtype=torch.int32, device="cuda")
t0 = time.perf_counter()
for _ in range(N):
    pin[:2].copy_(y, non_blocking=True)
torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"copy_ GPU->pinned : {(t1-t0)/N*1e6:6.2f} us (dispatch only, synced)")
