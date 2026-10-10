"""Test: can a CUDA graph capture a D2H copy_ to pinned host memory? And its cost."""
import sys, time
sys.path.insert(0, "/home/Tokn")
import torch as T

T.cuda.init()
dev = T.device("cuda")
gpu = T.zeros(8, dtype=T.int32, device=dev)
pin = T.zeros(8, dtype=T.int32, pin_memory=True)
pin_np = pin.numpy()

# Try capturing a D2H inside a graph
try:
    graph = T.cuda.CUDAGraph()
    with T.cuda.graph(graph):
        gpu[:] = T.tensor([1,2,3,4,5,6,7,8], dtype=T.int32, device=dev)
        pin[:8].copy_(gpu, non_blocking=True)
    T.cuda.synchronize()
    gpu[:] = T.zeros(8, dtype=T.int32, device=dev)
    T.cuda.synchronize()
    graph.replay()
    T.cuda.synchronize()
    print("graph D2H capture+replay OK; pinned values:", pin_np.tolist())
except Exception as e:
    print("graph D2H FAILED:", type(e).__name__, str(e)[:200])

# Cost: D2H in graph vs not
if True:
    try:
        g2 = T.cuda.CUDAGraph()
        with T.cuda.graph(g2):
            gpu[:] = T.arange(8, dtype=T.int32, device=dev)
            pin[:8].copy_(gpu, non_blocking=True)
        T.cuda.synchronize()
        ev = T.cuda.Event(enable_timing=True); ev2 = T.cuda.Event(enable_timing=True)
        ev.record()
        for _ in range(500): g2.replay()
        ev2.record(); T.cuda.synchronize()
        print(f"D2H-in-graph replay cost: {ev.elapsed_time(ev2)/500*1000:.2f}us")
    except Exception as e:
        print("cost test failed:", e)

# Cost of numpy read after replay
gpu[:] = T.arange(8, dtype=T.int32, device=dev)
T.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20000):
    r = pin_np[:8].tolist()
t1 = time.perf_counter()
print(f"pinned numpy tolist: {(t1-t0)/20000*1e6:.2f}us")
