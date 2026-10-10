import sys, time
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
device = T.device("cuda")

def time_pool(pool, b):
    graph = pool[b]
    ev = T.cuda.Event(enable_timing=True); ev2 = T.cuda.Event(enable_timing=True)
    ev.record()
    for _ in range(100):
        graph.replay()
    ev2.record(); T.cuda.synchronize()
    return ev.elapsed_time(ev2)/100*1000

with T.inference_mode():
    for b in (1,2,3,4,5):
        state = RUN.graph_vars["tbuf_state"]
        for i in range(b):
            state[i,0] = 280+i*10; state[i,1] = 281+i*10
            RUN.graph_vars["tbuf_tok"][i] = i+1
        T.cuda.synchronize()
        tf = time_pool(RUN.graphs_fast, b)
        ts = time_pool(RUN.graphs, b)
        print(f"bs={b} ctx~281+: fast(ns=1)={tf:.2f}us splitkv(ns=0)={ts:.2f}us diff={ts-tf:.2f}us")

    # Same at lower ctx (fast path region) for reference
    for b in (1,2,3,4,5):
        state = RUN.graph_vars["tbuf_state"]
        for i in range(b):
            state[i,0] = 200+i*10; state[i,1] = 201+i*10
        T.cuda.synchronize()
        ts = time_pool(RUN.graphs, b)  # splitkv graph at low ctx still uses its kernel
        tf = time_pool(RUN.graphs_fast, b)
        print(f"bs={b} ctx~201:  fast={tf:.2f}us splitkv={ts:.2f}us diff={ts-tf:.2f}us")
