"""Break down run_model CPU time in the real engine."""
import asyncio, random, sys, time
sys.path.insert(0, "/home/Tokn")
import torch
from src.llm import LLM
from utils.sampling_params import SamplingParams
from utils.configs import ServerConfigs
from utils.context import reset_context

def load_questions(path, n):
    import json
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line)["question"])
            if len(out) >= n: break
    return out

cfg = ServerConfigs(
    model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda",
    max_length=2048, max_model_len=2048, tensor_parallel_size=1,
    enforce_eager=False, max_num_seqs=5, max_num_batched_tokens=1024,
)
llm = LLM(model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda", max_model_len=2048, config=cfg)
RUN = llm.RUN
orig_run_model = RUN.run_model
orig_prepare = RUN.prepare_decode

stats = {"gen":0.0,"dict":0.0,"replay":0.0,"return":0.0,"n":0}

def timed_run_model(input_ids, positions, is_prefill):
    if is_prefill:
        return orig_run_model(input_ids, positions, is_prefill)
    t0 = time.perf_counter()
    bs = input_ids.size(0)
    context = llm.RUN._last_ctx  # set by timed_prepare
    t1 = time.perf_counter()
    pool = RUN.graphs_fast if context.use_fast_flash else RUN.graphs
    use_bs = next(x for x in RUN.graph_bs if x >= bs)
    graph = pool[use_bs]
    gv = RUN.graph_vars
    t2 = time.perf_counter()
    graph.replay()
    t3 = time.perf_counter()
    out = gv["outputs"][:bs]
    t4 = time.perf_counter()
    stats["gen"] += (t1-t0)*1e6; stats["dict"] += (t2-t1)*1e6
    stats["replay"] += (t3-t2)*1e6; stats["return"] += (t4-t3)*1e6
    stats["n"] += 1
    return out

def timed_prepare(seqs, token_only=False, spec_next=False):
    if token_only or spec_next:
        return orig_prepare(seqs, token_only=token_only, spec_next=spec_next)
    from utils.context import get_context
    r = orig_prepare(seqs, token_only=token_only, spec_next=spec_next)
    RUN._last_ctx = get_context()
    return r

RUN.run_model = timed_run_model
RUN.prepare_decode = timed_prepare

prompts = load_questions("data/questions.jsonl", 120)
params = [SamplingParams(temperature=0.0, max_tokens=64) for _ in prompts]
async def run():
    if llm.background_task is None:
        llm.background_task = asyncio.create_task(llm.run_loop())
    rng = random.Random(0)
    delays = [rng.expovariate(2.0) for _ in prompts]
    for (p, prm), d in zip(zip(prompts, params), delays):
        if d > 0: await asyncio.sleep(d)
        await llm.add_async_request(p, prm)
    states = list(llm.request_states.values())
    await asyncio.gather(*(st.done.wait() for st in states))
asyncio.run(run())
n = stats["n"]
print(f"run_model decode calls: {n}")
for k in ("gen","dict","replay","return"):
    print(f"  {k:8s} mean {stats[k]/n:6.2f} us")
