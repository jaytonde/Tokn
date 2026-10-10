"""Find which steps make up the ITL p99: record per-step wall, bs, max_ctx, mode."""
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

stepinfo = []  # (wall_us, bs, max_ctx, fast, is_prefill)

async def timed_split():
    t_start = time.perf_counter()
    loop = asyncio.get_running_loop()
    seqs, is_prefill = llm.scheduler.schedule()
    if not seqs:
        return []
    if is_prefill:
        fin = await loop.run_in_executor(None, llm._step_with_seqs, seqs, True)
        stepinfo.append(((time.perf_counter()-t_start)*1e6, len(seqs), -1, -1, True))
        return fin
    with torch.inference_mode():
        spec = tuple(s.seq_id for s in seqs)
        if llm._spec_ids == spec:
            input_ids, positions = llm.RUN.prepare_decode(seqs, token_only=True)
        else:
            llm._spec_ids = None
            input_ids, positions = llm.RUN.prepare_decode(seqs)
        try:
            logits = llm.RUN.run_model(input_ids, positions, False)
        finally:
            reset_context()
        ev = torch.cuda.Event(); ev.record()
    ctx_max = max(len(s.token_ids) for s in seqs)
    if not ev.query():
        try:
            with torch.inference_mode():
                llm._spec_ids = spec
                llm.RUN.prepare_decode(seqs, spec_next=True)
        except Exception:
            llm._spec_ids = None
    else:
        llm._spec_ids = None
    while not ev.query():
        await asyncio.sleep(0)
    with torch.inference_mode():
        next_token_ids = llm.RUN.graph_vars["tbuf_tok"][:len(seqs)].tolist()
        finished = llm.scheduler.postprocess(seqs, next_token_ids, False)
    for seq in seqs:
        start_block = seq.num_hashed_blocks
        end_block = seq.num_cached_tokens // llm.block_size
        if end_block > start_block:
            llm.block_manager.hash_completed_blocks(
                token_ids=seq.token_ids, block_table=seq.block_table_ids,
                start_block=start_block, end_block=end_block)
            seq.num_hashed_blocks = end_block
    for seq in finished:
        llm.block_manager.deallocate(seq.block_table_ids)
        llm.RUN._padded_bt.pop(seq.seq_id, None)
    stepinfo.append(((time.perf_counter()-t_start)*1e6, len(seqs), ctx_max,
                     ctx_max <= llm.block_size, False))
    return finished

llm._step_split = timed_split
prompts = load_questions("data/questions.jsonl", 150)
params = [SamplingParams(temperature=0.0, max_tokens=256) for _ in prompts]

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

walls = sorted((w for w,bs,c,f,pre in stepinfo if not pre))
n = len(walls)
def pct(q): return walls[min(n-1,int((n-1)*q))]
print(f"decode steps: {n}  p50={pct(0.5):.0f} p90={pct(0.9):.0f} p95={pct(0.95):.0f} p99={pct(0.99):.0f}us")
thr = pct(0.99)
slow = [(w,bs,c,f) for w,bs,c,f,pre in stepinfo if not pre and w >= thr*0.97]
print(f"\nsteps within 3% of p99 wall ({thr*0.97:.0f}us): {len(slow)}")
from collections import Counter
print("  (bs, fast?) count:")
for k,v in sorted(Counter((bs, f) for w,bs,c,f in slow).items()):
    print(f"    bs={k[0]} fast={k[1]}: {v}")
print("  ctx distribution of those:")
for k,v in sorted(Counter((c//64)*64 for w,bs,c,f in slow).items()):
    print(f"    ctx~{k}-{k+64}: {v}")
print("\nALL steps by (bs,fast):")
for k,v in sorted(Counter((bs, f) for w,bs,c,f,pre in stepinfo if not pre).items()):
    print(f"  bs={k[0]} fast={k[1]}: {v}")
