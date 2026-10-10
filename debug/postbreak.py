"""Break down the post-GPU phase in the real split-mode run_loop (post exp 47)."""
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

acc = {"tolist":0.0,"schedpost":0.0,"hash":0.0,"dealloc":0.0,"loop":0.0,"n":0}
printed = 0

async def timed_split():
    global printed
    t0 = time.perf_counter()
    loop = asyncio.get_running_loop()
    seqs, is_prefill = llm.scheduler.schedule()
    if not seqs: return []
    if is_prefill:
        return await loop.run_in_executor(None, llm._step_with_seqs, seqs, True)
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
        t_a = time.perf_counter()
        next_token_ids = llm.RUN.graph_vars["tbuf_tok"][:len(seqs)].tolist()
        t_b = time.perf_counter()
        finished = llm.scheduler.postprocess(seqs, next_token_ids, False)
        t_c = time.perf_counter()
    for seq in seqs:
        start_block = seq.num_hashed_blocks
        end_block = seq.num_cached_tokens // llm.block_size
        if end_block > start_block:
            llm.block_manager.hash_completed_blocks(
                token_ids=seq.token_ids, block_table=seq.block_table_ids,
                start_block=start_block, end_block=end_block)
            seq.num_hashed_blocks = end_block
    t_d = time.perf_counter()
    for seq in finished:
        llm.block_manager.deallocate(seq.block_table_ids)
        llm.RUN._padded_bt.pop(seq.seq_id, None)
    t_e = time.perf_counter()
    acc["tolist"] += (t_b-t_a)*1e6; acc["schedpost"] += (t_c-t_b)*1e6
    acc["hash"] += (t_d-t_c)*1e6; acc["dealloc"] += (t_e-t_d)*1e6
    acc["n"] += 1
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
n = acc["n"]
print(f"decode steps: {n}")
for k in ("tolist","schedpost","hash","dealloc"):
    print(f"  {k:10s} mean {acc[k]/n:6.2f} us/step  total {acc[k]:8.1f}")
