"""Time graph decode phases (buffer fills / replay / lm_head) with CUDA events."""
import argparse, asyncio, sys, json
sys.path.insert(0, "/home/Tokn")
import torch
from utils.configs import ServerConfigs
from src.engine import Engine
from utils.sampling_params import SamplingParams
from utils.context import get_context

def load_questions(path, n):
    out=[]
    with open(path) as f:
        for line in f:
            out.append(json.loads(line)["question"])
            if len(out)>=n: break
    return out

def main():
    cfg = ServerConfigs(model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda",
        max_length=2048, max_model_len=2048, tensor_parallel_size=1,
        enforce_eager=False, max_num_seqs=5, max_num_batched_tokens=1024)
    engine = Engine(model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda",
        max_model_len=2048, config=cfg)
    run = engine.RUN

    async def run_all():
        prompts = load_questions("data/questions.jsonl", 8)
        for p in prompts:
            await engine.add_async_request(p, SamplingParams(temperature=0.0, max_tokens=400))

        # Run until we are in steady decode.
        for _ in range(40):
            engine.step()

        t_fill = t_replay = t_lm = 0.0
        n = 0
        lens = []
        for _ in range(60):
            seqs, is_prefill = engine.scheduler.schedule()
            if not seqs or is_prefill:
                engine.scheduler.postprocess(seqs, [], is_prefill)
                continue
            lens.append(len(seqs[0].token_ids))
            with torch.inference_mode():
                input_ids, positions = run.prepare_decode(seqs)
                ctx = get_context()
                bs = input_ids.size(0)
                gv = run.graph_vars
                e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
                e2 = torch.cuda.Event(enable_timing=True); e3 = torch.cuda.Event(enable_timing=True)
                e0.record()
                gv["input_ids"][:bs] = input_ids
                gv["positions"][:bs] = positions
                gv["slot_mapping"].fill_(-1)
                gv["slot_mapping"][:bs] = ctx.slot_mapping
                gv["context_lens"].zero_()
                gv["context_lens"][:bs] = ctx.context_lens
                gv["block_tables"].fill_(-1)
                gv["block_tables"][:bs, :ctx.block_tables.size(1)] = ctx.block_tables
                e1.record()
                pool = run.graphs_fast if ctx.use_fast_flash else run.graphs
                pool[next(x for x in run.graph_bs if x >= bs)].replay()
                e2.record()
                logits = gv["outputs"][:bs]
                e3.record()
            torch.cuda.synchronize()
            t_fill += e0.elapsed_time(e1)
            t_replay += e1.elapsed_time(e2)
            t_lm += e2.elapsed_time(e3)
            toks = torch.argmax(logits, dim=-1).tolist()
            engine.scheduler.postprocess(seqs, toks, False)
            n += 1
        print(f"profiled decode steps: {n}, bs~{bs}, seq lens ~ {sorted(set(lens))}", flush=True)
        if n:
            print(f"per decode step: fills {t_fill/n:.3f} ms, graph replay {t_replay/n:.3f} ms, "
                  f"lm_head {t_lm/n:.3f} ms, TOTAL { (t_fill+t_replay+t_lm)/n:.3f} ms", flush=True)

    asyncio.run(run_all())

if __name__ == "__main__":
    main()
