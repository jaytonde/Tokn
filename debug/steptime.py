"""Measure per-step wall-time components in the real engine (sync driver).

Usage: PYTHONPATH=/home/Tokn .venv/bin/python debug/steptime.py
"""

import sys
sys.path.insert(0, "/home/Tokn")
import time
import torch
from utils.configs import ServerConfigs
from src.engine import Engine
from utils.sampling_params import SamplingParams


def load_questions(n=30):
    import json
    out = []
    with open("data/questions.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line)["question"])
            if len(out) >= n:
                break
    return out


def main():
    cfg = ServerConfigs(
        model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda",
        max_length=2048, max_model_len=2048, tensor_parallel_size=1,
        enforce_eager=False, max_num_seqs=5, max_num_batched_tokens=1024,
    )
    engine = Engine(
        model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda",
        max_model_len=2048, config=cfg,
    )
    prompts = load_questions(30)

    async def run():
        for p in prompts:
            await engine.add_async_request(p, SamplingParams(temperature=0.0, max_tokens=256))

        comps = {"schedule": [], "prepare": [], "replay": [], "argmax": [], "post": [], "step": []}
        guard = 0
        while not engine.scheduler.is_finished():
            t0 = time.perf_counter()
            seqs, is_prefill = engine.scheduler.schedule()
            t1 = time.perf_counter()
            if not seqs:
                break
            with torch.inference_mode():
                if is_prefill:
                    inp, pos = engine.RUN.prepare_prefill(seqs)
                    t2 = time.perf_counter()
                    lg = engine.RUN.run_model(inp, pos, True)
                    torch.cuda.synchronize()
                    t3 = time.perf_counter()
                    toks = torch.argmax(lg, dim=-1).tolist()
                else:
                    inp, pos = engine.RUN.prepare_decode(seqs)
                    t2 = time.perf_counter()
                    logits = engine.RUN.run_model(inp, pos, False)
                    t3 = time.perf_counter()
                    toks = torch.argmax(logits, dim=-1).tolist()
                torch.cuda.synchronize()
                t4 = time.perf_counter()
                engine.scheduler.postprocess(seqs, toks, is_prefill)
                t5 = time.perf_counter()
            if not is_prefill:
                comps["schedule"].append(t1 - t0)
                comps["prepare"].append(t2 - t1)
                comps["replay"].append(t3 - t2)
                comps["argmax"].append(t4 - t3)
                comps["post"].append(t5 - t4)
                comps["step"].append(t5 - t0)
            guard += 1
            if guard > 2000:
                break

        for k, v in comps.items():
            if v:
                print(f"{k:10s}: mean {sum(v)/len(v)*1000:7.2f} ms   n={len(v)}")
        total_step = sum(comps["step"]) / len(comps["step"])
        print(f"replay share: {sum(comps['replay'])/len(comps['replay'])/total_step*100:.1f}%")

    import asyncio
    asyncio.run(run())


if __name__ == "__main__":
    main()
