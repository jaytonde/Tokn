"""Measure decode graph replay time with CUDA events (no profiler overhead).

Reach a steady bs=5 decode state, then time run_model (graph replay) over many
steps with CUDA events, grouped by context length bucket.

Usage: PYTHONPATH=/home/Tokn .venv/bin/python debug/decode_time.py
"""

import sys
sys.path.insert(0, "/home/Tokn")

import torch
from utils.configs import ServerConfigs
from src.engine import Engine
from utils.sampling_params import SamplingParams


def load_questions(n=20):
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
    prompts = load_questions(20)

    async def run():
        for p in prompts:
            await engine.add_async_request(p, SamplingParams(temperature=0.0, max_tokens=256))

        # Run to a bs=5 steady decode state.
        guard = 0
        while not engine.scheduler.is_finished():
            seqs, is_prefill = engine.scheduler.schedule()
            if not seqs:
                break
            with torch.inference_mode():
                if is_prefill:
                    input_ids, positions = engine.RUN.prepare_prefill(seqs)
                    logits = engine.RUN.run_model(input_ids, positions, is_prefill=True)
                else:
                    input_ids, positions = engine.RUN.prepare_decode(seqs)
                    logits = engine.RUN.run_model(input_ids, positions, is_prefill=False)
                engine.scheduler.postprocess(seqs, torch.argmax(logits, dim=-1).tolist(), is_prefill)
            if not is_prefill:
                break
            guard += 1
            if guard > 200:
                raise RuntimeError("too many prefill steps")

        # Time decode replays, bucketed by context length.
        buckets = {}
        guard = 0
        n = 0
        while not engine.scheduler.is_finished() and n < 3000:
            seqs, is_prefill = engine.scheduler.schedule()
            if not seqs:
                break
            with torch.inference_mode():
                if is_prefill:
                    input_ids, positions = engine.RUN.prepare_prefill(seqs)
                    logits = engine.RUN.run_model(input_ids, positions, is_prefill=True)
                    engine.scheduler.postprocess(seqs, torch.argmax(logits, dim=-1).tolist(), True)
                    continue
                ctx_len = max(s.num_cached_tokens + s.num_scheduled_tokens for s in seqs)
                input_ids, positions = engine.RUN.prepare_decode(seqs)
                # warmup
                engine.RUN.run_model(input_ids, positions, is_prefill=False)
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(10):
                    engine.RUN.run_model(input_ids, positions, is_prefill=False)
                end.record()
                torch.cuda.synchronize()
                dt = start.elapsed_time(end) / 10
                engine.scheduler.postprocess(seqs, torch.argmax(logits, dim=-1).tolist(), False)
                b = ctx_len // 64 * 64
                buckets.setdefault(b, []).append(dt)
                n += 10
            guard += 1
            if guard > 5000:
                break

        tot = 0
        cnt = 0
        for b in sorted(buckets):
            v = buckets[b]
            mean = sum(v) / len(v)
            print(f"ctx~{b:4d}: {len(v):4d} samples  mean {mean*1000:7.2f} us")
            tot += sum(v); cnt += len(v)
        print(f"TOTAL: {cnt} samples  mean {tot/cnt*1000:7.2f} us")

    import asyncio
    asyncio.run(run())


if __name__ == "__main__":
    main()
