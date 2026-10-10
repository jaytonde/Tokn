"""Count distinct prefill (num_seqs, total_tokens) shapes in a benchmark-like run."""
import argparse
import asyncio
import json
import random
import sys
import time

sys.path.insert(0, "/home/Tokn")

from utils.configs import ServerConfigs
from src.engine import Engine
from utils.sampling_params import SamplingParams


def load_questions(path: str, n: int):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line)["question"])
            if len(out) >= n:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-prompts", type=int, default=400)
    ap.add_argument("--request-rate", type=float, default=2.0)
    args = ap.parse_args()

    cfg = ServerConfigs(
        model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda",
        max_length=2048, max_model_len=2048, tensor_parallel_size=1,
        enforce_eager=False, max_num_seqs=5, max_num_batched_tokens=1024,
    )
    engine = Engine(model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda",
                    max_model_len=2048, config=cfg)

    shapes = {}  # (num_seqs, total_tokens) -> count
    seq_counts = {}

    orig_sched = engine.scheduler.schedule

    def patched_schedule():
        seqs, is_prefill = orig_sched()
        if seqs and is_prefill:
            n = len(seqs)
            # total scheduled tokens
            total = sum(s.num_scheduled_tokens for s in seqs)
            key = (n, total)
            shapes[key] = shapes.get(key, 0) + 1
            seq_counts[n] = seq_counts.get(n, 0) + 1
        return seqs, is_prefill

    engine.scheduler.schedule = patched_schedule

    prompts = load_questions("data/questions.jsonl", args.num_prompts)
    rng = random.Random(0)
    delays = [rng.expovariate(args.request_rate) for _ in prompts]

    async def run():
        if engine.background_task is None:
            engine.background_task = asyncio.create_task(engine.run_loop())
        seq_ids = []
        for (p, delay) in zip(prompts, delays):
            if delay > 0:
                await asyncio.sleep(delay)
            seq_ids.append(await engine.add_async_request(p, SamplingParams(temperature=0.0, max_tokens=256)))
        states = [engine.request_states[sid] for sid in seq_ids]
        await asyncio.gather(*(s.done.wait() for s in states))

    asyncio.run(run())

    total = sum(shapes.values())
    print(f"total prefill steps: {total}")
    print(f"distinct (num_seqs, total_tokens) shapes: {len(shapes)}")
    print(f"distinct by num_seqs: { {k: sum(1 for (n,t) in shapes if n==k) for k in sorted(set(n for n,_ in shapes))} }")
    print(f"seq_counts: {dict(sorted(seq_counts.items()))}")
    # distribution of tokens
    toks = sorted(t for (n, t) in shapes)
    n = len(toks)
    print(f"total_tokens dist: min={toks[0]} p50={toks[n//2]} p90={toks[int(n*0.9)]} max={toks[-1]}")


if __name__ == "__main__":
    main()
