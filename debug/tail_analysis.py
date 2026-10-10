"""Analyze the benchmark tail: per-request arrival / first-token / finish times.

Replicates debug/benchmark.py's tokn path (same params) but records the last
few requests' timing to see why elapsed exceeds the arrival span by ~11s.

Usage: PYTHONPATH=/home/Tokn .venv/bin/python debug/tail_analysis.py --num-prompts 400
"""

import argparse
import asyncio
import json
import random
import time
import sys

sys.path.insert(0, "/home/Tokn")

from src.llm import LLM
from utils.sampling_params import SamplingParams
from utils.configs import ServerConfigs


def load_questions(path, n):
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line)["question"])
    if n:
        prompts = prompts[:n]
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-prompts", type=int, default=400)
    ap.add_argument("--request-rate", type=float, default=2)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    prompts = load_questions("data/questions.jsonl", args.num_prompts)
    cfg = ServerConfigs(
        model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda",
        max_length=2048, max_model_len=2048, tensor_parallel_size=1,
        enforce_eager=False, max_num_seqs=5, max_num_batched_tokens=1024,
    )
    llm = LLM(model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda",
              max_model_len=2048, config=cfg)
    sampling = [SamplingParams(temperature=0.0, max_tokens=args.max_tokens) for _ in prompts]

    rng = random.Random(0)
    delays = [rng.expovariate(args.request_rate) for _ in prompts]

    async def run():
        if llm.background_task is None:
            llm.background_task = asyncio.create_task(llm.run_loop())

        seq_ids = []
        t_start = time.perf_counter()
        for (p, sp), d in zip(zip(prompts, sampling), delays):
            if d > 0:
                await asyncio.sleep(d)
            sid = await llm.add_async_request(p, sp)
            seq_ids.append(sid)

        last_arrival = time.perf_counter() - t_start

        states = [llm.request_states[sid] for sid in seq_ids]
        await asyncio.gather(*(s.done.wait() for s in states))
        elapsed = time.perf_counter() - t_start

        rows = []
        for sid in seq_ids:
            st = llm.request_states.get(sid)
            if st is None:
                continue
            seq = st.seq
            arrival = seq.arrival_time - t_start
            first = seq.token_timestamps[0] - t_start if seq.token_timestamps else None
            last_ts = seq.token_timestamps[-1] - t_start if seq.token_timestamps else None
            finish = seq.token_timestamps[-1] if seq.token_timestamps else None
            rows.append((arrival, first, last_ts, seq.num_generated_tokens))

        rows.sort(key=lambda r: r[0])
        print(f"t_start offset used, last_arrival={last_arrival:.3f}s elapsed={elapsed:.3f}s "
              f"tail={elapsed-last_arrival:.3f}s")
        print("last 12 by arrival:")
        print(f"  {'arrival':>9} {'first':>9} {'last':>9} {'#tok':>5}")
        for arrival, first, last_ts, nt in rows[-12:]:
            print(f"  {arrival:9.3f} {str(first):>9} {str(last_ts):>9} {nt:5d}")
        # last-finish vs last-arrival
        lf = max(r[2] for r in rows if r[2] is not None)
        print(f"last finish={lf:.3f}s, last arrival={rows[-1][0]:.3f}s, gap={lf-rows[-1][0]:.3f}s")

        async with llm.request_lock:
            for sid in seq_ids:
                llm.request_states.pop(sid, None)

    asyncio.run(run())


if __name__ == "__main__":
    main()
