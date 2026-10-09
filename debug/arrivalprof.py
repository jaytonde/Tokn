"""Simulate the benchmark's arrival pattern and profile step/idle dynamics.

Submits N prompts at request-rate R, drives the engine synchronously, and logs
step counts, batch sizes, idle waits, and GPU utilization over time.

Usage: PYTHONPATH=/home/Tokn .venv/bin/python debug/arrivalprof.py --num 60 --rate 2
"""

import argparse
import asyncio
import json
import sys
import time

sys.path.insert(0, "/home/Tokn")

import torch

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
    ap.add_argument("--num", type=int, default=60)
    ap.add_argument("--rate", type=float, default=2.0)
    args = ap.parse_args()

    import random
    rng = random.Random(0)
    gaps = [rng.expovariate(args.rate) for _ in range(args.num)]
    arrival_span = sum(gaps)

    cfg = ServerConfigs(
        model="Qwen/Qwen3-0.6B",
        dtype="bf16",
        device="cuda",
        max_length=2048,
        max_model_len=2048,
        tensor_parallel_size=1,
        enforce_eager=False,
        max_num_seqs=5,
        max_num_batched_tokens=1024,
    )
    engine = Engine(
        model="Qwen/Qwen3-0.6B",
        dtype="bf16",
        device="cuda",
        max_model_len=2048,
        config=cfg,
    )
    # record schedule decisions
    _orig_schedule = engine.scheduler.schedule
    engine.scheduler._last = (None, None)
    def _rec_schedule():
        s = _orig_schedule()
        engine.scheduler._last = s
        return s
    engine.scheduler.schedule = _rec_schedule
    prompts = load_questions("data/questions.jsonl", args.num)

    async def run():
        t_start = time.perf_counter()
        n_decode = n_prefill = 0
        decode_bs = []
        idle_wait_s = 0.0
        next_arrival = 0.0
        idx = 0
        finished = 0
        t_last = time.perf_counter()
        step_times = []
        max_seqlen = 0

        while finished < args.num:
            now = time.perf_counter() - t_start
            # submit requests on schedule (exponential, matches benchmark seed 0)
            while idx < args.num and now >= next_arrival:
                await engine.add_async_request(
                    prompts[idx], SamplingParams(temperature=0.0, max_tokens=256)
                )
                idx += 1
                next_arrival += gaps[idx - 1] if idx - 1 < len(gaps) else 1.0 / args.rate

            if engine.scheduler.is_finished() and idx >= args.num:
                break
            if engine.scheduler.is_finished():
                # idle: no work yet
                t_wait0 = time.perf_counter()
                await asyncio.sleep(0.01)
                idle_wait_s += time.perf_counter() - t_wait0
                continue

            t0 = time.perf_counter()
            fin = engine.step()
            step_ms = (time.perf_counter() - t0) * 1000
            step_times.append(step_ms)
            finished += len(fin)
            seqs, is_prefill = engine.scheduler._last
            if is_prefill:
                n_prefill += 1
            else:
                n_decode += 1
                decode_bs.append(len(seqs) if seqs else 0)
                if seqs:
                    max_seqlen = max(max_seqlen, len(seqs[0].token_ids))

        elapsed = time.perf_counter() - t_start
        import statistics
        print(f"elapsed: {elapsed:.2f}s  (arrival span {arrival_span:.1f}s  lag {elapsed-arrival_span:.2f}s)")
        print(f"steps: decode={n_decode} prefill={n_prefill}")
        if decode_bs:
            print(f"decode batch size: mean={statistics.mean(decode_bs):.2f} "
                  f"p50={sorted(decode_bs)[len(decode_bs)//2]} max={max(decode_bs)}")
        print(f"step time (all): mean={statistics.mean(step_times):.3f}ms "
              f"p50={sorted(step_times)[len(step_times)//2]:.3f} p99={sorted(step_times)[int(len(step_times)*0.99)]:.3f}ms")
        print(f"max seqlen seen: {max_seqlen}")
        print(f"idle wait: {idle_wait_s:.2f}s")

    asyncio.run(run())


if __name__ == "__main__":
    main()
