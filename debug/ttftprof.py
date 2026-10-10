"""Break down TTFT into queue wait vs prefill time.

Monkey-patches the engine to record when each sequence's prefill actually runs
and how long the prefill step takes, then reports per-request TTFT components.

Usage: PYTHONPATH=/home/Tokn .venv/bin/python debug/ttftprof.py --num-prompts 60
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
    ap.add_argument("--num-prompts", type=int, default=60)
    ap.add_argument("--request-rate", type=float, default=2.0)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

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

    # instrument: record prefill-start time and prefill wall per step
    events = []  # (seq_id, kind, t, extra)

    orig_sched = engine.scheduler.schedule

    def patched_schedule():
        seqs, is_prefill = orig_sched()
        if seqs and is_prefill:
            for s in seqs:
                if s.block_table is not None and s.num_cached_tokens == 0 and not getattr(s, "_seen_prefill", False):
                    s._seen_prefill = True
                    events.append((s.seq_id, "prefill_start", time.perf_counter(), len(seqs)))
        return seqs, is_prefill

    engine.scheduler.schedule = patched_schedule

    orig_step = engine.step

    def patched_step():
        t0 = time.perf_counter()
        r = orig_step()
        dur = time.perf_counter() - t0
        events.append(("step", "decode_or_prefill", time.perf_counter(), round(dur * 1000, 3)))
        return r

    engine.step = patched_step

    prompts = load_questions("data/questions.jsonl", args.num_prompts)
    import random
    rng = random.Random(0)
    delays = [rng.expovariate(args.request_rate) for _ in prompts]

    async def run():
        if engine.background_task is None:
            engine.background_task = asyncio.create_task(engine.run_loop())
        seq_ids = []
        for (p, delay) in zip(prompts, delays):
            if delay > 0:
                await asyncio.sleep(delay)
            seq_ids.append(await engine.add_async_request(p, SamplingParams(temperature=0.0, max_tokens=args.max_tokens)))
        states = [engine.request_states[sid] for sid in seq_ids]
        await asyncio.gather(*(s.done.wait() for s in states))
        return [engine.request_states.get(sid) for sid in seq_ids]

    asyncio.run(run())

    # analyze
    # arrival times
    arr = {}
    for e in events:
        pass
    starts = {}
    for e in events:
        if e[1] == "prefill_start":
            starts[e[0]] = e[2]

    # collect from sequences
    waits = []
    for sid, state in list(engine.request_states.items()):
        s = state.seq
        if s.arrival_time is not None and s.token_timestamps:
            st = starts.get(sid)
            qw = (st - s.arrival_time) * 1000 if st else None
            waits.append(qw if qw is not None else -1)
    print(f"requests tracked: {len(waits)}")
    if waits:
        ws = sorted(w for w in waits if w >= 0)
        n = len(ws)
        print(f"queue-wait ms: mean={sum(ws)/n:.1f} p50={ws[n//2]:.1f} p99={ws[int((n-1)*0.99)]:.1f} max={ws[-1]:.1f}")
    # prefill step durations: for each prefill_start, the next step event's duration
    pref_durs = []
    for i, e in enumerate(events):
        if e[1] == "prefill_start":
            for e2 in events[i + 1:]:
                if e2[1] == "decode_or_prefill":
                    pref_durs.append(e2[3])
                    break
    if pref_durs:
        ps = sorted(pref_durs)
        n = len(ps)
        print(f"prefill step ms: count={n} mean={sum(ps)/n:.2f} p50={ps[n//2]:.2f} p99={ps[int((n-1)*0.99)]:.2f} max={ps[-1]:.2f}")
    print(f"total prefill_start events: {sum(1 for e in events if e[1]=='prefill_start')}")
    print(f"total steps: {sum(1 for e in events if e[1]=='decode_or_prefill')}")


if __name__ == "__main__":
    main()
