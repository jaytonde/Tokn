"""Replicate benchmark.py's exact driver (run_loop asyncio task + exponential
submissions) with instrumentation to find the elapsed-vs-arrival-span gap.

Usage: PYTHONPATH=/home/Tokn .venv/bin/python debug/benchsim.py --num 400
"""
import argparse
import asyncio
import json
import random
import sys
import time

sys.path.insert(0, "/home/Tokn")

import torch

from utils.configs import ServerConfigs
from src.engine import Engine
from utils.sampling_params import SamplingParams


def load_questions(path, n):
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line)["question"])
            if len(out) >= n:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=400)
    ap.add_argument("--rate", type=float, default=2.0)
    args = ap.parse_args()

    cfg = ServerConfigs(
        model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda",
        max_length=2048, max_model_len=2048, tensor_parallel_size=1,
        enforce_eager=False, max_num_seqs=5, max_num_batched_tokens=1024,
    )
    engine = Engine(
        model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda",
        max_model_len=2048, config=cfg,
    )
    prompts = load_questions("data/questions.jsonl", args.num)
    rng = random.Random(0)
    delays = [rng.expovariate(args.rate) for _ in range(args.num)]
    arrival_span = sum(delays)
    print(f"arrival span: {arrival_span:.2f}s", flush=True)

    # Instrument step timing.
    t_step = 0.0
    n_step = 0
    t_decode_text = 0.0
    n_finished = 0
    _orig_step = engine.step
    def step():
        nonlocal t_step, n_step
        t0 = time.perf_counter()
        out = _orig_step()
        t_step += time.perf_counter() - t0
        n_step += 1
        return out
    engine.step = step
    _orig_decode = engine.tokenizer.decode
    def decode(ids, **kw):
        nonlocal t_decode_text, n_finished
        t0 = time.perf_counter()
        out = _orig_decode(ids, **kw)
        t_decode_text += time.perf_counter() - t0
        n_finished += 1
        return out
    engine.tokenizer.decode = decode

    async def run():
        t0 = time.perf_counter()
        if engine.background_task is None:
            engine.background_task = asyncio.create_task(engine.run_loop())
        seq_ids = []
        t_submit = 0.0
        arrival_times = []
        sched_times = []
        _t_sched = 0.0
        for i, (p, delay) in enumerate(zip(prompts, delays)):
            _t_sched += delay
            if delay > 0:
                await asyncio.sleep(delay)
            s0 = time.perf_counter()
            sid = await engine.add_async_request(p, SamplingParams(temperature=0.0, max_tokens=256))
            t_submit += time.perf_counter() - s0
            seq_ids.append(sid)
            arrival_times.append(time.perf_counter())
            sched_times.append(_t_sched)
        # arrival delay stats
        lat = [arrival_times[i]-t0 - sched_times[i] for i in range(len(seq_ids))]
        print(f"arrival delay: last={lat[-1]:.3f}s  mean={sum(lat)/len(lat):.3f}s", flush=True)
        states = [engine.request_states[s] for s in seq_ids]
        # last request's timeline
        last = states[-1].seq
        import threading
        def watch():
            last.done.wait()
            print(f"LAST REQ: arrived at {arrival_times[-1]-t0:.2f}s, done at {time.perf_counter()-t0:.2f}s, "
                  f"queue-wait {last.ttft:.2f}s, n_tokens {last.num_generated_tokens}", flush=True)
        threading.Thread(target=watch, daemon=True).start()
        all_done = asyncio.ensure_future(asyncio.gather(*(s.done.wait() for s in states)))
        await asyncio.wait({all_done, engine.background_task}, return_when=asyncio.FIRST_COMPLETED)
        if not all_done.done():
            engine.background_task.result()
        elapsed = time.perf_counter() - t0
        print(f"elapsed: {elapsed:.2f}s  lag vs arrival {elapsed-arrival_span:.2f}s", flush=True)
        ls = states[-1].seq
        print(f"LAST REQ: ttft={ls.ttft:.2f}s n_tok={ls.num_generated_tokens} "
              f"last_ts={ls.token_timestamps[-1]-arrival_times[-1]:.2f}s after arrival", flush=True)
        # max TTFT
        tts = [(s.seq.ttft or 0, i) for i, s in enumerate(states)]
        mtt, mi = max(tts)
        print(f"MAX TTFT: {mtt:.2f}s at request idx {mi} (arrived {arrival_times[mi]-t0:.2f}s)", flush=True)
        # completion times (relative to t0) of the last few requests, with arrival
        cts = sorted((s.seq.token_timestamps[-1] if s.seq.token_timestamps else 0, i) for i, s in enumerate(states))
        print("last 5 done (rel time, req idx, arrived rel, ttft):")
        for ct, i in cts[-5:]:
            print(f"  done@ {ct-t0:7.2f}s  idx={i}  arr@ {arrival_times[i]-t0:7.2f}s  ttft={(states[i].seq.ttft or 0):.2f}s", flush=True)
        print(f"steps: {n_step}, step time total {t_step:.2f}s ({t_step/n_step*1000:.3f} ms/step)", flush=True)
        print(f"decode text total {t_decode_text:.2f}s ({n_finished} seqs, {t_decode_text/n_finished*1000:.3f} ms/seq)", flush=True)

    asyncio.run(run())


if __name__ == "__main__":
    main()
