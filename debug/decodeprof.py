"""Profile CUDA-graph decode replay: kernel breakdown per batch size.

Drives the engine synchronously, prefills N prompts, then profiles N decode
steps with torch.profiler, reporting the top CUDA kernels by time for each
batch size encountered.

Usage: PYTHONPATH=/home/Tokn .venv/bin/python debug/decodeprof.py --num-prompts 20
"""

import argparse
import asyncio
import sys
import time

sys.path.insert(0, "/home/Tokn")

import torch

from utils.configs import ServerConfigs
from src.engine import Engine
from utils.sampling_params import SamplingParams


def load_questions(path: str, n: int):
    import json
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line)["question"])
            if len(out) >= n:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-prompts", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--bs", type=int, default=None, help="only profile steps with this batch size")
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
    prompts = load_questions("data/questions.jsonl", args.num_prompts)

    async def run():
        for p in prompts:
            await engine.add_async_request(p, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))

        # Phase 1: no profiling, just get to a steady decode state.
        guard = 0
        while not engine.scheduler.is_finished():
            seqs, is_prefill = engine.scheduler.schedule()
            if not seqs:
                break
            if not is_prefill:
                break
            with torch.inference_mode():
                input_ids, positions = engine.RUN.prepare_prefill(seqs)
                logits = engine.RUN.run_model(input_ids, positions, is_prefill=True)
                engine.scheduler.postprocess(seqs, torch.argmax(logits, dim=-1).tolist(), True)
            guard += 1
            if guard > 100:
                raise RuntimeError("too many prefill steps")

        # Phase 2: profile decode steps, group by batch size.
        per_bs = {}
        guard = 0
        n_decode = 0
        while not engine.scheduler.is_finished():
            seqs, is_prefill = engine.scheduler.schedule()
            if not seqs:
                break
            if is_prefill:
                with torch.inference_mode():
                    input_ids, positions = engine.RUN.prepare_prefill(seqs)
                    logits = engine.RUN.run_model(input_ids, positions, is_prefill=True)
                    engine.scheduler.postprocess(seqs, torch.argmax(logits, dim=-1).tolist(), True)
                continue

            bs = len(seqs)
            if args.bs is not None and bs != args.bs:
                # still advance the system without profiling
                with torch.inference_mode():
                    input_ids, positions = engine.RUN.prepare_decode(seqs)
                    logits = engine.RUN.run_model(input_ids, positions, is_prefill=False)
                    engine.scheduler.postprocess(seqs, torch.argmax(logits, dim=-1).tolist(), False)
                guard += 1
                if guard > 10000:
                    raise RuntimeError("too many steps")
                continue

            with torch.inference_mode():
                input_ids, positions = engine.RUN.prepare_decode(seqs)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    logits = engine.RUN.run_model(input_ids, positions, is_prefill=False)
                engine.scheduler.postprocess(seqs, torch.argmax(logits, dim=-1).tolist(), False)

            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            torch.cuda.synchronize()

            if bs not in per_bs:
                per_bs[bs] = {"events": [], "kernels": {}}
            per_bs[bs]["events"].append(ev.elapsed_time(ev))
            for e in prof.key_averages():
                if e.device_type == torch.autograd.DeviceType.CUDA:
                    per_bs[bs]["kernels"][e.key] = per_bs[bs]["kernels"].get(e.key, 0) + e.self_device_time_total

            n_decode += 1
            guard += 1
            if guard > 10000:
                raise RuntimeError("too many steps")
            if args.bs is not None and n_decode > 15:
                break
            if args.bs is None and n_decode > 30:
                break

        for bs in sorted(per_bs):
            d = per_bs[bs]
            evs = d["events"]
            print(f"\n=== batch size {bs}: {len(evs)} profiled steps, mean cuda ev {sum(evs)/len(evs):.3f} ms ===")
            tot = sum(d["kernels"].values())
            for k, v in sorted(d["kernels"].items(), key=lambda kv: -kv[1])[:22]:
                print(f"  {v/len(evs):9.3f} ms/step ({v/tot*100:5.1f}%)  {k}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
