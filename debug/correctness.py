"""
Correctness gate for tokn optimizations.

Drives the engine synchronously (no run_loop) and writes greedy output token-ids
per prompt to a JSON file, or compares a run against a reference file.

Modes:
  --mode eager   : enforce_eager=True  (reference / known-good path)
  --mode graph   : enforce_eager=False (CUDA graph decode path)
  --mode compare : --ref <file> compares the produced outputs to a reference

Prompts = data/prefix_cache_test_prompts.txt + first N of data/questions.jsonl.
This exercises prefix caching, chunked prefill, and multi-seq batching.

Usage:
  PYTHONPATH=/home/Tokn .venv/bin/python debug/correctness.py --mode eager --out /tmp/ref.json
  PYTHONPATH=/home/Tokn .venv/bin/python debug/correctness.py --mode graph --out /tmp/graph.json
  PYTHONPATH=/home/Tokn .venv/bin/python debug/correctness.py --mode compare --ref /tmp/ref.json --out /tmp/graph.json
"""

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, "/home/Tokn")

from utils.configs import ServerConfigs
from src.engine import Engine
from utils.sampling_params import SamplingParams


def load_prompts(num_questions: int = 20) -> list[str]:
    prompts: list[str] = []
    with open("data/prefix_cache_test_prompts.txt", "r", encoding="utf-8") as f:
        prompts = [line.rstrip("\n") for line in f if line.strip()]
    with open("data/questions.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= num_questions + 17:
                break
            prompts.append(json.loads(line)["question"])
    return prompts


async def generate(engine: Engine, prompts: list[str], max_tokens: int) -> dict:
    seq_ids = []
    for p in prompts:
        seq_id = await engine.add_async_request(
            p, SamplingParams(temperature=0.0, max_tokens=max_tokens)
        )
        seq_ids.append(seq_id)

    outputs: dict[int, list[int]] = {}
    guard = 0
    while not engine.scheduler.is_finished():
        finished = engine.step()
        for seq in finished:
            outputs[seq.seq_id] = seq.completion_token_ids
        guard += 1
        if guard > 100000:
            raise RuntimeError("correctness driver did not finish")
    return outputs


async def generate_one_at_a_time(
    engine: Engine, prompts: list[str], max_tokens: int
) -> dict:
    """Batch size 1: process each prompt alone so decode never batches.

    This is the gate for bf16 numerics: model-math changes must be bit-identical
    here (no near-tie ambiguity from batching).
    """
    outputs: dict[int, list[int]] = {}
    for p in prompts:
        seq_id = await engine.add_async_request(
            p, SamplingParams(temperature=0.0, max_tokens=max_tokens)
        )
        guard = 0
        while seq_id not in outputs:
            finished = engine.step()
            for seq in finished:
                outputs[seq.seq_id] = seq.completion_token_ids
            guard += 1
            if guard > 10000:
                raise RuntimeError("did not finish seq")
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eager", "graph", "compare"], required=True)
    ap.add_argument("--out", default="/tmp/correctness_out.json")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--num-questions", type=int, default=20)
    ap.add_argument("--seq-at-a-time", action="store_true",
                    help="process one prompt at a time (batch size 1)")
    args = ap.parse_args()

    prompts = load_prompts(args.num_questions)

    if args.mode == "compare":
        with open(args.ref, "r") as f:
            ref = json.load(f)
        with open(args.out, "r") as f:
            got = json.load(f)
        if list(got.keys()) != list(ref.keys()):
            print(f"FAIL: key order/ids differ ({len(got)} vs {len(ref)})")
            sys.exit(1)
        n_diff = 0
        for k in ref:
            if ref[k] != got[k]:
                n_diff += 1
                if n_diff <= 10:
                    print(f"  diff seq {k}: ref len={len(ref[k])} got len={len(got[k])}")
                    print(f"    ref={ref[k][:20]}")
                    print(f"    got={got[k][:20]}")
        if n_diff:
            print(f"CORRECTNESS FAIL: {n_diff}/{len(ref)} sequences differ")
            sys.exit(1)
        print(f"CORRECTNESS PASS: all {len(ref)} sequences identical")
        sys.exit(0)

    enforce_eager = args.mode == "eager"
    cfg = ServerConfigs(
        model="Qwen/Qwen3-0.6B",
        dtype="bf16",
        device="cuda",
        max_length=2048,
        max_model_len=2048,
        tensor_parallel_size=1,
        enforce_eager=enforce_eager,
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
    if args.seq_at_a_time:
        outputs = asyncio.run(generate_one_at_a_time(engine, prompts, args.max_tokens))
    else:
        outputs = asyncio.run(generate(engine, prompts, args.max_tokens))

    ordered = {str(seq_id): outputs[seq_id] for seq_id in sorted(outputs)}
    with open(args.out, "w") as f:
        json.dump(ordered, f)

    total = sum(len(v) for v in ordered.values())
    print(f"{args.mode}: {len(ordered)} sequences, {total} output tokens -> {args.out}")

    if args.mode == "graph":
        print("graph mode: graph_bs=", engine.RUN.graph_bs)


if __name__ == "__main__":
    main()
