"""Correctness gate that exercises the RUN_LOOP (not engine.step directly).

Submits all 37 prompts to LLM, lets the background run_loop drive them (with
whatever TOKN_STEP_MODE the caller set), and compares the greedy completion
token ids to a reference JSON.

Usage: PYTHONPATH=/home/Tokn TOKN_STEP_MODE=split .venv/bin/python debug/correctness_runloop.py --ref /tmp/out_kvstore_bs1.json
"""
import argparse
import asyncio
import json
import sys

sys.path.insert(0, "/home/Tokn")
from src.llm import LLM
from utils.sampling_params import SamplingParams
from utils.configs import ServerConfigs


def load_prompts():
    prompts = []
    with open("data/prefix_cache_test_prompts.txt", "r", encoding="utf-8") as f:
        prompts = [line.rstrip("\n") for line in f if line.strip()]
    with open("data/questions.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= 37:
                break
            prompts.append(json.loads(line)["question"])
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None)
    ap.add_argument("--out", default="/tmp/runloop_out.json")
    ap.add_argument("--max-tokens", type=int, default=48)
    args = ap.parse_args()

    cfg = ServerConfigs(
        model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda",
        max_length=2048, max_model_len=2048, tensor_parallel_size=1,
        enforce_eager=False, max_num_seqs=5, max_num_batched_tokens=1024,
    )
    llm = LLM(
        model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda",
        max_model_len=2048, config=cfg,
    )
    prompts = load_prompts()

    async def run():
        if llm.background_task is None:
            llm.background_task = asyncio.create_task(llm.run_loop())
        seq_ids = []
        for p in prompts:
            seq_ids.append(await llm.add_async_request(
                p, SamplingParams(temperature=0.0, max_tokens=args.max_tokens)))
        states = [llm.request_states[s] for s in seq_ids]
        await asyncio.gather(*(s.done.wait() for s in states))
        outputs = {}
        for s in states:
            if s.error is not None:
                raise s.error
            outputs[str(s.seq.seq_id)] = s.seq.completion_token_ids
        return outputs

    outputs = asyncio.run(run())
    ordered = {k: outputs[k] for k in sorted(outputs, key=int)}
    with open(args.out, "w") as f:
        json.dump(ordered, f)
    total = sum(len(v) for v in ordered.values())
    print(f"runloop({__import__('os').environ.get('TOKN_STEP_MODE','split')}): "
          f"{len(ordered)} sequences, {total} tokens -> {args.out}")

    if args.ref:
        with open(args.ref) as f:
            ref = json.load(f)
        if list(ordered.keys()) != list(ref.keys()):
            print(f"FAIL: key order/ids differ ({len(ordered)} vs {len(ref)})")
            sys.exit(1)
        n_diff = 0
        for k in ref:
            if ref[k] != ordered[k]:
                n_diff += 1
                if n_diff <= 5:
                    print(f"  diff seq {k}: ref={ref[k][:12]} got={ordered[k][:12]}")
        if n_diff:
            print(f"CORRECTNESS FAIL: {n_diff}/{len(ref)} sequences differ")
            sys.exit(1)
        print(f"CORRECTNESS PASS: all {len(ref)} sequences identical")


if __name__ == "__main__":
    main()
