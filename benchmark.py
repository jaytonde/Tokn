"""
Online/continuous-batching benchmark: tokn vs official vLLM.

Runs the same prompts (data/questions.jsonl) through both engines and reports
throughput so they can be compared on equal footing.

vLLM uses its default compiled execution path unless --vllm-enforce-eager is
passed. vLLM is also configured to match tokn's feature set: prefix caching +
chunked prefill, same max_num_seqs and max_num_batched_tokens, and greedy
decoding (temperature=0).

Each framework runs in its own subprocess so that torch.distributed init,
CUDA context, and model memory from one engine never interfere with the other.

Usage:
    # benchmark both (default)
    python benchmark.py --model Qwen/Qwen3-0.6B --dtype bf16 --device cuda --max-tokens 256

    # only one engine
    python benchmark.py --framework tokn ...
    python benchmark.py --framework vllm ...
"""

import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
import time

# Keep tokn's scheduler defaults so vLLM is configured identically.
TOKN_MAX_NUM_SEQS = 5
TOKN_MAX_NUM_BATCHED_TOKENS = 1024

RESULT_MARKER = "BENCH_RESULT "


def arrival_delays(n: int, request_rate: float | None, seed: int = 0) -> list[float]:
    """Inter-arrival gaps (seconds) before each request is submitted.

    request_rate is in requests/second. None or inf means "submit everything at
    once" (the original burst behaviour). A finite rate draws exponential
    inter-arrival times (Poisson process), matching vLLM's benchmark_serving.
    """
    if request_rate is None or request_rate == float("inf") or request_rate <= 0:
        return [0.0] * n
    rng = random.Random(seed)
    return [rng.expovariate(request_rate) for _ in range(n)]


def load_questions(path: str, num_prompts: int | None) -> list[str]:
    prompts: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(obj["question"])
    if num_prompts is not None:
        prompts = prompts[:num_prompts]
    return prompts


def count_output_tokens(tokenizer, texts: list[str]) -> int:
    """Count generated tokens by re-encoding output text.

    Done identically for both engines so the throughput metric is apples-to-apples.
    """
    total = 0
    for t in texts:
        total += len(tokenizer.encode(t, add_special_tokens=False))
    return total


def _percentile(values, q):
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = (len(s) - 1) * (q / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def summarize(values):
    """mean / p50 / p99 summary for a list of latency samples (seconds)."""
    if not values:
        return {"mean": None, "p50": None, "p99": None}
    return {
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 50),
        "p99": _percentile(values, 99),
    }



# --------------------------------------------------------------------------- #
# tokn
# --------------------------------------------------------------------------- #
def run_tokn(args) -> dict:
    from llm import LLM
    from sampling_params import SamplingParams

    prompts = load_questions(args.questions, args.num_prompts)

    llm = LLM(
        model=args.model,
        dtype=args.dtype,                 # tokn expects "fp16" / "bf16"
        device=args.device,
        max_model_len=args.max_model_len,
    )

    sampling = [
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
        for _ in prompts
    ]

    async def run_continuous_batch():
        if llm.background_task is None:
            llm.background_task = asyncio.create_task(llm.run_loop())

        delays = arrival_delays(len(prompts), args.request_rate)

        # Submit requests over time (staggered) while the background loop is
        # already draining the queue. Each seq's arrival_time is stamped inside
        # add_async_request, so TTFT reflects real first-token latency rather
        # than how long a burst sat in the queue.
        seq_ids = []
        for (prompt, params), delay in zip(zip(prompts, sampling), delays):
            if delay > 0:
                await asyncio.sleep(delay)
            seq_id = await llm.add_async_request(prompt, params)
            seq_ids.append(seq_id)

        states = [llm.request_states[seq_id] for seq_id in seq_ids]
        await asyncio.gather(*(state.done.wait() for state in states))

        for state in states:
            if state.error is not None:
                raise state.error

        texts = [state.output or "" for state in states]
        out_tokens = sum(state.seq.num_generated_tokens for state in states)

        async with llm.request_lock:
            for seq_id in seq_ids:
                llm.request_states.pop(seq_id, None)

        finished_seqs = [state.seq for state in states]
        return texts, out_tokens, finished_seqs

    t0 = time.perf_counter()
    texts, out_tokens, finished_seqs = asyncio.run(run_continuous_batch())
    elapsed = time.perf_counter() - t0

    ttfts = [s.ttft for s in finished_seqs if s.ttft is not None]
    tpots = [s.tpot for s in finished_seqs if s.tpot is not None]
    itls = [gap for s in finished_seqs for gap in s.itls]

    return {
        "framework": "tokn",
        "num_prompts": len(prompts),
        "elapsed_s": elapsed,
        "output_tokens": out_tokens,
        "throughput_tok_s": out_tokens / elapsed if elapsed > 0 else 0.0,
        "self_reported_throughput_tok_s": None,
        "ttft_s": summarize(ttfts),
        "tpot_s": summarize(tpots),
        "itl_s": summarize(itls),
    }


# --------------------------------------------------------------------------- #
# official vLLM
# --------------------------------------------------------------------------- #
_VLLM_DTYPE = {"fp16": "float16", "bf16": "bfloat16", "float16": "float16", "bfloat16": "bfloat16"}


def _extract_vllm_metrics(request_outputs):
    """Pull TTFT / TPOT / ITL (seconds) out of vLLM RequestOutput.metrics."""
    ttfts, tpots, itls = [], [], []
    for o in request_outputs:
        if o is None:
            continue
        met = getattr(o, "metrics", None)
        if met is None:
            continue
        n_out = len(o.outputs[0].token_ids)

        ttft = getattr(met, "first_token_latency", None)
        if not ttft:
            # Standard vLLM RequestMetrics: derive from timestamps.
            arrival = getattr(met, "arrival_time", None)
            first_ts = getattr(met, "first_token_time", None) or getattr(met, "first_token_ts", None)
            if arrival is not None and first_ts is not None:
                ttft = first_ts - arrival
        if ttft:
            ttfts.append(ttft)

        first_ts = getattr(met, "first_token_ts", None) or getattr(met, "first_token_time", None)
        last_ts = getattr(met, "last_token_ts", None) or getattr(met, "last_token_time", None)
        if first_ts and last_ts and n_out > 1:
            tpot = (last_ts - first_ts) / (n_out - 1)
            tpots.append(tpot)
            itls.append(tpot)
    return ttfts, tpots, itls


def run_vllm(args) -> dict:
    prompts = load_questions(args.questions, args.num_prompts)

    if args.request_rate is not None:
        return _run_vllm_async(args, prompts)

    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=args.model,
        dtype=_VLLM_DTYPE.get(args.dtype, args.dtype),
        max_model_len=args.max_model_len,
        # False lets vLLM use torch.compile / CUDA graph capture when available.
        enforce_eager=args.vllm_enforce_eager,
        # Match tokn feature set.
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_seqs=TOKN_MAX_NUM_SEQS,
        max_num_batched_tokens=TOKN_MAX_NUM_BATCHED_TOKENS,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # Required so RequestOutput.metrics (TTFT / token timestamps) is populated.
        disable_log_stats=False,
    )
    if args.block_size is not None:
        llm_kwargs["block_size"] = args.block_size

    llm = LLM(**llm_kwargs)

    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    # Apply the model chat template, same as tokn's add_request does.
    conversations = [[{"role": "user", "content": q}] for q in prompts]
    request_outputs = []
    t0 = time.perf_counter()
    request_outputs.extend(llm.chat(conversations, sampling))
    elapsed = time.perf_counter() - t0

    out_tokens = sum(len(o.outputs[0].token_ids) for o in request_outputs)
    ttfts, tpots, itls = _extract_vllm_metrics(request_outputs)

    return {
        "framework": "vllm",
        "num_prompts": len(prompts),
        "elapsed_s": elapsed,
        "output_tokens": out_tokens,
        "throughput_tok_s": out_tokens / elapsed if elapsed > 0 else 0.0,
        "self_reported_throughput_tok_s": None,
        "ttft_s": summarize(ttfts),
        "tpot_s": summarize(tpots),
        "itl_s": summarize(itls),
    }


def _run_vllm_async(args, prompts) -> dict:
    """Staggered-arrival vLLM run via AsyncLLMEngine (request-rate mode)."""
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    engine_args = AsyncEngineArgs(
        model=args.model,
        dtype=_VLLM_DTYPE.get(args.dtype, args.dtype),
        max_model_len=args.max_model_len,
        enforce_eager=args.vllm_enforce_eager,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_seqs=TOKN_MAX_NUM_SEQS,
        max_num_batched_tokens=TOKN_MAX_NUM_BATCHED_TOKENS,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_log_stats=False,
        **({"block_size": args.block_size} if args.block_size is not None else {}),
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    # Pre-render chat templates so submission timing isn't skewed by tokenizing.
    text_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in prompts
    ]

    async def run_async():
        results = [None] * len(text_prompts)

        async def submit(i, prompt):
            final = None
            async for out in engine.generate(prompt, sampling, request_id=str(i)):
                final = out
            results[i] = final

        delays = arrival_delays(len(text_prompts), args.request_rate)
        tasks = []
        for i, (prompt, delay) in enumerate(zip(text_prompts, delays)):
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(submit(i, prompt)))
        await asyncio.gather(*tasks)
        return results

    t0 = time.perf_counter()
    request_outputs = asyncio.run(run_async())
    elapsed = time.perf_counter() - t0

    out_tokens = sum(len(o.outputs[0].token_ids) for o in request_outputs if o is not None)
    ttfts, tpots, itls = _extract_vllm_metrics(request_outputs)

    return {
        "framework": "vllm",
        "num_prompts": len(prompts),
        "elapsed_s": elapsed,
        "output_tokens": out_tokens,
        "throughput_tok_s": out_tokens / elapsed if elapsed > 0 else 0.0,
        "self_reported_throughput_tok_s": None,
        "ttft_s": summarize(ttfts),
        "tpot_s": summarize(tpots),
        "itl_s": summarize(itls),
    }


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run_single(args) -> dict:
    if args.framework == "tokn":
        return run_tokn(args)
    elif args.framework == "vllm":
        return run_vllm(args)
    raise ValueError(f"Unknown framework for single run: {args.framework}")


def run_in_subprocess(framework: str, args) -> dict | None:
    # tokn and vLLM may live in different conda envs (different torch builds),
    # so each framework is launched with its own interpreter when provided.
    python_exe = sys.executable
    if framework == "tokn" and args.tokn_python:
        python_exe = args.tokn_python
    elif framework == "vllm" and args.vllm_python:
        python_exe = args.vllm_python

    cmd = [
        python_exe, os.path.abspath(__file__),
        "--framework", framework,
        "--model", args.model,
        "--dtype", args.dtype,
        "--device", args.device,
        "--max-tokens", str(args.max_tokens),
        "--max-model-len", str(args.max_model_len),
        "--questions", args.questions,
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--emit-json",
    ]
    if args.request_rate is not None:
        cmd += ["--request-rate", str(args.request_rate)]
    if args.num_prompts is not None:
        cmd += ["--num-prompts", str(args.num_prompts)]
    if args.block_size is not None:
        cmd += ["--block-size", str(args.block_size)]
    if args.vllm_enforce_eager:
        cmd += ["--vllm-enforce-eager"]

    print(f"\n=== Running {framework} in subprocess ({python_exe}) ===", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        print(f"!! {framework} subprocess failed (exit {proc.returncode})")
        return None

    result = None
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_MARKER):
            result = json.loads(line[len(RESULT_MARKER):])
    if result is None:
        sys.stderr.write(proc.stderr)
        print(f"!! Could not parse result from {framework} output")
    return result


def print_comparison(results: list[dict], request_rate: float | None = None):
    results = [r for r in results if r]
    if not results:
        print("No results to compare.")
        return

    print("\n" + "=" * 72)
    print("BENCHMARK COMPARISON")
    workload = f"request-rate={request_rate} req/s" if request_rate is not None else "all-at-once burst"
    print(f"workload: {workload}")
    print("=" * 72)
    header = f"{'framework':<10}{'prompts':>9}{'out_tokens':>12}{'elapsed_s':>12}{'tok/s':>12}"
    print(header)
    print("-" * 72)
    for r in results:
        print(f"{r['framework']:<10}{r['num_prompts']:>9}{r['output_tokens']:>12}"
              f"{r['elapsed_s']:>12.3f}{r['throughput_tok_s']:>12.2f}")
    print("-" * 72)

    # Latency table (TTFT / TPOT / ITL), values in milliseconds.
    def ms(x):
        return f"{x * 1000:.2f}" if x is not None else "n/a"

    print("\nLatency (ms)   metric        mean         p50         p99")
    print("-" * 72)
    for r in results:
        for label, key in (("TTFT", "ttft_s"), ("TPOT", "tpot_s"), ("ITL", "itl_s")):
            s = r.get(key) or {"mean": None, "p50": None, "p99": None}
            print(f"{r['framework']:<10}{label:<10}"
                  f"{ms(s['mean']):>12}{ms(s['p50']):>12}{ms(s['p99']):>12}")
        print("-" * 72)

    by_name = {r["framework"]: r for r in results}
    if "tokn" in by_name and "vllm" in by_name:
        t = by_name["tokn"]["throughput_tok_s"]
        v = by_name["vllm"]["throughput_tok_s"]
        if t > 0:
            print(f"vLLM is {v / t:.2f}x tokn throughput "
                  f"(tokn={t:.2f} tok/s, vllm={v:.2f} tok/s)")
    print("=" * 72)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark tokn vs official vLLM (offline inference).")
    p.add_argument("--framework", choices=["tokn", "vllm", "both"], default="both")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--dtype", default="bf16", help="fp16 or bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--num-prompts", type=int, default=None, help="Limit number of prompts.")
    p.add_argument("--questions", default=os.path.join("data", "questions.jsonl"))
    p.add_argument("--request-rate", type=float, default=None,
                   help="Requests/second (Poisson arrivals). Omit for an all-at-once burst. "
                        "Use a finite rate to measure TTFT under realistic staggered load.")
    p.add_argument("--block-size", type=int, default=None, help="KV block size for vLLM (tokn uses 256).")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--tokn-python", default=None,
                   help="Python interpreter to run tokn (its conda env). Used in 'both' mode.")
    p.add_argument("--vllm-python", default=None,
                   help="Python interpreter to run vLLM (its conda env). Used in 'both' mode.")
    p.add_argument("--vllm-enforce-eager", action="store_true",
                   help="Force vLLM eager mode. Omit this to allow vLLM torch.compile/CUDA graphs.")
    p.add_argument("--emit-json", action="store_true",
                   help="Print machine-readable result line (used by subprocess runner).")
    return p


def main():
    args = build_parser().parse_args()

    if args.framework == "both":
        results = [
            run_in_subprocess("tokn", args),
            run_in_subprocess("vllm", args),
        ]
        print_comparison(results, request_rate=args.request_rate)
        return

    result = run_single(args)

    print(f"\n[{result['framework']}] "
          f"prompts={result['num_prompts']} "
          f"output_tokens={result['output_tokens']} "
          f"elapsed={result['elapsed_s']:.3f}s "
          f"throughput={result['throughput_tok_s']:.2f} tok/s")

    def ms(x):
        return f"{x * 1000:.2f}ms" if x is not None else "n/a"

    for label, key in (("TTFT", "ttft_s"), ("TPOT", "tpot_s"), ("ITL", "itl_s")):
        s = result.get(key) or {"mean": None, "p50": None, "p99": None}
        print(f"  {label:<5} mean={ms(s['mean'])} p50={ms(s['p50'])} p99={ms(s['p99'])}")

    if args.emit_json:
        print(RESULT_MARKER + json.dumps(result))


if __name__ == "__main__":
    main()
