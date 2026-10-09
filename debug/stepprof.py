"""Break down one engine decode step: scheduler, prepare, GPU, argmax, postproc.

Drives the engine synchronously (no run_loop), submits N requests, and after a
warm prefill runs M decode steps while timing each phase with perf_counter and
CUDA events around the graph replay.

Usage: PYTHONPATH=/home/Tokn .venv/bin/python debug/stepprof.py --num-prompts 20
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
            await engine.add_async_request(p, SamplingParams(temperature=0.0, max_tokens=64))

        t_sched = t_prep = t_gpu = t_samp = t_post = t_hash = t_dealloc = 0.0
        n_decode = 0
        gpu_ev = []
        guard = 0
        while not engine.scheduler.is_finished():
            t0 = time.perf_counter()
            seqs, is_prefill = engine.scheduler.schedule()
            t_sched += time.perf_counter() - t0
            if not seqs:
                break
            if not is_prefill:
                n_decode += 1

            t0 = time.perf_counter()
            input_ids, positions = engine.RUN.prepare_decode(seqs) if not is_prefill else (None, None)
            if is_prefill:
                input_ids, positions = engine.RUN.prepare_prefill(seqs)
            t_prep += time.perf_counter() - t0

            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            with torch.inference_mode():
                ev0.record()
                logits = engine.RUN.run_model(input_ids, positions, is_prefill)
                ev1.record()
            torch.cuda.synchronize()
            t_gpu += time.perf_counter() - t0
            gpu_ev.append(ev0.elapsed_time(ev1))

            t0 = time.perf_counter()
            next_token_ids = torch.argmax(logits, dim=-1).tolist()
            t_samp += time.perf_counter() - t0

            t0 = time.perf_counter()
            finished = engine.scheduler.postprocess(
                seqs=seqs, token_ids=next_token_ids, is_prefill=is_prefill
            )
            t_post += time.perf_counter() - t0

            t0 = time.perf_counter()
            for seq in seqs:
                start_block = seq.num_hashed_blocks
                end_block = seq.num_cached_tokens // engine.block_size
                if end_block > start_block:
                    engine.block_manager.hash_completed_blocks(
                        token_ids=seq.token_ids,
                        block_table=seq.block_table_ids,
                        start_block=start_block,
                        end_block=end_block,
                    )
                    seq.num_hashed_blocks = end_block
            t_hash += time.perf_counter() - t0

            t0 = time.perf_counter()
            for seq in finished:
                engine.block_manager.deallocate(seq.block_table_ids)
            t_dealloc += time.perf_counter() - t0

            guard += 1
            if guard > 100000:
                raise RuntimeError("too many steps")
            if n_decode > 60:
                break

        tot = t_sched + t_prep + t_gpu + t_samp + t_post + t_hash + t_dealloc
        print(f"decode steps profiled: {n_decode}")
        print(f"  schedule    {t_sched/n_decode*1e3:7.3f} ms/step ({t_sched/tot*100:5.1f}%)")
        print(f"  prepare     {t_prep/n_decode*1e3:7.3f} ms/step ({t_prep/tot*100:5.1f}%)")
        print(f"  gpu+python  {t_gpu/n_decode*1e3:7.3f} ms/step ({t_gpu/tot*100:5.1f}%)  [cuda ev mean {sum(gpu_ev)/len(gpu_ev):.3f} ms]")
        print(f"  argmax/tol  {t_samp/n_decode*1e3:7.3f} ms/step ({t_samp/tot*100:5.1f}%)")
        print(f"  postprocess {t_post/n_decode*1e3:7.3f} ms/step ({t_post/tot*100:5.1f}%)")
        print(f"  hash        {t_hash/n_decode*1e3:7.3f} ms/step ({t_hash/tot*100:5.1f}%)")
        print(f"  dealloc     {t_dealloc/n_decode*1e3:7.3f} ms/step ({t_dealloc/tot*100:5.1f}%)")

    asyncio.run(run())


if __name__ == "__main__":
    main()
