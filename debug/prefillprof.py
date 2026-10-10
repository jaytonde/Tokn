"""Directly time/profiler a prefill forward on real tokenized prompts."""
import sys
import time

sys.path.insert(0, "/home/Tokn")

import torch

from utils.configs import ServerConfigs
from src.run import Run
from utils.sequence import Sequence
from utils.context import reset_context


def main() -> None:
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
    run = Run(cfg, 0, [])

    import json
    texts = []
    with open("data/questions.jsonl") as f:
        for line in f:
            texts.append(json.loads(line)["question"])
            if len(texts) >= 6:
                break

    def tokenize(text):
        t = run.tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                              tokenize=False, add_generation_prompt=True)
        return run.tokenizer([t], return_tensors="pt")["input_ids"][0].tolist()

    def build_seq(token_ids, max_tokens=256):
        seq = Sequence(token_ids, max_tokens=max_tokens)
        alloc = token_ids + [0] * max_tokens
        cb = run.block_manager.match_prefix(token_ids)
        bt = run.block_manager.allocate(token_ids=alloc, num_cached_blocks=cb)
        seq.block_table_ids = bt
        seq.block_table = run.block_manager.make_block_table_tensor(bt, torch.device(run.device))
        seq.num_cached_tokens = cb * run.block_manager.block_size
        seq.num_hashed_blocks = cb
        seq.num_scheduled_tokens = seq.num_tokens - seq.num_cached_tokens
        return seq

    seqs = [build_seq(tokenize(t)) for t in texts]
    print(f"prompt lengths: {[len(s.token_ids) for s in seqs]}")

    with torch.inference_mode():
        for n in (1, 2, 3, 5):
            sel = seqs[:n]
            reset_context()
            t0 = time.perf_counter()
            input_ids, positions = run.prepare_prefill(sel)
            t_prep = time.perf_counter() - t0
            # warmup
            logits = run.run_model(input_ids, positions, True)
            torch.cuda.synchronize()
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            ev0.record()
            logits = run.run_model(input_ids, positions, True)
            ev1.record()
            torch.cuda.synchronize()
            gpu_ms = ev0.elapsed_time(ev1)
            wall = time.perf_counter() - t0
            print(f"n={n} tokens={input_ids.numel():4d} prep={t_prep*1e3:6.2f}ms gpu={gpu_ms:6.2f}ms wall={wall*1e3:6.2f}ms")

        # kernel breakdown for n=3
        sel = seqs[:3]
        reset_context()
        input_ids, positions = run.prepare_prefill(sel)
        with torch.inference_mode():
            logits = run.run_model(input_ids, positions, True)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                logits = run.run_model(input_ids, positions, True)
                torch.cuda.synchronize()
        tot = 0
        byname = {}
        for e in prof.key_averages():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                tot += e.self_device_time_total
                byname[e.key] = byname.get(e.key, 0) + e.self_device_time_total
        print(f"\n=== prefill n=3 tokens={input_ids.numel()}: {len(byname)} kernel types, sum={tot/1000:.1f}ms ===")
        for k, v in sorted(byname.items(), key=lambda kv: -kv[1])[:18]:
            print(f"  {v/1000:9.2f} ms ({v/tot*100:5.1f}%)  {k[:90]}")


if __name__ == "__main__":
    main()
