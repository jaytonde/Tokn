"""Eager decode phase timing at bs=5 with CUDA events (no graph, no profiler).

Loads the engine, prefills 5 seqs to a decode state, then times the eager
model forward phase-by-phase: embed, per-layer (norm/qkv+attn/o+norm+mlp),
final norm, lm_head.

Usage: PYTHONPATH=/home/Tokn .venv/bin/python debug/eager_phase.py
"""

import sys
sys.path.insert(0, "/home/Tokn")

import torch
from utils.configs import ServerConfigs
from src.engine import Engine
from utils.sampling_params import SamplingParams


def load_questions(n=20):
    import json
    out = []
    with open("data/questions.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line)["question"])
            if len(out) >= n:
                break
    return out


def main():
    cfg = ServerConfigs(
        model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda",
        max_length=2048, max_model_len=2048, tensor_parallel_size=1,
        enforce_eager=False, max_num_seqs=5, max_num_batched_tokens=1024,
    )
    engine = Engine(
        model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda",
        max_model_len=2048, config=cfg,
    )
    run = engine.RUN

    async def run_all():
        prompts = load_questions(20)
        for p in prompts:
            await engine.add_async_request(p, SamplingParams(temperature=0.0, max_tokens=256))
        # Steady bs=5 decode state.
        guard = 0
        while True:
            seqs, is_prefill = engine.scheduler.schedule()
            if not seqs:
                break
            with torch.inference_mode():
                if is_prefill:
                    input_ids, positions = run.prepare_prefill(seqs)
                    logits = run.run_model(input_ids, positions, True)
                else:
                    input_ids, positions = run.prepare_decode(seqs)
                    logits = run.run_model(input_ids, positions, False)
                engine.scheduler.postprocess(seqs, torch.argmax(logits, dim=-1).tolist(), is_prefill)
            if not is_prefill:
                break
            guard += 1
            if guard > 200:
                raise RuntimeError("prefill loop")
        print("steady decode reached, bs=", len(seqs))

        # Now time eager forward phase by phase.
        model = run.custom_model
        layers = model.model.layers

        def ev():
            return torch.cuda.Event(enable_timing=True)

        totals = {k: 0.0 for k in
                  ["embed", "attn_norm", "attn", "mlp_norm", "mlp", "final_norm", "lm_head"]}
        N = 0
        for _ in range(30):
            seqs, is_prefill = engine.scheduler.schedule()
            if not seqs or is_prefill:
                if seqs:
                    with torch.inference_mode():
                        input_ids, positions = run.prepare_prefill(seqs)
                        logits = run.run_model(input_ids, positions, True)
                        engine.scheduler.postprocess(seqs, torch.argmax(logits, dim=-1).tolist(), True)
                continue
            bs = len(seqs)
            with torch.inference_mode():
                input_ids, positions = run.prepare_decode(seqs)
                e0, e1, e2, e3, e4 = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
                e5, e6, e7, e8 = [torch.cuda.Event(enable_timing=True) for _ in range(4)]

                e0.record()
                hidden = model.model.embed_tokens(input_ids)
                e1.record()
                residual = None
                for layer in layers:
                    if residual is None:
                        hidden, residual = layer.input_layernorm(hidden), hidden
                    else:
                        hidden, residual = layer.input_layernorm(hidden, residual)
                    e2.record()
                    hidden = layer.self_attn(positions, hidden)
                    e3.record()
                    hidden, residual = layer.post_attention_layernorm(hidden, residual)
                    e4.record()
                    hidden = layer.mlp(hidden)
                    e5.record()
                    totals["attn_norm"] += e2.elapsed_time(e3)
                    totals["attn"] += e3.elapsed_time(e4)
                    totals["mlp_norm"] += e4.elapsed_time(e5)
                totals["embed"] += e0.elapsed_time(e1)
                totals["attn_norm"] += e1.elapsed_time(e2)
                e6.record()
                hidden, _ = model.model.norm(hidden, residual)
                e7.record()
                logits = model.lm_head(hidden)
                e8.record()
            torch.cuda.synchronize()
            totals["final_norm"] += e6.elapsed_time(e7)
            totals["lm_head"] += e7.elapsed_time(e8)
            N += 1
            engine.scheduler.postprocess(seqs, torch.argmax(logits, dim=-1).tolist(), False)

        print(f"\neager decode bs={bs}, {N} steps:")
        tot = sum(totals.values()) / N
        for k, v in totals.items():
            print(f"  {k:12s}: {v/N*1000:8.1f} us  ({v/tot/N*100:5.1f}%)")
        print(f"  {'TOTAL':12s}: {tot*1000:8.1f} us")

    import asyncio
    asyncio.run(run_all())


if __name__ == "__main__":
    main()
