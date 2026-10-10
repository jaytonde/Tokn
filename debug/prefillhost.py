"""Count kernel launches and phase-level host time in eager prefill."""
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
            if len(texts) >= 5:
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
    with torch.inference_mode():
        reset_context()
        input_ids, positions = run.prepare_prefill(seqs[:5])
        # warmup (JIT compile triton kernels)
        logits = run.run_model(input_ids, positions, True)
        torch.cuda.synchronize()

        # count launches
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            logits = run.run_model(input_ids, positions, True)
            torch.cuda.synchronize()
        evts = [e for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA]
        total_launches = sum(e.count for e in evts)
        print(f"prefill S={input_ids.numel()}: total kernel launches={total_launches}, kernel types={len(evts)}")

        # phase host timing (no sync between phases, so this is enqueue time)
        model = run.custom_model.model
        def time_phase(name, fn):
            t0 = time.perf_counter()
            fn()
            return (time.perf_counter() - t0) * 1000

        hs = model.embed_tokens(input_ids)
        residual = None
        layers = list(model.layers)
        layer_ms = []
        t_layer0 = time.perf_counter()
        for li, layer in enumerate(layers):
            t0 = time.perf_counter()
            hs, residual = layer(positions, hs, residual)
            layer_ms.append((time.perf_counter() - t0) * 1000)
        hs, _ = model.norm(hs, residual)
        t_norm = (time.perf_counter() - t_layer0) * 1000 - sum(layer_ms)
        lgs = run.custom_model.compute_logits(hs)
        torch.cuda.synchronize()
        print(f"phase host-enqueue times (no sync): embed+28 layers total={sum(layer_ms):.2f}ms "
              f"mean/layer={sum(layer_ms)/28:.2f}ms norm={t_norm:.2f}ms lm_head enqueue ~0 (async)")
        print(f"  slowest layers: {sorted(layer_ms, reverse=True)[:5]}")


if __name__ == "__main__":
    main()
