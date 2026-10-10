"""Test: does CUDA-graph replay eliminate the ~34ms eager prefill launch overhead?

Captures a prefill graph at a fixed token count and compares replay wall vs eager.
Also times eager vs graph for a real (padded) prefill shape.
"""
import sys
import time

sys.path.insert(0, "/home/Tokn")

import torch

from utils.configs import ServerConfigs
from src.run import Run
from utils.sequence import Sequence
from utils.context import get_context, set_context, reset_context


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
    dev = torch.device("cuda")

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
    lengths = [len(s.token_ids) for s in seqs]
    print(f"lengths={lengths}")

    # prepare a prefill for all 5 (batched, real cu_seqlens)
    with torch.inference_mode():
        reset_context()
        input_ids, positions = run.prepare_prefill(seqs[:5])
        S = input_ids.numel()
        print(f"batched prefill S={S}")

        # eager timing
        for _ in range(3):
            logits = run.run_model(input_ids, positions, True)
        torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        logits_e = run.run_model(input_ids, positions, True)
        ev1.record()
        torch.cuda.synchronize()
        print(f"eager prefill: {ev0.elapsed_time(ev1):.2f} ms")

        # capture a graph for this exact prefill shape
        ctx = get_context()
        g_input = torch.zeros_like(input_ids)
        g_pos = torch.zeros_like(positions)
        g_cu_q = ctx.cu_seqlens_q.clone()
        g_cu_k = ctx.cu_seqlens_k.clone()
        g_bt = ctx.block_tables.clone()
        g_slot = ctx.slot_mapping.clone()
        g_out = torch.zeros_like(logits_e)
        set_context(is_prefill=True, cu_seqlens_q=g_cu_q, cu_seqlens_k=g_cu_k,
                    max_seqlen_q=ctx.max_seqlen_q, max_seqlen_k=ctx.max_seqlen_k,
                    slot_mapping=g_slot, block_tables=g_bt)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            g_out[:] = run.custom_model(g_input, g_pos)
            with torch.cuda.graph(graph):
                g_out[:] = run.custom_model(g_input, g_pos)
        torch.cuda.synchronize()
        # replay
        for _ in range(3):
            g_input[:] = input_ids
            g_pos[:] = positions
            graph.replay()
        torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
        g_input[:] = input_ids; g_pos[:] = positions
        ev0.record()
        graph.replay()
        ev1.record()
        torch.cuda.synchronize()
        print(f"graph prefill replay: {ev0.elapsed_time(ev1):.2f} ms")

        # compare outputs
        reset_context()
        logits_g = g_out.detach().clone()
        print(f"eager vs graph prefill logits bit_eq={torch.equal(logits_e, logits_g)} max|d|={(logits_e.float()-logits_g.float()).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
