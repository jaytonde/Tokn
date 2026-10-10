"""Break down prefill-graph capture cost: warmup vs capture vs sync."""
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
        model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda",
        max_length=2048, max_model_len=2048, tensor_parallel_size=1,
        enforce_eager=False, max_num_seqs=5, max_num_batched_tokens=1024,
    )
    run = Run(cfg, 0, [])
    dev = torch.device("cuda")

    import json
    texts = []
    with open("data/questions.jsonl") as f:
        for line in f:
            texts.append(json.loads(line)["question"])
            if len(texts) >= 3:
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
        input_ids, positions = run.prepare_prefill(seqs[:2])
        S = input_ids.numel()
        ctx = get_context()

        # capture like run._capture_prefill_graph but timed
        num_seqs, max_q, max_k = 2, ctx.max_seqlen_q, ctx.max_seqlen_k
        g_input = torch.zeros(S, dtype=torch.long, device=dev)
        g_pos = torch.zeros(S, dtype=torch.long, device=dev)
        g_cu_q = torch.zeros(num_seqs + 1, dtype=torch.int32, device=dev)
        g_cu_k = torch.zeros(num_seqs + 1, dtype=torch.int32, device=dev)
        g_slot = torch.full((S,), -1, dtype=torch.int32, device=dev)
        g_bt = torch.zeros(num_seqs, run.max_num_blocks, dtype=torch.int32, device=dev)
        g_out = torch.zeros(num_seqs, run.hf_config.vocab_size, dtype=run.dtype, device=dev)
        g_input[:] = input_ids; g_pos[:] = positions
        g_cu_q[:] = ctx.cu_seqlens_q; g_cu_k[:] = ctx.cu_seqlens_k
        g_bt[:num_seqs] = ctx.block_tables
        set_context(is_prefill=True, cu_seqlens_q=g_cu_q, cu_seqlens_k=g_cu_k,
                    max_seqlen_q=max_q, max_seqlen_k=max_k,
                    slot_mapping=g_slot, block_tables=g_bt)

        t0 = time.perf_counter()
        g_out[:] = run.custom_model(g_input, g_pos)
        torch.cuda.synchronize()
        print(f"warmup forward: {(time.perf_counter()-t0)*1e3:.1f} ms (S={S})")

        t0 = time.perf_counter()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            g_out[:] = run.custom_model(g_input, g_pos)
        t_cap = time.perf_counter() - t0
        torch.cuda.synchronize()
        t_sync = time.perf_counter() - t0
        print(f"capture body: {t_cap*1e3:.1f} ms, +sync: {t_sync*1e3:.1f} ms")

        # replay timing
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        print(f"replay: {(time.perf_counter()-t0)*1e3:.1f} ms")

        reset_context()

        # now capture a few more shapes to separate JIT from steady-state capture cost
        for k in range(1, 4):
            sel = seqs[:1]
            reset_context()
            input_ids2, positions2 = run.prepare_prefill(sel)
            S2 = input_ids2.numel()
            ctx2 = get_context()
            gi = torch.zeros(S2, dtype=torch.long, device=dev)
            gp = torch.zeros(S2, dtype=torch.long, device=dev)
            gcq = torch.zeros(2, dtype=torch.int32, device=dev)
            gck = torch.zeros(2, dtype=torch.int32, device=dev)
            gs = torch.full((S2,), -1, dtype=torch.int32, device=dev)
            gb = torch.zeros(1, run.max_num_blocks, dtype=torch.int32, device=dev)
            go = torch.zeros(1, run.hf_config.vocab_size, dtype=run.dtype, device=dev)
            gi[:] = input_ids2; gp[:] = positions2
            gcq[:] = ctx2.cu_seqlens_q; gck[:] = ctx2.cu_seqlens_k
            gb[:1] = ctx2.block_tables
            set_context(is_prefill=True, cu_seqlens_q=gcq, cu_seqlens_k=gck,
                        max_seqlen_q=ctx2.max_seqlen_q, max_seqlen_k=ctx2.max_seqlen_k,
                        slot_mapping=gs, block_tables=gb)
            t0 = time.perf_counter()
            go[:] = run.custom_model(gi, gp)
            torch.cuda.synchronize()
            w = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter()
            g2 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g2):
                go[:] = run.custom_model(gi, gp)
            torch.cuda.synchronize()
            c = (time.perf_counter() - t0) * 1e3
            print(f"shape {k}: S={S2} warmup={w:.1f}ms capture={c:.1f}ms")
            reset_context()


if __name__ == "__main__":
    main()
