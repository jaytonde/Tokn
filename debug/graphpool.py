"""Test whether shared-pool multi-graph capture causes the decode divergence.

Builds a Run (graphs captured the standard way), reproduces a decode state,
then captures a FRESH bs=1 graph in its own private pool with the same inputs
and compares its output to (a) the standard run's bs=1 graph and (b) eager.

Usage:
  PYTHONPATH=/home/Tokn .venv/bin/python debug/graphpool.py --prompt-line 5 --decode-step 10
"""

import argparse
import sys

sys.path.insert(0, "/home/Tokn")

import torch

from utils.configs import ServerConfigs
from src.run import Run
from utils.sequence import Sequence
from utils.context import get_context, set_context as sc, reset_context


def load_prompt(line_no: int) -> str:
    with open("data/prefix_cache_test_prompts.txt", "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    return lines[line_no]


def tokenize(run: Run, prompt_text: str) -> list[int]:
    text = run.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return run.tokenizer([text], return_tensors="pt")["input_ids"][0].tolist()


def build_seq(run: Run, token_ids: list[int], max_tokens: int = 64):
    seq = Sequence(token_ids, max_tokens=max_tokens)
    alloc_token_ids = token_ids + [0] * max_tokens
    cached_blocks = run.block_manager.match_prefix(token_ids)
    block_table_ids = run.block_manager.allocate(
        token_ids=alloc_token_ids, num_cached_blocks=cached_blocks
    )
    seq.block_table_ids = block_table_ids
    seq.block_table = run.block_manager.make_block_table_tensor(
        block_table_ids, torch.device(run.device)
    )
    seq.num_cached_tokens = cached_blocks * run.block_manager.block_size
    seq.num_hashed_blocks = cached_blocks
    seq.num_scheduled_tokens = seq.num_tokens - seq.num_cached_tokens
    return seq


def prefill(run: Run, seq: Sequence):
    with torch.inference_mode():
        input_ids, positions = run.prepare_prefill([seq])
        logits = run.run_model(input_ids, positions, is_prefill=True)
        seq.num_cached_tokens += seq.num_scheduled_tokens
        seq.num_scheduled_tokens = 0
        tok = int(torch.argmax(logits[0]).item())
        seq.append_token(tok)
        run.block_manager.hash_completed_blocks(
            token_ids=seq.token_ids,
            block_table=seq.block_table_ids,
            start_block=seq.num_hashed_blocks,
            end_block=len(seq.token_ids) // run.block_size,
        )
        seq.num_hashed_blocks = len(seq.token_ids) // run.block_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-line", type=int, default=5)
    ap.add_argument("--decode-step", type=int, default=10)
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
    run = Run(cfg, 0, [])

    token_ids = tokenize(run, load_prompt(args.prompt_line))
    seq = build_seq(run, token_ids)
    prefill(run, seq)

    base_kv = run.kv_cache.clone()
    with torch.inference_mode():
        for _ in range(args.decode_step):
            run.enforce_eager = True
            run.kv_cache.copy_(base_kv)
            input_ids, positions = run.prepare_decode([seq])
            logits = run.run_model(input_ids, positions, is_prefill=False)
            tok = int(torch.argmax(logits[0]).item())
            seq.append_token(tok)
            base_kv = run.kv_cache.clone()
    print(f"seq length at probe: {len(seq.token_ids)}", flush=True)
    torch.cuda.synchronize()

    input_ids, positions = run.prepare_decode([seq])
    ctx = get_context()
    hidden = run.hf_config.hidden_size

    # Eager reference.
    run.enforce_eager = True
    with torch.inference_mode():
        hs_e = run.custom_model.model(input_ids, positions)
    torch.cuda.synchronize()

    # Fresh bs=1 graph, private pool.
    bs = 1
    gv = run.graph_vars
    with torch.inference_mode():
        gv["input_ids"][:bs] = input_ids
        gv["positions"][:bs] = positions
        gv["slot_mapping"].fill_(-1)
        gv["slot_mapping"][:bs] = ctx.slot_mapping
        gv["context_lens"].zero_()
        gv["context_lens"][:bs] = ctx.context_lens
        gv["block_tables"][:bs, : ctx.block_tables.size(1)] = ctx.block_tables

    # capture a fresh bs=1 graph with a private pool
    fresh_graph = torch.cuda.CUDAGraph()
    fresh_out = torch.zeros(bs, hidden, dtype=run.dtype, device=run.device)
    with torch.inference_mode():
        fresh_out[:bs] = run.custom_model.model(gv["input_ids"][:bs], gv["positions"][:bs])
        with torch.cuda.graph(fresh_graph):
            fresh_out[:bs] = run.custom_model.model(gv["input_ids"][:bs], gv["positions"][:bs])

    # Replay the STANDARD bs=1 graph.
    run.enforce_eager = False
    std_graph = run.graphs[1]
    with torch.inference_mode():
        gv["input_ids"][:bs] = input_ids
        gv["positions"][:bs] = positions
        gv["slot_mapping"].fill_(-1)
        gv["slot_mapping"][:bs] = ctx.slot_mapping
        gv["context_lens"].zero_()
        gv["context_lens"][:bs] = ctx.context_lens
        gv["block_tables"][:bs, : ctx.block_tables.size(1)] = ctx.block_tables
        std_graph.replay()
        std_out = gv["outputs"][:bs].detach().clone()

    # Replay the fresh bs=1 graph.
    with torch.inference_mode():
        gv["input_ids"][:bs] = input_ids
        gv["positions"][:bs] = positions
        gv["slot_mapping"].fill_(-1)
        gv["slot_mapping"][:bs] = ctx.slot_mapping
        gv["context_lens"].zero_()
        gv["context_lens"][:bs] = ctx.context_lens
        gv["block_tables"][:bs, : ctx.block_tables.size(1)] = ctx.block_tables
        fresh_graph.replay()
        fresh_out_val = fresh_out[:bs].detach().clone()
    torch.cuda.synchronize()

    for name, a, b in [("std-graph vs eager", std_out, hs_e),
                       ("fresh-graph vs eager", fresh_out_val, hs_e),
                       ("std-graph vs fresh-graph", std_out, fresh_out_val)]:
        md = (a - b).abs().max().item()
        rel = md / (b.abs().max().item() + 1e-9)
        print(f"{name}: max|d|= {md:.5g} rel= {rel:.2e} bit_eq={torch.equal(a, b)}", flush=True)

    # Compare K/V written to the cache at the current slot.
    pos = len(seq.token_ids) - 1
    logical_block_id = pos // run.block_size
    block_offset = pos % run.block_size
    physical_block_id = seq.block_table[0, logical_block_id].item()
    slot = physical_block_id * run.block_size + block_offset

    # eager KV
    run.kv_cache.copy_(base_kv)
    with torch.inference_mode():
        run.enforce_eager = True
        input_ids2, positions2 = run.prepare_decode([seq])
        run.custom_model(input_ids2, positions2)
    torch.cuda.synchronize()
    ke = run.kv_cache[0, :, physical_block_id, block_offset].clone()

    # std-graph KV
    run.kv_cache.copy_(base_kv)
    with torch.inference_mode():
        run.enforce_eager = False
        gv["input_ids"][:bs] = input_ids
        gv["positions"][:bs] = positions
        gv["slot_mapping"].fill_(-1)
        gv["slot_mapping"][:bs] = ctx.slot_mapping
        gv["context_lens"].zero_()
        gv["context_lens"][:bs] = ctx.context_lens
        gv["block_tables"][:bs, : ctx.block_tables.size(1)] = ctx.block_tables
        std_graph.replay()
    torch.cuda.synchronize()
    ks = run.kv_cache[0, :, physical_block_id, block_offset].clone()

    # fresh-graph KV
    run.kv_cache.copy_(base_kv)
    with torch.inference_mode():
        gv["input_ids"][:bs] = input_ids
        gv["positions"][:bs] = positions
        gv["slot_mapping"].fill_(-1)
        gv["slot_mapping"][:bs] = ctx.slot_mapping
        gv["context_lens"].zero_()
        gv["context_lens"][:bs] = ctx.context_lens
        gv["block_tables"][:bs, : ctx.block_tables.size(1)] = ctx.block_tables
        fresh_graph.replay()
    torch.cuda.synchronize()
    kf = run.kv_cache[0, :, physical_block_id, block_offset].clone()

    for name, a, b in [("std K vs eager K", ks, ke),
                       ("fresh K vs eager K", kf, ke),
                       ("std K vs fresh K", ks, kf)]:
        md = (a - b).abs().max().item()
        rel = md / (b.abs().max().item() + 1e-9)
        print(f"{name}: max|d|= {md:.5g} rel= {rel:.2e} bit_eq={torch.equal(a, b)}", flush=True)


if __name__ == "__main__":
    main()
