"""Decisive: capture a SINGLE bs=1 graph at init (before any real forward) and
check if replay is bit-identical to eager at a probe decode step.

Run is built with enforce_eager=True so no std graphs are captured. A single
private-pool graph is captured right after init with realistic buffer values.

Usage: PYTHONPATH=/home/Tokn .venv/bin/python debug/onegraph.py --prompt-line 5
"""

import argparse
import sys

sys.path.insert(0, "/home/Tokn")

import torch

from utils.configs import ServerConfigs
from src.run import Run
from utils.sequence import Sequence
from utils.context import get_context, set_context as sc


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
    args = ap.parse_args()

    cfg = ServerConfigs(
        model="Qwen/Qwen3-0.6B",
        dtype="bf16",
        device="cuda",
        max_length=2048,
        max_model_len=2048,
        tensor_parallel_size=1,
        enforce_eager=True,  # do NOT capture std graphs in Run.__init__
        max_num_seqs=5,
        max_num_batched_tokens=1024,
    )
    run = Run(cfg, 0, [])
    hidden = run.hf_config.hidden_size
    bs = 1

    # Capture ONE bs=1 graph at init, private pool, realistic values.
    g_input = torch.full((bs,), 1000, dtype=torch.long, device=run.device)
    g_pos = torch.full((bs,), 1000, dtype=torch.long, device=run.device)
    g_slot = torch.zeros(bs, dtype=torch.int32, device=run.device)
    g_lens = torch.full((bs,), 1000, dtype=torch.int32, device=run.device)
    g_bt = torch.zeros(bs, 8, dtype=torch.int32, device=run.device)
    g_out = torch.zeros(bs, hidden, dtype=run.dtype, device=run.device)
    sc(is_prefill=False, slot_mapping=g_slot, context_lens=g_lens, block_tables=g_bt)
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode():
        g_out[:bs] = run.custom_model.model(g_input[:bs], g_pos[:bs])
        with torch.cuda.graph(g):
            g_out[:bs] = run.custom_model.model(g_input[:bs], g_pos[:bs])
    torch.cuda.synchronize()
    print("captured single init graph", flush=True)

    # Real prefill + decode steps.
    token_ids = tokenize(run, load_prompt(args.prompt_line))
    seq = build_seq(run, token_ids)
    prefill(run, seq)
    base_kv = run.kv_cache.clone()
    with torch.inference_mode():
        for _ in range(5):
            run.enforce_eager = True
            run.kv_cache.copy_(base_kv)
            input_ids, positions = run.prepare_decode([seq])
            logits = run.run_model(input_ids, positions, is_prefill=False)
            tok = int(torch.argmax(logits[0]).item())
            seq.append_token(tok)
            base_kv = run.kv_cache.clone()
    torch.cuda.synchronize()

    input_ids, positions = run.prepare_decode([seq])
    ctx = get_context()

    # Eager reference.
    run.enforce_eager = True
    with torch.inference_mode():
        hs_e = run.custom_model.model(input_ids, positions)
    torch.cuda.synchronize()

    # Replay init graph with the real inputs.
    with torch.inference_mode():
        g_input[:bs] = input_ids
        g_pos[:bs] = positions
        g_slot[:bs] = ctx.slot_mapping
        g_lens[:bs] = ctx.context_lens
        g_bt[:bs, : ctx.block_tables.size(1)] = ctx.block_tables
        g.replay()
    torch.cuda.synchronize()
    md = (hs_e - g_out[:bs]).abs().max().item()
    print(f"single init graph vs eager: max|d|= {md:.5g} bit_eq={torch.equal(hs_e, g_out[:bs])} "
          f"rel= {md / (hs_e.abs().max().item() + 1e-9):.2e}", flush=True)


if __name__ == "__main__":
    main()
