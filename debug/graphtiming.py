"""Decisive probe: is a fresh private-pool bs=1 graph bit-identical to eager
when captured (a) right after Run init (before any real forward) and
(b) after a real decode forward?

Captures TWO fresh graphs at different times and compares each to eager at a
probe decode step.

Usage: PYTHONPATH=/home/Tokn .venv/bin/python debug/graphtiming.py --prompt-line 5
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


def capture_fresh(run, ctx, bs=1):
    """Capture a fresh private-pool bs=1 graph reading the given context tensors."""
    hidden = run.hf_config.hidden_size
    g_input = torch.zeros(bs, dtype=torch.long, device=run.device)
    g_pos = torch.zeros(bs, dtype=torch.long, device=run.device)
    g_out = torch.zeros(bs, hidden, dtype=run.dtype, device=run.device)
    sc(is_prefill=False,
       slot_mapping=ctx.slot_mapping.clone(),
       context_lens=ctx.context_lens.clone(),
       block_tables=ctx.block_tables.clone())
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode():
        g_out[:bs] = run.custom_model.model(g_input[:bs], g_pos[:bs])
        with torch.cuda.graph(g):
            g_out[:bs] = run.custom_model.model(g_input[:bs], g_pos[:bs])
    return g, g_input, g_pos, g_out


def replay_fresh(run, g, g_input, g_pos, g_out, input_ids, positions, ctx):
    with torch.inference_mode():
        g_input[:bs] = input_ids
        g_pos[:bs] = positions
        g.replay()
        return g_out[:bs].detach().clone()


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
        enforce_eager=False,
        max_num_seqs=5,
        max_num_batched_tokens=1024,
    )
    run = Run(cfg, 0, [])
    hidden = run.hf_config.hidden_size
    bs = 1

    # Build a seq but do NOT run any forward yet. Prepare a decode context.
    token_ids = tokenize(run, load_prompt(args.prompt_line))
    seq = build_seq(run, token_ids)
    # We need a decode context without a real forward: fake one.
    # Instead, capture the "at-init" graph using a synthetic context.
    fake_ctx = type("C", (), {})()
    fake_ctx.slot_mapping = torch.zeros(bs, dtype=torch.int32, device=run.device)
    fake_ctx.context_lens = torch.full((bs,), 100, dtype=torch.int32, device=run.device)
    fake_ctx.block_tables = torch.zeros(bs, 8, dtype=torch.int32, device=run.device)

    # Graph captured AT INIT (before any real forward).
    g_init, gi, gp, go = capture_fresh(run, fake_ctx, bs)

    # Now do a real prefill + a few decode steps eagerly.
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

    # Probe decode state.
    input_ids, positions = run.prepare_decode([seq])
    ctx = get_context()

    # Eager reference.
    run.enforce_eager = True
    with torch.inference_mode():
        hs_e = run.custom_model.model(input_ids, positions)
    torch.cuda.synchronize()

    # Replay the at-init graph (its context is the synthetic one; the real
    # context values must be copied into ITS buffers -- but the synthetic
    # context is a clone made inside capture_fresh). Re-create graph reading
    # the REAL context now.
    run.enforce_eager = False
    # fresh graph captured AFTER real forward, reading real context
    g_late, li, lp, lo = capture_fresh(run, ctx, bs)
    out_late = replay_fresh(run, g_late, li, lp, lo, input_ids, positions, ctx)
    torch.cuda.synchronize()

    md_late = (hs_e - out_late).abs().max().item()
    print(f"post-forward fresh graph vs eager: max|d|= {md_late:.5g} bit_eq={torch.equal(hs_e, out_late)}",
          flush=True)

    # The at-init graph used the synthetic (all-zero-ish) context. For a fair
    # comparison we need its capture-time context values to match the real one.
    # That graph is not directly usable; instead, capture another fresh graph
    # at-init-style but with real values is what capture_cudagraph does after
    # my edit. We already know that path. Skip the at-init graph comparison.


if __name__ == "__main__":
    main()
