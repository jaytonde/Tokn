"""Per-layer isolation of eager-vs-graph decode divergence (all layers).

Runs the eager decoder chain layer-by-layer recording inputs/outputs. Then for
EACH layer captures a single-layer CUDA graph, replays it with the recorded
eager inputs, and compares to the eager output. Reports the first divergent
layer.

Usage:
  PYTHONPATH=/home/Tokn .venv/bin/python debug/graphlayer.py --prompt-line 5 --decode-step 10
"""

import argparse
import sys

sys.path.insert(0, "/home/Tokn")

import torch

from utils.configs import ServerConfigs
from src.run import Run
from utils.sequence import Sequence
from utils.context import get_context, reset_context


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

    # Prepare decode context (positions, slot, lens, block table)
    input_ids, positions = run.prepare_decode([seq])
    ctx = get_context()

    n_layers = len(run.custom_model.model.layers)

    # Eager chain, layer by layer, recording inputs/outputs.
    run.enforce_eager = True
    hs = run.custom_model.model.embed_tokens(input_ids)
    residuals = [None]
    hs_in = [hs.detach().clone()]
    eager_out = []
    with torch.inference_mode():
        for i, layer in enumerate(run.custom_model.model.layers):
            h_out, res = layer(positions, hs, residuals[-1])
            eager_out.append(h_out.detach().clone())
            hs_in.append(h_out.detach().clone())
            residuals.append(res.detach().clone())
            hs = h_out
    torch.cuda.synchronize()

    # Now for each layer, capture a single-layer graph and compare.
    first_diff = None
    for i, layer in enumerate(run.custom_model.model.layers):
        run.kv_cache.copy_(base_kv)
        in_hs = hs_in[i].clone().contiguous()
        in_res = residuals[i]
        if in_res is not None:
            in_res = in_res.clone().contiguous()
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            go, gr = layer(positions, in_hs, in_res)  # warmup
            with torch.cuda.graph(graph):
                go, gr = layer(positions, in_hs, in_res)
            in_hs.copy_(hs_in[i])
            if in_res is not None:
                in_res.copy_(residuals[i])
            graph.replay()
        torch.cuda.synchronize()
        ref = eager_out[i]
        md = (ref - go).abs().max().item()
        bit_eq = torch.equal(ref, go)
        rel = md / (ref.abs().max().item() + 1e-9)
        mark = "  <-- FIRST DIFF" if not bit_eq and first_diff is None else ""
        if not bit_eq and first_diff is None:
            first_diff = i
        if not bit_eq or i < 2:
            print(f"layer {i:2d} bit_eq={bit_eq} max|d|= {md:.5g} rel= {rel:.2e}{mark}", flush=True)

    print(f"first divergent layer: {first_diff}", flush=True)


if __name__ == "__main__":
    main()
