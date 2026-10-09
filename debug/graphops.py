"""Find the exact op where an init-captured (pre-forward) CUDA graph diverges
from eager, by patching qkv_proj / attention / mlp to copy outputs to external
buffers that survive capture+replay.

Usage: PYTHONPATH=/home/Tokn .venv/bin/python debug/graphops.py --prompt-line 5
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
        enforce_eager=True,
        max_num_seqs=5,
        max_num_batched_tokens=1024,
    )
    run = Run(cfg, 0, [])
    hidden = run.hf_config.hidden_size
    bs = 1

    # ---- capture ONE init graph, patched to record op outputs ----
    n_layers = len(run.custom_model.model.layers)
    buf_store = {}
    def buf_for(key, ref):
        if key not in buf_store or buf_store[key].shape != tuple(ref.shape):
            buf_store[key] = torch.empty_like(ref)
        return buf_store[key]

    orig_qkv = [l.self_attn.qkv_proj.forward for l in run.custom_model.model.layers]
    orig_attn = [l.self_attn.attn.forward for l in run.custom_model.model.layers]
    orig_mlp = [l.mlp.forward for l in run.custom_model.model.layers]

    def mk_qkv(i):
        def fwd(x):
            y = orig_qkv[i](x)
            buf_for(("qkv", i), y).copy_(y)
            return y
        return fwd

    def mk_attn(i):
        def fwd(q, k, v):
            y = orig_attn[i](q, k, v)
            buf_for(("attn", i), y).copy_(y)
            return y
        return fwd

    def mk_mlp(i):
        def fwd(x):
            y = orig_mlp[i](x)
            buf_for(("mlp", i), y).copy_(y)
            return y
        return fwd

    for i in range(n_layers):
        run.custom_model.model.layers[i].self_attn.qkv_proj.forward = mk_qkv(i)
        run.custom_model.model.layers[i].self_attn.attn.forward = mk_attn(i)
        run.custom_model.model.layers[i].mlp.forward = mk_mlp(i)

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

    # restore eager forward
    for i in range(n_layers):
        run.custom_model.model.layers[i].self_attn.qkv_proj.forward = orig_qkv[i]
        run.custom_model.model.layers[i].self_attn.attn.forward = orig_attn[i]
        run.custom_model.model.layers[i].mlp.forward = orig_mlp[i]

    # ---- real prefill + decode ----
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

    # ---- eager reference (with op hooks) ----
    e_store = {}
    for i in range(n_layers):
        run.custom_model.model.layers[i].self_attn.qkv_proj.forward = mk_qkv(i)
        run.custom_model.model.layers[i].self_attn.attn.forward = mk_attn(i)
        run.custom_model.model.layers[i].mlp.forward = mk_mlp(i)
    # record references for eager buffers by capturing into e_store before graph replay overwrites
    for i in range(n_layers):
        for key in [("qkv", i), ("attn", i), ("mlp", i)]:
            e_store[key] = buf_store[key].clone()

    run.enforce_eager = True
    with torch.inference_mode():
        hs_e = run.custom_model.model(input_ids, positions)
    torch.cuda.synchronize()
    # now buf_store holds eager outputs; move them to e_store
    for i in range(n_layers):
        for key in [("qkv", i), ("attn", i), ("mlp", i)]:
            e_store[key] = buf_store[key].clone()

    # ---- replay init graph with real inputs ----
    with torch.inference_mode():
        g_input[:bs] = input_ids
        g_pos[:bs] = positions
        g_slot[:bs] = ctx.slot_mapping
        g_lens[:bs] = ctx.context_lens
        g_bt[:bs, : ctx.block_tables.size(1)] = ctx.block_tables
        g.replay()
    torch.cuda.synchronize()

    md_out = (hs_e - g_out[:bs]).abs().max().item()
    print(f"final out vs eager: max|d|= {md_out:.5g}", flush=True)
    first = None
    for i in range(n_layers):
        for name, key in [("qkv", ("qkv", i)), ("attn", ("attn", i)), ("mlp", ("mlp", i))]:
            if key not in buf_store or key not in e_store:
                continue
            ge = buf_store[key]
            ee = e_store[key]
            md = (ge - ee).abs().max().item()
            bit = torch.equal(ge, ee)
            if not bit and first is None:
                first = (i, name, md)
                print(f"FIRST DIFF layer {i} {name}: max|d|= {md:.5g}", flush=True)
            if i < 1 or (not bit and i < 3):
                print(f"layer {i} {name}: bit={bit} max|d|= {md:.5g}", flush=True)
    print(f"first diff: {first}", flush=True)

    # --- compare K/V written to cache at probe slot (layer 0) ---
    pos = len(seq.token_ids) - 1
    logical_block_id = pos // run.block_size
    block_offset = pos % run.block_size
    physical_block_id = seq.block_table[0, logical_block_id].item()
    print(f"probe slot block={physical_block_id} offset={block_offset}", flush=True)

    # eager-written K/V for layer 0 (already done above: e_store has k/v? no).
    # Re-run eager forward and read cache, then replay graph and read cache.
    run.enforce_eager = True
    with torch.inference_mode():
        run.kv_cache.copy_(base_kv)
        run.custom_model.model(input_ids, positions)
    torch.cuda.synchronize()
    ke0 = run.kv_cache[0, 0, physical_block_id, block_offset].clone()
    ve0 = run.kv_cache[1, 0, physical_block_id, block_offset].clone()

    with torch.inference_mode():
        run.kv_cache.copy_(base_kv)
        g_input[:bs] = input_ids
        g_pos[:bs] = positions
        g_slot[:bs] = ctx.slot_mapping
        g_lens[:bs] = ctx.context_lens
        g_bt[:bs, : ctx.block_tables.size(1)] = ctx.block_tables
        g.replay()
    torch.cuda.synchronize()
    kg0 = run.kv_cache[0, 0, physical_block_id, block_offset].clone()
    vg0 = run.kv_cache[1, 0, physical_block_id, block_offset].clone()

    md_k = (ke0 - kg0).abs().max().item()
    md_v = (ve0 - vg0).abs().max().item()
    print(f"layer0 K at slot: max|d|= {md_k:.5g} bit_eq={torch.equal(ke0, kg0)} "
          f"rel= {md_k/(ke0.abs().max().item()+1e-9):.2e}", flush=True)
    print(f"layer0 V at slot: max|d|= {md_v:.5g} bit_eq={torch.equal(ve0, vg0)} "
          f"rel= {md_v/(ve0.abs().max().item()+1e-9):.2e}", flush=True)


if __name__ == "__main__":
    main()
