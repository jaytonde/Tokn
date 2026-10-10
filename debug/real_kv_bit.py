"""Logit-level bit-exactness: fast(ns=1) vs splitkv(ns=0) for REAL KV across ctx range."""
import sys
sys.path.insert(0, "/home/Tokn")
import torch as T
from src.llm import LLM
from utils.configs import ServerConfigs
from utils.context import reset_context, set_context

cfg = ServerConfigs(
    model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda",
    max_length=2048, max_model_len=2048, tensor_parallel_size=1,
    enforce_eager=False, max_num_seqs=5, max_num_batched_tokens=1024,
)
llm = LLM(model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda", max_model_len=2048, config=cfg)
RUN = llm.RUN
tok = llm.tokenizer

def run_trials(text, max_ctx, n_steps):
    ids = tok(text, add_special_tokens=False).input_ids
    while len(ids) < 300:
        ids = ids + ids
    ids = ids[:300]
    from utils.sequence import Sequence
    seq = Sequence(token_ids=ids, max_tokens=400)
    bt = llm.block_manager.allocate(token_ids=ids + [0]*400, num_cached_blocks=0)
    seq.block_table_ids = bt
    seq.block_table = llm.block_manager.make_block_table_tensor(block_table=bt, device="cuda")
    seq.num_scheduled_tokens = len(ids)
    gv = RUN.graph_vars
    with T.inference_mode():
        RUN.call("run", [seq], True)
        seq.num_cached_tokens = 300
        seq.num_scheduled_tokens = 0
        ndiff = ntest = maxdiff = 0
        mism = []
        for step in range(n_steps):
            ctx = len(seq.token_ids)
            if ctx > max_ctx:
                break
            if ctx >= 257:
                input_ids, positions = RUN.prepare_decode([seq])
                reset_context()
                logits_fast = None
                for run_fast, label in ((True, "fast"), (False, "split")):
                    set_context(is_prefill=False, slot_mapping=gv["g_slot"][:1],
                                context_lens=gv["g_ctx"][:1], block_tables=gv["block_tables"][:1],
                                use_fast_flash=run_fast)
                    gv["tbuf_tok"][:1].copy_(T.tensor([seq.last_token], dtype=T.int32, device="cuda"))
                    T.cuda.synchronize()
                    try:
                        RUN.run_model(input_ids, positions, False)
                    finally:
                        reset_context()
                    lg = gv["outputs"][:1].clone()
                    if run_fast:
                        logits_fast = lg
                    else:
                        eq = T.equal(logits_fast, lg)
                        md = (logits_fast.float()-lg.float()).abs().max().item()
                        ntest += 1
                        maxdiff = max(maxdiff, md)
                        if not eq:
                            ndiff += 1
                            mism.append((ctx, md))
            nid = int(gv["tbuf_tok"][0].item())
            seq.append_token(nid)
        return ntest, ndiff, maxdiff, mism[:6]

with T.inference_mode():
    for text, max_ctx, steps, tag in [
        ("The quick brown fox jumps over the lazy dog. " * 14, 370, 140, "A"),
        ("What is the capital of France? The answer involves history and geography. " * 10, 370, 140, "B"),
        ("def fibonacci(n): return n if n < 2 else fibonacci(n-1)+fibonacci(n-2) " * 12, 522, 320, "gate-range"),
    ]:
        ntest, ndiff, maxdiff, mism = run_trials(text, max_ctx, steps)
        print(f"[{tag}] ctx257-{max_ctx}: tested={ntest} logit_diffs={ndiff} maxdiff={maxdiff:.2e} {mism[:4]}")
