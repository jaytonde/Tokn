# Tokn Optimization Log

## Scoreboard

| # | Change | Cumulative tok/s | TTFT mean (ms) | TPOT mean (ms) | vs vLLM (tok/s ratio) |
|---|--------|-----------------|----------------|----------------|----------------------|
| 0 | baseline (eager, forced by benchmark, after KV-admission fix) | 121.53 | 264765 | 40.72 | 0.26x |
| 1 | CUDA-graph decode fix (block_table padding) + enable graphs | 441.89 | 242.09 | 7.51 | 0.97x |
| - | vLLM 0.31.0 target (this machine, 2026-10-09) | 457.86 | 17.87 | 3.18 | 1.00x |

## Experiments tried

| N | Idea | Area (graphs/kernel/scheduler/sync/memory/compile/server/config) | Result | Delta tok/s | Why |
|---|------|------|--------|-------------|-----|
| 1 | CUDA-graph decode not bit-identical to eager -> root-caused to block_table width (eager (1,nblocks) vs graph (1,8) padded); pad both to max_num_blocks | graphs | KEPT | +320.4 | flash_attn_with_kvcache yields different results for padded vs unpadded block_table at seqlen>=256; matching shapes makes graph bit-identical |

## Details

(append-only, newest at the bottom)

## [1] 2026-10-09 14:05  Fix CUDA-graph decode: block_table width mismatch (KEPT)
- Hypothesis and expected gain: graph decode was not bit-identical to eager at batch size 1 (7/37 seqs diverged, session-1 finding). If made bit-identical, benchmark can run enforce_eager=False -> ~450 tok/s vs 121 eager (3.7x).
- Profile evidence: debug/graphdiag.py showed eager-vs-graph decode logits differ every step (rel ~1e-2, 893/1024 hidden elements). debug/graphops.py: layer-0 qkv bit-identical, layer-0 attention FIRST diff (max|d|=2.4e-4). K/V written to cache bit-identical. => flash_attn_with_kvcache itself diverges.
- Root cause (via debug/flashtest2/4/5.py): flash_attn_with_kvcache produces different output for a (1,nblocks) block_table vs a (1,8) zero-padded block_table whenever seqlen spans >= 2 blocks (seqlen >= 256). Eager decode used context.block_tables (1,nblocks); the CUDA graph used gv["block_tables"] (1,8) zero-padded (captured with max_num_blocks width). Different shapes -> different kernel numerics -> graph != eager.
- Related earlier entries: none (session-1 log ended without an entry; finding carried over).
- Change: src/run.py - pad block_tables to self.max_num_blocks (with -1) in prepare_prefill and prepare_decode; fill_(-1) gv["block_tables"] in run_model before partial copy. debug/benchmark.py: enforce_eager=False. Added debug/correctness.py + diagnostics (graphdiag, graphlayer, graphfull, graphpool, onegraph, graphops, flashtest*, tritongraph, decodeprobe).
- Correctness: PASS - bs=1 eager vs graph 0/37 differ; batched eager vs graph 0/37; batched graph run1 vs run2 0/37 (deterministic).
- Benchmark before (eager): 121.53 tok/s, TTFT 264765/529783, TPOT 40.72/48.96, ITL 40.71/79.47
- Benchmark after (graphs): 441.89 tok/s, TTFT mean 242.09 p99 1708.04, TPOT 7.51/8.06, ITL 7.50/46.45
- Gap to vLLM now: throughput 0.97x, TPOT 2.36x, TTFT 13.5x
- Outcome: KEPT
- Why it worked: matching block_table shape between eager and graph makes flash_attn bit-identical; the graph decode path is now numerically equivalent to eager.
- Worth retrying? NO (fixed).

## Backlog

- (to be filled)

## [loop] 2026-10-09 13:07  Session 1 ended without a status line
- Outcome: SESSION CRASHED or cut off. Transcript: debug/opencode_runs/iter_001_20261009_124714.log
- Any experiment in progress was not finished. Check the transcript before retrying it.
