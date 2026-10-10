# Tokn Optimization Log

## Scoreboard

| # | Change | Cumulative tok/s | TTFT mean (ms) | TPOT mean (ms) | vs vLLM (tok/s ratio) |
|---|--------|-----------------|----------------|----------------|----------------------|
| 0 | baseline (eager, forced by benchmark, after KV-admission fix) | 121.53 | 264765 | 40.72 | 0.26x |
| 1 | CUDA-graph decode fix (block_table padding) + enable graphs | 441.89 | 242.09 | 7.51 | 0.97x |
| 2 | fused Triton RoPE kernel | 446.12 | 116.36 | 6.03 | 0.97x |
| 3 | threaded run_loop step + RMSNorm tail fusion + lm_head-in-graph + sync removal | 456.57 | 94.27 | 5.53 | 0.997x |
| 4 | drop GPU round-trip in add_async_request tokenization | 456.70 | 95.77 | 5.53 | 0.998x |
| 5 | fuse RMSNorm convert+residual-add, fold rsqrt into tail | 459.65 | 62.50 | 4.67 | 1.004x  MILESTONE (beats vLLM) |
| 6 | fold RMSNorm pow into convert kernel (drop separate pow launch) | 459.42 | 54.07 | 4.47 | 1.003x |
| 7 | CUDA-graph prefill (pre-captured exact-shape graphs; eager prefill was host-bound ~40ms/step) | 459.27 | 19.86 | 4.28 | 1.003x |
| - | vLLM 0.31.0 target (this machine, 2026-10-09) | 457.86 | 17.87 | 3.18 | 1.00x |
| 8 | decode flash single-pass (num_splits=1) for seqlen<=256 via dual-graph hybrid | 459.02 | 17.72 | 4.11 | 1.003x |
| 9 | merge q_norm+k_norm into one fused QKNorm call | 459.39 | 13.28 | 3.99 | 1.003x |
| 10 | cache padded block tables + CPU slot computation in prepare_decode | 459.05 | 13.00 | 3.82 | 1.003x |
| 11 | fuse rotary cos/sin cache lookup into RoPE kernel | 459.58 | 11.15 | 3.78 | 1.003x |
| 12 | QKNorm reads strided q/k from qkv output (drop 56 copy kernels) | 459.07 | 9.27 | 3.64 | 1.003x |
| 13 | merge q+k rotary into one kernel launch | 460.10 | 8.77 | 3.62 | 1.005x |
| 14 | fuse qk-norm scale into rotary kernel (scale+rotate) | 460.13 | 9.39 | 3.57 | 1.005x |
| 15 | single-tensor prepare_decode (one H2D in graph mode) | 459.51 | 8.93 | 3.52 | 1.004x |
| 16 | drop redundant graph-vars fills (exact-bs graphs) | 460.10 | 9.39 | 3.51 | 1.005x |
| 17 | fuse paged-KV store into scale+rotary kernel (drop store_kvcache) | 459.94 | 8.58 | 3.47 | 1.005x |
| 18 | fuse bs=1 qk-norm+rotary+KV-store (8->4 kernels/layer at bs=1) | 459.73 | 8.30 | 3.465 | 1.004x |
| 29 | split-step (decode on loop, prefill in executor) + eager prefill fallback on multi-seq graph miss | 460.35 | 6.86 | 3.39 | 1.005x |
| 30 | cache padded block-table stacks by seq-set (skip torch.stack on decode) | 459.9 | 6.87 | 3.36 | 1.005x |
| 31 | copy-graph: one (bs,12) H2D + captured copy-graph distributes decode input (drop 5 per-step copies) | 459.4 | 6.74 | 3.30 | 1.004x |
| 36 | fuse decode-input copies into each decode graph (one replay, rmodel 39->31us) | 459.7 | 6.69 | 3.30 | 1.004x |
| 38 | busy-poll CUDA event in split-step (drop run_in_executor wait) | 459.5 | 6.95 | 3.21 | 1.004x |
| 41 | speculative decode prep (hide state build under GPU wait) | 459.2 | 6.97 | 3.19 | 1.003x |
| 43 | skip redundant prefill store_kvcache (scale+rotate already stores) | 459.2 | 6.54 | 3.19 | 1.003x |
| 47 | in-graph greedy argmax writes tbuf_tok (drop per-step argmax dispatch + token H2D) | 459.6 | 6.78 | 3.15 | 1.004x |

## Experiments tried

| N | Idea | Area (graphs/kernel/scheduler/sync/memory/compile/server/config) | Result | Delta tok/s | Why |
|---|------|------|--------|-------------|-----|
| 1 | CUDA-graph decode not bit-identical to eager -> root-caused to block_table width (eager (1,nblocks) vs graph (1,8) padded); pad both to max_num_blocks | graphs | KEPT | +320.4 | flash_attn_with_kvcache yields different results for padded vs unpadded block_table at seqlen>=256; matching shapes makes graph bit-identical |
| 2 | Fused Triton RoPE (7 torch kernels -> 1), elementwise fp32 + single bf16 round | kernel | KEPT | +4.2 | bit-identical to torch; graph replay 6.17->4.87 ms, TPOT 7.51->6.03 |
| 3 | Threaded run_loop step (submission starvation) + RMSNorm tail fusion + lm_head in graph + GPU slot mapping | sync/compile | KEPT | +10.4 | run_loop's blocking step starved the submission coroutine; arrivals drifted ~5.6s late -> 217s. Threaded step + faster decode -> 212s |
| 4 | Drop GPU round-trip in add_async_request tokenization | sync | KEPT | +0.1 | tokenizer ids moved to CUDA and back for no reason; also put a GPU sync on the submission path. sum_submit 0.85->0.30s |
| 5 | Fuse RMSNorm convert+residual-add, fold rsqrt into tail (tl.rsqrt == torch.rsqrt) | kernel | KEPT | +2.9 | RMSNorm ~7->4 kernels/call; graph replay 4.65->3.84ms; TPOT 5.53->4.66ms. First time beats vLLM (MILESTONE) |
| 12 | Fold RMSNorm pow (x*x) into convert_add kernel; keep mean reduction in torch | kernel | KEPT | ~0 (TPOT -4%) | x*x bit-identical to torch.pow(x,2); removes 113 pow launches/step. TPOT 4.66->4.47ms, throughput neutral |
| 13 | CUDA-graph prefill: pre-captured exact-shape graphs (eager prefill host-bound ~40ms/step) | graphs | KEPT | ~0 (TTFT 54->20ms) | eager prefill = 680 kernel launches at ~66us each (~40ms) for only ~7ms GPU; exact-shape graph replay is 8ms and bit-identical. TTFT mean 54->20ms, TPOT 4.47->4.27ms |
| 14 | decode flash single-pass (num_splits=1) for seqlen<=256 via dual-graph hybrid | kernel | KEPT | ~0 (TPOT -4.2%, TTFT -19%) | splitkv path costs split+combine kernels (~700us/step at bs=5); single-pass kernel is bit-identical for seqlen<=256 and ~7us/call faster. Dual-graph hybrid keeps splitkv for seqlen>256 (reference) -> gate passes. TPOT 4.28->4.11, TTFT 19.9->17.7 |
| 15 | merge q_norm+k_norm into one fused QKNorm call (3 kernels/layer instead of 6) | kernel | KEPT | ~0 (TPOT -3.5%, TTFT -25%) | q_norm+k_norm are 56 tiny RMSNorm calls (~583us/step) dominated by fixed per-call kernel cost; merging rows into one convert+mean+scale (per-row-group weights) halves calls to 28. Per-row mean reduction unchanged -> bit-identical for bs>=2; bs=1 falls back to separate path (torch reduce config differs for 8-row k). TPOT 4.11->3.99, TTFT 17.7->13.3 |
| 16 | flash_attn_with_kvcache in-kernel KV store (drop store_kvcache kernel) | kernel | REVERTED (FAILED correctness) | - | passing k/v to flash stores cache in-kernel and skips the 84us store_kvcache kernel (~51us/step faster); but flash computes the new token's attention from registers instead of re-reading the bf16 cache -> maxdiff 1e-3, 11/37 bs=1 sequences differ. Not bit-identical, fails gate |
| 17 | cache padded block tables + CPU slot computation in prepare_decode | sync | KEPT | ~0 (TPOT -4.3%) | prepare_decode was 0.46ms/step (11% of step, serialized): per-seq torch.full+cat block padding (constant per seq, rebuilt every step) + GPU slot arithmetic. Caching padded rows + computing slots on CPU from block_table_ids (no GPU ops, no sync) -> 0.12ms. TPOT 3.99->3.82 |
| 18 | fuse rotary cos/sin cache lookup into RoPE kernel | kernel | KEPT | ~0 (TPOT -1%, TTFT -14%) | decode is per-kernel-cost bound (~650 kernels x ~5us); the cos/sin indexSelect gather was 1 kernel/layer. Fused lookup by position inside the Triton RoPE kernel -> 28 fewer kernels/step. TPOT 3.82->3.78, TTFT 13.0->11.2 |
| 19 | (investigation) benchmark tail analysis | server | - | - | debug/tail_analysis.py: last arrival is at 210.37s (Poisson delays with seed=0 sum to 210s, not 200s), last request drains in 0.92s. Elapsed 211.3s = arrival span + drain. Throughput ceiling ~460.7 tok/s; tokn at 459.6 is 99.8% of it. No throughput headroom left; TPOT/ITL are the remaining metrics |
| 20 | QKNorm reads strided q/k from qkv output directly (no reshape copy) | kernel | KEPT | ~0 (TPOT -4%, TTFT -16%) | the decode graph had 59 direct_copy elementwise kernels/step (~250us profiler). Root cause: q/k are strided views of the qkv GEMM output (dim-0 stride 4096), so QKNorm's q.reshape(-1,hd) forced a copy per q and k (2/layer = 56). Rewrote qk_convert_kernel to compute strided offsets (b*stride + h*cols) -> no copy, bit-identical. Copy kernels 59->3, decode_time 3330->3184us. TPOT 3.78->3.64, TTFT 11.2->9.3 |
| 21 | merge q+k rotary into one kernel launch | kernel | KEPT | ~0 (TPOT -0.8%, TTFT -11%) | rotary was 2 kernels/layer (q and k separately) = 56/step; merged into one launch mapping head -> q or k tensor (same cos/sin lookup, same math) -> 28 kernels/step. Bit-identical. decode_time 3184->3146us. TPOT 3.64->3.62, TTFT 9.3->8.8. Throughput 460.1 (1.005x vLLM) |
| 22 | skip graph-vars fills when batch is full (bs==max_num_seqs) | sync | REVERTED (no gain) | ~0 | bs=5 steps are rare in the benchmark (steady state ~2 concurrent per Little's law), so skipping the 3 fills on full batches is neutral. Two runs TPOT 3.64/3.59 (avg 3.615) vs baseline 3.61/3.62. No gain, reverted for simplicity. |
| 23 | fuse qk-norm scale into rotary kernel (scale+rotate in one launch) | kernel | KEPT | ~0 (TPOT -1.4%, TTFT +) | QKNorm (bs>=2) now emits fp32 xf/var via prepare(); a new scale_rotate_kernel does the RMSNorm scale AND RoPE in one launch (was qk_scale 28 kernels + rotary 28 = 56 -> 28). Bit-identical (bf16 roundings match qk_scale exactly). decode_time 3136->3089us. TPOT 3.62->3.57, throughput 460.5 (1.006x vLLM) |
| 24 | single-tensor prepare_decode (one H2D in graph mode) | sync | KEPT | ~0 (TPOT -1.3%) | prepare built 4 GPU tensors (input_ids/positions/context_lens/slot_mapping) = 4 H2D dispatches (~100us). Graph mode now builds one (bs,4) int32 tensor + strided column views (the graph-vars copies handle strides and int32->long casts); eager mode keeps the contiguous path. prepare 0.12->0.08ms. TPOT 3.57->3.52, TTFT 9.4->8.9 |
| 25 | drop redundant graph-vars fills (exact-bs graphs never read padding) | sync | KEPT | ~0 (TPOT -0.3%) | graph_bs=[1,2,3,4,5] gives an exact graph for every bs, so the graphs never read beyond row bs; the 3 fills (slot_mapping/context_lens/block_tables) only reset rows the graphs never touch. Removed them (kept only when use_bs != bs, i.e. larger-graph padding). replay-launch 0.11->0.08ms, step 3.46->3.41. TPOT 3.525->3.515, throughput 460.1 |
| 26 | fuse paged-KV store into scale+rotary kernel (drop store_kvcache) | kernel | KEPT | ~0 (TPOT -1.4%) | the decode q/k chain was prepare + scale_rotate + store_kvcache (28 kernels). scale_rotate now also writes the rotated k and v into the paged KV cache at the token slot for k-heads (same values/slot/offsets as store_kvcache) -> store_kvcache dropped from decode (kept for prefill and the bs=1 separate path via a context.kv_stored flag). Unit test: full attention chain bit-identical at seqlen 50/120/256. decode_time 3089->3039us. TPOT 3.515->3.465, throughput 460.3 |
| 27 | fuse bs=1 qk-norm+rotary+KV-store (prepare_bs1 + scale_rotate with separate var tensors) | kernel | KEPT | ~0 (bs=1 replay -5.2%, exec -0.12ms) | the bs=1 decode path still used the OLD 8-kernel/layer layout (q _rms_one 3 + k _rms_one 3 + rotary 1 + store_kvcache 1). The fused bs>=2 path (qk_convert + mean + scale_rotate) needs the merged q+k mean, which at bs=1 changes the k-reduce config -> bit-identity fails. Key insight: the CONVERT is elementwise per-row (merge-safe), only the MEANS must stay separate torch calls. prepare_bs1 does one qk_convert + separate q/k means; scale_rotate now takes var_q/var_k pointers (bs>=2 passes slices of the merged var as views). 8->4 kernels/layer, 571->459 kernels/step at bs=1. Benchmark TPOT neutral (at ceiling, within noise); bs=1 replay 2973->2818us, exec (benchmark env) 3.46->3.34ms |
| 28 | split-step: enqueue on event loop + executor only for GPU wait (ev.synchronize) | sync | REVERTED (TTFT p99 regression) | TPOT -0.09ms, TTFT p99 +70ms | running step() inline (no executor) gives TPOT 3.25 but regresses throughput 459.9->454.8 (arrival drift +2.1s, submission coroutine starved by the blocking step). A hybrid (_step_split: schedule/prepare/replay/argmax on the event loop, then await run_in_executor(ev.synchronize)) keeps the loop free and got TPOT 3.37 (vs 3.46 executor) at throughput 460.5 (ceiling) BUT regressed TTFT p99 140->217ms consistently (2 runs). Likely cause: the lazy multi-seq prefill graph capture (~190ms) now runs on the event loop (run_prefill inside _step_split) instead of in the executor thread, blocking arrivals. Would need prefill capture moved off the loop. Reverted per "no latency metric regresses" rule. |
| 29 | split-step: decode enqueues on event loop (GPU wait in executor), PREFILL steps run wholly in executor, AND multi-seq prefill cache misses fall back to EAGER prefill (~40ms) instead of synchronous graph capture (~190ms) | sync | KEPT | TPOT -2.2% (3.465->3.39), TTFT p99 -40% (148/140->99/69), TTFT mean -17% (8.3->6.9) | retry of 28 with two material changes: (a) prefills run in the executor so lazy graph capture can't block the event loop; (b) run_prefill returns EAGER on cache miss (multi-seq shapes outside the pre-captured (1,S<=128) grid) instead of a synchronous ~190ms capture - the benchmark's 3 multi-seq shapes never repeat so capture was pure stall. TPOT 3.37/3.41 (2 runs), TTFT p99 98.7/68.6 (2 runs), throughput 460.87/459.83 (ceiling), no metric regresses. debug/ttft_tail.py confirms the tail was the 3 captures (~215ms each, worst TTFTs land at capture times). |
| 30 | cache padded block-table stacks by seq-id tuple (skip torch.stack; block tables are constant during decode) + cache torch.device | sync | KEPT | TPOT -0.9% (3.39->3.36), TTFT p99 56/97 | splitprof: prep 122->87us/step (torch.stack was 27us and rebuilt the same (bs,8) every step even though padded rows never change). Cache keyed by ordered seq-id tuple (seq_ids unique -> no stale hit; order mismatch = miss). Also a fused graph-vars copy kernel was tried (1 Triton launch instead of 4 __setitem__) but it REGRESSED rmodel 110->145us (Triton dispatch overhead on a 1-block grid) - reverted. TPOT 3.38/3.34 (2 runs), TTFT p99 55.85/96.94, throughput 460.03/459.78 (ceiling). |
| 31 | copy-graph: assemble the whole decode input (token,pos,ctx_len,slot,block-table row) into ONE (bs,12) CPU tensor, single H2D into a stable tbuf, and a captured copy-graph distributes it into the model's input buffers | sync | KEPT | TPOT -1.8% (3.36->3.30), prep+rmodel 197->139us | the decode step had 5 per-step __setitem__ copies (~68us) + torch.tensor + torch.stack to populate the model's 4 graph-vars buffers. New: prepare builds (bs, 4+8) int32 rows on CPU (incl. the padded block-table row - no GPU stack at all), H2D's them into tbuf, and a captured copy-graph (5 tiny copy ops, ~3us replay) distributes tbuf into g_input/g_pos/g_slot/g_ctx/block_tables before the main graph replays. run_model drops from 110us to 38us. TPOT 3.30/3.29 (2 runs), TTFT 6.74/6.74 (p99 98/96), throughput 459.27/459.51 (ceiling). |
| 32 | (investigation) single-pass flash for ctx>256 (drop splitkv) | kernel | FAILED (correctness) | - | forced use_fast_flash=True for all contexts; bs=1 gate 4/37 differ (the 474-token prefix prompts; splitkv is required there). The benchmark's ctx 256-369 range is between the gate's tested ranges, so a threshold would need bespoke testing; splitkv penalty is only ~114us at bs=1 (measured) so not worth it. Reverted. |
| 33 | prep micro: non_blocking=True on the tbuf H2D | sync | REVERTED (ambiguous) | ~0 (TPOT 3.30 then 3.37) | splitprof prep 100.8->97.7us but the full benchmark showed TPOT 3.30/3.37 vs exp 31's 3.30/3.29 - ambiguous (could be noise), and the benefit (~3us CPU) did not justify any risk. Reverted; exp 31's plain copy_ stands. |
| 34 | graph-argmax: capture the greedy argmax as the last decode-graph op, read g_tokens instead of per-step torch.argmax dispatch | sync | REVERTED (ambiguous) | ~0 (TPOT 3.29/3.33/3.51 vs exp31 3.30/3.29) | removed the ~23us argmax dispatch from _step_split but added a ~5us argmax kernel to EVERY decode graph replay; bs=1 gate PASSED and run_loop deterministic, but 3 benchmark runs (3.29/3.33/3.51, mean 3.377) were not better than exp 31's (3.30/3.29) and one run was a clear outlier. Net ~18us CPU saving is below benchmark noise; the extra complexity (branches in run()/_step_split) was not justified. Reverted. |
| 35 | graph-argmax + pinned D2H token read + reused event (retry of 34 with the tolist also removed) | sync | REVERTED (neutral) | ~0 (TPOT 3.36/3.27, mean 3.315 vs baseline 3.32) | exp 34 was below noise; this version ALSO H2D-copies the graphed g_tokens into a pinned CPU buffer (async non_blocking) and reads it after the step event, removing the ~14us GPU-syncing .tolist() too. Gates PASSED (bs=1 0/37, run_loop deterministic 0/37). But profiling showed the D2H copy lands on the GPU critical path (+11us rtt 2828->2839) and the CPU savings are only ~20us net: argmax dispatch gone (-17us) + tolist gone (-14us) - copy_ dispatch (+5us) - D2H on critical path (+11us) = ~ -15us. Two benchmark runs 3.36/3.27 (mean 3.315) = neutral vs baseline 3.32. Reverted (ambiguous, added complexity). |
| 36 | fuse decode-input copy ops INTO each decode graph (drop the separate copy_graph replay) | graphs | KEPT (neutral, simpler) | ~0 (TPOT 3.31/3.29 vs 3.32; rmodel 39->31us) | the copy-graph (tbuf -> g_input/g_pos/g_ctx/g_slot/block_tables) was a separate graph replayed every step before the decode graph. Capturing those 5 tiny copy kernels at the FRONT of each decode graph means run_model replays ONE graph instead of two: rmodel 39->31.6us CPU, rtt +5us (2828->2833, the copies now inside the replayed graph). Net ~neutral but strictly fewer launches; TPOT 3.31/3.29, TTFT p99 98/100 (noise band), throughput 459.55/459.94. Keep rule: no metric regressed beyond noise. |
| 38 | busy-poll the CUDA event in _step_split instead of run_in_executor(None, ev.synchronize) | sync | KEPT | TPOT -2.4% (3.29->3.21), TTFT p99 -12% | the executor wait tax (~0.1-0.4ms/step) was directly on the step critical path. ev.query() busy-poll with await asyncio.sleep(0) each iteration keeps the loop free for arrivals (no starvation, unlike inline) while removing the executor round-trip. Two runs: TPOT 3.21/3.22, TTFT 6.96/6.93 (p99 86/88), throughput 459.27/459.71 (ceiling). |
| 39 | gemvx-kernel investigation (bs=1 lm_head / layer GEMM dispatch) | kernel | REVERTED (no win - misinterpreted) | 0 | profile showed a single gemvx entry at ~844us which looked like the lm_head; root-caused to 56 small o/down gemvx calls (2/layer) at their NORMAL ~15us each. lm_head is actually the fast GEMM (~390us, at the 834GB/s floor). M=2 padding of o/down does not help (o GEMM slower 14vs11.4us; down not bit-identical) and lm_head padding leaves the engine's gemvx unchanged. No win. |
| 40 | CPU micro-opts on the exposed path (pinned+numpy H2D, reusable context object) | sync | REVERTED (neutral, sub-noise) | 0 | prep 76.6->72.4us (pinned/numpy) then 78.4us (context reuse) - all within noise. The exposed CPU (~140us/step: prep 77 + rmodel 24 + post 43) delays the GPU start ~1:1 but every reduction is <15us and below the ~40us benchmark noise. |
| 41 | speculative decode prep: split tbuf into tbuf_tok/tbuf_state, build the next step's state columns during the GPU wait (pinned non_blocking H2D), next step only rewrites the token | sync | KEPT | TPOT -0.6% (3.21->3.19), TTFT p99 -20% | the exposed CPU (~140us/step) delays the GPU start ~1:1; the state columns (pos/ctx/slot/block-table) depend only on the running set + token counts, not on this step's tokens, so they can be built during the GPU wait. First attempt used a PAGEABLE H2D which is synchronous -> blocked the loop for the full GPU wait (throughput 459.5->455.4); fixed with a pinned + non_blocking copy. Also split the token into a contiguous buffer (strided column write was 41us vs 22us contiguous). token_only fast path fires 97.7% of decode steps. Two runs: TPOT 3.19/3.19, TTFT 6.93/7.00 (p99 79.9/80.0), throughput 459.12/459.20 (ceiling). |
| 42 | side-stream prefill: run prefills on a second CUDA stream concurrent with the decode replay, combined decode+prefill step | sync | REVERTED (TTFT regression) | +3.6% tok/s (ARTIFACT) | removed the decode stall during prefills (~400x) and prefilled requests sooner, but: (1) the throughput "gain" is 100% a token-count artifact (output_tokens 96789->100298 from near-tie flips; elapsed UNCHANGED 210.80s), (2) TTFT regressed hard (mean 6.95->11.22, p99 79.9->144.8): the prefill and decode are BOTH memory-bound on the same 882MB weights, so concurrency just slows each down (a full-bandwidth ~3.3ms prefill becomes ~5-6ms shared), and the decode postprocess waited for the prefill future. Reverted; experiments/42_side_prefill/. |
| 43 | skip redundant prefill store_kvcache: the prefill's fused scale+rotate kernel already writes K/V to the paged cache (kv_stored=True), but Attention's prefill branch called store_kvcache UNCONDITIONALLY (unlike decode which gates on kv_stored) - double-write of identical bytes | kernel | KEPT | TTFT mean -6% (6.95->6.5), p50 -0.1ms (5.06->4.97) | removes 28 store_kvcache kernels + the double-write bandwidth per prefill (~100us/prefill at S=100), bit-identical (same values, same slots; cache ends up identical). TTFT mean 6.33/6.75 (baseline 6.93/7.00), p50 4.96/4.98 (baseline 5.06), p99 unchanged (noise: 43.8/81.5 vs 79.9), TPOT 3.19 unchanged, throughput 459.2 unchanged (ceiling). Gates: bs=1 0/37, run_loop 0/37 vs pre-change reference (no batching change -> NO near-tie flips at all). |
| 44 | bs=1 merged q/k mean (single mean over merged (24,128) rows instead of separate (16,128)+(8,128)) - TOKEN gate instead of bit-identity | kernel | REVERTED (FAILED correctness) | - | the bs>=2 path already merges q+k into one mean; bs=1 (46% of decode steps) kept 2 separate means (28 extra reduce kernels, ~118us/step at bs=1) because the merged (24,128) mean changes the k reduce_kernel config vs standalone (8,128). Retried with the TOKEN gate (identical tokens at bs=1) instead of bit-identity. FAIL: 1/37 sequences differ (seq 8, near-tie flip at pos 40: token 13->518). Confirms near-tie tokens are dense enough in this model that even ~1e-7 relative var changes flip greedy tokens; bit-identity is effectively required. Reverted (git checkout layers/layernorm.py). |
| 45 | prefill GEMM M-scaling microbenchmark | kernel | REVERTED (no bit-identity-safe fix) | 0 | debug/gemmbench3.py: padding M 100->128 keeps SAME kernel AND same time (53.2us) but real rows NOT bit-identical (cuBLAS changes per-row accumulation with exact M); M=256 is SLOWER (74.8us, different 128x128 kernel). Larger M is both slower and non-identical -> no bit-identity-safe prefill GEMM speedup. |
| 46 | batch queued prefills to raise prefill M | scheduler | ABANDONED | - | precluded by [45]: larger M slower + non-identical; scheduler already batches multi-seq prefills when several wait; TTFT p99 tail is admission-bound burst queueing, not prefill-M. |
| 47 | in-graph greedy argmax: capture argmax as the LAST decode-graph op writing tbuf_tok, drop the per-step torch.argmax dispatch + the token H2D (GPU->CPU->GPU round-trip) | sync | KEPT | TPOT -4% (3.19->3.15), TTFT p99 -12% (~80->72) | the decode step did argmax (GPU kernel, on stream) then tolist (D2H) then postprocess, then the NEXT step H2D'd the token back into tbuf_tok (22us) + re-dispatched torch.argmax (17-29us). Capturing argmax in-graph costs no GPU time (kernel already on the stream) and removes both CPU costs; the engine reads tokens from tbuf_tok after the wait. Gates: bs=1 0/37, run_loop 0/37. 2 full runs: TPOT 3.15/3.15, TTFT 6.84/6.72 (p99 71.7/72.5), throughput 459.43/459.71 (ceiling). |
| 48 | Triton GEMV to replace the o/down gemvx (23% of bs=1 decode) | kernel | REVERTED (no win) | 0 | debug/gemv_triton.py + gemv_tune.py: cuBLAS's gemvx order IS reproducible - a Triton sequential/chunk-tree GEMV is bit-identical (maxd=0) for EVERY BLOCK_N/BLOCK_K config (32..256 chunks), so the 2048-dot-product has slack. BUT every Triton config is ~2x SLOWER (30us vs 15.8us F.linear): Triton's (BLOCK_N,BLOCK_K) tile load + dispatch can't beat cuBLAS's 235GB/s GEMV on this shape, and more programs (small BLOCK_N) hit poor coalescing (matrix-vector). No config beat F.linear. |
| 49 | extend use_fast_flash (num_splits=1) above ctx=256 to cut the splitkv penalty at ctx 257-369 | kernel | REVERTED (unsafe) | 0 | debug/splitkv_threshold.py: with RANDOM cache data, ns=1 vs ns=0 diverges (max|d|~1e-3, ndiff 225-500) at EVERY seqlen 250-400 - the bit-identity of single-pass vs splitkv is DATA-DEPENDENT (exp 14's <=256 bit_eq used real structured KV). So there is NO safe threshold above 256: benchmark contexts 257-369 would flip tokens undetected by the gate (whose prompts are short (48 tok) or long (474)). Confirms exp 32's conclusion. The ITL p99 gap (3.48 vs vLLM 3.39) from splitkv steps is blocked. |
| 50 | submission-path tokenization cost on the event loop | server | NOT A LEVER | 0 | debug/tokenizer_cost.py: apply_chat_template+tokenizer ~0.3ms/request mean, 0.12s total event-loop blocking over 400 requests (0.06% of 210s). Negligible. |
| 51 | reuse a persistent CUDA event for the decode GPU-wait poll (drop per-step torch.cuda.Event() create, ~12us/step) | sync | REVERTED (neutral) | 0 | two 400-prompt runs TPOT 3.16/3.15, ITL p99 3.69/3.50 vs exp 47's 3.15/3.15, 3.48/3.48 - neutral (one run's 3.69 ITL p99 is burst noise). Saving below the ~40us noise. Reverted (git checkout src/engine.py) to keep the clean exp-47 state. |
| 52 | force explicit per-bs num_splits on the splitkv decode graphs (ctx>256) to shave the flash splitkv penalty on the slowest (high-bs) steps | kernel | ABANDONED (sub-noise, fragile) | 0 | debug/num_splits_test.py + num_splits_all.py: explicit ns values that match the ns=0 heuristic's split count ARE bit-identical across ctx 257-369 (bs=1/2/3: ns=8, bs=4/5: ns=3; bs=1 ns=8 also bit-identical to ctx 522 = the gate's prefix-prompt range) and ~0.6-2.2us/call faster -> 17-61us/step at splitkv steps. But that is only ~1-2us of the ~13us/call single-pass-vs-splitkv gap (ns=1 is the only fast option and is NOT bit-identical), so the average TPOT gain is ~5-10us (sub-noise) and the ITL p99 gain ~0.04ms. The per-bs ns mapping is fragile and the bit-identity is data-dependent (exp 14/49 lesson) -> risk of silent token flips. Abandoned. |

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

## [2] 2026-10-09 14:40  Fused Triton RoPE kernel (KEPT)
- Hypothesis and expected gain: torch apply_rotary_emb launches ~7 kernels per call (float, chunk, mul/sub, mul/add, cat, to) x 56 calls/step (q+k per layer) = ~390 kernel launches/step, a large share of the ~4ms/step elementwise "kernel soup" seen in eagerprof. A single fused elementwise Triton kernel (fp32, one bf16 round at the end) should be bit-identical and cut ~1.3ms/step.
- Profile evidence: eagerprof showed ~224 elementwise + 113 reduce + 200 vectorized elementwise kernels/step (~4ms). Rotary was a major contributor.
- Related earlier entries: [1]
- Change: layers/rotary_embedding.py - fused rotary_emb_kernel replacing apply_rotary_emb. Bit-identical verified on decode (bs=1,5) and prefill (483 tokens) q/k shapes vs the torch reference.
- Correctness: PASS - bs=1 vs pre-change reference 0/37; graph vs eager bs=1 0/37; batched deterministic 0/37. Batched eager-vs-graph shows 9/37 near-tie flips (bf16 batch-invariance from differing engine timing between the two runs; recorded, gate is bs-1 identity + batched determinism).
- Benchmark before: 441.89 tok/s, TTFT 242.09/1708, TPOT 7.51/8.06, ITL 7.50/46.45
- Benchmark after: 446.12 tok/s, TTFT 116.36/1078.95, TPOT 6.03/6.60, ITL 6.03/41.56
- Gap to vLLM now: throughput 0.97x, TPOT 1.90x, TTFT 6.5x
- Outcome: KEPT
- Why it worked: pure elementwise fusion with matching fp32 op order and a single bf16 rounding is bit-identical to torch; no reduction-order ambiguity.
- Worth retrying? NO.

## [3] 2026-10-09 14:20  Fused Triton RMSNorm (REVERTED - fails gate)
- Hypothesis and expected gain: 56 RMSNorm calls/step x ~6 kernels each. A fused Triton RMSNorm+residual would cut ~1ms/step.
- Change: layers/layernorm.py fused rms_norm_kernel.
- Correctness: FAIL - bit-identical on unit test with unit weights but 12/37 seqs differ at bs=1 vs pre-change reference with real weights. Root cause: tl.sum(x*x) reduction order differs from torch's mean_kernel (rel ~1e-7), which flips bf16 tokens after cascading through 28 layers. Matching torch's exact reduction tree is impractical in Triton.
- Outcome: REVERTED (correctness gate). All debug state in debug/rmsdump.py.
- Worth retrying? YES IF torch.compile-level reduction matching is used, or if the gate is relaxed to allclose.

## [4] 2026-10-09 16:00  Threaded run_loop step fixes submission starvation (KEPT)
- Hypothesis and expected gain: the benchmark was stuck at 217s (446 tok/s) while a tight-loop driver hit 210s with identical step times. The run_loop's synchronous 5.5ms step() blocks the event loop, starving the benchmark's submission coroutine; arrivals (asyncio.sleep timers) drift late and the drift accumulates over the run. Fix -> elapsed ~211s, ~458 tok/s (beats vLLM).
- Profile evidence: debug/benchsim.py measured arrival delay growing to 5.59s for the last request (mean 2.78s) over 400 requests; step time 5.86ms and idle 14s accounted for all of the elapsed, yet the last-completed request finished 6.9s after the last scheduled arrival because arrivals happened late. Tight-loop driver (debug/arrivalprof.py) got 210s with the same step times.
- Related earlier entries: none
- Change: src/engine.py run_loop runs step() in asyncio's default executor thread. Also fixed a lost-wakeup race (re-check is_finished() after has_work.clear()). src/scheduler.py refills freed slots by prefilling immediately when running < max. 
- Correctness: PASS - bs-1 eager vs graph 0/37; batched deterministic 0/37.
- Benchmark before: 447.17 tok/s (446.12 after rotary), TTFT 91.83, TPOT 5.47
- Benchmark after: 456.57 tok/s, TTFT 94.27/990.00, TPOT 5.53/5.99, ITL 5.53/6.43
- Gap to vLLM now: throughput 0.997x, TPOT 1.74x, TTFT 5.3x
- Outcome: KEPT
- Why it worked: the event loop stays free to service the submission coroutine, so arrivals happen on schedule instead of drifting ~5.6s late. This was the dominant term in end-to-end latency.
- Worth retrying? NO.

## [5] 2026-10-09 15:40  RMSNorm tail fusion (Triton) (KEPT)
- Hypothesis and expected gain: fuse the post-stat RMSNorm elementwise tail (mul rsqrt, to bf16, mul weight = ~3-4 kernels x 56 calls) into one Triton kernel. Full fusion failed (reduction order), but keeping the var/inv reduction in torch and fusing only the tail is bit-identical.
- Profile evidence: eagerprof showed ~4ms/step of small elementwise/reduce kernels; RMSNorm contributes heavily.
- Change: layers/layernorm.py - rms_scale_kernel; torch computes var+inv (bit-identical to old), Triton does round_bf16(x*inv)*w.
- Correctness: PASS - 0/37 vs pre-change reference at bs=1; graph bs=1 0/37. Key fix: must round (x*inv) to bf16 BEFORE the weight multiply (two roundings) and use explicit tl.bfloat16 cast (x_ptr is fp32 there).
- Benchmark: graph replay 5.24->4.65 ms; part of the 447->456 package.
- Outcome: KEPT.

## [6] 2026-10-09 15:25  lm_head inside CUDA graph (KEPT, neutral on GPU)
- Capture self.custom_model (incl. lm_head) so the vocab GEMM runs in the graph. GPU-neutral (0.376ms GEMM costs the same in/out) but removes the eager compute_logits Python call.
- Correctness: PASS 0/37 bs-1.

## [7] 2026-10-09 15:10  GPU slot mapping (remove .item()/.tolist() syncs) (KEPT, neutral)
- prepare_prefill/prepare_decode now compute slot_mapping with GPU integer ops instead of per-token .item(). Benchmark-neutral (prep wall time went up slightly but GPU overlaps). Required padding per-seq block tables to max_num_blocks before torch.stack.
- Correctness: PASS 0/37.

## [8] 2026-10-09 14:50  Scheduler: refill freed slots immediately (KEPT, no measured gain)
- should_decode_first now allows a prefill when running < max_num_seqs and requests are waiting, instead of waiting out prefill_interval=8. Benchmarked neutral on its own; kept as it improved max batch 4->5 in the tight-loop driver.

## [9] 2026-10-09 16:40  Fuse RMSNorm convert+residual-add and fold rsqrt into tail (KEPT, MILESTONE)
- Hypothesis and expected gain: RMSNorm was still ~7 kernels/call (x.float, residual.float, add, res.to(bf16), pow, mean, rsqrt, tail). Folding the rsqrt into the Triton tail and fusing the convert+add cuts it to ~4 kernels -> ~0.8ms/step.
- Profile evidence: eagerprof showed reduce_kernel + unrolled_elementwise + 3x vectorized_elementwise at 565 calls/step (the RMSNorm var computation) ~1.1ms/step.
- Key discovery: tl.rsqrt is bit-identical to torch.rsqrt (unlike 1/tl.sqrt which differs by 1 ULP), so inv = tl.rsqrt(var+eps) can be computed inside the tail kernel.
- Change: layers/layernorm.py - convert_add_kernel (x.to(f32) [+ r.to(f32)], res=round_bf16) + rms_scale_kernel takes var and computes inv via tl.rsqrt. var still computed in torch (bit-identical reduction).
- Correctness: PASS - 0/37 vs pre-change reference at bs=1; graph bs=1 0/37; batched deterministic 0/37.
- Benchmark before: 456.70 tok/s, TTFT 95.77/1001, TPOT 5.53/6.00
- Benchmark after: 459.59 tok/s (459.19 confirm), TTFT 61.06/611.76, TPOT 4.66/5.19, ITL 4.66/5.35
- Gap to vLLM now: throughput 1.004x (MILESTONE - first time beating vLLM), TPOT 1.47x, TTFT 3.4x
- Outcome: KEPT
- Why it worked: folding the (bit-identical) rsqrt into the tail and fusing the elementwise convert+add removed ~3 kernels x 113 RMSNorm calls/step.
- Worth retrying? NO.

## [10] 2026-10-09 17:10  Pre-allocated decode buffers (REVERTED - slower)
- Hypothesis: avoid per-step torch.tensor allocation in prepare_decode.
- Change: pre-allocated _dec_* buffers in Run; per-element GPU writes in prepare_decode.
- Result: prepare wall time INCREASED (stepprof 0.79ms -> 1.26ms) because per-element scalar GPU writes (self._dec_input_ids[i] = int) are more expensive than one batched torch.tensor H2D for bs<=5.
- Outcome: REVERTED. The batched tensor approach stays.

## [11] 2026-10-09 17:15  Keep slot arithmetic in int32 (KEPT, neutral)
- Avoid int64 promotion+cast in slot_mapping computation. Bit-identical (0/37). Benchmark-neutral (459.33 tok/s).

## [12] 2026-10-10 03:40  Fold RMSNorm pow into convert_add kernel (KEPT, TPOT -4%)
- Hypothesis and expected gain: decode-step profile showed the RMSNorm var computation as reduce_kernel (mean) + a separate pow_tensor_scalar kernel = ~15% of decode at bs=5. Folding the square into convert_add (which already computes xf) removes one kernel launch per RMSNorm call (113 calls/step) -> ~5% decode, better TPOT.
- Profile evidence: debug/decodeprof.py at bs=5: reduce_kernel 9.9% + pow_tensor_scalar 5.0% of graph-replay device time; convert_add 5.5%, rms_scale 5.6%. ~675 kernel launches/step total, RMSNorm ~452 of them.
- Key prerequisite (debug/microtest.py): torch.pow(x,2) IS bit-identical to x*x for fp32 on this GPU (all shapes incl. negatives). So writing x2 = xf*xf in convert_add and doing torch's x2.mean(-1) gives an identical var to the old xf.pow(2).mean(-1) (same reduce order, same inputs).
- Change: layers/layernorm.py - convert_add_kernel gains an x2_ptr output (x*x); forward allocates x2 and computes var = x2.mean(-1). xf/xf2/res stores all still bf16/fp32 as before.
- Correctness: PASS - bs=1 eager 0/37 vs pre-change bs=1 reference (generated by swapping in git HEAD layernorm.py); bs=1 graph 0/37 vs new eager; batched graph run1 vs run2 0/37. (Note: an earlier FAIL was a mistaken compare of new bs=1 vs the OLD BATCHED reference -- batched runs flip near-tie tokens; the correct gate is bs=1 vs bs=1.)
- Benchmark before (exp 5 final): 459.65 tok/s, TTFT 62.50/611.76, TPOT 4.66/5.19, ITL 4.66/5.35
- Benchmark after (2 runs): 459.42 then 459.14 tok/s, TTFT mean 54.07 p99 526.6, TPOT 4.47/4.97, ITL 4.47/4.80; 2nd run TPOT 4.48/4.97, TTFT 54.58/530.2
- Gap to vLLM now: throughput 1.003x, TPOT 1.41x, TTFT 3.0x
- Outcome: KEPT (throughput neutral within noise, TPOT consistently -4%, no latency regression)
- Why it worked: x*x is bit-identical to torch.pow(x,2), and the mean reduction is unchanged, so the numerics are untouched; only a redundant kernel launch is removed.
- Worth retrying? NO (done).

## [13] 2026-10-10 04:40  CUDA-graph prefill (pre-captured exact shapes) (KEPT, TTFT 54->20ms)
- Hypothesis and expected gain: eager prefill is host-bound. debug/prefillprof.py: a prefill forward of just 26-134 tokens costs 41ms GPU wall (CUDA event) yet the kernels sum to only ~7ms -> ~34ms of launch/dispatch gaps. Prefill steps happen 396x in the benchmark, so each 41ms step also stalls the run_loop pipeline. Exact-shape CUDA-graph replay measured 8ms and bit-identical -> TTFT should drop from ~54ms to ~15ms and free ~13s of pipeline time.
- Profile evidence: prefillhost.py: 680 kernel launches/step at ~66us host cost each = 45ms enqueue (mean 1.62ms/layer). prefillgraph.py: eager 42.3ms vs graph replay 7.98ms, bit_eq=True. capturetiming.py: capture body ~190ms (inherent), warmup 44ms after JIT. prefillshapes.py: benchmark has only 54 distinct (num_seqs, total_tokens) prefill shapes, almost all bs=1, tokens 13-113.
- Related earlier entries: [1] (decode graphs), none for prefill.
- Change: src/run.py - run_model routes prefill through run_prefill(); new _capture_prefill_graph (captures one graph per exact shape into a shared pool, slot_mapping forced -1 so capture never writes KV) and pre_capture_prefill_graphs (pre-captures (1, S) for S=1..128 at init, invisible to benchmark elapsed which is measured after engine init). Lazy capture remains as fallback for shapes outside the grid (multi-seq, longer prompts), capped at 256 graphs. Init went 30->37.5s (pre-capture efficient, ~55ms/shape).
- Correctness: PASS - bs=1 graph 0/37 vs eager reference; batched graph run1 vs run2 0/37. (Padding to buckets was rejected: flash_attn_varlen is NOT bit-identical when padding crosses a 256-token boundary, debug/prefillpadtest.py; exact shapes only.)
- Benchmark before (exp 12): 459.14-459.42 tok/s, TTFT 54.1/31.1/526, TPOT 4.47/4.97, ITL 4.47/4.80
- Benchmark after (2 runs): 458.57 then 459.27 tok/s, TTFT mean 19.9-20.5 p50 5.1-5.4 p99 370-404, TPOT 4.25-4.28/4.6-5.1, ITL 4.25-4.28/4.8-5.0
- Gap to vLLM now: throughput 1.003x, TPOT 1.34x (was 1.47x), TTFT 1.12x (was 3.4x)  <- TTFT near-parity, p50 (5.1ms) far better than vLLM mean
- Outcome: KEPT
- Why it worked: prefill was dominated by host dispatch (680 launches * ~66us) not GPU work; pre-recording the ops into a CUDA graph removes all launch overhead while keeping the exact same kernels/shapes, hence bit-identical output. Throughput stayed flat (~459) because the benchmark tail is decode-bound, not prefill-bound; the win is on latency (TTFT mean 54->20ms, p50 31->5ms).
- Worth retrying? NO (done). Follow-up: TTFT p99 tail (370ms) is scheduler queueing when all 5 decode slots are full; separate scheduler experiment.

## [14] 2026-10-10 06:40  Decode flash single-pass (num_splits=1) for short contexts (KEPT)
- Hypothesis and expected gain: the backlog flagged that splitkv attention adds a combine kernel (~0.18ms/step). Profile at bs=5 confirmed flash_fwd_splitkv_kernel 465us + splitkv_combine 252us = 717us/step (14%). Forcing num_splits=1 (single-pass kernel) should be ~7us/call faster -> ~200-400us/step, bit-identical for seqlen<=256.
- Profile evidence: debug/decodeprof.py at bs=5: splitkv 465us + combine 252us of ~5060us total. debug/splitkvtest.py: ns=0 vs ns=1 bit_eq=True (maxdiff=0) for seqlen 50/120/200/256, differs (1.2e-4) for 300/400. debug/decode_time.py (CUDA events, in-graph): ns=1 3495us vs ns=0 3733us at bs=5 across ctx 0-320 (~237us/step saved, holds at all context lengths).
- Related earlier entries: none (backlog item).
- Change: layers/attention.py (num_splits from context flag), utils/context.py (use_fast_flash field), src/run.py (capture TWO decode graphs per bs: graphs_fast[num_splits=1], graphs[num_splits=0]; run_model selects by context.use_fast_flash; prepare_decode sets the flag from CPU max_ctx<=256 with no GPU sync). commit 85d57c4.
- Correctness: PASS - bs=1 graph vs pre-change reference 0/37 (hybrid: short ctx via fast graph bit-identical, long ctx via splitkv graph unchanged). Batched graph run1 vs run2 0/37. (A naive full num_splits=1 FAILED: 4/37 long prefix-cache prompts >256 diverged; that's why the hybrid exists.)
- Benchmark before (exp 13): 458.57-459.27 tok/s, TTFT 19.9/370-404, TPOT 4.25-4.28/4.6-5.1, ITL 4.25-4.28
- Benchmark after (2 runs): 459.33 then 458.70 tok/s, TTFT 16.14/315.96 then 19.29/376.66, TPOT 4.10/4.44 then 4.12/4.52, ITL 4.10/4.57 then 4.13/4.77
- Gap to vLLM now: throughput 1.003x, TPOT 1.29x (was 1.34x), TTFT 0.99x (was 1.12x)  <- TTFT now at/below vLLM mean
- Outcome: KEPT
- Why it worked: for seqlen<=256 the splitkv heuristic adds a redundant split+combine pair; the single-pass kernel is numerically identical there (flash-attn only splits at longer seqlen), so a graph per variant lets us take the fast path exactly where it's bit-identical and keep the reference path elsewhere.
- Worth retrying? NO (done).

## [15] 2026-10-10 07:10  Merge q_norm+k_norm into one fused QKNorm call (KEPT)
- Hypothesis and expected gain: RMSNorm is ~1.2ms/step at bs=5 (34% of decode) per in-graph measurement (10.8us/hidden call x57 + 10.4us/head_dim call x56). The 56 q_norm+k_norm calls cost ~583us/step and are dominated by FIXED per-call kernel cost (convert+mean+scale ~10us regardless of 128 vs 1024 cols). Merging q+k into one row-block per layer (one convert, one torch mean, one scale with per-row-group weights) halves calls 56->28 -> expected ~250-280us/step saved.
- Profile evidence: debug/op_micro.py + in-graph captures: RMSNorm hidden in-graph 10.8us/call, head_dim 10.4us/call; q_norm+k_norm 56 calls = 583us/step. decode_time.py: 3526us/step (hybrid) vs 3381us (QKNorm) = -145us/step.
- Related earlier entries: [12] (RMSNorm pow fold), [5] (RMSNorm tail fusion), [3] (full fusion FAILED - reduction order).
- Change: layers/layernorm.py - qk_convert_kernel, qk_scale_kernel, QKNorm module (merged convert/mean/scale with w_q for q rows, w_k for k rows; bs=1 nk<16 falls back to separate _rms_one path). src/qwen3.py - replace q_norm/k_norm with qk_norm, keeping q_norm/k_norm as shared-storage submodule aliases so utils/loader.py finds the checkpoint's q_norm.weight/k_norm.weight (real non-identity values). commit fb33d9c.
- Correctness: PASS - bs=1 graph 0/37 vs pre-change reference (fallback path); batched run1 vs run2 0/37. Key subtlety found: merging the mean over (24,128) vs standalone (8,128) is NOT bit-identical at bs=1 (torch picks a different reduce_kernel config for 8 rows) - hence the bs=1 fallback. bs=2-5 merged mean is bit-identical (verified per-bs).
- Benchmark before (exp 14): 458.70-459.33 tok/s, TTFT 16.14/19.29, TPOT 4.10/4.12, ITL 4.10/4.13
- Benchmark after (2 runs): 459.54 then 459.24 tok/s, TTFT 13.68/12.88 (p99 272/258), TPOT 3.97/4.00 (p99 4.27/4.31), ITL 3.97/4.00
- Gap to vLLM now: throughput 1.003x, TPOT 1.25x (was 1.29x), TTFT 0.74x (was 0.99x)  <- TTFT now well below vLLM
- Outcome: KEPT
- Why it worked: per-row mean reduction is independent of row count (same reduce kernel for the rows), so merging q+k rows into one tensor keeps per-row numerics identical for bs>=2; the fallback covers the one shape (bs=1, 8 k-rows) where torch's reduce config changes. Halved the fixed kernel cost of the two smallest norm calls.
- Worth retrying? NO (done).

## [16] 2026-10-10 07:20  Flash in-kernel KV store (drop store_kvcache) (REVERTED - FAILED correctness)
- Hypothesis and expected gain: the decode path calls a Triton store_kvcache kernel (84us/step profiler, ~51us eager) then flash_attn_with_kvcache reads the cache. flash_attn_with_kvcache accepts k/v and stores+attends in ONE kernel -> drop store_kvcache, ~50-84us/step saved.
- Profile evidence: debug/inkernel_kv.py: A(store+flash)=71.7us vs B(in-kernel)=20.3us at bs=5 (~51us saved); kc/vc caches end up bit-identical, BUT attention output differs maxdiff 1e-3..3.5e-3 (flash uses the k/v from registers for the new token instead of re-reading the bf16 cache).
- Related earlier entries: none.
- Change: layers/attention.py - decode passes k=k.unsqueeze(1), v=v.unsqueeze(1) to flash_attn_with_kvcache; store_kvcache kept only for prefill. REVERTED.
- Correctness: FAIL - 11/37 bs=1 sequences differ (even short contexts; the register-vs-cache numerics flip near-tie tokens).
- Benchmark: NOT RUN (failed gate).
- Outcome: REVERTED (FAILED - correctness)
- Why it failed: the in-kernel store computes the freshly-written token's attention from the raw k/v argument rather than the stored bf16 cache value, which is a different numeric path (maxdiff up to 3.5e-3). Not bit-identical, so near-tie greedy tokens flip at bs=1.
- Worth retrying? NO - the numerics are inherently different (register vs cache read).

## [18] 2026-10-10 07:50  Fuse rotary cos/sin cache lookup into RoPE kernel (KEPT)
- Hypothesis and expected gain: decode graph is per-kernel-cost bound (~650 kernels x ~5us each, NOT compute bound). The rotary path launched a cos/sin indexSelect gather (1 kernel) per layer -> 28 kernels/step. Fusing the lookup by position inside the Triton RoPE kernel removes them -> ~140us/step.
- Profile evidence: single-layer in-graph cost 137us x 28 = 3843us vs full graph 3690us -> layers are the whole replay; per-layer ~137us from ~20-30 kernels. decode_time: 3381us (QKNorm) -> 3330us (fused) = -51us/step (gather was cheaper than estimated; fused kernel reads cache directly).
- Related earlier entries: [2] (fused RoPE), [17].
- Change: layers/rotary_embedding.py - rotary_emb_fused_kernel takes cos_sin_cache + positions and reads cache[pos, h]/cache[pos, h+half_dim] in-kernel; RotaryEmbedding.forward no longer gathers/chunks. commit 2427151.
- Correctness: PASS - fused vs old (gather) bit-identical at bs=1 and bs=5; bs=1 graph 0/37 vs reference; batched run1 vs run2 0/37.
- Benchmark before (exp 17): 458.86 tok/s, TTFT 13.00/259.46, TPOT 3.82/4.09, ITL 3.82/4.22
- Benchmark after (2 runs): 459.54/459.62 tok/s, TTFT 9.90/12.40 (p99 177/236), TPOT 3.77/3.78 (p99 4.04/4.05), ITL 3.77/3.78
- Gap to vLLM now: throughput 1.004x, TPOT 1.19x (was 1.20x), TTFT 0.62x
- Outcome: KEPT
- Why it worked: the gather read the same bf16 cache values the kernel now reads directly; the fused kernel is bit-identical (same values, same fp32 math). Removing 28 kernels from a per-kernel-cost-bound graph is a direct win.
- Worth retrying? NO (done).

## [20] 2026-10-10 08:40  QKNorm reads strided q/k from qkv output (drop 56 copy kernels) (KEPT)
- Hypothesis and expected gain: kernel-count profile showed 59 direct_copy elementwise kernels/step (~250us profiler, ~6% of decode). If located and removed, ~150-200us/step saved. The decode graph is per-kernel-cost bound, so every kernel removed is ~4-5us.
- Profile evidence: copy kernels 59/step at 4.3us each = 254us (profiler). Isolating each layer component in a graph: attention/GEMMs/norms/rotary all had 0 copies; qk_norm had 2 copies/layer. Root cause: q/k are strided views of the qkv GEMM output (qkv is (bs,4096) contiguous; q=qkv[:,:2048].view(-1,16,128) has dim-0 stride 4096, not 2048), so QKNorm's q.reshape(-1,hd) made a full copy per tensor. 2 copies/layer x 28 = 56.
- Related earlier entries: [15] (QKNorm merge - which introduced the reshape).
- Change: layers/layernorm.py - qk_convert_kernel now takes q_heads/k_heads/q_stride/k_stride and computes offsets b*stride + h*cols; QKNorm.forward passes the strided q/k directly (no reshape). commit 5bad8e0.
- Correctness: PASS - strided-input QKNorm bit-identical to old contiguous path (bs=1,5, random weights); bs=1 graph 0/37 vs reference; batched run1 vs run2 0/37. Copy kernels 59->3 (remaining 3 are the graph-vars fills).
- Benchmark before (exp 18): 459.54/459.62 tok/s, TTFT 9.90/12.40, TPOT 3.77/3.78, ITL 3.77/3.78
- Benchmark after (2 runs): 459.28/458.85 tok/s, TTFT 9.28/9.25 (p99 194), TPOT 3.64/3.63 (p99 3.86/3.87), ITL 3.64/3.64
- Gap to vLLM now: throughput 1.003x, TPOT 1.14x (was 1.19x), TTFT 0.52x
- Outcome: KEPT
- Why it worked: qkv.split produces strided views; the old reshape paid a copy every step to make them contiguous for the norm, even though the norm kernel only needs per-row contiguous reads. Computing the strided offsets in Triton reads the same values with no copy.
- Worth retrying? NO (done).

## [21] 2026-10-10 09:00  Merge q+k rotary into one kernel launch (KEPT)
- Hypothesis and expected gain: the decode graph is per-kernel-cost bound (~587 kernels x ~3-5us). Rotary ran 2 kernels/layer (q and k) = 56/step. A single merged launch (program h in [0, q_heads+k_heads], head<h_q -> q else k) halves that to 28 -> ~28 x 3us = ~84us/step.
- Profile evidence: kernel categories: rotary 56 kernels, 169us profiler (3us each). decode_time 3184us (exp 20) -> 3146us = -38us/step (kernel was cheaper than the rough estimate).
- Related earlier entries: [2], [18] (fused rotary), [15]/[20] (kernel-count reductions).
- Change: layers/rotary_embedding.py - rotary_emb_merged_kernel handles q and k in one launch (same cos/sin cache lookup by position, branches on head index); RotaryEmbedding.forward calls it. commit 6417ac1.
- Correctness: PASS - merged vs old (gather) and vs exp-18 fused, bit-identical at bs=1 and bs=5; bs=1 graph 0/37 vs reference; batched run1 vs run2 0/37.
- Benchmark before (exp 20): 459.28/458.85 tok/s, TTFT 9.28/9.25, TPOT 3.64/3.63, ITL 3.64/3.64
- Benchmark after (2 runs): 459.94/460.26 tok/s, TTFT 7.95/9.59 (p99 127/208), TPOT 3.61/3.62 (p99 3.88/3.87), ITL 3.61/3.62
- Gap to vLLM now: throughput 1.005x, TPOT 1.14x, TTFT 0.49x
- Outcome: KEPT
- Why it worked: q and k rotary share the same positions -> identical cos/sin lookup; computing both in one launch removes a duplicate lookup+launch with no numeric change.
- Worth retrying? NO (done).

## [23] 2026-10-10 09:30  Fuse qk-norm scale into rotary kernel (KEPT)
- Hypothesis and expected gain: the q/k chain was qk_convert (28) + qk_mean (28) + qk_scale (28) + rotary (28) = 112 kernels/step. The scale output feeds directly into rotary (same data), so fusing them -> 28 fewer kernels -> ~80us/step.
- Profile evidence: kernel categories qk_norm 56 kernels/174us + rotary 28/169us. decode_time 3136us (exp 22 baseline) -> 3089us = -47us/step.
- Related earlier entries: [15] (QKNorm), [20] (strided read), [21] (merged rotary), [18].
- Change: layers/layernorm.py - QKNorm.prepare() returns fp32 xf/var (convert+mean only, no scale/y). layers/rotary_embedding.py - scale_rotate_kernel does RMSNorm scale (round_bf16(x*inv)*w, matching qk_scale exactly) then RoPE in one launch; RotaryEmbedding.scale_rotate(). src/qwen3.py - Qwen3Attention.forward uses prepare+scale_rotate for bs>=2, keeps separate norm+rotary for bs=1. commit 015996f.
- Correctness: PASS - fused vs (scale then rotary) bit-identical at bs=2,5 with random weights; bs=1 gate 0/37 (uses separate path); batched run1 vs run2 0/37. First version had a k-output offset bug (wrote out of bounds using xf row index for the outk tensor) - caught by the unit test, fixed with a separate out_base.
- Benchmark before (exp 21): 459.94/460.26 tok/s, TTFT 7.95/9.59, TPOT 3.61/3.62, ITL 3.61/3.62
- Benchmark after (2 runs): 460.48/459.78 tok/s, TTFT 8.73/10.04 (p99 169/196), TPOT 3.55/3.59 (p99 3.75/3.80), ITL 3.55/3.59
- Gap to vLLM now: throughput 1.006x, TPOT 1.12x (was 1.14x), TTFT 0.53x
- Outcome: KEPT
- Why it worked: scale and rotary are sequential on the same values with no other consumer between them; doing both in one kernel removes a launch and a memory round-trip (scaled q/k no longer written to global then re-read). The bf16 rounding must happen at the same points as the old two kernels (round after scale, fp32 rotary math, round at store).
- Worth retrying? NO (done).

## [24] 2026-10-10 09:45  Single-tensor prepare_decode in graph mode (KEPT)
- Hypothesis and expected gain: the per-step exposed CPU (prepare 0.14ms + replay-launch 0.11ms) is now the biggest remaining lever; the GPU replay (~3.09ms) is already below vLLM's TPOT (3.18). prepare built 4 GPU tensors via torch.tensor (4 H2D dispatches ~100us). Building one (bs,4) int32 tensor and slicing columns saves 3 dispatches.
- Profile evidence: cProfile: prepare_decode 4 torch.tensor calls = 0.10ms of the 0.12ms. After change: 0.083ms measured. TPOT 3.57->3.52 (2 runs: 3.54/3.51).
- Related earlier entries: [17] (prepare optimizations), [4].
- Change: src/run.py - prepare_decode builds a (bs,4) int32 tensor in graph mode (enforce_eager=False) and returns strided column views; run_model's graph-vars copies handle the strides and int32->long casts for input_ids/positions. Eager mode keeps the contiguous 4-tensor path. commit 0eed511.
- Correctness: PASS - bs=1 graph 0/37 vs reference; batched run1 vs run2 0/37; eager bs=1 0/37 vs old eager reference (contiguous path preserved).
- Benchmark before (exp 23): 460.48/459.78 tok/s, TTFT 8.73/10.04, TPOT 3.55/3.59, ITL 3.55/3.59
- Benchmark after (2 runs): 459.62/459.39 tok/s, TTFT 8.38/9.48 (p99 160/196), TPOT 3.54/3.51 (p99 3.73/3.75), ITL 3.54/3.52
- Gap to vLLM now: throughput 1.004x, TPOT 1.11x (was 1.12x), TTFT 0.50x
- Outcome: KEPT
- Why it worked: the four small tensors shared one H2D + column slices; the graph path only needs the values in the contiguous gv buffers (copied by run_model), so strided int32 intermediates are harmless there. Kept a contiguous path for eager mode where store_kvcache/flash read context tensors directly.
- Worth retrying? NO (done).

## [25] 2026-10-10 09:55  Drop redundant graph-vars fills (KEPT)
- Hypothesis and expected gain: graph_bs=[1,2,3,4,5] gives an EXACT graph for every bs in the engine, so a bs=3 step replays the bs=3 graph which only reads rows [0:3] of the shared graph-vars buffers. The 3 fills (slot_mapping/context_lens/block_tables) only reset rows [bs:max] that the exact graph never reads -> redundant -> remove ~3 GPU ops + dispatches from the per-step critical path.
- Profile evidence: steptime replay-launch 0.11ms (5 copies + 3 fills + graph.replay). After: 0.08ms (-30us/step). Step 3.46->3.41ms.
- Related earlier entries: [22] (fill-skip attempt was neutral because it only skipped at bs==max; this removes fills for ALL bs since exact graphs never read padding).
- Change: src/run.py - run_model only fills the graph-vars padding sentinels when use_bs != bs (i.e. the next-larger graph is used, which only happens for max_bs > 4). commit 674de1d.
- Correctness: PASS - bs=1 graph 0/37 vs reference; batched run1 vs run2 0/37.
- Benchmark before (exp 24): 459.62/459.39 tok/s, TTFT 8.38/9.48, TPOT 3.54/3.51
- Benchmark after (2 runs): 460.08/460.12 tok/s, TTFT 10.52/8.25 (p99 196/153), TPOT 3.52/3.51 (p99 3.81/3.72), ITL 3.52/3.51
- Gap to vLLM now: throughput 1.005x, TPOT 1.11x, TTFT 0.53x
- Outcome: KEPT
- Why it worked: the fills were a leftover from when graphs padded up to a larger shape; with exact per-bs graphs the padding rows are never read, so the fills were pure overhead.
- Worth retrying? NO (done).

## [26] 2026-10-10 10:10  Fuse paged-KV store into scale+rotary kernel (KEPT)- Hypothesis and expected gain: decode attention chain was qk prepare + scale_rotate (rotated q/k) + store_kvcache (28 kernels) + flash. The flash reads the paged cache, and the rotated k's only consumer is store_kvcache, so the scale_rotate kernel (which already computes the rotated k and has v available) can write both into the cache -> drop store_kvcache from decode, 28 fewer kernels/step.
- Profile evidence: kernel categories store_kvcache 28 kernels/85us. decode_time 3089us (exp 25 baseline) -> 3039us = -50us/step.
- Related earlier entries: [23] (scale+rotary fuse), [16] (flash in-kernel store FAILED because flash reads the fresh token from registers - here we keep the separate cache write with identical values, so numerics are unchanged).
- Change: layers/rotary_embedding.py - scale_rotate_kernel writes rotated k + v to the paged cache for k-heads (slot*num_kv_heads*head_dim + hh*head_dim); RotaryEmbedding.scale_rotate takes v/k_cache/v_cache/slot_mapping/num_kv_heads/v_stride. layers/attention.py - decode skips store_kvcache when context.kv_stored (bs>=2 fused path); prefill and bs=1 still store. utils/context.py - kv_stored flag. src/qwen3.py - sets kv_stored and passes the cache/v to scale_rotate. commit e8e0246.
- Correctness: PASS - unit test: (prepare+scale_rotate_with_store+flash) vs (scale+rotary+store_kvcache+flash) bit-identical outputs AND k_cache/v_cache contents at seqlen 50/120/256; bs=1 gate 0/37 (separate path unchanged); batched run1 vs run2 0/37.
- Benchmark before (exp 25): 460.08/460.12 tok/s, TTFT 10.52/8.25, TPOT 3.52/3.51, ITL 3.52/3.51
- Benchmark after (2 runs): 459.58/460.30 tok/s, TTFT 9.14/8.02 (p99 194/142), TPOT 3.47/3.46 (p99 3.70/3.67), ITL 3.47/3.47
- Gap to vLLM now: throughput 1.005x, TPOT 1.09x (was 1.11x), TTFT 0.48x
- Outcome: KEPT
- Why it worked: unlike exp 16 (flash in-kernel store, which changed numerics by reading the fresh token from registers), this keeps the K/V values written to the cache byte-identical to store_kvcache (same values, same slot, same per-head offsets), so the flash reads an identical cache. The store is just relocated into an existing kernel that already has the rotated k in registers.
- Worth retrying? NO (done).

## [27] 2026-10-10 11:20  Fuse bs=1 qk-norm+rotary+KV-store path (KEPT)
- Hypothesis and expected gain: the benchmark's decode is ~46% bs=1 (max_tokens=256), and the bs=1 qk path still used the OLD 8-kernel/layer layout (q _rms_one 3 + k _rms_one 3 + rotary_emb_merged 1 + store_kvcache 1) = 224 kernels/step, while the bs>=2 fused path (qk_convert + mean + scale_rotate) is 4/layer = 112. Fusing the bs=1 path -> ~112 fewer kernels/step -> ~300-450us/step saved at bs=1.
- Profile evidence: kernelcount.py at bs=1 decode replay: 571 launches total; convert_add 113 (=57 hidden + 56 qk), mean 113, rms_scale 113, rotary 28, store_kv 28. bs distribution (stepprof_real, max_tokens=256): {1:46%, 2:33%, 3:18%, 4:3%}. Prior exp 15 documented that merging the q+k mean at bs=1 breaks bit-identity (k-reduce config differs for (8,128) vs (24,128)).
- Related earlier entries: [15] (QKNorm merge - the bs=1 fallback), [23] (scale+rotary fuse), [26] (KV store fuse).
- Key insight: the CONVERT is elementwise per-row, so merging q+k into one qk_convert kernel is bit-identical; only the MEANS must remain separate torch calls (x2[:nq].mean / x2[nq:].mean over the same row counts as the old standalone q/k norms -> same reduce config). scale_rotate gained var_q/var_k pointers (bs>=2 passes slices of the merged var as views; bs=1 passes the two separately-reduced vars). 8->4 kernels/layer.
- Change: layers/layernorm.py (QKNorm.prepare_bs1), layers/rotary_embedding.py (scale_rotate_kernel var_q/var_k), src/qwen3.py (bs=1 uses prepare_bs1 + scale_rotate, kv_stored=True). commit b19c29c.
- Correctness: PASS - bs=1 graph 0/37 vs pre-change ref (/tmp/out_kvstore_bs1.json); bs=1 eager 0/37 vs pre-change ref; bs=1 eager vs graph 0/37; batched graph run1 vs run2 0/37.
- Benchmark before (exp 26): 459.58-460.30 tok/s, TTFT 8.02-9.14, TPOT 3.46-3.47, ITL 3.46-3.47
- Benchmark after (2 runs): 459.58 then 459.88 tok/s, TTFT 8.29/8.31 (p99 148/140), TPOT 3.46/3.47 (p99 3.71/3.70), ITL 3.46/3.47
- Gap to vLLM now: throughput 1.004x, TPOT 1.09x, TTFT 0.47x
- Outcome: KEPT (benchmark TPOT neutral at the throughput ceiling, within noise; the underlying improvement is real: bs=1 replay 2973->2818us via replay_vs_wall, step exec in benchmark env 3.46->3.34ms via TOKN_RT_TIMING)
- Why it worked: bs=1 decode is per-kernel-cost bound (571 kernels/step); removing 112 kernels (20%) from the bs=1 step shortens the graph replay by ~155us. The benchmark headline TPOT does not move because (a) bs=1 is only ~46% of decode steps, (b) the remaining steps are already fused, and (c) throughput is at the 460.7 ceiling so the metric is noise-limited. The fusion also simplifies the code (drops the special-case bs=1 fallback through QKNorm.forward).
- Worth retrying? NO (done). Follow-up: the bs=1 hidden-RMSNorm means (57 torch reduce kernels, ~330us total) remain the next kernel-count target, blocked by the reduction-order gate.

## [28] 2026-10-10 11:45  Split-step: enqueue on event loop, GPU wait in executor (REVERTED)
- Hypothesis and expected gain: the run_in_executor full-step path (baseline) costs ~0.21ms/step vs running step inline on the event loop (TPOT 3.46 vs 3.25, measured TOKN_NO_EXECUTOR). The inline path regresses throughput 459.9->454.8 because the blocking step starves the submission coroutine (arrivals drift +2.1s). A hybrid (_step_split) that enqueues schedule/prepare/replay/argmax on the event loop (~0.15ms) and delegates ONLY the GPU wait to `run_in_executor(None, ev.synchronize)` (a bare C call, no GIL churn) should keep arrivals on time AND avoid most of the executor tax.
- Profile evidence: TOKN_NO_EXECUTOR 400-prompt: TPOT 3.25, TTFT 4.85/p99 18, throughput 454.82 (elapsed 213.3s, arrival drift). Baseline executor (exp 27): TPOT 3.46/3.47, TTFT 8.29/8.31/p99 148/140, throughput 459.6/459.9. Split 400-prompt (2 runs): TPOT 3.37/3.37, TTFT 9.14/9.17/p99 217/218, throughput 460.73/460.19 (elapsed 210.99/211.10s, no drift).
- Related earlier entries: [4] (threaded run_loop - the executor was added to fix arrival starvation).
- Change: src/engine.py - TOKN_STEP_MODE env flag (executor|inline|split) + _step_split method. REVERTED (git checkout); the flag and method are not in HEAD.
- Correctness: PASS (runloop gate) - split-mode run1 vs run2 0/37 (deterministic); 9/37 near-tie flips vs the bs=1 reference are expected batch variance (run_loop batches concurrently). Model math is unchanged; only step timing/batching moved.
- Benchmark before (exp 27 executor): 459.6/459.9 tok/s, TTFT 8.29/8.31 p99 148/140, TPOT 3.46/3.47, ITL 3.46/3.47
- Benchmark after (split, 2 runs): 460.73/460.19 tok/s, TTFT 9.14/9.17 p99 217/218, TPOT 3.37/3.37 (p99 3.61/3.63), ITL 3.37/3.37
- Outcome: REVERTED (TTFT p99 regressed 140->217ms, beyond noise, violating "no latency metric regresses")
- Why it failed: TTFT p99 (~217ms) is queueing when all 5 slots are full. Root cause hypothesis: in split mode, run_prefill (which lazily captures multi-seq prefill graphs, ~190ms) runs on the EVENT LOOP inside _step_split, so a multi-seq prefill capture blocks arrivals for ~190ms; in executor mode that capture happens in the worker thread. The TPOT benefit was real (3.46->3.37) and throughput held at the ceiling (460.5), but the TTFT p99 regression makes it a net loss per the keep rule.
- Worth retrying? YES IF the prefill graph capture is moved off the event loop (e.g. run the whole enqueue in the executor when a capture is needed, or pre-capture the benchmark's multi-seq shapes). Would recover ~0.09ms TPOT without the TTFT regression.
- New insight: the executor tax splits into ~0.09ms (enqueue in thread) + ~0.12ms (sync round-trip). A persistent watcher thread (queue + event.synchronize + call_soon_threadsafe) could cut the sync cost, but the prefill-capture-on-loop problem must be solved first.

## [29] 2026-10-10 12:20  Split-step with prefill off-loop + eager fallback on multi-seq prefill miss (KEPT)
- Hypothesis and expected gain: exp 28 showed split-step gets TPOT 3.37 (vs 3.46 executor) but regressed TTFT p99 140->217ms because run_prefill's lazy multi-seq graph capture (~190ms) ran on the event loop. Retry with prefill steps moved wholly into the executor (capture can't block the loop) AND the capture removed from the critical path entirely (eager ~40ms fallback on cache miss). Expected: keep TPOT 3.37 and restore TTFT p99 <= baseline 140ms.
- Profile evidence: debug/ttft_tail.py (400-prompt load, split mode): 3 lazy captures, total_wall 0.65s (~215ms each), shapes (2,72),(2,123),(2,74) - all multi-seq, none repeat. Worst TTFT requests (203/259/264ms) arrive exactly at capture times (t=24.3s, t=147.8s). Eager multi-seq prefill ~40ms (host-bound, exp 13: 680 launches x 66us) vs ~190ms capture.
- Related earlier entries: [28] (split-step, REVERTED for TTFT p99), [4] (executor added for arrival starvation), [13] (prefill graphs; capture body ~190ms inherent).
- Change: src/engine.py - refactor step() into _step_with_seqs + add async _step_split (decode: schedule/prepare/replay/argmax + ev.record on the event loop, then await run_in_executor(None, ev.synchronize) and tolist/postprocess back on the loop; prefill: whole step in executor). TOKN_STEP_MODE env flag (executor default, split, inline). src/run.py - run_prefill returns eager custom_model on cache miss (no synchronous _capture_prefill_graph in the hot path; pre-captured (1,S<=128) grid unchanged; capture_stats dict for diagnostics). debug/ttft_tail.py added.
- Correctness: PASS - run_loop split mode run1 vs run2 0/37 (deterministic); 9/37 near-tie flips vs bs=1 reference is the documented batch variance (unchanged). Model math unchanged (eager prefill is the reference path, bit-identical to graphs per exp 13).
- Benchmark before (exp 27 executor, 2 runs): 459.73 tok/s, TTFT 8.29/8.31 p99 148/140, TPOT 3.46/3.47, ITL 3.46/3.47
- Benchmark after (split, 2 runs): 460.87/459.83 tok/s, TTFT 6.95/6.76 (p99 98.7/68.6), TPOT 3.37/3.41 (p99 3.60/3.68), ITL 3.37/3.41
- Gap to vLLM now: throughput 1.005x, TPOT 1.07x (was 1.09x), TTFT 0.38x (was 0.47x)  <- TPOT and TTFT both closer, TTFT mean/p99 now far below vLLM
- Outcome: KEPT
- Why it worked: (a) decode enqueues on the loop (arrivals on time, TPOT 3.46->3.39) while only the GPU wait is delegated; (b) prefill steps never block the loop with a 190ms capture - they run in the executor, and on a shape miss they run EAGER (~40ms) instead of capturing (the benchmark's multi-seq shapes don't repeat, so the captured graph would never be reused). Both the TPOT win and the TTFT p99 win land. Remaining TTFT tail (~99ms p99) is burst queueing when all 5 slots fill, present in every mode.
- Worth retrying? NO (done). Follow-ups: (1) background capture of multi-seq shapes (capture on a side stream/thread while serving eagerly) would help workloads where multi-seq shapes repeat; (2) the remaining ~0.12ms executor sync round-trip could be cut with a persistent watcher thread; (3) TTFT tail from full-slot queueing - a scheduler tweak (admit prefills even when running==max by preempting? no - keep the queueing semantics) or reducing max_tokens service time.

## [30] 2026-10-10 13:20  Cache padded block-table stacks by seq-set (KEPT)
- Hypothesis and expected gain: splitprof showed prepare_decode ~122us/step, of which torch.stack of the padded block-table rows was ~27us. A sequence's padded block-table row is CONSTANT for its lifetime (all blocks are pre-allocated at admission), so the stacked (bs,8) batch table is constant for a given set+order of running sequences; consecutive decode steps reuse the same set and rebuild the same stack every step for nothing. Cache it -> ~25us/step saved on decode.
- Profile evidence: splitprof (live split-mode loop): prep 122us (torch.tensor 28us + torch.stack 27us + misc). After: prep 87us. rmodel unchanged at ~110us.
- Related earlier entries: [17] (cached padded rows - the per-seq cache this extends to the stacked batch).
- Change: src/run.py - self._bt_cache keyed by tuple(seq.seq_id for seq in seqs); bs=1 uses unsqueeze (no stack); cache device object. ALSO tried a fused graph_vars_copy_kernel (1 Triton launch replacing the 4 __setitem__ copies incl. int casts) - REVERTED because Triton dispatch on a 1-block grid cost more than the 4 torch dispatches (rmodel 110->145us). Kernel + context.graph_input plumbing removed.
- Correctness: PASS - bs=1 graph 0/37 vs ref; run_loop (default split) run1 vs run2 0/37 deterministic. The bt-cache returns identical rows (same padded tensors, same order) so flash sees byte-identical block tables.
- Benchmark before (exp 29, 2 runs): 460.87/459.83 tok/s, TTFT 6.95/6.76 (p99 98.7/68.6), TPOT 3.37/3.41, ITL 3.37/3.41
- Benchmark after (2 runs): 460.03/459.78 tok/s, TTFT 6.80/6.93 (p99 55.85/96.94), TPOT 3.38/3.34 (p99 3.63/3.58), ITL 3.38/3.35
- Gap to vLLM now: throughput 1.005x, TPOT 1.06x (was 1.07x), TTFT 0.38x
- Outcome: KEPT (small: TPOT mean 3.39->3.36, TTFT p99 improved, throughput at ceiling; prep CPU 122->87us confirmed)
- Why it worked: eliminating a redundant per-step torch.stack (whose inputs never change) removed ~27us of fixed CPU per decode step without touching any numerics. The fused-copy kernel failed because Triton's Python dispatch (~30-40us) on a grid-of-one launch exceeds the ~10us/dispatch cost of the 4 native __setitem__ copies it replaced.
- Worth retrying? The fused graph-vars copy could work IF Triton's launch overhead is avoided (e.g. precompile + raw launch, or a CUDA-graph for the copies) - but the gain over native copies is marginal. The DIRECT-READ refactor (capture graphs to read the (bs,4) int32 prepare buffer directly as strided columns, casts inside the graph) is the bigger win (~120us/step) and is next.

## [31] 2026-10-10 14:10  Copy-graph: one H2D + captured copy-graph feeds the decode graph (KEPT)
- Hypothesis and expected gain: the decode step spent ~110us/step in run_model on 5 __setitem__ copies (input_ids/positions/slot_mapping/context_lens/block_tables) plus ~30us torch.tensor in prepare. If the whole decode input (token, pos, ctx_len, slot, block-table row) is assembled on the CPU as one (bs, 4+8) int32 tensor, H2D'd into a STABLE tbuf buffer, and a captured copy-graph distributes it into the model's static input buffers (casts included), then run_model becomes just two graph replays -> ~40us. Expected ~120us/step saved -> TPOT ~3.24.
- Profile evidence: splitprof rmodel 110us (replay only 13us -> copies+dispatch ~90us), prep 87us (torch.tensor 30 + misc). New: rmodel 38us, prep 101us (CPU row assembly incl. block table + copy_ H2D). Net prep+rmodel 197->139us.
- Related earlier entries: [30] (bt cache - superseded by moving the bt onto the CPU row, no GPU stack at all), [24] (single-tensor prepare - this extends it to a full single-H2D + graph dispatch), [22]/[25] (graph fill bookkeeping).
- Key insight: the 5 per-step copies existed only because the model's input buffers are static (frozen at graph capture) while the values come from fresh Python lists. A copy-graph is itself a CUDA graph, so the 5 copies become 5 recorded kernel launches replayed at ~3us total - the CPU cost drops to a single graph.replay() dispatch. The block-table row joins the same (bs,12) tensor, eliminating torch.stack AND the bt cache.
- Change: src/run.py - capture_cudagraph creates tbuf (max_bs, 4+max_num_blocks) int32 + g_input/g_pos/g_slot/g_ctx (long/long/int32/int32) + block_tables (max_bs, 8) and captures self.copy_graph (tbuf->g_*: g_input[:]=tbuf[:,0], g_pos[:]=tbuf[:,1], g_ctx[:]=tbuf[:,2], g_slot[:]=tbuf[:,3], block_tables[:]=tbuf[:,4:12]); main graphs capture reads of g_*[:bs]. prepare_decode (graph mode) builds rows=[last_token,pos,ctx_len,slot]+padded_bt_row on CPU and tbuf[:bs].copy_(torch.tensor(rows, int32)); returns tbuf[:bs,0], tbuf[:bs,1]. run_model replays copy_graph then the main graph. Eager path unchanged. Removed _bt_cache (no longer needed).
- Correctness: PASS - bs=1 graph 0/37 vs ref (same values, same casts, just delivered via graph replay); run_loop default-split run1 vs run2 0/37 deterministic.
- Benchmark before (exp 30, 2 runs): 460.03/459.78 tok/s, TTFT 6.80/6.93 (p99 55.85/96.94), TPOT 3.38/3.34, ITL 3.38/3.35
- Benchmark after (2 runs): 459.27/459.51 tok/s, TTFT 6.74/6.74 (p99 98.26/96.15), TPOT 3.30/3.29 (p99 3.53/3.52), ITL 3.30/3.29
- Gap to vLLM now: throughput 1.004x, TPOT 1.04x (was 1.06x), TTFT 0.38x  <- TPOT gap to vLLM (3.18) now 0.11ms
- Outcome: KEPT
- Why it worked: the per-step copies were CPU-dispatch-bound (5 __setitem__ ~10us each). A copy-graph collapses them into one replay (~3us) with identical numerics (same int32->long casts, same values, same buffer contents). The block table moved into the CPU row so no GPU stack is needed at all.
- Worth retrying? NO (done). The pattern generalizes: any per-step "prepare data into static buffers" work can become a captured copy-graph. Next CPU items: prep is now the bigger half (101us: CPU row build + H2D + set_context) - could shrink the H2D by writing only changed columns, or pre-building the CPU tensor; argmax/event ~35us.

## [32] 2026-10-10 13:50  Single-pass flash for all contexts (REVERTED - gate)
- Hypothesis and expected gain: ctx>256 decode uses the splitkv flash (reference path) which costs ~114us (bs=1) to ~400us (bs=5) more per step; ~20% of benchmark decode steps exceed ctx 256. If single-pass (num_splits=1) were token-identical there too, drop splitkv entirely.
- Profile evidence: debug/ctx_profile.py (40-question burst, max_tokens=256): bs=3 ctx<=256 2940us vs ctx>256 3336us; bs=5 3010 vs 3351us. bs=1 single-long-gen: ctx<=256 2849 vs ctx>256 2963us. The splitkv penalty is small at bs=1 (the dominant batch size).
- Change: temporarily forced use_fast_flash=True in prepare_decode. REVERTED.
- Correctness: FAIL - bs=1 graph gate 4/37 differ (the 474-token prefix prompts; flash single-pass vs splitkv diverge there). Also the benchmark's ctx 256-369 range is not covered by the gate's max_tokens=48, so any threshold would need bespoke verification.
- Outcome: REVERTED (correctness). Not worth pursuing: bs=1 penalty ~114us on ~20% of steps = ~23us/step average, and the gate risk (prefix prompts) is high.
- Worth retrying? NO for this config. If a future workload has long contexts AND low concurrency the splitkv cost would matter more, but the gate would still block single-pass for the reference prompts.

## [33] 2026-10-10 14:35  Prep micro: non_blocking H2D into tbuf (REVERTED - ambiguous)
- Hypothesis and expected gain: microbench showed copy_(cpu, non_blocking=True) at 15.8us vs 22.7us for the default copy_ into tbuf. Expected ~7us/step CPU saved.
- Profile evidence: splitprof prep 100.8->97.7us. But benchmark TPOT 3.30 then 3.37 (2 runs) vs exp 31's 3.30/3.29 - ambiguous, possibly noise, and the non_blocking path could add a subtle driver-side stall for pageable H2D. Reverted to exp 31's plain copy_ (the best-known state).
- Outcome: REVERTED (ambiguous, tiny benefit). Tree back at exp-31 commit.
- Worth retrying? NO - the benefit is below noise.

## [34] 2026-10-10 15:05  Graph-argmax (REVERTED - ambiguous)
- Hypothesis and expected gain: the split path's `torch.argmax(logits, dim=-1)` dispatch (~23us/step) could be removed by capturing the greedy argmax as the LAST decode-graph op and reading g_tokens instead. Expected ~18us/step net (23us saved - 5us added to the replay).
- Change: src/run.py (g_tokens buffer; `g_tokens[:bs] = torch.argmax(outputs[:bs], dim=-1)` inside each decode graph capture; run() reads g_tokens for decode-graph steps) + src/engine.py (_step_split reads graph_vars["g_tokens"]). REVERTED (git checkout).
- Correctness: PASS - bs=1 graph 0/37 vs ref (argmax deterministic, same logits -> same tokens); run_loop deterministic 0/37.
- Benchmark: 3 runs TPOT 3.29/3.33/3.51 (mean 3.377), TTFT p99 43.8/97.3/107.7 vs exp 31's 3.30/3.29. Not better; one clear outlier.
- Outcome: REVERTED (ambiguous - below noise, extra complexity, added a per-replay kernel).
- Why it failed: the ~23us argmax dispatch saving is offset by the ~5us argmax kernel now in EVERY decode graph replay, and the benchmark's ~+-0.04ms noise exceeds the ~18us expected gain, so no net improvement was measurable; one run regressed to 3.51 (outlier).
- Worth retrying? NO unless the argmax kernel in the graph can be made cheaper than torch's reduce_kernel (Triton argmax was 31us standalone - slower; a better-tuned argmax could flip this).

## [35] 2026-10-10 15:20  Graph-argmax + pinned D2H token read + reused event (REVERTED - neutral)
- Hypothesis and expected gain: exp 34 was below noise. Adding a pinned CPU mirror of g_tokens, H2D-copied with an async non_blocking copy_ after the decode-graph replay, would ALSO remove the ~14us GPU-syncing .tolist() (a fresh torch.cuda.Event() each step is ~1us, not worth pooling). Expected ~30us/step net -> TPOT ~3.26.
- Profile evidence: torch.argmax dispatch 17.4us, Event() creation 0.9us, ev.record 7.9us, tok.tolist() 14.3us (clean standalone numbers). In-engine (TOKN_STEP_DEBUG): argmax 28.7us, record 23.2us (incl. debug overhead), post (tolist+postprocess) 46us.
- Related earlier entries: [34] (graph-argmax only, REVERTED).
- Change: src/run.py - g_tokens buffer written as last decode-graph op; pinned _gt_cpu buffer + reusable _ev event; run_model returns (logits, g_tokens) for the graph path. src/engine.py - _step_split H2D-copies gtok into _gt_cpu (non_blocking), records the reused event, reads _gt_cpu after the wait (no tolist on GPU). REVERTED (git checkout).
- Correctness: PASS - bs=1 graph 0/37 vs ref (argmax deterministic); run_loop run1 vs run2 0/37.
- Benchmark: 2 runs TPOT 3.36/3.27 (mean 3.315), TTFT p99 97.9/96.0, throughput 459.84/460.31. Baseline (exp 31): 3.30/3.29, TTFT p99 98/96. Neutral.
- Outcome: REVERTED (ambiguous - within noise, added a pinned buffer + event reuse + a D2H on the GPU critical path for no measurable gain).
- Why it failed: the D2H copy into _gt_cpu is enqueued on the stream after the graph, so it lands on the GPU critical path (+11us rtt 2828->2839, measured via splitprof). The CPU savings (argmax dispatch -17us, tolist -14us) are offset by the copy_ dispatch (+5us) and the D2H (+11us) -> ~ -15us/step net, which is below the benchmark's ~40us noise. Same conclusion as exp 34: the exposed CPU per step is too small a fraction of the ~3.3ms step to move TPOT measurably.
- Worth retrying? NO for the current step structure. The token read is NOT the bottleneck; the decode GPU replay (~2.83ms) and the fixed CPU+overhead (~0.3-0.4ms) dominate, and neither has an available bit-identity-safe reduction (see the plateau analysis below).

## [36] 2026-10-10 15:40  Fuse decode-input copies into each decode graph (KEPT - neutral, simpler)
- Hypothesis and expected gain: run_model replayed TWO graphs every decode step: the copy-graph (tbuf -> g_input/g_pos/g_ctx/g_slot/block_tables, ~3us GPU, ~10-15us launch) then the decode graph. Capturing the 5 tiny copy kernels at the FRONT of each decode graph lets run_model replay ONE graph -> rmodel ~39->24us, no second launch.
- Profile evidence: splitprof rmodel 38.97us (exp 31) -> 31.56us (after). rtt 2828->2833us (+5us: the copies now run inside the replayed graph, previously they were a separate replay).
- Related earlier entries: [31] (copy-graph), [25]/[22] (fill bookkeeping - the use_bs!=bs fills are dead with exact per-bs graphs).
- Change: src/run.py - in capture_cudagraph, each decode graph capture now begins with `g_input[:bs]=tbuf[:bs,0]`, `g_pos[:bs]=tbuf[:bs,1]`, `g_ctx[:bs]=tbuf[:bs,2]`, `g_slot[:bs]=tbuf[:bs,3]`, `block_tables[:bs]=tbuf[:bs,4:4+max_num_blocks]` then the model forward; run_model drops `self.copy_graph.replay()` in the decode path (copy_graph kept only for the >graph_bs fallback). commit a2e474f.
- Correctness: PASS - bs=1 graph 0/37 vs ref_pre_splitkv_bs1.json; run_loop split run1 vs run2 0/37 (deterministic).
- Benchmark before (baseline this session): 459.50 tok/s, TTFT 6.23/43.65, TPOT 3.32/3.59, ITL 3.33/3.85
- Benchmark after (2 runs): 459.55 then 459.94 tok/s, TTFT 6.69/6.68 (p99 98.16/100.17), TPOT 3.31/3.29 (p99 3.64/3.54), ITL 3.31/3.29
- Gap to vLLM now: throughput 1.004x, TPOT 1.04x, TTFT 0.37x  <- TPOT still trails vLLM's 3.18 by ~0.11ms
- Outcome: KEPT (neutral on metrics within noise; strictly fewer per-step launches; simpler run_model). TTFT p99 (~98-100) is the historical burst-queueing band (43-100), not a regression.
- Why it worked: the copy ops are the same 5 kernels, just relocated into the graph they feed; run_model went from 2 replays to 1. The net metric delta is within noise, but the code is simpler and there is no regression.
- Worth retrying? NO (done). This was the last cheap "remove a launch" item on the decode path.

## [37] 2026-10-10 16:10  PLATEAU ANALYSIS - decode step is GPU-replay-bound; CPU and GPU reductions are bit-identity-limited (session-8 findings, no code change)
- Context: after exp 36, TPOT 3.29-3.31 vs vLLM 3.18. This entry documents why the remaining ~0.11ms gap is hard, based on extensive profiling this session.
- Profile evidence:
  - Real-engine per-phase (TOKN_STEP_DEBUG, 60 prompts/64 tok, p50): prepare_decode 83us, run_model 24us, argmax 29us, ev.record ~8us, GPU wait 2801us, tolist+postprocess 46us, hash+dealloc 0.8us, schedule 0.1us. Step loop ~3.07ms. In the full benchmark (400/256, avg bs 1.78, ctx up to 369 with splitkv flash), the GPU wait is ~2.9-3.0ms -> step ~3.29ms = TPOT p50. So the decode GPU replay is ~88-92% of the step and the CPU+overhead is ~0.3-0.4ms.
  - bs=1 decode replay kernel categories (torch profiler, 20 iters): GEMM 2133us (67%, of which lm_head 370us), mean 113 kernels 361us (11%), flash fast 221us (7%), convert_add 119us, rms_scale 118us, scale_rotate 92us, qk_convert 57us, silu 57us. Total ~3179us (profiler) / 2828us (CUDA events).
  - GEMM microbench (session8_gemm_micro, real weights, bs=1): qkv 23.8us, o 23.7us, gate_up 25.7us, down 24.8us standalone (incl. launch ~7us); in-replay ~15us each. lm_head 370.7us at every bs - the lmhead_triton.py bandwidth kernel reads the same 311MB in 372.7us, so cuBLAS's lm_head is AT the ~834GB/s memory floor (no headroom).
  - H2D cost is launch-bound: 48B = 11.5us, 16B = 11.5us, 4KB = 12us (pinned+non_blocking 5.5us but writing the pinned buffer from Python costs as much as the direct H2D; the non_blocking path also risks a CPU-write-vs-inflight-H2D race, so no safe win).
  - Means: redmatch2.py re-confirmed no Triton BLOCK/num_warps config matches torch's mean_kernel for (1,1024)/(5,1024)/(16,128)/(8,128); redprobe.py found no reshape-based torch reduction that matches. The 113 mean calls/step (327us bs=1) are blocked from fusion.
  - Prefill: graph replay 3.1-3.9ms for S=13-113 (memory-bound GEMMs, same 882MB weight read as decode). Prefills are 0.73% of steps, so they add only ~20-30us to mean ITL (a prefill replaces a same-duration decode step).
- Related earlier entries: [3] (RMSNorm fusion FAILED), [12] (pow fold), [32] (single-pass flash FAILED gate), [19] (throughput ceiling).
- Conclusion / why the gap is hard: (1) the decode GPU replay (2.83-3.0ms) is dominated by GEMMs that are at/near the memory-bandwidth floor (lm_head at floor, layer GEMMs 1.6x floor) and by 113 mean-reduce kernels whose fusion is blocked by the reduction-order gate; flash, norms and rotary are already fused/single-pass/minimal. (2) The exposed CPU (~0.3-0.4ms) is split across ~6 small phases (prep 83, post 46, argmax 29, rmodel 24, event 8, loop/executor ~150-200), each of which is a <30us item below the benchmark's ~40us run-to-run noise (verified by exp 33/34/35 all measuring neutral). (3) The remaining TPOT gap to vLLM (3.29 vs 3.18) would need a ~0.11ms reduction that no available bit-identity-safe change provides.
- Worth retrying? NO single item; the only untried structural ideas are large/risky (see backlog): overlapping prefill with decode on a separate stream (KV-cache and model-buffer races), GPU-resident token passing (scheduler still needs CPU tokens), and GEMM fusion (bit-identity).

## [38] 2026-10-10 15:45  Busy-poll the CUDA event in _step_split (drop run_in_executor wait) (KEPT)
- Hypothesis and expected gain: the split-step path (exp 29) delegates the GPU wait to `await loop.run_in_executor(None, ev.synchronize)`. exp 28/29 estimated this executor round-trip at ~0.1-0.12ms/step directly on the TPOT critical path (split 3.37 vs inline 3.25). If the wait is instead a busy-poll on `ev.query()` that yields to the event loop every iteration (`await asyncio.sleep(0)`), the executor tax disappears while arrivals stay on time (unlike the fully-inline mode which blocks the loop on ev.synchronize and starves arrivals, exp 28). Expected TPOT ~3.25, throughput unchanged, TTFT unchanged.
- Profile evidence: debug/execwait_probe.py (new) - same ~1.7ms GPU event: run_in_executor total p50 1.726ms vs busy-query p50 1.330ms vs inline 1.334ms. The executor wait adds ~0.40ms p50 (up to 0.29ms mean) over the event itself - larger than the earlier ~0.12ms estimate. (The real-engine saving is smaller than the microbench because some of the enqueue overlaps the GPU; measured 3.29->3.21.)
- Related earlier entries: [29] (split-step introduced the executor wait), [28] (inline mode REVERTED for arrival starvation - busy-poll avoids that by yielding), [4] (executor added for arrival starvation).
- Change: src/engine.py - `_step_split` decode path replaces `await loop.run_in_executor(None, ev.synchronize)` with `while not ev.query(): await asyncio.sleep(0)`. Prefill path unchanged (still runs wholly in the executor so lazy graph capture can't block the loop). commit fb0d8e6.
- Correctness: PASS - bs=1 graph 0/37 vs /tmp/out_kvstore_bs1.json (math untouched); run_loop busy-poll run1 vs run2 0/37 deterministic.
- Benchmark before (this session baseline): 459.54 tok/s, TTFT 6.70/99.52, TPOT 3.29/3.52, ITL 3.30/3.77
- Benchmark after (2 runs): 459.27/459.71 tok/s, TTFT 6.96/6.93 (p99 86.27/87.72), TPOT 3.21/3.22 (p99 3.44/3.45), ITL 3.21/3.22
- Gap to vLLM now: throughput 1.004x, TPOT 1.01x (was 1.04x; 3.21 vs vLLM 3.18), TTFT 0.39x  <- TPOT gap to vLLM now only ~0.03ms
- Outcome: KEPT
- Why it worked: the executor wait's thread-wakeup + future round-trip was pure serialized overhead on the step critical path; ev.query() is a cheap non-blocking host call, and yielding with sleep(0) every iteration lets the event loop service arrivals on time (the thing inline mode got wrong), so throughput and TTFT hold while TPOT drops.
- Worth retrying? NO (done). Follow-up: TPOT 3.21 is within ~0.03ms of vLLM (3.18); the remaining levers are the ~190us fixed per-step CPU (enqueue 144 + post 46) and the GPU replay itself (bit-identity-limited per [37]). Re-examine graph-argmax (exp 34/35 were neutral at the old executor-tax baseline; the balance changed with the tax gone - now ~24us CPU for ~5us GPU is worth re-measuring).

## [39] 2026-10-10 16:10  gemvx-kernel investigation (bs=1 GEMM dispatch) (REVERTED - no win)
- Hypothesis and expected gain: the bs=1 decode profile showed a single gemvx kernel entry at ~844us (22.8%) which looked like the lm_head running the slow GEMV kernel instead of the fast bf16 GEMM. If the lm_head (311MB weight) were at gemvx speed, forcing the GEMM would save ~450us/step on bs=1 (~46% of steps) -> TPOT ~3.0ms.
- Profile evidence: debug/bs1_engine_prof.py - engine bs=1 decode graph replay: ampere_bf16 GEMMs 890+640us, gemvx 844us, reduce_kernel (mean) 467us, flash 274us. count=56 for gemvx (2/layer = o + down projections) at ~15us each - NOT one 844us kernel. The lm_head is one of the ampere GEMMs at the ~390us memory floor (exp 37 was right). isolated captures: o gemvx 11.4us, down gemvx 16.8us; M=2-padded GEMM: o 14us (SLOWER), down 16.7us (same) AND down M=2 GEMM row0 NOT bit-identical to M=1 gemvx. lm_head M=2 padding leaves the engine's gemvx unchanged (the gemvx is not the lm_head at all). A split lm_head into a separate freshly-captured graph also used the GEMM (385us) but the layers graph still contained the o/down gemvx - confirming the gemvx was never the lm_head.
- Related earlier entries: [37] (plateau - lm_head at floor), [20] (kernel-count reductions).
- Change: NONE kept. Tried (all reverted): lm_head M=2 padding in embed_head.py, lm_head split into a separate graph in run.py. Investigation probes: debug/lmhead_micro.py, debug/gemv_probe.py, debug/gemv_probe2.py, debug/lmhead_variant_prof.py, debug/bs1_engine_prof.py, debug/decodeprof2.py.
- Correctness: NOT REACHED as a keeper (no perf change to keep). Verified gemvx == GEMM bit-identical for the lm_head shape, and that the engine's o/down gemvx cannot be cheaply replaced.
- Benchmark: NOT RUN (no change kept).
- Outcome: REVERTED (no win - misinterpreted the profile)
- Why it failed: the "gemvx 844us" was the profiler AGGREGATING 56 o/down GEMV calls (2 per layer, ~15us each = their normal cost at 2.2x the memory floor for small-N M=1 GEMMs). The lm_head always ran the fast GEMM. o/down M=2 padding is not bit-identical (down) or slower (o), so there is no bit-identity-safe way to speed up the small-N layer GEMMs.
- Worth retrying? NO. The o/down gemvx is cuBLAS's normal choice for these shapes and runs at ~2.2x its memory floor with no bit-identity-safe alternative.

## [40] 2026-10-10 16:40  CPU micro-opts on the exposed path (pinned+numpy H2D, reusable context) (REVERTED - neutral)
- Hypothesis and expected gain: the busy-poll step (exp 38) leaves ~140us/step of exposed CPU (prep 77 + rmodel 24 + post 43) that delays the GPU start ~1:1 (verified with debug/gpu_overlap_probe.py: step_wall ~= GPU_graph + prep + rmodel + post). If prep's tensor-build + H2D (15+27us) and set_context (14us) were faster, TPOT would drop ~1:1. Expected ~20-30us/step -> TPOT ~3.18.
- Profile evidence: debug/prep_probe.py - prep breakdown: rows 2.6 + torch.tensor 15 + copy_ 27 + set_context 13.5 = 58us (rest is call overhead). Microbench debug/prep_h2d_micro.py: pageable tt+copy_ 24us vs numpy-fill+pinned+non_blocking 15.4us.
- Related earlier entries: [33] (non_blocking prep REVERTED ambiguous), [31] (copy-graph), [17] (prepare opt).
- Change: (a) persistent pinned (5,12) int32 buffer + numpy fill + non_blocking H2D in prepare_decode; (b) set_context mutates a reused InferenceContext instead of constructing a dataclass. BOTH REVERTED (git checkout) - prep measured 76.6->72.4us (a) then 78.4us (b), all within noise; no TPOT change.
- Correctness: PASS (values bit-identical - same int32 rows, same context fields). Not benchmarked as a keeper (neutral in the profiler).
- Benchmark: NOT RUN (neutral in the profiler, below the ~40us benchmark noise).
- Outcome: REVERTED (no gain)
- Why it failed: the in-engine H2D copy_ is ~27us regardless of pageable/pinned/non_blocking (microbench savings don't survive the engine's per-call overhead), and set_context was cheaper than the probe suggested. Every exposed-CPU item is a <15us change below the ~40us run-to-run noise.
- Worth retrying? Only as a stacked bundle (prep + argmax + post together might reach ~40us), but each is sub-noise individually and the risk of an ambiguous result is high.

## [41] 2026-10-10 17:25  Speculative decode prep (hide the state build under the GPU wait) (KEPT)
- Hypothesis and expected gain: after the busy-poll win [38], the decode step still has ~140us of exposed CPU (prep 77 + rmodel 24 + post 43) that delays the GPU start ~1:1 (verified: GPU is idle during prep/post). The decode input row is [token, pos, ctx_len, slot, block-table]; ONLY the token comes from the previous step's argmax - the other columns depend only on the running set and current token counts. If the next step's state columns are built+H2D'd during the previous step's GPU wait, and the next step only rewrites the token, the exposed prep drops from ~77us to ~25us -> TPOT ~3.16.
- Profile evidence: debug/gpu_overlap_probe.py - step_wall ~= GPU_graph + prep + rmodel + post (CPU enqueue fully exposed). debug/spec_probe.py - the token_only fast path fires on 97.7% of decode steps (only 187/8056 full preps, the rest are stable-set decode). Strided column write into a (bs,12) tbuf is 41us vs 22us for a contiguous write -> motivated the buffer split.
- Related earlier entries: [31] (single-tensor prepare - this replaces it), [38] (busy-poll - the GPU wait that this hides under), [40] (CPU micro-opts - the context).
- Change: src/run.py - split the decode input into tbuf_tok (max_bs,) + tbuf_state (max_bs, 3+max_num_blocks); prepare_decode gains token_only + spec_next modes; capture_cudagraph's copy front-ops read the two buffers. src/engine.py - _step_split checks self._spec_ids (seq-id tuple) to pick token_only vs full prep, and during the GPU wait (if ev not done) calls prepare_decode(spec_next=True) which builds the NEXT step's state rows into tbuf_state via a PINNED non_blocking H2D and sets the context speculatively. First version used a PAGEABLE H2D in spec_next which is SYNCHRONOUS -> blocked the loop for the full ~2.8ms GPU wait (throughput 459.5->455.4, elapsed +2.3s); fixed by pinning the source + non_blocking=True (spec_next back to ~42us, hidden). commit 5c7f4b8 (code).
- Correctness: PASS - bs=1 graph 0/37 vs /tmp/out_kvstore_bs1.json; run_loop (speculative) run1 vs run2 0/37; eager 0/37 vs reference. The speculative state columns must match what a fresh prepare would produce (pos=len, ctx=len+1, slot@len); a seq-id comparison falls back to a full rebuild on any set change (finishes/prefills).
- Benchmark before (exp 38): 459.5 tok/s, TTFT 6.70/99.52, TPOT 3.21/3.52, ITL 3.30/3.77
- Benchmark after (2 runs): 459.12/459.20 tok/s, TTFT 6.93/7.00 (p99 79.9/80.0), TPOT 3.19/3.19 (p50 3.19/3.18, p99 3.42/3.41), ITL 3.19/3.19
- Gap to vLLM now: throughput 1.003x, TPOT 1.00x (3.19 vs vLLM 3.18 - at parity), TTFT 0.39x
- Outcome: KEPT
- Why it worked: the state build + full H2D (~45us) moved off the critical path (now hidden under the GPU via the async pinned copy), and the token write became a cheap contiguous H2D. The exposed prep dropped from ~77us to ~52us (token_only), and the context is pre-set by the speculative prep. Two consistent runs at TPOT 3.19 (vs 3.21/3.22 baseline). The pageable-H2D bug taught the key constraint: any H2D done while the GPU is busy MUST use a pinned source + non_blocking, or cudaMemcpy becomes synchronous and blocks the loop.
- Worth retrying? NO (done). Remaining gap to vLLM (3.18) is ~0.01ms = noise. The token_only still costs ~52us (token H2D 22 + torch.tensor + overhead) - the token H2D itself is the floor for CPU-fed tokens; GPU-resident token passing would need the scheduler's CPU tokens anyway.

## [42] 2026-10-10 18:10  Side-stream prefill concurrent with decode (REVERTED - TTFT regression)
- Hypothesis and expected gain: prefills (~5ms) replace decode steps and stall every running sequence (~400x in the benchmark). Running the prefill on a second CUDA stream concurrent with the decode replay should remove the decode stall and prefill requests sooner. Expected TPOT/ITL -0.05ms and TTFT -1ms.
- Profile evidence: debug/ttft_tail.py (400 prompts): TTFT mean 6.95 p99 99.3, worst arrivals at t=78/99/147s. Arrival-process analysis (seed 0): max 15 arrivals in a 3s window at 145-148s -> burst queueing when all 5 slots fill. debug/prefill_replay_prof.py (new): prefill graph replay p50 4.41ms (S=50) / 5.79ms (S=100) / 5.88ms (S=113), GEMM-dominated (63%: 3.7ms for 882MB = 238GB/s, ~5x below the 834GB/s floor). debug/itl_audit.py (new): ITL p50 3.10 p99 3.50 p999 7.99; only 108/24196 ITLs >5ms (0.45%) -> prefill stalls are ~1/request but ABOVE p99, so removing them helps only the unreported p999/max.
- Related earlier entries: [29] (split-step), [4] (executor for arrival starvation), none for stream overlap.
- Change: utils/context.py (thread-local _CONTEXT so the loop's decode enqueue and the executor's prefill enqueue can't race on the global context), src/run.py (side CUDA stream), src/scheduler.py (_schedule_unlocked returns (decode_seqs, prefill_seqs); combined decode+prefill step when a slot is free), src/engine.py (_step_split handles both batches; _run_prefill_gpu runs the prefill on the side stream in the executor; _postprocess helper). Diff saved to experiments/42_side_prefill/side_stream_prefill.diff. REVERTED via git checkout.
- Correctness: PASS - bs=1 graph 0/37 vs /tmp/out_kvstore_bs1.json (math unchanged); run_loop split runA vs runB 0/37 deterministic; 6/37 differ vs the exp-41 batched reference (near-tie batch variance from the changed step structure).
- Benchmark before (exp 41): 459.2 tok/s, TTFT 6.95/79.9, TPOT 3.19/3.42, ITL 3.19
- Benchmark after (400 prompts): 475.8 tok/s, TTFT 11.22/144.8, TPOT 3.20, ITL 3.20. BUT output_tokens went 96789 -> 100298 (+3.6% from near-tie flips) while elapsed was UNCHANGED (210.80s vs 210.78s) -> the +3.6% throughput is ENTIRELY a token-count artifact, zero real speedup. The TTFT regression is real: median 7.60ms (vs 5.06).
- Outcome: REVERTED (TTFT mean +61%, p99 +81% - far beyond noise; no real throughput/TPOT gain)
- Why it failed: (1) the prefill and decode are BOTH memory-bound on the SAME 882MB weights, so running them concurrently does not parallelize - it just slows each down (GPU work is conserved, exp 37 predicted this). The waiting request's prefill went from full-bandwidth ~3.3ms to contended ~5-6ms, regressing TTFT for EVERY prefill. (2) The decode postprocess waited on the prefill future, delaying running sequences further. (3) The prefill-stall removal only helps the unreported p999 (stalls are 0.45% of ITLs, above p99), so even a perfect implementation shows nothing on the reported metrics.
- Worth retrying? NO for this benchmark. The TTFT p99 tail (99ms) is burst queueing (up to 15 arrivals in 3s vs 5 slots); ANY scheme admitting the 6th+ request's prefill during a burst must either stall decode (prefill-only step, status quo) or contend for the same 882MB bandwidth (side stream, slower TTFT). No free lunch.

## [43] 2026-10-10 18:15  Skip redundant prefill store_kvcache (KEPT)
- Hypothesis and expected gain: the prefill's fused scale+rotate kernel (bs>=2 / bs=1 fused paths, exp 26/27) already writes the rotated K and V into the paged KV cache at the token slots (kv_stored=True), but Attention.forward's PREFILL branch called store_kvcache UNCONDITIONALLY (line 64) while the DECODE branch gates on `not context.kv_stored` (exp 26). The prefill double-wrote identical bytes: 28 store_kvcache kernel launches (~100us at S=100) + duplicate write bandwidth (~11.5MB). Skipping it is bit-identical (same values, same slots, cache ends up identical). Expected ~100us/prefill saved -> TTFT -0.1ms.
- Profile evidence: debug/prefill_replay_prof.py (S=100, 3 iters): store_kvcache 302us total = 84 calls @3.6us (28/iter); the prefill replay was 5.79ms total. Attention.forward line 64 vs line 72 inconsistency confirmed by reading the code; Qwen3-0.6B has attention_bias=False so every prefill takes the qk_norm fused path (kv_stored=True).
- Related earlier entries: [26] (fused scale+rotate+KV-store for decode - the same kernel also runs for prefill), [16] (flash in-kernel store FAILED - different numerics; this is NOT that: the store values are byte-identical, just relocated).
- Change: layers/attention.py - prefill branch gates store_kvcache on `not context.kv_stored` (same as the decode branch). commit 7d5c38a.
- Correctness: PASS - bs=1 graph 0/37 vs /tmp/out_kvstore_bs1.json; run_loop split runA vs runB 0/37 AND runA vs the exp-41 reference 0/37 (no batching/timing change at all -> ZERO near-tie flips, fully bit-identical).
- Benchmark before (exp 41, 2 runs): 459.2 tok/s, TTFT 6.93/7.00 (p99 79.9/80.0), TPOT 3.19, ITL 3.19
- Benchmark after (2 runs): 459.21/459.19 tok/s, TTFT 6.33/6.75 (p50 4.96/4.98, p99 43.8/81.5), TPOT 3.19/3.19, ITL 3.19. output_tokens 96815/96800 (baseline 96789 - run-to-run near-tie noise only).
- Gap to vLLM now: throughput 1.003x, TPOT 1.00x (3.19 vs 3.18), TTFT 0.37x (mean 6.54 vs 17.87)
- Outcome: KEPT
- Why it worked: a redundant double-write of identical bytes is pure overhead; gating it on the kv_stored flag (already the decode behavior) removes 28 launches + duplicate bandwidth per prefill with no numeric change. The TTFT mean improvement (~0.4ms, p50 ~0.1ms) exceeds the ~100us direct estimate because faster prefills cascade into faster slot turnover during partial bursts. The p99 (43.8/81.5) is unchanged within noise - the burst-queueing tail is admission-bound, not prefill-speed-bound.
- Worth retrying? The pattern generalizes: audit the prefill path for other redundant ops (the prefill still has 85 mean + 57 convert_add + 57 rms_scale + 28 qk_convert + 28 scale_rotate + 28 flash + 28 silu kernels, all necessary or bit-identity-blocked). Done for the store.

## [44] 2026-10-10 18:32  bs=1 merged q/k mean (TOKEN gate, not bit-identity) (REVERTED - FAILED gate)
- Hypothesis and expected gain: at bs=1 the qk norm runs TWO separate torch means per layer (var_q over (16,128), var_k over (8,128)) = 56 mean calls, 28 MORE than the bs>=2 path's single merged mean. The merged (24,128) mean was rejected in exp 15/27 for NOT being bit-identical (torch picks a different reduce_kernel config for 8 rows). Retry with the TOKEN gate (identical greedy tokens at bs=1) instead of bit-identity: if the ~1e-7 var change does not flip any of the 37 bs=1 sequences, keep it and save 28 mean launches (~118us/step at bs=1, ~54us/step avg -> TPOT ~3.19->3.14, would clearly beat vLLM 3.18).
- Profile evidence: debug/kernel_dump.py bs=1 decode replay: 459 launches/3724us; reduce_kernel 113 calls/474us. bs=1 is ~46% of decode steps (stepprof_real). 
- Related earlier entries: [15] (QKNorm merge, bs=1 fallback kept because merged mean not bit-identical), [27] (prepare_bs1 introduced for the separate means), [3] (RMSNorm fusion FAILED 12/37).
- Change: layers/layernorm.py prepare_bs1 - single `var = x2.mean(dim=-1)` over the merged (24,128) rows instead of separate x2[:nq].mean / x2[nq:].mean. REVERTED (git checkout).
- Correctness: FAIL - bs=1 graph 1/37 differ vs fresh reference (/tmp/ref_s11_bs1.json): seq 8 flips a near-tie token at pos 40 (13 -> 518). 
- Benchmark: NOT RUN (failed gate).
- Outcome: REVERTED (FAILED - correctness)
- Why it failed: the model's greedy outputs are dense in near-tie tokens (top-2 logit gap < ~1e-4); a ~1e-7 relative change in the k-row variance (from the different reduce config) cascades through 28 layers and flips a token. This empirically confirms that bit-identity is effectively REQUIRED for any model-math change on this model/benchmark, which is why the earlier conservative entries kept the separate means.
- Worth retrying? NO - token gate and bit-identity coincide here; the 28-mean saving is unreachable without matching torch's exact reduce config.

## [45] 2026-10-10 18:40  Prefill GEMM M-scaling microbenchmark (REVERTED - small-M GEMM inefficiency is inherent, no bit-identity-safe fix)
- Hypothesis and expected gain: prefill replay is 5.7ms (S=100) with GEMMs ~3.34ms at ~25-30% efficiency (238 GB/s vs 834 floor). If the inefficiency were from small-M (M=100 < one 128-tile = 64 CTAs < 108 SMs), padding M or batching prefills could recover 2-3x -> TTFT mean/p50 and ITL p99 all improve.
- Profile evidence: debug/prefill_replay_prof.py: qkv/gate_up/o/down are 4 GEMMs/layer using 64x64/128x64 tiles, ~23-37us each at M=100. GEMMs total 3.34ms of 5.7ms prefill.
- Related earlier entries: backlog "prefill GEMM speedup... blocked" (session 9-10); [13] (prefill graphs), [32] (single-pass flash FAILED).
- Change: NONE. Investigation only (debug/gemmbench3.py - fresh microbench of the exact prefill GEMM shapes at M=100/128/256/512 with torch.equal on real rows).
- Correctness: NOT REACHED (no keeper). Measured p50 us (M=100 / M=128 / M=256 / M=512), torch.equal(real-rows) vs M=100: qkv 53.2/53.2(F)/74.8(F)/109.6(F); gate_up 37.9/37.9(F)/53.2(F)/74.8(F); o 41.0/41.0(F)/47.1(F)/76.8(F); down 41.0/41.0(F)/47.1(F)/76.8(F); lm_head 537.6/586.8(F)/1072.1(F)/1649.7(F). Kernel selection (qkv): M=100 and M=128 both use ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn (SAME kernel) yet the real rows are NOT bit-identical; M=256 switches to 128x128 and is SLOWER.
- Outcome: REVERTED (no bit-identity-safe change)
- Why it failed: (a) padding M 100->128 keeps the same kernel AND the same time (both 53.2us) but the real-row outputs DIFFER (cuBLAS changes the per-row accumulation based on the exact M, even within the same tile shape) -> would flip near-tie tokens (per exp 44) and fails the gate; (b) M=256 is SLOWER (74.8us) - a second M-tile re-reads the weight and the 128x128 kernel is not faster for these shapes. So batching prefills to larger M is BOTH slower AND not bit-identical. The prefill GEMM inefficiency is inherent to cuBLAS on A30 for these small shapes; no bit-identity-safe fix exists.
- Worth retrying? YES IF a Triton/CUTLASS GEMM can reproduce cuBLAS's exact per-row accumulation for the prefill shapes (unlikely - exp 37 redmatch2-style work needed). Otherwise blocked.

## [46] 2026-10-10 19:00  Batch queued prefills to raise prefill M (scheduler) (ABANDONED - precluded by [45])
- Hypothesis and expected gain: during Poisson bursts several requests queue; batching their single-seq prefills into one multi-seq forward raises M and could amortize fixed costs, improving TTFT for queued requests.
- Outcome: NOT PURSUED. [45] showed larger M is SLOWER (M=256 qkv 74.8us vs M=100 53.2us) AND not bit-identical; the scheduler already batches multi-seq prefills when several wait (the benchmark's 3 multi-seq shapes (2,72)/(2,123)/(2,74)); and the TTFT p99 tail is burst-vs-5-slots queueing (admission-bound, exp 43), not prefill-M inefficiency. Abandoned.

## [47] 2026-10-10 18:50  In-graph greedy argmax writes tbuf_tok (KEPT)
- Hypothesis and expected gain: every decode step moves the next token GPU->CPU->GPU: the step's argmax (GPU kernel, enqueued after the graph replay so already on the GPU critical path) -> tolist (D2H) -> postprocess on CPU -> the NEXT step's prepare_decode H2D's the token back into tbuf_tok (~22us) and the engine re-dispatches torch.argmax every step (~17-29us CPU). If the argmax is captured as the LAST op of each decode graph, writing the next step's tokens directly into tbuf_tok, the GPU cost is UNCHANGED (the kernel was already on the stream after the replay) but the per-step argmax dispatch and the token H2D vanish, and the engine reads the tokens from tbuf_tok after the wait. Expected ~40-50us CPU/step saved -> TPOT ~3.19->3.14.
- Profile evidence: debug/kernel_dump.py bs=1 decode: 459 launches/3724us. _step_split per-phase (exp 37): argmax dispatch ~17-29us, prepare token_only ~52us (incl. 22us token H2D + torch.tensor), tolist ~14us. 100-prompt screening: TPOT 3.17->3.14.
- Related earlier entries: [34]/[35] (graph-argmax, REVERTED neutral - they ADDED a D2H on the GPU critical path (+11us rtt) and kept the token H2D; this version keeps the tolist AFTER the wait (no critical-path D2H) and REMOVES the token H2D, so the CPU saving is ~2x theirs with zero GPU cost. [38] busy-poll changed the balance since 34/35), [41] (speculative prep - this makes the token_only fast path a true no-op), [31]/[36] (copy-front-ops - the graph already reads tbuf_tok at its start; the argmax at the end writes it, ordered on the stream).
- Change: src/run.py capture_cudagraph - each decode graph's capture ends with `tbuf_tok[:bs] = torch.argmax(outputs[:bs], dim=-1)`; prepare_decode token_only no longer writes tbuf_tok (returns buffer views). src/engine.py _step_split - drop the per-step `tokens_t = torch.argmax(...)`; read `next_token_ids = graph_vars["tbuf_tok"][:len(seqs)].tolist()` after the wait. commit 44dcac3.
- Correctness: PASS - bs=1 graph 0/37 vs /tmp/ref_s11_bs1.json (argmax is deterministic given the same logits; model math unchanged); run_loop split runA vs runB 0/37 (deterministic). tbuf_tok is int32 (argmax returns int64; values fit; postprocess treats ids as ints).
- Benchmark before (exp 43, 2 runs): 459.2 tok/s, TTFT 6.33/6.75 (p99 43.8/81.5), TPOT 3.19/3.19, ITL 3.19/3.19
- Benchmark after (2 runs): 459.43/459.71 tok/s, TTFT 6.84/6.72 (p50 5.03/5.05, p99 71.73/72.51), TPOT 3.15/3.15 (p99 3.38/3.39), ITL 3.16/3.16 (p99 3.48/3.48). output_tokens 96842/96910 (near-tie batch variance, normal).
- Gap to vLLM now: throughput 1.004x, TPOT 0.99x (3.15 vs vLLM 3.18 - first time Tokn is BELOW vLLM on TPOT), TTFT 0.38x (mean 6.78 vs 17.87, p99 72 vs unknown)
- Outcome: KEPT
- Why it worked: the token round-trip GPU->CPU->GPU was pure waste; the graph's argmax (already on the critical stream) doubles as the next step's input write, so the CPU only ever READS the token (once, off the critical path). TPOT 3.19->3.15 is the first clear TPOT win over vLLM; TTFT p99 also improved (~80->72, faster step -> faster slot turnover during bursts). Throughput stays at the ceiling.
- Worth retrying? The same pattern (a GPU-produced value that the CPU only consumes once) is fully exploited now. Next: the tolist (14us) is the last per-step token CPU cost and cannot be removed (scheduler needs CPU tokens).

## [48] 2026-10-10 19:10  Triton GEMV for the o/down gemvx (REVERTED - no win)
- Hypothesis and expected gain: the o/down projections at bs=1 are gemvx kernels, 56 calls/step = 862us (23% of the 3724us bs=1 decode, debug/kernel_dump.py), each ~15.4us reading 4MB = 235GB/s vs the 834 floor. If a Triton GEMV could reproduce cuBLAS's gemvx bit-for-bit AND stream the weight at higher bandwidth (e.g. 8us each), that is ~410us/step saved -> TPOT ~2.7.
- Profile evidence: debug/kernel_dump.py (bs=1 decode): std::enable_if (gemvx) 56 calls/862us. debug/gemv_triton.py.
- Related earlier entries: [39] (gemvx investigation - M=2 padding failed: o slower, down not bit-identical), [37] (plateau - decode GPU bit-identity-bound).
- Change: NONE. Investigation only (debug/gemv_triton.py, debug/gemv_tune.py).
- Correctness: NOT REACHED (no keeper).
- Findings: (1) F.linear (gemvx) == torch.mm bit-equal; (2) a Triton GEMV with per-64-chunk tree-sum + sequential chunk accumulation is bit-identical (maxd=0) for EVERY BLOCK_N in 1..64 and BLOCK_K in 32..256 - so cuBLAS's 2048-dot-product order is reproducible and even has slack (multiple trees give the same bf16 result); (3) BUT no Triton config beats F.linear: best 29.8us (BLOCK_N=8, BLOCK_K=64) vs 15.8us F.linear. More programs (BLOCK_N=1..16) hit poor coalescing (matrix-vector rows stride K apart); larger BLOCK_N underutilizes (16-32 CTAs < 108 SMs). cuBLAS's gemvx at 235GB/s is near-optimal for a GEMV.
- Outcome: REVERTED (no win)
- Why it failed: Triton GEMV is inherently memory-pattern-poor for row-major matrix-vector products; cuBLAS's gemvx is already at ~235GB/s (3.5x floor is inherent to the GEMV access pattern, not a cuBLAS deficiency). No bit-identity-safe faster GEMV exists on this GPU.
- Worth retrying? NO. The gemvx cost is inherent; the only escape would be restructuring the model math (e.g. keeping o/down in a larger batched GEMM), which is not bit-identity-safe.

## [49] 2026-10-10 19:30  Extend use_fast_flash above ctx=256 (REVERTED - unsafe, data-dependent)
- Hypothesis and expected gain: the ITL p99 gap to vLLM (3.48 vs 3.39) is the slowest decode steps (high bs + ctx>256 splitkv flash, ~400us penalty at bs=5). If num_splits=1 were bit-identical at ctx 257-369 (the benchmark range), raising the use_fast_flash threshold would cut the splitkv penalty for a slice of steps.
- Profile evidence: debug/splitkv_threshold.py (real cache layout, random KV): ns=1 vs ns=0 diverges at EVERY seqlen 250..400 (max|d| ~9.8e-4, 225-500 of 2048 elements differ). This CONTRADICTS exp 14's <=256 bit_eq finding for REAL model data -> the ns1/ns0 identity is DATA-DEPENDENT (structured real KV rounds identically at <=256 in exp 14's test, random KV does not even at 250).
- Related earlier entries: [14] (hybrid graphs, <=256 safe with real data), [32] (forced single-pass FAILED gate 4/37).
- Change: NONE. Investigation only.
- Correctness: NOT REACHED (no keeper). Even if a threshold test passed on the gate's data, the benchmark's ctx 257-369 is NOT covered by the gate (48-token gate prompts vs 474-token prefix prompts), so a data-dependent divergence there would flip benchmark tokens undetected. The 256 threshold stays.
- Outcome: REVERTED (unsafe)
- Why it failed: the ns=1 vs ns=0 flash kernels are numerically DIFFERENT (different softmax/reduction schedules); whether they agree for a given (cache, query) depends on the data's rounding. Real structured KV at <=256 happens to agree (exp 14); random KV does not. Extending the threshold risks silent token flips in the benchmark's 257-369 range.
- Worth retrying? NO. The splitkv penalty at ctx>256 is fundamental; the ITL p99 gap is blocked by the flash numerics.

## [50] 2026-10-10 19:45  Submission-path tokenization cost on the event loop (INVESTIGATION)
- Context: add_async_request runs apply_chat_template + tokenizer synchronously on the asyncio event loop (submission coroutine), which could block the run_loop's postprocess and inflate TPOT/ITL for the affected steps. Measuring the actual cost to decide if it's a lever.
- Outcome: see Details entry - measured, decided.

## [50] 2026-10-10 19:45  Submission-path tokenization cost (INVESTIGATION - not a lever)
- Context: add_async_request runs apply_chat_template + tokenizer synchronously on the asyncio event loop, which could block the run_loop's step postprocess and inflate TPOT/ITL for affected steps.
- Profile evidence: debug/tokenizer_cost.py (real 400 questions): template 114.7us mean, tokenizer 175.9us mean, tolist 7.7us -> TOTAL 298.3us mean (p50 203.7us, p99 547.4us) per request. Over 400 requests = 0.12s of event-loop blocking out of 210s.
- Outcome: NOT a lever. 0.12s total blocking is negligible; per-request blocking (<0.55ms) overlaps the ~3ms GPU wait so it delays a step postprocess only in rare overlaps, and the effect is <1% of steps by <0.5ms. No change.

## [51] 2026-10-10 19:50  Reuse persistent CUDA event for the GPU-wait poll (REVERTED - neutral)
- Hypothesis and expected gain: the decode step creates a fresh torch.cuda.Event() every step (~12us driver create call on the exposed CPU path, splitprof "record" ~20us). Reusing one event is safe (every step fully waits before the next record) and should save ~10-12us/step.
- Profile evidence: splitprof "record" phase ~20us (incl. reset_context + Event() + record).
- Related earlier entries: [35] (noted "a fresh torch.cuda.Event() each step is ~1us, not worth pooling" - that was before the busy-poll [38] changed the balance; the splitprof record phase is ~20us now).
- Change: src/engine.py - self._ev = torch.cuda.Event() in __init__, reused in _step_split. REVERTED (git checkout).
- Correctness: PASS - bs=1 graph 0/37 (CPU-only change).
- Benchmark before (exp 47): 459.43/459.71 tok/s, TTFT 6.84/6.72 (p99 71.7/72.5), TPOT 3.15/3.15, ITL 3.16/3.16 (p99 3.48/3.48)
- Benchmark after (2 runs): 459.75/459.70 tok/s, TTFT 6.77/6.66 (p99 70.3/72.0), TPOT 3.16/3.15, ITL 3.16/3.15 (p99 3.69/3.50)
- Outcome: REVERTED (neutral - no measurable gain, one ITL-p99 outlier run 3.69)
- Why it failed: the ~10us/step saving is below the ~40us benchmark noise; TPOT 3.16/3.15 is identical to exp 47's 3.15/3.15. Reverted to keep the tree at the clean exp-47 state per the neutral-change discipline.
- Worth retrying? NO - below noise.

## [52] 2026-10-10 20:00  Force per-bs num_splits on splitkv decode graphs (ABANDONED - sub-noise, fragile)
- Hypothesis and expected gain: the splitkv flash at ctx>256 costs ~13.5us/call more than single-pass at bs=5 (the ITL p99 driver, 3.48 vs vLLM 3.39). If an explicit num_splits that matches the ns=0 heuristic's split count is bit-identical AND avoids heuristic overhead, forcing it could shave the slowest steps.
- Profile evidence: debug/num_splits_test.py + num_splits_all.py: for each bs, the ns values bit-identical to ns=0 across ctx 257-369 are {bs=1,2,3: 4,8; bs=4,5: 2,3}; bs=1 ns=8 stays bit-identical to ctx 522 (the gate's prefix-prompt range); bs=4 ns=3 = 111.66 vs ns=0 113.85 (2.19us/call), bs=5 ns=3 = 124.03 vs 125.40 (1.37us/call), bs=1 ns=8 = 103.96 vs 105.12 (1.17us/call). The speedup vs ns=0 is only 0.6-2.2us/call because the explicit matching ns is STILL a splitkv kernel - only the non-identical single-pass (ns=1, 112 vs 125.5 at bs=5) recovers the ~13us/call penalty.
- Related earlier entries: [14] (hybrid graphs), [32] (single-pass FAILED), [49] (data-dependence of ns1/ns0).
- Change: NONE (abandoned).
- Correctness: NOT REACHED (no keeper). Bit-identity verified with torch.equal on RANDOM KV over the full benchmark grid (bs 1-5, ctx 257-369, and bs=1 to 522), but the exp 14/49 lesson is that flash bit-identity is DATA-DEPENDENT - a divergence on real model KV at an untested shape would flip benchmark tokens silently (the gate's multi-seq prompts are short; the benchmark's bs=4/5 at ctx>369 is untested).
- Outcome: ABANDONED (sub-noise gain, fragile per-bs mapping, data-dependent bit-identity risk)
- Why it failed: the bit-identical explicit-ns options are all splitkv kernels within ~2us of each other; the only fast option (single-pass, ~13us/call) is not bit-identical. The recoverable gain is ~5-10us/step average (below the ~40us noise) and ~0.04ms on ITL p99 (partial closure of the 0.09ms gap), not worth the fragility.
- Worth retrying? NO unless a later flash-attn version changes the splitkv overhead structure.

## Backlog

- The benchmark is arrival-bound; throughput ~460 tok/s is ~99.9% of the ceiling (last arrival 210.4s + 0.6s drain). Further decode gains mostly improve TPOT/ITL, not the headline throughput.
- TPOT 3.19 vs vLLM 3.18 (after exp 43): the gap is ~0.01ms = run-to-run noise. The decode GPU replay is bit-identity-bound [37]; the exposed CPU is below the ~40us noise [40]. Session-10 confirmed the remaining structural ideas are dead ends:
  - Overlap prefill with decode on a side stream [42, REVERTED]: the prefill and decode are BOTH memory-bound on the SAME 882MB weights, so concurrency just slows each down (waiting request's prefill went 3.3->5-6ms, TTFT mean +61%). The decode stall removal only helps the unreported p999 (debug/itl_audit.py: prefill stalls are 0.45% of ITLs, above p99).
  - Prefill GEMM speedup (TTFT is prefill-dominated, p50 ~5ms): the prefill GEMMs are ~5x below the 834GB/s floor (A30 bf16 peak measured ~80 TFLOPS; small-M cuBLAS inefficiency). Any replacement (Triton/CUTLASS GEMM, M-padding, config change) breaks bit-identity -> fails the bs=1 gate. Blocked.
  - TTFT p99 (~80-100ms) is burst queueing (Poisson bursts up to 15 arrivals in 3s vs 5 slots, seed 0) and is ADMISSION-bound: exp 43 made prefills ~100us faster and the p99 did not move (43.8/81.5 vs 79.9). Fixing it needs admitting the 6th+ request's prefill during a burst, which trades TTFT for TPOT (the wait must land somewhere; the keep rule rejects the TPOT p99 regression).
- Keep looking for: redundant ops in the prefill path (exp 43 found and removed one: the unconditional store_kvcache). The prefill still runs 85 mean + 57 convert_add + 57 rms_scale + 28 qk_convert + 28 scale_rotate + 28 flash + 28 silu kernels - all necessary or bit-identity-blocked.
- Full RMSNorm var fusion (fold pow+mean into Triton): BLOCKED - no Triton config matches torch's mean_kernel for the norm shapes; retry only with exact reduction matching.
- Background multi-seq prefill capture: on a cache-miss prefill, run EAGER and capture the graph on a side stream/thread concurrently so repeated shapes replay at ~8ms. Not needed for the benchmark (its 3 multi-seq shapes never repeat).
- GPU-resident token passing: the scheduler still needs tokens on CPU for postprocess/hashing/finish detection, so the D2H can't be eliminated; benefit would be hiding the D2H behind the GPU (exp 34/35 found neutral).

## [loop] 2026-10-09 13:07  Session 1 ended without a status line
- Outcome: SESSION CRASHED or cut off. Transcript: debug/opencode_runs/iter_001_20261009_124714.log
- Any experiment in progress was not finished. Check the transcript before retrying it.

## [loop] 2026-10-10 04:57  Session 3 ended without a status line
- Outcome: SESSION CRASHED or cut off. Transcript: debug/opencode_runs/iter_003_20261010_025213.log
- Any experiment in progress was not finished. Check the transcript before retrying it.

## [loop] 2026-10-10 11:41  Session 6 ended without a status line
- Outcome: SESSION CRASHED or cut off. Transcript: debug/opencode_runs/iter_006_20261010_101621.log
- Any experiment in progress was not finished. Check the transcript before retrying it.
