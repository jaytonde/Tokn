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

## Backlog

- Fused MLP (gate_up + silu + down in one kernel) and fused QKV GEMM: would cut ~1ms/step of the decode GEMM time. BLOCKED by the bs-1 bit-identity gate (Triton GEMM accumulation differs from cuBLAS -> flips near-tie tokens). Could be retried IF the gate is satisfied via matching reduction, or with torch.compile fusions that preserve tokens.
- Full RMSNorm var fusion (fold pow+mean into Triton): ~0.5ms/step. BLOCKED: Triton sum reduction differs from torch's mean_kernel; no reshape/num_warps combination matched (debug/redmatch.py). Retry only with exact reduction matching.
- Reduce run_loop per-step event-loop overhead (~8.5us/step, ~0.28s over 400-request run) to shrink residual arrival delay. Small.
- Reduce TTFT tail (p99 ~600ms, 3.4x vLLM): admit prefills faster during bursts or interleave prefill+decode (mixed batching). Larger change.
- Flash attention splitkv path: DONE in [14] (single-pass for seqlen<=256 via dual-graph hybrid, TPOT -4.2%, TTFT -19%). Remaining: >256 contexts still use splitkv; could extend the fast path to >256 by re-capturing with a higher threshold IF the gate allowed it (it doesn't, without changing the reference).
- The benchmark is arrival-bound; throughput is near the ceiling (~461 tok/s theoretical). Further decode gains mostly improve TPOT/ITL and the tail, not the headline throughput.

## [loop] 2026-10-09 13:07  Session 1 ended without a status line
- Outcome: SESSION CRASHED or cut off. Transcript: debug/opencode_runs/iter_001_20261009_124714.log
- Any experiment in progress was not finished. Check the transcript before retrying it.

## [loop] 2026-10-10 04:57  Session 3 ended without a status line
- Outcome: SESSION CRASHED or cut off. Transcript: debug/opencode_runs/iter_003_20261010_025213.log
- Any experiment in progress was not finished. Check the transcript before retrying it.
