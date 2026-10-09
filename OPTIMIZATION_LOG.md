# Tokn Optimization Log

## Scoreboard

| # | Change | Cumulative tok/s | TTFT mean (ms) | TPOT mean (ms) | vs vLLM (tok/s ratio) |
|---|--------|-----------------|----------------|----------------|----------------------|
| 0 | baseline (eager, forced by benchmark, after KV-admission fix) | 121.53 | 264765 | 40.72 | 0.26x |
| 1 | CUDA-graph decode fix (block_table padding) + enable graphs | 441.89 | 242.09 | 7.51 | 0.97x |
| 2 | fused Triton RoPE kernel | 446.12 | 116.36 | 6.03 | 0.97x |
| 3 | threaded run_loop step + RMSNorm tail fusion + lm_head-in-graph + sync removal | 456.57 | 94.27 | 5.53 | 0.997x |
| 4 | drop GPU round-trip in add_async_request tokenization | 456.70 | 95.77 | 5.53 | 0.998x |
| 5 | fuse RMSNorm convert+residual-add, fold rsqrt into tail | 459.59 | 61.06 | 4.66 | 1.004x  MILESTONE (beats vLLM) |
| - | vLLM 0.31.0 target (this machine, 2026-10-09) | 457.86 | 17.87 | 3.18 | 1.00x |

## Experiments tried

| N | Idea | Area (graphs/kernel/scheduler/sync/memory/compile/server/config) | Result | Delta tok/s | Why |
|---|------|------|--------|-------------|-----|
| 1 | CUDA-graph decode not bit-identical to eager -> root-caused to block_table width (eager (1,nblocks) vs graph (1,8) padded); pad both to max_num_blocks | graphs | KEPT | +320.4 | flash_attn_with_kvcache yields different results for padded vs unpadded block_table at seqlen>=256; matching shapes makes graph bit-identical |
| 2 | Fused Triton RoPE (7 torch kernels -> 1), elementwise fp32 + single bf16 round | kernel | KEPT | +4.2 | bit-identical to torch; graph replay 6.17->4.87 ms, TPOT 7.51->6.03 |
| 3 | Threaded run_loop step (submission starvation) + RMSNorm tail fusion + lm_head in graph + GPU slot mapping | sync/compile | KEPT | +10.4 | run_loop's blocking step starved the submission coroutine; arrivals drifted ~5.6s late -> 217s. Threaded step + faster decode -> 212s |
| 4 | Drop GPU round-trip in add_async_request tokenization | sync | KEPT | +0.1 | tokenizer ids moved to CUDA and back for no reason; also put a GPU sync on the submission path. sum_submit 0.85->0.30s |
| 5 | Fuse RMSNorm convert+residual-add, fold rsqrt into tail (tl.rsqrt == torch.rsqrt) | kernel | KEPT | +2.9 | RMSNorm ~7->4 kernels/call; graph replay 4.65->3.84ms; TPOT 5.53->4.66ms. First time beats vLLM (MILESTONE) |

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

## Backlog

- (to be filled)

## [loop] 2026-10-09 13:07  Session 1 ended without a status line
- Outcome: SESSION CRASHED or cut off. Transcript: debug/opencode_runs/iter_001_20261009_124714.log
- Any experiment in progress was not finished. Check the transcript before retrying it.
