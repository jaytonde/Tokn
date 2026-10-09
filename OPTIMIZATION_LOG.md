# Tokn Optimization Log

## Scoreboard

| # | Change | Cumulative tok/s | TTFT mean (ms) | TPOT mean (ms) | vs vLLM (tok/s ratio) |
|---|--------|-----------------|----------------|----------------|----------------------|
| 0 | baseline (eager, forced by benchmark, after KV-admission fix) | 121.53 | 264765 | 40.72 | 0.26x |
| - | vLLM 0.31.0 target (this machine, 2026-10-09) | 457.86 | 17.87 | 3.18 | 1.00x |

## Experiments tried

| N | Idea | Area (graphs/kernel/scheduler/sync/memory/compile/server/config) | Result | Delta tok/s | Why |
|---|------|------|--------|-------------|-----|
| (empty) | | | | | |

## Details

(append-only, newest at the bottom)

## Backlog

- (to be filled)

## [loop] 2026-10-09 13:07  Session 1 ended without a status line
- Outcome: SESSION CRASHED or cut off. Transcript: debug/opencode_runs/iter_001_20261009_124714.log
- Any experiment in progress was not finished. Check the transcript before retrying it.
