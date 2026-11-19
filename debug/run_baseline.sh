#!/bin/sh
# Baseline on real ShareGPT prompts, burst mode (all requests at once).
# Each config runs in a fresh process. Results: results/<tag>.log (BENCH_RESULT line)
# and results/<tag>.outputs.jsonl (generated texts, for correctness diffs).
set -u
cd "$(dirname "$0")/.."
TOKN_PY=${TOKN_PY:-/home/Tokn/.venv/bin/python}
VLLM_PY=${VLLM_PY:-/home/Tokn/.venv-vllm/bin/python}
RUN=${RUN:-1}
CONFIGS=${CONFIGS:-"tokn_eager tokn_graphs vllm"}
COMMON="--model Qwen/Qwen3-0.6B --dtype bf16 --device cuda --max-tokens 256 --max-model-len 2048 --questions data/sharegpt_400.jsonl --num-prompts 400 --emit-json"
mkdir -p results
export PYTHONPATH=.

bench() {
    tag=$1; py=$2; shift 2
    case " $CONFIGS " in *" ${tag%_r*} "*) ;; *) return ;; esac
    echo "=== $tag start $(date -u +%H:%M:%S)"
    env -u LD_LIBRARY_PATH -u CUDA_HOME -u CUDA_PATH \
        $py debug/benchmark.py $COMMON --save-outputs results/$tag.outputs.jsonl "$@" \
        > results/$tag.log 2>&1
    echo "=== $tag exit=$? $(date -u +%H:%M:%S)"
    grep "^BENCH_RESULT" results/$tag.log || tail -20 results/$tag.log
}

bench tokn_eager_r$RUN  $TOKN_PY --framework tokn --tokn-enforce-eager
bench tokn_graphs_r$RUN $TOKN_PY --framework tokn
bench vllm_r$RUN        $VLLM_PY --framework vllm
