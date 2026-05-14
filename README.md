# Tokn
LLM Inference server
Developing my own LLM Inference server like vLLM. 

## Right now it supports
1. Online and Offline mode.
2. KV Caching
3. Multiple requests processing.
4. Scheduler to schedule requests.
5. Seperate prefill and Decode.

## Things are coming

1. Prefix Caching
2. Continuous batching
3. Speculative decoding
4. Quantization
5. CUDA Graphs

## Current State

![image](llm_benchmarks)

total request : 102
tokn : 457 tok/sec.
vLLM : 30,000 tok /sec.

