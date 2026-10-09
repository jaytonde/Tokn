
python serve.py --model Qwen/Qwen2.5-0.5B --dtype bf16 --max_length 2048

python serve.py --model Qwen/Qwen3-0.6B --dtype bf16 --max_length 2048 --device cuda:1

# --------------------------------------------------------------------------- #
# Benchmark commands (continuous batching + staggered arrivals, request-rate=2)
# --------------------------------------------------------------------------- #

# tokn  (uses inference_optimization conda env, clears conflicting bashrc CUDA paths)
cd /home/Tokn
env -u LD_LIBRARY_PATH -u CUDA_HOME -u CUDA_PATH \
	PYTHONPATH=/home/Tokn \
	/home/Tokn/.venv/bin/python debug/benchmark.py \
	--framework tokn \
	--model Qwen/Qwen3-0.6B \
	--dtype bf16 \
	--device cuda \
	--max-tokens 256 \
	--max-model-len 2048 \
	--questions data/questions.jsonl \
	--num-prompts 400 \
	--request-rate 2


# vLLM  (uses vllm_env conda env)
/data1/home/mahek.shah/conda-envs/vllm_env/bin/python benchmark.py \
	--framework vllm \
	--model Qwen/Qwen3-0.6B \
	--dtype bf16 \
	--device cuda \
	--max-tokens 256 \
	--max-model-len 2048 \
	--questions data/questions.jsonl \
	--num-prompts 400 \
	--request-rate 2


#Tensor Parallelism
NCCL_P2P_DISABLE=1 python serve.py --model Qwen/Qwen3-0.6B --dtype bf16 \
  --max_length 2048 --device cuda --tensor_parallel_size 2 \
  --host 127.0.0.1 --port 8080 --enforce_eager