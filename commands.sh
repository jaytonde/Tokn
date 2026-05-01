#To run the server
python serve.py --model Qwen/Qwen2.5-0.5B --dtype bf16 --max_length 2048

python serve.py --model Qwen/Qwen3-0.6B --dtype bf16 --max_length 2048 --device cuda