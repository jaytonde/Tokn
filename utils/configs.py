
import argparse

class ServerConfigs:
    def __init__(self, model=None, host="127.0.0.1", port=8080, dtype="float16", max_length=2048, device="cpu",
                 tensor_parallel_size=1, enforce_eager=False, max_num_seqs=5,
                 max_num_batched_tokens=1024, kv_cache_block_size=256, max_model_len=2048):
        self.model = model
        self.dtype = dtype
        self.max_length = max_length
        self.host = host
        self.port = port
        self.device = device
        self.tensor_parallel_size = tensor_parallel_size
        self.enforce_eager = enforce_eager
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.kv_cache_block_size = kv_cache_block_size
        self.max_model_len = max_model_len

    @classmethod
    def from_cli(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", required=True)
        parser.add_argument("--dtype", default="float16")
        parser.add_argument("--max_length", default=2048)
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8080)
        parser.add_argument("--device", default="cpu")
        parser.add_argument("--tensor_parallel_size", type=int, default=1)
        parser.add_argument("--enforce_eager", action="store_true")
        parser.add_argument("--max_num_seqs", type=int, default=5)
        parser.add_argument("--max_num_batched_tokens", type=int, default=1024)
        parser.add_argument("--kv_cache_block_size", type=int, default=256)
        parser.add_argument("--max_model_len", type=int, default=2048)
        args = parser.parse_args()
        return cls(model=args.model, host=args.host, port=args.port, device=args.device,
                   tensor_parallel_size=args.tensor_parallel_size, enforce_eager=args.enforce_eager,
                   max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens,
                   kv_cache_block_size=args.kv_cache_block_size, max_model_len=args.max_model_len)
    