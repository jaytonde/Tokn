
import argparse

class ServerConfigs:
    def __init__(self, model=None, host="127.0.0.1", port=8080, dtype="float16", max_length=2048, device="cpu"):
        self.model = model
        self.dtype = dtype
        self.max_length = max_length
        self.host = host
        self.port = port
        self.device = device

    @classmethod
    def from_cli(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", required=True)
        parser.add_argument("--dtype", default="float16")
        parser.add_argument("--max_length", default=2048)
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8080)
        parser.add_argument("--device", default="cpu")
        args = parser.parse_args()
        return cls(model=args.model, host=args.host, port=args.port)
    