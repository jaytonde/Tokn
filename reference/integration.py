import time
import torch
import asyncio
import threading
from dataclasses import dataclass

from reference.tokenizer_workers import TokenizerPool
from reference.engine import EngineConfig, GenerationRequest, InferenceEngine
from reference.api_design import ChatCompletionRequest, ChatCompletionResponse
from reference.fastapi_server import InferenceServer, RateLimiter, ServerMetrics


@dataclass
class ServerConfig:
    dtype: torch.dtype = torch.float16
    block_size: int = 16
    max_batch_size: int = 256
    max_tokens_per_batch: int = 4096
    max_seq_len: int = 4096
    tokenizer_workers: int = 4
    vocab_size: int = 32000
    hidden_dim: int = 4096
    rate_limit_rpm: int = 60
    host: str = "0.0.0.0"
    port: int = 8000

class LLMServer:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.running = False

        engine_config = EngineConfig(
            max_batch_size=config.max_batch_size,
            max_seq_len=config.max_seq_len,
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim
        )

        self.engine = InferenceEngine(engine_config)

        self.tokenizer_pool = TokenizerPool(
            num_workers=config.tokenizer_workers
        )

        self.server = InferenceServer(
            engine=self.engine,
            rate_limiter=RateLimiter(max_requests_per_minute=config.rate_limit_rpm)
        )

        self.engine_thread: threading.Thread | None = None

    def start(self):
        self.running = True
        self.engine_thread = threading.Thread(
            target=self.engine_loop,
            daemon=True #It will run in background. when main program ends it will stop automatically
        )
        self.engine_thread.start()

    def engine_loop(self):
        while self.running:
            if self.engine.pending_queue.empty():
                time.sleep(0.001)
                continue

            request = self.engine.pending_queue.get()
            result = self.engine.generate(request)
            self.engine.results[result.request_id] = result

    def stop(self):
        self.running = False
        self.tokenizer_pool.shutdown()
        if self.engine_thread is not None:
            self.engine_thread.join(timeout=0.5)

    async def handle_request(
            self,
            request: ChatCompletionRequest) -> ChatCompletionResponse:
        
        prompt_text = " ".join(m.content for m in request.messages)
        prompt_tokens = self.tokenizer_pool.tokenize(prompt_text)

        gen_request = GenerationRequest(
            request_id=self.engine.next_request_id,
            prompt_tokens=prompt_tokens,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )

        result = self.engine.generate(gen_request)

        return ChatCompletionResponse.create(
            model=request.model,
            content=f"generated {result.completion_tokens} tokens",
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            finish_reason=result.finish_reason
        )

    def get_status(self):
        return {
            "running": self.running,
            "config": {
                "max_batch_size": self.config.max_batch_size,
                "max_seq_len": self.config.max_seq_len,
                "vocab_size": self.config.vocab_size,
                "tokenizer_workers": self.config.tokenizer_workers,
                "rate_limit_rpm": self.config.rate_limit_rpm,
            },
            "engine_stats": self.engine.get_stats(),
            "tokenizer_stats": self.tokenizer_pool.get_stats(),
            "server_metrics": self.server.metrics.snapshot(),
        }


def explain_integration() -> str:
    return """
            LLM Server Integration

            Components wired together:
            1. InferenceEngine: runs model forward passes
            2. TokenizerPool: parallel text-to-token conversion
            3. InferenceServer: async HTTP handling
            4. RateLimiter: per-client request throttling
            5. ServerMetrics: monitoring and observability

            Startup sequence:
            1. Load model onto GPU
            2. Initialize KV cache (sized from model config)
            3. Create scheduler
            4. Start tokenizer worker pool
            5. Start engine thread (daemon)
            6. Start FastAPI event loop

            Request lifecycle:
            HTTP -> tokenize (async) -> schedule -> generate (GPU) -> stream back

            Configuration:
            - ServerConfig bundles all tunable parameters
            - Block size: memory fragmentation tradeoff
            - Max batch size: concurrent request limit
            - Max tokens per batch: compute per step limit
            - Tokenizer workers: CPU parallelism for encoding

            Production deployment:
            uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
            Single worker because GPU is the bottleneck.
            Add nginx reverse proxy and Prometheus for production.
            """

if __name__ == "__main__":
    print(explain_integration())

    print("\n" + "=" * 60)
    print("LLM Server Integration Demo")
    print("-" * 60)

    config = ServerConfig(
        vocab_size=1000,
        max_batch_size=8,
        tokenizer_workers=2,
    )

    server = LLMServer(config)
    print(f"Server created with config:")
    print(f"  max_batch_size: {config.max_batch_size}")
    print(f"  tokenizer_workers: {config.tokenizer_workers}")
    print(f"  vocab_size: {config.vocab_size}")

    server.start()
    print(f"\nServer started, engine thread running")

    status = server.get_status()
    print(f"Status: running={status['running']}")
    print(f"Engine stats: {status['engine_stats']}")
    print(f"Tokenizer stats: {status['tokenizer_stats']}")

    server.stop()
    print(f"\nServer stopped")
