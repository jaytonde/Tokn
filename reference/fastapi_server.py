import asyncio
import json
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

from reference.api_design import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    StreamChoice,
    StreamDelta
)

from reference.engine import EngineConfig, GenerationRequest, InferenceEngine

@dataclass
class RateLimiter:
    max_requests_per_minute: int = 60
    window_seconds: float = 60.0
    timestamps: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def allow(self, client_id: str) -> bool:
        now = time.time()
        cutoff = now - self.window_seconds

        self.timestamps[client_id] = [t for t in self.timestamps[client_id] if t > cutoff]

        if len(self.timestamps[client_id]) >= self.max_requests_per_minute:
            return False
        
        self.timestamps[client_id].append(now)

        return True

@dataclass
class ServerMetrics:
    total_requests: int = 0
    active_requests: int = 0
    total_tokens_generated: int = 0
    total_prompt_tokens: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record_requests(self):
        with self.lock:
            self.total_requests += 1
            self.active_requests += 1

    def complete_request(self, prompt_tokens: int, completion_tokens: int):
        with self.lock:
            self.active_requests -= 1
            self.total_prompt_tokens += prompt_tokens
            self.total_tokens_generated += completion_tokens

    def record_error(self):
        with self.lock:
            self.active_requests -= 1
            self.errors += 1

    def snapshot(self) -> dict:
        with self.lock:
            uptime = time.time() - self.start_time
            return {
                "total_requests": self.total_requests,
                "active_requests": self.active_requests,
                "total_tokens_generated": self.total_tokens_generated,
                "total_prompt_tokens": self.total_prompt_tokens,
                "errors": self.errors,
                "uptime_seconds": uptime,
                "requests_per_seconds": self.total_requests / uptime if uptime > 0 else 0
            }

@dataclass
class InferenceServer:
    def __init__(self, engine: InferenceEngine, rate_limiter: RateLimiter | None = None):
        self.engine = engine
        self.rate_limiter = rate_limiter or RateLimiter()
        self.metrics = ServerMetrics()

        self.output_queues: dict[str, asyncio.Queue] = {}
        self.loop: asyncio.AbstractEventLoop | None = None #TO-DO

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    def emit_token(self, request_id: str, token: int | None):
        if self.loop is None:
            return
        queue = self.output_queues.get(request_id)

        if queue is not None:
            self.loop.call_soon_threadsafe(queue.put_nowait, token)

    async def submit_request(self, request: ChatCompletionRequest, prompt_tokens: list[int]) -> str:
        request_id : f"req-{uuid.uuid4().hex[:12]}"
        self.output_queues[request_id] = asyncio.Queue()
        self.metrics.record_requests()

        gen_request = GenerationRequest(
            request_id=hash(request_id) & 0xFFFFFFFF,
            prompt_tokens=prompt_tokens,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop_token_ids=[]
        )


        def run_generation():
            try:
                for token in self.engine.generate_stream(gen_request):
                    self.emit_token(request_id, token)
                self.emit_token(request_id, None)
            except Exception:
                self.emit_token(request_id, None)

        threading.Thread(target=run_generation, daemon=True).start()
        return request_id
    
    async def get_tokens(self, request_id: str):
        queue = self.output_queues[request_id]
        while True:
            token = await queue.get()
            if token is None:
                break
            yield token

        del self.output_queues[request_id]

    async def cancel_request(self, request_id: str):
        queue = self.output_queues.get(request_id)
        if queue is not None:
            await queue.put(None)

    def format_sse_chunk(
        self,
        request_id: str,
        model: str,
        token_text: str,
        finish_reason: str | None = None,
    ) -> str:
        chunk = ChatCompletionChunk(
            id=request_id,
            object="chat.completion.chunk",
            created=int(time.time()),
            model=model,
            choices=[
                StreamChoice(
                    index=0,
                    delta=StreamDelta(content=token_text),
                    finish_reason=finish_reason,
                )
            ],
        )
        return chunk.to_sse()

    


def validate_request(request: ChatCompletionRequest) -> str | None:
    if not request.messages:
        return "messages must not be empty"
    if request.temperature < 0 and request.temperature > 2.0:
        return "temperature must be between 0 and 2.0"
    if request.top_p < 0 and request.top_p > 1.0:
        return "top_p must be between 0 and 1.0"
    if request.max_tokens < 1:
        return "max tokens must be at leat 1"
    if request.stop is not None and len(request.stop) > 4: #TO-DO why 4?
        return "stop must have at most 4 sequences"
    
    return None

def create_error_response(status_code: int, message: str) -> dict:
    return {
        "error" : {
            "message": message,
            "type": "invalid_request_error",
            "code": status_code
        }
    }

def explain_fastapi_server() -> str:
    return """
            FastAPI Server for LLM Inference

            Architecture:
            - Async HTTP handling with FastAPI
            - Engine runs in separate thread
            - Async queues bridge thread boundary
            - SSE streaming for token-by-token delivery

            Request flow:
            1. HTTP request arrives
            2. Rate limiter checks client quota
            3. Request validated (params, format)
            4. Prompt tokenized (async, in pool)
            5. Request submitted to engine
            6. Tokens stream back via async queue
            7. SSE chunks sent to client

            Key patterns:
            - call_soon_threadsafe: engine thread -> event loop
            - asyncio.Queue per request: decouple generation from HTTP
            - StreamingResponse: yield SSE chunks as generated
            - request.is_disconnected(): detect dropped clients

            Rate limiting:
            - Sliding window per client
            - Reject with HTTP 429 when exceeded
            - Prevents single client from monopolizing GPU

            Metrics:
            - Total/active requests
            - Tokens generated
            - Error count
            - Uptime and throughput

            Error handling:
            - Validation errors: 400 with specific message
            - Rate limit: 429 with retry hint
            - Engine errors: 500 with safe message
            - Disconnect: cancel request, free resources
            """



if __name__ == "__main__":
    print(explain_fastapi_server())

    print("\n" + "=" * 60)
    print("FastAPI Server Demo")
    print("-" * 60)

    config = EngineConfig(vocab_size=1000)
    engine = InferenceEngine(config)
    server = InferenceServer(engine)

    limiter = RateLimiter(max_requests_per_minute=10)
    print(f"Rate limit check: {limiter.allow('client-1')}")
    print(f"Rate limit check: {limiter.allow('client-1')}")

    request_data = {
        "model": "qwen-30b",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 50,
        "temperature": 0.8,
    }
    request = ChatCompletionRequest.from_dict(request_data)
    error = validate_request(request)
    print(f"Validation: {'passed' if error is None else error}")

    bad_request = ChatCompletionRequest.from_dict({
        "model": "test",
        "messages": [],
        "temperature": 5.0,
    })
    error = validate_request(bad_request)
    print(f"Bad request validation: {error}")

    print(f"\nMetrics: {server.metrics.snapshot()}")