
import uuid
import time
import json
import asyncio
from schema import CompletionResponse, CompletionRequest
from configs import ServerConfig

from fastapi import FastAPI


app = FastAPI()


from engine import Engine




class APIError:
    pass


class StreamingResponse:
    pass


class ContinuousBatchingScheduler:
    pass


class TokenizerPool:
    pass


class LLMServer:
    def __init__(self, model_path: str, config: ServerConfig):
        self.model = load_model(model_path, config.dtype)
        self.kv_cache = PagedKVCache(
            self.model.num_layers,
            self.model.num_kv_heads,
            self.model.head_dim
        )
        self.scheduler = ContinuousBatchingScheduler()
        self.tokenizerPool = TokenizerPool()

    def process_output():
        pass

    def engine_loop(self):
        while self.running:
            batch = self.scheduler.get_batch()

            if batch is None:
                time.sleep(0.001)
                continue

            outputs = self.model.forward(batch.input_ids)

            for i, request_id in enumerate(batch. request_ids):
                self.process_output(request_id, outputs.logits[i])



async def create_server(model_path: str, config: ServerConfig):
    """
    we created LLM server here and saves it in fastapi instances to access it across the endpoints
    """
    app = FastAPI()
    server = LLMServer(model_path, config)
    server.start()

    app.state.server = server
    app.state.loop = asyncio.get_event_loop()

    return app



class Server:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.request_queues = dict[str, asyncio.Queue] = {}


    async def submit_request(self, request: CompletionRequest) -> str:
        request_id = str(uuid.uuid4())
        self.request_queues[request_id] = asyncio.Queue()

        await self.engine.add_request(request_id, request.prompt)

        return request_id
    

    async def get_tokens(self, request_id: str):
        queue = self.request_queues[request_id]

        while True:
            token = await queue.get()
            if token is None:
                break
            yield token

    def emit_token(self, request_id: str, token: str):
        queue = self.output_queues[request_id]
        self.loop.call_soon_threadsafe(queue.put_nowwait, token)


async def stream_response(request_id: str, engine: Engine):
    async for token in engine.generate_stream(request_id):
        chunk = {
            "choices" : [{
                "delta" : {
                    "content" : token
                }
            }]
        }
        yield f"data: " f"{json.dumps(chunk)} \n\n"
    yield "data : [Done] \n\n"


def create_error_response(message: str, code: int) -> dict:
    return {
        "error" : {
            "message" : message,
            "code" : code
        }
    }


def validate_request(request : CompletionRequest):
    if not request.prompt:
        raise APIError("prompt cannot be empty", 400)
    if request.max_tokens < 1 or request.max_tokens > 4096:
        raise APIError("max_tokens out of range", 400)
    

def stream_tokens():
    pass

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """
    Endpoint to return response to the user
    """
    server = Server()
    request_id = await server.submit_request(request)

    if request.stream:
        return StreamingResponse(
            stream_tokens(server, request_id),
            media_type = "text/event-stream"
        )

    tokens = [t async for t in server.get_tokens(request_id)]

    return CompletionResponse(text = "".join(tokens))


@app.on_event("startup")
async def startup():
    """
    This function runs before FastAPI server starts.
    doemon process means it dies when main thread exits.
    """
    engine = Engine(model, scheduler)
    engine.set_event_loop(asyncio.get_event_loop())
    threading.Thread(
        target = engine_loop,
        daemon = True
    ).start()
    app.state.server = Server(engine)



