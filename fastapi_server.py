from fastapi import FastAPI
from engine import Engine
from contextlib import asynccontextmanager
from pydantic import BaseModel
import uvicorn

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = Engine()
    yield

app = FastAPI(lifespan=lifespan)

class CompletionRequest(BaseModel):
    prompt: str


@app.post("/completions")
async def completions(req: CompletionRequest):
    response = app.state.engine.generate(req.prompt)
    return response


if __name__ == "__main__":
    uvicorn.run("fastapi_server:app", host="127.0.0.1", port=8080, reload=True, workers=2)