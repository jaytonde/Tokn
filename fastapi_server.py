from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn




class CompletionRequest(BaseModel):
    prompt: str


def build_app(engine):

    app = FastAPI()

    @app.post("/completions")
    async def completions(req: CompletionRequest):
        response = engine.generate(req.prompt)
        return response
    
    return app