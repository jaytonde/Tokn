from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn




class CompletionRequest(BaseModel):
    prompt: str


def build_app(engine):

    app = FastAPI()

    @app.post("/completions")
    async def completions(req: CompletionRequest):
        print(f"\n\nRequest received for with the prompt : {req.prompt}")
        response = engine.custom_generate(req.prompt)
        return response
    
    return app