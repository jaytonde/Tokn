from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List




class CompletionRequest(BaseModel):
    prompts: List[str]


def build_app(engine):

    app = FastAPI()

    @app.post("/completions")
    async def completions(req: CompletionRequest):
        print(f"\n\nRequest received for with the prompt : {req.prompts}")
        response = engine.generate_v2(req.prompts)
        return response
    
    return app