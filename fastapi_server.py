from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List

import asyncio




class CompletionRequest(BaseModel):
    prompts: List[str]


def build_app(engine):

    app = FastAPI()

    @app.on_event("startup")
    async def start_engine_loop():
        if engine.background_task is None:
            engine.background_task = asyncio.create_task(engine.run_loop())

    @app.post("/completions")
    async def completions(req: CompletionRequest):
        print(f"\n\nRequest received for with the prompt : {req.prompts}")

        tasks = [
            engine.generate_async(prompt)
            for prompt in req.prompts
        ]

        outputs = await asyncio.gather(*tasks)

        return {
            idx: output
            for idx, output in enumerate(outputs)
        }
    
    return app