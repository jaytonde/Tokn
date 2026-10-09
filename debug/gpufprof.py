"""torch.profiler breakdown of one decode step (graph path)."""
import argparse, asyncio, sys, json
sys.path.insert(0, "/home/Tokn")
import torch
from utils.configs import ServerConfigs
from src.engine import Engine
from utils.sampling_params import SamplingParams

def load_questions(path, n):
    out=[]
    with open(path) as f:
        for line in f:
            out.append(json.loads(line)["question"])
            if len(out)>=n: break
    return out

def main():
    cfg = ServerConfigs(model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda",
        max_length=2048, max_model_len=2048, tensor_parallel_size=1,
        enforce_eager=False, max_num_seqs=5, max_num_batched_tokens=1024)
    engine = Engine(model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda",
        max_model_len=2048, config=cfg)
    prompts = load_questions("data/questions.jsonl", 30)

    async def run():
        for p in prompts:
            await engine.add_async_request(p, SamplingParams(temperature=0.0, max_tokens=32))
        # run to decode
        steps = 0
        while not engine.scheduler.is_finished():
            engine.step()
            steps += 1
            if steps > 200: break
        print(f"steps to drain: {steps}", flush=True)
        # now profile decode steps
        from torch.profiler import profile, ProfilerActivity
        for _ in range(3):
            engine.step()
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
            for _ in range(5):
                engine.step()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30), flush=True)
    asyncio.run(run())

if __name__ == "__main__":
    main()
