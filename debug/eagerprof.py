"""Profile the EAGER decode forward (model + lm_head) with torch.profiler to see
per-op GPU time at bs=5, short context.
"""
import asyncio, sys, json
sys.path.insert(0, "/home/Tokn")
import torch
from utils.configs import ServerConfigs
from src.engine import Engine
from utils.sampling_params import SamplingParams
from utils.context import get_context

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
    run = engine.RUN

    async def run_all():
        prompts = load_questions("data/questions.jsonl", 8)
        for p in prompts:
            await engine.add_async_request(p, SamplingParams(temperature=0.0, max_tokens=400))
        for _ in range(40):
            engine.step()
        # find a decode batch
        seqs, is_prefill = engine.scheduler.schedule()
        if is_prefill:
            engine.scheduler.postprocess(seqs, [], True)
            seqs, is_prefill = engine.scheduler.schedule()
        with torch.inference_mode():
            input_ids, positions = run.prepare_decode(seqs)
        bs = input_ids.size(0)
        print(f"bs={bs} len={len(seqs[0].token_ids)}", flush=True)
        from torch.profiler import profile, ProfilerActivity
        with torch.inference_mode():
            for _ in range(3):
                run.custom_model(input_ids, positions)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(5):
                    run.custom_model(input_ids, positions)
                torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25), flush=True)
    asyncio.run(run_all())

if __name__ == "__main__":
    main()
