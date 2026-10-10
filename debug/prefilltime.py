"""Time prefill steps through the engine in graph mode (capture vs replay)."""
import asyncio
import json
import sys
import time

sys.path.insert(0, "/home/Tokn")

from utils.configs import ServerConfigs
from src.engine import Engine
from utils.sampling_params import SamplingParams


def main() -> None:
    cfg = ServerConfigs(
        model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda",
        max_length=2048, max_model_len=2048, tensor_parallel_size=1,
        enforce_eager=False, max_num_seqs=5, max_num_batched_tokens=1024,
    )
    engine = Engine(model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda",
                    max_model_len=2048, config=cfg)

    prompts = []
    with open("data/questions.jsonl") as f:
        for line in f:
            prompts.append(json.loads(line)["question"])
            if len(prompts) >= 60:
                break

    async def run():
        # submit all, let the run_loop handle scheduling; measure prefill step walls
        for p in prompts:
            await engine.add_async_request(p, SamplingParams(temperature=0.0, max_tokens=64))
        states = [engine.request_states[sid] for sid in list(engine.request_states)]

        prefill_walls = []
        guard = 0
        while not engine.scheduler.is_finished():
            seqs, is_prefill = engine.scheduler.schedule()
            if not seqs:
                break
            t0 = time.perf_counter()
            if is_prefill:
                with __import__("torch").inference_mode():
                    input_ids, positions = engine.RUN.prepare_prefill(seqs)
                    logits = engine.RUN.run_model(input_ids, positions, True)
                    engine.scheduler.postprocess(seqs, torch_argmax(logits), True)
                prefill_walls.append(((time.perf_counter() - t0) * 1000, len(seqs), input_ids.numel()))
            else:
                with __import__("torch").inference_mode():
                    input_ids, positions = engine.RUN.prepare_decode(seqs)
                    logits = engine.RUN.run_model(input_ids, positions, False)
                    engine.scheduler.postprocess(seqs, torch_argmax(logits), False)
            guard += 1
            if guard > 100000:
                raise RuntimeError("too many steps")

        n_graphs = len(engine.RUN.prefill_graphs)
        n_prefills = len(prefill_walls)
        if n_prefills:
            # distinguish captures: first occurrence of each shape is slower
            walls = [w for w, n, s in prefill_walls]
            walls.sort()
            print(f"prefill steps: {n_prefills}, distinct shapes captured: {n_graphs}")
            print(f"prefill wall ms: mean={sum(walls)/len(walls):.2f} p50={walls[len(walls)//2]:.2f} min={walls[0]:.2f} max={walls[-1]:.2f}")
            # estimate replay-only cost: exclude the slowest n_graphs (captures)
            slow = sorted(walls, reverse=True)[:n_graphs]
            fast = sorted(walls)[:-n_graphs] if len(walls) > n_graphs else []
            if fast:
                print(f"post-capture prefill wall ms: mean={sum(fast)/len(fast):.2f} (n={len(fast)})")

    def torch_argmax(logits):
        import torch
        return torch.argmax(logits, dim=-1).tolist()

    asyncio.run(run())


if __name__ == "__main__":
    main()
