"""Time parts of prepare_decode for a decode step."""
import asyncio, sys, json, time
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
    run = engine.RUN
    async def run_all():
        for p in load_questions("data/questions.jsonl", 8):
            await engine.add_async_request(p, SamplingParams(temperature=0.0, max_tokens=400))
        for _ in range(40):
            engine.step()
        seqs, is_prefill = engine.scheduler.schedule()
        if is_prefill:
            engine.scheduler.postprocess(seqs, [], True)
            seqs, is_prefill = engine.scheduler.schedule()
        t_list = t_tt = t_stack = t_slot = 0.0
        N = 200
        for _ in range(N):
            # measure prepare_decode parts
            t0 = time.perf_counter()
            input_ids_l = []
            positions_l = []
            context_lens_l = []
            bts = []
            for seq in seqs:
                pos = len(seq.token_ids) - 1
                input_ids_l.append(seq.last_token)
                positions_l.append(pos)
                context_lens_l.append(len(seq.token_ids))
                bt_row = seq.block_table[0]
                if bt_row.size(0) < run.max_num_blocks:
                    pad = torch.full((run.max_num_blocks - bt_row.size(0),), -1,
                                     dtype=torch.int32, device=run.device)
                    bt_row = torch.cat([bt_row, pad], dim=0)
                bts.append(bt_row)
            t_list += time.perf_counter() - t0
            t0 = time.perf_counter()
            device = torch.device(run.device)
            input_ids = torch.tensor(input_ids_l, dtype=torch.long, device=device)
            positions = torch.tensor(positions_l, dtype=torch.long, device=device)
            context_lens = torch.tensor(context_lens_l, dtype=torch.int32, device=device)
            t_tt += time.perf_counter() - t0
            t0 = time.perf_counter()
            bt = torch.stack(bts, dim=0)
            t_stack += time.perf_counter() - t0
            t0 = time.perf_counter()
            logical_ids = positions // run.block_size
            offsets = positions % run.block_size
            row_ids = torch.arange(positions.size(0), device=device)
            slot_mapping = (bt[row_ids, logical_ids] * run.block_size + offsets).to(torch.int32)
            t_slot += time.perf_counter() - t0
        print(f"prepare parts per step (bs={len(seqs)}): list {t_list/N*1000:.3f}ms "
              f"tensor-create {t_tt/N*1000:.3f}ms stack {t_stack/N*1000:.3f}ms slot {t_slot/N*1000:.3f}ms "
              f"total {(t_list+t_tt+t_stack+t_slot)/N*1000:.3f}ms", flush=True)
    asyncio.run(run_all())

if __name__ == "__main__":
    main()
