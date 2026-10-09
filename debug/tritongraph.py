"""Test whether two CUDA graphs containing the SAME Triton kernel interfere via
shared launch-parameter state.

Captures two graphs, each running store_kvcache-like Triton writes to different
cache slots with different values. Replays alternately and checks for
corruption of the first graph's results.

Usage: PYTHONPATH=/home/Tokn .venv/bin/python debug/tritongraph.py
"""

import sys

sys.path.insert(0, "/home/Tokn")

import torch
import triton
import triton.language as tl


@triton.jit
def store_kv_kernel(val_ptr, val_stride, cache_ptr, slot_mapping_ptr, D: tl.constexpr):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    offs = idx * val_stride + tl.arange(0, D)
    v = tl.load(val_ptr + offs)
    tl.store(cache_ptr + slot * D + tl.arange(0, D), v)


def main():
    dev = "cuda"
    D = 1024
    cache = torch.zeros(2048 * D, device=dev, dtype=torch.bfloat16)
    val_a = torch.arange(D, device=dev, dtype=torch.bfloat16) + 100
    val_b = torch.arange(D, device=dev, dtype=torch.bfloat16) + 200

    slot_buf = torch.zeros(1, device=dev, dtype=torch.int32)
    ga = torch.cuda.CUDAGraph()
    gb = torch.cuda.CUDAGraph()

    with torch.inference_mode():
        # graph A: write val_a to slot 10
        slot_buf[0] = 10
        store_kv_kernel[(1,)](val_a, D, cache, slot_buf, D)  # warmup
        with torch.cuda.graph(ga):
            store_kv_kernel[(1,)](val_a, D, cache, slot_buf, D)

        # graph B: write val_b to slot 20
        slot_buf[0] = 20
        store_kv_kernel[(1,)](val_b, D, cache, slot_buf, D)  # warmup
        with torch.cuda.graph(gb):
            store_kv_kernel[(1,)](val_b, D, cache, slot_buf, D)

        # replay A
        slot_buf[0] = 10
        ga.replay()
        torch.cuda.synchronize()
        a1 = cache[10 * D:11 * D].clone()

        # replay B
        slot_buf[0] = 20
        gb.replay()
        torch.cuda.synchronize()
        b1 = cache[20 * D:21 * D].clone()

        # replay A again
        slot_buf[0] = 10
        ga.replay()
        torch.cuda.synchronize()
        a2 = cache[10 * D:11 * D].clone()

        # also check B's target slot
        slot_buf[0] = 20
        gb.replay()
        torch.cuda.synchronize()
        b2 = cache[20 * D:21 * D].clone()

    print("A vs val_a:", torch.equal(a1, val_a))
    print("B vs val_b:", torch.equal(b1, val_b))
    print("A after B replay vs val_a:", torch.equal(a2, val_a))
    print("B after A replay vs val_b:", torch.equal(b2, val_b))
    # did replaying B clobber A's slot?
    print("A slot after B replay:", torch.equal(cache[10*D:11*D], val_a))
    print("B slot after A replay:", torch.equal(cache[20*D:21*D], val_b))


if __name__ == "__main__":
    main()
