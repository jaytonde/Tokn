import sys
sys.path.insert(0, "/home/Tokn")
import torch
from flash_attn import flash_attn_with_kvcache

torch.manual_seed(0)
dev = "cuda"
dtype = torch.bfloat16
num_blocks = 8
block_size = 256
num_kv_heads = 8
head_dim = 128
cache_shape = (2, 1, num_blocks, block_size, num_kv_heads, head_dim)
kvcache = torch.zeros(cache_shape, device=dev, dtype=dtype)
kvcache[0, 0].normal_()
kvcache[1, 0].normal_()
scale = head_dim ** -0.5
q = torch.randn(1, 16, head_dim, device=dev, dtype=dtype) * 0.1

def flash(cache_seqlen, bt):
    cache_seqlens = torch.tensor([cache_seqlen], device=dev, dtype=torch.int32)
    return flash_attn_with_kvcache(
        q.unsqueeze(1), kvcache[0, 0], kvcache[1, 0],
        cache_seqlens=cache_seqlens, block_table=bt,
        softmax_scale=scale, causal=True,
    ).squeeze(1)

def main():
    for seqlen in [256, 300, 494, 512, 700]:
        nblocks = (seqlen + 255) // 256
        bt_eager = torch.zeros(1, 8, device=dev, dtype=torch.int32)
        bt_eager[:1, :nblocks] = torch.arange(nblocks, device=dev, dtype=torch.int32)
        # graph: capture+replay with same padded (1,8) table
        bt_buf = torch.zeros(1, 8, device=dev, dtype=torch.int32)
        bt_buf.copy_(bt_eager)
        lens_buf = torch.zeros(1, device=dev, dtype=torch.int32)
        lens_buf[0] = seqlen
        out = torch.zeros(1, 16, head_dim, device=dev, dtype=dtype)
        seqlens_buf = torch.zeros(1, device=dev, dtype=torch.int32)
        seqlens_buf[0] = seqlen
        def f():
            return flash_attn_with_kvcache(
                q.unsqueeze(1), kvcache[0, 0], kvcache[1, 0],
                cache_seqlens=seqlens_buf, block_table=bt_buf,
                softmax_scale=scale, causal=True,
            ).squeeze(1)
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            out[:] = f()
            with torch.cuda.graph(g):
                out[:] = f()
            g.replay()
        torch.cuda.synchronize()
        ref = flash(seqlen, bt_eager)
        md = (ref - out).abs().max().item()
        print(f"seqlen={seqlen}: padded-eager vs graph(padded) max|d|= {md:.5g} bit_eq={torch.equal(ref, out)}", flush=True)

if __name__ == "__main__":
    main()
