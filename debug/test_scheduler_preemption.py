"""
CPU-only stress test for lazy KV allocation + preemption (src/scheduler.py).

A fake model replaces the GPU. It writes (position, token) into a simulated
KV store at the slots given by each sequence's block table, and computes the
next token from what it reads back through the block table. So any wrong
block mapping, lost KV, or overwritten shared block changes the outputs.

Every sequence's output is compared against a reference generated directly
from its prompt (no scheduler, unlimited memory).

Usage:
    PYTHONPATH=. python debug/test_scheduler_preemption.py
"""

import random
import xxhash

from src.block_manager import BlockManager
from src.scheduler import Scheduler
from utils.sequence import Sequence

VOCAB = 50
EOS = 0


def next_token(history: list[int]) -> int:
    h = xxhash.xxh64(bytes(t % 256 for t in history)).intdigest()
    # ~2% EOS so lengths vary; never EOS on the first token.
    if len(history) > 1 and h % 50 == 0:
        return EOS
    return 1 + h % (VOCAB - 1)


def reference(prompt: list[int], max_tokens: int) -> list[int]:
    toks = list(prompt)
    out = []
    while len(out) < max_tokens:
        t = next_token(toks)
        toks.append(t)
        out.append(t)
        if t == EOS:
            break
    return out


class FakeModel:
    def __init__(self, bm: BlockManager):
        self.bm = bm
        self.kv = {}    # slot -> (position, token)

    def _slot(self, seq, pos):
        table = seq.block_table_ids
        block = table[pos // self.bm.block_size]
        return block * self.bm.block_size + pos % self.bm.block_size

    def _write(self, seq, pos):
        block = seq.block_table_ids[pos // self.bm.block_size]
        assert self.bm.blocks[block].ref_count == 1, f"write into shared block {block}"
        self.kv[self._slot(seq, pos)] = (pos, seq.token_ids[pos])

    def _read(self, seq, end):
        hist = []
        for p in range(end):
            pos, tok = self.kv[self._slot(seq, p)]
            assert pos == p, f"seq {seq.seq_id}: slot for pos {p} holds pos {pos}"
            hist.append(tok)
        assert hist == seq.token_ids[:end], f"seq {seq.seq_id}: KV content differs from tokens"
        return hist

    def run(self, seqs, is_prefill):
        out = []
        for seq in seqs:
            if is_prefill:
                start = seq.num_cached_tokens
                end = start + seq.num_scheduled_tokens
                for p in range(start, end):
                    self._write(seq, p)
            else:
                end = seq.num_tokens
                self._write(seq, end - 1)
            out.append(next_token(self._read(seq, end)))
        return out


def check_invariants(bm: BlockManager, sched: Scheduler, in_flight=()):
    holders = {}
    for seq in [*sched.running, *sched.waiting, *in_flight]:
        if getattr(seq, "block_table_ids", None) is None:
            continue
        for b in seq.block_table_ids:
            holders[b] = holders.get(b, 0) + 1
    for b, n in holders.items():
        assert bm.blocks[b].ref_count == n, f"block {b}: ref_count {bm.blocks[b].ref_count} != holders {n}"
        assert b in bm.used_block_ids
    assert set(holders) == bm.used_block_ids, "used blocks without a holder (leak)"
    assert not (set(bm.free_block_ids) & bm.used_block_ids)
    assert len(bm.free_block_ids) + len(bm.used_block_ids) == bm.num_blocks
    num_active = sum(1 for s in {id(s): s for s in [*sched.running, *sched.waiting, *in_flight]}.values()
                     if getattr(s, "block_table_ids", None) is not None)
    assert num_active <= sched.max_num_seqs, f"{num_active} seqs hold KV blocks > max_num_seqs {sched.max_num_seqs}"


def run_case(seed, num_blocks, block_size, max_num_seqs, max_batched, n_req, shared_prefix):
    rng = random.Random(seed)
    bm = BlockManager(num_blocks=num_blocks, block_size=block_size)
    sched = Scheduler(max_num_seqs, max_batched, EOS, bm, device="cpu")
    model = FakeModel(bm)

    prefix = [rng.randint(1, VOCAB - 1) for _ in range(block_size * 2)]
    refs = {}
    pending = []
    for _ in range(n_req):
        if pending and rng.random() < 0.2:
            # Exact repeat of an earlier prompt: a full prefix-cache hit.
            prompt = list(rng.choice(pending)[0])
        else:
            # Lengths include exact multiples of block_size (block-aligned prompts).
            n_body = rng.choice([rng.randint(1, block_size * 3), block_size * rng.randint(1, 3)])
            body = [rng.randint(1, VOCAB - 1) for _ in range(n_body)]
            prompt = (prefix + body) if shared_prefix and rng.random() < 0.5 else body
        max_tokens = rng.randint(1, block_size * 4)
        pending.append((prompt, max_tokens))

    finished = []
    steps = 0
    while pending or not sched.is_finished():
        # Requests arrive over time.
        if pending and (rng.random() < 0.3 or sched.is_finished()):
            prompt, max_tokens = pending.pop(0)
            seq = Sequence(prompt, max_tokens)
            refs[seq.seq_id] = reference(prompt, max_tokens)
            sched.add(seq)

        batch, is_prefill = sched.schedule()
        check_invariants(bm, sched, in_flight=batch)
        if batch:
            toks = model.run(batch, is_prefill)
            done = sched.postprocess(batch, toks, is_prefill)
            for seq in batch:   # mirror Engine.step
                end_block = seq.num_cached_tokens // block_size
                if end_block > seq.num_hashed_blocks:
                    bm.hash_completed_blocks(seq.token_ids, seq.block_table_ids, seq.num_hashed_blocks, end_block)
                    seq.num_hashed_blocks = end_block
            for seq in done:
                bm.deallocate(seq.block_table_ids)
                seq.block_table_ids = None
                finished.append(seq)
            check_invariants(bm, sched)
        steps += 1
        assert steps < 200_000, "no progress (livelock?)"

    assert len(finished) == n_req
    for seq in finished:
        assert seq.completion_token_ids == refs[seq.seq_id], f"seq {seq.seq_id} output differs from reference"
    assert len(bm.used_block_ids) == 0 and len(bm.free_block_ids) == num_blocks
    return sched.num_preemptions


def main():
    total_preempt = 0
    cases = 0
    for seed in range(300):
        rng = random.Random(10_000 + seed)
        block_size = rng.choice([2, 4, 8])
        max_num_seqs = rng.randint(1, 6)
        # Smallest cache that still fits one worst-case request + watermark.
        worst = (block_size * 5 + block_size * 4 + block_size - 1) // block_size
        num_blocks = worst + 1 + rng.randint(0, 12)
        max_batched = rng.choice([block_size, block_size * 3, 1024])
        total_preempt += run_case(seed, num_blocks, block_size, max_num_seqs, max_batched,
                                  n_req=rng.randint(5, 40), shared_prefix=rng.random() < 0.7)
        cases += 1
    print(f"OK: {cases} random cases passed, {total_preempt} preemptions exercised")
    assert total_preempt > 0, "test never triggered preemption"


if __name__ == "__main__":
    main()
