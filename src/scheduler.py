import threading
from collections import deque

class Scheduler:
    
    def __init__(self, max_num_seqs, max_num_batched_tokens, eos_token_id, block_manager, device):
        self.waiting = deque()
        self.running = deque()

        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.eos_token_id = eos_token_id

        self.block_manager = block_manager
        self.device = device

        self.decode_steps_since_prefill = 0
        self.prefill_interval = 8

        # Engine.step runs schedule()/postprocess() in a worker thread while
        # add() runs on the asyncio event loop thread. Both mutate the waiting
        # and running deques, so every access goes through this lock.
        self.lock = threading.Lock()

    def add(self, seq):
        with self.lock:
            self.waiting.append(seq)

    def _schedule_unlocked(self):
        scheduled = []

        # Decode active requests first to avoid starving in-flight generations.
        # Refill a freed slot immediately: if a running request just finished
        # (running below max) and requests are waiting, prefill now instead of
        # waiting out the prefill interval, keeping the decode batch full.
        should_decode_first = (
            self.running
            and (
                not self.waiting
                or (
                    len(self.running) >= self.max_num_seqs
                    and self.decode_steps_since_prefill < self.prefill_interval
                )
            )
        )

        # DECODE
        if should_decode_first:
            while self.running and len(scheduled) < self.max_num_seqs:
                seq = self.running.popleft()
                seq.num_scheduled_tokens = 1
                scheduled.append(seq)

            self.decode_steps_since_prefill += 1
            return scheduled, False

        # PREFILL
        if self.waiting:
            num_batched_tokens = 0

            # Sequences that already hold KV blocks: running ones, plus chunked
            # prefills waiting for their next chunk. The KV cache is sized for
            # max_num_seqs of them, so new requests are only admitted below that
            # count and only when their blocks are actually free. Without this,
            # steady arrivals exhaust the cache and allocate() raises.
            num_active = len(self.running) + sum(1 for s in self.waiting if s.block_table is not None)
            admit_new = True

            for seq in list(self.waiting):
                if len(scheduled) >= self.max_num_seqs:
                    break

                if seq.block_table is None:
                    if not admit_new:
                        # Keep FIFO order among new requests, but still let
                        # already-admitted chunked prefills further back proceed.
                        continue

                    cached_blocks   = self.block_manager.match_prefix(seq.token_ids)
                    alloc_token_ids = seq.token_ids + ([0] * seq.max_tokens)

                    if num_active >= self.max_num_seqs or not self.block_manager.can_allocate(alloc_token_ids, cached_blocks):
                        admit_new = False
                        continue

                    block_table_ids       = self.block_manager.allocate(token_ids=alloc_token_ids, num_cached_blocks=cached_blocks)
                    seq.block_table_ids   = block_table_ids
                    seq.block_table       = self.block_manager.make_block_table_tensor(block_table=block_table_ids,device=self.device)
                    seq.num_cached_tokens = cached_blocks * self.block_manager.block_size
                    seq.num_hashed_blocks = cached_blocks
                    num_active += 1

                num_tokens = seq.num_tokens - seq.num_cached_tokens

                if num_tokens <= 0:
                    self.waiting.remove(seq)
                    seq.status = "RUNNING"
                    seq.num_scheduled_tokens = 0
                    self.running.append(seq)
                    continue

                budget_left = self.max_num_batched_tokens - num_batched_tokens
                if budget_left <= 0:
                    break

                # Chunk large prefill requests so they fit token budget.
                num_tokens = min(num_tokens, budget_left)

                self.waiting.remove(seq)
                seq.status = "PREFILL"
                seq.num_scheduled_tokens = num_tokens

                scheduled.append(seq)
                num_batched_tokens += num_tokens

            if scheduled:
                self.decode_steps_since_prefill = 0
                return scheduled, True

        # Nothing could be prefilled (cache full or at max_num_seqs): keep
        # decoding so running sequences finish and free their blocks.
        if self.running:
            while self.running and len(scheduled) < self.max_num_seqs:
                seq = self.running.popleft()
                seq.num_scheduled_tokens = 1
                scheduled.append(seq)

            self.decode_steps_since_prefill += 1
            return scheduled, False

        return scheduled, False
     
    def _is_finished_unlocked(self):
        return not self.waiting and not self.running

    def _postprocess_unlocked(self, seqs, token_ids, is_prefill):
        finished = []

        for seq, token_id in zip(seqs, token_ids):
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                seq.status = "WAITING"
                self.waiting.append(seq)
                continue

            seq.append_token(token_id)

            done = (
                token_id == self.eos_token_id
                or
                seq.num_generated_tokens >= seq.max_tokens
            )

            if done:
                seq.status = "FINISHED"
                finished.append(seq)
            else:
                seq.status = "RUNNING"
                self.running.append(seq)

        return finished

    def schedule(self):
        with self.lock:
            return self._schedule_unlocked()

    def is_finished(self):
        with self.lock:
            return self._is_finished_unlocked()

    def postprocess(self, seqs, token_ids, is_prefill):
        with self.lock:
            return self._postprocess_unlocked(seqs, token_ids, is_prefill)
