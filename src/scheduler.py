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

    def add(self, seq):
        self.waiting.append(seq)

    def schedule(self):
        scheduled = []

        # Decode active requests first to avoid starving in-flight generations.
        should_decode_first = (
            self.running
            and (
                not self.waiting
                or self.decode_steps_since_prefill < self.prefill_interval
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

            while self.waiting and len(scheduled) < self.max_num_seqs:
                seq = self.waiting[0]

                if seq.block_table is None:
                    cached_blocks         = self.block_manager.match_prefix(seq.token_ids)
                    alloc_token_ids       = seq.token_ids + ([0] * seq.max_tokens)
                    block_table_ids       = self.block_manager.allocate(token_ids=alloc_token_ids, num_cached_blocks=cached_blocks)
                    seq.block_table_ids   = block_table_ids
                    seq.block_table       = self.block_manager.make_block_table_tensor(block_table=block_table_ids,device=self.device)
                    seq.num_cached_tokens = cached_blocks * self.block_manager.block_size
                    seq.num_hashed_blocks = cached_blocks

                num_tokens = seq.num_tokens - seq.num_cached_tokens

                if num_tokens <= 0:
                    self.waiting.popleft()
                    seq.status = "RUNNING"
                    seq.num_scheduled_tokens = 0
                    self.running.append(seq)
                    continue

                budget_left = self.max_num_batched_tokens - num_batched_tokens
                if budget_left <= 0:
                    break

                # Chunk large prefill requests so they fit token budget.
                num_tokens = min(num_tokens, budget_left)

                if num_batched_tokens + num_tokens > self.max_num_batched_tokens:
                    break

                self.waiting.popleft()
                seq.status = "PREFILL"
                seq.num_scheduled_tokens = num_tokens

                scheduled.append(seq)
                num_batched_tokens += num_tokens

            if scheduled:
                self.decode_steps_since_prefill = 0
                return scheduled, True

        return scheduled, False
     
    def is_finished(self):
        return not self.waiting and not self.running

    def postprocess(self, seqs, token_ids, is_prefill):
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
