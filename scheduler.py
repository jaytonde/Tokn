from collections import deque

class Scheduler:
    def __init__(self, max_num_seqs, max_num_batched_tokens, eos_token_id):
        self.waiting = deque()
        self.running = deque()

        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.eos_token_id = eos_token_id

    def add(self, seq):
        self.waiting.append(seq)

    def schedule_prefill(self):
        if not self.waiting:
            return []

        seq = self.waiting.popleft()
        seq.status = "PREFILL"

        return [seq]

    def schedule_decode(self):
        if not self.waiting:
            return []

        return list(self.running)

    def schedule(self):
        scheduled = []

        #PREFILL
        if self.waiting:
            num_batched_tokens = 0

            while self.waiting and len(scheduled) < self.max_num_seqs:
                seq = self.waiting[0]
                num_tokens = seq.num_tokens - seq.num_cached_tokens

                if num_batched_tokens + num_tokens > self.max_num_batched_tokens:
                    break

                self.waiting.popleft()
                seq.status = "RUNNING"
                seq.num_scheduled_tokens = num_tokens

                scheduled.append(seq)
                self.running.append(seq)
                num_batched_tokens += num_tokens
            
            return scheduled, True

        #DECODE
        while self.running and len(scheduled) < self.max_num_seqs:
            seq = self.running.popleft()
            seq.num_scheduled_tokens = 1
            scheduled.append(seq)

        return scheduled, False
        

    def is_finished(self):
        return not self.waiting and not self.running

    def postprocess(self, seqs, token_ids, is_prefill):
        finished = []

        for seq, token_id in zip(seqs, token_ids):
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

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
