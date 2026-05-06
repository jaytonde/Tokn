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
        
        if self.waiting:
            seq = self.waiting.popleft()
            self.running.append(seq)
            return [seq], True
        else:
            seq = self.running.popleft()
            return [seq], False

    def is_finished(self):
        return True if not self.waiting else False

    def postprocess(self, seqs, token_ids, is_prefill):
        self.running.popleft()
