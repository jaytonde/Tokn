from collections import deque

class Scheduler:
    def __init__(self):
        self.waiting = deque()
        self.running = deque()

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
            self.running.add(seq)
            return [seq], True
        else:
            return [seq], False

    def is_finished(self):
        return True if not self.waiting

    def postprocess(self, seqs, token_ids, is_prefill):
        self.running.popleft()
        