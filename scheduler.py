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