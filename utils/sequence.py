import time
from itertools import count
from sampling_params import SamplingParams

class Sequence:
    counter = count()

    def __init__(self, token_ids, max_tokens):

        self.seq_id = next(Sequence.counter)

        self.token_ids = list(token_ids)
        self.num_prompt_tokens = len(self.token_ids)
        self.num_tokens = len(self.token_ids)

        self.prompt_len = len(token_ids)
        self.max_tokens = max_tokens

        self.block_table = None
        self.num_generated_tokens = 0

        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0

        self.block_table = None
        self.status = "WAITING"

        self.last_token = self.token_ids[-1]
        self.sampling_params = SamplingParams()

        self.arrival_time = None
        self.token_timestamps = []

    def append_token(self, token_id):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
        self.num_generated_tokens += 1
        self.token_timestamps.append(time.perf_counter())

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def ttft(self):
        """Time To First Token: arrival -> first generated token (seconds)."""
        if self.arrival_time is None or not self.token_timestamps:
            return None
        return self.token_timestamps[0] - self.arrival_time

    @property
    def itls(self):
        """Inter-Token Latencies: gaps between consecutive generated tokens (seconds)."""
        ts = self.token_timestamps
        return [ts[i] - ts[i - 1] for i in range(1, len(ts))]

    @property
    def tpot(self):
        """Time Per Output Token: mean decode-step latency = mean(ITL) (seconds)."""
        if len(self.token_timestamps) < 2:
            return None
        decode_time = self.token_timestamps[-1] - self.token_timestamps[0]
        return decode_time / (len(self.token_timestamps) - 1)