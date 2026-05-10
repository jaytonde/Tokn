from itertools import count

class Sequence:

    counter = count()

    def __init__(self, token_ids, max_tokens):

        self.seq_id = next(Sequence.counter)

        self.token_ids = list[token_ids]
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

    @property
    def last_token(self):
        return self.token_ids[-1]

    def append_token(self, token_id):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
        self.num_generated_tokens += 1

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]