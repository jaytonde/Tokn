

class Sequence:
    def __init__(self, token_ids, max_tokens):
        self.token_ids = token_ids
        self.prompt_len = len(token_ids)
        self.max_tokens = max_tokens

        self.block_table = None
        self.num_generated_tokens = 0
        self.status = "WAITING"

    @property
    def last_token(self):
        return self.token_ids[-1]

    def append_token(self, token_id):
        self.token_ids.append(token_id)
        self.num_generated_tokens += 1