

class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

    def update(self, block_hash: int, token_ids: list[int]):
        self.hash = block_hash
        self.token_ids = list(token_ids)