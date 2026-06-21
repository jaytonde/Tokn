
import xxhash
import torch
import numpy as np

from utils.block import Block
from collections import deque


class BlockManager:

    def __init__(self, num_blocks, block_size):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id = {}
        self.free_block_ids = deque(range(num_blocks))
        self.used_block_ids = set()

    def compute_hash(self, block_tokens: list[int], prev_hash: int = -1) -> int:
        """
        Computes hash for one KV cache block.

        prev_hash makes this a chained prefix hash.

        Example:
            block0_hash = hash(block0_tokens)
            block1_hash = hash(block0_hash + block1_tokens)
            block2_hash = hash(block1_hash + block2_tokens)

        So same block tokens at different prefix positions will not collide logically.
        """
        h = xxhash.xxh64()

        if prev_hash != -1:
            h.update(prev_hash.to_bytes(8, "little", signed=False)) #TO-DO

        tokens_array = np.asarray(block_tokens, dtype=np.int64)
        h.update(tokens_array.tobytes())

        return h.intdigest()

    def _block_tokens(self, token_ids: list[int], block_idx: int) -> list[int]:
        """
        Returns token IDs for one logical block.

        Example:
            block_size = 4
            token_ids = [10, 11, 12, 13, 14, 15]

            block_idx = 0 -> [10, 11, 12, 13]
            block_idx = 1 -> [14, 15]
        """
        start = block_idx * self.block_size
        end = min(start + self.block_size, len(token_ids))

        return token_ids[start:end]

    def _num_blocks(self, token_ids: list[int]) -> int:
        """
        Returns number of KV blocks needed for this sequence.
        """
        return (len(token_ids) + self.block_size - 1) // self.block_size

    def _allocate_one_block(self) -> int:
        """
        Allocates one physical KV cache block.

        This block may have old cached metadata from a previous request.
        If we reuse it, we must remove its old hash mapping because the
        physical KV memory will be overwritten.
        """
        
        if not self.free_block_ids:
            raise RuntimeError("No free KV cache blocks available")

        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]

        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]

        block.reset()
        self.used_block_ids.add(block_id)

        return block_id

    def match_prefix(self, token_ids: list[int]) -> int:

        """
        Returns how many FULL prefix blocks already exist in KV cache.

        Important:
        - Only full blocks are reusable.
        - Last partial block is ignored.
        - We verify token_ids after hash lookup to protect against hash collision.
        """

        num_full_blocks = len(token_ids) // self.block_size

        matched_blocks = 0
        prev_hash = -1

        for block_idx in range(num_full_blocks):
            block_tokens = self._block_tokens(token_ids, block_idx)
            block_hash = self.compute_hash(block_tokens, prev_hash)

            block_id = self.hash_to_block_id.get(block_hash, -1)

            if block_id == -1:
                break

            block = self.blocks[block_id]

            #Collision safety check.
            if block.token_ids != block_tokens:
                break

            matched_blocks += 1
            prev_hash = block_hash

        return matched_blocks

    def allocate(self, token_ids: list[int], num_cached_blocks: int) -> list[int]:

        """
        Builds block_table for one request.

        Cached prefix blocks are reused.
        Remaining blocks are newly allocated.

        Returns:
            block_table where:
            logical block index -> physical KV cache block id
        """

        block_table: list[int] = []

        num_blocks = self._num_blocks(token_ids)
        prev_hash = -1

        # 1. Reuse cached prefix blocks
        for block_idx in range(num_cached_blocks):
            block_tokens = self._block_tokens(token_ids, block_idx)
            block_hash = self.compute_hash(block_tokens, prev_hash)

            block_id = self.hash_to_block_id[block_hash]
            block = self.blocks[block_id]

            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)

            block_table.append(block_id)

            prev_hash = block_hash

        # 2. Allocate new blocks for uncached suffix
        for _ in range(num_cached_blocks, num_blocks):
            block_id = self._allocate_one_block()
            block_table.append(block_id)

        return block_table

    def hash_completed_blocks(self, token_ids: list[int], block_table: list[int], start_block: int, end_block: int):

        """
        Hash newly completed FULL blocks and register them in hash_to_block_id.

        start_block: first logical block index to hash
        end_block: exclusive logical block index

        Example:
            block_size = 4
            token_ids = [1,2,3,4,5,6,7,8,9]

            full blocks are:
            block 0 -> [1,2,3,4]
            block 1 -> [5,6,7,8]

            block 2 -> [9] is partial, so do not hash it.
        """

        num_full_blocks = len(token_ids) // self.block_size
        end_block = min(end_block, num_full_blocks)

        if start_block >= end_block:
            return

        if start_block > 0:
            prev_block_id = block_table[start_block - 1]
            prev_hash = self.blocks[prev_block_id].hash
        else:
            prev_hash = -1
 
        for block_idx in range(start_block, end_block):

            block_tokens = self._block_tokens(token_ids, block_idx)
            block_hash = self.compute_hash(block_tokens, prev_hash)

            physical_block_id = block_table[block_idx]
            block = self.blocks[physical_block_id]

            block.update(block_hash, block_tokens)
            self.hash_to_block_id[block_hash] = physical_block_id

            prev_hash = block_hash

    def deallocate(self, block_table: list[int]):

        for block_id in reversed(block_table):
            block = self.blocks[block_id]

            if block.ref_count <= 0:
                raise RuntimeError(
                    f"Trying to deallocate block {block_id}, "
                    f"but ref_count={block.ref_count}"
                )

            block.ref_count -= 1

            if block.ref_count == 0:
                self.used_block_ids.remove(block_id)
                self.free_block_ids.append(block_id)

    def make_block_table_tensor(
        self,
        block_table: list[int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Converts Python block table to FlashAttention-compatible tensor.

        Shape:
            [batch_size, num_blocks]

        For your current single-request Tokn engine:
            batch_size = 1
        """
        return torch.tensor(
            [block_table],
            device=device,
            dtype=torch.int32,
        )








    






if __name__=="__main__":
    bm = BlockManager(num_blocks=10, block_size=4)

    h0 = bm.compute_hash([1, 2, 3, 4])
    h1 = bm.compute_hash([5, 6, 7, 8], h0)

    print(h0)
    print(h1)

    token_ids = [10, 11, 12, 13, 14, 15]

    print(bm._block_tokens(token_ids, 0))
    print(bm._block_tokens(token_ids, 1))
    print(bm._num_blocks(token_ids))

    b0 = bm._allocate_one_block()
    b1 = bm._allocate_one_block()

    print(b0, b1)
    print(list(bm.free_block_ids))
    print(bm.used_block_ids)
    print(bm.blocks[b0].ref_count)


    block0_tokens = [1, 2, 3, 4]
    h0 = bm.compute_hash(block0_tokens)

    block_id = bm._allocate_one_block()
    bm.blocks[block_id].update(h0, block0_tokens)
    bm.hash_to_block_id[h0] = block_id

    print(bm.match_prefix(token_ids))

    num_cached_blocks = bm.match_prefix(token_ids)
    block_table = bm.allocate(token_ids, num_cached_blocks)

    print("cached:", num_cached_blocks)
    print("block_table:", block_table)
    print("free:", list(bm.free_block_ids))
    print("used:", bm.used_block_ids)


    token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    num_cached_blocks = bm.match_prefix(token_ids)
    block_table = bm.allocate(token_ids, num_cached_blocks)

    # Hash only full blocks after prefill.
    start_block = num_cached_blocks
    end_block = len(token_ids) // bm.block_size

    bm.hash_completed_blocks(
        token_ids=token_ids,
        block_table=block_table,
        start_block=start_block,
        end_block=end_block,
    )

    print("hash_to_block_id:", bm.hash_to_block_id)
    print("block0 tokens:", bm.blocks[block_table[0]].token_ids)
    print("block1 tokens:", bm.blocks[block_table[1]].token_ids)
    print("block2 tokens:", bm.blocks[block_table[2]].token_ids)


    print("before deallocate")
    print("block_table:", block_table)
    print("used:", bm.used_block_ids)
    print("free:", list(bm.free_block_ids))
    print("matched:", bm.match_prefix(token_ids))

    bm.deallocate(block_table)

    print("after deallocate")
    print("used:", bm.used_block_ids)
    print("free:", list(bm.free_block_ids))
    print("matched:", bm.match_prefix(token_ids))
    
