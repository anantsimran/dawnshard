"""Paged KV-cache bookkeeping: fixed-size blocks, block tables, refcounts, prefix reuse.

The KV cache is a pool of `num_blocks` blocks of `block_size` token slots. A sequence
owns a block table, the list of block ids holding its tokens in order, so its cache
need not be contiguous and grows one block at a time: a request reserves only what it
has used, not its maximum length, which is what lets more sequences share memory.

```
request A tokens:  [t0 … t15][t16 … t31][t32 t33]        block_size = 16
block table A:     [   7    ][    2    ][   9   ]        any free blocks, in any order
request B (same 16-token system prompt):
block table B:     [   7    ][   4   ]                   block 7 shared, refcount 2
```

Prefix caching: every *full* block is hashed together with its parent block's hash,
so a hash names the whole prefix up to that block, not just its 16 tokens. A new
request walks its prompt block by block and reuses any block whose chain hash is
cached, skipping that prefill. Freed blocks whose content is cached stay in an LRU
list; they are reused on a hit, and evicted (oldest first) only when the free list
is empty. A block is reclaimed only when its refcount reaches zero, so freeing one
request never frees a block another still reads.
"""

import hashlib
from collections import OrderedDict, deque
from typing import Optional


def block_hash(parent: Optional[str], tokens: tuple[int, ...]) -> str:
    """Hash of a full block given its parent's hash: names the entire prefix."""
    return hashlib.sha256(f"{parent}|{','.join(map(str, tokens))}".encode()).hexdigest()  # noqa: NAR001


class BlockManager:
    """Allocates KV blocks to sequences and shares full prefix blocks between them."""

    def __init__(self, num_blocks: int, block_size: int = 16, prefix_caching: bool = True) -> None:
        """Start with every block free."""
        self.block_size, self.prefix_caching = block_size, prefix_caching
        self.free: deque[int] = deque(range(num_blocks))  # noqa: NAR001
        self.refcount = [0] * num_blocks
        self.tables: dict[str, list[int]] = {}
        self.cached: dict[str, int] = {}  # chain hash -> block id
        self.hash_of: dict[int, str] = {}  # block id -> chain hash, for full blocks
        self.evictable: OrderedDict[int, None] = OrderedDict()  # refcount 0, still cached
        self.hits = self.lookups = 0

    @property
    def num_free(self) -> int:
        """Blocks that can be handed out: free plus evictable."""
        return len(self.free) + len(self.evictable)  # noqa: NAR001

    def blocks_needed(self, num_tokens: int) -> int:
        """Blocks a sequence of `num_tokens` occupies."""
        return -(-num_tokens // self.block_size)

    def _take(self) -> int:
        """A fresh block: from the free list, else evict the least recently freed cached one."""
        if self.free:
            return self.free.popleft()
        block, _ = self.evictable.popitem(last=False)
        del self.cached[self.hash_of.pop(block)]  # noqa: NAR001
        return block

    def _ref(self, block: int) -> None:
        """Take a reference to `block`, pulling it off the evictable list."""
        self.refcount[block] += 1
        self.evictable.pop(block, None)  # noqa: NAR001

    def allocate(self, seq_id: str, prompt: list[int]) -> int:
        """Give `seq_id` blocks for `prompt`; return how many leading tokens were cached.

        Raises:
            MemoryError: If the uncached part does not fit; nothing is allocated then.
        """
        table, parent, cached_tokens = [], None, 0
        full = len(prompt) // self.block_size  # noqa: NAR001
        for index in range(full if self.prefix_caching else 0):  # noqa: NAR001
            chain = block_hash(
                parent=parent,
                tokens=tuple(prompt[index * self.block_size : (index + 1) * self.block_size]),  # noqa: NAR001
            )  # noqa: NAR001
            self.lookups += 1
            if chain not in self.cached:
                break
            self.hits += 1
            table.append(self.cached[chain])  # noqa: NAR001
            parent, cached_tokens = chain, cached_tokens + self.block_size
        if self.blocks_needed(num_tokens=len(prompt)) - len(table) > self.num_free - sum(  # noqa: NAR001
            1 for b in table if b in self.evictable
        ):  # noqa: NAR001
            raise MemoryError(f"no room for {seq_id}")  # noqa: NAR001
        for block in table:
            self._ref(block=block)
        while len(table) < self.blocks_needed(num_tokens=len(prompt)):  # noqa: NAR001
            block = self._take()
            self._ref(block=block)
            table.append(block)  # noqa: NAR001
        self.tables[seq_id] = table
        return cached_tokens

    def can_append(self, seq_id: str, num_tokens: int) -> bool:
        """Whether `seq_id` can grow to `num_tokens` with the blocks left."""
        return self.blocks_needed(num_tokens=num_tokens) - len(self.tables[seq_id]) <= self.num_free  # noqa: NAR001

    def append(self, seq_id: str, num_tokens: int) -> None:
        """Grow `seq_id`'s table to hold `num_tokens` tokens.

        Raises:
            MemoryError: If a needed block is not available.
        """
        table = self.tables[seq_id]
        while len(table) < self.blocks_needed(num_tokens=num_tokens):  # noqa: NAR001
            if not self.num_free:
                raise MemoryError(f"no block for {seq_id}")  # noqa: NAR001
            block = self._take()
            self._ref(block=block)
            table.append(block)  # noqa: NAR001

    def commit_full_blocks(self, seq_id: str, tokens: list[int]) -> None:
        """Record chain hashes for `seq_id`'s full blocks so later requests can reuse them."""
        if not self.prefix_caching:
            return
        parent = None
        for index, block in enumerate(self.tables[seq_id][: len(tokens) // self.block_size]):  # noqa: NAR001
            chain = block_hash(
                parent=parent,
                tokens=tuple(tokens[index * self.block_size : (index + 1) * self.block_size]),  # noqa: NAR001
            )  # noqa: NAR001
            if block not in self.hash_of and chain not in self.cached:
                self.cached[chain], self.hash_of[block] = block, chain
            parent = chain

    def free_sequence(self, seq_id: str) -> None:
        """Drop `seq_id`'s references; a block with none left is freed or kept as evictable."""
        for block in reversed(self.tables.pop(seq_id)):  # noqa: NAR001
            self.refcount[block] -= 1
            if self.refcount[block] == 0:
                if block in self.hash_of:
                    self.evictable[block] = None
                else:
                    self.free.append(block)  # noqa: NAR001
