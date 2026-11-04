import random

import pytest
from training.serve.block_manager import BlockManager, block_hash


def test_ten_thousand_random_operations_leak_nothing():
    rng = random.Random(0)
    blocks = BlockManager(num_blocks=64, block_size=4)
    prefixes = [[rng.randrange(50) for _ in range(12)] for _ in range(3)]
    live = {}
    for step in range(10_000):
        if live and (rng.random() < 0.45 or blocks.num_free < 8):
            seq = rng.choice(list(live))
            blocks.free_sequence(seq_id=seq)
            del live[seq]
            continue
        prompt = rng.choice(prefixes)[: rng.randrange(4, 13)] + [rng.randrange(50) for _ in range(rng.randrange(0, 6))]
        seq = f"s{step}"
        try:
            blocks.allocate(seq_id=seq, prompt=prompt)
        except MemoryError:
            continue
        blocks.commit_full_blocks(seq_id=seq, tokens=prompt)
        if rng.random() < 0.5 and blocks.can_append(seq_id=seq, num_tokens=len(prompt) + 5):
            blocks.append(seq_id=seq, num_tokens=len(prompt) + 5)
        live[seq] = prompt
        held = [b for table in blocks.tables.values() for b in table]
        assert all(blocks.refcount[b] == held.count(b) for b in set(held))
    for seq in list(live):
        blocks.free_sequence(seq_id=seq)
    assert all(count == 0 for count in blocks.refcount)
    assert blocks.num_free == 64
    assert len(set(blocks.free) | set(blocks.evictable)) == 64


def test_prefix_hash_chain_is_stable_and_diverges_after_a_one_token_change():
    a = block_hash(parent=block_hash(parent=None, tokens=(1, 2)), tokens=(3, 4))
    assert a == block_hash(parent=block_hash(parent=None, tokens=(1, 2)), tokens=(3, 4))
    assert a != block_hash(parent=block_hash(parent=None, tokens=(1, 9)), tokens=(3, 4))


def test_a_shared_prefix_is_reused_and_survives_the_first_owner():
    blocks = BlockManager(num_blocks=8, block_size=4)
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert blocks.allocate(seq_id="a", prompt=prompt) == 0
    blocks.commit_full_blocks(seq_id="a", tokens=prompt)
    assert blocks.allocate(seq_id="b", prompt=prompt[:8] + [42]) == 8
    assert blocks.tables["a"][:2] == blocks.tables["b"][:2]
    blocks.free_sequence(seq_id="a")
    assert blocks.refcount[blocks.tables["b"][0]] == 1
    blocks.free_sequence(seq_id="b")
    assert blocks.allocate(seq_id="c", prompt=prompt) == 8  # evictable blocks are still hits


def test_allocation_that_does_not_fit_changes_nothing():
    blocks = BlockManager(num_blocks=2, block_size=4)
    with pytest.raises(MemoryError):
        blocks.allocate(seq_id="a", prompt=list(range(9)))
    assert blocks.num_free == 2 and blocks.tables == {}
