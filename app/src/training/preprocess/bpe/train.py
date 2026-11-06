"""Learn BPE merges from pre-tokenized chunks: `build_tokenizer`.

Training counts every adjacent pair within each chunk, merges the most frequent pair
everywhere it occurs, gives it the next id, and repeats until `num_merges` merges exist
or no pair occurs twice. With a `saved_state`, the chunks are first encoded with the
saved merges and training continues from there:

```
chunks (list[bytes])
│
▼
Counter(chunks)                          each distinct chunk once, with its freq
│
▼
encode_chunk(saved)                      bytes, or saved merges with saved_state
│  (ids, freq) per distinct chunk
▼
_count_pairs                             a linked list of PairNodes per chunk,
│  counts, nodes_of                      and each pair's count summed over freq
▼
heapq.heappop(heap) ◄────────┐           most frequent pair; stop below 2
│  best                      │
▼                            │
rewrite the neighbours of    │           new_id = FIRST_MERGE_ID + len(learned)
every node of best, push the │
changed counts ──────────────┘           until len(learned) == num_merges
│
▼  BpeTokenizer(merges=learned)
```

A merge touches only the nodes where its pair occurs, so no step recounts the text.
Pairs are tuples of ints, and each distinct chunk is linked once and weighted by its
count, which keeps training fast on large corpora.

Choices that look like bugs but are intentional (the chunking choices are listed in
`gpt4_preprocessor.py`):

- **Overlapping runs merge left to right.** "aaa" becomes "aa" + "a".
- **Ties go to the smallest pair.** On equal counts, the smaller
  `(first_id, second_id)` merges first, so the merges depend only on the counts.
  Continuing from a prefix of a run's own `state()` therefore reproduces the straight
  run. Version 1.0.0 broke ties by heap order, so a fresh run can learn different
  merges than it did; merges saved by 1.0.0 still load and encode unchanged.

References:

- Sennrich et al., 2016, "Neural Machine Translation of Rare Words with Subword
  Units" (arXiv 1508.07909): BPE for translation, and joint source-target BPE.
"""

import heapq
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import DefaultDict, List, Optional, Sequence, Tuple

from loguru import logger
from training.preprocess.bpe.encode import encode_chunk
from training.preprocess.bpe.tokenizer import BpeTokenizer
from training.preprocess.constants import FIRST_MERGE_ID

DEFAULT_NUM_MERGES = 500


# eq=False keeps identity comparison: a field-wise __eq__ would recurse through
# prev/next.
@dataclass(eq=False, slots=True)
class PairNode:
    """One adjacent pair of token ids in a distinct chunk; internal to `_learn_merges`.

    `pair` is (left id, right id). `_learn_merges` rewrites it in place when a merge
    absorbs one of its two tokens. `freq` is how many times the chunk holding this
    node occurs in the corpus, so the node stands for that many occurrences.
    """

    pair: Tuple[int, int]
    freq: int
    prev: Optional["PairNode"] = None
    next: Optional["PairNode"] = None


def build_tokenizer(
    chunks: Optional[List[bytes]] = None,
    num_merges: Optional[int] = None,
    saved_state: Optional[Sequence[Tuple[int, int]]] = None,
) -> BpeTokenizer:
    """Train a tokenizer, load a saved one, or load one and keep training it.

    Merges never cross a chunk boundary. Each step merges the most frequent adjacent
    pair everywhere it occurs, left to right, so an overlapping run like "aaa"
    becomes "aa" + "a". Training stops early once no pair occurs at least twice.

    Args:
        chunks: Training corpus, as `preprocess` returns it. Without it, nothing is
            learned.
        num_merges: Total merges to end with, saved ones included. Defaults to
            DEFAULT_NUM_MERGES when chunks is given; must be absent otherwise.
        saved_state: Merges as `BpeTokenizer.state` returns them. With chunks, the
            chunks are encoded with these first and training continues from them.

    Returns:
        The tokenizer. If saved_state holds more than num_merges merges, it holds
        all of them and a warning is logged.

    Raises:
        ValueError: If neither chunks nor saved_state is given, num_merges is given
            without chunks or is negative, or a saved pair is not two known ids.
        TypeError: If chunks holds anything but bytes, such as a raw str.
    """
    _validate_build_args(chunks=chunks, num_merges=num_merges, saved_state=saved_state)
    saved = BpeTokenizer(merges=saved_state or [])
    if chunks is None:
        return saved
    target = DEFAULT_NUM_MERGES if num_merges is None else num_merges
    if len(saved.merges) >= target:  # noqa: NAR001
        if len(saved.merges) > target:  # noqa: NAR001
            logger.warning(  # noqa: NAR001
                "saved_state has {} merges, more than num_merges={}; loading all of "
                "them and learning none",
                len(saved.merges),  # noqa: NAR001
                target,
            )
        return saved
    # A repeated chunk always encodes, and always merges, the same way, so each
    # distinct chunk is encoded and linked once and weighted by its count.
    sequences = [
        (encode_chunk(tokenizer=saved, chunk=chunk), freq)
        for chunk, freq in Counter(chunks).items()  # noqa: NAR001
    ]
    pairs = _learn_merges(sequences=sequences, pairs=saved.state(), num_merges=target)
    return BpeTokenizer(merges=pairs)


def _validate_build_args(
    chunks: Optional[List[bytes]],
    num_merges: Optional[int],
    saved_state: Optional[Sequence[Tuple[int, int]]],
) -> None:
    """Raise unless `build_tokenizer`'s arguments are usable; see its Raises section."""
    if chunks is None and saved_state is None:
        raise ValueError(  # noqa: NAR001
            "build_tokenizer needs chunks, saved_state, or both"
        )
    if num_merges is not None and chunks is None:
        raise ValueError(  # noqa: NAR001
            "num_merges needs chunks to learn from; saved_state alone loads as is"
        )
    if num_merges is not None and num_merges < 0:
        raise ValueError(f"num_merges must be >= 0, got {num_merges}")  # noqa: NAR001
    if chunks is not None and not all(isinstance(c, bytes) for c in chunks):  # noqa: NAR001
        raise TypeError(  # noqa: NAR001
            "chunks must be a list of bytes, as preprocess(text=...) returns"
        )
    for rank, pair in enumerate(saved_state or []):  # noqa: NAR001
        # A pair may name bytes or merges learned before it, never a protected id.
        valid = len(pair) == 2 and all(  # noqa: NAR001
            0 <= token_id < 256 or FIRST_MERGE_ID <= token_id < FIRST_MERGE_ID + rank
            for token_id in pair
        )
        if not valid:
            raise ValueError(  # noqa: NAR001
                f"saved_state[{rank}] = {pair} is not two ids that are bytes or earlier merges"
            )


def _count_pairs(
    sequences: List[Tuple[List[int], int]],
) -> Tuple[DefaultDict[Tuple[int, int], int], DefaultDict[Tuple[int, int], List[PairNode]]]:
    """Link adjacent id pairs into one doubly linked list per distinct chunk, by pair.

    Node i of a chunk holds the pair (ids[i], ids[i + 1]), so neighbouring nodes
    share one id. Each list ends at its chunk's boundary, so no pair spans two
    chunks.

    Args:
        sequences: (ids, freq) per distinct chunk: its token ids, and how many times
            the chunk occurs in the corpus.

    Returns:
        (counts, nodes_of): each pair's occurrences in the corpus, every node's freq
        summed, and the nodes where each pair occurs, in text order within a chunk.
    """
    counts: DefaultDict[Tuple[int, int], int] = defaultdict(int)  # noqa: NAR001
    nodes_of: DefaultDict[Tuple[int, int], List[PairNode]] = defaultdict(list)  # noqa: NAR001
    for ids, freq in sequences:
        prev: Optional[PairNode] = None
        for pair in zip(ids, ids[1:]):  # noqa: NAR001
            node = PairNode(pair=pair, freq=freq, prev=prev)
            if prev is not None:
                prev.next = node
            prev = node
            counts[pair] += freq
            nodes_of[pair].append(node)  # noqa: NAR001
    return counts, nodes_of


def _learn_merges(
    sequences: List[Tuple[List[int], int]],
    pairs: List[Tuple[int, int]],
    num_merges: int,
) -> List[Tuple[int, int]]:
    """Extend pairs with merges learned from sequences until it has num_merges entries.

    Each step merges the most frequent pair. On a tie, the smallest
    (first_id, second_id) wins, so the result depends only on the pair counts, not
    on the order of the chunks or of heap operations.

    Args:
        sequences: (ids, freq) per distinct chunk, already encoded with pairs, so no
            adjacent pair is an existing merge.
        pairs: Merges to continue from, as `BpeTokenizer.state` returns them; not
            mutated.
        num_merges: Total length to stop at. Stops earlier once no pair occurs at
            least twice.

    Returns:
        pairs followed by the newly learned merges' (first_id, second_id).
    """
    learned = list(pairs)  # noqa: NAR001
    counts, nodes_of = _count_pairs(sequences=sequences)
    # A max-heap by count via negation; tuple order breaks ties by the smaller pair.
    # Entries are never updated in place: a changed count pushes a new entry, and
    # pop skips any entry whose count no longer matches `counts`.
    heap = [(-count, pair) for pair, count in counts.items()]
    heapq.heapify(heap)  # noqa: NAR001
    while len(learned) < num_merges and heap:  # noqa: NAR001
        neg_count, best = heapq.heappop(heap)  # noqa: NAR001
        if counts.get(best) != -neg_count:  # noqa: NAR001
            continue
        if -neg_count < 2:
            break
        new_id = FIRST_MERGE_ID + len(learned)  # noqa: NAR001
        learned.append(best)  # noqa: NAR001
        touched: set[Tuple[int, int]] = set()  # noqa: NAR001

        # Lists are in text order, so overlapping runs ("aaa") merge left to right.
        # A node whose pair has changed since it was listed, e.g. absorbed by the
        # overlapping occurrence to its left, is skipped.
        for node in nodes_of.pop(best):  # noqa: NAR001
            if node.pair != best:
                continue
            prev, nxt = node.prev, node.next
            if prev is not None:
                counts[prev.pair] -= node.freq
                touched.add(prev.pair)  # noqa: NAR001
                prev.pair = (prev.pair[0], new_id)
                counts[prev.pair] += node.freq
                touched.add(prev.pair)  # noqa: NAR001
                nodes_of[prev.pair].append(prev)  # noqa: NAR001
                prev.next = nxt
            if nxt is not None:
                counts[nxt.pair] -= node.freq
                touched.add(nxt.pair)  # noqa: NAR001
                nxt.pair = (new_id, nxt.pair[1])
                counts[nxt.pair] += node.freq
                touched.add(nxt.pair)  # noqa: NAR001
                nodes_of[nxt.pair].append(nxt)  # noqa: NAR001
                nxt.prev = prev
        # Every pair created from here on contains new_id, so best never recurs.
        counts.pop(best, None)  # noqa: NAR001
        touched.discard(best)  # noqa: NAR001
        for pair in touched:
            if counts[pair] > 0:
                heapq.heappush(heap, (-counts[pair], pair))  # noqa: NAR001
            else:
                del counts[pair]
    return learned
