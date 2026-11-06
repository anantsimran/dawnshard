"""Encode pre-tokenized chunks with a BPE tokenizer's merges.

`BpeTokenizer.encode` preprocesses text and calls `encode_chunks`. Call
`encode_chunks` directly when you already have chunks, or when the chunks are not
the ones `gpt4_preprocessor.preprocess` would produce.

Within each chunk, `encode_chunk` applies the earliest-learned merge present until
none applies, left to right, so an overlapping run like "aaa" becomes "aa" + "a". A
merge can only create pairs that contain its own output token, and every merge
that uses that token ranks higher, so this matches replaying all merges in the order
they were learned, skipping those that never occur. That reproduces the segmentation
training produced (`train.py`). Every byte is in the base vocab, so there are no
unknown tokens.

The cache is keyed by the tokenizer's `tokenizer_id` and the chunks, not the text, so
a change to the pre-tokenizer changes the key instead of serving stale ids.
"""

from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple

from training.dal.cache import cached
from training.dal.dao.dao import Dao
from training.preprocess.constants import FIRST_MERGE_ID

if TYPE_CHECKING:
    from training.preprocess.bpe.tokenizer import BpeTokenizer


def encode_chunks(
    tokenizer: "BpeTokenizer",
    chunks: Iterable[Sequence[bytes]],
    cache: Optional[Dao] = None,
) -> List[List[int]]:
    """Tokenize each text's chunks; one id list per text, in order.

    Each distinct text is encoded once, and the distinct texts are sorted before
    they are keyed, so a reordered or duplicated input still hits the cache. The
    ids are the same with or without `cache`.

    Args:
        chunks: One chunk list per text, from the same pre-tokenizer the tokenizer
            was trained with, such as `[preprocess(text=t) for t in texts]`.
        cache: Where the encodings are stored; None stores nothing. The caller must
            keep it open for the call.

    Returns:
        Token ids per text; ids below 256 are raw bytes, FIRST_MERGE_ID + k is
        merges[k]. Protected ids are never produced. An empty input returns `[]`
        without touching the cache.

    Raises:
        TypeError: If an item is a bytes or str rather than a chunk list, as when
            one text's chunks are passed without wrapping them in a list.

    Side effects:
        With `cache`, a miss stores the encodings of all distinct texts as one entry.
    """
    texts: List[Tuple[bytes, ...]] = []
    for text_chunks in chunks:
        if isinstance(text_chunks, (bytes, str)):  # noqa: NAR001
            raise TypeError(  # noqa: NAR001
                "encode_chunks takes one chunk list per text; wrap a single text's chunks in a list"
            )
        texts.append(tuple(text_chunks))  # noqa: NAR001
    if not texts:
        return []
    unique = sorted(set(texts))  # noqa: NAR001
    encoded = _encode_unique(tokenizer=tokenizer, texts=unique, cache=cache)
    return [list(encoded[text]) for text in texts]  # noqa: NAR001


@cached(version="1")
def _encode_unique(
    tokenizer: "BpeTokenizer", texts: List[Tuple[bytes, ...]]
) -> Dict[Tuple[bytes, ...], List[int]]:
    """Encode each distinct text once; the unit `encode_chunks` caches.

    Bump this decorator's version when the ids produced for the same merges change in
    a way `tokenizer.__version__` does not already capture.

    Args:
        texts: Distinct chunk tuples, sorted so the cache key does not depend on the
            order the caller passed them in.

    Returns:
        Each text's chunk tuple mapped to its token ids.
    """
    return {
        text: [  # noqa: NAR001
            token_id
            for chunk in text
            for token_id in encode_chunk(tokenizer=tokenizer, chunk=chunk)
        ]
        for text in texts
    }


def encode_chunk(tokenizer: "BpeTokenizer", chunk: bytes) -> List[int]:
    """Encode one chunk with the tokenizer's merges; see the module docstring."""
    no_merge = len(tokenizer.merges)  # noqa: NAR001
    ids = list(chunk)  # noqa: NAR001
    while len(ids) >= 2:  # noqa: NAR001
        rank = min(  # noqa: NAR001
            tokenizer.rank_of.get(pair, no_merge)  # noqa: NAR001
            for pair in zip(ids, ids[1:])  # noqa: NAR001
        )
        if rank == no_merge:
            break
        pair = tokenizer.merges[rank]
        new_id = FIRST_MERGE_ID + rank
        merged: List[int] = []
        i = 0
        while i < len(ids):  # noqa: NAR001
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:  # noqa: NAR001
                merged.append(new_id)  # noqa: NAR001
                i += 2
            else:
                merged.append(ids[i])  # noqa: NAR001
                i += 1
        ids = merged
    return ids
