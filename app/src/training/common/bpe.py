"""Byte-level Byte Pair Encoding: fit once, encode/decode many times.

BPE turns text into a list of integer token ids. It starts from the 256 possible
bytes and learns "merges": pairs of adjacent tokens that occur often enough to earn a
single id of their own. Frequent words end up as one token. Rare words fall back to
smaller pieces, down to single bytes, so no input is ever unknown.

Usage:

```python
from training.common import bpe

merges, vocab = bpe.train(text="low lower lowest", num_merges=4)
ids = bpe.encode(text="lowest\tlow", merges=merges)  # [261, 101, 115, 116, 262]
[vocab[i] for i in ids]                               # ['low', 'e', 's', 't', ' low']
bpe.decode(ids=ids, vocab=vocab)                      # 'lowest low' (tab lost)
```

`train` runs once, on a training corpus. `encode` and `decode` run on every text
afterwards, and never learn anything new. Requires the `regex` package.

What train returns:

- `merges`: a list of `Phrase`, in the order they were learned. `merges[k]` is
  token id `FIRST_MERGE_ID + k`. A `Phrase` is a binary tree: a leaf holds one byte,
  a merge holds the two phrases it joined. `merges[1]` above is
  `Phrase(first=<"lo">, second=<byte "w">)`. `encode` needs only this list.
- `vocab`: a dict of token id to its string, used by `decode` and for printing.
  `len(vocab)` is the embedding table size, since it counts the protected ids too.

To save a tokenizer, store each merge as its `(first_id, second_id)` pair and rebuild
both return values with `tokenizer_from_pairs`.

Token ids:

```
  0 ─ 255                 raw bytes; byte b is vocab[b] = chr(b)
  256                     PAD_ID, vocab ""
  257                     MASK_ID, vocab "[MASK]"
  258                     CLS_ID, vocab "[CLS]"
  259                     SEP_ID, vocab "[SEP]"
  260 ─ 259+len(merges)   merges; FIRST_MERGE_ID + k is merges[k]
```

The protected ids live in `constants.py` next to this file. They sit between the
bytes and the merges, so they keep the same ids whatever `num_merges` is. `encode`
never produces them and `decode` drops them. Because vocab stores one `chr(b)` per
byte, a multi-byte UTF-8 character shows up in vocab as Latin-1 mojibake: "é" is
bytes 195, 169, which print as 'Ã' and '©'. `decode` undoes this, so its output is
correct.

The algorithm, training:

1. Replace every whitespace character with a space (`EOW`), split the text into
   chunks with the GPT-4 pre-tokenization pattern (`_CHUNK_PATTERN`), and split
   each chunk into UTF-8 bytes. Each byte is one token.
1. Count every adjacent pair of tokens within a chunk.
1. Take the most frequent pair, give it the next free id, and replace every
   occurrence of it in the text with the new token, left to right.
1. Recount the pairs that the replacement changed, and repeat from step 3 until
   `num_merges` merges exist or no pair occurs twice.

For "low lower lowest" this learns 'lo', then 'low', then ' low', then ' lowe'.

Recounting the whole text after every merge would cost O(text) per merge. Instead,
`train` keeps the pairs in one doubly linked list of `PhraseNode`s per chunk,
indexes the nodes by pair, and keeps the counts in a max-heap. A merge then visits
only the nodes where that pair occurs, and rewrites each one's two neighbours in
place.

The algorithm, encoding: normalize and split into chunks and bytes the same way,
then, within each chunk, repeatedly apply the earliest-learned merge that appears,
until none do. This reproduces the segmentation training would have produced for
that text.

Why pre-tokenize: without chunks, the vocabulary is spent on variants such as
"dog.", "dog!" and "dog?". With them, " dog" is learned once and punctuation
separately. Numbers are always built from the same pieces of at most 3 digits.
Every token lies inside one chunk, so tokens line up with words, which whole-word
masking needs. And token length is bounded by chunk length, which keeps the
recursion in `Phrase.to_str` shallow (a long symbol run such as "=======" is still
one chunk). The README next to this file shows how text is chunked, with
examples, and how to use the tokenizer for batching, saving, and translation.

Choices that look like bugs but are intentional:

- **Whitespace is lossy.** Tabs and newlines come back from `decode` as spaces.
- **Merges never cross a chunk.** "The dog." is chunked as 'The', ' dog', '.'.
  One leading space or symbol can join a word (' low' above, '(hello'), but
  nothing joins the end of one.
- **Numbers split into runs of at most 3 digits.** "12345" is chunked as '123',
  '45', and the space before a number is its own chunk.
- **Overlapping runs merge left to right.** "aaa" becomes "aa" + "a".
- **Ties break by heap order.** When two pairs have equal counts, the heap picks
  which merges first. That is the same on every run, but may differ from other
  BPE libraries.

References:

- tiktoken, `tiktoken_ext/openai_public.py`: the `cl100k_base` pattern, which
  `_CHUNK_PATTERN` copies character for character.
- Radford et al., 2019, "Language Models are Unsupervised Multitask Learners",
  section 2.2: why BPE should not merge across character categories.
- Sennrich et al., 2016, "Neural Machine Translation of Rare Words with Subword
  Units" (arXiv 1508.07909): BPE for translation, and joint source-target BPE.
"""

import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Optional, Tuple

import regex
from training.common.constants import CLS_ID, MASK_ID, PAD_ID, PROTECTED_IDS, SEP_ID
from training.common.priority_queue import MaxHeap, PrioritizedItem

UNK = "<unk>"
# vocab string for each protected id; decode drops all of them.
_PROTECTED_STRINGS = {PAD_ID: "", MASK_ID: "[MASK]", CLS_ID: "[CLS]", SEP_ID: "[SEP]"}
FIRST_MERGE_ID = 256 + len(PROTECTED_IDS)  # noqa: NAR001
EOW = " "  # all whitespace is normalized to it; chunks attach it to the next word
_WHITESPACE_TO_EOW = {
    c: EOW
    for c in range(sys.maxunicode + 1)  # noqa: NAR001
    if chr(c).isspace()  # noqa: NAR001
}
# GPT-4 (tiktoken cl100k_base) pre-tokenization pattern. Merges never cross the
# chunks it produces: contractions, letter runs with one optional leading
# non-letter (usually the space), digit runs of at most 3, punctuation runs, and
# whitespace. `regex` is needed because stdlib `re` has no \p{L} / \p{N}.
_CHUNK_PATTERN = regex.compile(  # noqa: NAR001
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
)


@dataclass(frozen=True)
class Phrase:
    """A token, as a binary tree of the merges that built it.

    A leaf sets `encoding` to a byte (0-255). A merge sets `first` and `second` to
    the two phrases it joined, and its id is its position in `merges`, not a field.
    Phrases are frozen and compare by value, so one can be a dict key.
    """

    encoding: Optional[int] = None
    first: Optional["Phrase"] = None
    second: Optional["Phrase"] = None

    def __post_init__(self) -> None:
        """Raises ValueError unless exactly one of `encoding` or `first`+`second` is set."""
        is_leaf = self.encoding is not None
        is_merge = self.first is not None and self.second is not None
        if is_leaf == is_merge:
            raise ValueError(  # noqa: NAR001
                "Phrase needs encoding, or first and second, but not both"
            )

    def to_str(self, vocab: Dict[int, str]) -> str:
        """Render this phrase as a string, resolving merges recursively."""
        if self.encoding is not None:
            return vocab[self.encoding]
        assert self.first is not None and self.second is not None
        return self.first.to_str(vocab=vocab) + self.second.to_str(vocab=vocab)


# eq=False keeps identity comparison: a field-wise __eq__ would recurse through
# prev/next.
@dataclass(eq=False)
class PhraseNode:
    """One adjacent pair of tokens in the training text; internal to `train`.

    `phrase` is the pair, stored as a merge `Phrase`. `train` rewrites it in place
    when a merge absorbs one of its two tokens.
    """

    phrase: Phrase
    prev: Optional["PhraseNode"] = None
    next: Optional["PhraseNode"] = None


def _count_pairs(
    chunks: List[bytes],
) -> DefaultDict[Phrase, List[PhraseNode]]:
    """Link adjacent token pairs into one doubly linked list per chunk, by pair.

    Node i of a chunk holds the pair (chunk[i], chunk[i + 1]), so neighbouring
    nodes share one token. Each list ends at its chunk's boundary, so no pair
    spans two chunks.

    Args:
        chunks: Pre-tokenized text, one bytes object per chunk.

    Returns:
        A map from each pair to the nodes where it occurs, in text order.
    """
    phrase_counter: DefaultDict[Phrase, List[PhraseNode]]
    phrase_counter = defaultdict(list)  # noqa: NAR001
    for chunk in chunks:
        encoded_text = [Phrase(encoding=b) for b in chunk]
        head = PhraseNode(phrase=Phrase(encoding=0))
        prev = head
        for i in range(len(encoded_text) - 1):  # noqa: NAR001
            phrase = Phrase(first=encoded_text[i], second=encoded_text[i + 1])
            node = PhraseNode(phrase=phrase, prev=prev)
            prev.next = node
            prev = node
            phrase_counter[phrase].append(node)  # noqa: NAR001
        if head.next:
            head.next.prev = None
    return phrase_counter


# TODO: make it training time specific
def train(text: str, num_merges: int = 500) -> Tuple[List[Phrase], Dict[int, str]]:
    """Learn up to num_merges BPE merges from text.

    Whitespace is normalized to EOW, the text is split into chunks by
    _CHUNK_PATTERN, and each chunk into UTF-8 bytes. Each step merges the most
    frequent adjacent pair everywhere it occurs, left to right, so an overlapping
    run like "aaa" becomes "aa" + "a". Merges never cross a chunk boundary.
    Training stops early once no pair occurs at least twice.

    Args:
        text: Training corpus.
        num_merges: Maximum number of merges to learn.

    Returns:
        merges: Merged pairs in the order they were learned; merges[k] is token
            FIRST_MERGE_ID + k.
        vocab: Token id -> string. Ids 0-255 map byte b to chr(b), so multi-byte
            UTF-8 characters appear Latin-1 decoded. Protected ids map to their
            _PROTECTED_STRINGS entry, so len(vocab) is the embedding size.
    """
    vocab = _base_vocab()
    merges: List[Phrase] = []
    phrase_keeper = _count_pairs(chunks=_pretokenize(text=text))
    phrase_pq = MaxHeap(
        items=(
            PrioritizedItem(priority=len(nodes), item=phrase)  # noqa: NAR001
            for phrase, nodes in phrase_keeper.items()
        )
    )
    for _ in range(num_merges):  # noqa: NAR001
        if phrase_pq.size() == 0:
            break
        max_phrase = phrase_pq.pop()
        if max_phrase.priority < 2:
            break

        updates: DefaultDict[Phrase, int] = defaultdict(int)  # noqa: NAR001
        new_id = FIRST_MERGE_ID + len(merges)  # noqa: NAR001
        vocab[new_id] = max_phrase.item.to_str(vocab=vocab)
        merges.append(max_phrase.item)  # noqa: NAR001

        # Lists are in text order, so overlapping runs ("aaa") merge left to right.
        # Nodes are never removed from phrase_keeper, so skip any whose pair has
        # changed since, e.g. absorbed by the overlapping occurrence to its left.
        for phrase_node in phrase_keeper[max_phrase.item]:
            if phrase_node.phrase != max_phrase.item:
                continue
            if phrase_node.prev:
                updates[phrase_node.prev.phrase] -= 1
                new_prev_phrase = Phrase(
                    first=phrase_node.prev.phrase.first, second=phrase_node.phrase
                )
                phrase_node.prev.phrase = new_prev_phrase
                if new_prev_phrase in phrase_pq:
                    updates[new_prev_phrase] += 1
                else:
                    phrase_pq.add_item(item=new_prev_phrase, priority=1)
                phrase_node.prev.next = phrase_node.next
                phrase_keeper[new_prev_phrase].append(phrase_node.prev)  # noqa: NAR001
            if phrase_node.next:
                updates[phrase_node.next.phrase] -= 1
                new_next_phrase = Phrase(
                    first=phrase_node.phrase, second=phrase_node.next.phrase.second
                )
                phrase_node.next.phrase = new_next_phrase
                if new_next_phrase in phrase_pq:
                    updates[new_next_phrase] += 1
                else:
                    phrase_pq.add_item(item=new_next_phrase, priority=1)
                phrase_node.next.prev = phrase_node.prev
                phrase_keeper[new_next_phrase].append(phrase_node.next)  # noqa: NAR001
        for phrase, update in updates.items():
            if phrase == max_phrase.item:
                continue  # already popped; only decremented by an overlapping run
            phrase_pq.update_priority(item=phrase, relative=update)
            if phrase_pq.get_priority(item=phrase) == 0:
                # Drop dead pairs so size() counts only pairs that still occur.
                phrase_pq.remove(item=phrase)
    return merges, vocab


def tokenizer_from_pairs(
    pairs: List[Tuple[int, int]],
) -> Tuple[List[Phrase], Dict[int, str]]:
    """Rebuild (merges, vocab) as train returns them from each merge's id pair.

    Args:
        pairs: pairs[k] is the (first_id, second_id) of merges[k]; ids are bytes
            (0-255) or FIRST_MERGE_ID + j for an earlier merge j < k.

    Returns:
        merges and vocab, in the same form as train.
    """
    # Pair ids index straight into phrases.
    phrases = {b: Phrase(encoding=b) for b in range(256)}  # noqa: NAR001
    vocab = _base_vocab()
    merges = []
    for rank, (first, second) in enumerate(pairs):  # noqa: NAR001
        merge = Phrase(first=phrases[first], second=phrases[second])
        phrases[FIRST_MERGE_ID + rank] = merge
        vocab[FIRST_MERGE_ID + rank] = merge.to_str(vocab=vocab)
        merges.append(merge)  # noqa: NAR001
    return merges, vocab


def encode(text: str, merges: List[Phrase]) -> List[int]:
    """Tokenize text with merges learned by train.

    Within each chunk, repeatedly applies the lowest-rank merge present, left to
    right. A merge can only create pairs containing its own output token, and every
    merge using that token ranks higher, so this matches replaying all merges in
    the order they were learned, skipping those that never occur. That reproduces
    the segmentation train would have produced. Every byte is in the base vocab, so
    there are no unknown tokens.

    Args:
        text: Text to tokenize; normalized and split into chunks as in train.
        merges: Merge list from train.

    Returns:
        Token ids; ids below 256 are raw bytes, FIRST_MERGE_ID + k is merges[k].
        Protected ids are never produced.
    """
    phrase_ids: Dict[Phrase, int] = {
        Phrase(encoding=b): b
        for b in range(256)  # noqa: NAR001
    }
    pairs: List[Tuple[int, int]] = []
    for rank, merge in enumerate(merges):  # noqa: NAR001
        assert merge.first is not None and merge.second is not None
        pairs.append(  # noqa: NAR001
            (phrase_ids[merge.first], phrase_ids[merge.second])
        )
        phrase_ids[merge] = FIRST_MERGE_ID + rank
    rank_of = {pair: rank for rank, pair in enumerate(pairs)}  # noqa: NAR001
    no_merge = len(pairs)  # noqa: NAR001
    encoded: List[int] = []
    for chunk in _pretokenize(text=text):
        ids = list(chunk)  # noqa: NAR001
        while len(ids) >= 2:  # noqa: NAR001
            rank = min(  # noqa: NAR001
                rank_of.get(pair, no_merge)  # noqa: NAR001
                for pair in zip(ids, ids[1:])  # noqa: NAR001
            )
            if rank == no_merge:
                break
            pair = pairs[rank]
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
        encoded.extend(ids)  # noqa: NAR001
    return encoded


def decode(ids: List[int], vocab: Dict[int, str]) -> str:
    """Turn token ids back into text.

    Lossy: whitespace comes back as EOW, since train normalized it away.

    Args:
        ids: Token ids from encode.
        vocab: Vocab from train.

    Returns:
        The decoded text, with protected ids dropped. Byte sequences that are not
        valid UTF-8 (possible only for id lists encode did not produce) become
        U+FFFD.
    """
    # vocab strings hold one Latin-1 char per byte; undo that to get the bytes.
    latin1 = "".join(  # noqa: NAR001
        vocab[i] for i in ids if i not in _PROTECTED_STRINGS
    )
    return latin1.encode("latin-1").decode(  # noqa: NAR001
        "utf-8", errors="replace"
    )


def _base_vocab() -> Dict[int, str]:
    """Vocab before any merges: byte b maps to chr(b), plus _PROTECTED_STRINGS."""
    vocab = {i: chr(i) for i in range(256)}  # noqa: NAR001
    vocab.update(_PROTECTED_STRINGS)  # noqa: NAR001
    return vocab


def _pretokenize(text: str) -> List[bytes]:
    """Replace every whitespace char with EOW and split into _CHUNK_PATTERN chunks.

    Returns:
        Each chunk's UTF-8 bytes, in text order. Joined, they are the normalized
        text, since the pattern matches every character.
    """
    normalized = text.translate(_WHITESPACE_TO_EOW)  # noqa: NAR001
    return [
        chunk.encode("utf-8")  # noqa: NAR001
        for chunk in _CHUNK_PATTERN.findall(normalized)  # noqa: NAR001
    ]
