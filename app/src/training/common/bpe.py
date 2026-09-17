"""Character-level Byte Pair Encoding: fit once, encode/decode many times."""

import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Optional, Tuple

from training.common.priority_queue import MaxHeap, PrioritizedItem

UNK = "<unk>"
# Ids 0-255 are raw bytes. PAD_ID sits between them and the merges so it is fixed
# regardless of num_merges and never produced by encode.
PAD_ID = 256
FIRST_MERGE_ID = PAD_ID + 1
EOW = " "  # end-of-word marker; all whitespace is normalized to it
_WHITESPACE_TO_EOW = {
    c: EOW
    for c in range(sys.maxunicode + 1)  # noqa: NAR001
    if chr(c).isspace()  # noqa: NAR001
}


@dataclass(frozen=True)
class Phrase:
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
    phrase: Phrase
    prev: Optional["PhraseNode"] = None
    next: Optional["PhraseNode"] = None


def _count_pairs(
    encoded_text: List[Phrase],
) -> Tuple[Optional[PhraseNode], DefaultDict[Phrase, List[PhraseNode]]]:
    """Link every adjacent token pair into a doubly linked list, indexed by pair.

    Node i holds the pair (encoded_text[i], encoded_text[i + 1]), so neighbouring
    nodes share one token.

    Args:
        encoded_text: Token sequence, one leaf Phrase per byte.

    Returns:
        The first node (None if there are fewer than two tokens) and a map from
        each pair to the nodes where it occurs, in text order.
    """
    phrase_counter: DefaultDict[Phrase, List[PhraseNode]]
    phrase_counter = defaultdict(list)  # noqa: NAR001
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
    return head.next, phrase_counter


# TODO: make it training time specific
def train(text: str, num_merges: int = 500) -> Tuple[List[Phrase], Dict[int, str]]:
    """Learn up to num_merges BPE merges from text.

    Whitespace is normalized to EOW and the text is split into UTF-8 bytes. Each
    step merges the most frequent adjacent pair everywhere it occurs, left to
    right, so an overlapping run like "aaa" becomes "aa" + "a". Merges may cross
    word boundaries (EOW is an ordinary byte). Training stops early once fewer than
    two distinct pairs remain or no pair occurs at least twice.

    Args:
        text: Training corpus.
        num_merges: Maximum number of merges to learn.

    Returns:
        merges: Merged pairs in the order they were learned; merges[k] is token
            FIRST_MERGE_ID + k.
        vocab: Token id -> string. Ids 0-255 map byte b to chr(b), so multi-byte
            UTF-8 characters appear Latin-1 decoded. PAD_ID maps to "", so
            len(vocab) is the embedding size.
    """
    phrases = [Phrase(encoding=utfbyte) for utfbyte in _normalize(text=text)]
    vocab = _base_vocab()
    merges: List[Phrase] = []
    _, phrase_keeper = _count_pairs(encoded_text=phrases)
    phrase_pq = MaxHeap(
        items=(
            PrioritizedItem(priority=len(nodes), item=phrase)  # noqa: NAR001
            for phrase, nodes in phrase_keeper.items()
        )
    )
    for _ in range(num_merges):  # noqa: NAR001
        if phrase_pq.size() < 2:
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

    Repeatedly applies the lowest-rank merge present, left to right. A merge can
    only create pairs containing its own output token, and every merge using that
    token ranks higher, so this matches replaying all merges in the order they were
    learned, skipping those that never occur. That reproduces the segmentation
    train would have produced. Every byte is in the base vocab, so there are no
    unknown tokens.

    Args:
        text: Text to tokenize; whitespace is normalized to EOW as in train.
        merges: Merge list from train.

    Returns:
        Token ids; ids below 256 are raw bytes, FIRST_MERGE_ID + k is merges[k].
        PAD_ID is never produced.
    """
    ids = list(_normalize(text=text))  # noqa: NAR001
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
    return ids


def decode(ids: List[int], vocab: Dict[int, str]) -> str:
    """Turn token ids back into text.

    Lossy: whitespace comes back as EOW, since train normalized it away.

    Args:
        ids: Token ids from encode.
        vocab: Vocab from train.

    Returns:
        The decoded text, with PAD_ID ids dropped. Byte sequences that are not
        valid UTF-8 (possible only for id lists encode did not produce) become
        U+FFFD.
    """
    # vocab strings hold one Latin-1 char per byte; undo that to get the bytes.
    latin1 = "".join(vocab[i] for i in ids)  # noqa: NAR001
    return latin1.encode("latin-1").decode(  # noqa: NAR001
        "utf-8", errors="replace"
    )


def _base_vocab() -> Dict[int, str]:
    """Vocab before any merges: byte b maps to chr(b), and PAD_ID maps to ""."""
    vocab = {i: chr(i) for i in range(256)}  # noqa: NAR001
    vocab[PAD_ID] = ""
    return vocab


def _normalize(text: str) -> bytes:
    """Replace every whitespace char with EOW and return the UTF-8 bytes."""
    return text.translate(_WHITESPACE_TO_EOW).encode("utf-8")  # noqa: NAR001
