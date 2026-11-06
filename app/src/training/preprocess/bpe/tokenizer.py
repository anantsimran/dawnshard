"""Byte-level BPE: `BpeTokenizer`, the `Tokenizer` that `get_tokenizer` returns.

BPE starts from the 256 byte values and learns merges: adjacent token pairs frequent
enough to get an id of their own. Frequent words become one token, and rare ones fall
back to smaller pieces, down to single bytes, so no input is unknown.

The class holds the merges and the tables built from them, and does nothing else.
The work is in three modules of public functions, each with its own docstring:

- `train.py`: `build_tokenizer` learns merges from chunks, or loads saved ones.
- `encode.py`: `encode_chunks` turns chunks into ids, through a `Dao` if given.
- `decode.py`: `decode_ids` turns ids back into bytes.

`BpeTokenizer.encode` and `decode` wrap the last two with `gpt4_preprocessor`, so a
caller passes text and gets text back. The package README shows how to train, save,
extend, and batch.

```python
from training.preprocess.bpe import gpt4_preprocessor, train

chunks = gpt4_preprocessor.preprocess(text="low lower lowest")
tokenizer = train.build_tokenizer(chunks=chunks, num_merges=4)
tokenizer.merges  # [(108, 111), (262, 119), (32, 263), (264, 101)]
ids = tokenizer.encode(texts=["lowest\tlow"])
# [[263, 101, 115, 116, 264]]
[tokenizer.vocab[i] for i in ids[0]]  # ['low', 'e', 's', 't', ' low']
tokenizer.decode(ids=ids)             # ['lowest low'] (tab lost in preprocess)
```

Token ids:

```
  0 ─ 255                 raw bytes; vocab[b] = chr(b)
  256 ─ 261               protected: PAD, MASK, CLS, SEP, BOS, EOS (constants.py)
  262 ─ 261+len(merges)   merges; FIRST_MERGE_ID + k joins the pair merges[k]
```

The protected ids sit between the bytes and the merges, so they keep their ids whatever
`num_merges` is. `encode` never produces them and `decode` drops them. `vocab` holds
one Latin-1 character per byte, so "é" (bytes 195, 169) shows up there as 'Ã©';
`decode` returns the real text.

`__version__` changes whenever the ids `encode` produces, or the merges training learns
from the same chunks, change. Each tokenizer's `tokenizer_id` hashes it together with
the merges, so a cached encoding is never served for a tokenizer that would encode
differently.
"""

import hashlib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from training.dal.dao.dao import Dao
from training.preprocess.bpe import gpt4_preprocessor
from training.preprocess.bpe.decode import decode_ids
from training.preprocess.bpe.encode import encode_chunks
from training.preprocess.constants import (
    BOS_ID,
    CLS_ID,
    EOS_ID,
    FIRST_MERGE_ID,
    MASK_ID,
    PAD_ID,
    SEP_ID,
)

__version__ = "2.0.0"

# vocab string for each protected id; decode drops all of them.
_PROTECTED_STRINGS = {
    PAD_ID: "",
    MASK_ID: "[MASK]",
    CLS_ID: "[CLS]",
    SEP_ID: "[SEP]",
    BOS_ID: "[BOS]",
    EOS_ID: "[EOS]",
}


class BpeTokenizer:
    """A byte-level BPE tokenizer: encodes batches of texts, decodes batches of ids.

    Build one with `train.build_tokenizer`, which trains or loads the merges; the
    constructor takes them as they are. Nothing mutates a tokenizer after
    construction, so `encode` and `decode` are safe to share.

    Attributes:
        tokenizer_id: sha256 of `__version__` and `merges`; `cache_key` returns it, so
            two tokenizers with the same merges under the same `__version__` share
            cached encodings, and any other pair never does.
        merges: merges[k] is the (first_id, second_id) that token FIRST_MERGE_ID + k
            joins.
        rank_of: Each merge's pair mapped to k, the rank `encode_chunk` applies
            lowest first.
        vocab: Token id -> string. Ids 0-255 map byte b to chr(b), so multi-byte
            UTF-8 characters appear Latin-1 decoded. Protected ids map to their
            _PROTECTED_STRINGS entry, so len(vocab) is the embedding size.
    """

    def __init__(self, merges: Sequence[Sequence[int]]) -> None:
        """Build vocab and the rank table `encode_chunk` reads.

        Args:
            merges: (first_id, second_id) per merge, in rank order; each id is a byte
                or an earlier merge. Lists are accepted, as JSON loads them.
        """
        self.merges: List[Tuple[int, int]] = [(first, second) for first, second in merges]
        self.tokenizer_id = hashlib.sha256(
            string=f"{__version__}|{self.merges}".encode(encoding="utf-8")
        ).hexdigest()
        self.rank_of = {
            pair: rank
            for rank, pair in enumerate(self.merges)  # noqa: NAR001
        }
        self.vocab = _base_vocab()
        for rank, (first, second) in enumerate(self.merges):  # noqa: NAR001
            self.vocab[FIRST_MERGE_ID + rank] = self.vocab[first] + self.vocab[second]

    def cache_key(self) -> str:
        """Return `tokenizer_id`, so a tokenizer can stand in a `@cached` key."""
        return self.tokenizer_id

    def encode(self, texts: Sequence[str], cache: Optional[Dao] = None) -> List[List[int]]:
        """Preprocess each text and encode its chunks; one id list per text, in order.

        Args:
            texts: The texts to encode.
            cache: Where the encodings are stored, as `encode_chunks` describes; None
                stores nothing.

        Raises:
            TypeError: If texts is a single str rather than a sequence of them.
        """
        if isinstance(texts, str):  # noqa: NAR001
            raise TypeError(  # noqa: NAR001
                "encode takes a sequence of texts; wrap a single text in a list"
            )
        return encode_chunks(
            tokenizer=self,
            chunks=[gpt4_preprocessor.preprocess(text=text) for text in texts],
            cache=cache,
        )

    def decode(self, ids: Iterable[Sequence[int]]) -> List[str]:
        """Turn each id list back into text, dropping protected ids; one str per list.

        Whitespace comes back as single spaces, since `preprocess` normalized it away.
        """
        return [
            gpt4_preprocessor.to_text(chunks=data) for data in decode_ids(tokenizer=self, ids=ids)
        ]

    def state(self) -> List[Tuple[int, int]]:
        """Return each merge as its (first_id, second_id), for `build_tokenizer`'s saved_state."""
        return list(self.merges)  # noqa: NAR001


def _base_vocab() -> Dict[int, str]:
    """Vocab before any merges: byte b maps to chr(b), plus _PROTECTED_STRINGS."""
    vocab = {i: chr(i) for i in range(256)}  # noqa: NAR001
    vocab.update(_PROTECTED_STRINGS)  # noqa: NAR001
    return vocab
