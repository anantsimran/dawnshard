"""Turn BPE token ids back into the bytes they encode.

`BpeTokenizer.decode` calls `decode_ids` and turns the bytes into text with
`gpt4_preprocessor.to_text`. Call `decode_ids` directly when you need the bytes, such
as to compare them with the chunks `encode_chunks` took.
"""

from typing import TYPE_CHECKING, Iterable, List, Sequence

from training.preprocess.constants import PROTECTED_IDS

if TYPE_CHECKING:
    from training.preprocess.bpe.tokenizer import BpeTokenizer

_PROTECTED = frozenset(PROTECTED_IDS)  # noqa: NAR001


def decode_ids(tokenizer: "BpeTokenizer", ids: Iterable[Sequence[int]]) -> List[bytes]:
    """Return the bytes each id list encodes, one bytes per list, in order.

    Protected ids are dropped. The bytes need not be valid UTF-8, as for a truncated
    row that cut a character in two.

    Args:
        ids: Token id lists, as `encode_chunks` returns them.
    """
    # vocab strings hold one Latin-1 char per byte; encoding undoes that.
    return [
        "".join(  # noqa: NAR001
            tokenizer.vocab[i] for i in row if i not in _PROTECTED
        ).encode("latin-1")  # noqa: NAR001
        for row in ids
    ]
