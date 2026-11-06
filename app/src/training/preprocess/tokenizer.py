"""The `Tokenizer` interface callers code against, and `get_tokenizer`, which makes one.

A loader or a model needs four things from a tokenizer: text to ids, ids back to
text, a vocab whose length sizes the embedding, and a `cache_key` so encodings can be
cached. `Tokenizer` is exactly that. `BpeTokenizer` in `bpe/tokenizer.py` is the one
implementation; code that needs BPE's merges, chunks, or `state()` imports `bpe/`
directly.

`get_tokenizer` identifies a tokenizer by the file its merges live in. The caller picks
the path and puts in it whatever tells two tokenizers apart, such as the corpus name
and `num_merges`. An existing file loads as is, so the file, not the arguments, decides
what comes back: reusing a path after changing the training texts or `num_merges`
loads the old merges until the file is deleted. A missing file is trained from
`train_texts` and written.

The file is the JSON list of `[first_id, second_id]` pairs that `BpeTokenizer.state`
returns. Merges saved under an older `__version__` still load and encode unchanged.
Adding a protected id shifts every merge id, and loading a file saved before that
fails validation in `build_tokenizer` rather than encoding wrongly.
"""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Sequence

from loguru import logger
from training.dal.cache import CacheKeyed
from training.dal.dao.dao import Dao
from training.preprocess.bpe import gpt4_preprocessor, train
from training.utils.timing import log_elapsed


class Tokenizer(CacheKeyed, Protocol):
    """Text to ids and back, independent of how the ids were learned.

    Attributes:
        vocab: Token id -> string, protected ids included, so `len(vocab)` is the
            embedding size.
    """

    vocab: Dict[int, str]

    def encode(self, texts: Sequence[str], cache: Optional[Dao] = None) -> List[List[int]]:
        """Return one id list per text, in order; never a protected id.

        Args:
            cache: Where encodings are stored, keyed by `cache_key`; None stores
                nothing. The caller must keep it open for the call.
        """
        ...

    def decode(self, ids: Iterable[Sequence[int]]) -> List[str]:
        """Return the text each id list encodes, with protected ids dropped."""
        ...


def get_tokenizer(
    merges_path: Path,
    train_texts: Optional[List[str]] = None,
    num_merges: Optional[int] = None,
) -> Tokenizer:
    """Load the tokenizer saved at `merges_path`, training and saving it if absent.

    Args:
        merges_path: The JSON file of merges; see the module docstring.
        train_texts: Text to train on, one string per row. Read only when the file
            is missing.
        num_merges: Merges to train; None trains `train.DEFAULT_NUM_MERGES`. Ignored
            when the file exists.

    Raises:
        FileNotFoundError: If the file is missing and no train_texts are given.
        ValueError: If the file holds a pair that is not two known ids.

    Side effects:
        On a miss, trains and writes `merges_path`, creating its parent directories.
    """
    if merges_path.exists():
        merges = json.loads(s=merges_path.read_text(encoding="utf-8"))
        logger.info("loaded tokenizer {} ({} merges)", merges_path, len(merges))  # noqa: NAR001
        return train.build_tokenizer(saved_state=merges)
    if train_texts is None:
        raise FileNotFoundError(  # noqa: NAR001
            f"no merges at {merges_path}; pass train_texts to train and save them"
        )

    with log_elapsed(label=f"BPE trained merges for {merges_path.name}"):
        tokenizer = train.build_tokenizer(
            chunks=gpt4_preprocessor.preprocess(text="\n".join(train_texts)),  # noqa: NAR001
            num_merges=num_merges,
        )
    merges_path.parent.mkdir(parents=True, exist_ok=True)
    merges_path.write_text(data=json.dumps(obj=tokenizer.state()), encoding="utf-8")
    return tokenizer
