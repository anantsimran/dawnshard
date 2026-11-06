# Preprocess

This package owns tokenization. [tokenizer.py](tokenizer.py) holds the `Tokenizer`
interface that callers code against and `get_tokenizer`, which loads or trains one. The
byte-level BPE implementation lives in `bpe/`, and the protected token ids in
[constants.py](constants.py). A data pipeline trains a vocabulary once and applies the
same merges to every split without importing a model or a training loop.

| Owns | Imports from the library | Never sees |
|---|---|---|
| merges, vocabularies, encoding and decoding, protected ids | `dal`, `utils` | datasets, models, the training loop |

[test_layout.py](../../../tests/test_layout.py) fails if this package imports anything
from `training/` other than `dal` and `utils`, so the tokenizer stays usable on its own.
`dal` is how `encode` stores its results, and `utils` times training. For the full
architecture, see the [root README](../../../../README.md).

______________________________________________________________________

## Get a tokenizer

```python
from pathlib import Path

from training.preprocess.tokenizer import get_tokenizer

tokenizer = get_tokenizer(
    merges_path=Path("tokenizers/corpus-8000.json"), train_texts=texts, num_merges=8000
)
ids = tokenizer.encode(texts=["lowest\tlow"])
tokenizer.decode(ids=ids)  # ['lowest low']
```

The path identifies the tokenizer. The first call trains on `train_texts` and writes
the merges there; later calls load the file and ignore the other arguments. Put
whatever tells two tokenizers apart into the path, such as the corpus and `num_merges`,
and delete the file after changing the texts it trained on. Without `train_texts`, a
missing file raises `FileNotFoundError`.

`get_tokenizer` returns a `Tokenizer`: `encode` takes a list of texts, `decode` returns
one string per id list, `len(vocab)` is the embedding size, and `cache_key` keys cached
encodings. A loader or model that types against it never imports `bpe/`. BPE is the
only implementation today.

______________________________________________________________________

## BPE tokenizer

Byte-level Byte Pair Encoding with GPT-4 pre-tokenization. Each module owns one step:

- [gpt4_preprocessor.py](bpe/gpt4_preprocessor.py) owns text. `preprocess` turns a
  string into chunks of UTF-8 bytes, and `to_text` turns bytes back into a string.
- [train.py](bpe/train.py) owns learning. `build_tokenizer` learns merges from chunks,
  or loads saved ones.
- [encode.py](bpe/encode.py) turns chunks into ids with `encode_chunks`, and
  [decode.py](bpe/decode.py) turns ids back into bytes with `decode_ids`.
- [tokenizer.py](bpe/tokenizer.py) holds `BpeTokenizer`: the merges, the vocab, and
  the `encode` and `decode` that wrap the steps above for text.

Every byte is in the base vocabulary, so any text encodes and there is no unknown
token. The algorithm, the id layout, and the choices that look like bugs are in the
module docstrings.

### Quick start

```python
from training.preprocess.bpe import gpt4_preprocessor, train

chunks = gpt4_preprocessor.preprocess(text="low lower lowest")
tokenizer = train.build_tokenizer(chunks=chunks, num_merges=4)

ids = tokenizer.encode(texts=["lowest\tlow"])
# [[263, 101, 115, 116, 264]]
[tokenizer.vocab[i] for i in ids[0]]  # ['low', 'e', 's', 't', ' low']
tokenizer.decode(ids=ids)             # ['lowest low'] (tab lost)
```

`encode` takes a list of texts and `decode` returns one string per id list, both in
order. Passing a single string raises `TypeError`. To work with chunks and bytes
instead, call `encode_chunks` and `decode_ids`. `vocab` holds one Latin-1 character
per byte, so non-ASCII tokens look garbled there (" Männer" shows as `' MÃ¤nner'`);
`decode` shows them correctly.

### Build, load, or extend

| You pass | You get |
|---|---|
| `chunks`, optionally `num_merges` (default 500) | a tokenizer trained from scratch |
| `saved_state` | the saved tokenizer, unchanged |
| `chunks`, `saved_state`, and `num_merges` | the saved merges, plus new ones learned on `chunks` until there are `num_merges` in total |

Extending keeps every saved id, so an embedding table trained on them only needs rows
appended. `build_tokenizer` checks its arguments before doing any work, and if
`saved_state` already holds more than `num_merges` merges it logs a warning and keeps
them all.

### Train on a corpus

```python
with open(file="corpus.txt", encoding="utf-8") as f:
    text = f.read()
tokenizer = train.build_tokenizer(
    chunks=gpt4_preprocessor.preprocess(text=text), num_merges=8000
)
```

- **`num_merges` is an upper bound.** Training stops once no pair occurs twice.
- **The embedding table needs `len(tokenizer.vocab)` rows:** 256 bytes, the 6
  protected ids in [constants.py](constants.py), and one row per merge.
- **Train on the training split only,** so validation and test text don't shape the
  vocabulary.
- **The same chunks always give the same merges.** Ties go to the smaller
  `(first_id, second_id)`, so chunk order doesn't matter.

### Batch with padding

```python
from training.preprocess.constants import PAD_ID

batch = tokenizer.encode(texts=texts)
max_len = max(len(ids) for ids in batch)
padded = [ids + [PAD_ID] * (max_len - len(ids)) for ids in batch]
mask = [[i != PAD_ID for i in ids] for ids in padded]  # False at padding
```

`encode` never produces `PAD_ID` and `decode` drops it, so the mask marks exactly the
real tokens and padded rows decode cleanly. Truncating with `ids[:max_len]` can cut a
multi-byte character, and `decode` then shows U+FFFD in its place.

### Save and load

```python
import json

with open(file="tokenizer.json", mode="w") as f:
    json.dump(obj=tokenizer.state(), fp=f)

with open(file="tokenizer.json") as f:
    tokenizer = train.build_tokenizer(saved_state=json.load(fp=f))
```

`state()` returns every merge as a `(first_id, second_id)` pair, which is all a
tokenizer needs, and this is the file `get_tokenizer` writes. The pairs don't record
how the text was chunked, so delete the file and retrain when
`gpt4_preprocessor._CHUNK_PATTERN` changes.

### Cache encodings

```python
with SqliteDao(path=Path("cache.sqlite")) as cache:
    ids = tokenizer.encode(texts=texts, cache=cache)  # a repeat is a cache hit
```

With a `Dao`, `encode` stores what it returns, one entry per call. The key holds the
tokenizer's `tokenizer_id`, the sha256 of `__version__` and its merges, and the
distinct texts' chunks in sorted order. Two tokenizers with the same merges share
entries, and a reordered or duplicated batch still hits. The ids are the same with or
without a cache, and an empty input returns `[]` without touching it. Pass the `Dao`
while it is still open. When a change alters the ids `encode` produces, bump
`__version__` in [bpe/tokenizer.py](bpe/tokenizer.py).

### Two languages: source and target

| Option | How | Trade-off |
|---|---|---|
| Shared (joint) | One `build_tokenizer` on source and target training text together | Shared words segment the same way on both sides, and the embedding table can be shared, but the languages split one vocabulary budget |
| Separate | One `build_tokenizer` per language | Each language gets the full budget, at the cost of two embedding tables |

`BOS_ID` and `EOS_ID` are reserved in [constants.py](constants.py) for decoder
inputs and targets. `encode` never adds them; the caller places them around the
target ids.

### How text is chunked

Merges never cross a chunk boundary, so these splits decide what can merge:

| Input | Chunks |
|---|---|
| `The dog. The dog!` | `'The' ' dog' '.' ' The' ' dog' '!'` |
| `Zwei junge Männer` | `'Zwei' ' junge' ' Männer'` |
| `I don't know` | `'I' ' don' "'t" ' know'` |
| `Price: 12345 dollars` | `'Price' ':' ' ' '123' '45' ' dollars'` |
| `(hello) The dog.` | `'(hello' ')' ' The' ' dog' '.'` |
| `a\tb\nc` | `'a' ' b' ' c'` |

The rules behind them are in the [gpt4_preprocessor.py](bpe/gpt4_preprocessor.py)
docstring.

### Open items

Not implemented. Each needs a decision first:

1. **Word ids from `encode`,** for whole-word masking in MLM.
1. **A per-chunk encode cache,** since a repeated chunk always encodes the same way.
   The `Dao` cache stores whole batches of distinct texts today.
1. **Lossless whitespace:** drop `_WHITESPACE_TO_EOW` so tabs and newlines survive.
