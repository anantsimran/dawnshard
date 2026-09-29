# Common

Building blocks shared across the library: [bpe.py](bpe.py), a byte-level BPE
tokenizer, and [priority_queue.py](priority_queue.py), the indexed max-heap it trains
with. For the rest of the library, see the [root README](../../../../README.md).

______________________________________________________________________

## BPE tokenizer

Byte-level Byte Pair Encoding with GPT-4 pre-tokenization. `train` learns merges once
from a corpus. `encode` and `decode` turn text into token ids and back, and never
learn anything new. Every byte is in the base vocabulary, so any input can be encoded
and there is no unknown token.

The algorithm, the token id layout, and the choices that look like bugs live in the
module docstring of [bpe.py](bpe.py). This page covers how to use it.

### Quick start

```python
from training.common import bpe

merges, vocab = bpe.train(text="low lower lowest", num_merges=4)
ids = bpe.encode(text="lowest\tlow", merges=merges)  # [261, 101, 115, 116, 262]
[vocab[i] for i in ids]                               # ['low', 'e', 's', 't', ' low']
bpe.decode(ids=ids, vocab=vocab)                      # 'lowest low' (tab lost)
```

Pre-tokenization needs the `regex` package (stdlib `re` has no `\p{L}` or `\p{N}`).
It is a project dependency, so `uv sync` installs it.

### Train on a corpus

```python
with open(file="corpus.txt", encoding="utf-8") as f:
    text = f.read()
merges, vocab = bpe.train(text=text, num_merges=8000)
```

- **`num_merges` is an upper bound.** Training stops earlier once no pair occurs at
  least twice.
- **The embedding table has `len(vocab)` rows:** 256 bytes, the 4 protected ids in
  [constants.py](constants.py) (`PAD_ID`, `MASK_ID`, `CLS_ID`, `SEP_ID`), and
  `len(merges)` merges.
- **Train on the training split only,** so validation and test text do not shape the
  vocabulary.
- **Memory grows with corpus size:** one linked-list node per adjacent byte pair.

### Encode and decode

```python
ids = bpe.encode(text="A man in a blue shirt.", merges=merges)
text = bpe.decode(ids=ids, vocab=vocab)
```

`encode` needs only `merges`, and `decode` needs only `vocab`. `decode` returns
whitespace-normalized text: tabs and newlines come back as spaces.

### Inspect tokens

```python
[vocab[i] for i in ids]                          # raw vocab strings
[bpe.decode(ids=[i], vocab=vocab) for i in ids]  # readable pieces
```

`vocab` stores one Latin-1 character per byte, so non-ASCII text looks garbled there:
" Männer" appears as `' MÃ¤nner'`. `decode` shows it correctly. A single id that
holds only part of a multi-byte character decodes to U+FFFD.

### Batch with padding

```python
from training.common.constants import PAD_ID

batch = [bpe.encode(text=t, merges=merges) for t in texts]
max_len = max(len(ids) for ids in batch)
padded = [ids + [PAD_ID] * (max_len - len(ids)) for ids in batch]
mask = [[i != PAD_ID for i in ids] for ids in padded]  # False at padding
```

- **`encode` never produces `PAD_ID`,** so `ids != PAD_ID` marks exactly the real
  tokens. Use the mask to exclude padding from mean pooling and attention.
- **`decode` drops `PAD_ID`,** so padded rows decode cleanly.
- **Truncating can split a character.** `ids[:max_len]` can cut a character stored as
  separate byte tokens, and `decode` then emits U+FFFD for the broken bytes.

### Save and load a tokenizer

Store each merge as its `(first_id, second_id)` pair, then rebuild with
`tokenizer_from_pairs`:

```python
import json

ids_of = {bpe.Phrase(encoding=b): b for b in range(256)}
pairs = []
for k, merge in enumerate(merges):
    pairs.append((ids_of[merge.first], ids_of[merge.second]))
    ids_of[merge] = bpe.FIRST_MERGE_ID + k
with open(file="tokenizer.json", mode="w") as f:
    json.dump(obj=pairs, fp=f)

with open(file="tokenizer.json") as f:
    merges, vocab = bpe.tokenizer_from_pairs(pairs=[tuple(p) for p in json.load(fp=f)])
```

The reloaded `merges` and `vocab` equal the originals. The pairs do not record how
the text was chunked, and merges learned under one pre-tokenization misapply under
another. So save `_CHUNK_PATTERN.pattern` next to the pairs, and retrain when it no
longer matches.

### Two languages: source and target

The API supports both ways of building the vocabulary:

| Option | How | Trade-off |
|---|---|---|
| Shared (joint) | One `train` on source and target training text together; encode both sides with the same `merges` | Names, numbers, and shared words segment the same way on both sides, and source and target can share an embedding table. The two languages split one vocabulary budget |
| Separate | One `train` per language; encode each side with its own `merges` | Each language gets the full budget, at the cost of two embedding tables |

Shared:

```python
merges, vocab = bpe.train(text=src_train_text + "\n" + tgt_train_text, num_merges=8000)
src = bpe.encode(text=src_sentence, merges=merges)
tgt = bpe.encode(text=tgt_sentence, merges=merges)
```

- **Non-English words stay whole.** `\p{L}` covers all Unicode letters, so ä, ö, ü,
  and ß stay in their word's chunk: "Zwei junge Männer" becomes `'Zwei'`, `' junge'`,
  `' Männer'`.
- **Whitespace normalization loses nothing on single-line sentences,** which is what
  a parallel corpus of sentence pairs usually holds.
- **A decoder needs start and end tokens,** which `bpe.py` does not define yet. See
  Open items.

### How text is chunked

All whitespace becomes a space, then `_CHUNK_PATTERN` splits the text into chunks.
Merges never cross a chunk boundary.

| Input | Chunks |
|---|---|
| `The dog. The dog!` | `'The' ' dog' '.' ' The' ' dog' '!'` |
| `Zwei junge Männer` | `'Zwei' ' junge' ' Männer'` |
| `I don't know` | `'I' ' don' "'t" ' know'` |
| `Price: 12345 dollars` | `'Price' ':' ' ' '123' '45' ' dollars'` |
| `(hello) The dog.` | `'(hello' ')' ' The' ' dog' '.'` |
| `a\tb\nc` | `'a' ' b' ' c'` |

At each position, the pattern tries these rules in order:

1. Contractions: `'s 't 're 've 'm 'll 'd`, case-insensitive.
1. A run of letters, with at most one leading character that is neither a letter nor
   a digit. Usually that character is the space, sometimes a symbol such as `(`.
1. A run of 1 to 3 digits.
1. A run of punctuation or symbols, with an optional leading space.
1. Whitespace. A run of spaces leaves its last space for the following word, so
   `' b'` is the same chunk whether 1 or 10 spaces came before it.

The pattern matches every character, so the chunks joined together are exactly the
normalized text.

### Open items

Not implemented. Each needs a decision first:

1. **BOS and EOS.** A translation decoder needs them. Adding them to
   `PROTECTED_IDS` shifts `FIRST_MERGE_ID`, so existing tokenizers would need
   retraining.
1. **Word ids from `encode`,** for whole-word masking in MLM.
1. **Training on unique chunk counts** instead of the full text. Memory would scale
   with unique chunks, not corpus bytes. Not needed while the corpus fits in memory.
1. **A per-chunk encode cache,** since a repeated chunk always encodes the same way.
1. **Lossless whitespace:** drop `_WHITESPACE_TO_EOW` so tabs and newlines survive.
