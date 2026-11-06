"""Pre-tokenization for byte-level BPE: text in, one bytes object per chunk out.

This module owns the text side of tokenization, and `tokenizer.py` the id side.
`preprocess` runs before both training and encoding: it turns every whitespace
character into a space, splits the text with the GPT-4 (tiktoken `cl100k_base`)
pattern, and returns each chunk's UTF-8 bytes. BPE merges never cross a chunk.
`to_text` is the way back, from the bytes `decode_ids` returns to a str.

```python
from training.preprocess.bpe import gpt4_preprocessor

gpt4_preprocessor.preprocess(text="The dog.\tIt ran 12345 m")
# [b'The', b' dog', b'.', b' It', b' ran', b' ', b'123', b'45', b' m']
gpt4_preprocessor.to_text(chunks=b"The dog. It")  # 'The dog. It'
```

At each position, `_CHUNK_PATTERN` tries these rules in order:

1. Contractions: 's 't 're 've 'm 'll 'd, case-insensitive.
1. A run of letters, with at most one leading character that is neither a letter nor
   a digit, usually the space, sometimes a symbol such as "(".
1. A run of 1 to 3 digits.
1. A run of punctuation or symbols, with an optional leading space.
1. Whitespace. A run of spaces leaves its last space for the next word, so " b" is
   the same chunk whether 1 or 10 spaces came before it.

The pattern matches every character, so the chunks joined together are exactly the
normalized text. Chunks keep the vocabulary from spending merges on variants such as
"dog." and "dog!", and keep every token inside one word.

Choices that look like bugs but are intentional:

- **Whitespace is lossy.** Tabs and newlines become spaces, so `BpeTokenizer.decode`
  returns them as spaces.
- **A leading space or symbol joins the next word, never the previous one.** "The
  dog." is chunked as 'The', ' dog', '.', and "(hello" stays one chunk.
- **Numbers split into runs of at most 3 digits.** "12345" is chunked as '123', '45',
  and the space before a number is its own chunk.

Requires the `regex` package, because stdlib `re` has no `\\p{L}` / `\\p{N}`.

References:

- tiktoken, `tiktoken_ext/openai_public.py`: the `cl100k_base` pattern, which
  `_CHUNK_PATTERN` copies character for character.
- Radford et al., 2019, "Language Models are Unsupervised Multitask Learners",
  section 2.2: why BPE should not merge across character categories.
"""

import sys
from typing import List

import regex

EOW = " "  # all whitespace is normalized to it; chunks attach it to the next word
_WHITESPACE_TO_EOW = {
    c: EOW
    for c in range(sys.maxunicode + 1)  # noqa: NAR001
    if chr(c).isspace()  # noqa: NAR001
}
# GPT-4 (tiktoken cl100k_base) pre-tokenization pattern. Merges never cross the
# chunks it produces: contractions, letter runs with one optional leading
# non-letter (usually the space), digit runs of at most 3, punctuation runs, and
# whitespace.
_CHUNK_PATTERN = regex.compile(  # noqa: NAR001
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
)


def preprocess(text: str) -> List[bytes]:
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


def to_text(chunks: bytes) -> str:
    """Turn chunk bytes back into text, as `decode_ids` returns them.

    Whitespace comes back as single spaces, since `preprocess` normalized it away.
    Byte sequences that are not valid UTF-8, possible only for id lists `encode` did
    not produce or for a truncated row that cut a character, become U+FFFD.

    Args:
        chunks: The bytes of one or more consecutive chunks, joined.
    """
    return chunks.decode("utf-8", errors="replace")  # noqa: NAR001
