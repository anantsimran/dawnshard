# Transformer

This package owns the transformer building blocks: attention, masks, the encoder and
decoder blocks, the decoder's KV cache, position encodings, an attention-map probe,
and the axis names for tensor shapes. The attention function returns its weights, and the normalization,
residual, and mask operations are visible in the source.

| Owns | Imports from the library | Never sees |
|---|---|---|
| attention, masks, blocks, the KV cache, position encodings, axis names | nothing | the tokenizer, datasets, the training loop |

[test_layout.py](../../../tests/test_layout.py) fails if this package imports anything
else from `training/`. Attention does not know a loop exists. `save_attention_maps`
in [probes.py](probes.py) still plugs into the loop's `eval_probe` slot, because a
probe is any callable with the right signature. For the training architecture, see
the [root README](../../../../README.md).

______________________________________________________________________

## Attention

Attention is implemented directly so score calculation and mask semantics stay
visible.

`attention` is a plain function. It takes queries, keys, and values that are already
split into heads, with shape `(B, h, L, d_k)`, and returns the attention weights along
with the output, so you can look at what each head attends to. It scales the queries
before the matmul rather than the scores after it, which touches `B·h·L·d_k` elements
instead of `B·h·L·L`. The tests compare its output with
[`torch.nn.functional.scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).
That checks the calculation; it is not a performance benchmark.

`MultiHeadAttentionLayer` holds the parameters, and `EncoderBlock` wraps it with a
pre-norm residual and a position-wise feed-forward network. Both pass the weights up
as `(out, weights)`. The layer's query, key, and value weights have shape
`(h, d_model, d_k)`, a separate matrix for each head, so one broadcast matmul projects
the input into every head and there is no reshape-and-transpose step to split them.
The heads only meet again at the output projection.

A few smaller choices:

- `qk_norm=True` applies RMSNorm to each head's queries and keys, which keeps the
  attention scores bounded as the weights grow.
- `masked_mean` averages over real tokens only, and returns zeros for a row that's
  all padding.
- `EncoderBlock` and `DecoderBlock` each share one `nn.Dropout` across their
  sublayers, because a Dropout holds no state and samples a fresh mask on every call.

```python
from preprocess.constants import PAD_ID
from transformer.masks import mask_tokenizer_pad_ids
from transformer.modules import MultiHeadAttentionLayer

layer = MultiHeadAttentionLayer(d_model=128, h=8, qk_norm=True)
keep = mask_tokenizer_pad_ids(token_ids=token_ids, pad_id=PAD_ID)  # (B, 1, 1, L)
out, weights = layer(x=embeddings, mask=keep)  # (B, L, d_model), (B, h, L, L)
```

### Inside a block: norm in, dropout out

`EncoderBlock` and `DecoderBlock` are pre-norm, and every sublayer (self-attention,
cross-attention, or the FFN) is wrapped the same way:

```
x ──┬───────────────────────────────────────────► + ──► x
    │                                             ▲
    └─► <sublayer>_norm ─► <sublayer> ─► dropout ─┘
```

- **The norm goes before the sublayer.** It normalizes the sublayer's input and leaves
  the residual stream `x` alone, so the block's input reaches its output through
  adds only, with no norm on the way. Each sublayer reads `norm(x)` from the running
  sum, never the previous sublayer's raw output.
- **Dropout goes after the sublayer**, on its output and before the residual add. It
  drops part of what the sublayer contributes, never `x` itself.
- **The stack's output is not normalized.** Each block adds to `x` without norming
  it, so a final norm after the last block is the caller's job. In an
  encoder-decoder, the encoder's final-norm output is the `memory` every decoder
  block reads.

Each norm is named after the sublayer it feeds, and each dropout after the sublayer
whose output it drops:

| Block | Sublayer | Norm before it | Dropout after it |
|---|---|---|---|
| `EncoderBlock` | `attn` | `attn_in_norm` | `dropout`, shared |
| `EncoderBlock` | `ffn` | `ffn_norm` | `dropout`, shared |
| `DecoderBlock` | `self_attn` | `self_attn_norm` | `dropout`, shared |
| `DecoderBlock` | `cross_attn` | `cross_attn_norm` | `dropout`, shared |
| `DecoderBlock` | `ffn` | `ffn_norm` | `dropout`, shared |

### Masks

Masks are boolean, and `True` means a position can be attended to. A mask always
indexes the key axis and may index the query axis; its other axes are size 1, so it
broadcasts over the scores `(B, h, L, S)`. [masks.py](masks.py) builds two kinds:

- **`mask_tokenizer_pad_ids`**, shape `(B, 1, 1, L)`: `True` where the key is a real token.
  It hides key columns only, never whole query rows, because a fully masked row turns
  into NaN after softmax.
- **`mask_later_positions`**, shape `(1, 1, L, L)`: `True` at `(i, j)` iff `j <= i`, so
  no position sees a later one.

Combine them with `&`. Which mask a sublayer takes depends on where its keys come
from:

| Sublayer | Keys come from | Mask | Shape |
|---|---|---|---|
| Encoder self-attention | the source | source padding | `(B, 1, 1, L)` |
| Decoder self-attention | the target so far | causal `&` target padding | `(B, 1, L, L)` |
| Decoder cross-attention | the encoder output | source padding | `(B, 1, 1, S)` |

Because a mask describes keys, cross-attention takes the source padding mask, never
the target's: the decoder's queries may read every real source token, and only the
source pads are hidden. It takes no causal mask either, because the whole source is
known before decoding starts.

In decoder self-attention, the causal mask alone already keeps real queries off the
pads when the target is right-padded, since every pad comes after them. Adding the
target padding mask costs nothing and stays correct under left padding. It never
masks a whole row as long as the first target token is not padding, because every
query keeps key 0.

Build the masks once per batch and pass the same ones to every block in the stack:

```python
from preprocess.constants import PAD_ID
from transformer.masks import mask_later_positions, mask_tokenizer_pad_ids

src_keep = mask_tokenizer_pad_ids(token_ids=src_ids, pad_id=PAD_ID)  # (B, 1, 1, S)
tgt_keep = mask_later_positions(
    seq_len=tgt_ids.shape[1], device=tgt_ids.device
) & mask_tokenizer_pad_ids(token_ids=tgt_ids, pad_id=PAD_ID)  # (B, 1, L, L)

memory = src_embeddings
for block in encoder_blocks:
    memory, _ = block(batch=memory, pad_mask=src_keep)

x = tgt_embeddings
for block in decoder_blocks:
    x, _, _ = block(x=x, memory=memory, x_mask=tgt_keep, memory_mask=src_keep)
```

### KV cache

Decoding one token at a time changes only where attention gets its keys and
values. `MultiHeadAttentionLayer.forward` takes an optional
`KVCache`, and `DecoderBlock.forward` takes a `DecoderBlockCache` that holds one for
each attention sublayer. Without a cache, every call runs exactly as before, so
training never touches one.

- **The caller owns the cache.** It allocates the buffers for one decode, passes
  them in on every step, and drops them afterwards. A layer writes into the cache it
  receives and stores nothing on `self`, so modules still hold no state between
  calls, and two decodes can share one model.
- **Buffers start at full capacity.** `KVCache.append` writes the new rows after
  `length_filled` and returns a view of the filled rows, so no step concatenates or
  copies the cache. Self-attention needs room for the longest target the decode can
  reach. Cross-attention needs exactly S rows.
- **Cross-attention fills once.** `memory` stays fixed for a whole decode, so the
  first call projects it into the cache and later calls read it back. A later call
  raises if `memory` no longer has `length_filled` rows.
- **Keys are cached after `k_norm`.** The cache holds what attention consumed, so
  QK-norm runs once per token.
- **A single new position may pass `x_mask=None`**, since the newest token may read
  every cached key. For more than one position, `DecoderBlock.forward` raises on
  `None`: without a causal mask, earlier positions would read later ones.
- **Positions belong to the caller too.** The newest token is always at index 0 of
  what the block sees, so the caller adds row t of the position table itself, for
  example with `SinusoidalEmbedding.at_pos(pos=t)`.

With a cache, the self-attention weights are `(B, h, L, T)`, with L new queries
against the T rows filled so far. Without one, T equals L.

```python
from transformer.modules import DecoderBlockCache, KVCache


def empty_cache(rows):
    shape = (memory.shape[0], h, rows, d_k)
    return KVCache(k=memory.new_empty(size=shape), v=memory.new_empty(size=shape))


caches = [
    DecoderBlockCache(
        self_attn=empty_cache(rows=max_len),
        cross_attn=empty_cache(rows=memory.shape[1]),
    )
    for _ in decoder_blocks
]
for t in range(max_len):
    x = embed(token) + position_table.at_pos(pos=t)  # (B, 1, d_model)
    for block, cache in zip(decoder_blocks, caches):
        x, _, _ = block(
            x=x,
            memory=memory,
            x_mask=None,
            memory_mask=src_keep,
            decoder_block_cache=cache,
        )
    token = next_token_from(x)
```

______________________________________________________________________

## Axis names

Every tensor shape in the transformer package, whether in a jaxtyping annotation, a
docstring, or a comment, uses these symbols:

| Symbol | Constant | Name | Example value | Set by |
|---|---|---|---|---|
| `B` | `B` | batch | 32 | training loop |
| `L` | `L` | sequence length | 128 | tokenizer + padding |
| `S` | `S` | key/value length, equal to `L` in self-attention | 128 | memory's tokenizer + padding |
| `T` | `T` | self-attention key length with a KV cache, the prefix decoded so far; equal to `L` without one | 17 | decode step |
| `d_model` | `D_MODEL` | embedding width | 512 | model config |
| `h` | `H` | heads | 8 | model config |
| `d_k` | `D_K` | width per head | 64 | `d_model // h` |
| `d_ff` | `D_FF` | FFN hidden width | 2048 | `d_ff_multiplier * d_model` |
| `seq_length` | `SEQ_LENGTH` | rows of a position table, the longest `L` it accepts | 512 | model config |
| `vocab` | `V` | vocabulary size | 32000 | tokenizer |

The constants in [constants.py](constants.py) hold the axis *names*, not sizes, so
annotations build their shape strings from them. Sizes are bound per call, and
jaxtyping checks that the axes agree within that call:

```python
from training.transformer.constants import B, D_MODEL, L

def forward(
    self, batch: Float[torch.Tensor, f"{B} {L} {D_MODEL}"]
) -> Float[torch.Tensor, f"{B} {L} {D_MODEL}"]: ...
```

The forward-pass diagrams in the module docstrings label weights with the same
symbols. A label with a leading dot, such as `.weight (vocab, d_model)` beside
`embedding`, names a parameter or buffer of the module on that line, and its shape is
the stored shape that `named_parameters()` prints. `nn.Linear` stores `(out, in)` and
computes `x @ weight.T`, while `w_q` is stored `(h, d_model, d_k)` and applied as
`x @ w_q`, so the two read in opposite orders. A diagram labels only the tensors its
own module creates, which is why the attention projections appear in
`MultiHeadAttentionLayer` and nowhere else. Biases and norm scales are left out,
since each is a vector as wide as the axis it acts on.
