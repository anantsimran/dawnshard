# Transformer

Attention, masks, and the encoder and decoder blocks, built from scratch. For the rest of the
library, see the [root README](../../../../README.md).

______________________________________________________________________

## Attention

Attention is implemented directly instead of using `nn.MultiheadAttention` for learning purposes.

`attention` is a plain function. It takes queries, keys, and values that are already
split into heads, with shape `(B, h, L, d_k)`, and returns the attention weights along
with the output, so you can look at what each head attends to. It scales the queries
before the matmul rather than the scores after it, which touches `B·h·L·d_k` elements
instead of `B·h·L·L`. The test suite checks it against
`torch.nn.functional.scaled_dot_product_attention`.

`MultiHeadAttentionLayer` holds the parameters, and `EncoderBlock` wraps it with a
pre-norm residual and a position-wise feed-forward network. Both pass the weights up
as `(out, weights)`. The layer's query, key, and value weights have shape
`(h, d_model, d_k)`, a separate matrix for each head, so one broadcast matmul projects
the input into every head and there is no reshape-and-transpose step to split them.
The heads only meet again at the output projection.

A few smaller choices:

- `qk_norm=True` applies RMSNorm to each head's queries and keys, which keeps the
  attention scores bounded as the weights grow.
- `pad_masked_mean` averages over real tokens only, and returns zeros for a row that's
  all padding.
- `EncoderBlock` and `DecoderBlock` each share one `nn.Dropout` across their
  sublayers, because a Dropout holds no state and samples a fresh mask on every call.

```python
from common.constants import PAD_ID
from transformer.mask import padding_keep_mask
from transformer.modules import MultiHeadAttentionLayer

layer = MultiHeadAttentionLayer(d_model=128, h=8, qk_norm=True)
keep = padding_keep_mask(token_ids=token_ids, pad_id=PAD_ID)  # (B, 1, 1, L)
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
broadcasts over the scores `(B, h, L, S)`. [mask.py](mask.py) builds two kinds:

- **`padding_keep_mask`**, shape `(B, 1, 1, L)`: `True` where the key is a real token.
  It hides key columns only, never whole query rows, because a fully masked row turns
  into NaN after softmax.
- **`causal_keep_mask`**, shape `(1, 1, L, L)`: `True` at `(i, j)` iff `j <= i`, so
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
from common.constants import PAD_ID
from transformer.mask import causal_keep_mask, padding_keep_mask

src_keep = padding_keep_mask(token_ids=src_ids, pad_id=PAD_ID)  # (B, 1, 1, S)
tgt_keep = causal_keep_mask(
    seq_len=tgt_ids.shape[1], device=tgt_ids.device
) & padding_keep_mask(token_ids=tgt_ids, pad_id=PAD_ID)  # (B, 1, L, L)

memory = src_embeddings
for block in encoder_blocks:
    memory, _ = block(batch=memory, pad_mask=src_keep)

x = tgt_embeddings
for block in decoder_blocks:
    x, _, _ = block(x=x, memory=memory, x_mask=tgt_keep, memory_mask=src_keep)
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
| `d_model` | `D_MODEL` | embedding width | 512 | model config |
| `h` | `H` | heads | 8 | model config |
| `d_k` | `D_K` | width per head | 64 | `d_model // h` |
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
