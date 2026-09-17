# Transformer

Attention, masks, and the encoder block, built from scratch. For the rest of the
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

Masks are boolean, and `True` means a position can be attended to.
`padding_keep_mask` has shape `(B, 1, 1, L)` and `causal_keep_mask` has shape
`(1, 1, L, L)`; combine them with `&`. Padding masks hide key columns only, never
whole query rows, because a fully masked row turns into NaN after softmax.

A few smaller choices:

- `qk_norm=True` applies RMSNorm to each head's queries and keys, which keeps the
  attention scores bounded as the weights grow.
- `pad_masked_mean` averages over real tokens only, and returns zeros for a row that's
  all padding.
- `EncoderBlock` shares one `nn.Dropout` between its two sublayers, because a Dropout
  holds no state and samples a fresh mask on every call.

```python
from common import bpe
from transformer.mask import padding_keep_mask
from transformer.modules import MultiHeadAttentionLayer

layer = MultiHeadAttentionLayer(d_model=128, h=8, qk_norm=True)
keep = padding_keep_mask(token_ids=token_ids, pad_id=bpe.PAD_ID)  # (B, 1, 1, L)
out, weights = layer(batch=embeddings, pad_mask=keep)  # (B, L, d_model), (B, h, L, L)
```

______________________________________________________________________

## Axis names

Every tensor shape in the transformer package, whether in a jaxtyping annotation, a
docstring, or a comment, uses these symbols:

| Symbol | Constant | Name | Example value | Set by |
|---|---|---|---|---|
| `B` | `B` | batch | 32 | training loop |
| `L` | `L` | sequence length | 128 | tokenizer + padding |
| `d_model` | `D_MODEL` | embedding width | 512 | model config |
| `h` | `H` | heads | 8 | model config |
| `d_k` | `D_K` | width per head | 64 | `d_model // h` |

The constants in [constants.py](constants.py) hold the axis *names*, not sizes, so
annotations build their shape strings from them. Sizes are bound per call, and
jaxtyping checks that the axes agree within that call:

```python
from training.transformer.constants import B, D_MODEL, L

def forward(
    self, batch: Float[torch.Tensor, f"{B} {L} {D_MODEL}"]
) -> Float[torch.Tensor, f"{B} {L} {D_MODEL}"]: ...
```

`attention` also uses `d_v`, the value width. It is separate from `d_k` because
nothing forces them to be equal. In `MultiHeadAttentionLayer`, `d_v = d_k`.
