# Attention Masks and Cross-Attention — Q&A

Vocabulary, matching `app/src/training/transformer/constants.py`:

| Symbol | Meaning |
|---|---|
| `B` | batch |
| `L` | sequence length |
| `d_model` | embedding width |
| `h` | attention heads |
| `d_k` | width per head, `d_model // h` |

When the queries and keys come from different sequences, `L` splits into `L_q` / `L_k`
(or `L_tgt` / `L_src` for translation). Attention scores are always
`(B, h, L_q, L_k)`: **rows = queries (readers), columns = keys (advertisers)**.

______________________________________________________________________

## Cheat-Sheet

**The one test.** For each *column* of a score matrix, ask: at inference, while generating
token `t`, do I physically have this?

| Answer | Action |
|---|---|
| Have it, and it's real | leave it |
| Have it, but it's `<pad>` | padding mask |
| Don't have it yet | causal mask |

Run the test on every attention block and the mask setup falls out of it. You don't need to memorize it.

| Block | Columns hold | Mask |
|---|---|---|
| Encoder self-attn | source tokens | padding |
| Decoder self-attn | target tokens (only `< t` exist) | padding + causal |
| Cross-attn (in decoder) | source tokens (encoder output, all available) | padding only |

**Mask shapes.** Each `1` means "the mask is the same along this axis":

```
(B, 1, 1,   L_k)   pad mask     → kills columns; same for every head and every query
(1, 1, L_q, L_k)   causal mask  → kills upper triangle; same for every batch and head
(B, 1, L_q, 1  )   row mask     → don't. Fully masked rows give NaN.
```

Combine with `keep_pad & keep_causal` → broadcasts to `(B, 1, L_q, L_k)`.

______________________________________________________________________

## Q: ELI5 cross-attention. I understand attention as "how can a token be best represented by other tokens, for a head."

Your definition already covers it. Change one phrase:

> how can a token be best represented by tokens **from a different sequence**

Same machinery, different pool of neighbours.

### Why Q is the one that stays home

The output has **one row per query**. Scores are `(B, h, L_q, L_k)`, and the output is
`(B, h, L_q, d_k)`. The softmax-weighted average sums the `L_k` axis away.

So whoever supplies **Q is the thing being represented**. Whoever supplies **K and V is
the material** it's built from. In translation you're building a better vector for the
French word you're about to write, so the French (decoder) position supplies Q. It's
shopping, and the encoder output is the shelf.

### Kid version

You're writing a French sentence. A friend has read the English sentence and made one
note-card for each English word.

- At each blank you ask: "I need the subject, a furry animal." That's your **query**.
- Each card has a label on the front: "I'm a noun, animal, the subject." That's the
  **key**. You compare your question against all the labels at once.
- You flip over the cards that matched and read the back, which holds the actual
  content. That's the **value**. You blend the backs, weighted by how well each front
  matched. The blend is your improved vector for this blank.

Key = what you search by. Value = what you take away. Keeping them separate means a card
can be *easy to find* for one reason and *useful* for another.

### Worked example

```
English (encoder output):  the(0)  black(1)  cat(2)  is(3)
French, already written:   "Le"
French, writing position 1: ___
```

Position 1's query, roughly: "subject noun, singular masculine, animal."

```
                the    black    cat     is
  q_1  scores [ 0.4    1.1     4.2     0.6  ]
       softmax[ 0.02   0.04    0.91    0.03 ]
```

Output = `0.02·v_the + 0.04·v_black + 0.91·v_cat + 0.03·v_is` ≈ mostly `v_cat`. That
vector goes up the stack, and the output layer reads it as *chat*. Nothing was looked up in a
dictionary. It's a soft blend of meanings, weighted by relevance.

At position 2 the query changes to "the adjective describing that noun," and the weight
lands on *black*. Same shelf, different question.

### Two attentions, two jobs

Every decoder layer runs both, in this order:

| | Question it answers | Mask |
|---|---|---|
| **Self-attention** | "What have I written so far?" Keeps the output grammatical and coherent | causal, because your own future doesn't exist yet |
| **Cross-attention** | "What am I supposed to be saying?" Keeps the output faithful to the source | none (besides padding), because the whole source is available |

Self-attention is re-reading your own draft. Cross-attention is glancing back at the
original. Without the first you write nonsense with bad grammar. Without the second you write
beautiful French that translates a different sentence.

### Nothing here is specific to language

Q and K only need to live in the same vector space so the dot product means something:

| Task | Q | K / V |
|---|---|---|
| Image captioning | words being written | image patches |
| Speech recognition | text | audio frames |
| Diffusion image models | image patches | text prompt |

Cross-attention is the general "let this thing look at that thing" connector. Translation
is just the task it was invented for.

______________________________________________________________________

## Q: Explain the masks used in the encoder and decoder. Why are they set up this way?

### The mistake is attaching masks to blocks

A mask isn't a property of "the decoder." It's a property of **the column axis of a specific
score matrix**. The only question to ask is:

> What lives in the columns, and is it available at inference time?

Run that on the two decoder blocks:

| Block | Columns hold | Available when generating token `t`? |
|---|---|---|
| Decoder self-attn | target tokens | Only those `< t`. The rest don't exist yet → **causal mask** |
| Cross-attn | source tokens | All of them, always. The encoder ran once, before decoding started → **nothing to hide** |

Cross-attention sits *inside* the decoder, but its keys are the *encoder's* output. The
source sentence is the input, and you had it before you generated anything. Hiding part of it
protects nothing and throws away information you already have.

### "But doesn't looking at the whole source leak the answer?"

No. The thing that must never leak is **future target tokens**, since those are what you're
graded on predicting. The encoder output is built only from source tokens and never
contains target tokens. The cross-attention key/value matrix holds no future-target
information, so there's nothing for a mask to block.

The autoregressive constraint is already fully enforced earlier in the layer. Follow the data for target
position `t`:

```
target tokens ≤ t  →  [decoder self-attn, causally masked]  →  query vector q_t
```

`q_t` depends only on past targets, by construction. It then reads the whole source.
The result depends on (past targets, full source), which is exactly what you have
at inference. Nothing leaks.

Masking cross-attention too would enforce the same constraint twice, on an axis where it
doesn't apply.

### Why a causal mask in cross-attention is wrong, not just useless

A triangular mask says "row `i` may only see columns `≤ i`." That assumes target
index and source index use the same coordinates. They don't: the lengths differ and so does word order.

English → French, where the adjective moves after the noun:

```
source:  the(0)  black(1)  cat(2)
target:  le(0)   chat(1)   noir(2)
```

The alignment crosses:

```
            the    black   cat
   le    [   ●       ·      ·  ]
  chat   [   ·       ·      ●  ]   ← needs source col 2
  noir   [   ·       ●      ·  ]   ← needs source col 1
```

A causal mask would let row `chat` (index 1) see only columns 0 and 1. Column 2, the word
it actually needs, would be forbidden. The mask would make the translation impossible.

That example is the easy case, because the lengths are equal. Real pairs differ, e.g. `L_src = 9`,
`L_tgt = 14`. The score matrix is `(B, h, L_tgt, L_src)`, and `tril` only makes sense on
a square where both axes index the same sequence. A square triangular mask doesn't even have
the right shape here.

### So what is the source padding mask doing?

It does a different job. Columns `L_src … L_max` are `<pad>` slots. They aren't secrets, just
meaningless embeddings that would otherwise get averaged into the output. You mask them for the
same reason as in encoder self-attention: they're noise. Padding is a side effect of
batching, not a modelling choice.

Keep the two motives separate:

- **Padding mask**: "this column is meaningless." It comes from how the data is batched. Apply it wherever those keys
  appear.
- **Causal mask**: "this column is a secret." It comes from the task. Apply it only where the
  columns are generated one token at a time.

Cross-attention has meaningless columns (pad) but no secret ones, so it gets padding only.

______________________________________________________________________

## Q: Does padding mean "this position advertises nothing about itself" or "this position isn't looking for anything"?

**The first.** Padding mask = "this position advertises nothing." Nobody may read from
it. The pad's *query row* is deliberately left alone.

### The matrix picture

`L = 4`. Positions 0–2 are real and position 3 is pad. Rows = queries (readers), columns =
keys (advertisers):

```
           k0    k1    k2    k3·pad
q0          .     .     .     -inf
q1          .     .     .     -inf
q2          .     .     .     -inf
q3·pad      .     .     .     -inf
```

The mask strikes out a **column**, not a row. Softmax runs along each row (every query
normalizes over all keys), so a `-inf` column gets zero weight in every row. No one ever
attends to the pad.

That column is exactly what `(B, 1, 1, L)` describes. The last axis is `L_k`, and the two
`1`s mean "the same column is struck out for every head and every query row." The mask is indexed
by key position only, and that's the whole reason for its shape.

### "So shouldn't the last row be zeroed?" No, for two reasons

**Practical:** row 3's output is garbage, and nobody reads it. The loss ignores pad positions
(`ignore_index=PAD`), and the next layer masks column 3 again, so the garbage can never
reach a real position.

**Fatal:** if you mask row 3 entirely, the row becomes all `-inf`:

```
softmax([-inf, -inf, -inf, -inf]) = 0 / 0 = NaN
```

NaN spreads through the backward pass into every weight in the model. Masking an entire
query row is the classic way to blow up training. The pad row is left to compute
meaningless but finite numbers on purpose.

______________________________________________________________________

## Q: I can't get past the syntax of `padding_keep_mask`

```python
def padding_keep_mask(token_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Mark real (non-pad) key positions.

    Args:
        token_ids: (B, L) integer token ids, padded with pad_id.
        pad_id: the id reserved for padding.

    Returns:
        (B, 1, 1, L) bool; True = this key position is readable. Singleton h and L_q
        axes broadcast over heads and readers.
    """
    return (token_ids != pad_id)[:, None, None, :]


src_keep = padding_keep_mask(token_ids=src_ids, pad_id=PAD)  # (B, 1, 1, L_src)
scores = scores.masked_fill(~src_keep, float("-inf"))
```

The return line does three separate things.

### 1. `token_ids != pad_id`

An elementwise comparison gives a bool tensor with the same shape, `(B, L)`:

```
token_ids = [[5, 7, 9, 0],     != 0     [[T, T, T, F],
             [3, 4, 0, 0]]     ---->     [T, T, F, F]]    # (2, 4)
```

True = real token, False = padding. That's why it's called "keep".

### 2. `[:, None, None, :]`

`None` inside an index is `np.newaxis`. It inserts a size-1 axis at that position without
moving or copying data.

```
[:,   None,  None,  :  ]
 B    new    new    L
```

`(B, L) → (B, 1, 1, L)`. Equivalent to `.reshape(B, 1, 1, L)` or
`.unsqueeze(1).unsqueeze(2)`.

### 3. Why those exact positions

Broadcasting lines shapes up from the right and stretches any axis of size 1 (see
[broadcasting.md](../pytorch/broadcasting.md)):

```
scores  (B, h,   L_q, L_k)
keep    (B, 1,   1,   L_k)
         ^  ^    ^    ^
         =  all  all  =
            heads queries
```

The mask says: "for every head and every query row, these key columns are dead." There's one bool
per (batch, key position), reused along the other axes.

### 4. `~keep` and `masked_fill`

`~` is elementwise NOT and flips keep → drop. `masked_fill(mask, v)` writes `v` wherever `mask`
is True and leaves the rest alone. Pad positions get `-inf`, and after softmax
`exp(-inf) = 0`, so those keys get exactly zero attention weight.

One head, one query row:

```
scores     [2.1,  0.4,  1.3,  0.9 ]
~keep      [ F ,   F ,   F ,   T  ]
after fill [2.1,  0.4,  1.3,  -inf]
softmax    [0.61, 0.11, 0.28, 0.00]
```

**Key intuition:** you're masking a *column* of the score matrix, not a row. A column is
"how much everyone attends to key `j`." Killing the column removes that token from the
pool for every reader. Rows are queries. The causal mask is the `(1, 1, L_q, L_k)`
triangular one, and it broadcasts over the batch instead.

______________________________________________________________________

## Q: What do `~keep`, `.masked_fill`, and `float("-inf")` each do?

### `~keep`: bitwise NOT

`~` is Python's bitwise-NOT operator. PyTorch overloads it on bool tensors to mean
elementwise logical NOT:

```python
keep  = torch.tensor([True, True, False])
~keep  # tensor([False, False,  True])
```

You need the flip because `keep` and `masked_fill` use opposite conventions: `keep` is True
where you *preserve*, and `masked_fill` writes where the mask is True.

**Trap:** `~` means logical NOT only on `dtype=torch.bool`. On an int tensor it does real
bitwise NOT: `~torch.tensor([1, 0])` gives `tensor([-2, -1])`, not `[0, 1]`. If a mask ever
arrives as `int64`/`uint8`, the result is silently wrong. Call `.bool()` if unsure. `not keep` doesn't
work either: Python's `not` needs a single truth value and raises on a multi-element
tensor.

### `.masked_fill(mask, value)`

Writes `value` at every position where `mask` is True and leaves everything else as is.

```python
scores = torch.tensor([2.1, 0.4, 1.3, 0.9])
mask   = torch.tensor([False, False, False, True])
scores.masked_fill(mask, float("-inf"))
# tensor([2.1000, 0.4000, 1.3000,   -inf])
```

- The mask only has to be **broadcastable** to the tensor's shape, not equal to it. That's what lets
  `(B, 1, 1, L_k)` fill a `(B, h, L_q, L_k)` tensor.
- It's **not in-place**. It returns a new tensor, hence `scores = scores.masked_fill(...)`.
  The in-place variant is `masked_fill_`; a trailing underscore always means in-place in PyTorch. Prefer
  the out-of-place one: in-place ops on tensors that autograd saved for the backward pass
  can raise.

### `float("-inf")`

Python has no `-inf` literal, so you parse it from a string. `float("inf")`, `float("-inf")`,
and `float("nan")` are the special values the constructor accepts. Equivalent:
`-math.inf`, `-torch.inf`.

Why not a large negative number like `-1e9`? It only works by accident. `exp(-1e9)`
underflows to exactly `0.0` in fp32, so you get the right answer. In fp16 the largest
magnitude is about 65504, and `-1e9` overflows to `-inf` anyway. In the additive style
`scores + (~keep) * -1e9` that's worse: kept positions compute `0 * -inf = NaN`.
`float("-inf")` through `masked_fill` is correct in every dtype.

### Putting it together

```python
scores = scores.masked_fill(~keep, float("-inf"))
#        ──┬───            ──┬──   ────┬────
#          │                 │         └─ the value: becomes 0 after exp()
#          │                 └─ where to write: the positions we DON'T keep
#          └─ rebind, since the op returns a new tensor
```

Read aloud: "wherever `keep` is False, overwrite the score with negative infinity."

### Equivalent spellings

Roughly from most to least idiomatic:

```python
scores.masked_fill(~keep, float("-inf"))        # clearest
scores.masked_fill(keep.logical_not(), -torch.inf)
torch.where(keep, scores, float("-inf"))        # "keep ? scores : -inf"
scores + (~keep) * -1e9                         # additive-float style; fragile in fp16
```

`torch.where` takes `keep` directly with no `~`, because it *selects* values rather than
*overwriting* them. Some people find it easier to read for that reason.

______________________________________________________________________

## Q: Does `float("-inf")` work on Mac MPS?

**Yes.** `-inf` is an ordinary fp32/fp16/bf16 value, and MPS handles it. The well-known dtype gap on MPS
is float64, which it doesn't support at all. That's unrelated to infinities.

MPS does have a history of bugs around masking. The points below come from PyTorch issues
as reported when these notes were written. Recheck them against your installed version.

1. **Causal SDPA can leak future tokens in half precision** (pytorch#195910, reported
   against 2.11). `scaled_dot_product_attention(..., is_causal=True)` on MPS in
   float16/bfloat16 let query `i` attend through key `(i//4)*4+3`. float32 was
   unaffected, and passing the same causal mask explicitly via `attn_mask` gave correct results.
   → On a Mac, don't trust `is_causal=True` in half precision. Build the bool mask
   yourself, as this repo already does.
1. **Fully masked rows behave differently on MPS than on CPU/CUDA** (pytorch#111416). CPU returns NaN and MPS returns
   0\. MPS's behaviour looks nicer, and that's the danger: an all-pad row that silently gives zeros on
   your Mac will produce NaN when the same code runs on CUDA.
1. **`masked_fill` has a size limit** (pytorch#143477). Tensors with ≥ 2³² elements crash
   the MPS backend, e.g. `(48, 25, 1024, 1024)`. That's unlikely at learning-scale sizes, but it
   crashes instead of raising an error.
1. **Softmax on large tensors has returned NaN** on MPS, in both fp16 and fp32 (pytorch#96602).

**Practical advice:** use `float("-inf")`, stay in fp32 while learning on MPS, and keep a
CPU parity check in your tests. Treat CPU as ground truth whenever a number looks strange.

```python
def assert_device_parity(fn, *args, atol=1e-4):
    """Run fn on MPS and CPU, assert outputs match.

    Args:
        fn: callable taking tensors, returning one tensor.
        *args: tensors to feed; moved to each device in turn.
        atol: absolute tolerance for the comparison.

    Raises:
        AssertionError: if outputs diverge or either contains NaN.
    """
    out = {d: fn(*(a.to(d) for a in args)) for d in ("cpu", "mps")}
    assert not any(o.isnan().any() for o in out.values()), "NaN in output"
    torch.testing.assert_close(out["mps"].cpu(), out["cpu"], atol=atol, rtol=0)
```

______________________________________________________________________

## How This Maps to Our Code

`attention` in [functions.py](../../app/src/training/transformer/functions.py):

```python
if mask is not None:
    attn_scores.masked_fill_(mask=~mask, value=-torch.inf)
```

- **Same convention as these notes:** `mask` True = visible, flipped with `~`, filled with
  `-inf`. Its docstring already names the two shapes: `(1, 1, L, L)` causal and
  `(B, 1, 1, L)` padding.
- **One difference:** it uses the in-place `masked_fill_`. That's safe here because
  `attn_scores` is a fresh matmul result that nothing else has saved. See "in-place" above
  for when it isn't safe.
- **Self-attention only, as currently typed:** the signature uses one `L` for both `q` and
  `k`, so the scores are `(B, h, L, L)`. Cross-attention needs `L_q ≠ L_k` (`L_tgt` vs
  `L_src`), so the jaxtyping annotation would need separate query and key lengths.

______________________________________________________________________

## References

- Vaswani et al., *"Attention Is All You Need"* (2017), §3.2.3. Masking only inside
  decoder self-attention; in encoder-decoder attention, queries come from the decoder,
  keys/values from the encoder, and every decoder position attends over all input
  positions. arXiv:1706.03762
- Bahdanau et al., *"Neural Machine Translation by Jointly Learning to Align and
  Translate"* (2015). Its alignment heatmaps show the crossing, non-monotonic patterns from
  the French example. arXiv:1409.0473
- Rombach et al., *"High-Resolution Image Synthesis with Latent Diffusion Models"* (2022).
  Text conditioning via cross-attention. arXiv:2112.10752
- PyTorch docs: `Tensor.masked_fill`, `torch.where`, broadcasting semantics
  (docs.pytorch.org/docs/stable/notes/broadcasting.html)
- NumPy indexing with `newaxis` (PyTorch reuses the `None` convention)
- MPS issues: pytorch#195910 (causal SDPA leak), pytorch#111416 (masked softmax CPU/MPS
  mismatch), pytorch#143477 (`masked_fill` 2³² limit), pytorch#96602 (softmax NaN)
