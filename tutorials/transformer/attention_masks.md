# Attention Masks and Cross-Attention — Q&A

**Revision question:** For one query, which keys may it read? First identify what
the score matrix's columns represent; then decide whether any columns are padding
or future information.

**Reading route:** Use the [mask table](#cheat-sheet) to reorient, follow the
[cross-attention example](#worked-example), and then
[trace the mask decision](#the-mistake-is-attaching-masks-to-blocks).
The [padding matrix](#the-matrix-picture) resolves the most common shape mistake.
The Python syntax and MPS sections are
reference material for when you encounter those details in code. Finish with the
[recall check](#recall-check).

Vocabulary, matching `app/src/training/transformer/constants.py`:

| Symbol | Meaning |
|---|---|
| `B` | batch |
| `L` | sequence length |
| `d_model` | embedding width |
| `h` | attention heads |
| `d_k` | width per head, `d_model // h` |
| `d_v` | value width per head; often the same as `d_k` |

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
| Decoder self-attn | target-side decoder inputs (positions after the current query are hidden) | padding + causal |
| Cross-attn (in decoder) | source tokens (encoder output, all available) | padding only |

This table assumes full-source encoder-decoder generation, as in ordinary
translation. A streaming model may have only part of the source available and
would mask source keys according to that availability.

**Mask shapes.** Each `1` means "the mask is the same along this axis":

```
(B, 1, 1,   L_k)   pad mask     → kills columns; same for every head and every query
(1, 1, L_q, L_k)   causal mask  → kills upper triangle; same for every batch and head
(B, 1, L_q, 1  )   row mask     → avoid fully masked rows; ordinary softmax is undefined there
```

Combine with `keep_pad & keep_causal` → broadcasts to `(B, 1, L_q, L_k)`.

**Trace one batch through a decoder layer.** Suppose there are two heads, three
decoder input positions, and four source positions (the fourth is `<pad>`):

| Operation | Score shape | Readable key columns |
|---|---|---|
| Decoder self-attention | `(1, 2, 3, 3)` | Query rows `0`, `1`, `2` may read input columns `0`; `0–1`; `0–2`, respectively. A target pad column, if present, is removed too. |
| Cross-attention | `(1, 2, 3, 4)` | Every target query may read source columns `0–2`; source column `3` is pad. |

The causal triangle belongs to the first matrix because its row and column axes
both index decoder positions. It does not belong to the rectangular cross-attention
matrix: those axes index different sequences. In teacher forcing, decoder inputs
are shifted right, so input positions through `t` contain target tokens from
*before* the next token being predicted at position `t`.

______________________________________________________________________

## Q: ELI5 cross-attention. I understand attention as "how can a token be best represented by other tokens, for a head."

Your definition already covers it. Change one phrase:

> how can a token be best represented by tokens **from a different sequence**

Same machinery, different pool of neighbours.

### Why Q is the one that stays home

The output has **one row per query**. Scores are `(B, h, L_q, L_k)`, and the output is
`(B, h, L_q, d_v)`; `d_v = d_k` in this repo's attention layer. The
softmax-weighted average sums the `L_k` axis away.

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

The query and key projections produce compatible vectors for the dot product;
their original inputs need not be the same kind of data:

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
| Decoder self-attn | shifted target inputs | Only input positions through the current query position are available → **causal mask** |
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
decoder inputs at positions ≤ t  →  [decoder self-attn, causally masked]  →  query vector q_t
```

With shifted decoder inputs, `q_t` depends only on target tokens before the one
being predicted at `t`. It then reads the whole source. The result depends on
(past targets, full source), exactly what is available at inference. Nothing leaks.

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

That example is the easy case, because the lengths are equal. Real pairs differ,
e.g. `L_src = 9`, `L_tgt = 14`, so the score matrix is
`(B, h, L_tgt, L_src)`. A rectangular triangle can be constructed, but it still
has **no causal meaning** here: its two axes index different sequences, not two
positions on one timeline.

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

**Practical:** row 3's output is irrelevant to the task when padded query
positions are excluded from the loss (`ignore_index=PAD`) and every later
attention layer continues masking padded keys. Real query positions then cannot
read that row through attention. Other operations that mix positions would need
their own padding treatment.

**Fatal:** if you mask row 3 entirely, the row becomes all `-inf`:

```
softmax([-inf, -inf, -inf, -inf]) = 0 / 0 = NaN
```

NaNs can spread through later computation and gradients. Masking an entire query
row is a common way to break ordinary softmax attention. The pad row is left to
compute meaningless but finite numbers on purpose. Some specialized attention
kernels define a safe result for fully masked rows; the simple `masked_fill` plus
softmax path shown here does not.

______________________________________________________________________

## Q: I can't get past the syntax of `padding_keep_mask`

The concepts above are enough for a theory pass. The next two questions unpack
the tensor syntax used to express those masks in PyTorch.

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
finite magnitude is about 65504, so `-1e9` cannot be represented as a finite value.
In the additive style
`scores + (~keep) * -1e9` that's worse: kept positions compute `0 * -inf = NaN`.
`float("-inf")` through `masked_fill` is the intended sentinel for excluded keys
in the ordinary softmax path, provided every query row retains at least one key.

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

**Yes.** `-inf` is representable in fp32/fp16/bf16, and MPS supports using it
in these tensors. MPS does not support float64 tensors; that is unrelated to
infinities.

MPS has had version-specific masking and softmax bugs. These are **issue reports
from particular PyTorch versions**, not claims about every current release.
Recheck the linked issue and your installed version when diagnosing a mismatch.

1. **Causal SDPA leaked future tokens in half precision** ([pytorch#195910](https://github.com/pytorch/pytorch/issues/195910), reported
   against 2.11). `scaled_dot_product_attention(..., is_causal=True)` on the reported MPS setup in
   float16/bfloat16 let query `i` attend through key `(i//4)*4+3`. float32 was
   unaffected, and passing the same causal mask explicitly via `attn_mask` gave correct results.
   An explicit Boolean causal mask avoids that reported path, as this repo does.
1. **A fully masked `masked.log_softmax` differed across devices** ([pytorch#111416](https://github.com/pytorch/pytorch/issues/111416), reported in 2023).
   The reported CPU result was NaN and the MPS result was `0` for a masked scalar.
   This illustrates why a fully masked row should be treated as an error in the
   ordinary attention formulation, even if one backend yields a finite value.
1. **A very large `masked_fill` allocation failed** ([pytorch#143477](https://github.com/pytorch/pytorch/issues/143477), reported against a 2.6 development build).
   The limit in the reported failure was **2³² bytes**, not elements; the example
   score tensor had shape `(48, 25, 1024, 1024)`. This is far beyond these
   tutorials' tensor sizes.
1. **Softmax on a large tensor returned NaN** in both fp16 and fp32 on a reported MPS setup
   ([pytorch#96602](https://github.com/pytorch/pytorch/issues/96602), reported against a 2.1 development build).

**Practical check:** use `float("-inf")` for excluded keys, avoid fully masked
rows, and compare a small example on CPU if MPS results look strange. CPU is a
useful independent reference, not an infallible ground truth.

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

## Common confusions

| Tempting shortcut | Better question |
|---|---|
| “The decoder needs a causal mask everywhere.” | Which sequence supplies this attention matrix's **keys**? The whole source is available to cross-attention. |
| “Padding means the pad query should read nothing.” | Which **key columns** are meaningless? Remove those from every query's reading pool; do not make a softmax row all `-inf`. |
| “A triangle works whenever there are target rows.” | Do rows and columns index the **same ordered sequence**? If they index target and source, a triangle would impose a false alignment. |
| “True means masked.” | Check the convention at the API boundary. Here `keep=True` means readable; `masked_fill(~keep, -inf)` flips it. |

## Recall check

1. In decoder cross-attention with `L_tgt = 3` and `L_src = 4`, what is the score shape, and which axis holds source padding?
1. Why may target query position `1` read source column `2` in translation, even though it may not read a future target position?
1. A score row is `[-inf, -inf, -inf]`. What goes wrong with ordinary softmax, and how should a padded query be handled in this example?

<details markdown="1">
<summary>Answers</summary>

1. `(B, h, 3, 4)`. Source padding removes columns on the final `L_src` axis, with a keep mask shaped `(B, 1, 1, 4)`.
1. The entire source exists before decoding; source index `2` is not a future target. The causal constraint applies to target-side decoder inputs.
1. The denominator is zero, so ordinary softmax is undefined and often returns NaN. Keep at least one readable key for that row, ignore the padded query in the loss, and keep masking its key column in later attention layers.

</details>

**One-minute recap:** The score matrix has one row per query and one column per
key. A padding mask removes meaningless **columns** wherever padded keys appear.
A causal mask removes future **target columns** in decoder self-attention.
Cross-attention reads the fully available source, so it uses source padding but
no target causal triangle.

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
- MPS issues: [pytorch#195910](https://github.com/pytorch/pytorch/issues/195910)
  (causal SDPA leak), [pytorch#111416](https://github.com/pytorch/pytorch/issues/111416)
  (masked log-softmax CPU/MPS mismatch), [pytorch#143477](https://github.com/pytorch/pytorch/issues/143477)
  (`masked_fill` 2³²-byte limit in the report), [pytorch#96602](https://github.com/pytorch/pytorch/issues/96602)
  (softmax NaN in a reported setup)
