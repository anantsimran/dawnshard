# Transformers

Which tokens may attend to which, where normalization goes and what that does to
the gradient path, and what a run costs in FLOPs and bytes. For how these ideas
appear in this repo's source, see the
[attention implementation notes](../app/src/training/transformer/README.md).

**On this page:**

- [Attention Masks and Cross-Attention — Q&A](#attention-masks-and-cross-attention--qa)
- [Pre-Norm vs. Post-Norm](#pre-norm-vs-post-norm)
- [Transformer Compute and Memory](#transformer-compute-and-memory)

______________________________________________________________________

## Attention Masks and Cross-Attention — Q&A

**Revision question:** For one query, which keys may it read? First identify what
the score matrix's columns represent; then decide whether any columns are padding
or future information.

**Reading route:** Use the [mask table](#cheat-sheet-attention-masks) to reorient, follow the
[cross-attention example](#worked-example), and then
[trace the mask decision](#the-mistake-is-attaching-masks-to-blocks).
The [padding matrix](#the-matrix-picture) resolves the most common shape mistake.
For the decoder as a whole, read
[what flows through a decoder block](#q-what-flows-through-a-decoder-block-and-what-does-each-part-represent),
then [the drawn causal mask](#q-draw-a-causal-mask-does-the-position-predicting-t1-see-only-bos).
The Python syntax and MPS sections are
reference material for when you encounter those details in code. Finish with the
[recall check](#recall-check-attention-masks).

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

### Cheat-sheet: Attention Masks

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

### Q: ELI5 cross-attention. I understand attention as "how can a token be best represented by other tokens, for a head."

Your definition already covers it. Change one phrase:

> how can a token be best represented by tokens **from a different sequence**

Same machinery, different pool of neighbours.

#### Why Q is the one that stays home

The output has **one row per query**. Scores are `(B, h, L_q, L_k)`, and the output is
`(B, h, L_q, d_k)`. The softmax-weighted average sums the `L_k` axis away.

So whoever supplies **Q is the thing being represented**. Whoever supplies **K and V is
the material** it's built from. In translation you're building a better vector for the
French word you're about to write, so the French (decoder) position supplies Q. It's
shopping, and the encoder output is the shelf.

#### Kid version

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

#### Worked example

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

#### Two attentions, two jobs

Every decoder layer runs both, in this order:

| | Question it answers | Mask |
|---|---|---|
| **Self-attention** | "What have I written so far?" Keeps the output grammatical and coherent | causal, because your own future doesn't exist yet |
| **Cross-attention** | "What am I supposed to be saying?" Keeps the output faithful to the source | none (besides padding), because the whole source is available |

Self-attention is re-reading your own draft. Cross-attention is glancing back at the
original. Without the first you write nonsense with bad grammar. Without the second you write
beautiful French that translates a different sentence.

#### Nothing here is specific to language

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

### Q: What flows through a decoder block, and what does each part represent?

A decoder block is three sublayers in a fixed order: masked self-attention, then
cross-attention, then a feed-forward network. Each is wrapped in a residual connection
and a LayerNorm (where the norm goes is the subject of
[Pre-Norm vs. Post-Norm](#pre-norm-vs-post-norm)). You stack `N` blocks: the paper used
6, and 20 works the same way. Each block reads the previous block's output. The encoder
runs **once**, and its final output, usually called `memory`, feeds the
cross-attention of **every** decoder block. The decoder never reads the encoder's
intermediate layers.

```
tgt_in  (B, L_tgt)                           src  (B, L_src)
   │                                         │
   ▼  embedding + positional encoding        ▼  encoder, runs once
x  (B, L_tgt, d_model)                       memory  (B, L_src, d_model)
   │                                         │
┌──▼─ decoder block 1 ──────────────────┐    │
│  x = x + masked_self_attn(x)          │    │
│  x = x + cross_attn(x, memory)  ◄─────┼────┤
│  x = x + ffn(x)                       │    │
└──┬────────────────────────────────────┘    │
   │  (B, L_tgt, d_model)                    │
   ▼                                         │
blocks 2 … N, same three steps  ◄────────────┘  the same memory in every block
   │
   ▼  Linear to vocab
logits  (B, L_tgt, vocab)  ──►  loss against tgt_out  (B, L_tgt)
```

The LayerNorms are left out of the picture. One running example carries through the rest
of this answer:

```
src     = [the, cat,  sleeps]
tgt_in  = [BOS, le,   chat,   dort]
tgt_out = [le,  chat, dort,   EOS]
```

Follow the position holding `le`. Its job is to output `chat`.

#### `tgt_out` is only the answer key

`tgt_out` never enters the network. It waits at the end and grades each position:
"position `le`, you should have said `chat`." At inference there is no `tgt_out` at all.
The model feeds its own predictions back in as `tgt_in`, one token at a time.

#### Block 1's input: embedding plus position

The first block reads `nn.Embedding(tgt_in)` plus a positional encoding, followed by
dropout. The paper multiplies the embedding by `√d_model` before adding the position
(§3.4). This repo's `SinusoidalEmbedding` in
[modules.py](../app/src/training/transformer/modules.py) explains why it skips that
scaling. Every later block reads the previous block's output.

#### What a position's vector holds: a note being edited

Think of each position as holding a note: "what I understand so far, and what I think
comes next."

- **Block 1:** the note is thin. "I'm the word `le`, in slot 1."
- **Deeper blocks:** the note gets richer. "I'm a masculine article. A noun is coming. The
  source is talking about a cat."

The residual connection is what makes it a note and not a fresh page. Each sublayer
computes `x + sublayer(x)`, adding edits instead of rewriting, so block `k`'s input is
the embedded word plus everything blocks `1 … k−1` wrote on it.

#### Self-attention: rereading your own draft

Q, K, and V all come from the decoder's own positions, and the causal mask limits each
row to itself and the positions to its left.

- **Q:** `le` asks, "which earlier words of my own sentence matter for what I say next?"
- **K:** each earlier target position advertises its role: "I'm the start," "I'm an
  article."
- **V:** what that position hands over if chosen, such as gender or number.

Row `le` reads `BOS` and itself, and can conclude: "I'm right after the start, I'm
masculine singular, so a masculine noun probably follows." It knows nothing about cats
yet.

#### Cross-attention: consulting the source

Q comes from the decoder, already enriched by self-attention. K and V come from
`memory`. There is no causal mask, so every source token is readable
([why](#q-explain-the-masks-used-in-the-encoder-and-decoder-why-are-they-set-up-this-way)).

- **Q:** `le` asks, "what in the source tells me which noun comes next?"
- **K:** each source token advertises itself: "I'm `cat`, a subject noun," "I'm
  `sleeps`, a verb."
- **V:** that source token's meaning, handed over.

The query matches `cat`'s key strongly, so `le` pulls in the meaning of `cat`. This is
the alignment step, where the model decides which source word it is translating right
now. The [note-card version](#kid-version) walks through the same lookup.

#### The block's output: the same note, better informed

The output has the same shape as the input, `(B, L_tgt, d_model)`, with the note
updated. After the attention sublayers gather information, the FFN works on each
position separately and digests it, turning "masculine article + cat concept + French
context" into something close to "`chat`."

- **An intermediate block's output** is a better-informed note, handed to the next
  block.
- **The last block's output** goes through the final Linear to vocab logits. Position
  `le` scores `chat` highest, and the loss compares it with `tgt_out[1]`.

The structure here (the shift, the masks, one `memory` shared by every block) comes from
the paper. What a particular head learns, such as gender agreement, is illustrative and
not guaranteed.

______________________________________________________________________

### Q: Why must cross-attention's values come from the source?

Because cross-attention has exactly one job: bring source information into the decoder.
If V came from the decoder, cross-attention would be self-attention again, and the
source sentence would never get in.

You're writing a French translation of an English book. Self-attention is rereading your
own draft; the content comes from you. Cross-attention is looking things up in the
English book. The question comes from you ("I just wrote `le`, which noun comes next?"),
but the answer has to come from the book. Looking up your own draft twice tells you
nothing new.

#### The residual keeps the decoder's own information

The block computes `x_new = x + cross_attn(x, memory)`, not `x_new = cross_attn(x, memory)`.

- `x` already holds "I'm `le`, a masculine article, in French."
- Cross-attention adds roughly `0.8·v_cat`: "the concept being translated is cat."
- The sum holds both: "masculine article + cat concept."

Decoder information and source information both end up in the vector. Cross-attention
only has to supply the missing half.

#### The value doesn't need to be in the target language

`v_cat` isn't a word. It is a meaning vector in the shared `d_model` space, where
"cat-ness" belongs to neither language. `W_v` and `W_o` shape what gets handed over, the
FFN and later blocks turn "masculine article + cat concept" into something `chat`-like,
and the final Linear is the only place the target vocabulary appears. The step into
target words happens at the end, not inside attention.

#### Keys and values must come from the same tokens

A row of cross-attention weights has one entry per source position: three numbers for
`[the, cat, sleeps]`. Those three weights multiply three rows of V, so V needs one row
per source token. A key is the label on a box, and the value is what's inside. A key from
the source paired with a value from the target would be a label on the wrong box, and
with `L_src ≠ L_tgt` the shapes would not even match.

______________________________________________________________________

### Q: Explain the masks used in the encoder and decoder. Why are they set up this way?

#### The mistake is attaching masks to blocks

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

#### "But doesn't looking at the whole source leak the answer?"

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

#### Why a causal mask in cross-attention is wrong, not just useless

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

#### So what is the source padding mask doing?

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

### Q: Draw a causal mask. Does the position predicting `t1` see only `BOS`?

Yes. For decoder inputs `[BOS, t1, t2, t3]`, rows are query positions, columns are key
positions, and `1` means "may attend":

```
         BOS   t1   t2   t3
BOS    [  1    0    0    0  ]   → predicts t1
t1     [  1    1    0    0  ]   → predicts t2
t2     [  1    1    1    0  ]   → predicts t3
t3     [  1    1    1    1  ]   → predicts t4
```

It is lower-triangular, `torch.tril(torch.ones(L, L, dtype=torch.bool))`: row `i` reads
columns `≤ i`. The row that predicts `t1` is `BOS`, and it sees only `BOS`.

#### What the zeros do to the scores

Scores are `QKᵀ / √d_k`. Every `0` in the mask becomes `-inf` in the scores before the
softmax, and `e^-inf = 0`, so masked columns get zero weight. Take row `t1` with raw
scores `[2, 1, 5, 9]`:

```
raw      [ 2      1      5      9    ]
masked   [ 2      1     -inf   -inf  ]
exp      [ 7.39   2.72   0      0    ]   sum 10.11
softmax  [ 0.73   0.27   0      0    ]
```

The large future scores, 5 and 9, contribute nothing. Row `BOS` has a single readable
column, so its weight is 1.0 and its attention output is exactly `BOS`'s own value
vector. For the PyTorch syntax, see
[`~keep`, `masked_fill`, and `-inf`](#q-what-do-keep-masked_fill-and-float-inf-each-do).

#### Why train with the triangle

Training feeds `[BOS, t1, t2, t3]` and grades against `[t1, t2, t3, t4]`, the same
sequence shifted by one. All four predictions come out of one parallel forward pass.
Without the mask, row `BOS` could read `t1` and copy the answer: the loss would drop
toward zero and nothing useful would be learned. At inference the future tokens don't
exist yet, so the mask makes training see exactly what inference will.

______________________________________________________________________

### Q: Row `le` can see itself. Can it copy the answer?

It can copy itself all it wants, and it gains nothing. Row `le` isn't trying to output
`le`. Its target is `chat`. Line up the decoder input and the target:

```
column:    0     1     2     3
tgt_in:   BOS   le    chat  dort
tgt_out:  le    chat  dort  EOS
```

| Row | Sees | Must predict | Is the answer visible? |
|---|---|---|---|
| `BOS` | `BOS` | `le` | No, `le` is column 1, masked |
| `le` | `BOS`, `le` | `chat` | No, `chat` is column 2, masked |
| `chat` | `BOS`, `le`, `chat` | `dort` | No, `dort` is column 3, masked |
| `dort` | all four | `EOS` | No, `EOS` is not in `tgt_in` at all |

The answer for row `i` is always `tgt_in[i+1]`, which is exactly the first hidden cell of
that row. The mask hides it right where it would be useful. Seeing the current word is
fine because it is the input, not the answer: to say what comes after `le`, you have to
know you're at `le`.

Two things work together:

- **The shift by one** puts each answer one column to the right of its row.
- **The triangle, diagonal included,** hides everything to the right.

Break either one and the model can cheat:

- **A mask that lets row `i` see `i+1`:** it copies the next word, and the loss goes to
  roughly zero without anything being learned.
- **No shift (target = the token itself) with the diagonal visible:** it copies itself.

______________________________________________________________________

### Q: Does padding mean "this position advertises nothing about itself" or "this position isn't looking for anything"?

**The first.** Padding mask = "this position advertises nothing." Nobody may read from
it. The pad's *query row* is deliberately left alone.

#### The matrix picture

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

#### "So shouldn't the last row be zeroed?" No, for two reasons

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

### Q: I can't get past the syntax of `padding_keep_mask`

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

#### 1. `token_ids != pad_id`

An elementwise comparison gives a bool tensor with the same shape, `(B, L)`:

```
token_ids = [[5, 7, 9, 0],     != 0     [[T, T, T, F],
             [3, 4, 0, 0]]     ---->     [T, T, F, F]]    # (2, 4)
```

True = real token, False = padding. That's why it's called "keep".

#### 2. `[:, None, None, :]`

`None` inside an index is `np.newaxis`. It inserts a size-1 axis at that position without
moving or copying data.

```
[:,   None,  None,  :  ]
 B    new    new    L
```

`(B, L) → (B, 1, 1, L)`. Equivalent to `.reshape(B, 1, 1, L)` or
`.unsqueeze(1).unsqueeze(2)`.

#### 3. Why those exact positions

Broadcasting lines shapes up from the right and stretches any axis of size 1 (see
[Shapes and Broadcasting](pytorch.md#shapes-and-broadcasting--qa)):

```
scores  (B, h,   L_q, L_k)
keep    (B, 1,   1,   L_k)
         ^  ^    ^    ^
         =  all  all  =
            heads queries
```

The mask says: "for every head and every query row, these key columns are dead." There's one bool
per (batch, key position), reused along the other axes.

#### 4. `~keep` and `masked_fill`

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

### Q: What do `~keep`, `.masked_fill`, and `float("-inf")` each do?

#### `~keep`: bitwise NOT

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

#### `.masked_fill(mask, value)`

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

#### `float("-inf")`

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

#### Putting it together

```python
scores = scores.masked_fill(~keep, float("-inf"))
#        ──┬───            ──┬──   ────┬────
#          │                 │         └─ the value: becomes 0 after exp()
#          │                 └─ where to write: the positions we DON'T keep
#          └─ rebind, since the op returns a new tensor
```

Read aloud: "wherever `keep` is False, overwrite the score with negative infinity."

#### Equivalent spellings

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

### Q: Does `float("-inf")` work on Mac MPS?

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

### How This Maps to Our Code

`attention` in [functions.py](../app/src/training/transformer/functions.py):

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

### Common confusions: Attention Masks

| Tempting shortcut | Better question |
|---|---|
| “The decoder needs a causal mask everywhere.” | Which sequence supplies this attention matrix's **keys**? The whole source is available to cross-attention. |
| “Padding means the pad query should read nothing.” | Which **key columns** are meaningless? Remove those from every query's reading pool; do not make a softmax row all `-inf`. |
| “A triangle works whenever there are target rows.” | Do rows and columns index the **same ordered sequence**? If they index target and source, a triangle would impose a false alignment. |
| “True means masked.” | Check the convention at the API boundary. Here `keep=True` means readable; `masked_fill(~keep, -inf)` flips it. |
| “The diagonal is visible, so a position can copy its answer.” | What is this row's **target**? With shifted targets it is the next token, which sits in the first column the triangle hides. |

### Recall check: Attention Masks

1. In decoder cross-attention with `L_tgt = 3` and `L_src = 4`, what is the score shape, and which axis holds source padding?
1. Why may target query position `1` read source column `2` in translation, even though it may not read a future target position?
1. A score row is `[-inf, -inf, -inf]`. What goes wrong with ordinary softmax, and how should a padded query be handled in this example?
1. With `tgt_in = [BOS, le, chat, dort]`, which columns may row `chat` read in decoder self-attention, and what must it predict?

<details markdown="1">
<summary>Answers</summary>

1. `(B, h, 3, 4)`. Source padding removes columns on the final `L_src` axis, with a keep mask shaped `(B, 1, 1, 4)`.
1. The entire source exists before decoding; source index `2` is not a future target. The causal constraint applies to target-side decoder inputs.
1. The denominator is zero, so ordinary softmax is undefined and often returns NaN. Keep at least one readable key for that row, ignore the padded query in the loss, and keep masking its key column in later attention layers.
1. Columns 0–2 (`BOS`, `le`, `chat`). It must predict `dort`, which sits in column 3, the first column the triangle hides.

</details>

**One-minute recap:** The score matrix has one row per query and one column per
key. A padding mask removes meaningless **columns** wherever padded keys appear.
A causal mask removes future **target columns** in decoder self-attention.
Cross-attention reads the fully available source, so it uses source padding but
no target causal triangle.

______________________________________________________________________

### References: Attention Masks

- Vaswani et al., *"Attention Is All You Need"* (2017), §3.2.3. Masking only inside
  decoder self-attention; in encoder-decoder attention, queries come from the decoder,
  keys/values from the encoder, and every decoder position attends over all input
  positions. arXiv:1706.03762
- Vaswani et al. (2017), §3.1 and §3.4. The decoder stack with a residual around each
  sublayer; output embeddings offset by one position, which together with the mask
  keeps position `i` from depending on later outputs; embeddings scaled by `√d_model`.
- Harvard NLP, *The Annotated Transformer*: nlp.seas.harvard.edu/annotated-transformer
  (the encoder's `memory` passed unchanged to every decoder layer)
- Karpathy, nanoGPT `model.py`: github.com/karpathy/nanoGPT (a `torch.tril` buffer and
  `masked_fill(..., -inf)` for the causal mask)
- Alammar, *The Illustrated Transformer*: jalammar.github.io/illustrated-transformer
  (decoder data-flow visuals)
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

______________________________________________________________________

## Pre-Norm vs. Post-Norm

**Revision question:** When a transformer block adds a new result to its residual
stream, does it normalize the running stream itself or only the copy read by the
next sublayer? That one placement choice changes the path gradients can take.

**Reading route:** Start with [one token's normalization](#what-layer-normalization-does),
then [trace two sublayers](#pre-norm-vs-post-norm-where-layernorm-goes), and finish
with the gradient-path question. The depth and FFN questions, the
[edge-set diagrams](#the-edge-set-for-two-sublayers), and the interpretability
discussion add detail when you need it. Use the [recall check](#recall-check-pre-norm-vs-post-norm)
after a break.

### What Layer Normalization Does

Think of one token's features as a row. LayerNorm compares each feature with the
other features **in that row**. For `x = [1, 2, 3]`, the mean is `2`, the variance
is `2/3`, and the centered, rescaled row is approximately
`[-1.225, 0, 1.225]` (ignoring `ε`). The row `[10, 20, 30]` gives the same
normalized values: its overall scale is gone, while its feature pattern remains.

Given an input vector `x ∈ ℝ^d` — one token's activations — the full operation is:

```text
LN(x) = γ ⊙ [(x − μ) / √(σ² + ε)] + β
```

where `μ = (1/d) Σ_i x_i` is the mean, `σ² = (1/d) Σ_i (x_i − μ)²` is the variance,
`γ` and `β` are learned per-dimension scale and shift parameters, `ε` is a small
constant for numerical stability, and `⊙` is element-wise multiplication.

Two stages, not one:

1. **Center and rescale.** Subtract the mean, divide by the standard deviation. The
   result has approximately zero mean and unit variance (exactly so when `ε = 0`
   and variance is nonzero) — no learned parameters involved yet.
1. **Learned affine transform.** `γ` and `β` shift and rescale the normalized
   vector per-dimension, letting the model undo the normalization in whichever
   directions it isn't helpful.

Layer normalization computes `μ` and `σ` **within one token's vector**,
independently of every other position and every other example in the batch —
the key axis difference from BatchNorm, which uses statistics across examples
for each feature instead. One consequence: apart from the small `ε` term,
multiplying the entire
input vector by a positive scalar leaves `LN(x)` unchanged before the affine
transform. A sublayer reading only the normalized vector has no way to recover
that overall scale — it's thrown away by construction.

**Vocabulary**, for the token × feature picture below:

- **Row** = one token. Its whole "meaning vector" at this point in the network.
- **Column** = one feature (channel / dimension) — `d_model` of them, e.g. 4096.
- **Feature** = a single column. "Feature 2 of token `cat`" is one cell.

```
                feat_1   feat_2   feat_3   feat_4   ...
token "the"   [  0.8     -1.2      3.5      0.1     ... ]
token "black" [ -2.1      0.3      0.0      4.7     ... ]
token "cat"   [  1.0      4.0     -0.6      0.2     ... ]
```

The row for `"cat"` is that token's whole vector, `[1.0, 4.0, -0.6, 0.2, ...]`. The
`feat_2` column read down every row is the same feature, compared across tokens.
`LN` normalizes **along a row** — across `feat_1 … feat_{d_model}` for one token —
never down a column.

#### Which axes: `normalized_shape`

```python
ln = nn.LayerNorm(normalized_shape=d_model)
out = ln(x)  # x: (B, L, d_model) → out: (B, L, d_model)
```

`normalized_shape` names the **trailing** dimensions that supply `μ` and `σ²`, and
it is also the shape of `γ` and `β`. Passing `d_model` gives `B × L` independent
normalizations, one per row of the picture above. Pooling over more axes would
break that:

- **Adding `L`** (`normalized_shape=(L, d_model)`) mixes tokens. One token's output
  would depend on every other token in the sequence, padding included, and on the
  sequence length. In a causal decoder, future tokens would leak into earlier
  positions through `μ` and `σ`.
- **Pooling across the batch** is BatchNorm's choice. One example's output would
  depend on which other examples share its batch, and training (batch statistics)
  would behave differently from inference (running averages).

#### Why it's called layer norm

If it normalizes per token, why the name? The name says which axis the statistics
pool over, in contrast with BatchNorm:

- **BatchNorm** (Ioffe & Szegedy, 2015): fix one hidden unit, pool over the
  **batch**. Each feature gets a mean and variance computed across examples.
- **LayerNorm** (Ba, Kiros & Hinton, 2016): fix one example, pool over every hidden
  unit in the **layer**. The statistics come from that layer's activations.

The paper targeted MLPs and RNNs, where a layer's output for one input is a single
vector, so "normalize over the layer" meant "normalize that vector." A transformer
adds a sequence axis: every position gets its own `d_model` vector from the layer,
and LayerNorm treats each position the way the paper treated one input. "Per token"
is what "per layer, per example" becomes once there is a sequence axis.

The name can mislead in the other direction too. In the paper's convolutional
setting, "the layer" meant all of `(C, H, W)`, so it pooled over spatial positions
as well as channels. Read `normalized_shape`, not the name.

______________________________________________________________________

### What Normalization Really Removes

After `x̂ = (x − μ) / σ`, computed across the `d_model` features of a single token,
that token's vector has mean 0 and variance 1 — so its length is pinned:
`‖x̂‖ = √d_model` when `ε` is ignored and variance is nonzero. The vectors lie
on a sphere within the mean-zero subspace. Direction is free; magnitude is not.

So it isn't quite right to say "feature 1 can't be large." A feature far above
the token's mean remains far above the mean after normalizing: subtracting one
mean and dividing by one standard deviation preserves the ordering and ratios of
*pairwise differences*. It does not preserve ratios of raw feature values, because
centering changes those. What's destroyed is the **token's original mean and
overall scale**, not its pattern of relative differences. That matters because
whatever reads `x̂` next is a matmul feeding an activation or a softmax, and both
of those care about absolute magnitude, not just direction.

`γ` and `β` let the model choose a shared scale and baseline for each feature.
They reshape and shift the normalized vectors, but they **cannot recover each
token's original mean or scale**: those were discarded before the affine step.
The resulting feature scales are learned by gradient descent rather than inherited
from the preceding layer.

______________________________________________________________________

### The Mental Model for `γ` and `β`

`γ_i` answers "how much should feature dimension `i` be amplified?" — that's a
property of the **feature**, not the token. `β_i` is the paired answer for "what
should the baseline value of feature `i` be?" Both are indexed by feature only:
shape `[d_model]`, no batch or sequence axis. The same `γ_i`, `β_i` apply to every
token, in every sequence, in every batch.

**Yes, they're learned.** `γ` and `β` are `nn.Parameter`, updated by the optimizer
exactly like any weight matrix. Writing `y = γ ⊙ x̂ + β` and `g = ∂L/∂y`, the
gradients are:

```
∂L/∂β_i = Σ_{b,l}  g[b, l, i]
∂L/∂γ_i = Σ_{b,l}  g[b, l, i] · x̂[b, l, i]
```

That sum over every `(b, l)` is the signature of a shared parameter: `γ_i` was
used at all `B × L` token positions, so it accumulates gradient contributions from
every one of them. This is exactly why the shapes work out — the upstream
gradient is `[B, L, d]`, and you reduce away the axes the parameter was broadcast
over (`B` and `L`) to land back at `[d]`, matching `γ`'s and `β`'s own shape.

Separately, gradient also flows backward through `x̂` into `x` — and *that* path is
not elementwise, because `μ` and `σ` are themselves functions of every element of
the token. Changing `x_j` changes `μ` and `σ`, which changes `x̂_i` for every `i`,
not just `i = j`. That's why `LayerNorm`'s backward pass into `x` has cross terms
between features, even though the forward pass and the `γ`/`β` gradients above are
purely elementwise or reduction-only.

______________________________________________________________________

### Pre-Norm vs. Post-Norm: Where LayerNorm Goes

Every transformer block wraps two sublayers — self-attention and a feedforward
network — in a residual connection plus a `LayerNorm`. Where exactly the `LayerNorm`
goes is not a cosmetic choice. It changes how gradients flow during training and
what the residual stream even *means* for interpretability.

`r_l` is the residual stream entering block `l`. `Sublayer` stands for either
self-attention or the FFN — the same analysis applies to both.

**Post-norm** (the original Transformer):

```text
r_{l+1} = LN(r_l + Sublayer(r_l))
```

**Pre-norm** (GPT-2 and most modern architectures):

```text
r_{l+1} = r_l + Sublayer(LN(r_l))
```

**Trace two sublayers, attention `A` then feedforward `F`:**

| Step | Pre-norm: normalize the branch being read | Post-norm: normalize the new stream |
|---|---|---|
| Attention | `a = r₀ + A(LN₁(r₀))` | `a = LN₁(r₀ + A(r₀))` |
| Feedforward | `r₂ = a + F(LN₂(a))` | `r₂ = LN₂(a + F(a))` |
| Skip path through both | `r₀ → + → + → r₂` | `r₀ → + → LN₁ → + → LN₂ → r₂` |

Both variants have residual additions. Only pre-norm has a path through the stack
that bypasses every internal `LayerNorm`. A pre-norm stack usually applies a final
normalization before its output head, so that final norm still sees the result.

______________________________________________________________________

### Cheat-sheet: Pre-Norm vs. Post-Norm

| | Post-Norm | Pre-Norm |
|---|---|---|
| Recurrence | `r_{l+1} = LN(r_l + Sublayer(r_l))` | `r_{l+1} = r_l + Sublayer(LN(r_l))` |
| Gradient path to `r_0` | Every layer's backward step is multiplied by `LN`'s Jacobian | Every layer contributes an identity term — a direct path back to the input |
| LR warmup | Often important for stable training near initialization | Less tied to norm placement, though still commonly used |
| Residual stream magnitude | Renormalized every sublayer | Can grow with depth; a final `LN` is common before the output head |
| Is the stream a sum of per-layer contributions? | Each addition is immediately renormalized | Yes, before the final `LN` — this is `hook_resid_pre` in TransformerLens |
| Used by | Vaswani et al. 2017 (original Transformer) | GPT-2 and most descendants |

______________________________________________________________________

### Q: What does "depth" mean here?

The next three questions supply vocabulary about depth and feedforward layers.
You can skip to the graph and gradient sections if those terms are already
familiar.

**Depth** is the number of stacked residual blocks (layers) in the model — the `L`
in "a 12-layer / 96-layer transformer." It has nothing to do with model width
(`d_model`) or sequence length.

> In this tutorial `L` counts layers. In the code, `L` in jaxtyping shapes is
> sequence length (`app/src/training/transformer/constants.py`). Same letter, unrelated
> quantity.

Each transformer layer has two sublayers, attention and the FFN, and each sublayer
gets its own residual connection and its own `LN`. So an `L`-layer model has `2L`
places where norm placement matters. The recurrences in this tutorial step once per
sublayer, which means a real `L`-layer model applies them `2L` times. The factor of
2 changes no argument below, so the math just writes `L`.

______________________________________________________________________

### Q: How wide is the FFN sublayer?

Width is a separate axis from depth. In *"Attention Is All You Need"*, the model
dimension was `d_model = 512` and the FFN's inner dimension was `d_ff = 2048`:

```
FFN(x) = W_2 · ReLU(W_1 · x + b_1) + b_2
W_1: d_model → d_ff      (512 → 2048)
W_2: d_ff → d_model      (2048 → 512)
```

This fourfold expansion, `d_ff = 4 × d_model`, is a common rule of thumb, though
variations exist. Expanding lets the FFN project each token into a
higher-dimensional space, where complex patterns may be easier to learn, before
projecting back down to `d_model`. The projection back is what lets the output be
added to the residual stream.

______________________________________________________________________

### Q: ReLU or GELU inside the FFN?

The nonlinearity between `W_1` and `W_2` is the only thing that stops the FFN from
collapsing into a single linear map. The original Transformer used ReLU. GPT-1,
GPT-2, and BERT switched to GELU.

**ReLU** gates each unit by the sign of its input:

```
ReLU(x) = max(0, x)
ReLU'(x) = 1 if x > 0, else 0
```

It's cheap and gives exact zeros, but negative inputs get exactly zero local
gradient. If a row of `W_1` has a negative pre-activation for every token, that
unit outputs 0 and receives no gradient through those inputs. It can stay inactive
unless its inputs change enough through updates elsewhere. This is a **dead
neuron**.

**GELU** (Gaussian Error Linear Unit) replaces the hard gate with a soft one:

```
GELU(x) = x · Φ(x)        Φ = standard normal CDF
```

ReLU multiplies `x` by 0 or 1. GELU multiplies `x` by `Φ(x)`, the probability that a
standard normal sample is below `x`. That factor slides smoothly from 0 to 1. For
large `|x|` the two functions agree. The difference is near zero:

| `x` | −2 | −1 | −0.5 | 0 | 0.5 | 1 | 2 |
|---|---|---|---|---|---|---|---|
| `ReLU(x)` | 0 | 0 | 0 | 0 | 0.5 | 1 | 2 |
| `GELU(x)` | −0.046 | −0.159 | −0.154 | 0 | 0.346 | 0.841 | 1.954 |
| `GELU'(x)` | −0.085 | −0.083 | 0.133 | 0.5 | 0.867 | 1.083 | 1.085 |

(Values computed with `torch.nn.functional.gelu`.)

Three consequences:

- **Smooth.** There's no kink at 0. The derivative there is 0.5.
- **Negative inputs still get gradient.** At `x = −2` the gradient is small but not
  zero, so a unit pushed negative can still be pulled back. It doesn't die the way
  a ReLU unit does.
- **Not monotonic.** GELU dips to a minimum of about −0.17 at `x ≈ −0.75` before
  rising back toward 0. Slightly negative inputs give slightly negative outputs, and
  very negative inputs give roughly zero.

The adoption of GELU in GPT and BERT is an empirical design choice, not a theorem
that it always trains better than ReLU. Smooth gradients and fewer completely
inactive units are common motivations.

In PyTorch:

```python
h = F.relu(input=W_1(x))
h = F.gelu(input=W_1(x))  # exact: x * Φ(x)
h = F.gelu(input=W_1(x), approximate="tanh")  # tanh approximation
```

The `tanh` approximation, `0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))`, is
within 5e-4 of exact GELU on `[−3, 1]`. The older `x · sigmoid(1.702 · x)` shortcut
is looser, at about 0.02. Many recent LLMs have moved on from both ReLU and GELU to
gated FFNs such as SwiGLU (Shazeer 2020).

______________________________________________________________________

### Q: The op sequence is the same either way — aren't the two graphs identical?

This is the deeper graph explanation behind the two-row trace above. Follow the
skip edge through the diagrams; counting `N`, `S`, and `+` alone loses it.

Write out the flat sequence of operations for a stack of `2L` sublayers, with `N`
for the norm, `S` for the sublayer body, and `+` for the residual add:

```
pre-norm (with its final norm):   N S +  N S +  N S +  …  N S +  N
post-norm:                          S + N  S + N  S + N  …  S + N
```

Same symbols, same order, same counts. The two sequences differ by exactly one
leading `N` — not "roughly," but identically up to that single boundary term. So
the question is a fair one, and the answer is that the flat reading is lossy.

#### `+` is a merge, not a unary op

Writing the ops in a line treats `+` as one more link in a chain. It isn't. It is
the one node with **two** incoming edges, and the linear text silently drops the
second one.

```
pre-norm                          post-norm

x ──────┬──────────► +            x ──────┬──────► + ──► N ──►
        │            ▲                    │        ▲
        └─► N ─► S ──┘                    └─► S ───┘

  branch point is BEFORE N          branch point is AFTER N
  → N sits on the sublayer branch   → N sits on the trunk
```

In pre-norm the skip connection taps `x` *upstream* of the norm, so the norm sits
on a side branch that dead-ends into the add. In post-norm the tap comes off the
norm's output, so the norm sits on the trunk and everything downstream — including
the next block's skip connection — inherits it.

#### The edge set, for two sublayers

Label the adds `A₁, A₂`, the norms `N₁, N₂`, the sublayers `S₁, S₂`:

```
pre-norm

x₀ ──┬──────────► A₁ ──┬──────────► A₂ ──► N_final ──►
     │            ▲     │           ▲
     └─ N₁ ─ S₁ ──┘     └─ N₂ ─ S₂ ─┘

post-norm

x₀ ──┬──────► A₁ ──► N₁ ──┬──────► A₂ ──► N₂ ──►
     │        ▲            │       ▲
     └─ S₁ ───┘            └─ S₂ ──┘
```

Same node types, same counts (modulo the one extra final norm). Different edges.
Pre-norm has a direct `A₁ → A₂` edge; post-norm has none, and every transit between
consecutive adds is forced through an `N`.

That makes the two DAGs non-isomorphic. One useful invariant is: *does a path from
the input to the last residual addition touch zero internal norm nodes?*

| | Post-norm | Pre-norm |
|---|---|---|
| Direct edge `Aᵢ → Aᵢ₊₁` | No — forced through an `N` | Yes |
| Norms on the identity path | `2L` | 0, at any depth |
| A norm-free path to the last add | None exists | `x₀ → A₁ → A₂ → … → A_2L` |

No relabeling of nodes repairs that. The difference lives in the edge set, not the
node list — which is why counting ops finds nothing and why the gap *widens* with
depth rather than washing out. The two are identical only at `L = 0`.

#### A witness you can run

Force every sublayer to output exactly zero (`S ≡ 0`) and the two stacks collapse
to different expressions:

```
pre-norm:   N_final(x₀)           one norm, whatever L is
post-norm:  N ∘ N ∘ … ∘ N (x₀)    2L norms
```

Symbolically those are very different. Numerically there is a catch worth knowing:
with freshly initialized parameters (`γ = 1`, `β = 0`), `LayerNorm` is idempotent —
`N(x)` already has zero mean and unit variance, so `N(N(x)) = N(x)` up to the `ε`
term. At init the two collapse to the *same* vector. Give the norms non-default
affine parameters, as training does, and they separate at once:

```
fresh init (γ = 1, β = 0):   max|pre − post| = 0.0000
random γ, β:                 max|pre − post| = 2.0439
```

(`d_model = 8`, 6 sublayers, `x₀ ~ 10 · N(0, 1)`, `γ ~ N(1, 0.3²)`, `β ~ N(0, 0.3²)`.)

So the zeroed-sublayer test witnesses that the *graphs* differ; it is not evidence
that the two models differ at step 0. The structural argument above holds either
way, and the gradient consequence in the next section is what bites during
training.

#### Where the distinction hides in the code

`nn.Sequential` trains you to read a network as a list of layers. A residual
network needs an explicit branch and addition, which is why this block writes
`forward` directly. In `EncoderBlock.forward`, the skip edge is the bare
`batch` on the left of the `+`:

```python
attn_in_normalized = self.attn_in_norm(batch)
attn, weights = self.attn(x=attn_in_normalized, mask=pad_mask)
batch = batch + self.dropout(attn)
```

The normalized value is bound to `attn_in_normalized` and never written back to `batch`, so
the `batch` on the right of the `+` is still the *unnormalized* value that entered
the block. A hypothetical post-norm variant moves normalization after the add:

```python
attn, weights = self.attn(x=batch, mask=pad_mask)
batch = post_attn_norm(batch + self.dropout(attn))
```

Which value the name `batch` refers to at the moment of the add is the entire
pre-norm/post-norm distinction, and it is invisible in any flattened listing of the
layers.

______________________________________________________________________

### Q: Mechanically, why does post-norm hurt gradient flow?

Write the local Jacobian of each recurrence with respect to its input `r_l`.

**Post-norm:**

```
r_{l+1} = LN(r_l + Sublayer(r_l))
∂r_{l+1}/∂r_l = J_LN · (I + J_Sublayer)
```

Backprop through `L` stacked layers multiplies `L` of these together:

```
∂r_L/∂r_0 = Π_l  J_LN,l · (I + J_Sublayer,l)
```

`LayerNorm`'s Jacobian is not the identity — it rescales by the input's own
standard deviation and removes the mean, so its singular values depend on the
local statistics of whatever it's fed. Every one of the `L` factors in that
product includes a non-identity `J_LN` term. Multiplying many non-identity
Jacobians together is exactly the setup that causes vanishing/exploding gradients
in any deep network — the transformer isn't special here, it's just deep.

**Pre-norm:**

```
r_{l+1} = r_l + Sublayer(LN(r_l))
∂r_{l+1}/∂r_l = I + J_Sublayer · J_LN
```

```
∂r_L/∂r_0 = Π_l  (I + J_Sublayer,l · J_LN,l)
```

Expand that product. Picking the `I` term at every single layer is one term in
the expansion with coefficient 1 no matter how deep the stack is: there is a
path from the last addition back to the input that never multiplies by a
sublayer or an internal `LayerNorm` Jacobian. That's the algebraic content of
"the residual connection gives gradients a highway." Other terms can still add
to or cancel this term, and a final output norm has its own Jacobian, so this
path does not guarantee a particular gradient norm. Post-norm has no such
internal identity path, because `LN` sits outside each `+`.

This is the same identity-mapping argument He et al. made for ResNets. Xiong et al.
(2020) worked out the transformer version, with explicit bounds covered in the next
section.

______________________________________________________________________

### Q: Why is the number of layers what breaks post-norm?

Both variants have a residual connection. What differs is whether the residual
stream is a clean identity path, and that depends on where the `LN` sits:

- **Pre-norm:** `x_{l+1} = x_l + F(LN(x_l))`. The Jacobian is
  `I + J_F · J_LN`, so the product through `L` layers includes an identity
  path. If the residual stream grows, `J_LN` often scales down the branch
  contribution; this can make training less sensitive to depth. The full
  product need not stay close to identity, and its actual gradient scale depends
  on the parameters, data, and final output path.
- **Post-norm:** `x_{l+1} = LN(x_l + F(x_l))`. The `LN` sits on the main path, so
  the backward pass is multiplied by the `LN` Jacobian (roughly a `1/σ` rescaling)
  at each of the `2L` sublayers. Those factors compound, so different layers get
  gradients of different scale, and the gap grows with `L`.

Xiong et al. (2020) make this precise at initialization. The gradient norm on the
last layer's parameters is bounded by:

```
post-norm:  O(d · √(ln d))        independent of L
pre-norm:   O(d · √(ln d / L))    shrinks as L grows
```

These are initialization-time bounds under the assumptions of that analysis,
not guarantees for every architecture throughout training. They explain why,
in Xiong et al.'s measurements, post-norm's per-layer gradient norm rises toward
the output layers while pre-norm's is more even across layers.

**Why warmup helps.** A large step size at step 0 can hit that big gradient near the
output and make training diverge. The original Transformer's answer was a
learning-rate **warmup schedule**: start the LR near zero and ramp it up over the
first few thousand steps, so the early updates stay small until the network is past
that fragile starting point. Skip warmup with post-norm and training often diverges
in the first few hundred steps.

**How deep before it breaks.** These are historical rules of thumb for
conventional post-norm training, not architectural limits:

| Depth | Post-norm |
|---|---|
| 6 encoder + 6 decoder (Vaswani et al. 2017) | Fine, with warmup |
| ~12–18 layers | Gets finicky |
| Past ~20–30 layers | Often needs extra care (Admin/DeepNet-style initialization, careful residual scaling) |

Liu et al. (2020) trace this to post-norm's output shift, which gets larger with
depth. Pre-norm, by contrast, trains 30+ layer encoders and today's 80–120 layer
LLMs. Those LLMs still usually use a warmup schedule, which also helps with
other early-training issues such as Adam's noisy early statistics. But with
pre-norm, warmup is no longer the only thing preventing divergence.

______________________________________________________________________

### Q: If pre-norm's stream is a raw, un-renormalized sum, why doesn't it blow up?

It does grow. Nothing in the pre-norm recurrence bounds `‖r_l‖` — each block adds
`Sublayer(LN(r_l))` on top of whatever was already there, and the addition is
never rescaled back down. Empirically, the residual-stream norm in pre-norm models
tends to grow across depth rather than staying flat.

Two things keep this from being a problem:

- **Every sublayer's own input is still normalized.** `LN(r_l)` is what
  self-attention and the FFN actually see, so a sublayer's forward pass operates
  on a unit-scale input regardless of how large `r_l` itself has gotten.
  `LayerNorm` here is a *read* operation — it doesn't touch what gets written back
  to the stream.
- **One more `LN` sits between the last block and the output head.** GPT-2-style
  stacks add a final `LN` (often called `ln_f`) after the last block, before the
  unembedding matrix. Without it, the ever-growing, unbounded-scale residual
  stream would be fed directly into the logit projection.

So pre-norm doesn't remove normalization — it moves every `LN` to be a *read*
at the top of a sublayer, plus one final read before the output, rather than a
*write* on the stream itself at every layer.

______________________________________________________________________

### Q: Why does this matter for interpretability specifically?

Because of what "the residual stream" is allowed to mean. Unroll the pre-norm
recurrence across all `L` layers:

```
r_L = r_0 + Sublayer_0(LN(r_0)) + Sublayer_1(LN(r_1)) + ... + Sublayer_{L-1}(LN(r_{L-1}))
```

That's an exact sum **at the residual stream**: the embedding and each sublayer
output are added without normalizing the running total. Attention outputs can be
decomposed further into head contributions. This supports direct logit
attribution (project a term onto the unembedding), activation patching (swap a
contribution and measure the downstream effect), and activation steering (add a
vector to the stream). The later sublayers depend on the changed stream, and the
final norm is nonlinear, so changing one term does **not** imply an exactly
additive change in final logits.
This is why MI code — TransformerLens included — reads the stream at
`hook_resid_pre`, the point right before a block's internal `LN`, since that value
equals the running sum of every contribution so far, untouched.

Unroll the post-norm recurrence and the sum breaks:

```
r_1 = LN(r_0 + Sublayer_0(r_0))
r_2 = LN(r_1 + Sublayer_1(r_1))
    = LN(LN(r_0 + Sublayer_0(r_0)) + Sublayer_1(r_1))
    ...
```

Every addition is immediately passed through a nonlinear `LN` before the next
term gets added. `Sublayer_0`'s contribution to `r_2` isn't the raw vector it
produced — it's whatever that vector became after being folded into `r_1`'s `LN`
and then partially re-expressed through `Sublayer_1`. There is no checkpoint where
"the stream" equals a clean sum of raw per-layer outputs. Attribution and patching
can still be applied to post-norm models, but they need to account for the
intervening normalization. This is one reason pre-norm residual streams are
especially convenient for interpretability work.

______________________________________________________________________

### Where this fits in our code

`app/src/training/transformer/modules.py` has `MultiHeadAttentionLayer`, which is only the
`Sublayer` term — it takes a batch and returns the attention output, with no
residual add and no block-level `LN` wrapping it. `EncoderBlock` in the same file
currently uses pre-norm for both sublayers. The two possible placements are:

```python
# Post-norm
x = ln(x + self_attn(x))
x = ln(x + ffn(x))

# Pre-norm
x = x + self_attn(ln(x))
x = x + ffn(ln(x))
```

For a decoder-only causal language model (see
[Attention Masks](#attention-masks-and-cross-attention--qa)), pre-norm plus a final `LN` before the
output projection is the GPT-2 convention. That final norm is a separate design
choice outside the `EncoderBlock` shown here.

______________________________________________________________________

### Common confusions: Pre-Norm vs. Post-Norm

| If you remember… | Check this distinction |
|---|---|
| “LayerNorm normalizes tokens together.” | It computes statistics across **features within each token**. Its learned `γ` and `β` are shared across tokens. |
| “Layer norm pools over the whole layer, so over every token.” | “Layer” is the contrast with “batch”: one example's layer activations. In a transformer that is one token's `d_model` vector. |
| “Pre-norm skips normalization.” | The sublayer reads `LN(r_l)`, and the stack commonly has a final `LN`; the running residual sum itself bypasses the internal norms. |
| “The identity path makes gradients constant.” | It supplies one direct path through the internal stack; other paths and the final norm still affect the total gradient. |
| “An additive stream means an additive change in logits.” | Contributions add at the stream, while later sublayers and the output normalization can respond nonlinearly. |

### Recall check: Pre-Norm vs. Post-Norm

1. For one token with features `[1, 2, 3]`, which axis supplies the mean and variance? Would multiplying that whole row by 10 change its normalized pattern much?
1. Trace the skip path across two pre-norm sublayers. Which operations does it bypass that a post-norm skip path cannot?
1. Why can a pre-norm stack have a growing residual stream without giving each sublayer a growing-scale input?

<details markdown="1">
<summary>Answers</summary>

1. The feature axis of that token. Ignoring `ε`, positive scaling gives the same normalized pattern; the learned affine parameters then act on it.
1. `r₀ → first add → second add → r₂`. It bypasses the two internal `LayerNorm` operations and both sublayer branches. Post-norm puts a `LayerNorm` after each add.
1. Each sublayer reads a normalized copy, `LN(r_l)`. The raw residual stream is carried through the additions, and a final normalization commonly sits before the output head.

</details>

**One-minute recap:** LayerNorm standardizes each token's features. Pre-norm uses
that standardized vector to *read* the residual stream, then adds a new result to
the raw stream. Post-norm normalizes the *new sum*. That placement determines
whether the internal skip path is an identity path and whether the running stream
remains an exact sum of sublayer outputs.

______________________________________________________________________

### References: Pre-Norm vs. Post-Norm

- Ba, Kiros & Hinton, *"Layer Normalization"* (2016). Introduces `LN`, normalizing
  within a single example's features rather than across the batch. arXiv:1607.06450
- Vaswani et al., *"Attention Is All You Need"* (2017). Original Transformer,
  post-norm. arXiv:1706.03762
- Xiong et al., *"On Layer Normalization in the Transformer Architecture"* (2020).
  Formal analysis of gradient norm at initialization for pre-norm vs. post-norm,
  and why post-norm needs warmup. arXiv:2002.04745
- Hendrycks & Gimpel, *"Gaussian Error Linear Units (GELUs)"* (2016). Defines
  `GELU(x) = x · Φ(x)` and its `tanh` and sigmoid approximations. arXiv:1606.08415
- Shazeer, *"GLU Variants Improve Transformer"* (2020). Gated FFN activations,
  including SwiGLU. arXiv:2002.05202
- Liu et al., *"Understanding the Difficulty of Training Transformers"* (Admin,
  EMNLP 2020). Post-norm's output shift gets larger with depth, and an
  initialization that tames it. arXiv:2004.08249
- Nguyen & Salazar, *"Transformers without Tears: Improving the Normalization of
  Self-Attention"* (IWSLT 2019). Empirical results showing pre-norm trains deeper
  models. arXiv:1910.05895
- He et al., *"Identity Mappings in Deep Residual Networks"* (2016). The
  identity-path argument for why residual connections stabilize gradients, applied
  here to `LayerNorm` placement instead of conv blocks. Tests variants that place
  BN/ReLU at different points around the add — the same edge-placement question,
  pre-dating transformers. arXiv:1603.05027
- Veit, Wilber & Belongie, *"Residual Networks Behave Like Ensembles of Relatively
  Shallow Networks"* (NeurIPS 2016). An `n`-block residual net is a DAG with `2ⁿ`
  input→output paths, not a chain of `n` — why reading a residual stack as a flat
  list of ops loses the structure. arXiv:1605.06431
- Wang et al., *"Learning Deep Transformer Models for Machine Translation"* (ACL
  2019). 30-layer pre-norm encoders that diverge under post-norm. arXiv:1906.01787
- Radford et al., *"Language Models are Unsupervised Multitask Learners"* (GPT-2,
  2019). Moved `LayerNorm` to the input of each sublayer and added a final `LN`
  after the last block.
- Elhage et al., *"A Mathematical Framework for Transformer Circuits"* (Anthropic,
  2021). The residual stream as a shared communication channel that every layer
  reads from and writes to additively. transformer-circuits.pub
- TransformerLens docs (Neel Nanda). `hook_resid_pre` / `hook_resid_post` naming
  and the standard MI convention of reading the pre-norm residual stream.

______________________________________________________________________

## Transformer Compute and Memory

**Revision question:** Given a model size, a token budget, and a set of GPUs, how
long might training take, and what must fit in each GPU's memory? Keep **total
work** (FLOPs), **work per second** (FLOP/s), and **bytes per GPU** separate.

**Reading route:** [Count the work](#count-the-training-work), then follow a
[one-billion-parameter memory budget](#a-worked-training-budget). Use
[activation memory](#activations-grow-with-the-microbatch) and
[sharding](#what-zero-shards) when the first estimate does not fit. Finish with
the [recall check](#recall-check-compute-and-memory). For the operations inside a block, see
[attention masks](#attention-masks-and-cross-attention--qa) and [pre-norm](#pre-norm-vs-post-norm).

These are planning estimates for a **dense transformer**. The arithmetic assumes
that matrix multiplications dominate training and that the model trains all of its
parameters. Attention kernels, context length, precision, optimizer, and parallel
layout change real costs. All GB figures below are decimal (`10⁹` bytes).

### Count the Training Work

| Symbol | Meaning | Unit |
|---|---|---|
| `P` | trainable parameters | parameters |
| `D` | tokens processed during training, including repeated passes | tokens |
| `C` | total training work | FLOPs |
| `N` | GPUs doing the work | GPUs |
| `r` | **sustained training throughput** per GPU | FLOP/s per GPU |
| `T` | elapsed training time | seconds |

For one token, a dense weight matrix needs roughly two floating-point operations
per weight in the forward pass: one multiply and one addition. Backpropagation
through that matrix computes both an input gradient and a weight gradient, adding
roughly twice the forward work. Across `D` tokens:

```text
forward   ≈ 2 P D FLOPs
backward  ≈ 4 P D FLOPs
training  ≈ 6 P D FLOPs
T         ≈ 6 P D / (N r) seconds
```

This is a useful first estimate, not an exact count. It omits some elementwise
operations, optimizer work, and attention's sequence-length-dependent cost. Long
contexts can make attention more significant. `r` must be a *measured* or
plausible sustained training rate for this workload; a GPU's advertised peak is
not the denominator to use for a wall-clock estimate.

#### Work through one run

Suppose `P = 1 billion`, `D = 20 billion`, `N = 8`, and a representative run
sustains `r = 50 trillion FLOP/s` on each GPU:

```text
C ≈ 6 × 10⁹ × 20 × 10⁹ = 1.2 × 10²⁰ FLOPs
N r = 8 × 50 × 10¹² = 4 × 10¹⁴ FLOP/s
T ≈ 1.2 × 10²⁰ / (4 × 10¹⁴) = 300,000 s ≈ 3.5 days
```

The units cancel: `FLOPs / (FLOP/s) = seconds`. A **petaFLOP-day** is
`10¹⁵ FLOP/s × 86,400 s = 8.64 × 10¹⁹ FLOPs`; this run uses about `1.39`
petaFLOP-days. A **GPU-hour** instead counts devices times elapsed hours and says
nothing about their throughput unless the GPU type and achieved rate are known.

#### Parameters and tokens are different choices

At fixed compute, `P × D` is approximately fixed: doubling `P` leaves roughly
half as many training tokens. The 2022 Chinchilla study found an approximate
`D ≈ 20P` balance for its compute-optimal setting. That is a result under its
training objective and budget, not a rule that every model should stop after
`20P` tokens. A smaller model trained longer may cost more compute to reach a
given loss, yet be cheaper to serve repeatedly because each generated token
reads fewer weights. Decide the inference target and training budget together.

The [source article's GPT-NeoX scaling plot](https://blog.eleuther.ai/transformer-math/#engineering-takeaways-for-compute-costs)
shows measured samples per second falling below ideal linear scaling as V100
GPU count rises. That is a historical measurement of one setup, so measure `r`
for the current model and cluster.

### Inference Memory: Weights and the KV Cache

**Weight memory** is `P × bytes per stored parameter`. The exponent determines
range and the mantissa determines precision, but only the *stored* format sets
weight bytes:

| Stored weights | Exponent bits | Mantissa bits | Bytes per parameter | `7B` parameters |
|---|---:|---:|---:|---:|
| FP32 | 8 | 23 | 4 | 28 GB |
| BF16 | 8 | 7 | 2 | 14 GB |
| FP16 | 5 | 10 | 2 | 14 GB |
| INT8 | — | — | about 1, plus scale metadata | about 7 GB plus metadata |

Every floating-point row also has one sign bit. TF32 is a **matrix-multiply
compute mode** with 8 exponent and 10 mantissa bits; it does not turn an FP32
weight tensor into a 19-bit storage format. FP32 weights still take 4 bytes each.
Quantized weights need scales and sometimes other metadata, so their exact memory
depends on the quantization method. The
[source's precision diagram](https://blog.eleuther.ai/transformer-math/#model-weights)
shows these bit layouts side by side.

For autoregressive inference, attention also keeps past keys and values. With
ordinary multi-head attention, an approximate KV-cache budget is:

```text
KV bytes ≈ 2 × layers × batch × cached_tokens × KV_width × bytes_per_value
```

The leading `2` counts keys and values. `KV_width = d_model` for ordinary
multi-head attention; grouped-query or multi-query attention uses fewer KV heads
and a smaller width. For `32` layers, batch `1`, `4096` cached tokens,
`KV_width = 4096`, and two-byte values, the cache alone is about `2.15 GB`.
A 7B BF16 model therefore needs more than its `14 GB` of weights, even before
temporary buffers and allocator overhead. The source article's `1.2 × weight memory` rule of thumb can miss a large cache at long context or high batch size.

### A Worked Training Budget

Training keeps more than weights. Use this **specific** mixed-precision Adam
layout: two-byte working weights and gradients, an FP32 master weight copy, and
FP32 Adam first and second moments. Before activations and temporary buffers:

| Resident item | Bytes per parameter | For `P = 1B` |
|---|---:|---:|
| Working weights | 2 | 2 GB |
| Gradients | 2 | 2 GB |
| FP32 master weights | 4 | 4 GB |
| Adam first moment | 4 | 4 GB |
| Adam second moment | 4 | 4 GB |
| **Total parameter state** | **16** | **16 GB** |

That `16P` figure belongs to this layout. Native autocast may instead keep FP32
weights and gradients with two FP32 moments (`4 + 4 + 8 = 16` bytes per
parameter), without a persistent low-precision weight copy. Optimizers with
different state, quantized moments, or frozen parameters change the total.

The full device estimate starts with four buckets:

```text
training memory ≈ weights + gradients + optimizer state + activations
                  + temporary/communication buffers
```

The first three buckets scale mostly with `P`. Activations depend on the
**microbatch on one GPU**, sequence length, hidden width, and layers. A model
that fits at batch size one may run out of memory when the microbatch grows,
even though `P` stays fixed.

#### Activations grow with the microbatch

Backpropagation needs intermediate forward results. Keeping all of them saves
compute but uses memory. **Activation checkpointing** saves selected boundaries
and recomputes discarded intermediates during backward. This trades extra
forward work for a smaller activation bucket.

One published estimate for FP16 activations, with no sequence parallelism, uses
`s` = sequence length, `b` = microbatch per GPU, `h` = hidden width, `L` = layers,
`a` = attention heads, and `t` = tensor-parallel degree:

```text
no recomputation         ≈ s b h L (10 + 24/t + 5 a s/(h t)) bytes
selective recomputation  ≈ s b h L (10 + 24/t) bytes
full recomputation       ≈ 2 s b h L bytes
```

The `s²` contribution in the first line comes from attention intermediates;
selective recomputation removes it in this model. These constants describe the
implementation analyzed by *Reducing Activation Recomputation in Large
Transformer Models*, so use them to reason about scaling rather than as an
allocator-exact promise for every attention kernel. Full recomputation can add
up to roughly one extra forward pass: the `6PD` training estimate can approach
`8PD` before other overheads.

For a roughly 1B-parameter shape with `h = 2048`, `L = 18`, `a = 16`,
`s = 2048`, `b = 2`, and `t = 1`, `s b h L = 150,994,944`. The formulas give:

| Activation policy | Approximate activation bytes per GPU | With the `16 GB` replicated parameter state |
|---|---:|---:|
| No recomputation | 17.2 GB | 33.2 GB |
| Selective recomputation | 5.13 GB | 21.1 GB |
| Full recomputation | 0.30 GB | 16.3 GB |

Even the `21.1 GB` selective estimate leaves less than 3 GB on a 24 GB GPU
for temporary buffers and allocator overhead. The
[source's activation chart](https://blog.eleuther.ai/transformer-math/#activations-and-batch-size)
illustrates the same tradeoff for larger models; its red line marks an 80 GB
GPU limit.

### What ZeRO Shards

Ordinary data parallelism gives each GPU a complete copy of the weights,
gradients, and optimizer state. Let `d` be the number of **data-parallel
replicas**, `W` the working-weight bytes, `G` the gradient bytes, `O` the
optimizer-state bytes, and `A` each GPU's activations. ZeRO progressively
partitions parameter state among those `d` ranks:

| Layout | Approximate resident bytes per GPU | What is partitioned |
|---|---|---|
| Replicated | `W + G + O + A` | nothing |
| ZeRO-1 | `W + G + O/d + A` | optimizer state |
| ZeRO-2 | `W + (G + O)/d + A` | optimizer state, gradients |
| ZeRO-3 | `(W + G + O)/d + A + live buffers` | optimizer state, gradients, weights |

ZeRO-3 gathers weights needed for the current computation and frees or reuses
them later. The live gathered weights and communication buffers mean its peak
memory is higher than the partitioned-state term alone. Sharding also adds
communication; the cheapest memory layout is not automatically the fastest run.

The [source article's ZeRO illustration](https://blog.eleuther.ai/transformer-math/#sharded-optimizers),
credited there to the [ZeRO paper](https://arxiv.org/abs/1910.02054), shows
the same progression: first the optimizer state, then gradients, then
parameters are partitioned across data-parallel ranks. Activations remain a
separate budget.

Return to the `P = 1B` layout above, with `d = 8` and no model parallelism.
Here `W = 2 GB`, `G = 2 GB`, and `O = 12 GB` (master weights plus two Adam
moments):

| Layout | Parameter-state calculation | Approximate bytes per GPU before activations |
|---|---|---:|
| Replicated | `2 + 2 + 12` | 16 GB |
| ZeRO-1 | `2 + 2 + 12/8` | 5.5 GB |
| ZeRO-2 | `2 + (2 + 12)/8` | 3.75 GB |
| ZeRO-3 | `(2 + 2 + 12)/8` | 2 GB **plus live buffers** |

Add each GPU's activation and temporary-buffer budget to every row. The table
is a storage estimate, not a guaranteed peak-memory measurement.

### How Data, Tensor, and Pipeline Parallelism Combine

| Dimension | Splits | Main consequence |
|---|---|---|
| Data parallel (`d`) | training examples across complete model replicas | higher token throughput; ZeRO can shard state across these ranks |
| Tensor parallel (`t`) | individual layer computations across GPUs | smaller local weight partitions, with communication inside layers |
| Pipeline parallel (`p`) | groups of layers across GPUs | smaller local layer sets, with in-flight microbatch activations |

If these axes form a regular mesh, `N = d × t × p`; therefore **ZeRO's shard
count is `d = N/(t p)`**, not `N`. For example, 32 GPUs with `t = 4` and
`p = 4` have only `d = 2` full model replicas. For a local microbatch of `b`
examples per replica and `k` gradient-accumulation microbatches, the global
batch is `b × d × k`. Tensor and pipeline ranks working on the same example do
not multiply that batch size.

The [source's 32-GPU diagram](https://blog.eleuther.ai/transformer-math/#sharded-optimizers--3d-parallelism)
draws these as three directions: data parallelism supplies independent examples
to complete model replicas; tensor and pipeline parallelism partition each
replica's computation.

As a first approximation, tensor and pipeline parallelism can reduce local
weight storage toward `W/(t p)` when layers and matrices partition evenly.
Actual peaks depend on unbalanced layers, pipeline microbatches, gathered
parameters, activations, and communication buffers. Measure a representative
step before committing to a cluster size.

### Common confusions: Compute and Memory

| Tempting shortcut | Check this instead |
|---|---|
| “Peak TFLOP/s tells me training time.” | Use sustained **training** FLOP/s measured for a similar workload and parallel layout. |
| “`D` is the number of unique tokens on disk.” | Count every token the model processes, including repeated epochs. |
| “TF32 means weights take fewer than four bytes.” | TF32 changes compute precision for FP32 matrix operations; FP32 storage still takes four bytes. |
| “A 7B BF16 model needs exactly 14 GB to serve.” | Add the KV cache, temporary buffers, and any quantization metadata. |
| “Training always costs 16 bytes per parameter.” | Name the weight, gradient, and optimizer dtypes and whether a master copy exists. |
| “Eight GPUs give eight ZeRO shards.” | Only the **data-parallel degree** counts; tensor and pipeline ranks split one replica. |

### Recall check: Compute and Memory

1. A dense model has `2B` parameters and trains on `40B` tokens. About how many training FLOPs does `6PD` predict?
1. Why can a model's inference memory rise while its weight file remains unchanged?
1. In the worked `1B`-parameter layout, which three buckets make up the `12 GB` optimizer state?
1. With `N = 32`, `t = 4`, and `p = 2`, how many data-parallel ranks can share ZeRO state?

<details markdown="1">
<summary>Answers</summary>

1. `6 × 2 × 10⁹ × 40 × 10⁹ = 4.8 × 10²⁰` FLOPs. Divide by *sustained aggregate* FLOP/s to estimate seconds.
1. The KV cache grows with the number of cached tokens and the inference batch. Temporary buffers can grow too.
1. The FP32 master weights, Adam first moment, and Adam second moment: `4 + 4 + 4` bytes per parameter.
1. `d = 32/(4 × 2) = 4` data-parallel ranks.

</details>

**One-minute recap:** `6PD` estimates dense-transformer training work; dividing
by sustained aggregate FLOP/s estimates time. Weight bytes are only the start of
an inference budget, because the KV cache grows with context and batch. Training
adds gradients, optimizer state, and activations. Checkpointing spends compute to
reduce activations; ZeRO partitions parameter state across **data-parallel**
ranks. Always state the precision and parallel layout behind a memory number.

______________________________________________________________________

### References: Compute and Memory

- Anthony, Biderman & Schoelkopf, [*Transformer Math 101*](https://blog.eleuther.ai/transformer-math/)
  (EleutherAI, 2023). Source for the planning framework, historical hardware
  measurements, and linked figures. Its `1.2×` inference heuristic describes
  the article's setting.
- Hoffmann et al., [*Training Compute-Optimal Large Language Models*](https://arxiv.org/abs/2203.15556)
  (2022). The parameter/token tradeoff behind the approximate `D ≈ 20P` result.
- Korthikanti et al., [*Reducing Activation Recomputation in Large Transformer
  Models*](https://arxiv.org/abs/2205.05198) (2022). Activation-memory formulas
  and selective recomputation.
- Rajbhandari et al., [*ZeRO: Memory Optimizations Toward Training Trillion
  Parameter Models*](https://arxiv.org/abs/1910.02054) (2020). The three
  progressive state-partitioning stages.
