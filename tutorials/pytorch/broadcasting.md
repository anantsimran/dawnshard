# Shapes and Broadcasting — Q&A

**One rule to carry through the page:** align shapes from the right. Equal sizes match; a size-1 axis can be reused; missing axes act like size 1. When a value was reused in the forward pass, its gradient adds those uses together on the way back.

| If the question is… | Read |
|---|---|
| Why `(3,)` differs from `(1,3)` | [shape notation](#q--shape-3-what-does-this-mean) |
| Whether two shapes combine | [broadcastability](#q-what-does-broadcastable-mean) and [`dim=k`](#q-what-does-dimk-actually-mean) |
| Whether expansion allocates | [`expand` versus `repeat`](#q-what-is-the-difference-between-expand-and-repeat) |
| Why attention masks need extra axes | [attention masks](#q-how-does-broadcasting-apply-to-attention-masks) |
| Why a bias gradient sums over examples | [broadcast gradients](#q-what-happens-to-gradients-when-a-tensor-is-broadcast) |

The sections are independent Q&A references. For a quick revision, follow the shape notation, broadcastability, and broadcast gradients questions in that order.

______________________________________________________________________

## Q: `# shape (3,)`. What does this mean?

That notation mixes two separate things.

### 1. The comma is Python syntax

`.shape` returns a tuple (`torch.Size` is a subclass of `tuple`). In Python, `(3)` is just
the number 3 in parentheses, which do nothing. A one-element tuple needs a
trailing comma:

```python
type((3))    # <class 'int'>    ← just a number
type((3,))   # <class 'tuple'>  ← a 1-element tuple
```

`(3,)` isn't "3 and then something omitted." The comma just tells Python it's a tuple.

### 2. What the tuple says

**Length of the tuple = number of dimensions (axes). Each number = how many elements lie
along that axis.**

```python
torch.tensor(5.).shape                    # ()      0 dims — a scalar
torch.tensor([1., 2., 3.]).shape          # (3,)    1 dim, 3 elements
torch.tensor([[1., 2., 3.]]).shape        # (1, 3)  2 dims: 1 row, 3 cols
torch.tensor([[1.], [2.], [3.]]).shape    # (3, 1)  2 dims: 3 rows, 1 col
```

Count how deeply the brackets are nested. That's the number of dims:

```
[1., 2., 3.]      one level of brackets   → (3,)
[[1., 2., 3.]]    two levels              → (1, 3)
```

### 3. Intuition: how many indices to reach one value

```python
a = torch.tensor([1., 2., 3.])     # (3,)
a[0]                               # tensor(1.) — one index, done

b = torch.tensor([[1., 2., 3.]])   # (1, 3)
b[0]                               # tensor([1., 2., 3.]) — still a row!
b[0][0]                            # tensor(1.) — needed two
```

Both hold the same 3 numbers. They differ in how the numbers are addressed, and
broadcasting cares about exactly that.

### 4. Why it matters: `(3,)`, `(1, 3)` and `(3, 1)` are different tensors

```python
a = torch.tensor([1., 2., 3.])         # (3,)
c = torch.tensor([[1.], [2.], [3.]])   # (3, 1)

a + c
# tensor([[2., 3., 4.],
#         [3., 4., 5.],
#         [4., 5., 6.]])   shape (3, 3)
```

Adding two tensors of 3 numbers each gave 9 numbers. `(3,)` is treated as `(1, 3)`, then `(1, 3)` vs
`(3, 1)` stretches both ways into `(3, 3)`. This is the most common silent bug in
numerical Python: the code runs, the shape quietly grows, and you find out 40 lines
later.

**Habit:** print `.shape` constantly while developing. Print the whole tuple, not `.shape[0]`.

______________________________________________________________________

## Q: What does "broadcastable" mean?

**Broadcastable = shapes that don't match but can be made to match by stretching size-1
dimensions.**

### Step 1: You already use it

```python
x = torch.tensor([1., 2., 3.])   # (3,)
x + 10                           # tensor([11., 12., 13.])
```

Elementwise addition should need matching shapes. This works because the scalar `10` is
conceptually copied to every position. That's broadcasting.

### Step 2: The rule (two lines)

Line up the shapes **right to left**. Two dimensions are compatible if:

1. they're equal, or
1. one of them is `1`.

Missing dimensions on the left count as `1`.

```
A:  (2, 3)
B:     (3,)   →  treated as (1, 3)
        ↑ align from the right
```

Dim -1: `3 == 3` ✓. Dim -2: `2` vs `1` ✓. Result: `(2, 3)`.

### Step 3: Why `1` is special

**A dimension of size 1 means "this value does not vary along this axis."**

If you have one row and need four, just reuse the same row four times. If you have 2 rows
and need 4, it's unclear which ones to duplicate. That's why only `1` stretches.

```python
row = torch.tensor([[10., 20., 30.]])   # (1, 3) — same for every row
col = torch.tensor([[1.], [2.]])        # (2, 1) — same for every column

row + col
# tensor([[11., 21., 31.],
#         [12., 22., 32.]])   shape (2, 3)
```

Both get stretched: `row` repeats down and `col` repeats across.

**The stretched operand is not copied in memory.** PyTorch can set the stretched axis's stride to 0, so every index along it reads the same memory. An arithmetic result such as `row + col` still allocates its output:

```python
torch.randn(3, 1).expand(3, 4).stride()   # (1, 0)
```

### Step 4: Practice

| Shape A | Shape B | Result | Why |
|---|---|---|---|
| `(5, 4)` | `(1,)` | `(5, 4)` | 4↔1 ✓, 5↔(missing = 1) ✓ |
| `(5, 4)` | `(4,)` | `(5, 4)` | 4↔4 ✓, 5↔1 ✓ |
| `(5, 4)` | `(5,)` | **error** | right-alignment puts 5 against 4 |
| `(15, 3, 5)` | `(3, 1)` | `(15, 3, 5)` | 5↔1 ✓, 3↔3 ✓, 15↔1 ✓ |
| `(2, 3)` | `(3, 2)` | **error** | 3↔2 ✗ |

Row 3 is the one that catches people: `(5,)` doesn't match `(5, 4)` even though both contain 5,
because alignment starts from the right.

______________________________________________________________________

## Q: What is the difference between `expand` and `repeat`?

Both turn `(3, 1)` into `(3, 4)`. Only one of them allocates.

```python
t = torch.randn(3, 1)

e = t.expand(3, 4)   # a view: stride (1, 0), still 3 floats of storage
r = t.repeat(1, 4)   # a copy: stride (4, 1), 12 floats of storage

e.data_ptr() == t.data_ptr()   # True  — same memory
r.data_ptr() == t.data_ptr()   # False — new memory
```

**`expand` is what broadcasting does under the hood.** It sets the stretched axis's stride
to 0, so every index along that axis reads the same element. Writing `a + b` on
broadcastable shapes performs this expansion for you, which is why you rarely call
`expand` by hand: you need it only when an op *doesn't* broadcast, such as the index
argument of `gather`, or `torch.cat`, which requires real sizes in every slot.

### What each one accepts

`expand` stretches size-1 axes and nothing else:

```python
torch.randn(3, 2).expand(3, 4)
# RuntimeError: The expanded size of the tensor (4) must match the existing size (2)
#               at non-singleton dimension 1.
```

`-1` keeps an axis as it is, and new axes may be added on the left, matching the
right-alignment rule:

```python
torch.randn(3, 1).expand(-1, 4).shape      # (3, 4)
torch.randn(3, 1).expand(5, 3, 4).shape    # (5, 3, 4)
```

`repeat` tiles the **whole tensor** a given number of times per axis. It is not NumPy's
`np.repeat`; the element-wise one is `repeat_interleave`:

```python
v = torch.tensor([1, 2, 3])
v.repeat(2)              # tensor([1, 2, 3, 1, 2, 3])
v.repeat_interleave(2)   # tensor([1, 1, 2, 2, 3, 3])
```

### The two gotchas of the cheap one

**An expanded view aliases.** Every stretched position is the same element, so one
in-place write lands in all of them, with no error:

```python
t = torch.zeros(3, 1)
e = t.expand(3, 4)
e[0, 0] = 5.0
e[0]          # tensor([5., 5., 5., 5.])  ← wrote one value, changed four
```

**An expanded view isn't contiguous.** `view` walks memory in order and refuses;
`reshape` silently falls back to a copy, which is the allocation you were avoiding:

```python
e.is_contiguous()   # False
e.view(12)          # RuntimeError: view size is not compatible with input tensor's
                    #               size and stride ...
e.reshape(12)       # works — copies
```

### Which to reach for

| You want | Use | Because |
|---|---|---|
| To line shapes up for an elementwise op | nothing | `+`, `*`, `where` broadcast already |
| A read-only stretch for an op that won't broadcast | `expand` | zero-copy |
| To write into the result, or hand it to `view` | `repeat` | you need real, independent elements |
| Each element duplicated in place (`[1,1,2,2]`) | `repeat_interleave` | `repeat` tiles, it doesn't interleave |

______________________________________________________________________

## Q: Does matmul broadcast the same way?

Almost. `@` splits the shape in two: **the last two slots are the matrix, and everything
to the left of them is batch and broadcasts by the usual right-to-left rule.**

```python
A = torch.randn(10, 3, 4)
B = torch.randn(4, 5)
(A @ B).shape                                          # (10, 3, 5)

torch.randn(10, 1, 3, 4) @ torch.randn(2, 4, 5)        # (10, 2, 3, 5)
```

In the second line, `(3, 4) @ (4, 5)` is the matrix part, and the batch parts `(10, 1)`
and `(2,)` broadcast to `(10, 2)`.

**The inner dimension never broadcasts.** The matrix rule is exact: the `4` in `(3, 4)`
must equal the `4` in `(4, 5)`. A size-1 inner dim is not stretched to fit.

```python
torch.randn(10, 3, 4) @ torch.randn(10, 5)
# RuntimeError: mat1 and mat2 shapes cannot be multiplied (30x4 and 10x5)
```

That error message is worth reading closely: the `10` on the right was treated as a
matrix row count, not a batch, because the right operand has only two dims. Batch dims
that genuinely conflict fail differently:

```python
torch.randn(2, 3, 4) @ torch.randn(3, 4, 5)
# RuntimeError: The size of tensor a (2) must match the size of tensor b (3)
#               at non-singleton dimension 0
```

### 1-D operands

A 1-D operand gets a dimension added for the duration and removed afterwards — prepended
on the left, appended on the right:

```python
torch.randn(3) @ torch.randn(3)          # ()      dot product
torch.randn(3) @ torch.randn(3, 5)       # (5,)    row vector times matrix
torch.randn(10, 3, 4) @ torch.randn(4)   # (10, 3) batched matrix times column
```

### Where this repo uses it

[modules.py:140](../../app/src/training/transformer/modules.py#L140) projects all heads
in one call instead of looping over them:

```python
v = batch.unsqueeze(dim=1) @ self.w_v
# (B, 1, L, d_model) @ (h, d_model, d_k) -> (B, h, L, d_k)
```

The matrix part is `(L, d_model) @ (d_model, d_k)`. The batch part is `(B, 1)` against
`(h,)`, which broadcasts to `(B, h)`: the `1` says "the same tokens go to every head" and
the `h` says "each head has its own weight." The head axis appears in the output without
a single reshape. It is exactly equal to stacking a per-head loop:

```python
torch.allclose(v, torch.stack([batch @ w_v[i] for i in range(h)], dim=1))   # True
```

The scores then use the same rule with no broadcasting at all, since `q` and `k` already
agree on `(B, h)`:

```python
q @ k.transpose(-2, -1)   # (B,h,L,d_k) @ (B,h,d_k,L) -> (B, h, L, L)
```

**`bmm` and `mm` do not broadcast.** They are the strict versions, which makes them a
useful assertion when you want a shape mistake to be loud:

```python
torch.bmm(torch.randn(10, 3, 4), torch.randn(4, 5))
# RuntimeError: batch2 must be a 3D tensor
```

______________________________________________________________________

## Q: How does broadcasting apply to attention masks?

See [attention_masks.md](../transformer/attention_masks.md) for the full mask story. On
the shape side:

```
scores:  (B, h, L_q, L_k)
mask:    (B, 1,   1, L_k)
          ✓  ✓    ✓   ✓
```

Read the `1`s as meaning, not as filler:

- `1` in the head slot = "this mask is the same for every attention head"
- `1` in the query slot = "every query position sees the same set of forbidden keys"
- `L_k` = "masking depends on which key" (e.g. whether it's a pad token)
- `B` = "masking depends on which sequence in the batch" (each one pads differently)

That's exactly a padding mask: whether key `j` is real depends on the sequence and on nothing
else.

A causal mask *does* depend on query position and is identical across the batch:

```
mask:    (1, 1, L_q, L_k)
```

### The gotcha this protects you from

A naively built padding mask is `(B, L_k)`. Pass that in:

```
scores:  (B, h, L_q, L_k)
mask:          (B,  L_k)   ← right-aligned
```

Now `B` sits against `L_q`. Usually that raises an error. But if the batch size equals the sequence
length (e.g. `B = 32`, `L_q = 32`), it silently broadcasts wrong and masks the wrong
things with no exception. That's why the code uses an explicit `mask[:, None, None, :]`.

**Rule of thumb:** always unsqueeze explicitly. Never rely on the missing left dimensions to guess your
intent.

______________________________________________________________________

## Q: What does `dim=k` actually mean?

### The one rule

A shape is a tuple. `dim=k` / `axis=k` names **slot k of that tuple**. Negative values
count from the right, so `dim=-1` is the last slot. That's the entire meaning of the
argument.

The part that varies is the second question: *what does the op do to that slot?*

| Op family | Effect on slot k |
|---|---|
| `sum`, `mean`, `max`, `argmax`, `norm` | slot **deleted** |
| same, with `keepdim=True` | slot set to **1** |
| `softmax`, `cumsum`, `sort`, `F.normalize` | slot **unchanged**, values rewritten along it |
| `unsqueeze`, `stack` | new slot **inserted** at k |
| `cat` | slot k **grows**, all others must match |
| `squeeze` | slot k deleted if it has size 1; otherwise a silent no-op in PyTorch (NumPy raises) |

### Reductions: the index that vanishes

Intuition: fix every *other* index and slide along slot k. That gives you a 1-D vector.
The op eats that vector and returns a scalar. Repeat for every combination of the other
indices.

In subscripts, for `x` of shape `(B, L, D)`:

```
out[b, d] = Σ_l  x[b, l, d]        # this is dim=1
```

The named index disappears from the left-hand side. Verify by crossing it out of the
tuple:

```python
x = np.arange(24).reshape(2, 3, 4)
x.sum(axis=0).shape   # (3, 4)
x.sum(axis=1).shape   # (2, 4)
x.sum(axis=2).shape   # (2, 3)
```

**Example:** `attn` is `(B, L, D_MODEL)`, so `.sum(dim=1)` crosses out `L` →
`(B, D_MODEL)`. Each token contributed a `D_MODEL`-vector, and you added them into one
vector per sequence. See the masked mean question below for the full pooling example.

### Softmax: the axis you're choosing among

`attn_scores` is `(B, h, L_q, L_k)`. `dim=-1` is `L_k`. Softmax normalizes rather than
reduces, so the shape is unchanged, but now `scores.softmax(dim=-1).sum(dim=-1)` is all
ones.

Meaning: each **query** spreads one unit of attention mass across all **keys**. `dim=-2`
would make each key's column sum to 1, i.e. attending over queries. That's the wrong
direction, and it won't raise, so nothing tells you.

Heuristic: softmax goes over the axis whose entries are the *candidates you're choosing
between*.

### Insertion and joining

Here `dim` means **where the new slot lands in the output**:

```python
torch.stack([a, b, c], dim=1)   # 3 tensors of (B, D) -> (B, 3, D)
torch.stack([a, b, c], dim=0)   # -> (3, B, D)
torch.cat([a, b], dim=-1)       # (B, D1) + (B, D2) -> (B, D1+D2)
```

`stack` needs identical shapes and adds a dimension. `cat` needs the same number of
dimensions and matching sizes in every slot except `k`.

### Shape ops

`transpose(k1, k2)` swaps two slots. `permute(...)` reorders all of them.
`reshape`/`view` ignore slots entirely and just re-window the flat memory buffer. That's
why the canonical multi-head split reads the way it does:

```python
x.view(B, L, h, D // h).transpose(1, 2)   # (B, L, D) -> (B, h, L, D_head)
```

You cannot do that in one reshape to `(B, h, L, D_head)`. `view` walks memory in order
and doesn't know your slots mean anything, so it would interleave heads with tokens.

### Habits that end guess-and-check

Assert the shape at the line that produces it:

```python
assert pooled.shape == (B, D_MODEL), pooled.shape
```

Or name your dims instead of numbering them, with
[einops](https://einops.rocks/) (not a dependency of this repo):

```python
from einops import rearrange, reduce
pooled = reduce(attn, "b l d -> b d", "sum")
x = rearrange(x, "b l (h dh) -> b h l dh", h=h)
```

The second line is the multi-head split above, and it documents itself.

### Not covered here

`gather` and `index_select` are the harder tier. For `gather`, the `dim` argument stops
meaning "slot that disappears" and starts meaning "slot whose indices you're replacing."

______________________________________________________________________

## Q: How does masked mean pooling work?

The goal: average each sequence's token vectors into one vector, ignoring padding.

```python
def masked_mean(attn, mask):
    # attn: (B, L, D_MODEL), mask: (B, L), True = real token
    m = mask.unsqueeze(-1)                                    # (B, L, 1)
    summed = attn.masked_fill(mask=~m, value=0.0).sum(dim=1)  # (B, D_MODEL)
    counts = mask.sum(dim=1, keepdim=True).clamp(min=1)       # (B, 1)
    return summed / counts                                    # (B, D_MODEL)
```

Yes, you can zero out the padding and then average, but you have to divide by the number
of real tokens yourself. `.mean(dim=1)` divides by `L`, which counts the pads too.

### Why `unsqueeze(-1)` and not `unsqueeze(1)`

`mask.unsqueeze(1)` is `(B, 1, L)`. Right-aligned against `(B, L, D_MODEL)`, that puts
`L` against `D_MODEL`. You need `(B, L, 1)` so the mask lines up on `B` and `L` and
stretches across the feature slot.

### What `sum(dim=1)` does, by hand

B=1, L=3, D_MODEL=2, last token is padding:

```python
attn = [[[1, 2],    # token 0
         [3, 4],    # token 1
         [9, 9]]]   # token 2 (pad)
mask = [[True, True, False]]
```

After `masked_fill`, the pad row is zeros:

```
[[[1, 2],
  [3, 4],
  [0, 0]]]
```

`.sum(dim=1)` adds the rows, one feature column at a time:

```
[[1+3+0, 2+4+0]] = [[4, 6]]      # shape (1, 2)
```

`mask.sum(dim=1, keepdim=True)` counts the `True`s per sequence: `[[2]]`, shape `(1, 1)`.

```
[[4, 6]] / [[2]] = [[2, 3]]
```

That's the mean over the real tokens only. `.mean(dim=1)` would give `[[4/3, 6/3]]`,
because it counts the pad row too.

### How `clamp(min=1)` prevents NaN

`clamp(min=1)` raises anything below 1 up to 1 and leaves everything else alone:

```python
torch.tensor([0, 1, 2, 5]).clamp(min=1)   # tensor([1, 1, 2, 5])
```

Counts are integers, so the only value it can change is `0`, which happens when a row is
entirely padding. Then `summed` is all zeros and the division is `0 / 0 = NaN`. NaN
spreads: every later layer, the loss, and the gradients go NaN, and training silently
breaks. With the clamp it's `0 / 1 = 0`, a zero vector. Rows with at least one real token
are unchanged.

### Why `keepdim=True` matters here

Broadcasting right-aligns shapes. `(B, 1)` against `(B, D_MODEL)` stretches correctly.
Without `keepdim` the count is `(B,)`, which right-aligns against `D_MODEL`: it raises if
`B != D_MODEL` and **silently divides the wrong entries** if `B == D_MODEL` (e.g. batch
size 512 with `d_model` 512).

The opposite mistake is just as quiet. `keepdim` is only right on the 2-D `mask`. On the
3-D `m`, `m.sum(dim=1, keepdim=True)` is `(B, 1, 1)`, and `(B, D_MODEL) / (B, 1, 1)`
broadcasts to `(B, B, D_MODEL)` with no error. `m.sum(dim=1)` without `keepdim` is
already `(B, 1)`, so either `mask.sum(dim=1, keepdim=True)` or `m.sum(dim=1)` is correct,
but not a mix of the two.

______________________________________________________________________

## Q: What happens to gradients when a tensor is broadcast?

**Forward stretches, backward sums.** A gradient always has the shape of the tensor it
belongs to, so the extra entries produced by stretching have to be folded back up.

```python
b = torch.zeros(3, requires_grad=True)   # (3,)
out = torch.ones(4, 3) + b               # (4, 3) — b stretched down 4 rows
out.sum().backward()

b.grad          # tensor([4., 4., 4.])
b.grad.shape    # (3,)  — not (4, 3)
```

`b` appears in 4 places in the output, each with an incoming gradient of 1, so its
gradient is their sum: 4. This is not a special case in autograd. Broadcasting inserts a
real `expand` node into the graph, and the backward of `expand` is a sum over the
stretched axes with `keepdim`, which restores the original shape by construction.

The sum is over exactly the axes that were stretched, weighted by whatever came back:

```python
g = torch.tensor([[1., 2., 3.],
                  [10., 20., 30.]])      # the incoming gradient

b = torch.zeros(3, requires_grad=True)   # (3,) stretched down -> column sums
# b.grad == tensor([11., 22., 33.])

c = torch.zeros(2, 1, requires_grad=True)  # (2,1) stretched across -> row sums
# c.grad == tensor([[6.], [60.]])
```

A `(3,)` operand varies along the last axis, so it sums down the rows. A `(2, 1)` operand
varies down the rows, so it sums across. **The axis with the `1` is the axis you sum
over**, which is the same reading as the forward rule: a `1` means "this value does not
vary along this axis," so every position along it contributed the same parameter.

### Why a bias grad grows with the batch

That is the rule behind one of the most-asked training questions. A bias is broadcast over
the batch, so its gradient is the *sum* over the batch, not the mean. Double the batch
size and the raw bias gradient roughly doubles.

Losses often offset this by reducing with `mean`, which divides by the number of examples in the ordinary unweighted case — the
default for common losses such as `MSELoss` and `CrossEntropyLoss`. Switch a loss to `reduction="sum"` and
every gradient in the model scales with batch size, so the learning rate that worked at
batch 32 blows up at batch 256. See
[training_loop.md › Aggregating Loss](../training/training_loop.md#aggregating-loss-correctly-across-an-epoch).

### The bug this makes silent

Because backward always repairs the shape, a wrong broadcast in the forward pass produces
a perfectly valid gradient of the right shape. Nothing downstream complains.

```python
pred = torch.randn(8, 1)   # model output kept its trailing dim
targ = torch.randn(8)      # labels came out of the loader flat

(pred - targ).shape        # (8, 8)  ← 64 pairwise differences, not 8 errors

F.mse_loss(pred, targ)              # averages pairwise errors (and warns about shape)
F.mse_loss(pred, targ.unsqueeze(1)) # averages the 8 aligned errors you meant
```

Backward on the wrong one returns a `(8, 1)` gradient for `pred`, exactly as a correct
run would. The model trains, the loss descends, and it is optimizing the wrong objective.

The built-in losses warn here (`Using a target size that is different to the input size...`), and that warning is worth treating as an error. Elementwise arithmetic gives
you nothing. The habit that catches it is the one from the top of this page: assert the
shape where it is produced.

```python
assert pred.shape == targ.shape, (pred.shape, targ.shape)
```

______________________________________________________________________

## Common confusions

- `(B,)` is aligned against the **last** axis of `(B, D)`, not the first. Add a singleton axis when the values belong to batch rows: `(B, 1)`.
- `expand` creates an aliased view; `repeat` creates independent storage. An elementwise operation may still allocate its own result after either kind of expansion.
- For `matmul`, only batch axes broadcast. Matrix inner dimensions must match exactly.
- A gradient with the expected shape does not prove the forward broadcast matched the intended meaning.

## Check your understanding

1. What shape results from `(4, 1, 3) + (2, 3)`? Which axis is reused?
1. Why does a `(3,)` bias used across 4 rows receive a gradient summed over those rows?
1. Why is a padding mask `(B, L)` ambiguous against scores `(B, h, L_q, L_k)`?

<details markdown="1"><summary>Answers</summary>

1. `(4, 2, 3)`. The `1` in the middle of the left operand is reused across the size-2 axis; the right operand is reused across the leading size-4 axis.
1. Each bias value contributes to four output positions, so the chain rule adds their four contributions back into that one parameter.
1. Right alignment puts `B` against `L_q`, not against the score tensor's batch axis. `(B, 1, 1, L_k)` expresses the intended axes.

</details>

**One-minute recap:** name axes, align from the right, treat singleton dimensions as reusable values, and expect backward to sum across axes that were reused.

______________________________________________________________________

## References

- NumPy, *Broadcasting*: numpy.org/doc/stable/user/basics.broadcasting.html (the
  right-to-left rule and the size-1 rule)
- PyTorch, *Broadcasting semantics*: docs.pytorch.org/docs/stable/notes/broadcasting.html
  (follows NumPy; `expand` is a zero-copy, stride-0 view)
- PyTorch, `torch.Tensor.expand`:
  docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html (size-1 axes only, `-1`
  to keep, new axes prepend)
- PyTorch, `torch.Tensor.repeat`:
  docs.pytorch.org/docs/stable/generated/torch.Tensor.repeat.html (tiles the tensor;
  contrast `torch.repeat_interleave`)
- PyTorch, `torch.matmul`: docs.pytorch.org/docs/stable/generated/torch.matmul.html (the
  batch-broadcast rule and the 1-D promotion rules)
- PyTorch, *Autograd mechanics*: docs.pytorch.org/docs/stable/notes/autograd.html (a
  gradient has the shape of its tensor)
- Python docs, *Tuples and Sequences*:
  docs.python.org/3/tutorial/datastructures.html#tuples-and-sequences (the one-element
  trailing-comma rule)
- PyTorch, `torch.Tensor.size` / `torch.Size`:
  docs.pytorch.org/docs/stable/generated/torch.Tensor.size.html
- NumPy, `numpy.sum`: numpy.org/doc/stable/reference/generated/numpy.sum.html (axis
  semantics)
- PyTorch, `torch.sum`: pytorch.org/docs/stable/generated/torch.sum.html (`dim`,
  `keepdim`)
- PyTorch, `torch.softmax`: pytorch.org/docs/stable/generated/torch.softmax.html
- einops: einops.rocks
- Vaswani et al., *"Attention Is All You Need"* (arXiv:1706.03762), §3.2 (attention shape
  conventions)
