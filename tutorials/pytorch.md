# PyTorch

The vocabulary every later page assumes: what a tensor carries, how shapes
combine, where a gradient comes from, how a module owns its parameters, and how
a `DataLoader` turns samples into batches. Read [Overview](#pytorch--overview)
first if the terms are new; otherwise jump to the section you need.

**On this page:**

- [PyTorch — Overview](#pytorch--overview)
- [Tensors](#tensors)
- [Shapes and Broadcasting — Q&A](#shapes-and-broadcasting--qa)
- [Gradients and Gradient Descent](#gradients-and-gradient-descent)
- [`nn.Module` and Multi-Layer Networks](#nnmodule-and-multi-layer-networks)
- [Where the Transpose Comes From](#where-the-transpose-comes-from)
- [ReLU and Dead Neurons](#relu-and-dead-neurons)
- [DataLoader — Batching and the Dataset Abstraction](#dataloader--batching-and-the-dataset-abstraction)
- [Random Streams and DataLoader Workers — Q&A](#random-streams-and-dataloader-workers--qa)
- [Inspecting Your Model — Best Practices](#inspecting-your-model--best-practices)

______________________________________________________________________

## PyTorch — Overview

**The picture to remember:** data enters as tensors; a module turns those tensors into predictions; a loss compares predictions with targets; autograd finds how each parameter affected the loss; an optimizer changes the parameters. The next batch repeats the cycle.

| If you are revising… | Start with |
|---|---|
| What a shape, device, or gradient flag means | [Tensors](#tensors) and [shapes and broadcasting](#shapes-and-broadcasting--qa) |
| Why a parameter receives a gradient | [Gradients and gradient descent](#gradients-and-gradient-descent) and [linear layer backward](#where-the-transpose-comes-from) |
| How layers become a model | [`nn.Module` and multi-layer networks](#nnmodule-and-multi-layer-networks) |
| Where batches and transforms happen | [DataLoader](#dataloader--batching-and-the-dataset-abstraction) and [random streams and workers](#random-streams-and-dataloader-workers--qa) |

This is a revision map. The examples in these pages illustrate ideas; the working code lives in the project.

______________________________________________________________________

### What PyTorch Is

Three things:

- **Tensors** — n-dimensional arrays (like `np.ndarray`) that can live on a GPU and participate in automatic differentiation
- **Autograd** — records operations involving tracked tensors into a graph so it can compute derivatives on demand
- **`nn`** — a class hierarchy for organizing model parameters and composing layers

In eager mode, PyTorch uses a **define-by-run** approach: the computation graph is built dynamically as operations execute. Normal Python control flow (`if`, `for`) works inside a model — debugging feels like ordinary Python. Compilation is an optional optimization.

For one batch, read the flow as `inputs → model → logits → loss → backward → optimizer step`. The graph belongs to that forward pass; parameters persist for the next one.

______________________________________________________________________

### The 5 Building Blocks

| What | PyTorch concept | Role in the cycle |
|---|---|---|
| Model | `nn.Module` subclass | owns parameters and maps inputs to predictions |
| Data | `Dataset` + `DataLoader` | supplies examples in batches |
| Loss | `nn.CrossEntropyLoss`, `nn.MSELoss`, etc. | turns prediction errors into a scalar |
| Optimizer | `torch.optim.Adam`, etc. | updates parameters from their gradients |
| Loop | plain Python `for` loop | repeats the cycle over batches |

______________________________________________________________________

### Where to Go Next

Every tutorial, and the section that answers a specific question, is listed in the [tutorials index](index.md).

### Check your understanding: PyTorch Overview

1. Which object persists between batches: the forward computation graph or the model's parameters?
1. Where does the optimizer get the information used to change parameters?
1. What does the loss compare?

<details markdown="1"><summary>Answers</summary>

1. Parameters persist. A new forward pass builds a new graph.
1. `backward()` writes gradients to the parameters; the optimizer reads those gradients.
1. The model's predictions and the target values.

</details>

**One-minute recap:** tensors carry values and shapes; modules own persistent state; autograd traces the current calculation; the loss and optimizer turn that trace into a parameter update.

______________________________________________________________________

### References: PyTorch Overview

- PyTorch autograd docs: pytorch.org/docs/stable/notes/autograd.html
- PyTorch data docs: pytorch.org/docs/stable/data.html
- Baydin et al., *"Automatic Differentiation in Machine Learning: a Survey"* (JMLR 2018)
- Bengio et al., *"Estimating or Propagating Gradients Through Stochastic Neurons"* (2013)
- Goodfellow, Bengio & Courville, *Deep Learning* (2016), Ch. 5–6

______________________________________________________________________

## Tensors

**Question to keep in mind:** what must you know about a tensor before combining it with another? Its **shape**, **dtype**, **device**, and whether autograd is tracking it.

Imagine a batch of 4 RGB images, each 28×28 pixels: its shape is `(4, 3, 28, 28)`. Indexing one image removes the batch slot and leaves `(3, 28, 28)`. Moving it to a GPU changes its device, not its shape. Setting `requires_grad` changes how later operations are tracked, not its values.

______________________________________________________________________

### Tensors vs NumPy

| | NumPy array | PyTorch tensor |
|---|---|---|
| Device | CPU only | CPU or GPU/accelerator |
| Autograd | none | tracks gradients via `requires_grad` |
| API | `np.*` | mostly mirrors NumPy |

Tensors are conceptually similar to NumPy ndarrays, with accelerator placement and optional gradient tracking. They are distinct objects, but the APIs overlap enough that NumPy experience transfers.

______________________________________________________________________

### Creating Tensors

```python
import torch

# from data
x = torch.tensor([1.0, 2.0, 3.0])

# from NumPy — shares memory on CPU, watch for aliasing bugs
import numpy as np
arr = np.array([1.0, 2.0])
t = torch.from_numpy(arr)
arr[0] = 99.0
print(t[0])   # tensor(99.) — same memory

# built-in constructors
torch.zeros(3, 4)
torch.randn(3, 4)    # standard normal
torch.arange(10)
```

______________________________________________________________________

### Device Placement

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(3, 4).to(device)
# or create directly on device:
x = torch.randn(3, 4, device=device)
```

**Pin dtype and device at boundaries.** Mixing `float32`/`float64` or CPU/GPU either raises or silently up/down-casts. Convert once when data enters your system:

```python
x = torch.as_tensor(arr, dtype=torch.float32, device=device)
```

______________________________________________________________________

### `requires_grad`

`requires_grad=True` tells autograd to track operations on this tensor and compute gradients during `.backward()`. You rarely set this manually — model parameters (`nn.Parameter`) have it set automatically.

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
x.grad   # tensor([4.])
```

See [Gradients and Gradient Descent](#gradients-and-gradient-descent) for how the graph works.

### Common confusions: Tensors

- A 1-D shape `(3,)` and a row shape `(1, 3)` hold three numbers but have different ranks; broadcasting may treat them differently. See [shapes and broadcasting](#shapes-and-broadcasting--qa).
- `.to(device)` returns a tensor on the requested device. Assign the result if later code should use it.
- `requires_grad=True` does not itself compute a gradient. A connected scalar result must be differentiated with `.backward()`.

### Check your understanding: Tensors

1. What is the shape of one image selected from `(4, 3, 28, 28)` with `images[0]`?
1. Why can `torch.from_numpy(arr)` reflect a later change to `arr`?
1. Does calling `.to("cuda")` change a tensor's number of dimensions?

<details markdown="1"><summary>Answers</summary>

1. `(3, 28, 28)`: indexing removed the leading batch dimension.
1. On the CPU, they can share the same underlying memory.
1. No. Device placement changes where values live, not the shape.

</details>

**One-minute recap:** shape describes indexing, dtype describes representation, device describes location, and autograd tracking describes whether operations can contribute to gradients.

______________________________________________________________________

## Shapes and Broadcasting — Q&A

**One rule to carry through the page:** align shapes from the right. Equal sizes match; a size-1 axis can be reused; missing axes act like size 1. When a value was reused in the forward pass, its gradient adds those uses together on the way back.

| If the question is… | Read |
|---|---|
| Why `(3,)` differs from `(1,3)` | [shape notation](#q--shape-3-what-does-this-mean) |
| Whether two shapes combine | [broadcastability](#q-what-does-broadcastable-mean) and [`dim=k`](#q-what-does-dimk-actually-mean) |
| Whether a 1-D tensor is a row or a column | [orientation](#q-is-torchtensor1-2-a-row-or-a-column) |
| Whether to join or to add an axis | [`cat` or `stack`](#q-cat-or-stack) |
| Whether expansion allocates | [`expand` versus `repeat`](#q-what-is-the-difference-between-expand-and-repeat) |
| Why attention masks need extra axes | [attention masks](#q-how-does-broadcasting-apply-to-attention-masks) |
| Why a bias gradient sums over examples | [broadcast gradients](#q-what-happens-to-gradients-when-a-tensor-is-broadcast) |
| How one `nn.Linear` serves every token | [shared weights](#q-how-does-nnlinear-apply-one-weight-to-every-token) |

The sections are independent Q&A references. For a quick revision, follow the shape notation, broadcastability, and broadcast gradients questions in that order.

______________________________________________________________________

### Q: `# shape (3,)`. What does this mean?

That notation mixes two separate things.

#### 1. The comma is Python syntax

`.shape` returns a tuple (`torch.Size` is a subclass of `tuple`). In Python, `(3)` is just
the number 3 in parentheses, which do nothing. A one-element tuple needs a
trailing comma:

```python
type((3))    # <class 'int'>    ← just a number
type((3,))   # <class 'tuple'>  ← a 1-element tuple
```

`(3,)` isn't "3 and then something omitted." The comma just tells Python it's a tuple.

#### 2. What the tuple says

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

#### 3. Intuition: how many indices to reach one value

```python
a = torch.tensor([1., 2., 3.])     # (3,)
a[0]                               # tensor(1.) — one index, done

b = torch.tensor([[1., 2., 3.]])   # (1, 3)
b[0]                               # tensor([1., 2., 3.]) — still a row!
b[0][0]                            # tensor(1.) — needed two
```

Both hold the same 3 numbers. They differ in how the numbers are addressed, and
broadcasting cares about exactly that.

#### 4. Why it matters: `(3,)`, `(1, 3)` and `(3, 1)` are different tensors

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

### Q: Is `torch.tensor([1, 2])` a row or a column?

Neither. Its shape is `(2,)`, which is 1-D. Rows and columns only exist once there are
two slots to name.

```python
a = torch.tensor([1, 2])
a.shape               # (2,)    1-D, no orientation
a.unsqueeze(0).shape  # (1, 2)  row vector: 1 row, 2 columns
a.unsqueeze(1).shape  # (2, 1)  column vector: 2 rows, 1 column
```

Intuition: a 1-D tensor is a ruler lying on a table. It has a length, but no up/down
versus left/right. Orientation appears only when you place it on a 2-D page. The
horizontal print is formatting, not a row.

#### Where the orientation gets decided

The op you pass it to picks one. `stack` picks by `dim`:

```python
torch.stack([a, b], dim=0)   # each vector becomes a row:    [[1, 2], [3, 4]]
torch.stack([a, b], dim=1)   # each vector becomes a column: [[1, 3], [2, 4]]
```

`matmul` picks whichever orientation makes the multiplication work, by position: a 1-D
left operand gets a `1` prepended (it becomes a row), a 1-D right operand gets a `1`
appended (it becomes a column).

```python
M = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
a = torch.tensor([1.0, 2.0])

a @ M   # a is a row (1, 2):    [1*1 + 2*3, 1*2 + 2*4] = [7., 10.]
M @ a   # a is a column (2, 1): [1*1 + 2*2, 3*1 + 4*2] = [5., 11.]
```

Same `a`, two orientations, and nothing in the tensor chose either one.

The confusion usually comes from linear algebra, where a vector is a column by default.
PyTorch and NumPy do not follow that convention. 1-D is orientation-free, and you
`unsqueeze` when the orientation matters.

______________________________________________________________________

### Q: What does "broadcastable" mean?

**Broadcastable = shapes that don't match but can be made to match by stretching size-1
dimensions.**

#### Step 1: You already use it

```python
x = torch.tensor([1., 2., 3.])   # (3,)
x + 10                           # tensor([11., 12., 13.])
```

Elementwise addition should need matching shapes. This works because the scalar `10` is
conceptually copied to every position. That's broadcasting.

#### Step 2: The rule (two lines)

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

#### Step 3: Why `1` is special

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

#### Step 4: Practice

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

### Q: What is the difference between `expand` and `repeat`?

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

#### What each one accepts

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

#### The two gotchas of the cheap one

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

#### Which to reach for

| You want | Use | Because |
|---|---|---|
| To line shapes up for an elementwise op | nothing | `+`, `*`, `where` broadcast already |
| A read-only stretch for an op that won't broadcast | `expand` | zero-copy |
| To write into the result, or hand it to `view` | `repeat` | you need real, independent elements |
| Each element duplicated in place (`[1,1,2,2]`) | `repeat_interleave` | `repeat` tiles, it doesn't interleave |

______________________________________________________________________

### Q: Does matmul broadcast the same way?

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

#### 1-D operands

A 1-D operand gets a dimension added for the duration and removed afterwards — prepended
on the left, appended on the right:

```python
torch.randn(3) @ torch.randn(3)          # ()      dot product
torch.randn(3) @ torch.randn(3, 5)       # (5,)    row vector times matrix
torch.randn(10, 3, 4) @ torch.randn(4)   # (10, 3) batched matrix times column
```

#### Where this repo uses it

[modules.py:140](../app/src/training/transformer/modules.py#L140) projects all heads
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

### Q: How does `nn.Linear` apply one weight to every token?

A language model's output head maps `(B, L, d_model)` to `(B, L, vocab)`. It holds one
weight `W` of shape `(vocab, d_model)` and one bias of shape `(vocab,)`, and every
position uses both:

```
out[b, l, :] = x[b, l, :] @ W.T + bias        for every b and every l
  (vocab,)      (d_model,)  (d_model, vocab)
```

**`nn.Linear` knows nothing about positions. Matmul and broadcasting do the work.** Its
`forward` is one line, `F.linear(input, self.weight, self.bias)`, which computes
`input @ weight.T + bias`. Two rules from this page do the rest:

- `@` treats every axis left of the matrix part as batch, and reuses a 2-D right operand
  across all of them ([matmul](#q-does-matmul-broadcast-the-same-way)):
  `(B, L, d_model) @ (d_model, vocab) → (B, L, vocab)`.
- `+ bias` aligns from the right, so `(vocab,)` acts as `(1, 1, vocab)` and is reused
  across `B` and `L`.

That is why the docs give the input shape as `(*, in_features)`: any number of leading
axes, and only the last one has to match.

**Two positions get different logits because their inputs differ, not their weights.**
By the time a token reaches the head, position encoding and attention have mixed context
into its `d_model` vector. The head is one function applied to `B × L` vectors. Sharing
it is also why sequence length doesn't change the parameter count, the same way a
convolution reuses one kernel at every pixel.

#### One product per token, one matmul per batch

Mathematically, each `(b, l)` gets its own vector × matrix product, and no product reads
another token's row. Computationally, the `B × L` products are not run one at a time.
Stack the rows and they are a single matrix × matrix product:

```
x.reshape(B*L, d_model)  @  W.T     →   (B*L, vocab)
 row 0     = token (0, 0)                 row 0     = logits for (0, 0)
 row 1     = token (0, 1)                 row 1     = logits for (0, 1)
 ...                                      ...
 row B*L−1 = token (B−1, L−1)             row B*L−1 = logits for (B−1, L−1)
```

Row `i` of the output depends only on row `i` of the input, so the result is identical
to doing them one by one:

```python
x = torch.randn(2, 3, 4)
W = torch.randn(5, 4)
b = torch.randn(5)

y = F.linear(x, W, b)                                              # (2, 3, 5)
torch.allclose(y, (x.reshape(-1, 4) @ W.T + b).reshape(2, 3, 5))   # True
```

Each output entry is one dot product of length `d_model`:
`out[b, l, v] = Σ_k x[b, l, k] · W[v, k] + bias[v]`. A GPU computes all `B × L × vocab`
of them as one large matmul, tiled across thousands of cores. That is where batching
pays: the arithmetic per token is unchanged, but it arrives as one big matmul instead of
`B × L` small ones.

#### Backward: one `W.grad` for every token

`W` was used at all `B × L` positions, so its gradient is the sum of `B × L`
contributions, the same rule as a [broadcast bias](#q-what-happens-to-gradients-when-a-tensor-is-broadcast).
There is one `W.grad` and one optimizer step per batch, not one per token. When the loss
averages over tokens, each contribution already carries the `1/N`, so that sum is an
average ([sequence outputs](#sequence-outputs-one-loss-per-token)).

LayerNorm splits the two cases: `γ` and `β` are shared across positions like `W`, but `μ`
and `σ` are computed fresh for each token. See
[Which axes: `normalized_shape`](transformer.md#which-axes-normalized_shape).

______________________________________________________________________

### Q: How does broadcasting apply to attention masks?

See [Attention Masks](transformer.md#attention-masks-and-cross-attention--qa) for the full mask story. On
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

#### The gotcha this protects you from

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

### Q: What does `dim=k` actually mean?

#### The one rule

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

#### Reductions: the index that vanishes

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

#### Softmax: the axis you're choosing among

`attn_scores` is `(B, h, L_q, L_k)`. `dim=-1` is `L_k`. Softmax normalizes rather than
reduces, so the shape is unchanged, but now `scores.softmax(dim=-1).sum(dim=-1)` is all
ones.

Meaning: each **query** spreads one unit of attention mass across all **keys**. `dim=-2`
would make each key's column sum to 1, i.e. attending over queries. That's the wrong
direction, and it won't raise, so nothing tells you.

Heuristic: softmax goes over the axis whose entries are the *candidates you're choosing
between*.

#### Insertion and joining

Here `dim` means **where the new slot lands in the output**:

```python
torch.stack([a, b, c], dim=1)   # 3 tensors of (B, D) -> (B, 3, D)
torch.stack([a, b, c], dim=0)   # -> (3, B, D)
torch.cat([a, b], dim=-1)       # (B, D1) + (B, D2) -> (B, D1+D2)
```

`stack` needs identical shapes and adds a dimension. `cat` needs the same number of
dimensions and matching sizes in every slot except `k`. The next question works through
what each one does to the values.

#### Shape ops

`transpose(k1, k2)` swaps two slots. `permute(...)` reorders all of them.
`reshape`/`view` ignore slots entirely and just re-window the flat memory buffer. That's
why the canonical multi-head split reads the way it does:

```python
x.view(B, L, h, D // h).transpose(1, 2)   # (B, L, D) -> (B, h, L, D_head)
```

You cannot do that in one reshape to `(B, h, L, D_head)`. `view` walks memory in order
and doesn't know your slots mean anything, so it would interleave heads with tokens.

#### Habits that end guess-and-check

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

#### Not covered here

`gather` and `index_select` are the harder tier. For `gather`, the `dim` argument stops
meaning "slot that disappears" and starts meaning "slot whose indices you're replacing."

______________________________________________________________________

### Q: `cat` or `stack`?

**`cat` joins along an existing slot, so `ndim` stays the same. `stack` creates a new
slot, so `ndim` goes up by one.**

`cat` is taping sheets of paper end to end: one longer sheet, still 2-D. `stack` is
piling sheets into a deck: a new "which sheet" axis, so 2-D becomes 3-D.

```python
a, b = torch.tensor([1, 2]), torch.tensor([3, 4])

torch.cat([a, b])            # [1, 2, 3, 4]       shape (4,)
torch.stack([a, b])          # [[1, 2], [3, 4]]   shape (2, 2)
torch.stack([a, b], dim=1)   # [[1, 3], [2, 4]]   shape (2, 2), pairs elements like zip
```

#### The shapes, for four tensors of the same shape

```python
xs = [torch.randn(2, 3) for _ in range(4)]

torch.cat(xs, 0).shape     # (8, 3)      slot 0 grows: 2 + 2 + 2 + 2
torch.cat(xs, 1).shape     # (2, 12)     slot 1 grows: 3 + 3 + 3 + 3
torch.stack(xs, 0).shape   # (4, 2, 3)   new slot of size 4 inserted at position 0
torch.stack(xs, 1).shape   # (2, 4, 3)
torch.stack(xs, 2).shape   # (2, 3, 4)
```

![Four 2x3 tensors combined six ways: torch.cat along dim 0 piles their rows into an
8x3 block and along dim 1 lines their columns up into 2x12, while torch.stack inserts a
size-4 axis at dim 0, 1, or 2, giving (4, 2, 3), (2, 4, 3), and (2, 3,
4).](assets/cat_vs_stack.png)

How to read the figure: number and color say which `xs[k]` a cell came from, hatching
marks row 1 of that tensor and plain marks row 0, and the 3-D results are drawn as
slices along dim 0 (`out[0]`, `out[1]`, …).

#### Why the slices look like that

`cat` adds no index. Only the joined slot grows, and every other slot must already
match. `stack(xs, d)` adds an index of size 4 at position `d`, and that index is `k`,
"which tensor":

| `d` | Subscripts | Each slice of the result |
|---|---|---|
| 0 | `out[k, i, j] = xs[k][i, j]` | slice `k` is tensor `k`, whole |
| 1 | `out[i, k, j] = xs[k][i, j]` | slice `i` is row `i` of all four tensors, as rows |
| 2 | `out[i, j, k] = xs[k][i, j]` | same rows, but each tensor's row becomes a column |

The hatching is the proof for `d=1`: `out[0]` is all plain and `out[1]` is all hatched,
because each slice collects one row index across the four tensors. For `d=2`,
`stack(xs, 2)[1, 2]` is `[xs[0][1, 2], xs[1][1, 2], xs[2][1, 2], xs[3][1, 2]]` — the same
position in four tensors grouped together, the way a channels-last image groups four
channels.

The shortcut that collapses both ops into one: insert a size-1 slot, then glue along it.

```python
for d in range(3):
    assert torch.equal(torch.stack(xs, d), torch.cat([x.unsqueeze(d) for x in xs], d))
```

#### What each one demands

`cat` wants matching sizes in every slot except the joined one, so `(2, 3)` and `(5, 3)`
concatenate along dim 0 into `(7, 3)`. `stack` wants identical shapes, so that same pair
raises `stack expects each tensor to be equal size`. Both allocate a new tensor; neither
returns a view.

#### Which one you want

- **`stack` when each tensor is one item** and you want a collection axis. The
  DataLoader's `default_collate` does exactly this, turning N tensors of `(C, H, W)` into
  `(N, C, H, W)`. See [DataLoader](#dataloader--batching-and-the-dataset-abstraction).
- **`cat` when each tensor is already a chunk** of the same thing: merging batches along
  dim 0, joining U-Net skip connections along the channel axis, or combining a
  bidirectional RNN's forward and backward hidden states.

______________________________________________________________________

### Q: How does masked mean pooling work?

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

#### Why `unsqueeze(-1)` and not `unsqueeze(1)`

`mask.unsqueeze(1)` is `(B, 1, L)`. Right-aligned against `(B, L, D_MODEL)`, that puts
`L` against `D_MODEL`. You need `(B, L, 1)` so the mask lines up on `B` and `L` and
stretches across the feature slot.

#### What `sum(dim=1)` does, by hand

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

#### How `clamp(min=1)` prevents NaN

`clamp(min=1)` raises anything below 1 up to 1 and leaves everything else alone:

```python
torch.tensor([0, 1, 2, 5]).clamp(min=1)   # tensor([1, 1, 2, 5])
```

Counts are integers, so the only value it can change is `0`, which happens when a row is
entirely padding. Then `summed` is all zeros and the division is `0 / 0 = NaN`. NaN
spreads: every later layer, the loss, and the gradients go NaN, and training silently
breaks. With the clamp it's `0 / 1 = 0`, a zero vector. Rows with at least one real token
are unchanged.

#### Why `keepdim=True` matters here

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

### Q: What happens to gradients when a tensor is broadcast?

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

#### Why a bias grad grows with the batch

That is the rule behind one of the most-asked training questions. A bias is broadcast over
the batch, so its gradient is the *sum* over the batch, not the mean. Double the batch
size and the raw bias gradient roughly doubles.

Losses often offset this by reducing with `mean`, which divides by the number of examples in the ordinary unweighted case — the
default for common losses such as `MSELoss` and `CrossEntropyLoss`. Switch a loss to `reduction="sum"` and
every gradient in the model scales with batch size, so the learning rate that worked at
batch 32 blows up at batch 256. See
[The Training Loop › Aggregating Loss](training.md#aggregating-loss-correctly-across-an-epoch).

#### The bug this makes silent

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

### Common confusions: Shapes and Broadcasting

- `(B,)` is aligned against the **last** axis of `(B, D)`, not the first. Add a singleton axis when the values belong to batch rows: `(B, 1)`.
- A 1-D tensor has no orientation. The op it is passed to decides whether it acts as a row or a column.
- `cat` keeps `ndim` and grows one slot; `stack` raises `ndim` by one. Reaching for `cat` when the inputs are single items silently builds a long sequence instead of a batch.
- `expand` creates an aliased view; `repeat` creates independent storage. An elementwise operation may still allocate its own result after either kind of expansion.
- For `matmul`, only batch axes broadcast. Matrix inner dimensions must match exactly.
- A gradient with the expected shape does not prove the forward broadcast matched the intended meaning.

### Check your understanding: Shapes and Broadcasting

1. What shape results from `(4, 1, 3) + (2, 3)`? Which axis is reused?
1. Why does a `(3,)` bias used across 4 rows receive a gradient summed over those rows?
1. Why is a padding mask `(B, L)` ambiguous against scores `(B, h, L_q, L_k)`?
1. You have 32 tensors of shape `(3, 224, 224)`, one per image. Which op builds the batch, and what does the other one give you?

<details markdown="1"><summary>Answers</summary>

1. `(4, 2, 3)`. The `1` in the middle of the left operand is reused across the size-2 axis; the right operand is reused across the leading size-4 axis.
1. Each bias value contributes to four output positions, so the chain rule adds their four contributions back into that one parameter.
1. Right alignment puts `B` against `L_q`, not against the score tensor's batch axis. `(B, 1, 1, L_k)` expresses the intended axes.
1. `torch.stack(images, dim=0)` gives `(32, 3, 224, 224)`, a new batch axis. `torch.cat(images, dim=0)` gives `(96, 224, 224)`, which stacks the channels of all 32 images into one image-shaped tensor and loses the sample boundary.

</details>

**One-minute recap:** name axes, align from the right, treat singleton dimensions as reusable values, and expect backward to sum across axes that were reused.

______________________________________________________________________

### References: Shapes and Broadcasting

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
- PyTorch, `torch.cat`: docs.pytorch.org/docs/stable/generated/torch.cat.html (joins along
  an existing dimension)
- PyTorch, `torch.stack`: docs.pytorch.org/docs/stable/generated/torch.stack.html
  (concatenates along a new dimension; all inputs must have the same shape)
- PyTorch, `torch.unsqueeze`:
  docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html
- PyTorch, `default_collate`:
  docs.pytorch.org/docs/stable/data.html#torch.utils.data.default_collate (builds a batch
  with `torch.stack`)
- einops: einops.rocks
- Vaswani et al., *"Attention Is All You Need"* (arXiv:1706.03762), §3.2 (attention shape
  conventions)

______________________________________________________________________

## Gradients and Gradient Descent

**The question:** how can a single loss tell each weight which way to move? Every forward operation records a local connection. Backward follows those connections in reverse, multiplying local derivatives and adding contributions wherever paths meet.

For a quick revision, read the [chain rule trace](#chain-rule-trace) first; then use the sections on leaves and differentiability to explain where gradients appear and when they stop carrying useful information.

______________________________________________________________________

### The Computation Graph

When you create a tensor with `requires_grad=True`, PyTorch tracks all operations on it. Each resulting tensor gets a `grad_fn` attribute referencing the operation that created it. These link together into an acyclic graph encoding the full computation history.

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2        # y.grad_fn = PowBackward0
y.backward()      # walk the graph backward; fill .grad on leaves
x.grad            # → tensor([4.])
```

Calling `.backward()` walks the graph in reverse and fills each **leaf** tensor's `.grad`.

______________________________________________________________________

### Why `.grad` Lands on the Leaf, Not the Output

`.backward()` computes the derivative *of the output* with respect to the *leaves*. The output is the numerator; the leaves are the denominators. The result gets stored on the **leaf** — the thing you differentiate against.

In real training: you call `loss.backward()`. `loss` is the output (numerator). Model parameters are the leaves (denominators). Those parameters need to know how to change — so that's where the gradient lands.

```python
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
y = a * b
y.backward()
print(a.grad)   # 3.0  — dy/da = b
print(b.grad)   # 2.0  — dy/db = a
# y.grad is None — output nodes don't store gradients by default
```

______________________________________________________________________

### Why Centered Inputs Stabilize Gradients

Take one neuron: `z = w·x + b`. The gradient of the loss with respect to weight `w` is:

```
∂L/∂w = ∂L/∂z · x
                 ↑
         the input value itself appears here
```

**The input directly scales the gradient.**

**Intuition:** for one example with positive inputs, every incoming weight gradient has the sign of `∂L/∂z`. Across examples, the sum can still have different signs, but strongly correlated input features can make weight updates poorly conditioned and cause a zig-zag path:

```
Uncentered inputs (all positive):     Centered inputs (mix of ±):
weights forced to move together        weights move independently
→ zig-zag path to minimum              → direct path

    \  /\                                   \
     \/  \                                   \
          \                                   \
           → slow                              → fast
```

Centering inputs around 0 allows positive and negative contributions to each weight gradient. It often reduces correlations and makes descent better conditioned; it does not guarantee a straight path or independent updates.

**Second effect — scale:** if `x` is huge, `∂L/∂w` is huge → giant weight jumps → instability. Dividing by std keeps gradient magnitudes in a sane range. This is the same reasoning behind BatchNorm. (LeCun et al., *Efficient BackProp*, 1998.)

______________________________________________________________________

### Chain Rule Trace

Autograd is the chain rule, automated. For `L = f(g(h(w)))`:

∂L/∂w = (∂L/∂f) · (∂f/∂g) · (∂g/∂h) · (∂h/∂w)

Each factor is **local** — it only depends on one operation. Autograd computes each independently, then multiplies them walking backward. The loss starts the relay by handing back `1.0` (since `∂L/∂L = 1`).

**Concrete trace** — `L = (w·x − y)²` with `x=3, y=10, w=2`:

```
Forward:
a = w * x      = 6
b = a - y      = -4
L = b ** 2     = 16

Backward:
∂L/∂L = 1
∂L/∂b = 2b          = -8
∂L/∂a = ∂L/∂b · 1   = -8
∂L/∂w = ∂L/∂a · x   = -24
```

`w.grad` → `-24`. Swap in cross-entropy and only the final node's local rule changes.

______________________________________________________________________

### Leaf vs Non-Leaf, `retain_grad`, and `no_grad`

**Leaf node:** a tensor you created directly, not produced by an operation. Model parameters are leaves — they sit at the edges of the computation graph.

**Non-leaf node:** any tensor that is the *result* of an operation. Intermediate activations and the loss itself are non-leaf.

```python
w = torch.tensor([1.0], requires_grad=True)  # leaf
x = torch.tensor([2.0])                       # leaf (requires_grad=False)
z = w * x                                      # non-leaf — result of multiply
loss = z.sum()                                 # non-leaf
```

`loss.backward()` walks the graph backward and **accumulates into `.grad` of leaf tensors only:**

```python
loss.backward()
print(w.grad)   # populated  — w is a leaf
print(z.grad)   # None       — z is non-leaf, grad discarded by default
```

Non-leaf grads are computed transiently during backprop (to propagate the chain rule) then discarded — memory savings. Add `z.retain_grad()` before `backward()` if you actually need a non-leaf's grad.

**`torch.no_grad()`:** disables graph recording inside its context. Results have `requires_grad=False`:

```python
with torch.no_grad():
    z = w * x
    print(z.requires_grad)   # False — no graph built
```

Two reasons to use at eval time: (1) no graph = less memory, (2) no graph-building overhead = faster. You're not training, so you never need the backward pass.

______________________________________________________________________

### Why Any Differentiable Loss Works

The loss isn't special to autograd — it's just the **final node** in the graph, the one that outputs a scalar. Autograd walks backward from any scalar through whatever operations produced it.

The real constraint: every operation from input to loss must have a known local derivative. PyTorch ships a derivative rule for each primitive op (`+`, `*`, `matmul`, `relu`, `exp`, `log`, ...). As long as your loss is built out of these, autograd handles it.

______________________________________________________________________

### Differentiability — Three Cases

#### Case 1: Differentiable everywhere

`x²`, `exp`, `sin`, `matmul` — smooth, single-valued derivative at every point. No issue.

#### Case 2: Differentiable almost everywhere (kinks)

`ReLU`, `abs` — undefined at exactly one point.

| Input | ReLU derivative |
|---|---:|
| x > 0 | 1 |
| x < 0 | 0 |
| x = 0 | undefined mathematically; PyTorch chooses 0 |

PyTorch resolves the tie by convention (returns `0` for `relu'(0)`). This is a **subgradient** — legitimate for convex kinks. In practice it never matters since you almost never land exactly on `x=0` in floating point.

```python
x = torch.tensor([0.0], requires_grad=True)
torch.relu(x).backward()
x.grad   # tensor([0.])  ← chosen subgradient
```

#### Case 3: Zero-gradient / non-differentiable — breaks training

`argmax`, `round`, `floor`, sampling.

These do not provide a useful path for ordinary backpropagation. `floor` and `round` have zero derivatives almost everywhere in autograd. `argmax` returns integer indices with no gradient path, so trying to backpropagate through the indices raises rather than silently returning zeros. Sampling a discrete choice similarly needs a special estimator if its choice affects the loss.

```python
x = torch.tensor([2.7], requires_grad=True)
y = torch.floor(x)
y.backward()
x.grad   # tensor([0.])  ← defined but useless
```

**Why this matters for Transformers:** picking a token with `argmax(logits)` does not carry a gradient to the logits. Cross-entropy instead operates on the full logit vector, so gradient flows back into the scores. Token selection is usually deferred to generation or inference.

When you need to backprop through a discrete choice: **straight-through estimator** (pretend the non-differentiable op was identity on the backward pass) and **Gumbel-softmax** (smooth, temperature-controlled approximation) exist for this.

### Common confusions: Gradients and Gradient Descent

- A gradient is a **slope at the current values**, not an update. The optimizer uses it to choose an update.
- Gradients **accumulate** in leaf `.grad` fields. A second backward pass adds to an existing gradient unless it is cleared.
- `torch.no_grad()` stops recording new operations; it does not change the `requires_grad` setting of an existing parameter.

### Check your understanding: Gradients and Gradient Descent

1. In `L = (w·x − y)²` with `w=2`, `x=3`, `y=10`, why is `∂L/∂w = −24`?
1. After `loss.backward()`, which tensor normally stores a parameter's derivative?
1. Why can cross-entropy train a classifier while `argmax(logits)` cannot carry its training signal?

<details markdown="1"><summary>Answers</summary>

1. The error is `−4`; squaring contributes `2(−4)=−8`, and multiplying by `x=3` gives `−24`.
1. The leaf parameter's `.grad` field.
1. Cross-entropy uses the continuous scores; `argmax` produces a discrete index with no gradient path.

</details>

**One-minute recap:** a forward pass builds a graph; backward applies local chain rules; leaf gradients accumulate; the optimizer makes the actual update.

______________________________________________________________________

## `nn.Module` and Multi-Layer Networks

**The mental model:** a module is a persistent tree of registered state and a callable description of data flow. With gradient recording enabled, each call creates a fresh autograd graph. Keeping those two lifetimes separate explains parameter registration, repeated calls, `state_dict()`, and why gradients accumulate.

**Reading route:** start with [parameters, buffers, and state](#parameters-buffers-and-state), then [what the graph saves](#the-module-doesnt-save-forward-results--the-graph-does). The sections on `Sequential`, hooks, and loss modules answer specific interface questions once that distinction is clear.

| At this point | What exists |
|---|---|
| After `__init__` | Registered parameters, buffers, and child modules |
| After `model(x)` | Those same objects plus this call's graph when gradients are recorded |
| After `loss.backward()` | Gradients accumulated on leaf parameters |
| After `optimizer.step()` | Updated parameter values for the next call |

______________________________________________________________________

### What `nn.Module` Is

`nn.Module` is the base class every layer and every model subclasses. You subclass it, fill `__init__` with layers, and write `forward()`. The backward pass is defined automatically by autograd.

**SWE analogy:** an abstract base class with inversion-of-control and lifecycle hooks — closest to a React `Component` or a Spring `@Component`. Like those, it:

- auto-discovers registered fields (parameters instead of props/beans)
- has mode switches: `train()` / `eval()` change behavior of Dropout and BatchNorm
- moves everything to device: `model.to("cuda")`
- serializes state: `state_dict()` / `load_state_dict()`

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()   # MUST call first
        # assign layers here

    def forward(self, x):
        # data flow — autograd builds the graph as this runs
```

______________________________________________________________________

### `__setattr__` Interception

`nn.Module` overrides Python's `__setattr__`. Every `self.x = y` in `__init__` is intercepted and classified by type:

- `nn.Parameter` → registered in `self._parameters`
- `nn.Module` (a sub-module) → registered in `self._modules`
- Anything else → stored as an ordinary attribute, **invisible to the framework**

The "magic": assigning `self.fc1 = nn.Linear(8, 16)` automatically registers that layer's weights and biases as parameters. No explicit registration call needed.

**The silent bug:**

```python
class Buggy(nn.Module):
    def __init__(self):
        super().__init__()
        self.good = nn.Parameter(torch.randn(3))        # registered ✓
        self.bad  = torch.randn(3, requires_grad=True)  # plain tensor ✗

print([n for n, _ in Buggy().named_parameters()])  # ['good']  ← 'bad' missing
```

If `bad` is a plain tensor:

1. **It never trains** — `model.parameters()` omits it, optimizer never updates it
1. **It stays on CPU** — `model.to("cuda")` won't move it → cryptic device mismatch crash in `forward()`
1. **It isn't saved** — `state_dict()` omits it, reload silently loses it

All three fail silently at definition time.

______________________________________________________________________

### Parameters, Buffers, and State

**Parameters** are tensors the model *learns*. Wrap in `nn.Parameter` (`requires_grad=True` by default). Sub-module parameters (e.g. `nn.Linear`'s `weight` and `bias`) are auto-registered when you assign the sub-module.

**Buffers** are persistent state the model *doesn't learn* — must be saved and moved to GPU, but not trained. Register with `self.register_buffer("name", tensor)`.

**Tensor state** = parameters + persistent buffers. `state_dict()` saves those tensors; the model class and its configuration must also be available to reconstruct the model.

```
state  (everything in state_dict — saved & device-moved)
├── parameters   → nn.Parameter, requires_grad=True, optimizer trains these
└── buffers      → register_buffer(), not trained but persistent
```

BatchNorm is the canonical example: `weight`/`bias` are parameters (learned), `running_mean`/`running_var` are buffers (updated during forward passes, not by the optimizer).

```python
bn = nn.BatchNorm1d(4)
print([n for n, _ in bn.named_parameters()])  # ['weight', 'bias']
print([n for n, _ in bn.named_buffers()])      # ['running_mean', 'running_var', ...]
print(list(bn.state_dict().keys()))            # all of the above
```

**Decision rule:**

- Should the optimizer learn it? → `nn.Parameter`
- Persistent, must save/move to GPU, but *not* learned? → `register_buffer`
- Throwaway intermediate in `forward`? → local variable, don't store on `self`

______________________________________________________________________

### A Module Is a Tree Node, Not a Layer

A layer is a Module, but so is the whole model. `nn.Linear`, `nn.Sequential`, and a ResNet are the same kind of object, and they nest through `_modules`. Each node owns its own `_parameters`, `_buffers`, and child `_modules`.

`.parameters()`, `.to(device)`, and `state_dict()` are just recursive walks over that tree. The registry is the whole point of the class.

______________________________________________________________________

### The Module Doesn't Save Forward Results — the Graph Does

Backward needs intermediate values from forward, such as the input to a matmul. A natural guess is that the module stores them. It doesn't. **Autograd attaches them to the output tensor.** Each op's output carries a `grad_fn`, and that backward node holds only what its gradient formula needs:

| Op | Backward node | Saves |
|---|---|---|
| `nn.Linear` | `AddmmBackward0` | both operands (`mat1` = input, `mat2` = weight) |
| `a @ b` | `MmBackward0` | both operands |
| `relu` | `ReluBackward0` | its output (`result`) |
| `a + b` | `AddBackward0` | no tensors |

```python
import torch, torch.nn as nn

lin = nn.Linear(3, 2)
x = torch.randn(1, 3)
y1, y2 = lin(x), lin(x)          # two independent graphs; lin recalls neither

print(y1.grad_fn)                                        # AddmmBackward0
print(y1.grad_fn is y2.grad_fn)                          # False
print(y1.grad_fn._saved_mat1.data_ptr() == x.data_ptr()) # True: graph holds the input

y1.sum().backward()
print(lin.weight.grad.shape)     # torch.Size([2, 3]): grads land on the parameter
```

Three consequences show that this state lives outside the module:

1. **Calling one module twice builds two independent graphs.** That's what makes RNN unrolling and weight tying work.
1. **`retain_graph=True` exists** because `backward()` frees the *graph's* saved tensors. A second `backward()` through the same graph raises `Trying to backward through the graph a second time`. The module is untouched.
1. **`del loss` frees activation memory while the model stays alive**, as long as nothing else holds the graph (a kept `logits` tensor does).

Some state does persist on the module: gradients. `backward()` accumulates into `param.grad`, which is why you call `optimizer.zero_grad()`.

**The split:**

```
Module  → persistent state: parameters, buffers, .grad   (optimizer reads this)
Graph   → per-call state:   saved activations            (backward() reads this)
```

This is why activation memory grows with batch size and depth, and parameter memory doesn't.

Sources: [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), [nn.Parameter](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html), [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html).

______________________________________________________________________

### Why You Need Nonlinearity

A single `nn.Linear` computes `y = xW^T + b` for row-shaped batches — an affine map. Writing one example as a column vector, two layers without an activation give:

W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)

It collapses to *one* linear layer. **Depth buys nothing without nonlinearity.**

**The fix:** activation functions between layers:

h = σ(W₁x + b₁), then y = W₂h + b₂.

The `σ` bends the space — the composition can't be flattened. Stacking nonlinear warps lets you carve arbitrarily complex decision boundaries.

______________________________________________________________________

### The Two Idioms

#### `nn.Sequential` — straight-line data flow

```python
self.net = nn.Sequential(
    nn.Linear(in_dim, hidden),
    nn.ReLU(),
    nn.Linear(hidden, out_dim),
)
# forward: return self.net(x)
```

PyTorch calls each layer's output as the next layer's input. You never write `forward()` — Sequential owns it. Think of it as a Unix pipe: `input | layer1 | layer2 | output`.

#### Explicit `forward()` — you control the data flow

```python
def forward(self, x):
    h = torch.relu(self.fc1(x))
    return self.fc2(h)
```

Identical in result to the Sequential above — but *you're writing the wiring*. This matters the moment your graph isn't a straight line.

#### Why Transformers Can't Use Sequential

The residual connection `x + sublayer(x)` requires **two references to `x`** — the original, and the transformed version. Sequential has already thrown `x` away by the time `sublayer(x)` is done.

```python
def forward(self, x):
    return x + self.attention(x)  # Sequential cannot express this
```

Sequential only knows: *"pass output of step N as input to step N+1."* It can't say *"also remember step N's input and add it back later."*

#### Rule of thumb

- Toy MLP, simple classifier → `Sequential` is fine
- Skip connections, multiple inputs/outputs, branching → explicit `forward()`
- Transformers → always explicit `forward()`

Key insight: **the training loop is identical in both cases**. The idiom only affects model definition.

______________________________________________________________________

#### Side-by-side

**Shared setup:**

```python
import torch
import torch.nn as nn

X = torch.randn(100, 8)
y = torch.randint(0, 2, (100,)).float()
criterion = nn.BCEWithLogitsLoss()
```

**Idiom A — `nn.Sequential`:**

```python
model = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    logits = model(X).squeeze()
    loss = criterion(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Idiom B — Explicit `forward()`:**

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    logits = model(X).squeeze()
    loss = criterion(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**What's actually the same:**

| | Sequential | Explicit |
|---|---|---|
| `model(X)` call | ✅ | ✅ |
| `loss.backward()` | ✅ | ✅ |
| `optimizer.step()` | ✅ | ✅ |
| `model.parameters()` | ✅ | ✅ |

**The only real difference:** in Idiom B you can write skip connections:

```python
def forward(self, x):
    h = torch.relu(self.fc1(x))
    return x[:, :1] + self.fc2(h)  # impossible in Sequential
```

______________________________________________________________________

### `model(x)` vs `model.forward(x)` — Never Call `forward` Directly

`self.net(x)` goes through `__call__`, which wraps `forward` with hook machinery and other module call behavior:

```python
# Pseudo-internals of nn.Module.__call__
def __call__(self, *args):
    for hook in self._forward_pre_hooks: args = hook(self, args) or args
    out = self.forward(*args)
    for hook in self._forward_hooks: out = hook(self, args, out) or out
    return out
```

Calling `model.forward(x)` directly bypasses module hooks. For example, a forward hook registered for shape inspection will fire for `model(x)` but not for `model.forward(x)`.

```python
with torch.no_grad():
    out = model(x)         # no graph built
    direct = model.forward(x) # also no graph: no_grad is a thread-local context
```

`no_grad` is a context setting, so it applies to either call. The practical reason to use `model(x)` is that it respects module call machinery, including hooks and framework wrappers that depend on `__call__`.

**One line:** always use `model(x)`, never `model.forward(x)`.

______________________________________________________________________

### Why Loss Functions Are `nn.Module`

**Short answer:** consistency and composability.

```python
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(pred, yb)   # calls loss_fn.forward(pred, yb) under the hood
```

Some loss functions have **persistent state**, such as fixed class weights:

```python
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]))
loss_fn.to(device)   # moves the weight tensor to GPU — works because it's a Module
```

A plain function can't do `.to(device)`. And because it's a Module, you can **embed it inside another model:**

```python
class ModelWithLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(8, 2)             # two class logits
        self.criterion = nn.CrossEntropyLoss()  # nested — fully tracked

    def forward(self, x, y):
        return self.criterion(self.net(x), y)
```

This pattern is common in HuggingFace and multi-GPU code (see [The Training Loop](training.md#why-embed-loss-inside-the-model)).

**One line:** `nn.Module` is PyTorch's universal interface for "anything with a forward pass" — loss functions qualify, so they inherit `.to(device)`, composability, and hook support for free.

______________________________________________________________________

### Shape Discipline

Every early bug will be a shape mismatch. Track dimensions explicitly:

```
x:    (B, in_dim)
W1:   (hidden, in_dim)   → fc1(x): (B, hidden)
ReLU: (B, hidden)        → shape unchanged (elementwise)
W2:   (out_dim, hidden)  → fc2:    (B, out_dim)
```

The batch dim `B` rides along untouched — layers operate on the last dim. Transformer tensors flow the same way, just with more dims: `(B, seq_len, d_model)`.

### Common confusions: nn.Module

- A plain tensor with `requires_grad=True` can receive a gradient but is not automatically a registered parameter. An optimizer built from `model.parameters()` will miss it.
- A buffer moves and saves with the module, but the optimizer does not learn it. BatchNorm running statistics are the standard example.
- `model.eval()` changes mode-sensitive modules such as Dropout and BatchNorm; it does not itself turn off autograd. Use `torch.no_grad()` or inference mode when gradients are unnecessary.

### Check your understanding: nn.Module

1. Why can one `nn.Linear` be called twice and still produce two independent forward graphs?
1. What happens if `self.weight` is a plain `torch.Tensor(requires_grad=True)` rather than `nn.Parameter`?
1. Why do two linear layers with no activation collapse into one affine transformation?

<details markdown="1"><summary>Answers</summary>

1. The module stores weights, while each call's output graph stores that call's operation history and saved values.
1. It is not registered in `model.parameters()` or `state_dict()`, and `model.to(device)` will not move it.
1. Composing affine maps gives another affine map: weights multiply and biases combine.

</details>

**One-minute recap:** modules own persistent registered state; graph nodes hold per-call backward information; nonlinear activations make stacked layers more expressive than one affine layer.

______________________________________________________________________

## Where the Transpose Comes From

**Remember the cause, not just the formula:** one weight is reused across every example, so its gradient sums over the batch. A transpose puts that batch index in the inner slot where matrix multiplication can perform the sum.

Read the [concrete numbers](#concrete-numbers) once, then use the [shape shortcut](#the-shortcut-everyone-actually-uses) for revision.

Take a linear layer `Y = X W` with a batch of inputs `X` (`n×d`), weights `W` (`d×k`), and output `Y` (`n×k`). Write `G = ∂L/∂Y` (`n×k`) for the gradient arriving from above. Backprop gives:

```
∂L/∂W = Xᵀ G
∂L/∂X = G Wᵀ
```

Why the transposes?

______________________________________________________________________

### One-Line Answer

The transpose isn't a rule you memorize. It's what happens when **the index you need to sum over is sitting in the wrong slot** for matrix multiplication.

Matmul has exactly one behavior: it sums over the *inner* index (columns of the left operand, rows of the right). So if your derivative sums over some index, you have to flip whichever matrix isn't storing that index in the inner position. That flip is the transpose.

______________________________________________________________________

### Where the Sum Comes From

Forward pass, written per-element:

```
Y[i,j] = Σ_a X[i,a] · W[a,j]
```

Now ask: **how many places does a single weight `W[a,j]` show up?** It's used by *every row* of the batch. All `n` examples multiply by the same weight, so its gradient collects a contribution from each one:

```
∂L/∂W[a,j] = Σ_i G[i,j] · X[i,a]        ← sum over i, the batch index
```

That sum runs over `i`. For a matmul to produce it, `i` must be the column index of the left operand. But in `X`, `i` is the **row** index. Flip it to `Xᵀ[a,i]`, and now:

```
Σ_i Xᵀ[a,i] · G[i,j]  =  (Xᵀ G)[a,j]  ✓
```

**Intuition:** the forward pass reads `X` *row by row* (one row = one example). The weight gradient needs `X` *column by column* (one column = one input feature, gathered across all examples). The transpose switches from the "per-example" view to the "per-feature" view.

______________________________________________________________________

### Concrete Numbers

```
X = [[1, 2],     W = [[10],     G = ∂L/∂Y = [[1],
     [3, 4]]          [20]]                  [2]]
n=2, d=2         d=2, k=1       n=2, k=1
```

Expand the forward pass by hand:

```
Y[0] = 1·w₀ + 2·w₁
Y[1] = 3·w₀ + 4·w₁
```

`w₀` appears in both rows, so the chain rule sums both:

```
∂L/∂w₀ = g₀·1 + g₁·3 = 1·1 + 2·3 = 7      ← 1 and 3 are X's first COLUMN
∂L/∂w₁ = g₀·2 + g₁·4 = 1·2 + 2·4 = 10     ← 2 and 4 are X's second COLUMN
```

And indeed `Xᵀ G = [[1,3],[2,4]] @ [[1],[2]] = [[7],[10]]`. ✓

The multipliers you needed were **columns of `X`**. `Xᵀ` is just "`X` with its columns turned into rows so a matmul can dot them."

______________________________________________________________________

### The Other One: `∂L/∂X = G Wᵀ`

Same story, different shared index. A single activation `X[i,a]` feeds *all `k` outputs* in its row, so its gradient sums over `j`:

```
∂L/∂X[i,a] = Σ_j G[i,j] · W[a,j]
```

Here `j` is already the column index of `G`, which is good because `G` is the left operand. But `j` is also the column index of `W`, and the right operand needs it as a *row*. Flip it: `Wᵀ`.

A deeper intuition worth holding onto: **forward, `W` maps `d → k`. Backward, gradient flows `k → d`.** You need the same weights with the arrows reversed, and reversing the arrows of a linear map *is* the transpose (its adjoint). Every backward pass through a linear layer runs the same wiring in reverse.

______________________________________________________________________

### The Shortcut Everyone Actually Uses

Once you trust the derivation, don't redo it. Just match shapes:

| Need | Have | Only option |
|---|---|---|
| `∂L/∂W` is `d×k` | `X` is `n×d`, `G` is `n×k` | `Xᵀ (d×n) @ G (n×k)` |
| `∂L/∂X` is `n×d` | `G` is `n×k`, `W` is `d×k` | `G (n×k) @ Wᵀ (k×d)` |

The gradient of a scalar loss with respect to a tensor always has the same shape as that tensor. Given that, dimensional analysis forces the transposes, as long as `n`, `d` and `k` are all different. With square shapes several arrangements type-check, so check with distinct sizes.

______________________________________________________________________

### In PyTorch: `nn.Linear` Stores the Transpose

`nn.Linear` keeps its weight as `(out, in)`, which is `k×d`, and computes `y = x Wᵀ + b` (see [`nn.Module` and Multi-Layer Networks](#why-you-need-nonlinearity)). Its `weight` is the transpose of the `W` above, so the same gradients come out rearranged:

```python
lin = nn.Linear(in_features=3, out_features=2)
x = torch.randn(4, 3, requires_grad=True)
g = torch.randn(4, 2)  # stands in for ∂L/∂y
lin(x).backward(gradient=g)

lin.weight.shape                                   # torch.Size([2, 3])  (out, in)
torch.allclose(lin.weight.grad, g.T @ x.detach())  # True: Gᵀ X, i.e. (Xᵀ G)ᵀ
torch.allclose(x.grad, g @ lin.weight.detach())    # True: G W, the transpose is already stored
torch.allclose(lin.bias.grad, g.sum(dim=0))        # True: shared across the batch, so summed over it
```

Same derivation, same shape-matching shortcut. Only the storage layout differs.

______________________________________________________________________

### Common confusions: the Transpose

- `XᵀG` comes from the convention `Y=XW`. `nn.Linear` stores weights as `(out, in)` and uses `XWᵀ`, so its stored weight gradient is `GᵀX`.
- The transpose changes which index matmul sums over. It does not mean the forward pass literally switched rows and columns in memory.
- A bias shared across `n` examples receives a sum of `n` contributions; a mean-reduced loss may already scale each contribution by `1/n`.

### Check your understanding: the Transpose

1. If `X` is `(5, 3)` and `G` is `(5, 2)`, what shape must `∂L/∂W` have for `Y=XW`?
1. Which index is summed in `∂L/∂W[a,j]`?
1. Why is `∂L/∂X = GWᵀ`?

<details markdown="1"><summary>Answers</summary>

1. `(3, 2)`, the shape of `W`; `XᵀG` has that shape.
1. The batch index `i`, because the same weight is used for every example.
1. Each input feature contributes to all output features, so the output index `j` is summed; `Wᵀ` puts `j` in matmul's inner position.

</details>

**One-minute recap:** forward sums over input features; weight backward sums over examples; input backward sums over output features. The required sum determines each transpose.

______________________________________________________________________

### References: the Transpose

- Stanford CS231n notes, *"Backpropagation, Intuitions"*. The matrix-multiply gradient (written `D = W X`, so `dW = dD.dot(X.T)`) and dimension analysis as the practical shortcut. cs231n.github.io/optimization-2/
- Goodfellow, Bengio & Courville, *Deep Learning* (2016), §6.5 "Back-Propagation and Other Differentiation Algorithms". Index-notation treatment of the chain rule over tensors.
- Petersen & Pedersen, *The Matrix Cookbook*, §2.4. Standard matrix derivative identities.

______________________________________________________________________

## ReLU and Dead Neurons

`ReLU(x) = max(0, x)` has gradient `0` for every negative input. The natural worry: doesn't that stop backprop?

For one inactive unit on one example, only that path is gated. Other active units and other examples can still contribute. The lasting problem is a unit that stays inactive over the data it sees.

**Trace to remember:** with three units, `z=[−0.4, 1.2, 0.8]`, only the first path is gated. The other two terms still add to the input gradient below.

______________________________________________________________________

### Why Backprop Doesn't Stop

The gate is **per-neuron** and **per-example**, and backprop through a layer is a **sum over paths**.

For hidden unit `j` with pre-activation `z_j = Σ_i w_ji·x_i + b_j` and `h_j = max(0, z_j)`:

∂L/∂xᵢ = Σⱼ (∂L/∂hⱼ) · 1[zⱼ > 0] · wⱼᵢ

The indicator kills **term `j`** of the sum, not the whole sum. An active unit can still pass gradient to the layer below when its upstream gradient and weight provide a signal.

**Concrete:** a layer of 3 units, upstream gradients `∂L/∂h = [2, −1, 3]`, pre-activations `z = [−0.4, 1.2, 0.8]`, and weights into `x_1` of `w_·1 = [0.5, −0.2, 0.9]`:

∂L/∂x₁ = 2(0)(0.5) + (−1)(1)(−0.2) + 3(1)(0.9) = 2.9

Unit 1 contributed nothing. The other two still contributed to the input gradient.

Two more reasons it self-heals:

- **Per-example.** With a batch of 64, unit 1 might be negative on 20 examples and positive on 44. The weight gradient is summed over the batch, so the unit still updates.
- **The gate moves.** Nothing pins `z_j` negative. Updates to earlier layers change `x`, so a unit that is silent this step can be active next step. With roughly symmetric pre-activations, about half the units may be on for a given example; the exact fraction depends on the data and initialization. Glorot, Bordes & Bengio (2011) argued this sparsity, with true zeros, is a feature.

______________________________________________________________________

### When a Unit Really Dies

If `z_j < 0` for **every input the unit sees**, its own incoming weights receive zero task gradient:

∂L/∂wⱼᵢ = (∂L/∂hⱼ) · 0 · xᵢ = 0

on every such example. With a fixed input representation, those weights cannot move through this loss, which is the **dying ReLU** problem. In a deeper network, updates to earlier shared layers can change the unit's inputs and sometimes reactivate it, so "dead" is a practical description rather than an absolute guarantee.

**Usual cause:** one oversized update (learning rate too high) shoves the unit's weights or bias far negative. Bad initialization does it too. Lu et al. (2019) show that the probability of a whole network being *born dead* goes to 1 as depth grows, and to 0 as width grows at fixed depth.

Mitigations, roughly in order of how often you'll reach for them:

| Fix | Mechanism |
|---|---|
| He initialization | Helps preserve activation variance at initialization; with roughly symmetric pre-activations, many units begin active |
| Lower LR / gradient clipping | Prevents the single oversized update that pushes weights or bias far negative |
| BatchNorm before activation | Re-centers `z` over each batch, so it's hard for a unit to end up negative on every input |
| Leaky ReLU, `max(0.01x, x)` | Gradient is `0.01` for `x < 0`, never `0`, so there's always a way back |
| GELU / SiLU | Smooth transitions that can pass gradient for many negative inputs |

GELU is covered in more depth in [Pre-Norm vs. Post-Norm](transformer.md#q-relu-or-gelu-inside-the-ffn).

______________________________________________________________________

### The `x = 0` Point

ReLU is non-differentiable exactly at zero. PyTorch chooses a subgradient of `0` there. This convention matters if an activation is exactly zero, but it does not change the distinction between a single gated example and a unit inactive across the data. See [Gradients and Gradient Descent](#case-2-differentiable-almost-everywhere-kinks) for the PyTorch check.

______________________________________________________________________

### Common confusions: ReLU and Dead Neurons

- An inactive ReLU blocks one term in a sum of gradient paths; it does not automatically block the whole layer.
- A unit inactive on today's batch may activate on another batch. Persistent inactivity across the data is the concern.
- Leaky ReLU supplies a negative-side slope; it does not guarantee that every layer's gradients are well behaved.

### Check your understanding: ReLU and Dead Neurons

1. If one of three ReLU units has a negative pre-activation, can the input below still receive a gradient?
1. Why do a persistently inactive unit's own incoming weights receive zero task gradient?
1. How could a unit in a deeper network become active again despite that zero incoming-weight gradient?

<details markdown="1"><summary>Answers</summary>

1. Yes. The active units' paths still contribute to the sum.
1. Its local ReLU derivative is zero for every example it sees, so the chain rule multiplies every such contribution by zero.
1. Earlier shared layers can change its input representation and move its pre-activation above zero.

</details>

**One-minute recap:** ReLU gates individual paths; repeated inactivity can stall a unit's own weights; initialization, learning rate, normalization, and activation choice affect how often that happens.

______________________________________________________________________

### References: ReLU and Dead Neurons

- Nair & Hinton, *"Rectified Linear Units Improve Restricted Boltzmann Machines"* (ICML 2010). Introduces rectified linear units, in RBMs.
- Glorot, Bordes & Bengio, *"Deep Sparse Rectifier Neural Networks"* (AISTATS 2011). The case for rectifiers in deep networks, including sparse representations with true zeros.
- Maas, Hannun & Ng, *"Rectifier Nonlinearities Improve Neural Network Acoustic Models"* (2013). Leaky ReLU.
- He et al., *"Delving Deep into Rectifiers"* (2015). He initialization and PReLU. arXiv:1502.01852
- Lu et al., *"Dying ReLU and Initialization: Theory and Numerical Examples"* (2019). Formal treatment of born-dead networks and how the probability scales with depth and width. arXiv:1903.06733
- Stanford CS231n notes, *"Neural Networks Part 1: Setting up the Architecture"*. The standard practitioner framing of dying ReLU. cs231n.github.io/neural-networks-1/

______________________________________________________________________

## DataLoader — Batching and the Dataset Abstraction

**The story of one example:** the dataset retrieves it, a transform prepares it, the loader groups it with other examples, the model produces logits, and the loss reduces the batch to a scalar. Keep each job separate when revising this page.

```
dataset[i] → (image: C×H×W, label: scalar)
collate B items → images: (B,C,H,W), labels: (B,)
model → logits: (B,num_classes)
loss → one scalar (with the default mean reduction)
```

Read [the two abstractions](#two-abstractions--keep-them-separate) and [how a batch becomes a scalar loss](#how-a-batch-becomes-a-scalar-loss) for the main flow; the intervening sections explain image transforms and normalization.

______________________________________________________________________

### Why mini-batches?

- **Full dataset (batch GD):** true gradient, but one update per full pass and memory blows up
- **One sample (pure SGD):** frequent updates, but jittery convergence
- **Mini-batch (common compromise):** for roughly independent examples, gradient estimate variance shrinks approximately like `1/B`, so quadrupling batch size halves the standard deviation of the noise. Correlation and dataset size affect the real gain.

### Two abstractions — keep them separate

| | Job | Analogy |
|---|---|---|
| `Dataset` | "give me item `i`" — random access to one sample | a list/array |
| `DataLoader` | batching, shuffling, parallel loading | an iterator that groups the list |

**`Dataset`** — subclass it, implement two methods:

```python
class MyDataset(Dataset):
    def __len__(self):           # total number of samples
    def __getitem__(self, idx):  # return (x, y) for one index
```

**`DataLoader`** — configure, don't subclass:

```python
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

It calls `__getitem__` under the hood and collates `B` samples into a batch. With simple tensor samples, default collation stacks each field and adds a leading batch dimension `(B, ...)`:

```python
for xb, yb in loader:   # xb: (B, features), yb: (B, ...)
    # 5-step loop here
```

**`shuffle=True` for training, `shuffle=False` for validation/test.** If data is ordered (e.g., all class-0 then all class-1), consecutive batches are biased. Shuffling each epoch decorrelates batches.

______________________________________________________________________

### Transforms — Format and Normalization Pipeline

`transforms` is a pipeline that processes raw data into what your model expects. Two jobs:

#### Format conversion — `ToTensor()`

```python
# Raw image: PIL Image, pixel values 0–255
# After ToTensor():
#   - converts to torch.Tensor
#   - for an 8-bit PIL image, scales values from [0, 255] → [0.0, 1.0]
#   - reshapes to (C, H, W)

transforms.ToTensor()
```

This is the classic `torchvision.transforms` API. In the newer `transforms.v2` API, the same idea is split into [`ToImage()` and `ToDtype(torch.float32, scale=True)`](https://docs.pytorch.org/tutorials/beginner/basics/transforms_tutorial.html). Scaling depends on the input image type; do not assume every tensor conversion divides by 255.

Neural nets train better on small float inputs than large integers — the scale directly affects gradient magnitude (see [Gradients and Gradient Descent](#why-centered-inputs-stabilize-gradients)).

#### Normalization — `Normalize(mean, std)`

```python
# Shifts each pixel so distribution is centered at 0
# (pixel - mean) / std   — applied independently per channel

transforms.Normalize((mean_ch1,), (std_ch1,))
```

**Why a tuple?** One value per channel. A single-channel image takes `(mean,)`. RGB takes three:

```python
transforms.Normalize((0.485, 0.456, 0.406),   # ImageNet R, G, B means
                     (0.229, 0.224, 0.225))    # R, G, B stds
```

#### Compose chains them

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])
# Applied in order, automatically, to every sample the DataLoader yields
```

#### Why it's the standard pattern — lazy + train/test split

Transforms are applied **lazily** — per sample as batches load, not upfront. This makes it natural to use different transforms for training vs evaluation:

```python
train_transform = transforms.Compose([
    transforms.RandomRotation(10),   # augmentation — training only
    transforms.ToTensor(),
    transforms.Normalize(...)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),           # no augmentation for test
    transforms.Normalize(...)
])
```

______________________________________________________________________

### How Augmentations Work

`Dataset.__getitem__(i)` is the hook point. The transform is stored on the dataset and called inside `__getitem__`:

```python
# Simplified torchvision internals
class SomeDataset(Dataset):
    def __getitem__(self, i):
        img, label = self.raw_data[i]      # raw, untouched on disk
        if self.transform:
            img = self.transform(img)      # called fresh, every access
        return img, label
```

Three consequences:

- **Stateless & per-call** — the raw bytes never change. Each `__getitem__` call produces a new transformed copy. A `RandomRotation` rolls a new angle *every epoch* → the model sees a slightly different image each time → free data variety without changing the dataset.
- **Lazy** — nothing runs until the DataLoader pulls index `i`. Memory holds raw data only; transformed tensors are ephemeral, created and discarded per batch.
- **Composable** — `Compose` is just function composition. Each transform is a callable:

```python
class Compose:
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
```

A list of callables applied in sequence — that's it.

______________________________________________________________________

### PIL Image

PIL = **Python Imaging Library** (the `Pillow` fork). It's a common Python object for representing an image in memory before tensor conversion. Many torchvision image datasets return PIL Images before a transform converts them.

```python
from torchvision import datasets

raw = datasets.MNIST(root="data", train=True, download=True)  # no transform
img, label = raw[0]

print(type(img))    # <class 'PIL.Image.Image'>
print(img.size)     # (28, 28)  — note: PIL is (W, H), not (H, W)
print(img.mode)     # 'L'  → 'L' means 8-bit grayscale
```

`type(img)` is how you confirm. After `ToTensor()` it becomes `torch.Tensor`.

______________________________________________________________________

### Computing Normalization Stats

**Can you compute mean/std at runtime?** Yes, from the training set. Weight every pixel equally; averaging batch standard deviations is not the standard deviation of the whole dataset, and averaging batch means equally is wrong when the final batch is smaller.

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ds = datasets.MNIST(root="data", train=True, download=True,
                    transform=transforms.ToTensor())
loader = DataLoader(ds, batch_size=1000)

total = torch.zeros(1)   # one entry per image channel (MNIST has one)
total_sq = torch.zeros(1)
count = 0
for xb, _ in loader:     # xb: (B,C,H,W)
    total += xb.sum(dim=(0, 2, 3))
    total_sq += xb.square().sum(dim=(0, 2, 3))
    count += xb.shape[0] * xb.shape[2] * xb.shape[3]
mean = total / count
std = (total_sq / count - mean.square()).clamp_min(0).sqrt()
```

**Do train and test stats differ?** Slightly — they're different image sets. **But always use training stats for both.** At inference you may have only one image; you can't compute meaningful stats from it. Test data is treated as unseen, so you normalize it with numbers derived only from training. Using test stats on test data is data leakage.

This is why constants like MNIST's 0.1307/0.3081 are hardcoded — computed once on train, reused everywhere.

______________________________________________________________________

### What are Logits?

Raw, unnormalized scores output by the model — one number per class, before any probability conversion.

**Example** — 3 classes (cat, dog, bird):

```
logits = [2.1, 0.3, 1.5]
           cat  dog  bird
```

These aren't probabilities — they don't sum to 1 and can be negative. To get probabilities, you pass them through softmax:

```
softmax([2.1, 0.3, 1.5]) ≈ [0.58, 0.10, 0.32]  ← now sums to 1
```

**Why keep them as logits and not convert first?**

`CrossEntropyLoss` takes raw logits and does the softmax internally — this is numerically more stable than doing `softmax → log → loss` yourself. So the convention is: **model outputs logits, loss function handles the conversion.**

______________________________________________________________________

### How a Batch Becomes a Scalar Loss

`loss = criterion(logits, yb)  # logits: (B, C), yb: (B,) → scalar`

Say `B=3`, `C=4` (3 samples, 4 classes). `CrossEntropyLoss` does this in two steps:

**Step 1 — loss per sample** (softmax + log + pick the correct class):

```
loss_0 = 0.43
loss_1 = 0.12
loss_2 = 0.67
```

**Step 2 — reduce to scalar:**

```
loss = mean(0.43, 0.12, 0.67) = 0.407   ← this is what .backward() sees
```

**Averaged, not summed — and that's the default.** PyTorch loss functions have a `reduction` argument:

```python
nn.CrossEntropyLoss(reduction='mean')   # default — divides by B
nn.CrossEntropyLoss(reduction='sum')    # sums instead
nn.CrossEntropyLoss(reduction='none')   # returns (B,), no reduction
```

**Why mean and not sum?** With sum, your loss (and gradients) scale with batch size — doubling `B` doubles the gradient magnitude. With mean, gradient scale stays constant regardless of batch size, so your learning rate stays stable.

#### Sequence outputs: one loss per token

A model that predicts at every position returns `(B, L, vocab)` logits against `(B, L)`
targets. `CrossEntropyLoss` wants classes on axis 1: `(N, C)` against `(N,)`, or
`(N, C, L)` against `(N, L)`. Passing `(B, L, vocab)` as-is reads `L` as the class axis
and fails with `Expected target size [B, vocab], got [B, L]`. Either of these fixes it,
and both give the same scalar:

```python
loss = criterion(logits.reshape(-1, vocab), targets.reshape(-1))  # (B*L, vocab), (B*L,)
loss = criterion(logits.transpose(1, 2), targets)                 # (B, vocab, L), (B, L)
```

Both reshapes are row-major, so row `b*L + l` of the logits and of the targets is the
same token.

Positions that shouldn't be scored, such as padding or the unmasked tokens of a
masked-token objective, get target `-100`, the default `ignore_index`. Those rows
contribute zero loss and zero gradient. Under `reduction='mean'`, the divisor is the
number of rows **kept**, not `B × L`:

```
loss = Σ_{kept rows} −log softmax(logits_row)[target_row]  /  N_kept
```

The gradient on each kept row's logits is `(softmax − one_hot) / N_kept`. A shared
weight such as the output head's `W` sums those over every kept token, so its update is
the average over scored tokens. See
[shared weights](#q-how-does-nnlinear-apply-one-weight-to-every-token).

### Common confusions: DataLoader

- `Dataset` supplies individual samples; `DataLoader` decides order, batching, and worker loading. A map-style dataset does not need to return a prebuilt batch.
- A random training transform is drawn on access; evaluation should use a stable transform. Both splits use normalization statistics calculated from **training data only**.
- Logits are scores, not probabilities. `CrossEntropyLoss` expects logits and class indices, and applies the needed log-softmax internally.

### Check your understanding: DataLoader

1. If images are `(3, 28, 28)` and the loader collates 32 of them, what shape reaches the model?
1. Why is averaging batch standard deviations insufficient for a dataset-wide standard deviation?
1. Why should test images use the training set's mean and standard deviation?

<details markdown="1"><summary>Answers</summary>

1. `(32, 3, 28, 28)`; the loader adds a leading batch axis.
1. Standard deviation depends on distances from the **global** mean; within-batch deviations omit differences between batch means.
1. Evaluation must treat test data as unseen, and an individual inference example may not support a meaningful new estimate.

</details>

**One-minute recap:** the dataset retrieves one item, transforms prepare it, the loader collates batches, and the loss reduces per-example errors. Estimate normalization statistics from training pixels, then reuse them.

______________________________________________________________________

## Random Streams and DataLoader Workers — Q&A

Two questions sit under every reproducible data pipeline: where the random numbers come
from, and which process draws them. This page answers both. For how a loader's splits,
samplers, and seeds are wired together, see the
[DataLoader internals](../app/src/training/dataload/README.md) page.

**Mental model:** a seed initializes a stream; each draw advances that stream. The main process chooses indices for a map-style dataset, while workers use their own streams for random work done while fetching those indices. A repeatable result depends on both the seed and which component consumes each draw.

**Reading route:** use the diagram below to locate the process, then read the [generator](#q-what-is-a-torchgenerator-and-why-use-one) and [worker seed](#q-why-does-the-dataloader-pass-an-rng-seed-to-workers-it-could-just-shuffle-the-indices-and-send-them) questions. The later sections explain process start methods and CUDA.

______________________________________________________________________

### Cheat-sheet: Random Streams

| You want | Do this |
|---|---|
| One component's randomness isolated from the rest | Give it its own `torch.Generator` and pass `generator=` |
| Different draws from a private generator in a loop | Create `g` once, outside the loop; a new unseeded `torch.Generator()` repeats the same value |
| A private generator that differs every run | `g = torch.Generator()`, then `g.seed()` |
| A reproducible train/val split | `random_split(..., generator=torch.Generator().manual_seed(0))` |
| A reproducible shuffle order | `DataLoader(..., shuffle=True, generator=torch.Generator().manual_seed(0))` |
| Reproducible dropout and weight init | `torch.manual_seed(seed)`; those draw from the global generator |
| To rewind a stream | `state = g.get_state()` … `g.set_state(state)` |
| Fresh random masks every epoch, reproducible per run | Draw from the global generator (`generator=None`) after one `torch.manual_seed` |
| The same masks at every evaluation | A private generator seeded with `seed + index` per sample |
| Validation that doesn't shift training's random stream | Wrap it in `torch.random.fork_rng(devices=[])` |
| CUDA inside a worker | `multiprocessing_context="spawn"`, or better, keep workers on the CPU |

**Where each thing happens:**

```
main process                                   worker process (one per num_workers)
────────────                                   ────────────────────────────────────
Sampler draws a permutation (loader.generator)
  │  indices, grouped into batches
  ▼
index queue ──────────────────────────────────► dataset[i] for each index
                                                 augmentations draw from the worker's
                                                 own seed: base_seed + worker_id
                                                 collate_fn → CPU tensors
result queue ◄──────────────────────────────── batch
  │
  ▼
pin_memory thread (if pin_memory=True)
  │
  ▼
batch.to("cuda")
```

Shuffling happens once, in the main process. Workers only fetch, transform, and collate.

______________________________________________________________________

### Q: After `torch.manual_seed(a)` and one draw, is the seed now `a + 1`?

No. The seed stays `a`. What changes is the generator's **state**, which is a different
thing.

- **The seed** is only the starting point. `manual_seed(a)` uses `a` to initialize the
  state, and `torch.initial_seed()` keeps returning `a`.
- **The state** is the generator's current position in a very long, fixed sequence. On
  the CPU, PyTorch uses the Mersenne Twister (MT19937), whose state is 624 32-bit words.
  On CUDA it uses Philox, which tracks its position as the seed plus an offset.
- **Advancing** means each draw consumes numbers from the sequence and moves the position
  forward. The next call continues from there.

Think of the seed as choosing a book and each draw as reading the next page.
`manual_seed(a + 1)` doesn't turn to the next page. It picks a different book whose pages
have no relation to book `a`.

```python
import torch

torch.manual_seed(seed=42)
x1, x2 = torch.rand(size=(1,)), torch.rand(size=(1,))
print(torch.initial_seed())  # 42: the seed is unchanged

torch.manual_seed(seed=42)  # back to page 1 of book 42
assert torch.equal(x1, torch.rand(size=(1,)))
assert torch.equal(x2, torch.rand(size=(1,)))  # the same sequence again

torch.manual_seed(seed=43)
print(torch.rand(size=(1,)))  # unrelated to x2

state = torch.get_rng_state()
torch.rand(size=(1,))
print(torch.equal(state, torch.get_rng_state()))  # False: the state advanced
```

A pseudo-random generator is a deterministic function, `state → (output, next_state)`.
The seed sets only the first state. After that the state evolves on its own, and nothing
writes back to the seed.

#### What "global" means

The global generator is `torch.default_generator`: one shared CPU generator per process.
Every random op called without `generator=` reads from it, including `torch.rand`,
`randint`, dropout, weight init, and `RandomSampler`. `torch.manual_seed(s)` resets it.

In the book picture, the global generator is one shared book on the process's desk. Any
call that doesn't bring its own book reads the next page of the shared one. So a draw
depends on every draw made before it in that process:

```python
torch.manual_seed(seed=0)
a = torch.rand(size=(1,))  # page 1

torch.manual_seed(seed=0)
torch.rand(size=(1,))  # some other code reads page 1 first
b = torch.rand(size=(1,))  # so this is page 2: b != a
```

Adding a layer, a random augmentation, or a debug `torch.rand` shifts every later draw,
including the next shuffle order.

"Global" means global per process. The main process and each worker have their own copy,
which is why workers are reseeded (see [below](#the-seed-is-for-randomness-inside-__getitem__)).
Python's `random` and NumPy's `np.random` are separate global generators, and each CUDA
device has its own default generator.

______________________________________________________________________

### Q: Does a generator store an array of random numbers?

It behaves like one: an endless array you read left to right. But nothing is stored. A
generator holds a small **state** and a **formula**. Each draw computes new numbers from
the state, then overwrites the state.

In the book picture, the book is never printed. It is a recipe, "next page =
recipe(current page)", and the bookmark is the state.

#### A toy generator

A linear congruential generator is the simplest example. PyTorch uses a stronger one,
but the structure is the same:

```python
state = 7  # the seed is the initial state


def rand():
    """Advance the state one step and return a float in [0, 1)."""
    global state
    state = (5 * state + 3) % 16  # update: the new state replaces the old one
    return state / 16  # output: the new state, scaled to [0, 1)
```

| Call | Old state | `5 · old + 3` | `mod 16` → new state | Output |
|---|---|---|---|---|
| 1 | 7 | 38 | 6 | 0.375 |
| 2 | 6 | 33 | 1 | 0.0625 |
| 3 | 1 | 8 | 8 | 0.5 |
| 4 | 8 | 43 | 11 | 0.6875 |
| 5 | 11 | 58 | 10 | 0.625 |

The whole generator is one integer. Seed 7 always produces this sequence, which is why
the same seed gives the same book, and no value exists before the call that computes it.

`mod 16` works like a 16-hour clock: 38 hours after midnight is hour 6, so the state
always stays in 0–15. The multiply-and-add jumps around the clock, and the wrap-around
makes the jumps look scattered.

**It eventually repeats.** From 10 the sequence continues 5, 12, 15, 14, 9, 0, 3, 2, 13,
4, then returns to 7: a cycle of 16 that visits every value once. That isn't luck. The
Hull–Dobell theorem says `state = (a · state + c) mod m` reaches the full period `m`
exactly when `c` and `m` share no factor, `a − 1` is divisible by every prime factor of
`m`, and `a − 1` is divisible by 4 if `m` is. Here `c = 3`, `m = 16`, and `a − 1 = 4`
satisfy all three. Real generators keep the structure and stretch the cycle: MT19937's
period is 2¹⁹⁹³⁷ − 1.

#### What PyTorch keeps

```python
g = torch.Generator()
print(g.get_state().numel())  # 5056 bytes
saved = g.get_state()
first = torch.rand(size=(10,), generator=g)
torch.rand(size=(10_000_000,), generator=g)
print(g.get_state().numel())  # still 5056: nothing grows or gets used up
g.set_state(saved)  # move the bookmark back
assert torch.equal(first, torch.rand(size=(10,), generator=g))
```

- **CPU: Mersenne Twister (MT19937).** The state is a fixed-size block, and a formula
  scrambles it into each next output. Ten million draws leave its size unchanged, and
  restoring it replays the same numbers, so the state alone decides every future output.
  `get_state` and `set_state` read and overwrite that state, which is why rewinding works.
- **CUDA: Philox.** Philox is counter-based: value `n` is `scramble(seed, n)`. It is the
  closest thing to the array picture, because it can jump straight to index `n` without
  computing the values before it. Thousands of GPU threads can each take their own slice,
  and the generator only tracks an offset.

______________________________________________________________________

### Q: What is a `torch.Generator`, and why use one?

A `torch.Generator` is an independent random number stream with its own state. Seed it,
then hand it to a random op through the `generator=` argument.

**The problem it solves.** `torch.manual_seed(42)` seeds PyTorch's default random streams. Random ops without an explicit generator draw from the default stream for their device. Draws are sequential within each stream, so one CPU consumer can change what a later CPU consumer sees. For example, additional CPU random initialization can change a later shuffle if that shuffle uses the default CPU generator. A private generator for the loader isolates its order from that unrelated draw.

#### Basic usage

```python
import torch

g = torch.Generator().manual_seed(42)  # CPU generator; manual_seed returns g
a = torch.rand(size=(3,), generator=g)
b = torch.randn(size=(2, 2), generator=g)
idx = torch.randperm(n=10, generator=g)

g.manual_seed(42)  # reseed: the same sequence again
assert torch.equal(a, torch.rand(size=(3,), generator=g))
```

Ops that take `generator=` include `torch.rand`, `randn`, `randint`, `randperm`,
`normal`, `bernoulli`, and `multinomial`, the in-place `Tensor.uniform_`, `normal_`,
and `random_`, and the `torch.nn.init` functions.

#### An unseeded generator always starts from the same seed

A generator is a reader holding one book (its seed) and a bookmark (its state).
`torch.rand()` without `generator=` still has a reader: the global one,
`torch.default_generator`. The two start differently:

- **The global generator** opens a random book. Its seed is non-deterministic and
  changes every time the process starts.
- **`torch.Generator()`** opens the same book every time, at page 1. Unseeded, it doesn't
  pick a random seed. It uses a constant, `67280421310721`, hardcoded as
  `default_rng_seed_val` in PyTorch's `c10/core/GeneratorImpl.h`.

That matters as soon as a generator is created inside a loop:

```python
for _ in range(3):
    torch.rand(size=(1,))  # A: the global generator

for _ in range(3):
    torch.rand(size=(1,), generator=torch.Generator())  # B: a new generator per draw

g = torch.Generator()
for _ in range(3):
    torch.rand(size=(1,), generator=g)  # C: one generator, reused
```

Two separate runs, with no `manual_seed` anywhere:

| | Run 1 | Run 2 |
|---|---|---|
| `torch.initial_seed()` | 10969210436945446397 | 6338311990818715765 |
| `torch.Generator().initial_seed()` | 67280421310721 | 67280421310721 |
| A: global generator | 0.6692, 0.6612, 0.9196 | 0.7824, 0.3618, 0.0007 |
| B: new generator per draw | 0.2673, 0.2673, 0.2673 | 0.2673, 0.2673, 0.2673 |
| C: one generator, reused | 0.2673, 0.8725, 0.3353 | 0.2673, 0.8725, 0.3353 |

| Pattern | Within a run | Across runs |
|---|---|---|
| A: `torch.rand()` in a loop | Different: the bookmark advances | Different |
| B: `generator=torch.Generator()` in a loop | Identical: a fresh copy of the same book, read from page 1 each time | Identical |
| C: one generator created outside the loop | Different | Identical: the fixed default seed |

Pattern B is almost always a bug. Create the generator once, then seed it on purpose:
`g.manual_seed(42)` for a reproducible stream, or `g.seed()` for one that differs every
run. `torch.manual_seed` doesn't reach a private generator.

#### A draw advances only the generator it reads

```python
g = torch.Generator()
g_before, global_before = g.get_state(), torch.get_rng_state()
torch.rand(size=(3,), generator=g)
print(torch.equal(g_before, g.get_state()))  # False: g advanced
print(torch.equal(global_before, torch.get_rng_state()))  # True: the global one didn't
```

In the [toy generator](#a-toy-generator), that is simply two state variables:

```python
global_state = ...  # random at startup, read by torch.rand()
g_state = 67280421310721  # read by torch.rand(generator=g)
```

Each call runs the update formula on the variable it was given. That independence is the
isolation described above: draws from the global generator can't shift `g`'s sequence,
and draws from `g` can't shift the global one.

The keyword is `generator=`. `torch.rand(3, gen=g)` raises a `TypeError`.

#### Three common uses

```python
from torch.utils.data import DataLoader, random_split

# 1. A reproducible train/val split
train_ds, val_ds = random_split(
    dataset=dataset,
    lengths=[0.8, 0.2],
    generator=torch.Generator().manual_seed(0),
)

# 2. A reproducible shuffle order, unaffected by the model's randomness
loader = DataLoader(
    dataset=train_ds,
    batch_size=32,
    shuffle=True,
    generator=torch.Generator().manual_seed(0),
)

# 3. Save and rewind the stream, e.g. to resume training
state = g.get_state()
x1 = torch.rand(size=(3,), generator=g)
g.set_state(state)
x2 = torch.rand(size=(3,), generator=g)
assert torch.equal(x1, x2)
```

#### Gotchas

- **The device must match.** A CUDA op needs `torch.Generator(device="cuda")`. Passing a
  CPU generator to an op that creates a CUDA tensor raises
  `Expected a 'cuda' device type for generator but found 'cpu'`.
- **Modules don't take one.** `nn.Dropout` and the default init in `nn.Linear.__init__`
  draw from the global generator, so full reproducibility still needs `torch.manual_seed`.
  Re-initializing weights yourself with `nn.init.*(..., generator=g)` is the exception.
- **A `DataLoader` generator does two jobs.** It drives the `RandomSampler`'s shuffle, and
  it draws the `base_seed` that seeds the workers (next question).
- **The other methods.** `g.initial_seed()` returns the seed in use, `g.seed()` reseeds
  from a non-deterministic source, and `torch.default_generator` is the global generator
  itself.

______________________________________________________________________

### Q: "The map-style dataset is then passed to the DataLoader to create batches." What does that mean?

PyTorch has two kinds of dataset, told apart by how you read a sample:

| Kind | You implement | You read a sample by |
|---|---|---|
| Map-style (`Dataset`) | `__getitem__(i)` and `__len__` | index: `dataset[i]`, like a dict or list |
| Iterable-style (`IterableDataset`) | `__iter__` | iterating: the next item in a stream |

"Map-style" means the dataset maps an index to a sample. It knows only one sample at a
time. Passing it to a `DataLoader` hands the batching to the loader: its sampler chooses
the indices, it calls `dataset[i]` for each, and `collate_fn` stacks the results into one
batch with a leading `B` axis. The dataset never sees a batch.

______________________________________________________________________

### Q: Why does the DataLoader pass an RNG seed to workers? It could just shuffle the indices and send them.

It does exactly that. For a map-style dataset with `shuffle=True`:

1. The main process owns the sampler (`RandomSampler`), which draws the permutation.
1. The main process groups the indices into batches and puts `(batch_idx, [indices])` on
   each worker's index queue.
1. Workers call `dataset[i]` for the indices they receive. They never shuffle.

Shuffling happens in one place. The seed workers get is for something else.

#### The seed is for randomness inside `__getitem__`

Each time a loader starts iterating, it draws a `base_seed` from its `generator`. Each
worker then seeds Python's `random` and `torch` with `base_seed + worker_id`, and NumPy
with a value derived from the same two numbers (NumPy seeding was added in PyTorch 1.9).
That randomness is what a random crop, a flip, added noise, or MLM masking inside
`__getitem__` or `collate_fn` draws from.

Without the reseed, forked workers would all start from a byte-for-byte copy of the
parent's RNG state:

```
worker 0: dataset[5]  → random_crop → crop at (12, 40)
worker 1: dataset[9]  → random_crop → crop at (12, 40)   same state, same "random" numbers
worker 2: dataset[17] → random_crop → crop at (12, 40)
```

Every worker produces the same sequence of augmentations, which silently cuts
augmentation diversity. This was a well-known bug with NumPy
before PyTorch 1.9 (pytorch/pytorch#5059). A per-worker seed breaks the symmetry, and
deriving every worker's seed from one `base_seed` keeps the run reproducible once you pass
`generator=`.

Inside a worker, `torch.utils.data.get_worker_info().seed` returns that worker's seed, if
another library needs seeding from it.

#### When workers do shuffle: `IterableDataset`

An iterable dataset has no indices and often no `__len__`: it streams from files, object
storage, or shards. The main process can't permute indices it doesn't have, so each worker
reads its own stream and shuffles locally, for example with a shuffle buffer.

An iterable dataset must partition its stream across workers to avoid duplicates. One design gives every worker the **same shuffled order**, then keeps every `num_workers`-th element; that design needs a shared shuffle seed or the slices can overlap and leave gaps. Another design assigns distinct source shards first, then each worker may shuffle its own shard independently. DataPipes supported by `DataLoader` include coordinated seeding and sharding for the former pattern; a custom `IterableDataset` must define its own ordering and partitioning.

**In short:** index-based data shuffles centrally. Stream-based data has no central index permutation, so its dataset controls ordering and must partition work across workers. Separately, per-worker seeds decorrelate augmentations.

______________________________________________________________________

### Q: Does a worker's randomness depend on the global generator or on the base seed?

Both. It's a chain:

```
torch.manual_seed(train_seed)
  │
  ▼
main process global generator    (or loader.generator, if you passed one)
  │  one draw each time an iterator is created, i.e. once per epoch
  ▼
base_seed
  │  worker i reseeds its own global generator with base_seed + i
  ▼
torch.rand / randint calls without generator= inside worker i
```

- Inside a worker, a draw without `generator=` reads that worker's global generator. It
  starts from `base_seed + worker_id`, so it depends directly on the base seed.
- The base seed is drawn from the main process's global generator when each epoch's
  iterator is created. So the worker depends indirectly on the main generator, and through
  it on `train_seed`.

In the book picture: at the start of each epoch, the main process reads one page of its
own book to get `base_seed`, then hands worker `i` book number `base_seed + i` for the
whole epoch.

**A consequence.** Epoch 2's `base_seed` depends on everything else that read the main
CPU generator during epoch 1: the sampler's shuffle and any CPU-side random op in the
training loop (CUDA ops use a separate GPU generator). The chain is still deterministic,
but a change to the training loop can change what the workers draw.

With `num_workers=0` there are no workers and no reseeding. `collate_fn` runs in the main
process and reads the main global generator directly. A private generator passed as
`generator=` sits outside the chain altogether.

______________________________________________________________________

### Q: A `collate_fn` seeds each sample with `seed + index`. Do its masks repeat?

Take a masking `collate_fn` that gives each sample its own generator:

```python
for index, ids in rows:
    generator = torch.Generator().manual_seed(seed + index)
    x, y = mask_row(ids=ids, generator=generator)
```

`index` is the dataset index, not the batch position. It follows the sample wherever the
sampler puts it, so `shuffle=True` changes nothing below.

- **Across rows and batches: no repeats.** Each sample gets its own seed, so each row
  gets a different mask.
- **Across epochs: every mask repeats.** Sample 42 always gets `seed + 42`, so its mask
  is identical every epoch. Passing a different `seed` at startup doesn't help; it's fixed
  for the whole run.

That is right for validation and test: every evaluation sees the same corruption, so
losses are comparable across checkpoints. For training it is **static masking**: the
model sees one corruption per sample for the whole run. RoBERTa (§4.1) found that dynamic
masking, a fresh mask each time a sequence is fed, matches or slightly beats static.
Original BERT softened static masking by duplicating the data 10 times with different
masks. The [DataLoader internals](../app/src/training/dataload/README.md) page covers
why the two splits want different masking.

#### Dropping `index` makes it worse

It is tempting to seed with `seed` alone to get "fresh noise". That does the opposite.
Every row gets the same seed, so every row in every batch and every epoch gets the same
position pattern: position 3 always masked, position 7 always replaced, and so on.

#### The fix: make the seed optional

```python
def collate_fn(rows, seed: int | None):
    """Mask each row.

    seed: an int masks every sample identically each epoch (validation, test).
        None draws from the global generator, so masks change every epoch (train).
    """
    for index, ids in rows:
        generator = None if seed is None else torch.Generator().manual_seed(seed + index)
        x, y = mask_row(ids=ids, generator=generator)
```

`torch.rand` and `torch.randint` accept `generator=None`, so `mask_row` only needs its
annotation widened to `torch.Generator | None`. Then:

```python
torch.manual_seed(seed=train_seed)  # once, at startup
train_dl = DataLoader(dataset=train_ds, collate_fn=partial(collate_fn, seed=None))
val_dl = DataLoader(dataset=val_ds, collate_fn=partial(collate_fn, seed=VAL_SEED))
```

Training masks are now fresh and still reproducible. With `num_workers=0`, the global
generator keeps advancing across epochs. With `num_workers > 0`, each epoch draws a new
`base_seed`, so workers mask differently each epoch (the chain in the previous question).
Either way, `train_seed` determines the whole run.

______________________________________________________________________

### Q: What does `torch.random.fork_rng` do?

It is a context manager that saves the global generator's state when the block starts and
restores it when the block ends. Draws inside the block still read the global generator,
but once you leave, it is as if they never happened.

In the book picture: put a bookmark in the shared book, read some pages inside the block,
and the bookmark goes back where it was on the way out.

```python
torch.manual_seed(seed=0)
with torch.random.fork_rng(devices=[]):
    torch.rand(size=(1000,))  # pages read inside the block
a = torch.rand(size=(1,))  # the same value as if the block never ran
```

It doesn't create a new generator; it only protects the existing one. By default it
saves and restores the CPU generator and every visible CUDA device's generator.
`devices=[]` forks only the CPU generator.

#### Where it helps: validating mid-training

Creating any `DataLoader` iterator draws a base seed from the loader's `generator`, or from
the main global generator if it has none. That holds even for a validation loader with
`shuffle=False` and `num_workers=0`. So each validation pass advances the main generator
by one draw, and the next training epoch's shuffle depends on how often you validated:
validating every epoch and every two epochs give different training runs.

```python
with torch.random.fork_rng(devices=[]):
    val_loss = evaluate(model=model, loader=val_dl)
```

Training's random stream is now the same whether or not validation ran. The alternative is
to give the train loader its own `generator=torch.Generator().manual_seed(train_seed)`, so
its shuffle and base seed never read the global generator at all.

A private per-sample generator, as in the previous question, doesn't need `fork_rng`: it
never reads the global generator and nothing else reads it. `fork_rng` with a
`manual_seed` inside would do the same job, more slowly, since it saves and restores the
state on every call.

______________________________________________________________________

### Q: With `num_workers > 0`, is each worker a thread or a process?

A process. Each worker is a separate OS process started by Python's `multiprocessing`.

#### Why processes and not threads

Python's GIL (global interpreter lock) lets only one thread in a process run Python bytecode at a time. Data loading mixes Python work, I/O, and library routines; some library routines release the GIL. Separate worker processes each have their own interpreter and GIL, allowing Python-heavy preparation to run on multiple cores.

#### How the processes start

The start method decides what a worker begins with. Override it with
`DataLoader(..., multiprocessing_context=...)`.

| Method | Default on | What happens | Consequence |
|---|---|---|---|
| `fork` | Linux, Python ≤ 3.13 | The child is a copy-on-write clone of the parent | Fast startup; the dataset is shared until something writes to it |
| `forkserver` | Linux, Python ≥ 3.14 | Children fork from a clean helper process started early | Safer than `fork`; the dataset is pickled and sent over |
| `spawn` | macOS, Windows | A fresh interpreter starts, and the dataset is pickled and sent to it | Slowest startup; the dataset must be picklable (no lambdas, no open file handles) |

Check yours with `torch.multiprocessing.get_start_method()`.

With `spawn` and `forkserver`, the child imports your main script, so the training code
needs a guard:

```python
if __name__ == "__main__":
    train()
```

Without it, each worker runs the top level of the script again and tries to start
training itself.

#### What that implies

- **No shared variables.** A worker that changes `self.counter` changes its own copy. The
  main process and the other workers never see it. The only way back is the result queue.
- **Batches cross a process boundary.** Tensors travel through shared memory rather than
  being copied byte by byte. That's why a Docker container sometimes needs a bigger
  `/dev/shm` (`--shm-size`).
- **Copy-on-write still leaks under `fork`.** Reading a Python object updates its
  reference count, which is a write, so the pages holding a big Python list get copied
  into each worker a little at a time. Memory grows per worker. Storing the data in a
  tensor or a NumPy array avoids it, because one array is one object.
- **Pinning uses a thread.** With `pin_memory=True`, the main process runs a background
  thread that copies finished batches into pinned memory. A thread works here because the
  copy runs in C and releases the GIL.

______________________________________________________________________

### Q: Isn't forking a process that uses CUDA forbidden?

Forking is fine. *Using* CUDA in the forked child is what's forbidden, and DataLoader
workers don't.

#### Why CUDA and `fork` don't mix

`fork` copies the parent's memory, but only the thread that called `fork`. The CUDA driver
keeps state that isn't plain memory: GPU contexts, background driver threads, device
handles, and locks. The child inherits memory that refers to that state, without the
threads and the GPU context behind it. So if a forked child touches CUDA, PyTorch refuses:

```
RuntimeError: Cannot re-initialize CUDA in forked subprocess.
To use CUDA with multiprocessing, you must use the 'spawn' start method
```

#### Why workers get away with it

The standard design keeps workers on the CPU:

```
worker (forked, CPU only):  read → decode → augment → CPU tensor
main (owns CUDA):           batch.to("cuda")    ← the GPU transfer happens here
```

The parent may have initialized CUDA. The children never call into it, so the state they
inherited is never touched.

#### When you need `spawn`

Any of these in a `Dataset` or `collate_fn` breaks a forked worker:

- Moving a tensor to the GPU (`.cuda()`, `.to("cuda")`).
- Running a model on the GPU, for GPU-side augmentation or feature extraction.
- Returning CUDA tensors from workers, which the docs advise against anyway.

If you need one of them:

```python
DataLoader(dataset=dataset, num_workers=4, multiprocessing_context="spawn")
```

`spawn` starts a fresh interpreter, so each worker creates its own CUDA context. Every
context costs GPU memory, and startup is slower. The usual better answer is to keep the
workers on the CPU and pass `pin_memory=True`, so the main process's copy to the GPU is
fast.

Python 3.14's `forkserver` default on Linux sidesteps the problem in a different way: the
helper that forks the workers is a fresh process that never initialized CUDA, so there is
no CUDA state to inherit.

______________________________________________________________________

### Common confusions: Random Streams

- **Seed versus state:** the seed starts a sequence; the state records where you currently are in it. Reseeding restarts, while saving and restoring state rewinds to an exact point.
- **Shuffle versus augmentation:** the main process chooses indices for map-style data; workers can still draw random crops or masks inside `__getitem__`.
- **Reproducible versus identical workers:** worker seeds differ so they do not produce the same random augmentation sequence, but they derive from a repeatable base seed.
- **Reproducible versus repeated:** a fixed seed per sample makes the run reproducible *and* repeats every mask each epoch. The global generator after one `manual_seed` is reproducible without repeating.
- **Unseeded versus random:** an unseeded `torch.Generator()` always starts from the same constant seed. Only the global generator gets a fresh seed when the process starts.
- **Stored versus computed:** a generator keeps a fixed-size state, not a list of numbers. Each draw computes the next values and overwrites the state.

### Check your understanding: Random Streams

1. Why can a private generator keep a loader's shuffle stable after an unrelated CPU random draw is added?
1. If the main process already shuffles sample indices, why seed each worker?
1. What must a worker avoid after being forked from a parent that has initialized CUDA?
1. After `torch.manual_seed(5)` and three draws, what does `torch.initial_seed()` return?
1. A masking `collate_fn` seeds each row with `seed + index`. What does training see across epochs, and what happens if you drop `index`?
1. Why can adding a validation pass change the next training epoch's shuffle order?
1. What does `torch.rand(size=(1,), generator=torch.Generator())` return inside a loop, and in a second run of the script?
1. After ten million draws, is a generator's state larger than before?

<details markdown="1"><summary>Answers</summary>

1. The loader consumes its own stream, so unrelated draws do not advance its state.
1. Random transforms and other work inside item retrieval need distinct, reproducible worker streams.
1. It must avoid using CUDA in that forked child; CPU loading followed by a main-process GPU transfer is the usual flow.
1. `5`. The draws advanced the state, not the seed.
1. The same mask for each sample every epoch (static masking). Without `index`, every row gets the same seed, so every row everywhere gets the same mask pattern.
1. Creating the validation iterator draws a base seed from the global generator, advancing it. `fork_rng`, or a private train-loader generator, removes the dependence.
1. The same value on every iteration, and the same value again in the second run. Each new generator starts from the fixed default seed, at page 1.
1. No. The state has a fixed size and each draw overwrites it, which is also why `get_state` and `set_state` can rewind a stream.

</details>

**One-minute recap:** a seed picks a sequence and draws advance the state; the sampler and workers consume different random streams; process boundaries explain why workers need seeds and why CPU loading is the usual CUDA-safe design.

______________________________________________________________________

### Sources: Random Streams

- PyTorch docs, [torch.Generator](https://pytorch.org/docs/stable/generated/torch.Generator.html):
  `manual_seed`, `get_state`, `set_state`, `initial_seed`, `seed`, and devices.
- PyTorch docs, [torch.utils.data](https://pytorch.org/docs/stable/data.html): dataset
  types, "Data Loading Order and Sampler", "Single- and Multi-process Data Loading",
  "Platform-specific behaviors", and "Randomness in multi-process data loading".
- PyTorch docs, [Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html):
  the global generators, `worker_init_fn`, and seeding a `DataLoader`.
- PyTorch docs, [torch.manual_seed](https://pytorch.org/docs/stable/generated/torch.manual_seed.html),
  [torch.initial_seed](https://pytorch.org/docs/stable/generated/torch.initial_seed.html),
  and [torch.random.fork_rng](https://pytorch.org/docs/stable/random.html#torch.random.fork_rng).
- PyTorch source: `aten/src/ATen/core/MT19937RNGEngine.h`, the CPU generator's engine.
- PyTorch source: `c10/core/GeneratorImpl.h`, `default_rng_seed_val = 67280421310721`,
  the seed of an unseeded `torch.Generator()`.
- Knuth, *The Art of Computer Programming*, Vol. 2, §3.2.1.2: the full-period
  (Hull–Dobell) theorem for linear congruential generators.
- Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3" (SC 2011): Philox, the
  counter-based generator behind CUDA's default.
- Liu et al., [RoBERTa](https://arxiv.org/abs/1907.11692) (2019), §4.1 "Static vs.
  Dynamic Masking".
- Devlin et al., [BERT](https://arxiv.org/abs/1810.04805) (2018), §3.1, and
  `dupe_factor=10` in google-research/bert's `create_pretraining_data.py`.
- PyTorch docs, [Multiprocessing best practices](https://pytorch.org/docs/stable/notes/multiprocessing.html):
  CUDA in multiprocessing and the `spawn` requirement.
- PyTorch source: `torch/utils/data/dataloader.py` (`_base_seed` drawn from
  `loader.generator`, the pin-memory thread) and `torch/utils/data/_utils/worker.py`
  (`_worker_loop` seeding `random`, `torch`, and NumPy from `base_seed + worker_id`).
- Python docs, [multiprocessing: contexts and start methods](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods):
  the default start method per platform and the 3.14 change to `forkserver`.
- [fork(2)](https://man7.org/linux/man-pages/man2/fork.2.html): the child gets only the
  calling thread.

______________________________________________________________________

## Inspecting Your Model — Best Practices

Most PyTorch bugs are silent shape/dtype/device mismatches that either crash 50 layers later or — worse — *broadcast successfully into wrong results*.

**Question to ask before inspecting code:** which boundary first changes the tensor from the shape or type you expected? Follow one batch from input to output, naming each axis. If that trace is correct, inspect parameter registration and then gradients.

| Symptom | First thing to inspect | Why |
|---|---|---|
| Loss computes but seems wrong | Prediction and target shapes together | Broadcasting can create pairwise comparisons |
| Model moves device but a tensor does not | `named_parameters()` and `named_buffers()` | A plain tensor attribute is not registered |
| Gradient missing | Leaf parameters, graph path, then hooks | The path may be unused or gradient recording disabled |
| Architecture hard to follow | Module tree or `grad_fn` graph | They show different structures |

______________________________________________________________________

### Shape Checking

**Assert shapes explicitly.** Broadcasting will happily turn a `(32,1)` vs `(32,)` mismatch into a `(32,32)` tensor with no error.

```python
def assert_shape(t: torch.Tensor, expected: tuple, name: str = "tensor") -> torch.Tensor:
    assert t.dim() == len(expected), \
        f"{name}: rank {t.dim()} != {len(expected)} {tuple(t.shape)}"
    for i, (a, e) in enumerate(zip(t.shape, expected)):
        assert e == -1 or a == e, \
            f"{name}: dim {i} is {a}, expected {e} {tuple(t.shape)}"
    return t   # returns tensor so calls can be chained
```

Usage: `assert_shape(logits, (batch, -1, vocab), "logits")`. Use `-1` as a wildcard dim.

**Name dimensions in comments** on every line that reshapes: `# x: (B, T, C)`. Reshape/permute bugs are the #1 time sink.

**Check a representative batch before training.** One forward and backward pass can catch many wiring mistakes early:

```python
x = torch.randn(4, 8)
out = model(x)
assert_shape(out, (4, 1), "out")
loss = criterion(out.squeeze(), torch.zeros(4))
loss.backward()
assert all(p.grad is not None for p in model.parameters())
```

**Verify parameter registration** after building a model to catch unregistered tensors early:

```python
def assert_param_names(model: nn.Module, expected: set[str]) -> None:
    actual = {name for name, _ in model.named_parameters()}
    assert actual == expected, \
        f"param mismatch: missing={expected - actual}, extra={actual - expected}"
```

______________________________________________________________________

### PyTorch Hooks

A hook is a **callback you register on a module or tensor** that fires automatically during forward or backward — think middleware or event listeners.

#### Three main types

```python
# 1. Forward hook — fires AFTER forward(), sees input & output
def fwd_hook(module, inp, out):
    print(f"{module.__class__.__name__}: out shape {out.shape}")

handle = model.net[1].register_forward_hook(fwd_hook)
model(xb)        # prints layer output shape automatically
handle.remove()  # always clean up
```

```python
# 2. Forward pre-hook — fires BEFORE forward(), can modify input
def pre_hook(module, inp):
    return (inp[0] * 2,)   # returning a value replaces the input

model.register_forward_pre_hook(pre_hook)
```

```python
# 3. Backward hook — fires during backward(), sees gradients
def bwd_hook(module, grad_in, grad_out):
    print(f"grad norm: {grad_out[0].norm()}")

model.net[1].register_full_backward_hook(bwd_hook)
```

#### What hooks are actually used for

- **Debugging** — inspect activations/gradients mid-network without editing `forward`
- **Feature extraction** — grab intermediate layer outputs (transfer learning, Grad-CAM)
- **Gradient monitoring** — detect vanishing/exploding gradients by logging norms
- **Profiling** — torchvision/profilers attach hooks to time each layer

This is also why `model(x)` matters over `model.forward(x)` — `__call__` is what runs registered hooks around `forward`. Bypass `__call__` and your hooks silently never fire. See [`nn.Module` and Multi-Layer Networks](#modelx-vs-modelforwardx--never-call-forward-directly).

______________________________________________________________________

### `jaxtyping` + `beartype` — Runtime Shape Contracts

Shape annotations as type hints, enforced at runtime. `jaxtyping` provides the vocabulary; `beartype` enforces it. Together they give typed contracts: wrong shape in → immediate, readable exception at the call site.

```bash
pip install jaxtyping beartype
```

```python
from jaxtyping import Float, jaxtyped
from beartype import beartype
from torch import Tensor

@jaxtyped(typechecker=beartype)
def linear(x: Float[Tensor, "batch in_dim"],
           w: Float[Tensor, "in_dim out_dim"]) -> Float[Tensor, "batch out_dim"]:
    return x @ w

x = torch.randn(32, 128)
w = torch.randn(128, 10)
linear(x, w)           # ✓

bad = torch.randn(64, 99)
linear(bad, w)         # ✗ raises immediately: in_dim is 99 here but 128 in w
```

The shared name `in_dim` means `x.shape[1]` must equal `w.shape[0]`. Same-name enforcement catches cross-argument relationship bugs that a standalone `assert` can't express cleanly.

Apply to `forward()` during development:

```python
class TinyNet(nn.Module):
    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "batch 2"]) -> Float[Tensor, "batch 1"]:
        ...
```

______________________________________________________________________

### Graph Visualization

#### Option 1 — `torchviz`: the actual backward graph

```bash
pip install torchviz   # also: brew install graphviz / apt-get install graphviz
```

```python
from torchviz import make_dot
y = model(torch.randn(1, 2))
dot = make_dot(y, params=dict(model.named_parameters()))
dot.render("graph", format="png")
```

Use `show_attrs=True, show_saved=True` to see what autograd saves for the backward pass. Those saved tensors drive GPU memory during training — useful when debugging OOMs.

#### Option 2 — `torchview`: module-level view with shapes

Better when you want to see your architecture rather than raw `grad_fn` nodes. `device='meta'` traces shapes with zero memory allocation.

```python
from torchview import draw_graph
g = draw_graph(model, input_size=(1, 2), device='meta')
g.visual_graph   # renders inline in a notebook
```

Both options need the `graphviz` system binary on your PATH — that's the #1 reason these silently error.

#### Option 3 — zero-dependency: walk `grad_fn` yourself

```python
def print_graph(t, depth=0, seen=None):
    fn = getattr(t, "grad_fn", t)
    if fn is None:
        return
    seen = set() if seen is None else seen
    if id(fn) in seen:
        return
    seen.add(id(fn))
    print("  " * depth + type(fn).__name__)
    for next_fn, _ in getattr(fn, "next_functions", ()):
        if next_fn is not None:
            print_graph(next_fn, depth + 1, seen)
```

```python
y = model(torch.randn(1, 2))
print_graph(y)
# AddmmBackward0
#   AccumulateGrad        ← leaf parameter (.grad lands here)
#   ReluBackward0
#     AddmmBackward0
#       ...
```

`AccumulateGrad` nodes are leaf parameters (where `.grad` lands). `*Backward0` nodes are operations. This is the same structure `torchviz` renders, just as text.

**Recommendation:** option 3 to understand graph structure, `torchview` for day-to-day architecture + shape debugging, `torchviz` only when you need backward/saved-tensor detail.

### Common confusions: Inspecting Your Model

- A **module tree** shows registered layers and parameters. An **autograd graph** shows operations from one particular forward call. A module can appear in many forward graphs.
- A shape assertion tells you whether dimensions match your expectation. It cannot tell you whether labels or masking values have the right meaning.
- A missing `.grad` may mean the parameter was unused, the graph was not recorded, or it was inspected before backward; a hook is useful after checking those basics.

### Check your understanding: Inspecting Your Model

1. Why might `(32, 1)` predictions and `(32,)` targets produce a loss without a shape error?
1. Which view would you inspect to see saved operations for backward: `torchview`'s module diagram or a `grad_fn`/`torchviz` graph?
1. What does `named_parameters()` reveal about a learnable tensor assigned as an ordinary attribute?

<details markdown="1"><summary>Answers</summary>

1. Broadcasting can expand them to `(32, 32)`, comparing every prediction with every target.
1. The autograd graph from `grad_fn` or `torchviz`.
1. It is absent, so an optimizer built from `model.parameters()` will not receive it.

</details>

**One-minute recap:** trace shapes at boundaries first, verify registered state second, and inspect the graph or hooks when a gradient path remains unclear.
