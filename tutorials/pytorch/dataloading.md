# DataLoader — Batching and the Dataset Abstraction

## Why mini-batches?

- **Full dataset (batch GD):** true gradient, but one update per full pass and memory blows up
- **One sample (pure SGD):** frequent updates, but jittery convergence
- **Mini-batch (sweet spot):** gradient estimate variance shrinks like `1/B` — quadrupling batch size halves noise, with diminishing returns past ~32–512

## Two abstractions — keep them separate

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

It calls `__getitem__` under the hood, stacks `B` samples into one tensor (adding the batch dimension at the front: `(B, ...)`), and hands it to you:

```python
for xb, yb in loader:   # xb: (B, features), yb: (B, ...)
    # 5-step loop here
```

**`shuffle=True` for training, `shuffle=False` for validation/test.** If data is ordered (e.g., all class-0 then all class-1), consecutive batches are biased. Shuffling each epoch decorrelates batches.

______________________________________________________________________

## What are Logits?

Raw, unnormalized scores output by the model — one number per class, before any probability conversion.

**Example** — 3 classes (cat, dog, bird):

```
logits = [2.1, 0.3, 1.5]
           cat  dog  bird
```

These aren't probabilities — they don't sum to 1 and can be negative. To get probabilities, you pass them through softmax:

```
softmax([2.1, 0.3, 1.5]) = [0.62, 0.11, 0.27]  ← now sums to 1
```

**Why keep them as logits and not convert first?**

`CrossEntropyLoss` takes raw logits and does the softmax internally — this is numerically more stable than doing `softmax → log → loss` yourself. So the convention is: **model outputs logits, loss function handles the conversion.**

______________________________________________________________________

## How a Batch Becomes a Scalar Loss

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
