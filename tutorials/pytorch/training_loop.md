# The Training Loop

## The 5-Step Rhythm

Every training loop in PyTorch follows **this exact pattern**, per batch:

```
1. forward(x)             →  get predictions ŷ
2. loss(ŷ, y)             →  compute scalar loss L
3. optimizer.zero_grad()  →  CLEAR old gradients (easy to forget)
4. loss.backward()        →  compute ∂L/∂w for all params
5. optimizer.step()       →  w ← w − lr · ∂L/∂w
```

**Why `zero_grad()`?** PyTorch *accumulates* gradients by default. If you don't zero them, gradients from the previous batch add to the current one — silent bug, wrong updates.

- `backward()` = chain rule over the computation graph → fills `.grad` on every parameter
- `optimizer.step()` = applies the update rule (Adam, SGD, etc.) using those `.grad` values

______________________________________________________________________

## `train()` and `eval()`

```python
model.train()   # enables dropout, batch norm training behavior
model.eval()    # disables them for inference
```

Forgetting `model.eval()` during validation is a classic bug. Always pair it with `torch.no_grad()` to skip graph construction during inference:

```python
model.eval()
with torch.no_grad():
    preds = model(x_val)
```

______________________________________________________________________

## DataLoader — Batching and the Dataset Abstraction

### Why mini-batches?

- **Full dataset (batch GD):** true gradient, but one update per full pass and memory blows up
- **One sample (pure SGD):** frequent updates, but jittery convergence
- **Mini-batch (sweet spot):** gradient estimate variance shrinks like `1/B` — quadrupling batch size halves noise, with diminishing returns past ~32–512

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

It calls `__getitem__` under the hood, stacks `B` samples into one tensor (adding the batch dimension at the front: `(B, ...)`), and hands it to you:

```python
for xb, yb in loader:   # xb: (B, features), yb: (B, ...)
    # 5-step loop here
```

**`shuffle=True` for training, `shuffle=False` for validation/test.** If data is ordered (e.g., all class-0 then all class-1), consecutive batches are biased. Shuffling each epoch decorrelates batches.

______________________________________________________________________

## Nested Loop Structure

```python
for epoch in range(N):
    model.train()
    for xb, yb in train_loader:
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            # compute validation loss / accuracy
```

One **epoch** = one full pass over the data = many batch updates.
