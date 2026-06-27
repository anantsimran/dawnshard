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
