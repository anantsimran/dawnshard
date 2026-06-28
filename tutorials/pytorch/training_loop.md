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

## `.to(device)` and the Optimizer

```python
model = MLP().to(device)
```

- `MLP()` — instantiates the module; all `nn.Linear` weights are created on **CPU** by default.
- `.to(device)` — walks every registered parameter and moves its storage to that device. Mutates in place, returns the same model.

```python
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
```

- `model.parameters()` — a generator yielding every learnable tensor (weights + biases).
- The optimizer stores *references* to those tensors — it doesn't copy them. `opt.step()` mutates them in place.

**Critical ordering gotcha:** call `.to(device)` *before* creating the optimizer, so the optimizer captures the GPU tensors, not stale CPU ones.

______________________________________________________________________

## Optimizer Algorithm

**Plain SGD:** `w ← w − lr · grad`. One global learning rate, no memory.

**Adam** adds two pieces of state per parameter:

- **Momentum (`m`)** — running average of past gradients. Smooths the path, like a ball carrying inertia.
- **Variance (`v`)** — running average of past *squared* gradients. Gives each parameter its own effective learning rate.

```
m ← β1·m + (1−β1)·grad           # smoothed direction
v ← β2·v + (1−β2)·grad²          # smoothed magnitude
w ← w − lr · m / (√v + ε)        # per-parameter adaptive step
```

Parameters with consistently large gradients get smaller steps; noisy ones get damped. This is why Adam "just works" with `lr=1e-3` and needs little tuning. (Kingma & Ba, *Adam*, 2014.)

______________________________________________________________________

## SE Mental Model — the Full Data Flow

| ML thing | SE equivalent | Where state lives |
|---|---|---|
| `model` (nn.Module) | Object with mutable fields | Weights = `nn.Parameter` tensors |
| `model.parameters()` | Getter returning refs to those fields | The tensors themselves |
| `forward(x)` | Pure-ish function: input → output | Reads weights, returns prediction |
| `loss` | Scalar + hidden DAG | Graph references intermediate tensors |
| `loss.backward()` | Traversal that writes into `.grad` fields | Mutates `param.grad` on each leaf |
| `optimizer` | Controller holding refs + its own buffers | `m`, `v` buffers per param (Adam) |
| `opt.step()` | Reads `.grad`, mutates weights in place | Writes new values into `param.data` |
| `opt.zero_grad()` | Clears the `.grad` fields | Resets accumulator to 0 |

**The data flow as a pipeline:**

```
xb ──forward──► pred ──loss_fn──► loss
                                    │
                              backward() (writes param.grad)
                                    │
                                    ▼
weights ◄──step() reads .grad────  optimizer (uses m, v state)
   │
   └──zero_grad() clears .grad, ready for next batch
```

The thing that confuses SEs: **gradients are stored on the parameters, not returned.** `backward()` returns nothing — it's a side-effecting traversal that deposits `.grad` onto each leaf tensor. The optimizer then reads those `.grad` fields. Three objects (model, loss graph, optimizer) communicate through shared mutable tensor state, not through return values.

______________________________________________________________________

## Why Embed Loss Inside the Model

Mostly two practical reasons:

**1. Multi-GPU efficiency.** `DataParallel` splits a batch across GPUs and runs `forward` on each. If loss is inside `forward`, the expensive computation gets parallelized and only a small scalar is gathered back. If loss is outside, the full logits tensor must be gathered to one GPU first — a memory bottleneck.

**2. Encapsulation for frameworks.** HuggingFace models return loss directly when you pass labels:

```python
output = model(input_ids=x, labels=y)
output.loss.backward()   # loss came from inside the model
```

This lets training loops be model-agnostic.

For most learning code, **keep loss outside** — it's clearer and makes the training loop explicit. Embedding is an optimization/framework pattern, not a default.

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

> For the *architecture* under this loop — how the optimizer shares tensors with the model, and how schedulers adjust the LR — see [optimizer_and_scheduler.md](optimizer_and_scheduler.md).

______________________________________________________________________

## Aggregating Loss Correctly Across an Epoch

Two separate collapses happen, and conflating them causes a real bug.

**Collapse A — samples → one number (the criterion).** The loss compares each prediction to its target, giving one value *per sample*; `reduction='mean'` (the default) averages those into one number for the batch. `loss.item()` pulls out that float. So `loss.item()` already means "average loss per sample, for this one batch."

**Collapse B — batches → one number (the epoch loss).** The naive move — averaging the per-batch means — **breaks when batches differ in size** (the last batch is usually smaller):

```
32-sample batch mean 0.5, then 8-sample batch mean 1.0

naive (mean of means):      (0.5 + 1.0) / 2            = 0.75   ← wrong
correct (sample-weighted):  (0.5*32 + 1.0*8) / (32+8)  = 0.60   ← right
```

The fix is a small sample-weighted accumulator. It also generalizes to accuracy, F1 — any "accumulate per batch, compute at the end" metric:

```python
class MeanMetric:
    """Sample-weighted running mean; correct even with unequal batch sizes."""
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        self.total += value * n      # weight each batch mean by its sample count
        self.count += n

    def compute(self) -> float:
        return self.total / self.count if self.count else 0.0
```

Have each step report its batch size (`n = y.size(0)`) so aggregation can weight by it.

______________________________________________________________________

## Structuring the Loop — Two Shapes

The loop body (iterate, accumulate, average) is *identical* for train and eval; only the per-batch operation differs. Two ways to factor that out — they're the same fork as "data + functions" vs "object with methods."

### Path C — functional core, imperative shell

A `@dataclass` bundles the mutable system (it's pure data — no methods — so the dataclass *signals* "functions live elsewhere"). The step holds all mutation and returns `(loss, n)`; `run_epoch` is dumb iteration, parameterized by which `step_fn` you pass:

```python
@dataclass
class TrainState:
    model: nn.Module
    optimizer: Optimizer
    criterion: nn.Module
    device: torch.device

def train_step(state: TrainState, batch) -> tuple[float, int]:
    x, y = (t.to(state.device) for t in batch)
    state.optimizer.zero_grad(set_to_none=True)
    loss = state.criterion(state.model(x), y)
    loss.backward()
    state.optimizer.step()
    return loss.item(), y.size(0)

@torch.no_grad()
def eval_step(state: TrainState, batch) -> tuple[float, int]:
    x, y = (t.to(state.device) for t in batch)
    return state.criterion(state.model(x), y).item(), y.size(0)

def run_epoch(state, loader, step_fn, *, train: bool) -> float:
    state.model.train(train)
    metric = MeanMetric()
    for batch in loader:
        metric.update(*step_fn(state, batch))   # unpacks (loss, n)
    return metric.compute()
```

`step_fn` is a function passed *as a value* (`train_step` or `eval_step`) — the Strategy pattern, same idea as passing a comparator to `sort()`. The loop is written **once**:

```python
for epoch in range(epochs):
    tr = run_epoch(state, train_loader, train_step, train=True)
    va = run_epoch(state, val_loader,   eval_step,  train=False)
```

### Path D — Trainer object

When you want the *process* to have an identity — logging, checkpointing, best-model tracking — wrap it in an object. State lives on `self`; `fit` owns the cross-cutting concerns. The key point: `save()` writes **two** `state_dict`s, which is exactly what lets you persist and resume model and optimizer independently:

```python
class Trainer:
    def fit(self, train_loader, val_loader, epochs):
        for epoch in range(1, epochs + 1):
            tr = self.run_epoch(train_loader, self.train_step, train=True)
            va = self.run_epoch(val_loader,   self.eval_step,  train=False)
            if va < self.best_val:
                self.best_val = va
                self.save("best.pt")          # model + optimizer state together

    def save(self, name):
        torch.save({"model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_val": self.best_val}, self.ckpt_dir / name)
```

In a real codebase, `Trainer.train_step` would **delegate** to the free `train_step` from Path C rather than re-implement it — keep the batch logic in one place and let the `Trainer` be a thin stateful shell.

**Which to use:** the dataclass-vs-class choice *is* the C-vs-D fork. Reach for C while learning (the mutation is named and isolated in `state`); reach for D once you need orchestration with state.
