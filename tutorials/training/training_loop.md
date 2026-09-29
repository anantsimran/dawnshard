# The Training Loop

**Revision question:** How does one batch change the model, and what state carries over to the next batch?

Think of a batch as a short chain: **predictions → loss → gradients → updated weights**. The model owns the weights, autograd writes their gradients, and the optimizer reads those gradients to update the weights. The next batch sees the updated model.

**Reading map:** Start with [the five-step rhythm](#the-5-step-rhythm) and [the two-batch trace](#follow-two-batches). Then use [train versus eval](#train-and-eval) and [epoch aggregation](#aggregating-loss-correctly-across-an-epoch) to explain what changes outside an individual update. [Loop structure](#structuring-the-loop--two-shapes) is a deeper design reference.

## The 5-Step Rhythm

For an ordinary training batch, these operations have to happen in this causal order (clearing gradients may also happen before the forward pass):

```
1. forward(x)             →  get predictions ŷ
2. loss(ŷ, y)             →  compute scalar loss L
3. optimizer.zero_grad()  →  clear gradients left by earlier batches
4. loss.backward()        →  compute ∂L/∂w for used trainable params
5. optimizer.step()       →  update weights using the optimizer's rule
```

**Why `zero_grad()`?** PyTorch *accumulates* gradients by default. Without clearing them, gradients from previous batches add to the current one. That is useful when you intentionally accumulate gradients over several batches; otherwise it changes the update you meant to make.

- `backward()` = chain rule over the computation graph → fills `.grad` on the leaf parameters that contributed to the loss
- `optimizer.step()` = applies the update rule (Adam, SGD, etc.) using those `.grad` values

### Follow two batches

For a one-weight model with prediction `w·x`, let every batch contain `x=1`, target `0`, and loss `L=½(w·x)²`. Start at `w=2` with SGD learning rate `0.1`:

| Event | Weight before update | Gradient used | Weight after update |
|---|---:|---:|---:|
| Batch 1 | 2.00 | 2.00 | 1.80 |
| Batch 2, after `zero_grad()` | 1.80 | 1.80 | 1.62 |
| Batch 2, if gradients were **not** cleared | 1.80 | 2.00 + 1.80 = 3.80 | 1.42 |

The second forward pass uses `w=1.80`, not the original `2.00`. The last row shows why `.grad` is state: `backward()` adds the new `1.80` to the old `2.00` unless that field is cleared.

______________________________________________________________________

## `.to(device)` and the Optimizer

```python
model = MLP().to(device)
```

- `MLP()` — instantiates the module; all `nn.Linear` weights are created on **CPU** by default.
- `.to(device)` — moves registered parameters and buffers to that device and returns the module. Device conversions can affect parameter objects, so optimizer references deserve attention.

```python
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
```

- `model.parameters()` — an iterator over registered parameters (weights, biases, and any other `nn.Parameter`s), including ones you may later freeze.
- The optimizer stores *references* to those tensors — it doesn't copy them. `opt.step()` mutates them in place.

**Safe ordering:** move the model before creating the optimizer, so the optimizer receives the parameters in their final placement. This project's `TrainState.create()` currently moves the model after optimizer construction; that pattern relies on the parameter references remaining valid for the chosen conversion. If changing devices or optimizer types, check the references rather than assuming every conversion behaves identically.

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

Adam normalizes each coordinate by its recent squared gradients, so the size of a raw gradient alone does not determine its step. `lr=1e-3` is a common starting point, but the loss curve still decides whether it fits the model and batch size. (Kingma & Ba, *Adam*, 2014.)

For bias correction, AdamW, and how to pick a learning rate, see [adam_and_learning_rate.md](adam_and_learning_rate.md).

______________________________________________________________________

## SE Mental Model — the Full Data Flow

| ML thing | SE equivalent | Where state lives |
|---|---|---|
| `model` (nn.Module) | Object with mutable fields | Weights = `nn.Parameter` tensors |
| `model.parameters()` | Getter returning refs to those fields | The tensors themselves |
| `forward(x)` | Pure-ish function: input → output | Reads weights, returns prediction |
| `loss` | Scalar + hidden DAG | Graph references intermediate tensors |
| `loss.backward()` | Traversal that writes into `.grad` fields | Mutates `param.grad` on used trainable leaf parameters |
| `optimizer` | Controller holding refs + its own buffers | `m`, `v` buffers per param (Adam) |
| `opt.step()` | Reads `.grad`, mutates weights in place | Changes parameter values without adding this update to the forward graph |
| `opt.zero_grad()` | Clears the `.grad` fields | Resets or removes the accumulator |

**The data flow as a pipeline:**

```
xb ──forward──► pred ──loss_fn──► loss
                                    │
                              backward() (writes param.grad)
                                    │
                                    ▼
weights ◄──step() reads .grad────  optimizer (uses m, v state)
   │
   └──zero_grad() clears .grad before the next backward pass
```

The thing that confuses SEs: **gradients are stored on the parameters, not returned.** `backward()` returns nothing — it's a side-effecting traversal that deposits `.grad` onto each leaf tensor. The optimizer then reads those `.grad` fields. Three objects (model, loss graph, optimizer) communicate through shared mutable tensor state, not through return values.

______________________________________________________________________

## Why Embed Loss Inside the Model

Mostly two practical reasons:

**1. Multi-GPU efficiency.** `DataParallel` splits a batch across GPUs and runs `forward` on each. If loss is inside `forward`, loss computation also happens on each device, and only small per-device loss values need gathering. The caller must still combine those losses correctly. If loss is outside, the full logits tensor is gathered first, which can be a memory bottleneck.

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

These control different things: `eval()` changes modules such as dropout and batch norm; `no_grad()` stops autograd from recording operations. Forgetting either can make validation misleading or waste memory. Pair them for ordinary validation:

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

**Collapse A — samples → one number (the criterion).** For a per-sample loss configured with `reduction='mean'` (the default for many PyTorch criteria), the criterion averages sample losses into one number for the batch. `loss.item()` pulls out that float. Under this assumption, `loss.item()` means "average loss per sample, for this one batch." Losses reduced over tokens or elements need a matching count instead.

**Collapse B — batches → one number (the epoch loss).** The naive move — averaging the per-batch means — **breaks when batches differ in size** (the last batch is usually smaller):

```
32-sample batch mean 0.5, then 8-sample batch mean 1.0

naive (mean of means):      (0.5 + 1.0) / 2            = 0.75   ← wrong
correct (sample-weighted):  (0.5*32 + 1.0*8) / (32+8)  = 0.60   ← right
```

The fix is a small sample-weighted accumulator. It also works for accuracy when each batch reports its fraction correct. **F1 is different:** aggregate true positives, false positives, and false negatives first, then compute F1 from those totals; averaging batch F1 scores can still be wrong.

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

______________________________________________________________________

## Common Confusions

| Easy to mix up | What to remember |
|---|---|
| `backward()` versus `step()` | `backward()` computes and stores gradients; `step()` changes weights. |
| `zero_grad()` versus `no_grad()` | The first clears stored gradients; the second prevents graph recording. |
| `eval()` versus `no_grad()` | The first changes model behavior; the second changes autograd behavior. |
| Batch loss versus epoch loss | A mean for each batch needs sample weighting before it becomes a mean for the epoch. |

## Check Your Recall

1. If two identical batches follow one SGD update, why is their second gradient different even when gradients are cleared?
1. What changes if `backward()` runs twice without `zero_grad()`?
1. Why do validation passes commonly use both `eval()` and `no_grad()`?

<details markdown="1">
<summary>Answers</summary>

1. The first update changes the weight, so the second forward pass and its gradient use a new value.
1. The second gradient is added to the first in each parameter's `.grad`; the following step uses their sum.
1. `eval()` sets inference behavior for modules such as dropout and batch norm, while `no_grad()` avoids recording a backward graph.

</details>

**One-minute recap:** A forward pass builds a temporary computation graph; `backward()` deposits gradients on persistent parameters; `step()` changes those parameters. Clear gradients unless accumulation is intentional. An epoch is many such updates, followed by correctly aggregated metrics and usually a separate evaluation pass.
