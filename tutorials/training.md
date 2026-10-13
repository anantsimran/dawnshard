# Training

What happens between a batch and a weight update, and what you can see while it
happens. The loop first, then the two objects that hold the mutable state
(optimizer and scheduler), then how to choose a learning rate and how to measure
a run that is slow or unstable.

**On this page:**

- [The Training Loop](#the-training-loop)
- [Optimizer & Scheduler — the Decoupling](#optimizer--scheduler--the-decoupling)
- [Adam and the Learning Rate](#adam-and-the-learning-rate)
- [Observability in the Training Loop](#observability-in-the-training-loop)

______________________________________________________________________

## The Training Loop

**Revision question:** How does one batch change the model, and what state carries over to the next batch?

Think of a batch as a short chain: **predictions → loss → gradients → updated weights**. The model owns the weights, autograd writes their gradients, and the optimizer reads those gradients to update the weights. The next batch sees the updated model.

**Reading map:** Start with [the five-step rhythm](#the-5-step-rhythm) and [the two-batch trace](#follow-two-batches). Then use [train versus eval](#train-and-eval) and [epoch aggregation](#aggregating-loss-correctly-across-an-epoch) to explain what changes outside an individual update. [Loop structure](#structuring-the-loop--two-shapes) is a deeper design reference.

### The 5-Step Rhythm

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

#### Follow two batches

For a one-weight model with prediction `w·x`, let every batch contain `x=1`, target `0`, and loss `L=½(w·x)²`. Start at `w=2` with SGD learning rate `0.1`:

| Event | Weight before update | Gradient used | Weight after update |
|---|---:|---:|---:|
| Batch 1 | 2.00 | 2.00 | 1.80 |
| Batch 2, after `zero_grad()` | 1.80 | 1.80 | 1.62 |
| Batch 2, if gradients were **not** cleared | 1.80 | 2.00 + 1.80 = 3.80 | 1.42 |

The second forward pass uses `w=1.80`, not the original `2.00`. The last row shows why `.grad` is state: `backward()` adds the new `1.80` to the old `2.00` unless that field is cleared.

______________________________________________________________________

### `.to(device)` and the Optimizer

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

### Optimizer Algorithm

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

For bias correction, AdamW, and how to pick a learning rate, see [Adam and the Learning Rate](#adam-and-the-learning-rate).

______________________________________________________________________

### SE Mental Model — the Full Data Flow

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

### Why Embed Loss Inside the Model

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

### `train()` and `eval()`

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

### Nested Loop Structure

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

> For the *architecture* under this loop — how the optimizer shares tensors with the model, and how schedulers adjust the LR — see [Optimizer & Scheduler](#optimizer--scheduler--the-decoupling).

______________________________________________________________________

### Aggregating Loss Correctly Across an Epoch

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

### Structuring the Loop — Two Shapes

The loop body (iterate, accumulate, average) is *identical* for train and eval; only the per-batch operation differs. Two ways to factor that out — they're the same fork as "data + functions" vs "object with methods."

#### Path C — functional core, imperative shell

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

#### Path D — Trainer object

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

### Common confusions: The Training Loop

| Easy to mix up | What to remember |
|---|---|
| `backward()` versus `step()` | `backward()` computes and stores gradients; `step()` changes weights. |
| `zero_grad()` versus `no_grad()` | The first clears stored gradients; the second prevents graph recording. |
| `eval()` versus `no_grad()` | The first changes model behavior; the second changes autograd behavior. |
| Batch loss versus epoch loss | A mean for each batch needs sample weighting before it becomes a mean for the epoch. |

### Check your recall: The Training Loop

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

______________________________________________________________________

## Optimizer & Scheduler — the Decoupling

**Revision question:** When the learning rate changes, which object changed—and which weight update will see the new value?

[The Training Loop](#the-training-loop) showed the five operations in a batch. The key relationship here is **model parameter → optimizer parameter group → scheduler**. The model owns parameters; the optimizer keeps references to them and their update state; the scheduler keeps a reference to the optimizer and edits its learning-rate field. The loop decides when each operation happens.

**Reading map:** [Where parameters live](#where-parameters-live) through [the scheduler](#the-scheduler) explains the shared state. [Ordering and cadence](#ordering-rule-and-cadence) determines which learning rate an update uses. Use [the built-in schedules](#the-built-in-schedules), [warmup and decay](#warmup-and-decay-how-to-build-them), and [the `fit` compatibility table](#using-a-scheduler-with-fit) as references once that picture is clear.

______________________________________________________________________

### Where Parameters Live

Parameters are owned by the **model**. Writing `self.fc = nn.Linear(10, 2)` registers `fc` as a child module; that `Linear` module registers its own `weight` and `bias` as parameters. Modules form a tree, and `model.parameters()` walks it recursively.

The tensor *data* is just a buffer in CPU/GPU memory. The model holds a Python *reference* to it. That distinction is the whole key.

```python
model.fc.weight        # the actual nn.Parameter object
model.parameters()     # iterator over every param tensor (walks the tree)
model.named_parameters()   # ('fc.weight', tensor), ('fc.bias', tensor), ...
model.state_dict()     # name -> parameter/buffer tensor, used for saving
```

______________________________________________________________________

### How the Optimizer "Knows" Which Params to Update

It **doesn't know about the model at all.** Look at construction:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

You hand it `model.parameters()` — the **actual tensor objects, by reference**. Not copies, not names, not the model. The optimizer stashes those references in `optimizer.param_groups[0]['params']`. They are the *same Python objects* the model uses:

```python
model = nn.Linear(2, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)

model.weight is opt.param_groups[0]['params'][0]   # True
```

Same object, two names — ordinary Python aliasing, exactly like `b = a` on a list. For plain SGD without momentum or weight decay, `optimizer.step()` conceptually applies `param ← param − lr·param.grad` without tracking the update in autograd. It changes *the exact parameter* the model uses in its next forward pass. The optimizer never looks up the model. **The shared parameter is the link.**

Each shared tensor carries **two slots**:

- `param.data` — the value (model reads it in forward; optimizer writes it in step)
- `param.grad` — the proposed change (`backward()` writes it; optimizer reads it)

Two parties passing notes through one tensor's two pockets. Neither imports the other; `autograd` is the third party that fills `.grad`, and the optimizer never touches the graph.

#### The compute graph is transient

The graph is the *ephemeral* part. Each recorded forward pass builds a fresh DAG (define-by-run); parameters are leaf tensors referenced by graph operations. Ordinarily `backward()` walks the graph, accumulates gradients on used leaf parameters, and releases the intermediate graph. The parameters and their `.grad` fields remain.

```
model + params:  persist for the whole run            (the storage)
compute graph:   born each forward, dies each backward (the wiring, ephemeral)
.grad:           written by backward, lives on the param, cleared by zero_grad
```

`.grad` lives on the tensor, not in the graph — which is *why* both model and optimizer can see it.

______________________________________________________________________

### Why the Optimizer Isn't Part of `nn.Module`

A model defines a *function*; an optimizer defines an *update rule*. They vary independently (the same ResNet trains under SGD/Adam/LAMB; the same Adam trains any model), so coupling them would multiply combinations. Four concrete reasons:

1. **It operates on a flat view of tensors, not the module tree.** The constructor takes an iterable of tensors or param-group dicts — it can optimize a subset, params spanning *multiple* models (GANs), or a bare `nn.Parameter` in no module at all.
1. **The interface is `.grad`, not a model method.** `backward()` writes `.grad`; `step()` reads `.grad` and changes parameter values. They coordinate through the shared tensor.
1. **Separately checkpointable state.** `model.state_dict()` is the learned function; `optimizer.state_dict()` is training scaffolding (Adam's `m`, `v`, which are *not* parameters). You often want weights for inference *without* dragging stale optimizer moments along.
1. **Param groups need different policies.** `add_param_group(...)` lets you unfreeze a backbone mid-training or set a lower LR for pretrained layers vs the head — an optimizer concern with no home in the module.

The SE analogy: **`nn.Module` is the data structure; the optimizer is an algorithm operating on it.** You don't bake quicksort into the array. (Keras made the opposite call — `model.compile(optimizer=...)` — trading flexibility for convenience.)

______________________________________________________________________

### What `step()` Actually Does — and Inspecting It

For standard optimizers, `step()` alters the **parameters**, not the scheduled learning rate. It reads the `lr` in `param_groups`. For vanilla SGD, the conceptual update is `param ← param − lr·param.grad`, performed without recording it in the forward graph.

Adam has **two** kinds of "rate," and only one is the scheduler's business:

- **Per-parameter adaptation** (`m`, `v` buffers in `optimizer.state[param]`) — automatic; each param gets its own effective step from its gradient history.
- **The base LR** (`lr` in `param_groups`) — one number per parameter group scaling its updates. *This* is what a scheduler rewrites.

So Adam adapts *relative* to a base LR; the scheduler moves the base. Two knobs, two objects.

```python
opt.param_groups[0]['lr']     # this group's base LR (what the scheduler edits)
opt.state[model.weight]       # Adam's m, v for that one param
```

**Gotcha:** for Adam, `optimizer.state` is usually **empty until after the first `step()`** — its moment buffers are created lazily on the first update.

______________________________________________________________________

### The Scheduler

A scheduler adjusts the **learning rate** over time. It does *not* touch parameters — same shared-state pattern, one level up. It wraps the optimizer and, on `scheduler.step()`, writes a new value into `optimizer.param_groups[i]['lr']`:

```
scheduler.step()   →  writes  param_groups[i]['lr']
optimizer.step()   →  reads   param_groups[i]['lr'],  writes  param.data
```

So the full picture is a chain of state changes. The scheduler *does* hold the optimizer and edit its parameter groups; the optimizer holds parameter references but no model reference:

```
scheduler →[lr in param_groups]→ optimizer →[.data on param]→ model
                                 optimizer ←[.grad on param]← backward(graph)
```

and your loop is the conductor making them fire in order.

#### Follow two updates

Suppose an SGD optimizer starts with `w=2` and `lr=0.1`, and the gradient on each update is `2`. A scheduler with `gamma=0.5` runs **after** each optimizer update:

| Event | Learning rate read by optimizer | Weight after `optimizer.step()` | Learning rate after `scheduler.step()` |
|---|---:|---:|---:|
| First update | 0.10 | 1.80 | 0.05 |
| Second update | 0.05 | 1.70 | 0.025 |

The scheduler did not change `w` directly. It changed the value the *next* optimizer update reads. In a real model the second gradient would usually change with the new weight; holding it at `2` isolates the scheduling effect.

**Why bother:** a schedule can make early updates large enough to move quickly and later updates small enough to settle. The useful shape depends on the optimizer, model, and run length.

#### Ordering rule and cadence

For the usual schedules, call `optimizer.step()` **before** `scheduler.step()`. Reversing them makes the first update use an already advanced schedule: a `StepLR(step_size=1, gamma=0.1)` built with `lr=0.1` gives that update `0.01`, and PyTorch warns about the order.

*How often* you call the scheduler is set by the schedule's design, not by SGD vs batch GD:

| Cadence | Schedules | Call `scheduler.step()` |
|---|---|---|
| **Per epoch** | `StepLR`, `MultiStepLR`, `ExponentialLR`, `ReduceLROnPlateau`, `CosineAnnealingLR` sized in epochs | once per epoch |
| **Per batch** | `OneCycleLR`, warmup, Noam, `CosineAnnealingLR` sized in steps | once per optimizer step |

Batch-cadence schedules must be *sized* to total steps, e.g. `OneCycleLR(optimizer=opt, max_lr=0.1, total_steps=epochs * len(loader))`. Rule of thumb: **the number of `scheduler.step()` calls must match how the schedule was parameterized.** A scheduler has no idea what an epoch is. It counts calls to `step()`, and the count lives in `scheduler.last_epoch` whatever you meant it to count.

With mini-batches an epoch is many updates, so per-batch ≠ per-epoch and you must choose. (Under full-batch GD there's one update per epoch, so the distinction vanishes.)

```python
for epoch in range(epochs):
    for x, y in loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()       # update weights with the current lr
        # scheduler.step()     # here for a per-batch schedule
    scheduler.step()           # here for a per-epoch schedule
```

______________________________________________________________________

### The Built-in Schedules

Every output in this and the following sections was produced by running the code with `torch==2.12.1`.

All of `torch.optim.lr_scheduler` does one of two things: multiply the `lr` by a factor on a timetable, or compute it from a formula of the step count. Each row below starts from `SGD(lr=0.1)` and lists the `lr` used by updates 0, 1, 2, …, calling `scheduler.step()` after each `optimizer.step()`:

| Scheduler | Arguments | `lr` per update |
|---|---|---|
| `StepLR` | `step_size=3, gamma=0.1` | `0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.001` |
| `MultiStepLR` | `milestones=[2, 5], gamma=0.1` | `0.1, 0.1, 0.01, 0.01, 0.01, 0.001` |
| `ExponentialLR` | `gamma=0.5` | `0.1, 0.05, 0.025, 0.0125` |
| `LinearLR` | `start_factor=0.25, total_iters=3` | `0.025, 0.05, 0.075, 0.1, 0.1` |
| `CosineAnnealingLR` | `T_max=4, eta_min=0` | `0.1, 0.0854, 0.05, 0.0146, 0.0`, then `0.0146, 0.05` |
| `OneCycleLR` | `max_lr=0.1, total_steps=10` | `0.004, 0.052, 0.1, 0.0950, 0.0812, 0.0611, 0.0389, 0.0188, 0.0050, ≈0` |
| `LambdaLR` | `lr_lambda=f` | `initial lr × f(step)` — see [the Noam schedule](#the-noam-schedule) |
| `ReduceLROnPlateau` | `factor, patience` | driven by a metric, not the step count — [its own section](#reducelronplateau) |

What to read off the table:

- **`StepLR` / `MultiStepLR`** are the classic vision recipe: train at a flat `lr`, divide by 10 at fixed epochs.
- **`CosineAnnealingLR` is periodic.** It reaches `eta_min` at `T_max` and then climbs back up. Set `T_max` to the full length of the run (in whatever unit you step it) or the tail of training runs at a rising `lr`.
- **`OneCycleLR` is a whole run in one object:** warm up to `max_lr` over the first 30% (`pct_start=0.3`), anneal to near zero, and also cycle SGD momentum inversely to the `lr`. It refuses to overrun: the 11th `step()` on `total_steps=10` raises `ValueError: Tried to step 11 times. The specified number of total steps is 10`.
- **`LinearLR`** is almost never used alone. It's the warmup half of a composed schedule.

#### Composing schedules

`SequentialLR` runs schedulers one after another, switching at `milestones`. Warmup 2 steps, then cosine over 4:

```python
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer=optimizer,
    schedulers=[
        torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer, start_factor=0.25, total_iters=2
        ),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=4),
    ],
    milestones=[2],
)
# lr per update: 0.025, 0.0625, 0.1, 0.0854, 0.05, 0.0146, 0.0, 0.0146
```

The cosine phase restarts its own count at the milestone, so its `T_max` is the length of *its* phase, not of the run. `ChainedScheduler` is the other combinator: it steps every scheduler on each call, so their factors multiply instead of taking turns.

______________________________________________________________________

### Warmup and Decay: how to build them

**Warmup** ramps the LR up from near zero over the first few thousand steps. It matters most for post-norm transformers: the gradient near the output is large at initialization no matter how deep the model is, so a full-size LR at step 0 can make training diverge. See [Pre-Norm vs. Post-Norm](transformer.md#q-why-is-the-number-of-layers-what-breaks-post-norm). **Decay** shrinks the LR late in training so the weights can settle instead of bouncing around a minimum.

[Adam and the Learning Rate](#warmup-and-decay-why-adam-wants-them) covers *why* Adam in particular wants both. This section is how to build them.

#### Warmup + cosine

Linear warmup followed by cosine decay, stepped **once per batch**:

```python
optimizer = torch.optim.AdamW(params=model.parameters(), lr=3e-4, weight_decay=0.01)
total_steps = num_epochs * len(train_loader)
warmup_steps = 100

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer=optimizer,
    schedulers=[
        torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer, start_factor=0.01, total_iters=warmup_steps
        ),
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=total_steps - warmup_steps, eta_min=3e-5
        ),
    ],
    milestones=[warmup_steps],
)
```

With `total_steps = 1000` the `lr` goes `3e-6 → 1.5e-4 (step 50) → 3e-4 (step 100) → 1.65e-4 (step 550) → 3e-5 (step 999)`.

#### The Noam schedule

The original Transformer (Vaswani et al., 2017, §5.3) used a different shape: linear warmup, then a decay that never reaches zero and needs no end date. It's usually called the **Noam schedule**, after Noam Shazeer, one of the paper's authors.

```
lr(step) = d_model^(−0.5) · min(step^(−0.5), step · warmup_steps^(−1.5))
```

The `min` picks between two curves that cross at `step = warmup_steps`:

- **Before the crossing**, the second term is smaller: `lr = step · d_model^(−0.5) · warmup_steps^(−1.5)`. That's a straight ramp up from zero.
- **After it**, the first term is smaller: `lr = d_model^(−0.5) · step^(−0.5)`. That's an inverse-square-root decay.

The same schedule is easier to read if you pull the peak out:

```
peak     = (d_model · warmup_steps)^(−0.5)
lr(step) = peak · min(step / warmup_steps, √(warmup_steps / step))
```

With the paper's `d_model = 512` and `warmup_steps = 4000`:

| Step | `lr` | Fraction of peak |
|---|---|---|
| 1 | `1.75e-7` | 1/4000 |
| 1000 | `1.75e-4` | 1/4 |
| 4000 | `6.99e-4` | **peak** |
| 8000 | `4.94e-4` | 1/√2 |
| 16000 | `3.49e-4` | 1/2 |
| 40000 | `2.21e-4` | 1/√10 |
| 100000 | `1.40e-4` | 1/5 (where the paper's base model stopped) |

Two things set it apart from warmup + cosine:

**1. It doesn't need `total_steps`.** The decay depends only on the current step, so you can train longer without re-planning the schedule. Cosine has to know where training ends before it starts. The flip side is that the `lr` never gets small: after 100k steps it's still a fifth of the peak, so there is no low-`lr` phase at the end for the weights to settle in (the plateau row in [Symptoms](#symptoms)).

**2. You don't set the peak directly.** It falls out of `d_model` and `warmup_steps`:

| `d_model` | `warmup_steps` | Peak `lr` |
|---|---|---|
| 512 | 4000 | `6.99e-4` |
| 512 | 8000 | `4.94e-4` |
| 1024 | 4000 | `4.94e-4` |
| 512 | 1000 | `1.40e-3` |

A wider model gets a smaller peak, which matches "bigger models need smaller values" in [Starting points](#starting-points). But the coupling also means that changing the warmup length changes the peak: doubling `warmup_steps` lowers it by √2. To change one without the other, use the peak form above and pick `peak` yourself.

PyTorch has no built-in Noam scheduler, so write it with `LambdaLR`. `LambdaLR` sets each group's `lr` to *the optimizer's `lr` × the lambda's return value*. Give the optimizer `lr=1.0` so the lambda's value is the learning rate:

```python
from collections.abc import Callable

import torch


def noam_lambda(d_model: int, warmup_steps: int) -> Callable[[int], float]:
    """Build the Noam schedule as a ``LambdaLR`` multiplier.

    Returns:
        A function from the scheduler's step count (starting at 0) to the
        learning rate. Pair it with an optimizer whose ``lr`` is 1.0.
    """

    def lr_lambda(step: int) -> float:
        step += 1  # LambdaLR calls this with 0 at construction, and 0 ** -0.5 raises
        return d_model**-0.5 * min(step**-0.5, step * warmup_steps**-1.5)

    return lr_lambda


optimizer = torch.optim.Adam(
    params=model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer=optimizer, lr_lambda=noam_lambda(d_model=512, warmup_steps=4000)
)
```

The optimizer settings are the paper's. Stepping this 100,000 times reproduces the first table above, and the largest `lr` is `6.99e-4`, used by update 4000.

Gotchas:

- **`lr=1.0` is only the base the lambda multiplies.** Read `optimizer.param_groups[0]["lr"]` to see the real value. If you leave the optimizer at `lr=1e-3`, every `lr` comes out 1000× too small.
- **Keep the `+ 1`.** Without it, building the `LambdaLR` raises `ZeroDivisionError: zero to a negative power`. With it, update *n* uses `lr(n)`.
- **Step it once per batch.** `warmup_steps` counts optimizer updates. `fit` calls `scheduler.step()` once per epoch, so handing it this scheduler would stretch the warmup over 4000 *epochs* (see [Using a Scheduler with `fit`](#using-a-scheduler-with-fit)). Call `scheduler.step()` right after each `optimizer.step()` instead.
- **Size the warmup to the run.** The paper's 4000 steps were 4% of a 100,000-step run. On a run of 3000 steps, `warmup_steps=4000` means training ends before warmup does.

______________________________________________________________________

### ReduceLROnPlateau

Every schedule above is a function of the step count, decided before training starts. `ReduceLROnPlateau` watches a number instead, usually validation loss, and cuts the `lr` when it stops improving:

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, mode="min", factor=0.5, patience=2
)
for epoch in range(epochs):
    train_one_epoch()
    val_loss = evaluate()
    scheduler.step(val_loss)   # the metric is required
```

Feed it validation losses `1.0, 0.9, 0.9, 0.9, …` starting from `lr=0.1`, and the `lr` per epoch is `0.1` ×5, then `0.05` ×3, then `0.025`. It cuts after `patience + 1` epochs without improvement, then resets its counter.

Two things set it apart:

- **`step()` needs the metric.** `scheduler.step()` with no argument raises `TypeError: ReduceLROnPlateau.step() missing 1 required positional argument: 'metrics'`.
- **It's per epoch by nature.** The metric only exists after validation, so stepping it per batch would mean judging "no improvement" on noisy batch losses.

______________________________________________________________________

### Using a Scheduler with `fit`

`TrainState` holds an optional `scheduler`, and `fit` (and `profiled_fit`) calls `state.scheduler.step()` with no arguments **once per epoch**, right after the training pass and before validation. That is the per-epoch row of the cadence table, and it decides what you can hand it:

| Scheduler | With `fit` |
|---|---|
| `StepLR`, `MultiStepLR`, `ExponentialLR` | Works. Size them in epochs |
| `CosineAnnealingLR` | Works with `T_max=num_epochs` |
| `OneCycleLR`, warmup, Noam sized in optimizer steps | **Silently wrong.** Only `num_epochs` steps ever happen |
| `ReduceLROnPlateau` | **Crashes** with the `TypeError` above: `fit` passes no metric, and it steps before validation runs anyway |

The silent case is the dangerous one. A `OneCycleLR(max_lr=0.1, total_steps=3 * len(loader))` run through `fit` for 3 epochs on a 10-batch loader ends with `last_epoch=3` of 30 and `lr=0.0336`: training finishes still in warmup, and nothing warns you.

For a per-batch schedule, write the loop yourself, or build the schedule in epochs. Stepping the scheduler inside a custom `step_fn` doesn't work either, because `fit` still adds its own per-epoch `step()` on top.

______________________________________________________________________

### Checkpointing and Reading the LR

A scheduler's `state_dict()` holds its counters and bookkeeping (`last_epoch`, `base_lrs`, `_last_lr`, plus its own arguments like `step_size`). `_last_lr` records what the scheduler last calculated, but **the operative learning rate** lives in the optimizer's `param_groups`, and the optimizer's `state_dict()` saves it.

So resume needs both. Restore only the scheduler on a fresh optimizer and the counters say "one decay has happened" while the optimizer still runs at the original `lr` (`get_last_lr()` says `0.05`, `param_groups` says `0.1`). Restore both and the resumed run follows the original exactly. `save_state` and `load` in `train_loop.py` already save and restore all three: model, optimizer, scheduler.

Reading the value:

```python
scheduler.get_last_lr()              # [lr for each param group], as of the last step()
optimizer.param_groups[0]["lr"]      # what the next optimizer.step() will use
optimizer.param_groups[0]["initial_lr"]  # added by the scheduler at construction
```

`get_last_lr()` returns a list because each param group is scheduled separately: with groups at `lr=0.01` and `lr=0.1`, one `StepLR(gamma=0.1)` step gives `[0.001, 0.01]`. Differential learning rates survive scheduling.

**Gotcha:** most schedulers are *chainable*. They multiply whatever `lr` is currently in `param_groups`. Set `param_groups[0]["lr"] = 1.0` by hand, and the next `ExponentialLR(gamma=0.5)` step gives `0.5`, not the value the schedule was planning. If you edit the `lr` manually, the scheduler follows your edit.

______________________________________________________________________

### Scheduler Cheat-Sheet

| Question | Answer |
|---|---|
| Order of calls | `optimizer.step()`, then `scheduler.step()` |
| Per batch or per epoch? | Whatever unit the schedule was sized in. It counts `step()` calls, nothing else |
| Simple vision recipe | `MultiStepLR` or `CosineAnnealingLR`, per epoch |
| Transformer | Linear warmup + cosine, or Noam, per batch |
| Don't know when training ends | Noam, or `ReduceLROnPlateau` |
| Using `fit` | Per-epoch schedules only; no `ReduceLROnPlateau` |
| Resuming a run | Load optimizer **and** scheduler state |
| Current `lr` | `optimizer.param_groups[i]["lr"]`; `scheduler.get_last_lr()` is the last value the scheduler calculated |

______________________________________________________________________

### Common confusions: Optimizer and Scheduler

| Easy to mix up | What to remember |
|---|---|
| Optimizer state versus model weights | Adam moments live with the optimizer; learned weights live with the model. Save both to resume training. |
| Scheduler count versus epoch count | A scheduler counts calls to `step()`. The loop gives those calls meaning as batches or epochs. |
| Current `lr` versus `base_lrs` | The optimizer's `param_groups[i]["lr"]` is used by the next update; `base_lrs` is scheduler bookkeeping. |
| Step-based versus metric-based schedules | `ReduceLROnPlateau` needs a validation metric; `fit` currently steps its scheduler before validation. |

### Check your recall: Optimizer and Scheduler

1. If `scheduler.step()` halves the learning rate after update 20, which update first uses the smaller rate?
1. Why can a per-batch warmup silently fail when passed to this project's `fit()`?
1. Why is restoring only scheduler state insufficient to resume at the same learning rate?

<details markdown="1">
<summary>Answers</summary>

1. Update 21. Update 20 already read the old rate when its optimizer step ran.
1. `fit()` steps the scheduler once per epoch, so a schedule parameterized in optimizer updates advances far too slowly.
1. The actual `lr` used by the optimizer lives in its parameter groups. Scheduler counters alone do not restore that value or Adam's optimizer state.

</details>

**One-minute recap:** The model owns parameters, `backward()` fills their gradients, and the optimizer reads those gradients and the current `lr`. A scheduler edits the optimizer's `lr` for later updates. Match scheduler calls to the unit it was configured for, and restore both optimizer and scheduler state when resuming.

______________________________________________________________________

### Sources: Optimizer and Scheduler

- Parameter update in `step()` and the per-parameter `optimizer.state` buffers: [pytorch/torch/optim/optimizer.py](https://github.com/pytorch/pytorch/blob/main/torch/optim/optimizer.py)
- Optimizer takes an iterable of tensors / param-group dicts, `add_param_group`, separate `state_dict`: [torch.optim docs](https://docs.pytorch.org/docs/stable/optim.html)
- Every scheduler above, `SequentialLR`, `ChainedScheduler`, `ReduceLROnPlateau`: [pytorch/torch/optim/lr_scheduler.py](https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py)
- Loshchilov & Hutter, *"SGDR: Stochastic Gradient Descent with Warm Restarts"* (ICLR 2017). Cosine annealing. arXiv:1608.03983
- Smith & Topin, *"Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates"* (2017). The one-cycle policy. arXiv:1708.07120
- Vaswani et al., *"Attention Is All You Need"* (2017), §5.3. The Noam schedule. arXiv:1706.03762

______________________________________________________________________

## Adam and the Learning Rate

**Revision question:** Why can a learning rate that works for SGD fail for Adam, and why does Adam still need a learning-rate schedule?

Think of the gradient as a **local direction**, the learning rate as the **overall step scale**, and Adam's moments as a **per-coordinate adjustment** based on recent gradients. Adam changes how the gradient becomes an update; it does not choose a safe global scale for every model or training phase.

**Reading map:** [The one-dimensional loss](#the-1-d-picture) makes large and small steps visible. [Momentum](#momentum-remembering-the-direction) and [Adam's update](#adam-line-by-line) explain how history changes an update. [AdamW](#adam-vs-adamw-where-weight-decay-goes), [learning-rate selection](#picking-a-learning-rate), and [warmup and decay](#warmup-and-decay-why-adam-wants-them) are reference sections for diagnosing a run. See [optimizer and scheduler](#optimizer--scheduler--the-decoupling) for where these values live in PyTorch.

Every output shown below was produced by running the code with `torch==2.12.1`.

______________________________________________________________________

### What the Learning Rate Controls

The update rule for plain gradient descent is one line:

```
w ← w − lr · ∂L/∂w
```

The gradient says *which way* is downhill. `lr` says *how far* to walk. The gradient is only valid locally, like a slope measured under your feet, so a step that's too long lands somewhere the slope said nothing about.

#### The 1-D picture

Take the simplest loss with a single minimum, a parabola: `L(w) = ½·a·w²`. Its gradient is `a·w`, so one step is:

```
w ← w − lr·a·w  =  (1 − lr·a) · w
```

Every step multiplies `w` by the same factor `(1 − lr·a)`. The whole story is in that factor. With curvature `a = 4`, starting from `w = 1`:

| `lr` | factor `1 − 4·lr` | first 4 steps | What happens |
|---|---|---|---|
| 0.1 | 0.6 | 0.6, 0.36, 0.216, 0.130 | smooth convergence |
| 0.4 | −0.6 | −0.6, 0.36, −0.216, 0.130 | overshoots, flips sign, still converges |
| 0.6 | −1.4 | −1.4, 1.96, −2.744, 3.842 | overshoots further each time: **diverges** |

So there is a hard ceiling: **gradient descent on this parabola converges when `0 < lr < 2/a`**. Steeper curvature means a lower ceiling.

#### Why one global `lr` is a compromise

A real network has millions of directions, each with its own curvature. The **steepest** direction sets the ceiling (`lr < 2/a_max`), and the **flattest** direction then crawls, because its factor `1 − lr·a_min` is barely below 1. The ratio `a_max / a_min` (the *condition number*) is how badly one shared `lr` fits everyone.

That mismatch is the problem both momentum and Adam attack.

#### Symptoms

| What the loss curve does | Likely cause |
|---|---|
| `nan` or `inf` within the first few hundred steps | `lr` may be too high; also check invalid operations and masking |
| Drops, then jumps up and oscillates at a high value | `lr` too high |
| Drops fast, then plateaus above where a smaller `lr` ends up | `lr` too high to settle; decay it (see [Schedules](#warmup-and-decay-why-adam-wants-them)) |
| Decreases in a slow, straight, boring line | `lr` too low |
| Train loss fine, val loss rising | usually overfitting, not the learning rate |

______________________________________________________________________

### Momentum: Remembering the Direction

SGD with momentum keeps a running sum of past gradients and steps along that instead of the raw gradient. PyTorch's form:

```
buf ← μ·buf + grad
w   ← w − lr · buf
```

- In a direction where the gradient keeps the same sign, `buf` grows to `grad / (1 − μ)`. With `μ = 0.9` that's a **10× longer step**, for free.
- In a direction where the gradient flips sign every step (the oscillating, steep direction), the terms cancel and `buf` stays small.

In this constant-gradient picture, momentum speeds up the persistent direction and cancels alternating directions. The steady-state buffer scales as `1/(1 − μ)`: changing `momentum` from 0.9 to 0.99 makes that scale 10× larger. Real gradients change over time, but the `lr` that worked before may no longer be safe.

______________________________________________________________________

### Adam, Line by Line

Adam (Kingma & Ba, 2015) keeps two running averages **per parameter** and uses them to give each parameter its own step size. PyTorch defaults: `lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0`.

```
t ← t + 1
m ← β1·m + (1 − β1)·grad          # 1st moment: smoothed direction
v ← β2·v + (1 − β2)·grad²         # 2nd moment: smoothed squared magnitude
m̂ ← m / (1 − β1^t)                # bias correction
v̂ ← v / (1 − β2^t)                # bias correction
w ← w − lr · m̂ / (√v̂ + ε)
```

These live in `optimizer.state[param]` under the keys `step`, `exp_avg` (`m`), and `exp_avg_sq` (`v`).

#### The key ratio: `m̂ / √v̂`

Read the last line as `w ← w − lr · (a normalized direction)`. That direction is often near `−1` to `1`, but it is not strictly bounded there.

- **Consistent gradient** (the same value `g` every step): `m̂ → g` and `√v̂ → |g|`, so the ratio is `±1`. The parameter moves by exactly `lr`.
- **Alternating gradients** (the sign keeps flipping, mean ≈ 0): `m̂` tends toward 0 but `√v̂` stays at the typical magnitude, so the systematic update is small. Individual noisy steps can still move the parameter.

**Follow two gradients:** start with zero moments and `lr=0.01`. A first gradient of `+2` gives bias-corrected `m̂=2`, `√v̂=2`, and an update of about `−0.01`. If the next gradient is `−2`, the first moment nearly cancels (`m̂≈−0.105`) while `√v̂` remains `2`; the update is only about `+0.00053`. Adam remembers the disagreement between those directions.

So **Adam's step is roughly `lr` times the gradient's persistent direction relative to its recent magnitude.** That has two consequences you'll use constantly.

**1. `lr` scales a normalized update.** With SGD, the step is `lr · grad`, so the right `lr` depends strongly on gradient magnitude. With Adam, a consistent gradient often gives a coordinate update near `lr`, regardless of that gradient's size. Transient updates can exceed `lr`, so it is a scale, not a cap.

**2. Adam is approximately insensitive to scaling the loss** when `ε` is negligible, weight decay is absent, and numeric precision is adequate. Multiply the loss by 100: `grad`, `m`, and `√v` all scale by 100, and the ratio stays nearly unchanged. On a small linear regression, 50 steps:

| | loss × 1 | loss × 100 |
|---|---|---|
| Adam, `lr=1e-2` | final weights match to 4e-8 | ← same |
| SGD, `lr=1e-2` | converging (first weights `≈ [0.42, −1.07]`) | diverged (`≈ 3.8e8`) |

A sudden large gradient after a long run of tiny ones illustrates why `lr` is not a hard cap: the ratio can reach about `(1 − β1)/√(1 − β2)`, or `3.2` with default betas.

#### Why bias correction exists

`m` and `v` start at zero, so early on they are averages of "a few real gradients plus a lot of zeros." Without correction, at step 1:

```
m = 0.1 · g          v = 0.001 · g²
m / √v = 0.1·g / (0.0316·|g|) ≈ 3.16 · sign(g)
```

The very first step would be **3.16× `lr`** on every parameter, exactly when the weights are random and the gradients least trustworthy. Dividing by `1 − β^t` undoes the zero-initialization: at `t = 1`, `m̂ = g` and `v̂ = g²`, so the first step is exactly `lr` in each coordinate. In PyTorch:

```python
w = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
opt = torch.optim.Adam(params=[w], lr=0.01)
w.grad = torch.tensor([0.5, -300.0, 1e-3])   # wildly different magnitudes
opt.step()
# every coordinate moves approximately 0.01, against the sign of its gradient
```

The correction fades as `β^t → 0`, but slowly for `β2 = 0.999`: `1 − 0.999^1000 ≈ 0.63`, so it's still doing real work after a thousand steps.

#### `ε`

`ε` stops the division from blowing up when `√v̂ ≈ 0`. For a parameter whose gradients are much smaller than `1e-8`, `ε` dominates the denominator and Adam turns into plain momentum-SGD with step `lr · m̂ / ε`. The default is fine for almost everything. The original Transformer used `1e-9`.

#### The betas

| Knob | Default | Averaging window ≈ `1/(1−β)` | Raise it when | Lower it when |
|---|---|---|---|---|
| `β1` | 0.9 | ~10 steps | gradients are very noisy (small batches) | you need fast reaction to changing gradients |
| `β2` | 0.999 | ~1000 steps | rarely | training is unstable with loss spikes; the original Transformer used 0.98 |

Lowering `β2` makes `v̂` track recent gradient sizes faster. A sudden gradient spike then gets normalized away sooner instead of producing that worst-case `3·lr` step.

______________________________________________________________________

### Adam vs AdamW: Where Weight Decay Goes

Weight decay pulls every weight toward zero a little each step, which discourages the model from relying on a few huge weights. There are two ways to add it, and in Adam they are **not** the same.

**`Adam(weight_decay=λ)`: L2 penalty, coupled** with the default `decoupled_weight_decay=False`. PyTorch adds `λ·w` to the gradient *before* the moments:

```
grad ← grad + λ·w
... then the normal Adam update
```

The decay term now goes through the `m̂ / √v̂` normalization. Two bad things follow:

- Weights with large gradients (large `√v̂`) get their decay divided away, so they're barely regularized. Weights with tiny gradients get decayed hard.
- A decay term that is tiny compared with `w` still gets normalized up to a full `lr`-sized step.

**`AdamW(weight_decay=λ)`: decoupled** (Loshchilov & Hutter, 2019). The decay is applied straight to the weight, outside the normalization:

```
w ← w · (1 − lr·λ)
... then the normal Adam update, using the raw grad
```

The difference is stark on a single weight `w = 2.0` with an explicit zero gradient tensor, `lr = 0.1`, `λ = 0.01`, after one step:

| Optimizer | `w` after one step | Why |
|---|---|---|
| `Adam(weight_decay=0.01)` | **1.900** | `grad = 0.02`, normalized to a full `lr` step |
| `AdamW(weight_decay=0.01)` | **1.998** | `2.0 · (1 − 0.1·0.01)` |

**Use `AdamW` whenever you want weight decay.** Its PyTorch default is `weight_decay=1e-2`, while `Adam`'s is `0`. `Adam(..., decoupled_weight_decay=True)` is the same algorithm as `AdamW`.

Common practice is to **not decay biases and normalization weights** (LayerNorm's `γ`/`β`). Their role differs from weight matrices, and pulling LayerNorm's scale toward zero can fight the normalization. Use two param groups:

```python
decay, no_decay = [], []
for param in model.parameters():
    (no_decay if param.ndim < 2 else decay).append(param)

optimizer = torch.optim.AdamW(
    params=[
        {"params": decay, "weight_decay": 0.01},
        {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=3e-4,
)
```

`param.ndim < 2` catches biases and LayerNorm parameters (all 1-D) and keeps embedding and linear weight matrices in the decayed group.

______________________________________________________________________

### Picking a Learning Rate

#### Starting points

These are well-trodden starting points, not answers. Always check the loss curve.

| Setting | Starting `lr` | Notes |
|---|---|---|
| Small MLP / CNN, Adam | `1e-3` | The default. What `mnist.py` uses |
| Transformer from scratch, AdamW | `1e-4` to `5e-4`, with warmup | Bigger models need smaller values |
| Fine-tuning a pretrained model | `1e-5` to `5e-5` | You're nudging good weights, not searching |
| SGD + momentum 0.9 (vision) | `0.1` at batch size 256 | Scale with batch size (below) |

For reference, the original Transformer (Vaswani et al., 2017, §5.3) used Adam with `β2 = 0.98`, `ε = 1e-9`, and a schedule ([the Noam schedule](#the-noam-schedule)) that peaks at about `7e-4` after 4000 warmup steps for `d_model = 512`.

#### The LR range test

Instead of guessing, measure (Smith, 2017). Start from a tiny `lr`, multiply it by a constant factor after every batch, and record the loss. Plot loss against `lr` on a log x-axis.

```python
import math

import torch
from torch import nn
from torch.utils.data import DataLoader


def lr_range_test(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    lr_min: float = 1e-7,
    lr_max: float = 1.0,
    num_steps: int = 200,
) -> tuple[list[float], list[float]]:
    """Sweep the learning rate exponentially and record the loss at each step.

    Trains the model in place, so pass a throwaway copy
    (``copy.deepcopy(model)``). The loader must yield at least
    ``num_steps`` batches.

    Returns:
        (lrs, losses), one entry per step; stops early once the loss blows up.
    """
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr_min)
    growth = (lr_max / lr_min) ** (1 / (num_steps - 1))
    lrs, losses = [], []
    for step, (x, y) in zip(range(num_steps), loader):
        lr = lr_min * growth**step
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        lrs.append(lr)
        losses.append(loss.item())
        if not math.isfinite(loss.item()) or loss.item() > 4 * min(losses):
            break
    return lrs, losses
```

Reading the plot:

```
loss
 |---------.
 |          \
 |           \           /
 |            \         /
 |             '--...--'
 +-------------------------- lr (log)
   flat    steep    min   explodes
```

- Flat on the left: `lr` too small to change anything.
- The loss falls fastest in the middle.
- It bottoms out, then explodes.

**Pick roughly 10× below the `lr` at the minimum**, or the point of steepest descent. The minimum itself is already on the edge of instability. Loss is noisy batch to batch, so smooth the curve (for example, an exponential moving average) before reading it.

#### Batch size moves the learning rate

A bigger batch gives a less noisy gradient, so you can afford a bigger step. These are heuristics, and both break down at very large batches:

- **SGD: linear scaling.** Batch ×k → `lr` ×k, plus warmup (Goyal et al., 2017). This is how they trained ImageNet at batch size 8192.
- **Adam: square-root scaling.** Batch ×k → `lr` ×√k (Malladi et al., 2022). Adam already normalizes by gradient magnitude, so it gets less benefit from the reduced noise.

If you change the batch size and keep the `lr`, you have changed the experiment.

______________________________________________________________________

### Warmup and Decay: why Adam wants them

[Optimizer & Scheduler](#the-scheduler) covers *how* schedulers edit `param_groups` and which one to pick. Here is *why* Adam in particular wants one.

**Warmup.** In the first steps, `v̂` is estimated from only a handful of gradients, so the per-parameter step sizes are themselves noisy. Bias correction fixes the *average* size, not the *variance*. Large, badly aimed steps at random initialization can push training into a bad region it never recovers from (Liu et al., 2020, the RAdam paper). Ramping `lr` up over the first few hundred to few thousand steps gives `v̂` time to settle. Transformers need this most. See [Pre-Norm vs. Post-Norm](transformer.md#q-why-is-the-number-of-layers-what-breaks-post-norm).

**Decay.** Late in training you're near a minimum. A large `lr` keeps bouncing around it (the plateau row in [Symptoms](#symptoms)). Shrinking `lr` lets the weights settle into the bottom of the basin.

For how to build these (warmup + cosine with `SequentialLR`, the Noam schedule with `LambdaLR`, and which cadence to step them at), see [Warmup and Decay](#warmup-and-decay-how-to-build-them) in the scheduler guide.

______________________________________________________________________

### Inspecting a Live Optimizer

```python
optimizer.param_groups[0]["lr"]                   # the current base lr (what a scheduler edits)
state = optimizer.state[model.fc.weight]          # buffers populated after the first step()
state["step"], state["exp_avg"], state["exp_avg_sq"]
```

A useful health check is the **update-to-weight ratio**: how much each weight matrix changes per step relative to its size. A rough heuristic from CS231n is around `1e-3`. Far below that suggests `lr` is too low; far above suggests it's too high.

```python
before = {n: p.detach().clone() for n, p in model.named_parameters()}
optimizer.step()
for name, param in model.named_parameters():
    ratio = (param.detach() - before[name]).norm() / before[name].norm()
    print(f"{name:30s} {ratio.item():.1e}")
```

Log this for a few steps at the start of training, not every step. The clone doubles parameter memory.

______________________________________________________________________

### Cheat-sheet: Adam and the Learning Rate

| Question | Answer |
|---|---|
| What does `lr` mean for Adam? | The scale of a normalized update; it is not a strict cap |
| First thing to try | `AdamW(lr=1e-3)` for small models; `3e-4` + warmup for transformers |
| Loss is `nan` | Check invalid operations and masking, then try a lower `lr` or warmup |
| Want weight decay | `AdamW`, not `Adam(weight_decay=...)`; skip biases and norms |
| Changed batch size | Recheck `lr`; √k for Adam and k for SGD are starting heuristics |
| Unsure about `lr` | Run the LR range test, pick ~10× below the minimum |
| Loss plateaus early | Add decay (cosine), or lower `lr` late in training |
| No fixed end to training | Noam: linear warmup, then `lr ∝ 1/√step`; step it per batch |

______________________________________________________________________

### Common confusions: Adam and the Learning Rate

| Easy to mix up | What to remember |
|---|---|
| Gradient magnitude versus Adam update size | The first Adam step with a consistent nonzero gradient is near `lr` per coordinate, regardless of raw gradient scale. Later steps depend on moment history. |
| Bias correction versus warmup | Bias correction removes the zero-initialization bias of the moments; warmup limits the global learning rate while early estimates are still noisy. |
| L2 decay versus AdamW decay | Coupled L2 enters Adam's moments; AdamW shrinks weights separately from gradient normalization. |
| A plateau versus overfitting | A high training loss that will not settle may call for decay; rising validation loss while training improves usually points to generalization. |

### Check your recall: Adam and the Learning Rate

1. Why does doubling a consistent gradient double an SGD step but often barely change an Adam step?
1. At the first Adam update, what does bias correction remove—and what does it leave uncertain?
1. Why does decoupled weight decay behave differently from adding `λ·w` to Adam's gradient?

<details markdown="1">
<summary>Answers</summary>

1. SGD multiplies the raw gradient by `lr`; Adam divides its smoothed gradient by the square root of smoothed squared gradients, so the common scale mostly cancels when `ε` is negligible.
1. It removes the systematic shrinkage caused by initializing `m` and `v` at zero. One gradient is still a noisy estimate of what future gradients will look like.
1. Adding `λ·w` changes Adam's moment estimates and is normalized with the task gradient. AdamW applies a separate weight shrinkage proportional to `lr·λ`.

</details>

**One-minute recap:** For a fixed loss curvature, too large an `lr` overshoots and too small an `lr` crawls. Adam uses gradient history to adjust each coordinate, but its global `lr` still controls overall step scale. Bias correction, warmup, decay, and AdamW solve different problems; choose them by the behavior you need to explain.

______________________________________________________________________

### Sources: Adam and the Learning Rate

- Kingma & Ba, *"Adam: A Method for Stochastic Optimization"* (ICLR 2015). The algorithm, bias correction, and step-size bound. arXiv:1412.6980
- Loshchilov & Hutter, *"Decoupled Weight Decay Regularization"* (ICLR 2019). AdamW. arXiv:1711.05101
- Smith, *"Cyclical Learning Rates for Training Neural Networks"* (WACV 2017). The LR range test. arXiv:1506.01186
- Goyal et al., *"Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"* (2017). Linear scaling and warmup. arXiv:1706.02677
- Liu et al., *"On the Variance of the Adaptive Learning Rate and Beyond"* (ICLR 2020). Why adaptive optimizers need warmup (RAdam). arXiv:1908.03265
- Malladi et al., *"On the SDEs and Scaling Rules for Adaptive Gradient Algorithms"* (NeurIPS 2022). Square-root scaling for Adam. arXiv:2205.10287
- Vaswani et al., *"Attention Is All You Need"* (2017), §5.3. Transformer optimizer settings and the Noam schedule. arXiv:1706.03762
- CS231n notes, *"Neural Networks Part 3: Learning and Evaluation"*. Update-to-weight ratio heuristic. cs231n.github.io/neural-networks-3/
- PyTorch source: `torch/optim/adam.py` (update rule, `decoupled_weight_decay`), `torch/optim/adamw.py` (`weight_decay=1e-2` default)

______________________________________________________________________

## Observability in the Training Loop

**Revision question:** When an epoch is slow, is the delay in loading data, launching work, or running kernels on the accelerator?

An epoch total tells you *that* training is slow. A step timeline gives you the first split: **wait for the next batch → run the step → record the result**. A profiler can then show where time went inside that step. The distinction matters because accelerator work is often queued asynchronously.

**Reading map:** Start with [what is already measured](#whats-already-there) and [how to time a setup block](#timing-setup-work-with-log_elapsed), then [the timing trap](#the-async-gotcha-that-invalidates-naive-timing). [Profiler versus structured telemetry](#profiler-vs-slog--why-theyre-not-the-same-tool) gives the tool choice. The four paths below are design sketches for adding detail, with [the recommendation](#recommendation) as a quick summary.

### What's Already There

`train_loop.py` ships with epoch-level telemetry out of the box:

- `epoch_duration_seconds` — `time.perf_counter()` brackets the training pass for an epoch, before validation ([train_loop.py:154-169](../app/src/training/train/train_loop.py#L154-L169))
- `collect_system_metrics()` — CPU%, RAM used, and GPU memory; CUDA peak stats reset at epoch start and may also include validation before collection ([environment.py:8-18](../app/src/training/train/environment.py#L8-L18))
- Three output sinks: loguru (human-readable), wandb (optional), history JSON (machine-readable)

For work outside the loop, [`log_elapsed`](../app/src/training/utils/timing.py)
reports a labelled block's elapsed time to loguru. The Multi30k loader uses it to
time downloads and BPE training. These INFO lines are separate from the epoch
records saved in history JSON.

So the gap is everything **below the epoch boundary**: per-step throughput, the forward/backward/optimizer split, and whether the accelerator is idling while the DataLoader does CPU work.

#### Timing setup work with `log_elapsed`

```python
from training.utils.timing import log_elapsed

with log_elapsed(label="Prepared dataset"):
    prepare_dataset()
```

On successful exit, this logs one INFO line such as `Prepared dataset in 1.2s`.
It measures the block with `time.perf_counter()` and displays seconds to one
decimal place. If the block raises an exception, no duration is logged. Use it
for coarse setup phases such as downloads or tokenizer training; it does not
synchronize CUDA or MPS work, so its output is not a device kernel time.

#### Follow a slow epoch backward

Suppose epoch duration rises but loss is unchanged. First compare **samples per second** across epochs; an epoch total alone may have changed because the number of batches changed. If throughput fell, split a representative step into data wait and step elapsed time. A long wait points toward loading or preprocessing; a long step points toward computation, synchronization, or host overhead. Only a profiler timeline can distinguish these inside a busy step or prove the accelerator was idle.

Two different tools fill different parts of that gap — and conflating them is the source of most confusion in this space.

______________________________________________________________________

### Profiler vs. Slog — Why They're Not the Same Tool

| | Profiler | Slog (structured telemetry) |
|---|---|---|
| **Answers** | Where did my ms go this step? | How is this run trending? |
| **Granularity** | Op/kernel level | Coarse counters (loss, samples/sec, data-wait) |
| **Overhead** | Noticeable during capture | Low when sampled |
| **Run duration** | Bounded window (a few steps) | Always on |
| **Output** | Chrome trace / flamegraph | JSON lines |

A profiler is a microscope. Structured telemetry is a flight recorder. The first diagnoses a short window; the second shows trends across a run. Use the measurements together: a trend tells you *when* to profile, and a trace helps explain *why* that trend appeared.

______________________________________________________________________

### The Async Gotcha That Invalidates Naive Timing

CUDA and MPS work can be **asynchronous**. A `time.perf_counter()` around `loss.backward()` may measure launch time rather than completed device work. Without a synchronization point, that duration cannot be read as kernel compute time.

```python
# MISLEADING on an asynchronous device — may measure launch time only
t0 = time.perf_counter()
loss.backward()
elapsed = time.perf_counter() - t0   # could be microseconds while GPU still runs
```

```python
# CUDA wall-clock section timing — synchronize on both sides
torch.cuda.synchronize()
t0 = time.perf_counter()
loss.backward()
torch.cuda.synchronize()   # blocks until GPU finishes
elapsed = time.perf_counter() - t0
```

For MPS, the corresponding barrier is `torch.mps.synchronize()`; `torch.cuda.synchronize()` applies only to CUDA. PyTorch's profiler uses device timing so you can inspect a trace without placing manual barriers around every operation.

In this repository, `train_step()` calls `loss.item()` after `optimizer.step()`. Reading a device scalar on the CPU synchronizes the queued work needed for that value, so `epoch_duration_seconds` is a meaningful wall-clock measure of the training pass. It already includes the cost of those per-step synchronizations. For a new fine-grained section timer, use appropriate device synchronization, knowing that barriers can reduce throughput.

______________________________________________________________________

### Path 1 — `torch.profiler` Scheduled Window

Concept sketch: profile a bounded window around the `run_epoch` batch loop. Include `ProfilerActivity.CUDA` only when CUDA is in use; other devices need their supported profiler activity and timing path.

```python
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3),
    on_trace_ready=tensorboard_trace_handler("./log/profiler"),
    record_shapes=True,
    with_stack=True,
) as prof:
    for batch_index, batch in enumerate(loader):
        batch_metrics = step_fn(state, config, batch, epoch, batch_index)
        prof.step()
```

Gives you an operator and kernel timeline, including the data-load-versus-compute split when the loader is marked in the trace. Device events account for asynchronous execution; manual `perf_counter()` sections do not.

**Trade-off:** heavyweight — unusable to leave on. A flag-gated diagnostic run, not production telemetry.

**Right when:** you have a concrete "why is this slow / why does it OOM" question.

______________________________________________________________________

### Path 2 — Lightweight Step-Level Slog (recommended)

Concept sketch: time the wait for a batch separately from the step, then report a compact record every N steps. This fits the existing `run_epoch`/`EpochSpec` split: `EpochSpec` can carry a reporting interval, while `run_epoch` owns the timing. The snippets below describe a possible extension; the current `EpochSpec` does not yet have `log_every_n_steps`.

```python
@dataclass
class EpochSpec:
    device: torch.device
    metrics_type: type[Metrics]
    calculate_metrics: MetricsCalculateFn
    accumulate_metrics: MetricsAccumulateFn
    reduce_metrics: MetricsReduceFn
    log_every_n_steps: int = 50       # 0 = disabled
```

Inside `run_epoch`:

```python
step_index = 0
data_wait_start = time.perf_counter()

for batch in loader:
    data_wait_seconds = time.perf_counter() - data_wait_start

    step_start = time.perf_counter()
    batch_metrics = step_fn(state, config, batch, epoch, step_index)
    # train_step() already calls loss.item(), synchronizing the loss read
    step_seconds = time.perf_counter() - step_start

    config.accumulate_metrics(accumulator, batch_metrics)

    if config.log_every_n_steps and step_index % config.log_every_n_steps == 0:
        samples_per_second = batch_metrics.count / (data_wait_seconds + step_seconds)
        logger.info(
            "step {:>5} | loss {:.4f} | {:.0f} samples/s | data wait {:.1f}ms",
            step_index, batch_metrics.loss, samples_per_second, data_wait_seconds * 1000,
        )

    step_index += 1
    data_wait_start = time.perf_counter()
```

The data-wait pattern measures time from the end of one step to the arrival of the next batch. A high value suggests a loading or preprocessing bottleneck, though host-side work in that gap also counts. The displayed throughput includes both the wait and the step; `count / step_seconds` would answer a different question: throughput *while inside the step*.

**Trade-off:** low logging overhead when sampled, but the current `train_step()` already synchronizes once per batch through `loss.item()`. This measurement will not tell you the forward-versus-backward split.

**Right when:** you want a continuous flight recorder and trend lines without the profiler's weight.

______________________________________________________________________

### Path 3 — Manual Section Timing in `train_step`

Concept sketch: time forward and backward separately with device-appropriate synchronization. The CUDA-only snippet below omits optimizer timing and the project's full step signature for clarity.

```python
def train_step(state, config, batch):
    x, y = (tensor.to(device=config.device) for tensor in batch)
    state.optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    t_fwd = time.perf_counter()
    predicted = state.model(x)
    torch.cuda.synchronize()
    fwd_seconds = time.perf_counter() - t_fwd

    loss = state.criterion(predicted, y)

    torch.cuda.synchronize()
    t_bwd = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    bwd_seconds = time.perf_counter() - t_bwd

    state.optimizer.step()
    return loss.item(), y.size(dim=0), {"fwd": fwd_seconds, "bwd": bwd_seconds}
```

**Trade-off:** barriers around every section can measurably slow training. They also make the step interface carry timing data. This sketch gives only two numbers and no kernel detail.

**Right when:** you just want a rough forward/backward ratio and don't mind the overhead.

______________________________________________________________________

### Path 4 — A `Telemetry` Callback Protocol

Define a protocol and inject it via `EpochSpec`, unifying the three sinks you already have:

```python
from typing import Protocol

class Telemetry(Protocol):
    def on_batch_end(self, step: int, loss: float, n: int, elapsed: float) -> None: ...
    def on_epoch_end(self, epoch: int, record: dict) -> None: ...

class NullTelemetry:
    def on_batch_end(self, *args, **kwargs): pass
    def on_epoch_end(self, *args, **kwargs): pass

class JsonlTelemetry:
    def on_batch_end(self, step, loss, n, elapsed):
        print(json.dumps({"step": step, "loss": loss, "samples_per_s": n / elapsed}))
    def on_epoch_end(self, epoch, record):
        print(json.dumps({"epoch": epoch, **record}))
```

`fit` stops knowing about wandb specifically — `WandbTelemetry` becomes just another impl.

**Trade-off:** a clean extension seam, but it touches the working wandb path and adds an abstraction to learn. It becomes useful when several signals or sinks need the same lifecycle events.

**Right when:** you expect to keep adding sinks/signals and want one extension point.

______________________________________________________________________

### Recommendation

Treat them as two separate jobs:

1. **Always-on slog → Path 2.** Slots into the seam already built (`EpochSpec` callbacks), stays cheap via sampling, reveals the one thing epoch telemetry hides — throughput and DataLoader stall.
1. **Diagnostic profiler → Path 1, flag-gated.** Keep as an opt-in window for a few steps when something's slow. Don't entangle it with the always-on path.

Hold Path 4 in reserve until several signals or sinks need the same events. Use Path 3 only for a short diagnostic when a profiler is unavailable or unnecessary.

______________________________________________________________________

### Common confusions: Observability

| Easy to mix up | What to remember |
|---|---|
| Host elapsed time versus device compute time | A host clock can stop while queued accelerator work is still running. Synchronize or inspect device events for section timing. |
| Step throughput versus end-to-end throughput | Excluding data wait can make compute look fast while the training run stays slow. State which interval the denominator covers. |
| Low GPU utilization versus a known cause | Utilization shows a symptom. Data-wait timing and a profiler trace help locate the gap. |
| Logging interval versus synchronization interval | Sampling log output does not remove synchronizations that already happen every step through `loss.item()`. |

### Check your recall: Observability

1. Why can a host timer around `backward()` report a short duration while the accelerator is still busy?
1. If data wait is 80 ms and step time is 20 ms, what does `count / step_seconds` hide?
1. When should you move from sampled step telemetry to a profiler trace?

<details markdown="1">
<summary>Answers</summary>

1. The host may only have queued asynchronous kernels. A device barrier or profiler event is needed to account for their completion.
1. It excludes the 80 ms spent waiting for data, so it overstates end-to-end throughput by a factor of five in this example.
1. When a trend shows a slow step or memory problem but coarse timing cannot identify which operation, kernel, or gap is responsible.

</details>

**One-minute recap:** Epoch timing tells you the size of a problem; sampled data-wait and step timing tell you where to look; a short profiler trace shows the operations behind it. On asynchronous devices, interpret host timers only after accounting for synchronization, including the `loss.item()` call already present in this loop.
