# Optimizer & Scheduler — the Decoupling

[training_loop.md](training_loop.md) showed the 5-step rhythm. This file answers the *architecture* question underneath it: **there is no god object.** There are three independent things — model, optimizer, scheduler — that share one set of tensors and never reach into each other. Your training loop is the only thing that sequences them.

______________________________________________________________________

## Where Parameters Live

Parameters are owned by the **model**, full stop. Writing `self.fc = nn.Linear(10, 2)` makes `nn.Module.__setattr__` file each `nn.Parameter` into an internal `OrderedDict` (`_parameters`), with submodules nesting recursively. The model is a **tree**; params hang off its nodes.

The tensor *data* is just a buffer in CPU/GPU memory. The model holds a Python *reference* to it. That distinction is the whole key.

```python
model.fc.weight        # the actual nn.Parameter object
model.parameters()     # iterator over every param tensor (walks the tree)
model.named_parameters()   # ('fc.weight', tensor), ('fc.bias', tensor), ...
model.state_dict()     # OrderedDict {name -> tensor}, used for saving
```

______________________________________________________________________

## How the Optimizer "Knows" Which Params to Update

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

Same object, two names — ordinary Python aliasing, exactly like `b = a` on a list. So when `optimizer.step()` runs `param.data -= lr * param.grad`, it mutates *the exact tensor* the model uses in its next forward pass, because it **is** that tensor. The optimizer never looks up the model. **The shared tensor *is* the link.**

Each shared tensor carries **two slots**:

- `param.data` — the value (model reads it in forward; optimizer writes it in step)
- `param.grad` — the proposed change (`backward()` writes it; optimizer reads it)

Two parties passing notes through one tensor's two pockets. Neither imports the other; `autograd` is the third party that fills `.grad`, and the optimizer never touches the graph.

### The compute graph is transient

The graph is the *ephemeral* part. Each forward pass builds a fresh DAG (define-by-run); params are its **leaves** but are not *stored in* it. `backward()` walks the graph to write each leaf's `.grad`, then the graph is freed. The params and their fresh `.grad` remain on the model.

```
model + params:  persist for the whole run            (the storage)
compute graph:   born each forward, dies each backward (the wiring, ephemeral)
.grad:           written by backward, lives on the param, cleared by zero_grad
```

`.grad` lives on the tensor, not in the graph — which is *why* both model and optimizer can see it.

______________________________________________________________________

## Why the Optimizer Isn't Part of `nn.Module`

A model defines a *function*; an optimizer defines an *update rule*. They vary independently (the same ResNet trains under SGD/Adam/LAMB; the same Adam trains any model), so coupling them would multiply combinations. Four concrete reasons:

1. **It operates on a flat view of tensors, not the module tree.** The constructor takes an iterable of tensors or param-group dicts — it can optimize a subset, params spanning *multiple* models (GANs), or a bare `nn.Parameter` in no module at all.
1. **The interface is `.grad`, not each other.** `backward()` writes `.grad`; `step()` reads `.grad` and writes `.data`. They coordinate through the shared tensor.
1. **Separately checkpointable state.** `model.state_dict()` is the learned function; `optimizer.state_dict()` is training scaffolding (Adam's `m`, `v`, which are *not* parameters). You often want weights for inference *without* dragging stale optimizer moments along.
1. **Param groups need different policies.** `add_param_group(...)` lets you unfreeze a backbone mid-training or set a lower LR for pretrained layers vs the head — an optimizer concern with no home in the module.

The SE analogy: **`nn.Module` is the data structure; the optimizer is an algorithm operating on it.** You don't bake quicksort into the array. (Keras made the opposite call — `model.compile(optimizer=...)` — trading flexibility for convenience.)

______________________________________________________________________

## What `step()` Actually Does — and Inspecting It

`step()` alters the **parameters**, never the learning rate. The LR is a value it *reads* from `param_groups`. For vanilla SGD, `step()` is literally `param.data = param.data - lr * param.grad`, under `no_grad` (the update is not tracked).

Adam has **two** kinds of "rate," and only one is the scheduler's business:

- **Per-parameter adaptation** (`m`, `v` buffers in `optimizer.state[param]`) — automatic; each param gets its own effective step from its gradient history.
- **The global base LR** (`lr` in `param_groups`) — one number multiplying everything. *This* is what a scheduler rewrites.

So Adam adapts *relative* to a base LR; the scheduler moves the base. Two knobs, two objects.

```python
opt.param_groups[0]['lr']     # the global base LR (what the scheduler edits)
opt.state[model.weight]       # Adam's m, v for that one param
```

**Gotcha:** `optimizer.state` is **empty until after the first `step()`** — the buffers are created lazily on the first update.

______________________________________________________________________

## The Scheduler

A scheduler adjusts the **learning rate** over time. It does *not* touch parameters — same shared-state pattern, one level up. It wraps the optimizer and, on `scheduler.step()`, writes a new value into `optimizer.param_groups[i]['lr']`:

```
scheduler.step()   →  writes  param_groups[i]['lr']
optimizer.step()   →  reads   param_groups[i]['lr'],  writes  param.data
```

So the full picture is three layers of one-directional message-passing, each through a shared slot, no object reaching into another:

```
scheduler →[lr in param_groups]→ optimizer →[.data on param]→ model
                                 optimizer ←[.grad on param]← backward(graph)
```

and your loop is the conductor making them fire in order.

**Why bother:** big steps when far from a good solution, small steps as you close in — coarse-to-fine. A high constant LR converges fast then bounces; a decaying LR lets it settle.

### Ordering rule and cadence

Since PyTorch 1.1, call `optimizer.step()` **before** `scheduler.step()`. *How often* you call the scheduler is set by the schedule's design, not by SGD vs batch GD:

| Cadence | Schedules | Call `scheduler.step()` |
|---|---|---|
| **Per epoch** | `StepLR`, `CosineAnnealingLR` | once per epoch |
| **Per batch** | `OneCycleLR`, warmup | once per optimizer step |

Batch-cadence schedules must be *sized* to total steps, e.g. `OneCycleLR(opt, total_steps=epochs * len(loader))`. Rule of thumb: **the number of `scheduler.step()` calls must match how the schedule was parameterized.**

With mini-batches an epoch is many updates, so per-batch ≠ per-epoch and you must choose. (Under full-batch GD there's one update per epoch, so the distinction vanishes.)

```python
for epoch in range(epochs):
    for batch in loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()       # update weights with the current lr
    scheduler.step()           # set the lr for the next epoch (epoch cadence)
```

______________________________________________________________________

## Sources

- Parameter update in `step()` and the per-parameter `optimizer.state` buffers: [pytorch/torch/optim/optimizer.py](https://github.com/pytorch/pytorch/blob/main/torch/optim/optimizer.py)
- Optimizer takes an iterable of tensors / param-group dicts, `add_param_group`, separate `state_dict`: [torch.optim docs](https://docs.pytorch.org/docs/stable/optim.html)
