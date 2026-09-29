# Optimizer & Scheduler — the Decoupling

**Revision question:** When the learning rate changes, which object changed—and which weight update will see the new value?

[training_loop.md](training_loop.md) showed the five operations in a batch. The key relationship here is **model parameter → optimizer parameter group → scheduler**. The model owns parameters; the optimizer keeps references to them and their update state; the scheduler keeps a reference to the optimizer and edits its learning-rate field. The loop decides when each operation happens.

**Reading map:** [Where parameters live](#where-parameters-live) through [the scheduler](#the-scheduler) explains the shared state. [Ordering and cadence](#ordering-rule-and-cadence) determines which learning rate an update uses. Use [the built-in schedules](#the-built-in-schedules), [warmup and decay](#warmup-and-decay), and [the `fit` compatibility table](#using-a-scheduler-with-fit) as references once that picture is clear.

______________________________________________________________________

## Where Parameters Live

Parameters are owned by the **model**. Writing `self.fc = nn.Linear(10, 2)` registers `fc` as a child module; that `Linear` module registers its own `weight` and `bias` as parameters. Modules form a tree, and `model.parameters()` walks it recursively.

The tensor *data* is just a buffer in CPU/GPU memory. The model holds a Python *reference* to it. That distinction is the whole key.

```python
model.fc.weight        # the actual nn.Parameter object
model.parameters()     # iterator over every param tensor (walks the tree)
model.named_parameters()   # ('fc.weight', tensor), ('fc.bias', tensor), ...
model.state_dict()     # name -> parameter/buffer tensor, used for saving
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

Same object, two names — ordinary Python aliasing, exactly like `b = a` on a list. For plain SGD without momentum or weight decay, `optimizer.step()` conceptually applies `param ← param − lr·param.grad` without tracking the update in autograd. It changes *the exact parameter* the model uses in its next forward pass. The optimizer never looks up the model. **The shared parameter is the link.**

Each shared tensor carries **two slots**:

- `param.data` — the value (model reads it in forward; optimizer writes it in step)
- `param.grad` — the proposed change (`backward()` writes it; optimizer reads it)

Two parties passing notes through one tensor's two pockets. Neither imports the other; `autograd` is the third party that fills `.grad`, and the optimizer never touches the graph.

### The compute graph is transient

The graph is the *ephemeral* part. Each recorded forward pass builds a fresh DAG (define-by-run); parameters are leaf tensors referenced by graph operations. Ordinarily `backward()` walks the graph, accumulates gradients on used leaf parameters, and releases the intermediate graph. The parameters and their `.grad` fields remain.

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
1. **The interface is `.grad`, not a model method.** `backward()` writes `.grad`; `step()` reads `.grad` and changes parameter values. They coordinate through the shared tensor.
1. **Separately checkpointable state.** `model.state_dict()` is the learned function; `optimizer.state_dict()` is training scaffolding (Adam's `m`, `v`, which are *not* parameters). You often want weights for inference *without* dragging stale optimizer moments along.
1. **Param groups need different policies.** `add_param_group(...)` lets you unfreeze a backbone mid-training or set a lower LR for pretrained layers vs the head — an optimizer concern with no home in the module.

The SE analogy: **`nn.Module` is the data structure; the optimizer is an algorithm operating on it.** You don't bake quicksort into the array. (Keras made the opposite call — `model.compile(optimizer=...)` — trading flexibility for convenience.)

______________________________________________________________________

## What `step()` Actually Does — and Inspecting It

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

## The Scheduler

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

### Follow two updates

Suppose an SGD optimizer starts with `w=2` and `lr=0.1`, and the gradient on each update is `2`. A scheduler with `gamma=0.5` runs **after** each optimizer update:

| Event | Learning rate read by optimizer | Weight after `optimizer.step()` | Learning rate after `scheduler.step()` |
|---|---:|---:|---:|
| First update | 0.10 | 1.80 | 0.05 |
| Second update | 0.05 | 1.70 | 0.025 |

The scheduler did not change `w` directly. It changed the value the *next* optimizer update reads. In a real model the second gradient would usually change with the new weight; holding it at `2` isolates the scheduling effect.

**Why bother:** a schedule can make early updates large enough to move quickly and later updates small enough to settle. The useful shape depends on the optimizer, model, and run length.

### Ordering rule and cadence

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

## The Built-in Schedules

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

### Composing schedules

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

## Warmup and Decay

**Warmup** ramps the LR up from near zero over the first few thousand steps. It matters most for post-norm transformers: the gradient near the output is large at initialization no matter how deep the model is, so a full-size LR at step 0 can make training diverge. See [transformer/norm.md](../transformer/norm.md#q-why-is-the-number-of-layers-what-breaks-post-norm). **Decay** shrinks the LR late in training so the weights can settle instead of bouncing around a minimum.

[adam_and_learning_rate.md](adam_and_learning_rate.md#warmup-and-decay) covers *why* Adam in particular wants both. This section is how to build them.

### Warmup + cosine

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

### The Noam schedule

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

**1. It doesn't need `total_steps`.** The decay depends only on the current step, so you can train longer without re-planning the schedule. Cosine has to know where training ends before it starts. The flip side is that the `lr` never gets small: after 100k steps it's still a fifth of the peak, so there is no low-`lr` phase at the end for the weights to settle in (the plateau row in [Symptoms](adam_and_learning_rate.md#symptoms)).

**2. You don't set the peak directly.** It falls out of `d_model` and `warmup_steps`:

| `d_model` | `warmup_steps` | Peak `lr` |
|---|---|---|
| 512 | 4000 | `6.99e-4` |
| 512 | 8000 | `4.94e-4` |
| 1024 | 4000 | `4.94e-4` |
| 512 | 1000 | `1.40e-3` |

A wider model gets a smaller peak, which matches "bigger models need smaller values" in [Starting points](adam_and_learning_rate.md#starting-points). But the coupling also means that changing the warmup length changes the peak: doubling `warmup_steps` lowers it by √2. To change one without the other, use the peak form above and pick `peak` yourself.

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

## ReduceLROnPlateau

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

## Using a Scheduler with `fit`

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

## Checkpointing and Reading the LR

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

## Scheduler Cheat-Sheet

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

## Common Confusions

| Easy to mix up | What to remember |
|---|---|
| Optimizer state versus model weights | Adam moments live with the optimizer; learned weights live with the model. Save both to resume training. |
| Scheduler count versus epoch count | A scheduler counts calls to `step()`. The loop gives those calls meaning as batches or epochs. |
| Current `lr` versus `base_lrs` | The optimizer's `param_groups[i]["lr"]` is used by the next update; `base_lrs` is scheduler bookkeeping. |
| Step-based versus metric-based schedules | `ReduceLROnPlateau` needs a validation metric; `fit` currently steps its scheduler before validation. |

## Check Your Recall

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

## Sources

- Parameter update in `step()` and the per-parameter `optimizer.state` buffers: [pytorch/torch/optim/optimizer.py](https://github.com/pytorch/pytorch/blob/main/torch/optim/optimizer.py)
- Optimizer takes an iterable of tensors / param-group dicts, `add_param_group`, separate `state_dict`: [torch.optim docs](https://docs.pytorch.org/docs/stable/optim.html)
- Every scheduler above, `SequentialLR`, `ChainedScheduler`, `ReduceLROnPlateau`: [pytorch/torch/optim/lr_scheduler.py](https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py)
- Loshchilov & Hutter, *"SGDR: Stochastic Gradient Descent with Warm Restarts"* (ICLR 2017). Cosine annealing. arXiv:1608.03983
- Smith & Topin, *"Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates"* (2017). The one-cycle policy. arXiv:1708.07120
- Vaswani et al., *"Attention Is All You Need"* (2017), §5.3. The Noam schedule. arXiv:1706.03762
