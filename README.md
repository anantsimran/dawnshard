# Dawnshard

**A top-down–designed ML infrastructure library.**

Dawnshard is an ML stack built from first principles and engineered like a real
codebase. Each one is typed down to tensor
shapes, tested, linted, containerized, and documented with the reasoning behind it.

Most ML code grows bottom-up: a script becomes a notebook becomes a tangle of
globals. Dawnshard goes the other way. Infra is *designed*, with clean seams between
configuration, state, and computation, and this README explains why each seam sits
where it does.

The long-term goal is to build toward **Constitutional AI and RLHF** on top of
this foundation. The training loop, observability, and reproducibility primitives
here are the groundwork for that.

______________________________________________________________________

## Why the name

In Brandon Sanderson's Cosmere, a **Dawnshard** is one of the ancient Commands
that predate creation itself — a fragment of Adonalsium, the god whose power
shattered into the Shards that now shape every world. Dawnshards are not
weapons or artifacts; they are *access*. A mortal who carries one holds a
sliver of divine creative force — the ability to reshape reality at a
fundamental level.

Deep learning is the same idea, expressed in math. A neural network is a mortal
tool for accessing something that otherwise looks god-like: the ability to
compress, generalize, and understand patterns at scales no human could process
unaided. The library's name is a reminder that the real goal isn't to run
training loops — it's to build the infrastructure that lets humans reach for
that kind of power responsibly.

______________________________________________________________________

## A training run in one screen

Here is the whole loop, for any `nn.Module` and any two `DataLoader`s that yield
`(input, target)` tuples.

```python
train_state = TrainState(
    model=model.to(device=DEVICE),
    optimizer=torch.optim.Adam(params=model.parameters(), lr=1e-3),
    criterion=nn.CrossEntropyLoss(),
)
epoch_spec = EpochSpec(
    device=DEVICE,
    metrics_type=ClassificationMetrics,
    calculate_metrics=calculate_metrics,
    accumulate_metrics=accumulate_metrics,
    reduce_metrics=reduce_metrics,
)
history = fit(
    state=train_state,
    epoch_spec=epoch_spec,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=15,
    val_epoch_list=[5, 10, 15],
)
```

Three objects, three jobs:

- `TrainState` is what training changes: the model, the optimizer, the loss, and an
  optional scheduler.
- `EpochSpec` is the policy: which device, how to score a batch, what to watch.
- `fit` is a function over the two. It returns one `EpochRecord` per epoch and writes
  a history file with the git commit and the machine's CPU, RAM, and GPU usage.

That is the entire API. The rest of this page explains why it is built this way.

______________________________________________________________________

## Separate state, policy, and mechanism

This is the principle the whole repo hangs on, and the part I am proudest of.

Operating-systems people have a name for it: **separation of policy and mechanism**,
from the Hydra kernel in 1974. Mechanism is the code that does the work. Policy is
the set of decisions about how. Keep them apart and you can change either without
touching the other. JAX and Flax add the third piece, **explicit state**: nothing
hides in `self`, and everything the loop mutates is passed in and handed back. Flax
even calls its struct `TrainState`.

Dawnshard applies all three to a PyTorch training loop, and every component maps to
exactly one of them.

### State: `TrainState`

Defined in [model.py](app/src/training/train/model.py). A dataclass with four
fields: `model`, `optimizer`, `criterion`, and `scheduler`. It is the only object the
loop mutates. A checkpoint is its `state_dict`s and nothing else (`save_state`,
`load`), so "what does resuming restore?" has a one-word answer.

### Policy: `EpochSpec`

Same file. It holds `device`, a `Metrics` type, the three functions that score a
batch and reduce an epoch, and two optional probes. The loop reads it and never
writes to it. Swap `ClassificationMetrics` for `Metrics` and the same loop trains a
regressor. Set `train_probe` and the same loop dumps gradient norms. Nothing in
`EpochSpec` knows which model it will drive.

______________________________________________________________________

## Exactly one place for everything

Because the seams are fixed, every tool you might want already has a home. There is
no hidden complexity to be afraid of in this repo.

### Metrics: a function of the outputs

A metric sees the predicted tensor and the target tensor and nothing else. Accuracy, precision, and recall are one
dataclass and three functions in
[classification_metrics.py](app/src/training/metrics/classification_metrics.py).
Mean absolute error would be another dataclass and three more. The loop calls them
on every batch and never learns what they compute.

### Probes: a function of the training state

A probe sees the training state at the one moment everything exists together: the
model, with this batch's gradients still on its parameters during training, the
input, the target, the detached prediction, and where in the run you are. It writes
what it wants to disk or to the log and returns nothing. Attention maps, gradient
norms, activations for a fixed input: same signature, same slot on `EpochSpec`.

### Everything else

- Need to resume? `save_state` and `load`, called where you decide.
- What happened in your run? The history file every run writes on its own.

### No hidden complexity

Nothing here asks you to read the loop, subclass anything, or work out which of a
dozen callbacks fires when. Know the three parts and you know where every tool
goes.

______________________________________________________________________

## The repo keeps itself clean

The goal is a codebase with one way to do each thing and a tool that checks it was
done that way. Here is what that looks like.

### The layout enforces the boundaries

`app/src/training/` has one package per concern, and dependencies point downward,
never sideways.

| Package | Owns |
|---|---|
| `train/` | the loop: `TrainState`, `EpochSpec`, the step functions, `fit`, checkpoints, history |
| `metrics/` | the `Metrics` subclasses and their three functions |
| `transformer/` | attention, the encoder block, masks, probes, axis names |
| `common/` | the BPE tokenizer and the indexed max-heap it trains with |
| `dataload/` | the dataset cache and the AG News pipeline |
| `model/` | the MNIST classifiers and the AG News classifier |
| `viz/` | every plot |
| `utils/` | the git commit and state serialization |

Four rules fall out of that layout:

- `train/` never imports `transformer/`, and `transformer/` never imports `train/`.
  The only modules that import both are the models.
- Paths live in one `constants.py`. Device selection lives in one `setup.py`.
- Everything a run produces lands in `app/history/` or `app/datasets/`, both
  gitignored.
- Tests live in `app/tests/`, one file per module. The tests for `mask.py` are in
  `test_mask.py` and nowhere else.

### Every rule has a tool behind it

| Rule | Enforced by |
|---|---|
| Every command runs through `uv run`. No bare `python`, no `pip`, no `sys.path` edits. | [.env](.env) sets `PYTHONPATH=app/src`, the [Dockerfile](Dockerfile) sets the same `UV_ENV_FILE`, and both run the same `uv sync --frozen` against the same lockfile. Imports resolve identically on a laptop and in a container. |
| Every workflow is a Make target. | The [Makefile](Makefile). Nobody remembers flags, and nobody's local variation drifts. |
| Every function call uses keyword arguments. | [check_named_args.py](scripts/check_named_args.py), an AST linter that fails on any positional argument. `# noqa: NAR001` is the escape for C builtins and `nn.Module.__call__`, where keywords don't exist. |
| Every function has a docstring. | [check_docstrings.py](scripts/check_docstrings.py), installed as a git pre-commit hook. Tests are exempt; nothing else is. |
| Every tensor has a declared shape. | `jaxtyping` annotations enforced at call time by `beartype` on every `forward` and every transformer function. Shape strings come from one set of axis names in [constants.py](app/src/training/transformer/constants.py), so `B`, `L`, `d_model`, `h`, and `d_k` mean the same thing everywhere. A mismatch raises at the call that made it. |
| Every run is recorded. | `fit` writes a history file with the git commit, the optimizer's hyperparameters, and per-epoch resource usage. A result you can't trace to a commit doesn't exist here. |
| Every PR passes the same gate. | `make precheck` runs pyright, ruff, and mdformat. A pre-PR hook ([check_tests_before_pr.py](.claude/hooks/check_tests_before_pr.py)) refuses to open the PR until the named-argument check and the tests pass, and until the README changed whenever `app/src/` did. |

______________________________________________________________________

## Debuggability

A run that goes wrong should be explainable from what it left behind, without running
it again. So every run leaves a trail, and each part of the trail has one job.

### The terminal: one line per epoch

loguru prints the epoch number, the training loss, and the validation loss when
there was a validation pass. That is enough to watch a run from your terminal.

### The history file: everything

Three functions build it:

1. `fit` makes an `EpochRecord` for each epoch: the reduced training metrics, the
   validation metrics if the epoch had a validation pass, the wall-clock duration,
   and a `SystemMetrics` snapshot from psutil and torch with CPU percent, RAM used,
   and peak GPU memory on CUDA or current allocation on MPS.
1. `serialize_epoch_record` flattens that into one dict, hoisting the system fields to
   the top level and dropping anything unset.
1. When `fit` returns, `save_history` writes the records to
   `app/history/<run-uuid>.json` under a `meta` block with the git commit, a summary
   of the `TrainState` (type names and the optimizer's parameter groups, never
   weights), and the device.

This is the run the screenshot below comes from, trimmed to its first epoch and
rounded:

```json
{
  "meta": {
    "git_commit": "a408911da11658f59bd1e36d6d4bfb66f7f3b93f",
    "train_state": {
      "model": "AGNewsClassifier",
      "optimizer": {"type": "Adam", "param_groups": [{"lr": 0.001, "betas": [0.9, 0.999], "weight_decay": 0}]},
      "criterion": "CrossEntropyLoss",
      "scheduler": null
    },
    "train_config": {"device": "mps"}
  },
  "history": [
    {
      "epoch": 1,
      "train_loss": 0.8845,
      "train_metrics": {"loss": 0.8845, "count": 20000, "correct": 12732, "accuracy": 0.6366, "precision": 0.6359, "recall": 0.6366},
      "val_loss": 0.5147,
      "val_metrics": {"loss": 0.5147, "count": 2000, "correct": 1627, "accuracy": 0.8135, "precision": 0.8145, "recall": 0.8142},
      "epoch_duration_seconds": 21.41,
      "cpu_usage_percent": 13.9,
      "ram_used_gb": 9.29,
      "gpu_memory_allocated_gb": 0.056
    }
  ]
}
```

### Plot it locally

`plot_metrics.py` takes one history file, draws every top-level numeric field against
the epoch, and opens the result in your browser. No server, no account, no upload.

```bash
uv run python app/src/training/viz/plot_metrics.py app/history/fd13cf78-a573-465a-b151-55439ff68d36.json
```

![plot_metrics output for the run above: training loss falls every epoch, validation loss bottoms out at epoch 4 and rises, epoch duration and GPU memory stay flat](tutorials/assets/plot_metrics.png)

The turn in the validation-loss panel is the overfitting the numbers showed. The flat
duration and memory panels rule out the run changing character halfway through.

### Compare two runs

`compare_runs.py` takes two history files and draws the metrics they share on the
same axes, one colour per run, labelled by file name. This is how a change gets judged:
run before, run after, compare. Both files stay in `app/history/` with the commit
that produced each.

```bash
uv run python app/src/training/viz/compare_runs.py app/history/<run-a>.json app/history/<run-b>.json
```

### Weights & Biases: the same dict, streamed

Pass `wandb_run=` to `fit` and it logs every epoch's dict, the same one that ends up
in the file, with `wandb_run.log` as the epoch finishes. The keys are identical, so
the dashboard and the file can never disagree. It is off by default. `init_wandb` in
[setup.py](app/src/training/setup.py) reads `WANDB_API_KEY`, each `main()` gates it
behind a flag, and nothing else in the repo knows W&B exists.

### When the numbers aren't enough

The same loop has two more levels. [Probes](#probes-look-inside-the-model) capture
whatever you want from inside the model on every batch, and
[`profiled_fit`](#profiling-is-a-separate-function) writes a Chrome trace of the
epochs you choose, with the dataloader and the step labelled separately.

______________________________________________________________________

## What's inside

| Part | What it is | Where |
|---|---|---|
| Training loop | `fit`, `profiled_fit`, `run_epoch`, `train_step`, `eval_step`; checkpoints, probes, and a history file per run | [train/](app/src/training/train/) |
| Metrics | loss-only, and classification with accuracy and macro-averaged precision and recall, each as three functions the loop injects | [metrics/](app/src/training/metrics/) |
| Attention | scaled dot-product attention, multi-head self-attention with optional QK norm, a pre-norm encoder block, padding and causal masks; design notes and axis names in the [transformer README](app/src/training/transformer/README.md) | [transformer/](app/src/training/transformer/) |
| Tokenizer | byte-level BPE, written from scratch, with an indexed max-heap so each merge updates pair counts in O(log n) | [common/](app/src/training/common/) |
| Data | MNIST through torchvision; AG News sampled, tokenized, and cached | [dataload/](app/src/training/dataload/), [mnist.py](app/src/training/model/mnist.py) |
| Models | three MNIST classifiers (two MLPs and a CNN) and a transformer classifier for AG News | [model/](app/src/training/model/) |
| Visualization | a run's metrics, two runs side by side, a model's autograd graph, attention heatmaps, BPE sequence length against merges | [viz/](app/src/training/viz/) |
| Tests | a pytest suite; `attention` is checked against `torch.nn.functional.scaled_dot_product_attention` | [tests/](app/tests/) |
| Tutorials | a learning track from tensors to attention masks, published as a website | [tutorials/](tutorials/) |

Logs go through loguru, Weights & Biases is optional, and everything runs with `uv`
locally or in Docker.

______________________________________________________________________

## Why the training loop is different

Most PyTorch training loops are a single class that owns the model, the optimizer,
the device, the metrics, and the loop body all at once. Dawnshard takes a
**functional** approach instead. `nn.Module` already gives us dependency injection,
so an extra object-oriented layer buys nothing.

### Design decisions

The loop is split into three concerns ([model.py](app/src/training/train/model.py),
[train_loop.py](app/src/training/train/train_loop.py)):

| Concern | Type | What it holds |
|---|---|---|
| **State** | `TrainState` | model, optimizer, criterion, scheduler — the things that mutate |
| **Config** | `EpochSpec` | device, the metrics type and its functions, optional probes — the policy |
| **Function** | `fit`, `profiled_fit`, `run_epoch`, `train_step`, `eval_step` | functions that take state + config + batch |

#### One loop for training and evaluation

`run_epoch` takes the step function as an argument, so the same loop drives training
with `train_step` and evaluation with `eval_step`. It switches the model into train or
eval mode, walks the loader, and passes each step the epoch number and the batch's
index. `eval_step` runs under `torch.no_grad()`. `fit` runs a training pass every
epoch, a validation pass only on the epochs in `val_epoch_list`, and steps the
scheduler once per epoch.

There is no `Trainer` class and no callback registry. To do something the loop
doesn't, you call a function before or after `fit`, or you add a probe.

#### Metrics are injected

`EpochSpec` holds a `Metrics` dataclass and three functions. `calculate_metrics`
turns one batch's predictions and targets into a `Metrics`, `accumulate_metrics` adds
it into the epoch's accumulator in place, and `reduce_metrics` turns the accumulator
into the epoch's result. Changing how a metric is computed never touches the loop.

The step fills in `loss` and `count` itself, so every `Metrics` carries both; subclass
it to add fields. `run_epoch` builds a fresh accumulator at the start of each epoch by
calling `metrics_type()`. An accumulator needs nothing to start, so the class itself is
the constructor and there is no separate init function.

[loss_only_metrics.py](app/src/training/metrics/loss_only_metrics.py) tracks the loss
only, and [classification_metrics.py](app/src/training/metrics/classification_metrics.py)
adds accuracy and macro-averaged precision and recall. Both weight each batch's loss
by its batch size before averaging, so a smaller last batch doesn't skew the epoch
loss.

`EpochSpec` is generic over its metrics type, written `EpochSpec[M: Metrics]`.
Callable parameters are contravariant, so a function that only accepts
`ClassificationMetrics` is not a `Callable[[Metrics], ...]`. Without the type
parameter, pyright rejects every spec built from a subclass's functions.

#### Probes look inside the model

Some things you want from a run aren't metrics: attention maps, gradient norms,
activations for a few fixed inputs. A probe is a plain function set as `train_probe`
or `eval_probe` on `EpochSpec`. The step calls it on every batch, after the forward
pass, as `probe(model, input, target, predicted, epoch, batch_index)`. `train_step`
calls it after the optimizer step, while that batch's gradients are still on the
parameters.

The probe runs inside the step because that is the only place where the model, the
batch, and its predictions exist together. The forward pass has already happened, so
a probe costs no extra one.

The loop passes `epoch` and `batch_index` instead of building a new probe object each
epoch. A probe that only wants the first batch checks `batch_index == 0`, and it keeps
no state between calls.

A probe returns nothing. Whatever it keeps goes to disk or to the log, never back
through `run_epoch`, so `fit` still returns only metrics and memory doesn't grow with
the number of epochs. Settings such as an output folder are bound with
`functools.partial`, and reading the results is a separate function you call after
`fit`.

A probe that needs values from inside `forward` reads them off the model. `forward`
has to keep returning only what the criterion expects, so `AGNewsClassifier` stores
each block's detached attention weights in `model.attention_map` instead of returning
them. That keeps one batch of weights per block on the device between calls.

#### Every run is recorded

`fit` gives each run a UUID and, when training finishes, writes
`app/history/<run-uuid>.json`. The file holds the git commit, a summary of the state
and config (model, optimizer and its hyperparameters, criterion, scheduler, device),
and one record per epoch with training and validation metrics, epoch duration, and
CPU, RAM, and GPU memory. Any result can be traced back to the code that produced it,
and any two runs can be compared. If you pass a Weights & Biases run, each epoch's
record is logged there as well.

#### Profiling is a separate function

`profiled_fit` takes the same arguments as `fit`, plus `profile_epoch_list`. On those
epochs it wraps the training pass in `torch.profiler.profile`, labels the dataloader
and step phases, and writes a Chrome trace to
`app/history/traces/<run-uuid>/epoch_N.json`. Open it in `chrome://tracing` or
Perfetto. Validation passes aren't profiled.

It writes the normal history file and a second `<run-uuid>_detailed.json`, in which
the profiled epochs also carry dataloader and step CPU time, CUDA time on a GPU, and
the trace path.

Profiling is its own function rather than a flag on `fit` for two reasons. The
profiler slows down every step it records, so you pick which epochs pay for it. And
the regular history file keeps `fit`'s format, so profiled runs compare directly with
normal ones.

#### Checkpoints are functions you call

`save_state` writes the model, optimizer, and scheduler state, and `load` restores
them. `fit` calls neither, so checkpointing happens where you decide, not through a
lifecycle hook.

### Using the loop

This walkthrough goes from a first run to custom metrics, probes, profiling, and
checkpoints. It assumes you have a `model`, a `train_loader`, and a `val_loader`.
Every batch must be an `(input, target)` tuple.

#### Train a model

```python
import torch
from torch import nn
from training.metrics.classification_metrics import (
    ClassificationMetrics,
    accumulate_metrics,
    calculate_metrics,
    reduce_metrics,
)
from training.setup import DEVICE
from training.train.model import EpochSpec, TrainState
from training.train.train_loop import fit

train_state = TrainState(
    model=model.to(device=DEVICE),
    optimizer=torch.optim.Adam(params=model.parameters(), lr=1e-3),
    criterion=nn.CrossEntropyLoss(),
)
epoch_spec = EpochSpec(
    device=DEVICE,
    metrics_type=ClassificationMetrics,
    calculate_metrics=calculate_metrics,
    accumulate_metrics=accumulate_metrics,
    reduce_metrics=reduce_metrics,
)
history = fit(
    state=train_state,
    epoch_spec=epoch_spec,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=15,
    val_epoch_list=[5, 10, 15],
)
```

With `val_epoch_list=[5, 10, 15]`, validation only runs on those three epochs. To use
a learning-rate scheduler, pass `scheduler=` to `TrainState`. For a model that doesn't
classify, import the same three functions from `training.metrics.loss_only_metrics`
and set `metrics_type=Metrics`.

#### Read the history

`fit` returns one `EpochRecord` per epoch and writes them to
`app/history/<run-uuid>.json`. It logs the path when it starts, and you can choose it
with `history_path=`. [Debuggability](#debuggability) shows what the file holds and
what the plots look like. To plot one run, or compare two:

```bash
uv run python app/src/training/viz/plot_metrics.py app/history/<run-uuid>.json
uv run python app/src/training/viz/compare_runs.py app/history/<run-a>.json app/history/<run-b>.json
```

#### Add a metric

Subclass `Metrics` and write the three functions. This one tracks mean absolute
error:

```python
from dataclasses import dataclass

from torch import Tensor
from training.train.model import Metrics


@dataclass
class MAEMetrics(Metrics):
    abs_error: float = 0.0


def calculate_mae(predicted: Tensor, target: Tensor) -> MAEMetrics:
    return MAEMetrics(abs_error=(predicted - target).abs().sum().item())


def accumulate_mae(acc: MAEMetrics, batch: MAEMetrics) -> None:
    acc.loss += batch.loss * batch.count
    acc.count += batch.count
    acc.abs_error += batch.abs_error


def reduce_mae(acc: MAEMetrics) -> MAEMetrics:
    if acc.count == 0:
        return MAEMetrics()
    return MAEMetrics(
        loss=acc.loss / acc.count,
        count=acc.count,
        abs_error=acc.abs_error / acc.count,
    )
```

The step sets `loss` and `count` after `calculate_metrics` returns, so
`accumulate_mae` still has to add them. Pass `metrics_type=MAEMetrics`,
`calculate_metrics=calculate_mae`, and so on to `EpochSpec`.

#### Add a probe

This probe logs the gradient norm on every `every_n`-th training batch:

```python
from functools import partial

from loguru import logger


def log_grad_norm(
    model: nn.Module,
    input_tensor: Tensor,
    target_tensor: Tensor,
    predicted: Tensor,
    epoch: int,
    batch_index: int,
    *,
    every_n: int,
) -> None:
    if batch_index % every_n != 0:
        return
    grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None]
    norm = torch.cat(tensors=grads).norm().item()
    logger.info("epoch {} batch {} | grad norm {:.3f}", epoch, batch_index, norm)


epoch_spec = EpochSpec(
    device=DEVICE,
    metrics_type=ClassificationMetrics,
    calculate_metrics=calculate_metrics,
    accumulate_metrics=accumulate_metrics,
    reduce_metrics=reduce_metrics,
    train_probe=partial(log_grad_norm, every_n=100),
)
```

For a probe that writes to disk and a function that reads the files back, see
`save_attention_maps` in [probes.py](app/src/training/transformer/probes.py) and
[Visualizing attention](#visualizing-attention).

#### Profile a few epochs

```python
from training.train.train_loop import profiled_fit

profiled_fit(
    state=train_state,
    epoch_spec=epoch_spec,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=5,
    val_epoch_list=[5],
    profile_epoch_list=[2],
)
```

This profiles epoch 2 rather than 1, which keeps one-time startup costs out of the
trace, and writes `app/history/traces/<run-uuid>/epoch_2.json`.

#### Save and restore a checkpoint

```python
from training.constants import CHECKPOINTS_PATH
from training.train.train_loop import load
from training.train.utils import save_state

checkpoint_path = CHECKPOINTS_PATH / "model.pt"
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
save_state(state=train_state, checkpoint_path=checkpoint_path)
load(state=train_state, epoch_spec=epoch_spec, checkpoint_path=checkpoint_path)
```

`save_state` doesn't create the folder, hence the `mkdir`. `load` restores into an
existing `TrainState`, so build the model and optimizer first.

______________________________________________________________________

## The BPE tokenizer

The tokenizer is byte-pair encoding, written from scratch. It works on UTF-8 bytes, so
any text can be encoded without an unknown token.

```python
from common import bpe

merges, vocab = bpe.train(text=corpus, num_merges=2000)
ids = bpe.encode(text="Stocks rally as tech earnings beat forecasts", merges=merges)
```

On the AG News train split, 2,000 merges bring the average sample down from 236 bytes
to 77 tokens.

______________________________________________________________________

## Getting started

### Install

Dawnshard uses [`uv`](https://docs.astral.sh/uv/) for everything. Install it
first, then export these three environment variables (add them to your shell
profile):

```bash
# Make `uv run` automatically load the project .env (which sets PYTHONPATH=app/src)
export UV_ENV_FILE=".env"

# Weights & Biases — set your key, or leave wandb disabled (see below)
export WANDB_API_KEY=<your-key>

# Enable Docker BuildKit for the cached, fast image builds
export DOCKER_BUILDKIT=1
```

Then sync dependencies:

```bash
uv sync          # runtime + dev dependencies
uv sync --no-dev # runtime only (matches the production Docker image)
```

`UV_ENV_FILE=".env"` is what makes imports like `from training.train.train_loop import fit`
resolve. [.env](.env) sets `PYTHONPATH=app/src`, and `uv run` loads it for every
command.

### Run something

Everything runs through `uv run`, never a bare `python`. With `UV_ENV_FILE` set,
`PYTHONPATH=app/src` is loaded automatically.

```bash
# Train the transformer classifier on AG News, saving attention maps as it goes
uv run python app/src/training/model/ag_news_classifier.py

# Train the MNIST model (CNN by default — edit main() to pick a model)
uv run python app/src/training/model/mnist.py

# Plot one run's per-epoch metrics
uv run python app/src/training/viz/plot_metrics.py app/history/<run-id>.json

# Compare two runs on the same axes
uv run python app/src/training/viz/compare_runs.py app/history/<run-a>.json app/history/<run-b>.json

# Plot mean BPE tokens per sample against the number of merges (AG News train)
uv run python app/src/training/viz/bpe_seq_len.py

# Run the test suite
make test
```

**Device selection is automatic** ([setup.py](app/src/training/setup.py)): CUDA → MPS →
CPU, in that order. Force CPU with `DISABLE_GPU=1`.

### Weights & Biases is optional

W&B logging is **off by default**. `main()` in [mnist.py](app/src/training/model/mnist.py)
gates it behind `SHOULD_LOG_WANDB = False`, and `fit(...)` simply skips logging when
no run is passed. To enable it, set `WANDB_API_KEY` and flip the flag. To stay fully
offline, leave the flag `False` (or run `wandb disabled` / `export WANDB_MODE=disabled`).

### The three MNIST models

All three share the same `(batch, 1, 28, 28) → (batch, 10)` signature and are
swappable in `main()` ([mnist.py](app/src/training/model/mnist.py)):

| Model | Architecture |
|---|---|
| `MNISTClassifier` | Flatten → 784→128 → ReLU → 128→64 → ReLU → 64→10 |
| `DeepMNISTClassifier` | Flatten → 784→256 → ReLU → 256→128 → ReLU → 128→10 |
| `ConvolutionalMNISTClassifier` | Conv(1→16) → Pool → Conv(16→32) → Pool → Linear→10 |

### Visualizing a model's graph

[model_graph.py](app/src/training/viz/model_graph.py) renders the autograd graph
(`torchviz`) to an HTML file and opens it in your browser:

```python
from viz.model_graph import visualize_model
from model.mnist import ConvolutionalMNISTClassifier

visualize_model(model=ConvolutionalMNISTClassifier(), input_shape=(1, 1, 28, 28))
```

For a quick layer-by-layer parameter table, `torchinfo.summary(...)` is the
faster check. See the module docstring for both.

### Visualizing attention

[attention_heatmap.py](app/src/training/viz/attention_heatmap.py) saves one layer's attention
weights as a grid of heatmaps, with one row per sentence and one column per head. All
cells share a single colour scale so heads can be compared, padding is dropped, and
spaces are drawn as `·` so word boundaries stay visible.

The AG News classifier records these maps during training with a probe:

1. Every `AGNewsClassifier.forward` overwrites `model.attention_map` with each block's
   detached `(B, h, L, L)` weights.
1. `main` sets `eval_probe=partial(save_attention_maps, out_dir=..., num_sentences=4)`
   ([probes.py](app/src/training/transformer/probes.py)), with `out_dir` a fresh folder
   under `app/history/attention_probes/` (`ATTENTION_PROBES_PATH` in
   [constants.py](app/src/training/constants.py)).
1. On each validation epoch, `save_attention_maps` ignores every batch except
   batch 0 and saves its first 4 sentences to `epoch_<NNN>.pt`. Each file holds the
   token ids and one weight tensor per block.
1. After `fit`, `plot_attention_probes` reads those files and writes
   `epoch_<NNN>_block_<i>.png` next to each one.

The validation loader isn't shuffled, so every capture shows the same sentences and
you can compare how attention changes between epochs.

### Docker

The [Dockerfile](Dockerfile) copies `uv` from the official image, uses a BuildKit
cache mount for fast rebuilds, and supports a `DEV` build arg to include or
exclude dev dependencies. Prod/dev parity is the point: the same
`uv sync --frozen` runs everywhere.

```bash
make docker-build           # production image (runtime deps only)
make docker-build-dev       # dev image (adds pytest, ruff, pyright, …)
make docker-run             # shell in the prod image
make docker-run-dev         # shell in the dev image, ./app mounted for live editing
make docker-build-dev-test  # build dev image and run the test suite inside it
```

(Remember `export DOCKER_BUILDKIT=1` so the cache mount is honored.)

### Make targets

The [Makefile](Makefile) is the single source of truth for common workflows, so
you never have to remember the underlying flags:

| Target | What it does |
|---|---|
| `make test` | Run pytest (`make test DIR=app/tests/foo` to scope it) |
| `make precheck` | `pyright` + `ruff check --fix` + `ruff format` + `mdformat`. Run it before every PR |
| `make check-named-args` | Enforce the keyword-argument rule (see [CLAUDE.md](CLAUDE.md)) |
| `make check-docstrings` | Enforce the docstring rule (see [CLAUDE.md](CLAUDE.md)) |
| `make install-hooks` | Run the docstring check as a git pre-commit hook (once after cloning) |
| `make docker-build*` / `make docker-run*` | Build and run the Docker images (above) |
| `make nb-to-py NB=…` / `make py-to-nb PY=…` | Convert between notebooks and scripts |

______________________________________________________________________

## Tutorials

The [tutorials/](tutorials/) directory is a learning track that builds the same
concepts the library uses, from the ground up. Its
[index](tutorials/index.md) lists every page and maps specific questions to the
section that answers them.

- [tutorials/pytorch/](tutorials/pytorch/) — a PyTorch series: tensors, shapes and
  broadcasting, autograd and gradient descent, `nn.Module` and multi-layer
  networks, where the transpose in a linear layer's backward pass comes from, ReLU
  and dead neurons, data loading, and best practices for inspecting a model.
- [tutorials/training/](tutorials/training/) — the training loop, optimizers and
  learning-rate schedulers, Adam and the learning rate, and what the training loop
  measures for free versus a profiler.
- [tutorials/math/](tutorials/math/) — the matrix calculus behind backprop, adapted
  from Parr & Howard with PyTorch checks.
- [tutorials/vision/](tutorials/vision/) — convolutional networks for MNIST.
- [tutorials/transformer/](tutorials/transformer/) — attention masks and
  cross-attention, and pre-norm vs post-norm.

The tutorials and this README are also published as a website at
[anantsimran.github.io/dawnshard](https://anantsimran.github.io/dawnshard/), rebuilt
on every push to `main` by [pages.yml](.github/workflows/pages.yml). Pages keep their
repo paths, so relative links work the same on the site as on GitHub, and links to
source files point at GitHub.

**When you add or change a tutorial, update
[tutorials/index.md](tutorials/index.md), and add a line for a new page to the `nav:`
list in [mkdocs.yml](mkdocs.yml).** Otherwise the page is built but doesn't appear in the
site's navigation. The build prints a warning naming any page left out. To preview
the site locally:

```bash
uvx --from mkdocs==1.6.1 --with mkdocs-material==9.7.7 --with pymdown-extensions==12.0 mkdocs serve
```

______________________________________________________________________

## Project structure

```
dawnshard/
├── app/
│   ├── src/
│   │   └── training/
│   │       ├── train/                # the functional training loop
│   │       │   ├── model.py          # TrainState, EpochSpec, Metrics, Probe, EpochRecord, SystemMetrics
│   │       │   ├── train_loop.py     # fit/profiled_fit/run_epoch/step
│   │       │   ├── environment.py    # per-epoch CPU/RAM/GPU snapshot
│   │       │   └── utils.py          # save_state, save_history
│   │       ├── metrics/              # Metrics subclasses + calculate/accumulate/reduce functions
│   │       ├── transformer/          # attention built from scratch
│   │       │   ├── README.md         # attention design choices and axis names
│   │       │   ├── functions.py      # scaled dot-product attention
│   │       │   ├── modules.py        # MultiHeadAttentionLayer (optional QK norm), EncoderBlock
│   │       │   ├── probes.py         # save_attention_maps: probe that saves attention maps per epoch
│   │       │   ├── mask.py           # padding and causal keep masks, pad-masked mean pool
│   │       │   └── constants.py      # axis names for shape strings: B, L, D_MODEL, H, D_K
│   │       ├── common/
│   │       │   ├── bpe.py            # byte-level BPE: train, encode, decode
│   │       │   └── priority_queue.py # indexed max-heap with O(log n) priority updates
│   │       ├── dataload/             # dataset cache location + AG News pipeline
│   │       ├── model/                # MNIST classifiers; AG News classifier (in progress)
│   │       ├── viz/                  # metrics, run comparison, model graph, attention, BPE length
│   │       ├── utils/                # git commit + state/config serialization
│   │       ├── setup.py              # device selection + wandb init
│   │       └── constants.py          # repo-root-relative paths (CHECKPOINTS_PATH, HISTORY_PATH, TRACES_PATH, ATTENTION_PROBES_PATH)
│   ├── tests/                    # pytest suite
│   ├── history/                  # per-run history JSON; traces/ and attention_probes/ (gitignored)
│   └── datasets/                 # downloaded dataset cache (gitignored)
├── tutorials/                    # learning track: pytorch/, training/, math/, vision/, transformer/; index.md
├── pages/                        # MkDocs hooks and theme overrides for the website
├── scripts/
│   ├── check_named_args.py       # keyword-argument linter
│   └── check_docstrings.py       # docstring-presence linter
├── .claude/hooks/                # pre-PR gate: named args, tests, README touched when source changed
├── .githooks/pre-commit          # runs the docstring check (make install-hooks)
├── .github/workflows/pages.yml   # builds and deploys the website on push to main
├── mkdocs.yml                    # website config and navigation
├── Dockerfile                    # uv-based prod/dev image
├── Makefile                      # test / precheck / lint / docker / notebook targets
├── pyproject.toml                # deps, ruff, pyright, pytest config
└── .env                          # PYTHONPATH=app/src (loaded via UV_ENV_FILE)
```
