# Dawnshard

**A top-down–designed ML infrastructure library.**

Dawnshard is where machine-learning infrastructure and software engineering meet.
Most ML code grows bottom-up — a script becomes a notebook becomes a tangle of
globals. Dawnshard goes the other way: infra is *designed*, with clean seams
between configuration, state, and computation, and engineered like a real
codebase (typed, tested, linted, containerized).

The long-term goal is to build toward **Constitutional AI and RLHF** on top of
this foundation. The training loop, observability, and reproducibility primitives
here are the groundwork for that. Today the library ships with MNIST reference
models so every primitive is exercised end-to-end.

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

## Why the training loop is different

Most PyTorch training loops are a single class that owns the model, the
optimizer, the device, the metrics, and the loop body all at once. Dawnshard
takes a **functional** approach instead — because `nn.Module` already gives us
dependency injection, an extra object-oriented layer buys nothing.

The loop is split into three clean concerns ([train_loop.py](app/src/train/train_loop.py)):

| Concern | Type | What it holds |
|---|---|---|
| **State** | `TrainState` | model, optimizer, criterion, scheduler — the things that mutate |
| **Config** | `RuntimeConfig` | device + injected loss `accumulate`/`reduce` functions — the policy |
| **Function** | `fit`, `run_epoch`, `train_step`, `eval_step` | pure-ish functions that take state + config + batch |

Because the step function is just a `Callable`, the *same* `run_epoch` drives both
training and evaluation — you pass `train_step` or `eval_step`. Loss reduction is
injected too (`accumulate_loss` / `compute_reduced_loss` in
[mean_loss.py](app/src/train/mean_loss.py)), so swapping how a metric is computed
never touches the loop. Checkpointing, history, and run metadata are separate
functions you opt into, not lifecycle hooks you inherit.

```python
runtime_config = RuntimeConfig(
    device=DEVICE,
    accumulate_loss=accumulate_loss,
    compute_reduced_loss=compute_reduced_loss,
)
train_state = TrainState(
    model=model,
    optimizer=torch.optim.Adam(params=model.parameters(), lr=1e-3),
    criterion=nn.CrossEntropyLoss(),
)
fit(state=train_state, config=runtime_config, train_loader=..., val_loader=...,
    num_epochs=15, val_epoch_list=[5, 10, 15])
```

Every run also captures the git commit, a serialized snapshot of state + config,
and per-epoch system metrics (CPU%, RAM, GPU peak memory) into a history JSON —
so any result is reproducible and any run is comparable.

______________________________________________________________________

## What's inside

- **Functional training loop** with checkpointing, resumable state, and
  reproducible per-run history — [app/src/train/](app/src/train/)
- **MNIST reference models** — simple MLP, deep MLP, and a CNN, all runtime
  shape-checked with `jaxtyping` + `beartype` — [app/src/model/mnist.py](app/src/model/mnist.py)
- **Visualization scripts** — plot a run's metrics, compare two runs, or render a
  model's autograd graph — [app/src/viz/](app/src/viz/)
- **Observability** — loguru for humans, optional Weights & Biases, history JSON
  for machines — built into the loop
- **Reproducible environment** — `uv` for dependency management, Docker for
  prod/dev parity, everything driven by `uv run`

______________________________________________________________________

## Setup

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

`UV_ENV_FILE=".env"` is what makes imports like `from train.train_loop import fit`
resolve — [.env](.env) sets `PYTHONPATH=app/src`, and `uv run` loads it for every
command.

### Weights & Biases is optional

W&B logging is **off by default** — `main()` in
[mnist.py](app/src/model/mnist.py) gates it behind `SHOULD_LOG_WANDB = False`, and
`fit(...)` simply skips logging when no run is passed. To enable it, set
`WANDB_API_KEY` and flip the flag. To stay fully offline, leave the flag `False`
(or run `wandb disabled` / `export WANDB_MODE=disabled`).

______________________________________________________________________

## Running things

Everything runs through `uv run` — never a bare `python`. With `UV_ENV_FILE` set,
`PYTHONPATH=app/src` is loaded automatically.

```bash
# Train the MNIST model (CNN by default — edit main() to pick a model)
uv run python app/src/model/mnist.py

# Plot one run's per-epoch metrics
uv run python app/src/viz/plot_metrics.py app/history/<run-id>.json

# Compare two runs on the same axes
uv run python app/src/viz/compare_runs.py app/history/<run-a>.json app/history/<run-b>.json

# Run the test suite
make test
```

**Device selection is automatic** ([setup.py](app/src/setup.py)): CUDA → MPS →
CPU, in that order. Force CPU with `DISABLE_GPU=1`.

### The three MNIST models

All three share the same `(batch, 1, 28, 28) → (batch, 10)` signature and are
swappable in `main()` ([mnist.py](app/src/model/mnist.py)):

| Model | Architecture |
|---|---|
| `MNISTClassifier` | Flatten → 784→128 → ReLU → 128→64 → ReLU → 64→10 |
| `DeepMNISTClassifier` | Flatten → 784→256 → ReLU → 256→128 → ReLU → 128→10 |
| `ConvolutionalMNISTClassifier` | Conv(1→16) → Pool → Conv(16→32) → Pool → Linear→10 |

### Visualizing a model's graph

[model_graph.py](app/src/viz/model_graph.py) renders the autograd graph
(`torchviz`) to an HTML file and opens it in your browser:

```python
from viz.model_graph import visualize_model
from model.mnist import ConvolutionalMNISTClassifier

visualize_model(model=ConvolutionalMNISTClassifier(), input_shape=(1, 1, 28, 28))
```

For a quick layer-by-layer parameter table, `torchinfo.summary(...)` is the
faster check — see the module docstring for both.

______________________________________________________________________

## Docker

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

______________________________________________________________________

## Make targets

The [Makefile](Makefile) is the single source of truth for common workflows, so
you never have to remember the underlying flags:

| Target | What it does |
|---|---|
| `make test` | Run pytest (`make test DIR=app/tests/foo` to scope it) |
| `make precheck` | `pyright` + `ruff check --fix` + `ruff format` + `mdformat` — run before every PR |
| `make check-named-args` | Enforce the keyword-argument rule (see [CLAUDE.md](CLAUDE.md)) |
| `make docker-build*` / `make docker-run*` | Build and run the Docker images (above) |
| `make nb-to-py NB=…` / `make py-to-nb PY=…` | Convert between notebooks and scripts |

______________________________________________________________________

## Tutorials

The [tutorials/](tutorials/) directory is a learning track that builds the same
concepts the library uses, from the ground up:

- [tutorials/pytorch/](tutorials/pytorch/) — a PyTorch series: tensors, autograd
  and gradient descent, the `nn.Module` and multi-layer networks, data loading,
  optimizers and schedulers, the training loop, and best practices for inspecting
  a model.
- [tutorials/cnn.md](tutorials/cnn.md) — convolutional networks for MNIST.
- [tutorials/observability.md](tutorials/observability.md) — what the training
  loop measures for free, and the difference between a profiler and structured
  telemetry.

______________________________________________________________________

## Project structure

```
dawnshard/
├── app/
│   ├── src/
│   │   ├── model/            # MNIST classifiers (MLP, deep MLP, CNN)
│   │   │   └── mnist.py
│   │   ├── train/            # the functional training loop
│   │   │   ├── train_loop.py # TrainState, RuntimeConfig, fit/run_epoch/step
│   │   │   └── mean_loss.py  # injectable loss accumulate/reduce functions
│   │   ├── dataload/         # dataset cache location + helpers
│   │   ├── viz/              # plot_metrics, compare_runs, model_graph
│   │   ├── utils/            # git commit + state/config serialization
│   │   ├── setup.py          # device selection + wandb init
│   │   └── constants.py      # repo-root-relative paths
│   ├── tests/                # pytest suite
│   ├── history/              # per-run history JSON (metrics + metadata)
│   └── datasets/             # downloaded dataset cache (gitignored)
├── tutorials/                # PyTorch learning track + CNN + observability
├── scripts/
│   └── check_named_args.py   # custom keyword-argument linter
├── Dockerfile                # uv-based prod/dev image
├── Makefile                  # test / precheck / docker / notebook targets
├── pyproject.toml            # deps, ruff, pyright, pytest config
└── .env                      # PYTHONPATH=app/src (loaded via UV_ENV_FILE)
```

______________________________________________________________________

## Conventions

- **Keyword arguments everywhere.** A custom linter
  ([scripts/check_named_args.py](scripts/check_named_args.py)) enforces it;
  suppress with `# noqa: NAR001` only where keywords are genuinely unavailable.
  See [CLAUDE.md](CLAUDE.md).
- **Runtime tensor shape checking via `@jaxtyped(typechecker=beartype)`.**
  `jaxtyping` provides the annotation syntax —
  `Float[torch.Tensor, "batch 1 28 28"]` — and `beartype` enforces it at
  call time. Every `forward()` in [app/src/model/mnist.py](app/src/model/mnist.py)
  is decorated this way: a shape mismatch raises immediately at the call site
  instead of surfacing as a cryptic downstream error.
- **Run `make precheck` before opening a PR.** Type-checking, linting,
  formatting, and markdown formatting all have to pass.
