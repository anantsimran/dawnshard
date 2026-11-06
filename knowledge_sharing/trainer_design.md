# Where a training loop keeps its decisions

**A change to a training run should land in the object that owns that concern, and
in no other.**

Take a classification run that already works and add one thing: log the global
gradient norm after every optimizer step. The gradients exist only between
`loss.backward()` and the next `optimizer.zero_grad()`, inside the training step.
So the question of where the new code goes is the question of which object owns the
step. In Keras, the step is `Model.train_step`, so the logging means a subclass of
the model. In PyTorch Lightning it is a hook on the `LightningModule` or on a
callback. In the Hugging Face `Trainer` it is a callback on the
`on_pre_optimizer_step` event. Every framework has an answer. They differ in which
object you open to make the change.

## Three concerns in every loop

Operating-systems designers named the split in the Hydra kernel in 1974:
**separation of policy and mechanism**. Mechanism is the code that does the work.
Policy is the set of decisions about how the work is done. JAX and Flax add a third
piece, **explicit state**: the objects a run changes are passed in rather than held
inside a trainer.

A training loop has all three. The state is the model's parameters, the optimizer's
moments, and the scheduler's position. The policy is how a batch is scored, what is
observed, and on which device it runs. The mechanism is the loop itself: iterate the
loader, run the step, accumulate metrics, record the epoch. A new loss or optimizer
changes state. A new metric or probe changes policy. A new way of running epochs,
such as profiling a few of them, changes mechanism. The design question is whether
those three kinds of change touch three different objects.

## Where the frameworks put them

![Five rows, one per framework, across columns for policy, mechanism, and state. Keras is one keras.Model box spanning all three columns. Lightning has a Trainer box for policy and the loop, and a LightningModule box for the step and state. Hugging Face has TrainingArguments for policy and one Trainer box for the loop and state. Flax has a dashed box for the caller's own loop and step, and a TrainState box. Dawnshard has three separate boxes: EpochSpec, the functions fit, run_epoch, and train_step, and TrainState.](assets/trainer-ownership.svg)

*A wide box is one object that holds several concerns. Only the bottom row gives
each concern its own object.*

Keras puts all three on the model. `compile` stores the optimizer, the loss, and
the metrics on the `keras.Model`. `fit` is a method of the same object, and it calls
`Model.train_step` for every batch. The Keras guide to customizing `fit` says to
override the training step function of the `Model` class, so a change to the step is
a change to a model subclass.

PyTorch Lightning splits the work between two classes. The `LightningModule` is an
`nn.Module` that also defines `training_step` and returns its optimizers from
`configure_optimizers`. The `Trainer` owns the loop and the run's flags: precision,
devices, how many batches to accumulate, how often to validate. The step's logic lives
on the model class, and the loop that calls it lives on the trainer.

The Hugging Face `Trainer` holds the model, optimizer, and scheduler, and runs
the loop. Policy arrives as one `TrainingArguments` object. The constructor also
accepts functions: `compute_metrics`, and `compute_loss_func` for a loss that differs
from the model's own. Its callbacks see the run through 16 events, and the
documentation calls them read-only apart from the `TrainerControl` flags they return.
For changes beyond those, the same documentation says to subclass `Trainer` and
override the methods you need.

Flax keeps state explicit. Its `TrainState` holds `params`, the Optax
transformation `tx`, and `opt_state`, and `apply_gradients` returns a new state rather
than mutating the old one. Flax ships no loop. Each project writes its own step and
its own loop around that state.

## Dawnshard keeps the split and ships the loop

Dawnshard takes Flax's explicit state and supplies the loop as plain functions. Each
concern is one object at the call site:

```python
StepFn = Callable[["TrainState", "EpochSpec", Batch, int, int, Optional[int]], "Metrics"]
```

[`TrainState`](../app/src/training/dataclass/model.py) holds the model, optimizer,
criterion, and optional scheduler. These are ordinary PyTorch objects owned by the
caller and changed in place, because PyTorch optimizers update parameters in place.
[`EpochSpec`](../app/src/training/dataclass/model.py) holds the device, a `Metrics` type,
three functions that score and reduce batches, and two optional probes. The loop
reads it and never writes to it. So does `FitConfig`, which holds the epoch count,
the validation epochs, the three seeds, and an optional early-stopping rule. It is one
section of a run's YAML-backed `TrainConfig`, whose other sections name the
architecture, optimizer, scheduler, batch size, and data. The entry point builds the
state and loaders from those sections; the loop never sees them.
[`train_step`, `eval_step`, `run_epoch`, and
`fit`](../app/src/training/train/train_loop.py) take both as arguments and keep no
training state of their own.

Every model in [model/](../app/src/training/model/) subclasses `nn.Module` and
nothing else. No base class connects a model to the loop, so the loop cannot ask the
model for anything beyond a forward pass.

## A change is one argument

The repo's four training entry points cover a CNN, a transformer classifier, a
masked-token predictor, and an encoder-decoder translator. All four pass the same
`nn.CrossEntropyLoss()` as the criterion. What differs between their runs is the
`Metrics` family they put in `EpochSpec` (classification, loss only, or masked-token
accuracy and perplexity), and whether a probe records attention maps. None of them
subclasses anything from the loop's package.

The gradient-norm logging from the opening is a `train_probe`. `train_step` calls it
after `optimizer.step()`, while the batch's gradients are still on the parameters, and
passes it the model, the batch, the detached prediction, the epoch, and the batch
index. Settings such as a logging interval are bound with `functools.partial`. The
model class does not change, and neither does any part of the loop.

## The extension points can be listed

`EpochSpec` has five callables, so the places where caller code runs inside the loop
can be listed by reading one dataclass. `grep epoch_spec. train_loop.py` finds every
line that calls one. A probe returns `None`, and `fit` returns only epoch records, so a
probe can observe a run without steering it. That matches the read-only rule the
Hugging Face documentation sets for its callbacks, with five fields in place of 16
events.

A change of mechanism gets a new function instead of a new hook.
[`profiled_fit`](../app/src/training/train/train_loop.py) wraps the epochs you select
in `torch.profiler.profile` and labels the loader and step phases. It reuses
`train_step` and `eval_step` unchanged and returns the same records as `fit`, with the
profiler's timings beside them.

## The loop runs without its infrastructure

`run_epoch` only iterates its loader, so
[test_train_loop.py](../app/tests/test_train_loop.py) passes it a plain list of
`(input, target)` tuples. A model needs no loop to be tested either:
[test_ag_news_classifier.py](../app/tests/test_ag_news_classifier.py) calls the
classifier's forward pass directly.

The same property makes a sweep an ordinary `for` loop.
[`grid_search`](../app/src/training/ablations/setup.py) takes three builder
functions, for the loaders, the state, and the spec, and calls `fit` once per config.
The 66 comparison runs behind the
[Multi30k ablation report](../app/src/training/ablations/reports/multi30k_ablations.md) went through it.

## The boundaries are tested

The split holds one level up, between packages.
[test_layout.py](../app/tests/test_layout.py) declares which packages each package
under `app/src/training/` may import and fails on any other import, including imports
needed only for type checking. The loop's package may import only `constants` and
`utils`, so `fit` cannot acquire a dependency on a model or a dataset by accident.
The transformer package imports nothing from the rest of the library, so attention
does not know a loop exists. The only modules that import both are the models and
the experiments that drive them.

## What the frameworks offer

The trainers hold several concerns in one object partly because their features need
all three at once. Mixed precision changes the step, the optimizer, and the loss
scale together. Distributed training wraps the model, shards the loader, and changes
when the optimizer steps. A class that owns all three can switch such a feature on
with one flag.

The table lists the features each framework ships and the setting that turns each
one on. The last column is what Dawnshard has today.

| Capability | Keras | Lightning | Hugging Face | Flax | Dawnshard |
|---|---|---|---|---|---|
| Mixed precision | a mixed-precision dtype policy | `precision="bf16-mixed"` | `bf16=True` or `fp16=True` | module `dtype` and `param_dtype` | not built in |
| Several devices | `keras.distribution`, JAX backend only | `strategy="ddp"` or `"fsdp"`, `devices`, `num_nodes` | `fsdp`, `deepspeed` | JAX sharding through `jax.jit` | not built in |
| Gradient accumulation | optimizer `gradient_accumulation_steps` | `accumulate_grad_batches` | `gradient_accumulation_steps` | `optax.MultiSteps` | not built in |
| Gradient clipping | optimizer `clipnorm` or `global_clipnorm` | `gradient_clip_val` | `max_grad_norm`, 1.0 by default | `optax.clip_by_global_norm` | not built in |
| Compilation | `compile(jit_compile=...)` | `torch.compile` the module first | `torch_compile=True` | `jax.jit` | not built in |
| Early stopping | `EarlyStopping` callback | `EarlyStopping` callback | `EarlyStoppingCallback` | `flax.training.early_stopping` | `FitConfig.early_stopping`, counted in validation passes |
| Checkpoints | `ModelCheckpoint` callback | a default `ModelCheckpoint` | `save_strategy`, `load_best_model_at_end` | Orbax, which Flax recommends | `fit` calls `checkpoint` after every validation pass; `load` restores one |
| Learning-rate schedules | a schedule as the optimizer's `learning_rate` | returned from `configure_optimizers` | `lr_scheduler_type`, `warmup_steps` | Optax schedules | `TrainState.scheduler`, stepped once per epoch |
| Validation cadence | `validation_freq` | `val_check_interval` | `eval_strategy` | your loop | `val_epoch_list` |
| Experiment tracking | callbacks such as `TensorBoard` | `logger=` | `report_to=` | your loop | `save_history`, called after `fit` with the run's whole config, plus an optional `wandb_run` |
| Seeding | `keras.utils.set_random_seed` | `seed_everything`, `deterministic=True` | `seed=`, 42 by default | explicit PRNG keys | three required seeds, written to the history file |
| Debugging runs | | `fast_dev_run`, `overfit_batches` | | | `run_epoch` accepts a plain list |

The Hugging Face `Trainer` can also push each saved model to the Hub with
`push_to_hub`, and Lightning runs two validation batches before training by default
(`num_sanity_val_steps`) to catch a broken validation loop early.

Under Dawnshard's split, each missing feature has a known home. Mixed precision,
clipping, and accumulation change what one step does, so each would be a new
`StepFn` beside `train_step`, with a variant of `fit` that runs it, as `profiled_fit`
did for profiling. Compilation and device placement act on the model before the run, which is state the
caller builds.

## Recap

**State** is what a run changes: parameters, optimizer moments, and scheduler
position. In Dawnshard it is `TrainState`, owned by the caller.

**Policy** is the set of decisions the loop reads without changing: scoring,
observation, and device, plus the run's epochs, seeds, and stopping rule. In
Dawnshard that is `EpochSpec` and `FitConfig`.

**Mechanism** is the code that runs the work: the loop and the step. In Dawnshard it
is `fit`, `run_epoch`, `train_step`, and `eval_step`, which take state and policy as
arguments.

**A trainer class** is an object that owns more than one of the three. Keras's
`Model`, Lightning's `LightningModule` and `Trainer`, and the Hugging Face `Trainer`
each do. Flax keeps state separate and leaves the loop to the caller.
