# ADR 0001: Save checkpoints as a plain dict, never as a pickled model

Status: accepted, 2026-11-03

Related: [model_config_spec.md](../../model_config_spec.md) (requirements),
[the pickle discussion](../discussions/checkpoint-pickle-vs-state-dict.md) (reasoning),
[the tokenizer discussion](../discussions/bpe-tokenizer-in-checkpoint.md) (not decided)

## Context

`save_state` writes the model, optimizer, and scheduler `state_dict`s. To use one of
those files you also need the constructor arguments, which live in a constant block and
in an ablation module, and you need to build a `TrainState` by hand before `load` can
fill it. The file does not say how to rebuild what it holds.

Saving the whole `nn.Module` with `torch.save(model)` looks like the shortcut, and it
fails in ways that are hard to see:

- Pickle stores the class's import path and the object's attributes, never its code.
  `forward` comes from whatever code exists at load time.
- Unpickling creates an empty object and copies attributes onto it. `__init__` never
  runs, so an attribute added to `__init__` after the save is missing, and the failure
  appears in the first `forward` call. The checks in `__init__` do not run either.
- A model trained by running its file directly records its class as
  `__main__.Multi30kTranslator`. Loading from any other script fails.
- Pickle can call any function while loading, so an untrusted `.pt` file can run code.
  `torch.load` defaults to `weights_only=True` for this reason, and under that default a
  pickled module or a dataclass instance raises.

Resuming a run also needs more than the three `state_dict`s. The epoch counter, the
seeds, and the random number generators have to come back, or a resumed run draws
different batches and different dropout masks than an uninterrupted one.

## Decision

A checkpoint is one file written with `torch.save` on a plain dict. The module is never
pickled. The dict holds:

- the model `state_dict`
- the config, as a plain dict of JSON-native values
- the optimizer `state_dict`
- the scheduler `state_dict`, when the run has a scheduler
- the epoch
- `train_seed` and `run_uuid`
- the RNG states: torch, CUDA, and Python `random`

Every checkpoint holds all of it. There is no separate weights-only file for inference.

The config is a frozen dataclass in code. Its fields mirror the model's constructor
parameters, so the dataclass is the one place that maps config to constructor. The
checkpoint stores `dataclasses.asdict(config)`, and a field that would otherwise hold a
class holds a string. The norm layer is `"layernorm"` or `"rmsnorm"`, not
`nn.LayerNorm`.

Loading rebuilds the model from the stored config by calling the current constructor,
then calls `load_state_dict` in strict mode. The structure comes from today's code and
only the values come from the file. `torch.load` runs with its default
`weights_only=True` and is never passed `weights_only=False`.

The saved device does not matter. The model is built and moved to the target device
first, `load_state_dict` copies the saved values into those parameters, and
`map_location` handles the tensors in the file. The optimizer's `load_state_dict` moves
its state to each parameter's device in the same way.

## Consequences

Loading fails early. A missing key, an extra key, or a shape that disagrees with the
config raises in `load_state_dict`, where the pickle route would fail later inside
`forward`. Opening a checkpoint cannot execute code.

The file is about three times the size of the weights, because Adam keeps two running
averages per parameter. A run that only wants to translate pays for state it never
reads. Writing one on every checkpoint is the price of not keeping two file types.

The checkpoint records arguments and not code. If the model class changes in a way that
keeps parameter names and shapes, old weights load without an error and compute
something different. The history file stores `git_commit` under the same `run_uuid`, so
the commit is recoverable, but nothing checks it at load time.

A model with a class in its config cannot be saved as it stands. Each such field needs a
string name and a lookup from name to class, which the norm layer already has.

The format covers `state_dict` use only. Exporting a graph that runs without the Python
class (`torch.export`) is a deployment step for later. The `beartype` checks and the
`attention_map` attribute that `forward` sets are likely to need changes first.

## Alternatives considered

Pickle the module with `torch.save(model)`. Rejected for the failure modes in the
context section. It stores the result of `__init__` and not the recipe for it.

Pickle the module and allowlist the classes with `torch.serialization.add_safe_globals`.
This lets `weights_only=True` accept the config's classes, but every new class needs an
allowlist entry, and the code-drift and `__main__` problems remain.

Two files, `best.pt` for inference (weights, config, tokenizer data) and `last.pt` for
resuming (everything). This halves the size of the inference file. It was dropped to
keep one format, one loader, and no rule about which file to open.

Save everything with `.cpu()` applied first. This costs a copy on every save and gives
nothing that loading with `map_location` does not already give.

## Not decided here

- How `fit` writes a checkpoint between epochs. `fit` has no hook for it today, and the
  spec's single save at the end of `main()` no longer covers the decision above. The
  choice reaches the loop's public signature, so the README, the landing page, and the
  trainer-design article move with it.
- Which package captures and restores the epoch, seeds, and RNG states, and so whether
  `train/` stays blind to everything in a checkpoint except what the model module hands
  it.
- Whether the tokenizers go in the checkpoint. The model loads against whatever merges
  file the loader finds, and a wrong match produces wrong output without an error. The
  [tokenizer discussion](../discussions/bpe-tokenizer-in-checkpoint.md) holds the
  options.
- Where the config sits in the dict, and whether the git commit is stored beside it.
- Converting existing checkpoints. They hold no config and stay in the old format.
