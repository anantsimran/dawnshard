# Discussion: pickling a model versus saving a state_dict

Raw notes from a design discussion on 2026-11-03, kept as the record behind
[ADR 0001](../adr/0001-self-describing-checkpoints.md). The text is the conversation as
it happened, lightly regrouped under headings and with the chat interface's labels
removed. It is not edited for style, and its claims have not been checked beyond what
the text says. The `torch.export` and `beartype` points are marked in the text as
inference.

The first prompt was: what is needed to reconstruct the model from scratch, without
discussing a KV cache? The idea of a model config dataclass came up, along with how to
save a `model.pt` between epochs, how to load it, whether config plus `model.pt` is
enough, and what to do about the tokenizers. The last question moved to
[the tokenizer discussion](bpe-tokenizer-in-checkpoint.md).

______________________________________________________________________

## The key idea

Pickle saves an object's data plus the name of its class, never the class's code.

### What pickle does

When `torch.save` writes an object, pickle records two things: a pointer to the class,
like `training.model.multi30k_translator.Multi30kTranslator`, and the object's
attributes. When loading, pickle imports that path, creates an empty object without
calling `__init__`, and copies the saved attributes onto it.

### Why saving the whole module is fragile

The file holds the class's name, not its code. `forward` and every other method come
from whatever code exists when you load. Loading breaks when:

- You move or rename the file or class. The import path no longer resolves.
- You change the attributes. Say you add a submodule in `__init__` later: the old pickle
  doesn't have it, and `forward` crashes with an `AttributeError`.
- You trained by running the file directly. That's how your `main` runs, so the class is
  recorded as `__main__.Multi30kTranslator`. Loading from any other script fails.

It's also unsafe. Pickle can be made to call any function while loading, so opening an
untrusted `.pt` file can run arbitrary code.

### "Saving how the module was created"

The whole-module pickle doesn't save that either. It saves the result of `__init__`, not
the recipe. The recipe has two parts:

- the arguments, which is your config dict
- the code, which is the git commit hash stored in the checkpoint

Config, commit hash, and `state_dict` recreate the model exactly. If you want the model
to run without your Python class at all, use `torch.export`, which saves the computation
graph with the weights. That's a deployment step, though, and your `beartype` decorators
and the `attention_map` side effect may get in its way.

______________________________________________________________________

## What `torch.load` does

`torch.save` and `torch.load` use Python's pickle by default. The `.pt` file is a zip
archive: one pickle with the object's structure, plus the raw tensor bytes stored
separately. `torch.load` unpickles the structure, fills in the tensor bytes, and returns
whatever you saved (usually your dict).

### `map_location`

Every saved tensor remembers its device, such as `cuda:0`. Load on a machine without
that GPU and it fails. `map_location` says where to put the tensors instead: `"cpu"`,
your device, or a mapping like `cuda:1` to `cuda:0`.

### `pickle_module`

Lets you swap in a different pickle implementation, like `dill`. You'll almost never
need it. Passing it also turns off the safe default described next.

### What `weights_only` is

It switches `torch.load` to a restricted unpickler. The `weights_only` unpickler rewrites
the stack machine of the unpickler so that no modules or functions are imported and only
certain allowlisted classes and functions (necessary for loading `state_dict`s of plain
tensors) are called.

What's allowed: tensors, ints, floats, strings, bools, `None`, and dicts, lists, and
tuples of those, plus a few torch types such as `dtype` and `device`. Anything that needs
a class reference is rejected. Loading serialized `nn.Module`s is against recommended
best practices and now explicitly fails.

### Why `norm=nn.LayerNorm` fails

Pickle saves a class object as a reference to `torch.nn...LayerNorm`. That's a class
lookup, which isn't on the allowlist, so loading raises an `UnpicklingError`. A dataclass
instance fails for the same reason.

Fixes:

- Simplest: store `"layernorm"` as a string and the config as a plain dict.
- Alternative: allowlist the class with `torch.serialization.add_safe_globals`.

### Adam state for inference

Not needed. Inference needs only the weights, the config, and the tokenizers.

Adam keeps two running averages for every parameter, each the same shape as the
parameter. That's where the 2x comes from. It only matters for resuming training. If you
resume without it, the averages restart at zero and you get a few rough steps. Bias
correction softens that, but the scheduler still needs its own state.

So the first proposal was:

- `best.pt` (for inference): weights, config, tokenizer hashes
- `last.pt` (for resuming): everything

The decision recorded in the ADR is simpler: every checkpoint holds everything.

______________________________________________________________________

## Why unpickling skips `__init__`

Creating an object normally takes two steps, and unpickling does only the first.

Normal creation. `Multi30kTranslator(...)` first calls `__new__`, which allocates a blank
object with no attributes. Then it calls `__init__` on that object, and your code runs:
it builds `self.dropout`, both embeddings, the encoder and decoder stacks, and
`self.pred`, fills the weights with random values, and checks that `d_model` divides
evenly by the head count.

Unpickling. Pickle calls `__new__` to get the blank object and skips `__init__`
completely. Then it copies the saved attributes straight onto the object: the submodules
(each unpickled the same way), the parameter tensors, `src_seq_length`, the `training`
flag, `attention_map`, and so on. None of your `__init__` lines run, so the object ends
up looking like it went through `__init__` only because the saved attributes say so.

Why that causes trouble. Anything `__init__` would do today doesn't happen. Suppose you
later add `self.output_dropout = nn.Dropout(...)` to `__init__` and use it in `forward`.
A model pickled before that change has no such attribute, because the file never had it
and `__init__` never ran to create it. The load succeeds, and the first forward pass
crashes. Checks in `__init__` also don't run on load. `nn.Module` does patch in a few of
PyTorch's own internal attributes when unpickling an older file, but none of yours.

The `state_dict` route works the other way round. You call
`Multi30kTranslator(**config)` yourself, so the current `__init__` runs: it builds every
layer from today's code, fills random weights, and runs its checks. Then
`load_state_dict` overwrites those random values with the saved tensors. The structure
comes from current code and only the numbers come from the file. That's why a mismatch is
caught: with strict loading, a missing or extra key raises an error immediately instead
of a crash later in `forward`.

______________________________________________________________________

## Why running the file directly records `__main__`

This comes from how Python names modules, and it only bites if you pickle the whole
model.

Every class records which module defined it. It's stored in
`Multi30kTranslator.__module__`, and pickle saves that string plus the class name.

A file you run directly isn't called by its path. When you run
`uv run python app/src/training/model/multi30k_translator.py`, Python loads that file as
a module named `__main__`, not `training.model.multi30k_translator`. That's also why
`if __name__ == "__main__":` works. So during that run, `Multi30kTranslator.__module__`
is `"__main__"`, and `torch.save(model)` records the class as
`__main__.Multi30kTranslator`.

Loading from another script then fails. Say you later write `translate.py` and call
`torch.load` there. Pickle looks up `__main__`, but now `__main__` is `translate.py`,
which has no `Multi30kTranslator`. You get PyTorch's well-known error:
`Can't get attribute 'Multi30kTranslator' on <module '__main__'>`. The same thing happens
in a notebook. Running with `python -m training.model.multi30k_translator` doesn't help,
because `-m` also names the module `__main__`.

The usual hack is to import the class into the loading script. That puts the name
`Multi30kTranslator` into that script's `__main__`, so the lookup happens to succeed. It
works, but only by accident.

Only the top-level class is affected. `EncoderBlock`, `DecoderBlock`, and
`SinusoidalEmbedding` are imported from `training.transformer.modules`, so they're
recorded under their real module path.

Two real fixes:

- Saving a `state_dict` avoids the whole problem. It holds only string keys and tensors,
  with no class references, so the name `__main__` never ends up in the file.
- If you ever do pickle a module, define the class in one file and run `main` from a
  different script that imports it. Then `__module__` is the real import path.

______________________________________________________________________

## Follow-up questions

### How can loading a pickle run code?

A pickle file isn't plain data. It's a short list of instructions that `torch.load`
executes to rebuild your objects. Two of those instructions do the work:

- `GLOBAL` imports any name from any module, for example `os.system`.
- `REDUCE` calls whatever was just imported, with arguments stored in the file.

Normal loading uses exactly these two to rebuild objects ("import this class, call it
with these arguments"). Python lets any class choose that callable and those arguments
through a method named `__reduce__`. An attacker defines a class whose `__reduce__`
returns `os.system` plus a shell command, pickles an instance, and renames the file
`model.pt`. When you load it, the command runs. Your machine doesn't need the attacker's
class; the file only refers to `os`.

`weights_only=True` stops this by refusing any `GLOBAL` that isn't on its short
allowlist. `safetensors` (what Hugging Face uses) goes further: the file is just a JSON
header plus raw tensor bytes, with no instructions to run.

### What does `torch.export` do?

`state_dict` saves only the numbers, so you need your class to run them. `torch.export`
saves the computation itself. You give it the model and example inputs, and it produces a
traced graph representing only the tensor computation of the function. That graph is a
flat list of low-level torch operations (`matmul`, `softmax`, `add`, and so on), with the
weights attached.

You save it with `torch.export.save` to a `.pt2` file and load it with
`torch.export.load`. What you get back is callable, and your `Multi30kTranslator` source
isn't needed. It's meant for deployment: compiling the model ahead of time, running it
from C++, or on mobile.

The catch: anything that isn't a tensor operation is decided once, at export time. Tensor
shapes are static by default unless marked dynamic with `dynamic_shapes`. Python
primitives such as `int` and `float` are hardcoded unless you use `SymInt`. And
`torch.export` may fail on untraceable code, requiring rewrites or extra information
(unlike `torch.compile`'s fallback).

### Why might `beartype` and `attention_map` get in the way of export?

Both are things your `forward` does that aren't tensor math. No documentation was found
on how export handles either specifically, so this is a likely outcome, not something
that was checked.

- `beartype` and `jaxtyping`. `@jaxtyped(typechecker=beartype)` wraps `forward` in
  ordinary Python that runs on each call: it parses `"B 2 L"`, compares shapes, and keeps
  track of which size each named dimension got bound to. Export either has to trace
  through that Python or run it once at export time. The shape comparisons can make
  export hardcode your batch size and sequence length into the graph, or fail outright.
  The usual fix is to turn the checks off while exporting.
- `attention_map`. `forward` stores a Python dict on `self` as a side effect. An exported
  program is a pure function: inputs go in, outputs come out. A side effect on a regular
  Python attribute has nowhere to live in that graph, so export either errors or silently
  drops it, and the exported model would have no attention maps. The fix is to return the
  maps as an extra output behind a flag, or to leave them out of the exported version.

None of this matters until you deploy. For training and checkpoints, `state_dict` is
enough.

### Can the saved device be made irrelevant?

The question: can this be remembered, since the model may have been trained on a
different device?

Yes. Always load to CPU (`map_location="cpu"`), whatever device trained the model. Two
reasons it works:

- `load_state_dict` copies values into the model's existing tensors. So build the model,
  move it to your device, then load. The numbers go wherever the model's parameters
  already are, whatever device they were saved from.
- The optimizer does the same. Its `load_state_dict` moves its saved state to each
  parameter's device.

That covers training on a GPU box and loading on a laptop. The alternative is calling
`.cpu()` on everything before saving, which costs a copy every epoch and gains nothing
over the load-side fix.

______________________________________________________________________

## Sources named in the discussion

- PyTorch serialization notes
- PyTorch dev-discuss: `weights_only` default flip
- PyTorch: Saving and loading models
- `torch.export` documentation
- DEV Community: `torch.export` summary
