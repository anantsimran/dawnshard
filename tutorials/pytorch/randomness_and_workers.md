# Random Streams and DataLoader Workers — Q&A

Two questions sit under every reproducible data pipeline: where the random numbers come
from, and which process draws them. This page answers both. For how a loader's splits,
samplers, and seeds are wired together, see the
[DataLoader internals](../../app/src/training/dataload/README.md) page.

**Mental model:** a seed initializes a stream; each draw advances that stream. The main process chooses indices for a map-style dataset, while workers use their own streams for random work done while fetching those indices. A repeatable result depends on both the seed and which component consumes each draw.

**Reading route:** use the diagram below to locate the process, then read the [generator](#q-what-is-a-torchgenerator-and-why-use-one) and [worker seed](#q-why-does-the-dataloader-pass-an-rng-seed-to-workers-it-could-just-shuffle-the-indices-and-send-them) questions. The later sections explain process start methods and CUDA.

______________________________________________________________________

## Cheat-Sheet

| You want | Do this |
|---|---|
| One component's randomness isolated from the rest | Give it its own `torch.Generator` and pass `generator=` |
| Different draws from a private generator in a loop | Create `g` once, outside the loop; a new unseeded `torch.Generator()` repeats the same value |
| A private generator that differs every run | `g = torch.Generator()`, then `g.seed()` |
| A reproducible train/val split | `random_split(..., generator=torch.Generator().manual_seed(0))` |
| A reproducible shuffle order | `DataLoader(..., shuffle=True, generator=torch.Generator().manual_seed(0))` |
| Reproducible dropout and weight init | `torch.manual_seed(seed)`; those draw from the global generator |
| To rewind a stream | `state = g.get_state()` … `g.set_state(state)` |
| Fresh random masks every epoch, reproducible per run | Draw from the global generator (`generator=None`) after one `torch.manual_seed` |
| The same masks at every evaluation | A private generator seeded with `seed + index` per sample |
| Validation that doesn't shift training's random stream | Wrap it in `torch.random.fork_rng(devices=[])` |
| CUDA inside a worker | `multiprocessing_context="spawn"`, or better, keep workers on the CPU |

**Where each thing happens:**

```
main process                                   worker process (one per num_workers)
────────────                                   ────────────────────────────────────
Sampler draws a permutation (loader.generator)
  │  indices, grouped into batches
  ▼
index queue ──────────────────────────────────► dataset[i] for each index
                                                 augmentations draw from the worker's
                                                 own seed: base_seed + worker_id
                                                 collate_fn → CPU tensors
result queue ◄──────────────────────────────── batch
  │
  ▼
pin_memory thread (if pin_memory=True)
  │
  ▼
batch.to("cuda")
```

Shuffling happens once, in the main process. Workers only fetch, transform, and collate.

______________________________________________________________________

## Q: After `torch.manual_seed(a)` and one draw, is the seed now `a + 1`?

No. The seed stays `a`. What changes is the generator's **state**, which is a different
thing.

- **The seed** is only the starting point. `manual_seed(a)` uses `a` to initialize the
  state, and `torch.initial_seed()` keeps returning `a`.
- **The state** is the generator's current position in a very long, fixed sequence. On
  the CPU, PyTorch uses the Mersenne Twister (MT19937), whose state is 624 32-bit words.
  On CUDA it uses Philox, which tracks its position as the seed plus an offset.
- **Advancing** means each draw consumes numbers from the sequence and moves the position
  forward. The next call continues from there.

Think of the seed as choosing a book and each draw as reading the next page.
`manual_seed(a + 1)` doesn't turn to the next page. It picks a different book whose pages
have no relation to book `a`.

```python
import torch

torch.manual_seed(seed=42)
x1, x2 = torch.rand(size=(1,)), torch.rand(size=(1,))
print(torch.initial_seed())  # 42: the seed is unchanged

torch.manual_seed(seed=42)  # back to page 1 of book 42
assert torch.equal(x1, torch.rand(size=(1,)))
assert torch.equal(x2, torch.rand(size=(1,)))  # the same sequence again

torch.manual_seed(seed=43)
print(torch.rand(size=(1,)))  # unrelated to x2

state = torch.get_rng_state()
torch.rand(size=(1,))
print(torch.equal(state, torch.get_rng_state()))  # False: the state advanced
```

A pseudo-random generator is a deterministic function, `state → (output, next_state)`.
The seed sets only the first state. After that the state evolves on its own, and nothing
writes back to the seed.

### What "global" means

The global generator is `torch.default_generator`: one shared CPU generator per process.
Every random op called without `generator=` reads from it, including `torch.rand`,
`randint`, dropout, weight init, and `RandomSampler`. `torch.manual_seed(s)` resets it.

In the book picture, the global generator is one shared book on the process's desk. Any
call that doesn't bring its own book reads the next page of the shared one. So a draw
depends on every draw made before it in that process:

```python
torch.manual_seed(seed=0)
a = torch.rand(size=(1,))  # page 1

torch.manual_seed(seed=0)
torch.rand(size=(1,))  # some other code reads page 1 first
b = torch.rand(size=(1,))  # so this is page 2: b != a
```

Adding a layer, a random augmentation, or a debug `torch.rand` shifts every later draw,
including the next shuffle order.

"Global" means global per process. The main process and each worker have their own copy,
which is why workers are reseeded (see [below](#the-seed-is-for-randomness-inside-__getitem__)).
Python's `random` and NumPy's `np.random` are separate global generators, and each CUDA
device has its own default generator.

______________________________________________________________________

## Q: Does a generator store an array of random numbers?

It behaves like one: an endless array you read left to right. But nothing is stored. A
generator holds a small **state** and a **formula**. Each draw computes new numbers from
the state, then overwrites the state.

In the book picture, the book is never printed. It is a recipe, "next page =
recipe(current page)", and the bookmark is the state.

### A toy generator

A linear congruential generator is the simplest example. PyTorch uses a stronger one,
but the structure is the same:

```python
state = 7  # the seed is the initial state


def rand():
    """Advance the state one step and return a float in [0, 1)."""
    global state
    state = (5 * state + 3) % 16  # update: the new state replaces the old one
    return state / 16  # output: the new state, scaled to [0, 1)
```

| Call | Old state | `5 · old + 3` | `mod 16` → new state | Output |
|---|---|---|---|---|
| 1 | 7 | 38 | 6 | 0.375 |
| 2 | 6 | 33 | 1 | 0.0625 |
| 3 | 1 | 8 | 8 | 0.5 |
| 4 | 8 | 43 | 11 | 0.6875 |
| 5 | 11 | 58 | 10 | 0.625 |

The whole generator is one integer. Seed 7 always produces this sequence, which is why
the same seed gives the same book, and no value exists before the call that computes it.

`mod 16` works like a 16-hour clock: 38 hours after midnight is hour 6, so the state
always stays in 0–15. The multiply-and-add jumps around the clock, and the wrap-around
makes the jumps look scattered.

**It eventually repeats.** From 10 the sequence continues 5, 12, 15, 14, 9, 0, 3, 2, 13,
4, then returns to 7: a cycle of 16 that visits every value once. That isn't luck. The
Hull–Dobell theorem says `state = (a · state + c) mod m` reaches the full period `m`
exactly when `c` and `m` share no factor, `a − 1` is divisible by every prime factor of
`m`, and `a − 1` is divisible by 4 if `m` is. Here `c = 3`, `m = 16`, and `a − 1 = 4`
satisfy all three. Real generators keep the structure and stretch the cycle: MT19937's
period is 2¹⁹⁹³⁷ − 1.

### What PyTorch keeps

```python
g = torch.Generator()
print(g.get_state().numel())  # 5056 bytes
saved = g.get_state()
first = torch.rand(size=(10,), generator=g)
torch.rand(size=(10_000_000,), generator=g)
print(g.get_state().numel())  # still 5056: nothing grows or gets used up
g.set_state(saved)  # move the bookmark back
assert torch.equal(first, torch.rand(size=(10,), generator=g))
```

- **CPU: Mersenne Twister (MT19937).** The state is a fixed-size block, and a formula
  scrambles it into each next output. Ten million draws leave its size unchanged, and
  restoring it replays the same numbers, so the state alone decides every future output.
  `get_state` and `set_state` read and overwrite that state, which is why rewinding works.
- **CUDA: Philox.** Philox is counter-based: value `n` is `scramble(seed, n)`. It is the
  closest thing to the array picture, because it can jump straight to index `n` without
  computing the values before it. Thousands of GPU threads can each take their own slice,
  and the generator only tracks an offset.

______________________________________________________________________

## Q: What is a `torch.Generator`, and why use one?

A `torch.Generator` is an independent random number stream with its own state. Seed it,
then hand it to a random op through the `generator=` argument.

**The problem it solves.** `torch.manual_seed(42)` seeds PyTorch's default random streams. Random ops without an explicit generator draw from the default stream for their device. Draws are sequential within each stream, so one CPU consumer can change what a later CPU consumer sees. For example, additional CPU random initialization can change a later shuffle if that shuffle uses the default CPU generator. A private generator for the loader isolates its order from that unrelated draw.

### Basic usage

```python
import torch

g = torch.Generator().manual_seed(42)  # CPU generator; manual_seed returns g
a = torch.rand(size=(3,), generator=g)
b = torch.randn(size=(2, 2), generator=g)
idx = torch.randperm(n=10, generator=g)

g.manual_seed(42)  # reseed: the same sequence again
assert torch.equal(a, torch.rand(size=(3,), generator=g))
```

Ops that take `generator=` include `torch.rand`, `randn`, `randint`, `randperm`,
`normal`, `bernoulli`, and `multinomial`, the in-place `Tensor.uniform_`, `normal_`,
and `random_`, and the `torch.nn.init` functions.

### An unseeded generator always starts from the same seed

A generator is a reader holding one book (its seed) and a bookmark (its state).
`torch.rand()` without `generator=` still has a reader: the global one,
`torch.default_generator`. The two start differently:

- **The global generator** opens a random book. Its seed is non-deterministic and
  changes every time the process starts.
- **`torch.Generator()`** opens the same book every time, at page 1. Unseeded, it doesn't
  pick a random seed. It uses a constant, `67280421310721`, hardcoded as
  `default_rng_seed_val` in PyTorch's `c10/core/GeneratorImpl.h`.

That matters as soon as a generator is created inside a loop:

```python
for _ in range(3):
    torch.rand(size=(1,))  # A: the global generator

for _ in range(3):
    torch.rand(size=(1,), generator=torch.Generator())  # B: a new generator per draw

g = torch.Generator()
for _ in range(3):
    torch.rand(size=(1,), generator=g)  # C: one generator, reused
```

Two separate runs, with no `manual_seed` anywhere:

| | Run 1 | Run 2 |
|---|---|---|
| `torch.initial_seed()` | 10969210436945446397 | 6338311990818715765 |
| `torch.Generator().initial_seed()` | 67280421310721 | 67280421310721 |
| A: global generator | 0.6692, 0.6612, 0.9196 | 0.7824, 0.3618, 0.0007 |
| B: new generator per draw | 0.2673, 0.2673, 0.2673 | 0.2673, 0.2673, 0.2673 |
| C: one generator, reused | 0.2673, 0.8725, 0.3353 | 0.2673, 0.8725, 0.3353 |

| Pattern | Within a run | Across runs |
|---|---|---|
| A: `torch.rand()` in a loop | Different: the bookmark advances | Different |
| B: `generator=torch.Generator()` in a loop | Identical: a fresh copy of the same book, read from page 1 each time | Identical |
| C: one generator created outside the loop | Different | Identical: the fixed default seed |

Pattern B is almost always a bug. Create the generator once, then seed it on purpose:
`g.manual_seed(42)` for a reproducible stream, or `g.seed()` for one that differs every
run. `torch.manual_seed` doesn't reach a private generator.

### A draw advances only the generator it reads

```python
g = torch.Generator()
g_before, global_before = g.get_state(), torch.get_rng_state()
torch.rand(size=(3,), generator=g)
print(torch.equal(g_before, g.get_state()))  # False: g advanced
print(torch.equal(global_before, torch.get_rng_state()))  # True: the global one didn't
```

In the [toy generator](#a-toy-generator), that is simply two state variables:

```python
global_state = ...  # random at startup, read by torch.rand()
g_state = 67280421310721  # read by torch.rand(generator=g)
```

Each call runs the update formula on the variable it was given. That independence is the
isolation described above: draws from the global generator can't shift `g`'s sequence,
and draws from `g` can't shift the global one.

The keyword is `generator=`. `torch.rand(3, gen=g)` raises a `TypeError`.

### Three common uses

```python
from torch.utils.data import DataLoader, random_split

# 1. A reproducible train/val split
train_ds, val_ds = random_split(
    dataset=dataset,
    lengths=[0.8, 0.2],
    generator=torch.Generator().manual_seed(0),
)

# 2. A reproducible shuffle order, unaffected by the model's randomness
loader = DataLoader(
    dataset=train_ds,
    batch_size=32,
    shuffle=True,
    generator=torch.Generator().manual_seed(0),
)

# 3. Save and rewind the stream, e.g. to resume training
state = g.get_state()
x1 = torch.rand(size=(3,), generator=g)
g.set_state(state)
x2 = torch.rand(size=(3,), generator=g)
assert torch.equal(x1, x2)
```

### Gotchas

- **The device must match.** A CUDA op needs `torch.Generator(device="cuda")`. Passing a
  CPU generator to an op that creates a CUDA tensor raises
  `Expected a 'cuda' device type for generator but found 'cpu'`.
- **Modules don't take one.** `nn.Dropout` and the default init in `nn.Linear.__init__`
  draw from the global generator, so full reproducibility still needs `torch.manual_seed`.
  Re-initializing weights yourself with `nn.init.*(..., generator=g)` is the exception.
- **A `DataLoader` generator does two jobs.** It drives the `RandomSampler`'s shuffle, and
  it draws the `base_seed` that seeds the workers (next question).
- **The other methods.** `g.initial_seed()` returns the seed in use, `g.seed()` reseeds
  from a non-deterministic source, and `torch.default_generator` is the global generator
  itself.

______________________________________________________________________

## Q: "The map-style dataset is then passed to the DataLoader to create batches." What does that mean?

PyTorch has two kinds of dataset, told apart by how you read a sample:

| Kind | You implement | You read a sample by |
|---|---|---|
| Map-style (`Dataset`) | `__getitem__(i)` and `__len__` | index: `dataset[i]`, like a dict or list |
| Iterable-style (`IterableDataset`) | `__iter__` | iterating: the next item in a stream |

"Map-style" means the dataset maps an index to a sample. It knows only one sample at a
time. Passing it to a `DataLoader` hands the batching to the loader: its sampler chooses
the indices, it calls `dataset[i]` for each, and `collate_fn` stacks the results into one
batch with a leading `B` axis. The dataset never sees a batch.

______________________________________________________________________

## Q: Why does the DataLoader pass an RNG seed to workers? It could just shuffle the indices and send them.

It does exactly that. For a map-style dataset with `shuffle=True`:

1. The main process owns the sampler (`RandomSampler`), which draws the permutation.
1. The main process groups the indices into batches and puts `(batch_idx, [indices])` on
   each worker's index queue.
1. Workers call `dataset[i]` for the indices they receive. They never shuffle.

Shuffling happens in one place. The seed workers get is for something else.

### The seed is for randomness inside `__getitem__`

Each time a loader starts iterating, it draws a `base_seed` from its `generator`. Each
worker then seeds Python's `random` and `torch` with `base_seed + worker_id`, and NumPy
with a value derived from the same two numbers (NumPy seeding was added in PyTorch 1.9).
That randomness is what a random crop, a flip, added noise, or MLM masking inside
`__getitem__` or `collate_fn` draws from.

Without the reseed, forked workers would all start from a byte-for-byte copy of the
parent's RNG state:

```
worker 0: dataset[5]  → random_crop → crop at (12, 40)
worker 1: dataset[9]  → random_crop → crop at (12, 40)   same state, same "random" numbers
worker 2: dataset[17] → random_crop → crop at (12, 40)
```

Every worker produces the same sequence of augmentations, which silently cuts
augmentation diversity. This was a well-known bug with NumPy
before PyTorch 1.9 (pytorch/pytorch#5059). A per-worker seed breaks the symmetry, and
deriving every worker's seed from one `base_seed` keeps the run reproducible once you pass
`generator=`.

Inside a worker, `torch.utils.data.get_worker_info().seed` returns that worker's seed, if
another library needs seeding from it.

### When workers do shuffle: `IterableDataset`

An iterable dataset has no indices and often no `__len__`: it streams from files, object
storage, or shards. The main process can't permute indices it doesn't have, so each worker
reads its own stream and shuffles locally, for example with a shuffle buffer.

An iterable dataset must partition its stream across workers to avoid duplicates. One design gives every worker the **same shuffled order**, then keeps every `num_workers`-th element; that design needs a shared shuffle seed or the slices can overlap and leave gaps. Another design assigns distinct source shards first, then each worker may shuffle its own shard independently. DataPipes supported by `DataLoader` include coordinated seeding and sharding for the former pattern; a custom `IterableDataset` must define its own ordering and partitioning.

**In short:** index-based data shuffles centrally. Stream-based data has no central index permutation, so its dataset controls ordering and must partition work across workers. Separately, per-worker seeds decorrelate augmentations.

______________________________________________________________________

## Q: Does a worker's randomness depend on the global generator or on the base seed?

Both. It's a chain:

```
torch.manual_seed(train_seed)
  │
  ▼
main process global generator    (or loader.generator, if you passed one)
  │  one draw each time an iterator is created, i.e. once per epoch
  ▼
base_seed
  │  worker i reseeds its own global generator with base_seed + i
  ▼
torch.rand / randint calls without generator= inside worker i
```

- Inside a worker, a draw without `generator=` reads that worker's global generator. It
  starts from `base_seed + worker_id`, so it depends directly on the base seed.
- The base seed is drawn from the main process's global generator when each epoch's
  iterator is created. So the worker depends indirectly on the main generator, and through
  it on `train_seed`.

In the book picture: at the start of each epoch, the main process reads one page of its
own book to get `base_seed`, then hands worker `i` book number `base_seed + i` for the
whole epoch.

**A consequence.** Epoch 2's `base_seed` depends on everything else that read the main
CPU generator during epoch 1: the sampler's shuffle and any CPU-side random op in the
training loop (CUDA ops use a separate GPU generator). The chain is still deterministic,
but a change to the training loop can change what the workers draw.

With `num_workers=0` there are no workers and no reseeding. `collate_fn` runs in the main
process and reads the main global generator directly. A private generator passed as
`generator=` sits outside the chain altogether.

______________________________________________________________________

## Q: A `collate_fn` seeds each sample with `seed + index`. Do its masks repeat?

Take a masking `collate_fn` that gives each sample its own generator:

```python
for index, ids in rows:
    generator = torch.Generator().manual_seed(seed + index)
    x, y = mask_row(ids=ids, generator=generator)
```

`index` is the dataset index, not the batch position. It follows the sample wherever the
sampler puts it, so `shuffle=True` changes nothing below.

- **Across rows and batches: no repeats.** Each sample gets its own seed, so each row
  gets a different mask.
- **Across epochs: every mask repeats.** Sample 42 always gets `seed + 42`, so its mask
  is identical every epoch. Passing a different `seed` at startup doesn't help; it's fixed
  for the whole run.

That is right for validation and test: every evaluation sees the same corruption, so
losses are comparable across checkpoints. For training it is **static masking**: the
model sees one corruption per sample for the whole run. RoBERTa (§4.1) found that dynamic
masking, a fresh mask each time a sequence is fed, matches or slightly beats static.
Original BERT softened static masking by duplicating the data 10 times with different
masks. The [DataLoader internals](../../app/src/training/dataload/README.md) page covers
why the two splits want different masking.

### Dropping `index` makes it worse

It is tempting to seed with `seed` alone to get "fresh noise". That does the opposite.
Every row gets the same seed, so every row in every batch and every epoch gets the same
position pattern: position 3 always masked, position 7 always replaced, and so on.

### The fix: make the seed optional

```python
def collate_fn(rows, seed: int | None):
    """Mask each row.

    seed: an int masks every sample identically each epoch (validation, test).
        None draws from the global generator, so masks change every epoch (train).
    """
    for index, ids in rows:
        generator = None if seed is None else torch.Generator().manual_seed(seed + index)
        x, y = mask_row(ids=ids, generator=generator)
```

`torch.rand` and `torch.randint` accept `generator=None`, so `mask_row` only needs its
annotation widened to `torch.Generator | None`. Then:

```python
torch.manual_seed(seed=train_seed)  # once, at startup
train_dl = DataLoader(dataset=train_ds, collate_fn=partial(collate_fn, seed=None))
val_dl = DataLoader(dataset=val_ds, collate_fn=partial(collate_fn, seed=VAL_SEED))
```

Training masks are now fresh and still reproducible. With `num_workers=0`, the global
generator keeps advancing across epochs. With `num_workers > 0`, each epoch draws a new
`base_seed`, so workers mask differently each epoch (the chain in the previous question).
Either way, `train_seed` determines the whole run.

______________________________________________________________________

## Q: What does `torch.random.fork_rng` do?

It is a context manager that saves the global generator's state when the block starts and
restores it when the block ends. Draws inside the block still read the global generator,
but once you leave, it is as if they never happened.

In the book picture: put a bookmark in the shared book, read some pages inside the block,
and the bookmark goes back where it was on the way out.

```python
torch.manual_seed(seed=0)
with torch.random.fork_rng(devices=[]):
    torch.rand(size=(1000,))  # pages read inside the block
a = torch.rand(size=(1,))  # the same value as if the block never ran
```

It doesn't create a new generator; it only protects the existing one. By default it
saves and restores the CPU generator and every visible CUDA device's generator.
`devices=[]` forks only the CPU generator.

### Where it helps: validating mid-training

Creating any `DataLoader` iterator draws a base seed from the loader's `generator`, or from
the main global generator if it has none. That holds even for a validation loader with
`shuffle=False` and `num_workers=0`. So each validation pass advances the main generator
by one draw, and the next training epoch's shuffle depends on how often you validated:
validating every epoch and every two epochs give different training runs.

```python
with torch.random.fork_rng(devices=[]):
    val_loss = evaluate(model=model, loader=val_dl)
```

Training's random stream is now the same whether or not validation ran. The alternative is
to give the train loader its own `generator=torch.Generator().manual_seed(train_seed)`, so
its shuffle and base seed never read the global generator at all.

A private per-sample generator, as in the previous question, doesn't need `fork_rng`: it
never reads the global generator and nothing else reads it. `fork_rng` with a
`manual_seed` inside would do the same job, more slowly, since it saves and restores the
state on every call.

______________________________________________________________________

## Q: With `num_workers > 0`, is each worker a thread or a process?

A process. Each worker is a separate OS process started by Python's `multiprocessing`.

### Why processes and not threads

Python's GIL (global interpreter lock) lets only one thread in a process run Python bytecode at a time. Data loading mixes Python work, I/O, and library routines; some library routines release the GIL. Separate worker processes each have their own interpreter and GIL, allowing Python-heavy preparation to run on multiple cores.

### How the processes start

The start method decides what a worker begins with. Override it with
`DataLoader(..., multiprocessing_context=...)`.

| Method | Default on | What happens | Consequence |
|---|---|---|---|
| `fork` | Linux, Python ≤ 3.13 | The child is a copy-on-write clone of the parent | Fast startup; the dataset is shared until something writes to it |
| `forkserver` | Linux, Python ≥ 3.14 | Children fork from a clean helper process started early | Safer than `fork`; the dataset is pickled and sent over |
| `spawn` | macOS, Windows | A fresh interpreter starts, and the dataset is pickled and sent to it | Slowest startup; the dataset must be picklable (no lambdas, no open file handles) |

Check yours with `torch.multiprocessing.get_start_method()`.

With `spawn` and `forkserver`, the child imports your main script, so the training code
needs a guard:

```python
if __name__ == "__main__":
    train()
```

Without it, each worker runs the top level of the script again and tries to start
training itself.

### What that implies

- **No shared variables.** A worker that changes `self.counter` changes its own copy. The
  main process and the other workers never see it. The only way back is the result queue.
- **Batches cross a process boundary.** Tensors travel through shared memory rather than
  being copied byte by byte. That's why a Docker container sometimes needs a bigger
  `/dev/shm` (`--shm-size`).
- **Copy-on-write still leaks under `fork`.** Reading a Python object updates its
  reference count, which is a write, so the pages holding a big Python list get copied
  into each worker a little at a time. Memory grows per worker. Storing the data in a
  tensor or a NumPy array avoids it, because one array is one object.
- **Pinning uses a thread.** With `pin_memory=True`, the main process runs a background
  thread that copies finished batches into pinned memory. A thread works here because the
  copy runs in C and releases the GIL.

______________________________________________________________________

## Q: Isn't forking a process that uses CUDA forbidden?

Forking is fine. *Using* CUDA in the forked child is what's forbidden, and DataLoader
workers don't.

### Why CUDA and `fork` don't mix

`fork` copies the parent's memory, but only the thread that called `fork`. The CUDA driver
keeps state that isn't plain memory: GPU contexts, background driver threads, device
handles, and locks. The child inherits memory that refers to that state, without the
threads and the GPU context behind it. So if a forked child touches CUDA, PyTorch refuses:

```
RuntimeError: Cannot re-initialize CUDA in forked subprocess.
To use CUDA with multiprocessing, you must use the 'spawn' start method
```

### Why workers get away with it

The standard design keeps workers on the CPU:

```
worker (forked, CPU only):  read → decode → augment → CPU tensor
main (owns CUDA):           batch.to("cuda")    ← the GPU transfer happens here
```

The parent may have initialized CUDA. The children never call into it, so the state they
inherited is never touched.

### When you need `spawn`

Any of these in a `Dataset` or `collate_fn` breaks a forked worker:

- Moving a tensor to the GPU (`.cuda()`, `.to("cuda")`).
- Running a model on the GPU, for GPU-side augmentation or feature extraction.
- Returning CUDA tensors from workers, which the docs advise against anyway.

If you need one of them:

```python
DataLoader(dataset=dataset, num_workers=4, multiprocessing_context="spawn")
```

`spawn` starts a fresh interpreter, so each worker creates its own CUDA context. Every
context costs GPU memory, and startup is slower. The usual better answer is to keep the
workers on the CPU and pass `pin_memory=True`, so the main process's copy to the GPU is
fast.

Python 3.14's `forkserver` default on Linux sidesteps the problem in a different way: the
helper that forks the workers is a fresh process that never initialized CUDA, so there is
no CUDA state to inherit.

______________________________________________________________________

## Common confusions

- **Seed versus state:** the seed starts a sequence; the state records where you currently are in it. Reseeding restarts, while saving and restoring state rewinds to an exact point.
- **Shuffle versus augmentation:** the main process chooses indices for map-style data; workers can still draw random crops or masks inside `__getitem__`.
- **Reproducible versus identical workers:** worker seeds differ so they do not produce the same random augmentation sequence, but they derive from a repeatable base seed.
- **Reproducible versus repeated:** a fixed seed per sample makes the run reproducible *and* repeats every mask each epoch. The global generator after one `manual_seed` is reproducible without repeating.
- **Unseeded versus random:** an unseeded `torch.Generator()` always starts from the same constant seed. Only the global generator gets a fresh seed when the process starts.
- **Stored versus computed:** a generator keeps a fixed-size state, not a list of numbers. Each draw computes the next values and overwrites the state.

## Check your understanding

1. Why can a private generator keep a loader's shuffle stable after an unrelated CPU random draw is added?
1. If the main process already shuffles sample indices, why seed each worker?
1. What must a worker avoid after being forked from a parent that has initialized CUDA?
1. After `torch.manual_seed(5)` and three draws, what does `torch.initial_seed()` return?
1. A masking `collate_fn` seeds each row with `seed + index`. What does training see across epochs, and what happens if you drop `index`?
1. Why can adding a validation pass change the next training epoch's shuffle order?
1. What does `torch.rand(size=(1,), generator=torch.Generator())` return inside a loop, and in a second run of the script?
1. After ten million draws, is a generator's state larger than before?

<details markdown="1"><summary>Answers</summary>

1. The loader consumes its own stream, so unrelated draws do not advance its state.
1. Random transforms and other work inside item retrieval need distinct, reproducible worker streams.
1. It must avoid using CUDA in that forked child; CPU loading followed by a main-process GPU transfer is the usual flow.
1. `5`. The draws advanced the state, not the seed.
1. The same mask for each sample every epoch (static masking). Without `index`, every row gets the same seed, so every row everywhere gets the same mask pattern.
1. Creating the validation iterator draws a base seed from the global generator, advancing it. `fork_rng`, or a private train-loader generator, removes the dependence.
1. The same value on every iteration, and the same value again in the second run. Each new generator starts from the fixed default seed, at page 1.
1. No. The state has a fixed size and each draw overwrites it, which is also why `get_state` and `set_state` can rewind a stream.

</details>

**One-minute recap:** a seed picks a sequence and draws advance the state; the sampler and workers consume different random streams; process boundaries explain why workers need seeds and why CPU loading is the usual CUDA-safe design.

______________________________________________________________________

## Sources

- PyTorch docs, [torch.Generator](https://pytorch.org/docs/stable/generated/torch.Generator.html):
  `manual_seed`, `get_state`, `set_state`, `initial_seed`, `seed`, and devices.
- PyTorch docs, [torch.utils.data](https://pytorch.org/docs/stable/data.html): dataset
  types, "Data Loading Order and Sampler", "Single- and Multi-process Data Loading",
  "Platform-specific behaviors", and "Randomness in multi-process data loading".
- PyTorch docs, [Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html):
  the global generators, `worker_init_fn`, and seeding a `DataLoader`.
- PyTorch docs, [torch.manual_seed](https://pytorch.org/docs/stable/generated/torch.manual_seed.html),
  [torch.initial_seed](https://pytorch.org/docs/stable/generated/torch.initial_seed.html),
  and [torch.random.fork_rng](https://pytorch.org/docs/stable/random.html#torch.random.fork_rng).
- PyTorch source: `aten/src/ATen/core/MT19937RNGEngine.h`, the CPU generator's engine.
- PyTorch source: `c10/core/GeneratorImpl.h`, `default_rng_seed_val = 67280421310721`,
  the seed of an unseeded `torch.Generator()`.
- Knuth, *The Art of Computer Programming*, Vol. 2, §3.2.1.2: the full-period
  (Hull–Dobell) theorem for linear congruential generators.
- Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3" (SC 2011): Philox, the
  counter-based generator behind CUDA's default.
- Liu et al., [RoBERTa](https://arxiv.org/abs/1907.11692) (2019), §4.1 "Static vs.
  Dynamic Masking".
- Devlin et al., [BERT](https://arxiv.org/abs/1810.04805) (2018), §3.1, and
  `dupe_factor=10` in google-research/bert's `create_pretraining_data.py`.
- PyTorch docs, [Multiprocessing best practices](https://pytorch.org/docs/stable/notes/multiprocessing.html):
  CUDA in multiprocessing and the `spawn` requirement.
- PyTorch source: `torch/utils/data/dataloader.py` (`_base_seed` drawn from
  `loader.generator`, the pin-memory thread) and `torch/utils/data/_utils/worker.py`
  (`_worker_loop` seeding `random`, `torch`, and NumPy from `base_seed + worker_id`).
- Python docs, [multiprocessing: contexts and start methods](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods):
  the default start method per platform and the 3.14 change to `forkserver`.
- [fork(2)](https://man7.org/linux/man-pages/man2/fork.2.html): the child gets only the
  calling thread.
