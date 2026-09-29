# Dataload

Dataset loading for the library. A loader module exposes one function that returns a
ready-made `torch.utils.data.DataLoader` per split, and caches downloads under
`DATASETS_CACHE_DIR` from [constants.py](constants.py). For the rest of the library,
see the [root README](../../../../README.md).

This page covers how a PyTorch `DataLoader` is put together, so a new loader can be
written without guessing, and what each split is for. For the basics (why
mini-batches, transforms, normalization), see the
[data loading tutorial](../../../../tutorials/pytorch/dataloading.md).
Choices specific to one dataset live in that module's docstring.

______________________________________________________________________

## How a DataLoader works

### The pipeline

A `DataLoader` runs three stages for every batch:

1. The **sampler** yields indices, say 32 of them.
1. The **dataset** turns each index into one sample: `dataset[i]`.
1. **`collate_fn`** merges the list of 32 samples into one batch.

The training loop receives whatever `collate_fn` returns. Every question about a
loader, such as batch size, order, padding, or seeding, belongs to exactly one of
these stages.

| Component | Knows about | Doesn't know about |
|---|---|---|
| `Dataset` | One example: loading, decoding, per-sample transforms | Batch size, order, workers |
| Sampler | Which indices, in what order | What the data is |
| `collate_fn` | How to merge N samples: stacking, padding | Where the samples came from |
| `DataLoader` | Orchestration: batching, workers, prefetch | The data format |

### A dataset returns one sample, the loader returns batches

`__getitem__(i)` returns sample `i`, typically `(x, y)`. It can be any structure: a
tuple, a dict such as `{"input_ids": ..., "label": ...}`, or a triple
`(x, y, mask)`. The default collate transposes and stacks, keeping the structure and
adding a batch axis:

| Stage | Value |
|---|---|
| `dataset[i]` | `(x: (3, 224, 224), y: 7)` |
| What the loader collects | a list of 32 such tuples |
| What default collate returns | `(X: (32, 3, 224, 224), Y: (32,))` |

Keeping the dataset per-sample is what lets batch size, shuffling, and padding change
without touching it.

**The exception is batched reads.** When one read of 32 rows costs about as much as
one row, as with a database, the dataset can return whole batches. Pass
`batch_size=None` to turn automatic batching off, and the loader then passes each item
through without collating it. Only do this when per-sample fetching is the
bottleneck.

______________________________________________________________________

## What you define

### A Dataset: the only required class

You never subclass `DataLoader`; you configure it. The one class to write is the
dataset, in one of two styles:

- **Map-style (`Dataset`)** implements `__len__` and `__getitem__(idx)`. Use it when
  samples can be indexed, such as files on disk or rows in a tensor. This is the
  common case.
- **Iterable-style (`IterableDataset`)** implements `__iter__`. Use it for streams,
  huge sharded files, or anything without random access.

### collate_fn: a function, not a class

```text
collate_fn(batch: list[Sample]) -> Batch
```

- **Input** is a Python list of whatever `__getitem__` returns. Its length is
  `batch_size`, except for the last batch, which is shorter unless `drop_last=True`.
- **Output** is anything. The training loop receives exactly what you return.
- **The default is enough when samples stack.** It stacks tensors and recurses into
  tuples and dicts. Write your own when they don't, typically for variable-length
  sequences: sentences of lengths 5, 8, and 3 cannot stack, so the collate pads them
  to 8 and also returns a mask or the lengths.
- **Extra arguments, such as a pad id,** are bound ahead of time with
  `functools.partial` or a small callable class, because the loader only ever passes
  the list.
- **Pad to the actual list length,** never to a hard-coded `batch_size`, or the short
  last batch breaks.
- **With `batch_size=None`,** collate is called on each single item instead of a
  list.

The same `collate_fn` serves every split, since all of them need the same padding.

### A Sampler: only for custom ordering

A `Sampler` yields indices, and a `BatchSampler` yields groups of indices. Write one
only for custom ordering, such as class-balanced batches or grouping similar lengths
to cut padding. `sampler=` and `shuffle=True` are mutually exclusive: with a custom
sampler, ordering is entirely its job.

______________________________________________________________________

## Train, validation, and test

### Three splits, three jobs

| Split | Used for | Influences the model |
|---|---|---|
| Train | Fitting the weights by gradient descent | Directly |
| Validation (val, dev) | Early stopping, picking a checkpoint, tuning hyperparameters | Indirectly, through every decision it drives |
| Test (held out) | One final, unbiased estimate of performance | Never. It runs once, at the end |

A split that drives any decision during development is validation, whatever the code
calls it; `fit` takes it as `val_loader`. A typical run passes over train every epoch,
runs validation every n epochs to get `val_loss`, and leaves test untouched until the
final checkpoint is chosen. With 10,000 examples, a common split is 8,000 / 1,000 /
1,000: fit on the 8,000, compare 20 configs on validation, and report the winner's
test score once.

Think of a student. Train is the textbook exercises, learned from directly.
Validation is the practice exams: no answers are learned from them, but their scores
steer how the student studies, so over many rounds the student fits them anyway. Test
is the final exam, and a student who peeked at it no longer gets a score that measures
what they know.

### Why validation and test stay separate

Try 100 learning rates and keep the one with the best validation accuracy, say 92%.
That 92% is optimistic: the winner was selected on the same data, so part of its
margin is luck on that particular split. Test chose nothing, so it gives the honest
number, say 89%. With a small dataset, k-fold cross-validation replaces the single
validation split, but a separate test set still stays out of it.

### How test leaks

- **Picking the checkpoint by test score.** Test becomes a second validation set, and
  its number inflates.
- **Iterating after reading test.** A bad test score that prompts an architecture
  change has made a decision. Over many cycles the model fits test, like a student who
  retakes the same final exam until the questions are memorized.
- **Fitting preprocessing on all the data.** Normalization statistics, a tokenizer's
  vocabulary, and feature scaling come from train alone, then apply unchanged to
  validation and test.

The check: if test were deleted right after training, would any code or decision
change? If not, it is properly separate.

______________________________________________________________________

## Order and randomness

### Shuffling

- **The default is `shuffle=False`.** It gives a `SequentialSampler`.
- **`shuffle=True` reshuffles every epoch by itself.** It uses a `RandomSampler`,
  which draws a new permutation each time iteration starts, with no extra work.
- **It shuffles samples, not batches.** Indices are permuted first, then chunked.
  With indices 0–5 and batch size 2, one epoch might give `[4, 1] [5, 0] [2, 3]` and
  the next `[0, 3] [1, 5] [4, 2]`.
- **An `IterableDataset` cannot use `shuffle=True`.** Shuffle inside `__iter__`
  instead.
- **Multi-GPU training uses `DistributedSampler`,** and needs `set_epoch(epoch)` every
  epoch. Without it, every epoch repeats the same order.

### Each split gets its own loader

A sampler belongs to one `DataLoader`, and nothing carries over. Build one loader per
split, each with its own sampler:

- **Train:** `shuffle=True`, or a custom sampler.
- **Validation and test:** `shuffle=False`.

Not reusing the train sampler for validation and test is deliberate. Class-balanced
sampling or oversampling changes which examples you see, so reusing it would bias the
metrics. Validation and test should measure the true distribution, once per example.

**Shuffling validation would not change its metrics.** Order only matters for
training, where it sets the order of gradient updates. Validation makes no updates,
and its loss is a mean over every example, which does not depend on order.
`shuffle=False` is for two practical reasons:

- **Reproducible batches.** Batch 7 is the same batch every pass, so per-batch logs
  compare across epochs and a bad example is easy to trace.
- **Partial validation.** When only the first k batches are scored, as with
  Lightning's `limit_val_batches`, shuffling picks a different subset each pass, and
  `val_loss` stops being comparable across epochs. This is the one case where
  shuffling validation breaks something.

**Distributed validation double-counts.** `DistributedSampler(val_ds, shuffle=False)`
pads the dataset with repeated samples so every GPU gets an equal share: 10 samples on
4 GPUs become 12, and 2 are counted twice unless you de-duplicate. On a small
validation set this skews the metrics slightly.

### Seeding

Pass `generator=torch.Generator().manual_seed(seed)` to the `DataLoader`.

- **Validation with `shuffle=False` is already deterministic.** A seed only matters if
  validation has randomness, such as random subsampling or augmentation.
- **A fresh generator per pass repeats the order.** Creating a new generator with the
  same seed before each validation pass gives the identical order every time. Reusing
  one generator advances its state, so each pass differs, but the whole run is still
  reproducible.
- **Workers seed torch and Python, not NumPy.** With `num_workers > 0`, each worker's
  `torch` and `random` are seeded from the generator. NumPy is not, so if
  `__getitem__` uses NumPy randomness, pass a `worker_init_fn` that seeds it per
  worker, or workers produce duplicate "random" augmentations.

### Random collates: mask validation and test once

Masked language modelling picks the tokens to mask inside `collate_fn`, so every call
draws fresh randomness. Treat the splits differently:

- **Train: keep it random.** Fresh masks every epoch give the model more variety.
  RoBERTa calls this dynamic masking and finds it matches or beats masking the data
  once.
- **Validation and test: fix the masks.** If masks change every pass, `val_loss`
  carries mask noise: one pass hides easy tokens like "the", another hides rare
  nouns. A drop from 2.31 to 2.29 could be a better model or easier masks, and early
  stopping reacts to the noise.

Hugging Face's `DataCollatorForLanguageModeling` samples masks on every call, so a
validation loader built on it already changes its masks every pass.

The simplest fix masks once and reuses the batches, which also sidesteps seeding
workers:

```python
def premask(dataset, collate_fn, seed, batch_size=256):
    """Collate dataset once, with masks drawn from a fixed seed.

    fork_rng restores the global RNG afterwards, so training randomness is
    unaffected. Returns the batches to reuse at every pass.
    """
    batches = []
    with torch.random.fork_rng():
        torch.manual_seed(seed=seed)
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            batches.append(collate_fn([dataset[i] for i in range(start, stop)]))
    return batches
```

Call it once for validation and once for test, with different seeds. The loop only
iterates over its loader, so the returned list can be passed where a `DataLoader` is
expected. Collation also runs once instead of every pass.

______________________________________________________________________

## Long documents: chunks

### Token, chunk, sample, batch

A model has a fixed maximum input length, say `max_len=512`. A longer document is cut
into chunks: contiguous slices of one document's tokens, each at most `max_len` long.
A 1000-token document with no overlap becomes:

- chunk 0: tokens 0–511 (512 tokens)
- chunk 1: tokens 512–999 (488 tokens)

A chunk is one sample, not a batch. The hierarchy is:

| Unit | What it is |
|---|---|
| Token | A word or word piece, such as `"play"` or `"##ing"` |
| Chunk | A sequence of tokens, capped at `max_len`. One chunk is one sample |
| Batch | A group of chunks, say 32, possibly from different documents |

"Chunk" is informal; Hugging Face also calls these features or blocks.

### Three ways to serve chunks

1. **Index the chunks.** The map-style dataset's indices count chunks, not
   documents, so `dataset[5]` is the sixth chunk, which might be the second half of
   document 2. Batches have a fixed size and `shuffle=True` mixes documents.
1. **Return every chunk of a document from one `__getitem__`.** One index yields a
   variable number of chunks, so batch sizes become irregular and a batch's chunks
   are correlated.
1. **An `IterableDataset` that yields one chunk per `yield`.** Batches still have a
   fixed size: the loader takes the next `batch_size` items from the stream, so it
   doesn't matter that one document gave 2 chunks and the next gave 7. Yielding a
   whole list of chunks per document brings back option 2's problems.

### Where an IterableDataset still gets short batches

- **The last batch,** as with any dataset, unless `drop_last=True`.
- **One per worker.** With `num_workers > 0`, each worker runs its own copy of the
  iterator and batches its own stream, so each can end on a short batch: up to 4
  undersized batches with 4 workers. `drop_last=True` drops each worker's last
  partial batch.
- **Every worker replays the whole stream unless you shard.** Without sharding, each
  chunk is yielded `num_workers` times. Shard inside `__iter__` with
  `torch.utils.data.get_worker_info()`, for example having worker `k` process only
  documents where `doc_idx % num_workers == k`.

______________________________________________________________________

## Writing a new loader

1. **Write the Dataset.** One sample per `__getitem__`, in the structure the model's
   forward pass expects.
1. **Pick the collate.** Use the default if samples already stack. Otherwise write a
   `collate_fn` that pads to the longest sample in the list and returns a mask.
1. **Build one DataLoader per split.** `shuffle=True` for train, `shuffle=False` for
   validation and test, with the same `collate_fn` on all three. If the collate is
   random, as with masking, premask validation and test once.
1. **Decide on workers.** Data already in memory gains nothing from
   `num_workers > 0`: workers add startup and pickling cost. Workers pay off when
   `__getitem__` reads from disk or does heavy per-sample work.
1. **Document the dataset's choices in the module docstring:** splits, caching, and
   anything a caller must pass to invalidate the cache.

______________________________________________________________________

## Sources

- PyTorch docs, [torch.utils.data](https://pytorch.org/docs/stable/data.html):
  dataset types, samplers, `collate_fn`, `shuffle`, `generator`, automatic batching,
  multi-process loading, and `DistributedSampler`.
- PyTorch docs,
  [Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html): worker
  seeding, the NumPy caveat, and `worker_init_fn`.
- Hugging Face LLM Course,
  [Chapter 7](https://huggingface.co/learn/llm-course/chapter7/7): splitting long
  inputs into fixed-length pieces.
- Goodfellow, Bengio, and Courville, *Deep Learning* (2016), §5.3, "Hyperparameters
  and Validation Sets": the roles of the three splits, and why validation is biased
  once it has been used to tune.
- Hastie, Tibshirani, and Friedman, *The Elements of Statistical Learning* (2nd ed.),
  §7.2: the three-way split, and keeping test for the final assessment.
- scikit-learn user guide, the cross-validation guide and "Common pitfalls and
  recommended practices: Data leakage": k-fold as a replacement for the validation
  split, and fitting preprocessing on train only.
- Liu et al., [RoBERTa](https://arxiv.org/abs/1907.11692) (2019), §4.1: static
  versus dynamic masking.
- Hugging Face docs, `DataCollatorForLanguageModeling`: masks are sampled per call.
- PyTorch Lightning docs, `Trainer(limit_val_batches=...)`: scoring a subset of the
  validation loader.
