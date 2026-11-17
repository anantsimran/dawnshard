# ADR 0002: Audit other loaders for exact resume instead of taking one open issue

Status: accepted, 2026-11-16

Related: [T6 design notes](../../experiments/t6-data-loading/DESIGN.md) (the loader the
audit compares against), [T6 results](../../experiments/t6-data-loading/RESULTS.md) (the
resume check), T11 Part G in issue #64 (the upstream goal this changes)

## Context

T11 Part G asks for one upstream PR to vLLM or SGLang. The target moved to data loading
because T6 already holds the tools for it. `resume` and `global_batch_hashes` in
[data_loading.py](../../app/src/training/experiments/data_loading.py) hash every global
batch of an uninterrupted run and of a run resumed from a saved loader state, then
compare the two step by step. On the T6 loader the 20 batches after a stop at step 137
match bit for bit, resumed on 4 ranks and on 2.

A search on 2026-11-16 for an open issue to take sorted the candidate repositories into
three groups.

- One has a free issue and no reviews. `meta-pytorch/data#1561` is a real bug. On main
  at `a0266d0` with torch 2.12.1, a `StatefulDataLoader` whose sampler shares the
  loader's generator resumes onto different batches. The sampler saves the generator
  state from before its first permutation, the loader then draws its base seed from the
  same generator, and the restore replays without that draw. `shuffle=True` with a
  `generator` builds this configuration by default: the interrupted epoch resumes
  correctly and every later epoch is shuffled differently. The reporter's fix was closed
  after their account was compromised. The maintainers said in June 2026 that they are
  not investing in new development, no PR from outside the core team has merged since
  April 2026, and 49 PRs are open.
- Several review quickly and have no free issue. In `huggingface/datasets` each
  streaming or resume issue checked had a PR within a day. In `pytorch/torchtitan` a
  maintainer's PR replaced an outside fix for #5623 two days after it was opened. In
  `pytorch/pytorch` two small DataLoader fixes have waited since 2026-09-05.
- `Lightning-AI/litData` reviews quickly and has one free item. Outside PRs merge there
  in two to nine days. The item is #797, a shuffle buffer, and no maintainer has agreed
  to it yet.

A PR against a single issue would either wait unreviewed or duplicate one that already
exists.

## Decision

The upstream contribution is a resume audit of four loaders. Its outputs are a results
table, an article, and an upstream report for each failure.

The loaders are torchdata's `StatefulDataLoader`, the Hugging Face `datasets` streaming
`IterableDataset`, litData's `StreamingDataLoader`, and MosaicML's `StreamingDataset`.
The T6 design notes already name the first and the last as alternatives, and T6 cut its
MosaicML comparison.

The check is the T6 one. A run iterates the loader and hashes every batch. A second run
stops at a chosen step, saves the loader's `state_dict`, builds a fresh loader, loads
the state, and hashes every batch from there. A cell passes when every hash after the
stop matches through the end of the following epoch. The following epoch is included
because the torchdata failure above appears only there when the loader builds its own
sampler.

Each loader runs with shuffling on and a fixed seed, with zero workers and with more
than one, and with the stop placed at the start of an epoch, in the middle, and at its
end. Where a loader documents resuming on a different number of ranks, the 4-to-2 resume
from T6 runs as well. Every row records the library version and commit.

A failure gets a minimal reproducer and an upstream report. The report is a comment on
the existing issue when there is one and a new issue otherwise. A fix PR is optional and
goes only to a repository whose maintainers review outside PRs.

The audit code is an experiment module under `app/src/training/experiments/`, and its
results go in `experiments/t11-upstream/`. `dataload/` does not change and imports none
of the four libraries. The write-up is a knowledge-sharing article that follows
[the reproducibility article](../../knowledge_sharing/reproducibility.md). That one
covers a reproducible start, and this one covers a reproducible resume.

## Consequences

The work finishes without a maintainer's reply. The table and the article exist whether
or not an upstream PR merges, which the "one merged PR" in T11 Part G did not allow.

One cell is known and three are not. torchdata fails with a shared generator. If the
other three loaders pass, the table is one failure and three passes, and the article
then rests on the design comparison between a loader position of two integers and saved
iterator and generator state.

The method is not new. `anishpentyala/resumecheck` compares resumed and uninterrupted
training runs. What this audit adds is a measured table with pinned versions.

Four third-party libraries enter the environment for one experiment. Their versions
move, so every result carries a date and a commit, and the table goes stale.

The audit hashes batches only. It says nothing about model, optimizer, or RNG state
after a resume, which T10 covers for this repo's own checkpoints.

## Alternatives considered

Fix `meta-pytorch/data#1561`. The bug reproduces, and a fix that saves the generator
state before the current permutation, the cursor, and the live generator state took the
failing checkpoints from 39 of 39 to 0 of 39 in an in-memory patch. It was rejected as
the main contribution because the repository is unlikely to review it and the reporter
has said they plan to fix it. The reproducer and the `shuffle=True` finding still go to
the issue as a comment.

Build the litData shuffle buffer (#797). It matches the T6 shuffle-quality study, and
the repository reviews quickly. No maintainer has agreed to the feature, and it touches
resume and chunk eviction, so it is a medium-sized change that may be declined. Its
same-chunk measurement fits in the article.

Reproduce `pytorch/torchtitan#6093`, where compiled FlexAttention makes a resumed run
differ from an uninterrupted one. The issue asks for the measurement that would choose
between two fixes. It needs Linux or a GPU, since `torch.compile` does not support the
CPU of the development machine, and a maintainer may fix it first.

Keep T11 Part G as written. vLLM had no unclaimed `good first issue`. SGLang's are
kernel and quantization features, most with closed PR attempts already.

## Not decided here

- How the four libraries are installed: a dependency group in `pyproject.toml`, or a
  separate environment that only the experiment runs in.
- How ranks are simulated for a loader that reads its rank from the process group.
- The stop steps, the worker counts, and the dataset behind the table.
- Which failures, if any, get a fix PR.
- Whether issue #64 is edited so that Part G names this audit.
- The article's title, and whether the landing page gets a card for it.
