# Discussion: where the BPE tokenizer lives relative to a checkpoint

Open notes from 2026-11-03, written from the repo and from
[the model_config_spec.md](../../model_config_spec.md), not from a recorded conversation.
A separate chat on this topic exists and has not been merged in. Treat everything here
as a starting point for that merge. [ADR 0001](../adr/0001-self-describing-checkpoints.md)
covers the model only and leaves the tokenizer to a later decision.

______________________________________________________________________

## Why the model file is not enough

Embedding rows are indexed by BPE ids. An id means whatever merges file the loader reads
at load time. The loader finds that file through `tokenizer_path.txt`, and the
translator's `main()` retrains both tokenizers and repoints that file on every run. A
checkpoint therefore loads fine against the wrong tokenizer. The shapes match, the
weights are valid, and every translation is wrong.

The config and weights describe the network. Nothing in them records which merges the
network was trained with.

## What is already in the repo

- The vocab size follows from the merges: `len(vocab) == FIRST_MERGE_ID + len(merges)`,
  with `FIRST_MERGE_ID` derived in `bpe.py`. The embedding sizes need not be stored
  separately.
- `bpe.tokenizer_from_pairs` rebuilds `(merges, vocab)` from merge id pairs.
- The inverse, merges to id pairs, exists only inline in `_save_merges` in
  `dataload/multi30k.py`. `dataload/ag_news.py` has a second copy that writes JSON, and
  `viz/bpe_seq_len.py` imports that copy's private helpers.
- The three train-or-load paths disagree on staleness. The AG News JSON records
  `num_merges`, `first_merge_id`, and the chunk pattern, and `_load_merges` rejects a
  mismatch. The Multi30k text file records nothing, so only `retrain_tokenizer=True`
  replaces it. Moving this logic into `bpe.py` would give the checkpoint one format to
  embed.
- `save_state` in `train/utils.py` writes only the model, optimizer, and scheduler
  state. The tokenizer reaches a checkpoint only if the model module adds it.

## Options on the table

1. Store the merge pairs inside the checkpoint. A list of `[first_id, second_id]`
   pairs in merge-rank order is plain ints, so it loads under `weights_only=True`. The
   file alone rebuilds the model and both tokenizers on a machine with no datasets. This
   is the spec's default (D5). It needs `bpe.pairs_from_merges` as the inverse of
   `tokenizer_from_pairs`.
   Store pairs, not pickled `Phrase` trees. Pairs are small, readable, independent of
   the Python version, and `tokenizer_from_pairs` already rebuilds `vocab` from them, so
   `vocab` is not saved. Record `first_merge_id` and the chunk pattern beside the pairs,
   as the AG News cache does, so a load can refuse merges written under a different id
   layout or pre-tokenization. Write the file from the model module, since `train/`
   stays tokenizer-agnostic. For the translator that means one entry per tokenizer, and
   the history file's `meta` can name each tokenizer's hash.
1. Store the path to the merges file. The file stays small, but the path points into
   `app/datasets/`, which is gitignored and not copied with the checkpoint.
1. Store a hash of the merges. The earlier discussion floated "tokenizer hashes". A
   hash detects a mismatch but cannot repair one: the loader still needs the merges from
   somewhere else.

## Open questions

- Does the checkpoint hold the pairs, or only a hash plus a separate tokenizer file that
  ships beside it?
- If the pairs go in the checkpoint, do both tokenizers live in the model's metadata, or
  in a separate top-level key so `train/` and the model module do not share a dict?
- Which tokenizer hyperparameters, if any, need recording beyond the pairs? The number of
  merges follows from the pairs, and the pre-tokenization pattern is fixed in code.
- Should the dataset-side cache also become content-addressed? Keying
  `bpe_<hash>.json` on the training text, `num_merges`, chunk pattern, and
  `FIRST_MERGE_ID` makes any file that matches valid by construction. It would replace
  the Multi30k pointer file, so "pin the default tokenizer" would move to the
  checkpoint. That changes behavior and needs a decision.
- Should `bpe.py` expose `save`, `load`, and `train_or_load`, or a frozen `Tokenizer`
  value holding `merges`, `vocab`, and the encode table? The checkpoint format is the
  same either way; the choice only changes the loader call sites.
- Should loading verify anything, such as the vocab size against the embedding shape?
  Strict `load_state_dict` already fails on a size mismatch, but not on same-size merges
  from a different corpus.
