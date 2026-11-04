# T6: results

The ticket's benchmark (1B FineWeb-Edu tokens on 4 GPUs) needs the GPUs and the
bucket. Every check below ran here, on CPU, on the 29,000 Multi30k English training
captions: BPE with 2,000 merges, about 430K tokens, 22 shards of 20K tokens, windows
of 64 + 1 tokens, global batch 32. Command:
`uv run python app/src/training/experiments/data_loading.py all`.

## Ownership audit: 4 ranks x 4 workers, one epoch ([audit.csv](audit.csv))

| Loader | Served | Distinct | Duplicates | Gaps | Most copies of one sample |
|---|---|---|---|---|---|
| rank x worker sharded | 7,744 | 7,744 | 0 | 0 | 1 |
| unsharded `IterableDataset` | 123,904 | 7,744 | 116,160 | 0 | 16 |

Prediction: zero duplicates and gaps; the unsharded loader serves every sample 16
times. Both hold. 7,744 is 242 whole global batches; the 17 windows that do not fill a
batch are dropped by design. The property test checks the same partition for every
world size from 1 to 8 and 0 to 4 workers.

## Shuffle quality against buffer size ([shuffle.csv](shuffle.csv))

Share of consecutive samples that come from the same shard (22 shards):

| Method | Buffer | Same-shard neighbours |
|---|---|---|
| no shuffle | 0 | 99.7% |
| shuffle buffer | 10 | 96.5% |
| shuffle buffer | 100 | 70.2% |
| shuffle buffer | 1,000 | 14.8% |
| global permutation | all 7,761 | 3.7% |

Prediction: quality improves with buffer size, memory grows linearly. It does, slowly:
with about 350 samples per shard, a buffer must be a few shards deep before
neighbours stop sharing a shard. A random order's floor is 1/22 = 4.5%; the global
permutation is at it.

## Resume: stop at step 137 ([resume.csv](resume.csv))

The rank-0 loader is advanced 137 steps and its `state_dict` saved:
`{"epoch": 0, "samples_consumed": 4384}`. Fresh loaders resume from it on 4 ranks and
on 2 ranks; each global batch is rebuilt by concatenating every rank's batch, and
hashed.

| Steps compared | Uninterrupted = resumed on 4 = resumed on 2 |
|---|---|
| 137 to 156 | 20 of 20 bitwise identical |

## Data stall ([stall.csv](stall.csv))

Tiny Qwen3 trained for 60 steps through `fit` with `measure_data_stall=True`, on a CPU
shared with another training job:

| Pipeline | Data stall | Tokens/s |
|---|---|---|
| v0: BPE on raw text in `__getitem__`, no workers | 46.7% | 575 |
| v1: pre-tokenized shards, `StreamingTokens` | 0.5% | 1,096 |

Prediction: above 30% for v0, under 2% for v1. Both hold. Tokenizing in the loader
nearly halves throughput even with a model this small.

## Tokenize script

`scripts/tokenize_shards.sh` ran two input files in parallel (`JOBS=2`) into shards
and a manifest in 6.9 s for 427,531 tokens. It is written for zsh; this machine has
no zsh, so it was run with bash, which its syntax also supports.

## Not run here

| Ticket item | Needs |
|---|---|
| Stall waterfall v0 to v7 on 4 GPUs, tokens/s | the 4x 24 GB box |
| v3 object storage: bandwidth per worker | a bucket; the HTTP range path is tested against a local server |
| v2 document masking: loss and tokens/s cost | GPUs |
| v7 MosaicML comparison, image side experiment | cut, as the ticket allows |
