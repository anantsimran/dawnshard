# T6: feeding GPUs at scale, design notes

This ticket adds offline tokenization into shards
([shards.py](../../app/src/training/dataload/shards.py)), a streaming loader with a
deterministic global order and exact resume
([streaming.py](../../app/src/training/dataload/streaming.py)), document masks for
packed sequences ([mask.py](../../app/src/training/transformer/mask.py)), a stall
meter in the loop, the loader's position in `TrainState` and every checkpoint, a
parallel tokenize script, and an audit experiment
([data_loading.py](../../app/src/training/experiments/data_loading.py)). Measured
numbers are in [RESULTS.md](RESULTS.md).

______________________________________________________________________

## Order is a pure function of (seed, epoch, index)

Sample `i` is token window `i` of the shard stream. Epoch `e`'s order is a NumPy
permutation seeded with `[seed, e]`. Global batch `k` is `order[k*B : (k+1)*B]`;
rank `r` takes its contiguous `B / R`; worker `w` builds steps `w, w + W, ...`.

- **Pros.** Nothing about the order lives in a worker, so there is no worker RNG to
  seed, save, or get wrong. Resume needs two integers. A different world size changes
  only the rank split, so a run resumes on 2 GPUs with the batches it would have seen
  on 4. Ownership is checkable by pure arithmetic, which the property test does for
  world sizes 1 to 8 and 0 to 4 workers.
- **Cons.** The permutation is materialized per epoch (one int64 per sample: 8 MB per
  million windows). Window boundaries are fixed by `seq_len`, so changing `seq_len`
  changes every sample. Holding the global batch fixed is the rule resume depends on.

**Alternatives considered.** A shuffle buffer over a shard-ordered stream: no index,
but its state is the buffer, and the shuffle-quality study shows how slowly it mixes
when neighbours share a shard. torchdata `StatefulDataLoader`: saves each worker's
iterator state, which works but ties resume to the worker count. MosaicML Streaming:
the same idea plus "canonical nodes"; a dependency for what is about 200 lines here.

## Workers produce whole batches

`StreamingTokens` yields per-rank batches, and its DataLoader uses
`batch_size=None`. Each worker owns whole steps, and the DataLoader's round-robin over
workers returns them in step order.

- **Pros.** No short batch per worker at the end of an epoch, and the batch a rank
  sees at step k is exactly its slice of global batch k.
- **Cons.** Batch construction (one `read` per sample) runs in Python in the worker;
  for text this is far below GPU speed, for images it would not be.

## The loader position lives in `TrainState`

`TrainState.loader` holds anything with `state_dict` / `load_state_dict`;
`save_state` and `load` include it. `StatefulLoader` counts a batch as consumed when
it hands it out, so a checkpoint after training on batch k resumes at k + 1.

**Alternative considered.** Saving the position from a callback in the loop. The
loop has no callbacks by design; the position is state, and state is `TrainState`.

## Shards: flat uint32 plus an index and a manifest

`.bin` files of uint32 ids, a `.idx` of document starts, and `manifest.json` written
last. Tokenization takes already-tokenized documents, so any tokenizer works; the
command-line entry uses the repo's BPE. Remote shards are read with byte-range GETs
or downloaded once into a cache.

- **Pros.** A window is one contiguous byte range: a memmap slice locally, one GET
  remotely. No decoding, no per-record framing.
- **Cons.** No compression and no per-record metadata, which image or multimodal data
  would need (WebDataset tars, MDS, Parquet). The HTTP reader covers public or
  presigned bucket URLs, not signed S3 or GCS API calls; adding `boto3` or `gcsfs` is
  one more `_read_bytes` branch.

## Packing with document masks

Documents are packed back to back with EOS separators; `document_keep_mask` is causal
and block-diagonal, and `document_positions` restarts positions per document.
`Qwen3(document_eos_id=...)` uses both, passing the mask to SDPA instead of
`is_causal`. A test shows a packed document's logits equal the same document alone.

- **Cons.** An explicit `(B, 1, L, L)` boolean mask rules out the flash kernel, so
  document masking costs speed; FlexAttention block masks or varlen `cu_seqlens`
  would keep a fused kernel.

## Stall meter

`EpochSpec.measure_data_stall` makes `run_epoch` time the wait for each batch and each
step (after a device synchronize) and `fit` record `data_stall_fraction`. Off by
default, so existing history files keep their format.

## Not built

The Ray Data image side experiment and the MosaicML comparison, which the ticket cuts
first when behind. The FineWeb-Edu and multi-GPU runs need the bucket and GPUs.

## Routes not to take

- **Tokenizing in `__getitem__`.** The study measured a 47% data stall against 0.5% from
  pre-tokenized shards; the training loop should never see text.
- **uint16 token files for a vocabulary above 65,535.** Ids wrap into wrong tokens and
  nothing crashes.
- **An `IterableDataset` that does not shard by rank and worker.** Every sample is
  served `ranks x workers` times; the audit measured 16 copies at 4 x 4.
- **RNG state inside workers, or a shuffle that depends on the worker count.** Resume
  then depends on things a checkpoint does not hold. Make the order a function of
  (seed, epoch, index).
- **Counting a batch as consumed after the training step.** A checkpoint taken after
  step k then replays batch k; count it when it is handed out.
- **Resuming on a different world size with a different global batch.** Only the
  per-rank split may change; the global batch is what keeps the order identical.
- **A small shuffle buffer over shard-ordered data.** With hundreds of samples per
  shard, neighbours keep coming from one shard; interleave shards or use a permutation.
- **An uneven last batch across ranks.** One rank waits in a collective for a batch the
  others never send; drop the tail.
