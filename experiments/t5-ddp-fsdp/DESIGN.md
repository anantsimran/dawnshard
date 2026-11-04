# T5: DDP to FSDP2, design notes

This ticket adds the `parallel` package
([app/src/training/parallel/](../../app/src/training/parallel/)): process-group setup,
data parallelism twice by hand (`NaiveDDP`, `BucketedDDP`), ZeRO-2 by hand
(`Zero2AdamW`), and `apply_fsdp` for FSDP2 and HSDP. `fit` runs unchanged under
`torchrun`. Measured numbers are in [RESULTS.md](RESULTS.md).

______________________________________________________________________

## The loop does not know it is distributed

`train/` never imports `parallel/`. Two small, duck-typed changes make `fit` correct
under `torchrun`:

- **Rank 0 writes.** `fit` and `profiled_fit` save history and log to W&B only when
  `torch.distributed` is not initialized or the rank is 0. Every rank logs its own
  peak memory each epoch.
- **`no_sync` during accumulation.** If the model has a `no_sync()` context, the
  micro-batches that do not step run inside it. PyTorch DDP and both hand-written
  wrappers have one, so all three communicate once per optimizer step.

**Alternative considered.** A `distributed: bool` field on `EpochSpec`, or a
`DistributedFit`. Both make the loop know about a concern it can detect.

## Hand DDP that needs no change to the step

Both wrappers broadcast rank 0's weights at construction and make `loss.backward()`
return with gradients already averaged, so `train_step` calls `backward` and `step`
exactly as before.

- **Naive:** the first post-accumulate-grad hook of a backward queues a callback on
  the autograd engine; the callback all-reduces every gradient, one collective per
  parameter, after backward.
- **Bucketed:** parameters are grouped into ~25 MB buckets in reverse registration
  order (roughly gradient-ready order). Each hook counts its parameter into its
  bucket; a full bucket is flattened and all-reduced asynchronously while backward
  continues. The queued callback waits for every handle, launches any bucket that
  never filled (zeros for missing gradients, so all ranks still issue the same
  collectives), and copies the averages back.

**Pros.** The overlap is visible in a trace and the mechanism is about 150 lines.
**Cons.** One extra copy per bucket (flatten, then copy back); PyTorch DDP avoids it
with `gradient_as_bucket_view=True`, which the `ddp-torch` strategy uses. No
`find_unused_parameters`: an unused parameter just delays its bucket to the callback.

## ZeRO-2 with whole-parameter ownership

Each parameter has exactly one owner rank, assigned largest-first to the least-loaded
rank. Its gradient is reduced to the owner the moment it is ready and freed on every
other rank; the owner updates an fp32 master copy with AdamW and broadcasts it.

- **Pros.** Easy to verify: one owner per parameter, held gradients equal owned
  parameters, and the result matches `torch.optim.AdamW` on the combined batch (all
  three are tests). Reduce plus broadcast is the same traffic as one all-reduce.
- **Cons.** Shards are uneven by up to one parameter (the embedding dominates a small
  model). Collectives are per parameter and synchronous in the hook, so there is no
  overlap; the point of this version is the memory, and `BucketedDDP` shows overlap.

**Alternative considered.** Flat-buffer sharding with `reduce_scatter_tensor` and
`all_gather_into_tensor`, as DeepSpeed and Megatron do. Even shards and bucketed
collectives, but it needs views from the flat buffer back into every parameter, which
is most of the code of a real implementation.

## FSDP2 through one function

`apply_fsdp(model, block_type, spec, device_type)` calls `fully_shard` on every block,
then on the root, with a `MixedPrecisionPolicy` (bf16 parameters, fp32 reduce by
default) and a 1-D mesh, or a 2-D (replicate, shard) mesh for HSDP. The optimizer is
built after it, because sharding replaces the parameters.

**Pros.** Any model with a block class can be sharded; nothing in `model/` changes.
**Cons.** Per-block wrapping only; a model whose memory is in one huge layer (a 150K
vocabulary embedding) keeps that layer in the root unit.

## Strategy memory as a function

`perf.static_bytes_per_parameter(strategy, world_size)` returns the ticket's table:
16, 4 + 12/N, 2 + 14/N, and 16/N bytes per parameter for DDP and ZeRO-1, -2, -3, under
bf16 weights and gradients with fp32 master weights and Adam state.

## Not built

The bus-bandwidth sweep, the 8-GPU NVLink session, and multi-node HSDP need GPUs;
`nccl-tests` is an external binary. The weight checksum at step 0 is logged by the
Qwen3 entry point on every rank.

## Routes not to take

- **Speed claims before loss parity.** A data-parallel bug that drops a gradient still
  runs fast. Parity against one process on the combined batch comes first.
- **All-reducing gradients one by one after backward.** Hundreds of latency-bound
  collectives with no overlap; bucket them and launch from gradient hooks.
- **Accumulating without `no_sync`.** Every micro-batch pays a full all-reduce.
- **Building the optimizer before `fully_shard`.** It holds the old, unsharded
  parameters and updates nothing that trains.
- **Calling `model.forward(x)` on an FSDP module.** It skips the hooks that gather the
  parameters; call `model(x)`.
- **Different initial weights per rank.** Broadcast from rank 0 (or seed identically)
  and log a checksum at step 0.
- **DDP for a model whose optimizer state does not fit.** 1.7B at 16 bytes per
  parameter is 27.5 GB per GPU; shard it.
- **Unlabeled multi-GPU numbers.** PCIe without P2P and NVLink differ by 30x in
  bandwidth; every number carries its interconnect.
