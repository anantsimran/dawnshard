# T10: distributed checkpointing and fault tolerance, design notes

This ticket adds one definition of what a checkpoint holds and a sharded, atomic,
optionally asynchronous checkpointer
([checkpoint.py](../../app/src/training/train/checkpoint.py)), a straggler detector
([straggler.py](../../app/src/training/train/straggler.py)), a seeded failure
injector ([chaos.py](../../scripts/chaos.py)), and a study that exercises all of them
on 2 to 4 CPU ranks ([fault_tolerance.py](../../app/src/training/experiments/fault_tolerance.py)).
Numbers are in [RESULTS.md](RESULTS.md).

______________________________________________________________________

## One list of contents, two storage paths

`capture` and `restore` define everything besides weights a bitwise resume reads: the
scheduler, the loader's position (T6), every RNG of the process (torch CPU and CUDA,
Python, NumPy), and the step. `save_state`/`load` (one file, one process) and
`Checkpointer` (sharded) both call them, so there is still one answer to "what is in
a checkpoint", which is what the ticket's "one way to checkpoint" protects.

- **Per-rank RNG under per-rank keys** (`rank{r}/rng`): each rank writes its own and
  reads its own. On a smaller world the missing ranks are simply absent; on a larger
  one the new ranks keep their fresh state.
- **The loader position is two integers**, so resharding it is free: global batch k
  is the same on any world size.

## Sharded writes through DCP, committed atomically

Every rank writes its shards with `torch.distributed.checkpoint`; `get_state_dict`
keeps FSDP's DTensors sharded, and `load_latest` reshards to the current world on
read. After every rank has finished, rank 0 writes a `COMMITTED` marker through a
temporary file and a rename. `latest_committed` only considers marked directories, so
a crash mid-write leaves the previous checkpoint as the latest, and the new partial
directory is ignored.

**Alternatives considered.** Rank 0 gathering the full state dict and `torch.save`
(kept as `save_full_on_rank0`, the v0 baseline): simple and portable, but serial and
a memory spike on rank 0 that grows with the model. Megatron's `dist_checkpointing`:
the same idea with more layout options; DCP is what FSDP2 and TorchTitan use.

## Async: copy in the call, write in the background, commit later

`dcp.async_save` stages the state to CPU before it returns, so the snapshot is
complete before the next optimizer step can change the weights, and writes it on a
background thread. The commit marker needs a barrier, and a barrier on a background
thread would race training's own collectives, so the marker is written at the next
`save` or at `wait()`, from the main thread.

## Short collective timeouts

The worker initializes its process group with a 30-second timeout. With the default
(30 minutes), a run in the chaos study hung for nine minutes after a peer died during a
collective before anything noticed; with 30 s, the survivor errors, `torchrun`
restarts both, and the job resumes from the last commit.

## Routes not to take

- **Saving only model and optimizer.** The loss curve after resume looks fine and the
  run silently retrains on data it has seen (no loader state) or draws different
  dropout and sampling (no RNG). The resume study compares losses and batch hashes bit
  for bit, which is the only test that catches this.
- **One RNG state for all ranks.** Ranks draw different streams; restoring rank 0's
  everywhere makes them identical.
- **Writing "latest" in place.** A crash mid-write corrupts the only checkpoint. Write
  a new directory, mark it last.
- **Rank-0 gather for large models.** The full state must fit in rank 0's memory and
  is written serially; it scales with model size, not with ranks.
- **Committing from the async thread.** It needs a barrier, which would interleave
  with training's collectives.
- **Default collective timeouts.** A dead peer becomes a 30-minute hang.
- **A fixed "every 30 minutes" interval.** The Young/Daly interval depends on the save
  cost and the failure rate; async saves shrink it from minutes to about one.
- **Restarting faster than the failure rate allows.** In one chaos run, kills landed
  during process-group initialization, each one used a `--max-restarts` slot, and the
  job ran out of restarts at step 48 of 60. Restart time is the next lever after
  async saves, and the restart budget has to cover the failure rate.

## Not built

The save-time table on GPUs and object storage, the 2-hour GPU runs under chaos, and
TorchFT (v7). The goodput-vs-interval curve is a simulation driven by one seeded
failure schedule; the real chaos run is on CPU and short.
