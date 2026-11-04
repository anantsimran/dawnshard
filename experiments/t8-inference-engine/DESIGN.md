# T8: inference engine from scratch, design notes

This ticket adds the `serve` package
([app/src/training/serve/](../../app/src/training/serve/)): a paged KV block manager
with prefix caching, an iteration-level scheduler with chunked prefill and
preemption, a ragged-batch runner, per-request sampling with UTF-8-safe streaming,
an OpenAI-compatible endpoint, and a seeded open-loop load generator. The Qwen3
decoder gains `forward_paged` and an official-checkpoint loader
([qwen3.py](../../app/src/training/model/qwen3.py)), and
[qwen3_serve.py](../../app/src/training/model/qwen3_serve.py) is the one command to
serve and the one to benchmark. Numbers are in [RESULTS.md](RESULTS.md).

______________________________________________________________________

## The engine owns memory; the model owns math

`Qwen3.forward_paged(input_ids, positions, attend)` runs a flat batch and calls
`attend(layer, q, k, v)` for attention. The runner's closure writes the new keys and
values into their slots and attends over each request's gathered blocks. The model
never sees a block, and `serve/` never imports `model/` (the layout test enforces
it), so another model only has to implement `forward_paged` and `logits`.

**Alternative considered.** A paged-attention-aware `Qwen3Attention`, as vLLM's model
files do. Simpler per model, but every model then carries the engine's cache layout.

## One scheduling unit: tokens to compute this step

A request in decode needs 1 token, a request in prefill needs the rest of its prompt,
cut to `chunk_size`. Each step fills a token budget, running requests first (oldest
first), then waiting ones. This makes continuous batching, chunked prefill, and
preemption three small rules instead of three code paths, which is how vLLM V1's
scheduler treats them.

- **Preemption by recompute.** When a running request needs a block and none is
  free, the newest running request is freed and requeued at the front, keeping its
  generated tokens. It costs FLOPs on re-prefill; swapping would cost a host-memory
  pool and PCIe bandwidth, and complicate the block manager.
- **Requests that can never fit are rejected at `add_request`.** The first test run
  hung: a 52-token request against a 9-block pool waited forever at the queue head.

## Blocks, refcounts, and chain hashes

16-token blocks; each full block is hashed with its parent's hash, so a hash names
the whole prefix. A block is freed only when its refcount reaches zero; a freed block
whose content is cached goes to an LRU list and is reused on a hit or evicted when the
free list runs out. A 10,000-operation random test checks refcounts against block
tables after every operation and that everything returns to free.

**Alternative considered.** SGLang's radix tree: finer reuse (token-level, any
branch), more code. Hash chains over full blocks are what vLLM does and are enough to
show the TTFT effect.

## Correctness gate before speed

The engine's greedy output must equal a full-sequence forward with no cache, token for
token, under continuous batching, chunked prefill with prefix caching, static
batching, and forced preemption. Official weights load through
`load_official_state_dict`, which fails on any missing or unexpected tensor name and
keeps a tied head tied. The Hugging Face comparison itself needs the weights and is in
"Not run here".

## Routes not to take

- **Static batching.** Every slot waits for the batch's longest request; with outputs
  from 50 to 500 tokens, 45% of slot-steps are wasted. The `static_batching` flag
  exists only as the baseline.
- **A contiguous cache preallocated to max length per request** (T1's layout). Correct,
  but reserves memory for tokens that are never generated, so fewer requests fit.
- **Freeing a block when its first owner finishes.** Another request may share it
  through the prefix cache. Refcounts, always.
- **Hashing a block on its own tokens.** Two requests with the same 16 tokens at
  different positions would share KV computed in different contexts. The hash must
  chain through the parent.
- **Decoding bytes token by token with `bytes.decode`.** A multi-byte character split
  across two tokens raises or prints replacement characters; use an incremental
  decoder and hold incomplete bytes.
- **Checking stop strings on each chunk alone.** "ST" + "OP" never matches; hold back
  `len(stop) - 1` characters.
- **Closed-loop benchmarks.** They slow down with the server and hide the queueing tail.
  The load generator is open-loop and seeded.
- **Comparing engines with different settings.** Same trace, `max_tokens`,
  `ignore_eos`, greedy, warmup excluded, stated once.

## Not built

CUDA graphs (v7) need a GPU. The vLLM and SGLang comparison (v9) and the Hugging Face
correctness gate need the GPU and the weights; `--weights` and
`scripts`-free `replay_http` are ready for them. Attention gathers each request's
blocks and calls SDPA; a paged kernel (FlashInfer) would replace `make_attend`.
