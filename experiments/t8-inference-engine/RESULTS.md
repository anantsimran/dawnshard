# T8: results

CPU only: no GPU, no Hugging Face access, so no official weights, no CUDA graphs, and
no vLLM or SGLang comparison. What ran is the full engine on a tiny random-init Qwen3
(2 layers, 64 wide) with a byte-level tokenizer, on a 4-vCPU machine shared with a
training job. Absolute numbers are CPU numbers; the comparisons between versions are
the result.

## Correctness gate ([test_engine.py](../../app/tests/test_engine.py))

For four prompts (two sharing a 21-token prefix), 12 greedy tokens each, the engine's
output equals a full-sequence forward with no cache, token for token, in every mode:

| Mode | Equal to no-cache greedy | Notes |
|---|---|---|
| continuous batching | yes | |
| chunked prefill (chunk 7, budget 12) + prefix caching | yes | second pass hits the prefix cache |
| static batching | yes | |
| 16-block pool | yes | forces preemption by recompute |

Every block's refcount returns to zero after each run. The official-weight loader maps
every released tensor name and fails on a missing or extra one
(`test_every_official_tensor_name_maps_and_mismatches_fail_loudly`); the comparison
against Hugging Face outputs needs the weights.

## Versions on two traces ([bench.csv](bench.csv))

`uv run python -m training.model.qwen3_serve bench --config tiny`: 60 requests per
trace at 6 requests/s, open-loop, greedy, `ignore_eos`. Trace A: independent prompts.
Trace B: 30% of requests start with one of 4 shared system prompts. SLO for goodput:
TTFT < 2 s, ITL p99 < 250 ms (scaled for CPU). The engine is saturated at this rate;
every version queues.

| Trace | Version | TTFT p50 | TTFT p99 | ITL p99 | Output tokens/s | Prefix hit rate |
|---|---|---|---|---|---|---|
| A | v0 static batching | 23.1 s | 53.2 s | 2.0 s | 22.0 | 0 |
| A | v2 continuous batching | 7.6 s | 14.7 s | 6.7 s | 25.0 | 0 |
| A | v5 + prefix caching | 6.3 s | 12.6 s | 6.9 s | 18.6 | 0.02 |
| A | v6 + chunked prefill 256 | 2.9 s | 19.1 s | 3.6 s | 21.6 | 0.02 |
| A | v6 + chunked prefill 64 | 56.9 s | 96.6 s | 1.8 s | 12.6 | 0.02 |
| B | v0 static batching | 12.8 s | 46.3 s | 1.9 s | 20.0 | 0 |
| B | v2 continuous batching | 1.0 s | 5.0 s | 3.9 s | 25.8 | 0 |
| B | v5 + prefix caching | 4.7 s | 9.7 s | 6.1 s | 32.9 | 0.78 |
| B | v6 + chunked prefill 256 | 6.9 s | 9.4 s | 3.3 s | 31.4 | 0.82 |
| B | v6 + chunked prefill 64 | 23.8 s | 61.6 s | 2.4 s | 15.3 | 0.82 |

- **Continuous batching cuts TTFT 3x on A and 13x on B against static batching**: a
  new request no longer waits for the whole running batch to finish.
- **Prefix caching hits 78-82% on trace B** and raises throughput by 27% (25.8 to 32.9
  tokens/s): shared prompt blocks are not prefilled again.
- **Chunked prefill trades TTFT for ITL.** Smaller chunks keep decodes moving
  (ITL p99 6.7 s, then 3.6 s, then 1.8 s on A) but stretch each prompt over more steps;
  at chunk 64 with a 128-token budget, prefill throughput collapses and TTFT explodes.
  The chunk size is a knob on that trade-off, not a free win.
- **The TTFT gaps between v2 and v5/v6 on B are noise at this saturation.** The CPU
  was shared with a training job and per-step time is dominated by Python bookkeeping,
  which a GPU run would make small next to the model. v0 to v2 and the hit rate are the
  robust results here.

## Not run here

| Ticket item | Needs |
|---|---|
| Correctness gate against Hugging Face for 0.6B and 1.7B | the weights (`--weights`) |
| Goodput vs request rate on a GPU, vLLM and SGLang on the same axes | a GPU; `replay_http` drives any OpenAI-compatible server with the same settings |
| v7 CUDA graphs for decode | a GPU |
| Benchmark settings stated once | in DESIGN.md and `replay_http`: same trace, `max_tokens`, greedy, `ignore_eos`, open-loop |
