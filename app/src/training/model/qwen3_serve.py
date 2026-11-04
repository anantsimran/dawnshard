"""
qwen3_serve.py
Serve the Qwen3 decoder with the `serve` engine, and benchmark engine versions.

`build_engine` puts a Qwen3 model behind the engine: random init from a config, or a
released checkpoint's safetensors files through `load_official_state_dict`. The
tokenizer is byte-level (one id per UTF-8 byte), which needs no download and keeps
multi-byte characters split across tokens, the case incremental detokenization
exists for. A released checkpoint needs its own tokenizer; `--tokenizer-json` loads
one with the `tokenizers` package when it is installed.

    uv run python -m training.model.qwen3_serve serve --config tiny --port 8000
    uv run python -m training.model.qwen3_serve bench --config tiny
"""

import argparse
import csv
import importlib
from dataclasses import replace
from pathlib import Path

import torch
from training.constants import REPO_ROOT
from training.model.qwen3 import CONFIGS, Qwen3, Qwen3Config, load_official_state_dict
from training.serve.api import serve
from training.serve.block_manager import BlockManager
from training.serve.engine import Engine
from training.serve.load_gen import make_trace, replay_engine, summarize
from training.serve.runner import KVPool
from training.serve.scheduler import SchedulerConfig

RESULTS_DIR = REPO_ROOT / "experiments" / "t8-inference-engine"
BYTE_EOS = 0  # the NUL byte never appears in text, so it can end a sequence


def encode_bytes(text: str) -> list[int]:
    """Byte-level tokenizer: one id per UTF-8 byte."""
    return list(text.encode(encoding="utf-8"))  # noqa: NAR001


def token_bytes(token: int) -> bytes:
    """The byte an id stands for; ids past 255 decode to nothing."""
    return bytes([token]) if token < 256 else b""  # noqa: NAR001


def build_engine(
    config: Qwen3Config,
    num_blocks: int,
    scheduler: SchedulerConfig,
    prefix_caching: bool,
    weights: list[Path] | None = None,
    block_size: int = 16,
) -> Engine:
    """Return an engine over a Qwen3 model, random init or loaded from safetensors files."""
    torch.manual_seed(seed=0)
    model = Qwen3(config=config).eval()
    if weights:
        safetensors = importlib.import_module(name="safetensors.torch")
        tensors = {k: v for path in weights for k, v in safetensors.load_file(str(path)).items()}  # noqa: NAR001
        load_official_state_dict(model=model, tensors=tensors)
    pool = KVPool(
        num_layers=config.num_hidden_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
    )
    blocks = BlockManager(
        num_blocks=num_blocks, block_size=block_size, prefix_caching=prefix_caching
    )
    return Engine(
        model=model,
        pool=pool,
        blocks=blocks,
        config=scheduler,
        eos_id=BYTE_EOS,
        token_bytes=token_bytes,
        max_model_len=config.max_position_embeddings,
    )


VERSIONS = {
    "v0 static batching": (
        SchedulerConfig(static_batching=True, chunk_size=4096, max_tokens_per_step=4096),
        False,
    ),
    "v2 continuous batching": (SchedulerConfig(chunk_size=4096, max_tokens_per_step=4096), False),
    "v5 + prefix caching": (SchedulerConfig(chunk_size=4096, max_tokens_per_step=4096), True),
    "v6 + chunked prefill 256": (SchedulerConfig(chunk_size=256, max_tokens_per_step=512), True),
    "v6 + chunked prefill 64": (SchedulerConfig(chunk_size=64, max_tokens_per_step=128), True),
}


def bench(config: Qwen3Config, num_requests: int, rate: float) -> list[dict]:
    """Replay traces A (independent prompts) and B (30% shared prefixes) through every version."""
    traces = {
        "A": make_trace(
            seed=0,
            num_requests=num_requests,
            rate=rate,
            prompt_words=(5, 60),
            output_tokens=(4, 48),
        ),
        "B": make_trace(
            seed=1,
            num_requests=num_requests,
            rate=rate,
            prompt_words=(5, 30),
            output_tokens=(4, 32),
            shared_prefix_share=0.3,
            num_prefixes=4,
            prefix_words=60,
        ),
    }
    rows = []
    for trace_name, trace in traces.items():
        for version, (scheduler, prefix) in VERSIONS.items():
            engine = build_engine(
                config=config, num_blocks=2048, scheduler=scheduler, prefix_caching=prefix
            )
            records, duration = replay_engine(trace=trace, engine=engine, encode=encode_bytes)
            blocks = engine.blocks
            hit_rate = blocks.hits / blocks.lookups if blocks.lookups else 0.0
            rows.append(  # noqa: NAR001
                {
                    "trace": trace_name,
                    "version": version,
                    **summarize(records=records, duration=duration, ttft_slo=2.0, itl_slo=0.25),
                    "prefix_hit_rate": round(hit_rate, 3),  # noqa: NAR001
                    "preemptions": engine.scheduler.preemptions,
                }
            )  # noqa: NAR001
            print(rows[-1])  # noqa: NAR001
    return rows


def main() -> None:
    """`serve` starts the HTTP endpoint; `bench` replays the traces through each version."""
    parser = argparse.ArgumentParser(description="Serve or benchmark Qwen3 on the serve engine.")
    parser.add_argument("command", choices=["serve", "bench"])  # noqa: NAR001
    parser.add_argument("--config", default="tiny", choices=list(CONFIGS))  # noqa: NAR001
    parser.add_argument("--weights", type=Path, nargs="*", help="Official safetensors files.")  # noqa: NAR001
    parser.add_argument("--port", type=int, default=8000)  # noqa: NAR001
    parser.add_argument("--num-blocks", type=int, default=4096)  # noqa: NAR001
    parser.add_argument("--requests", type=int, default=60)  # noqa: NAR001
    parser.add_argument("--rate", type=float, default=6.0)  # noqa: NAR001
    args = parser.parse_args()
    base = CONFIGS[args.config]
    config = replace(  # noqa: NAR001
        base,
        vocab_size=max(base.vocab_size, 256),  # noqa: NAR001
        max_position_embeddings=max(base.max_position_embeddings, 4096),  # noqa: NAR001
    )
    if args.command == "serve":
        engine = build_engine(
            config=config,
            num_blocks=args.num_blocks,
            scheduler=SchedulerConfig(),
            prefix_caching=True,
            weights=args.weights,
        )
        server = serve(
            engine=engine,
            encode=encode_bytes,
            model_name=f"qwen3-{args.config}",
            host="0.0.0.0",
            port=args.port,
        )
        print(f"serving on :{args.port}")  # noqa: NAR001
        server.serve_forever()
        return
    rows = bench(config=config, num_requests=args.requests, rate=args.rate)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(file=RESULTS_DIR / "bench.csv", mode="w", newline="") as handle:
        writer = csv.DictWriter(f=handle, fieldnames=list(rows[0]))  # noqa: NAR001
        writer.writeheader()
        writer.writerows(rowdicts=rows)


if __name__ == "__main__":
    main()
