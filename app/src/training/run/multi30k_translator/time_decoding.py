"""
time_decoding.py
Time cached against uncached greedy decoding on the best translator checkpoint.

Run from the repository root:

    uv run python app/src/training/run/multi30k_translator/time_decoding.py

Measures on the device the checkpoint loads to, after a warm-up:

1. End to end: `greedy_translate` and `translate_with_cache` on the first SENTENCES
   test captions, in alternating order, checking that both return the same text.
1. Per step: one decode step at every output position, forced to the full output
   limit so every position is measured, median of POSITION_REPEATS runs.
1. Cache size: the bytes `init_decoder_cache` allocates, against the formula.

Writes `sentences.csv` and `positions.csv` to ASSETS and prints the cache sizes, for
`ablations/reports/kv_cache_timing.md`. Draw the report's figures from those CSVs
afterwards with `ablations/reports/generate_kv_cache_timing.py`.
"""

import csv
import statistics
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import cast

import torch
from training.constants import REPO_ROOT
from training.dal.dao.sqlite_dao import SqliteDao
from training.dataload.constants import CACHE_PATH
from training.dataload.multi30k import get_multi30k_texts
from training.model.transformer_translator import TransformerTranslator
from training.preprocess.constants import BOS_ID
from training.preprocess.tokenizer import Tokenizer
from training.run.multi30k_translator.main import (
    BEST_TRANSLATOR_CHECKPOINT_PATH,
    BEST_TRANSLATOR_CONFIG_PATH,
    encode_source,
    greedy_translate,
    init_decoder_cache,
    translate_with_cache,
)
from training.run.multi30k_translator.setup import load_transformer
from training.transformer.masks import mask_later_positions

ASSETS = REPO_ROOT / "app" / "src" / "training" / "ablations" / "reports" / "assets"
ASSETS = ASSETS / "kv_cache_timing"
SENTENCES = 200
WARMUP = 10
POSITION_REPEATS = 20


def synchronize(device: torch.device) -> None:
    """Wait for queued GPU work, so a wall-clock reading covers it."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def timed_ms(fn: Callable[[], object], device: torch.device) -> tuple[float, object]:
    """Return `fn`'s wall-clock time in milliseconds and its result."""
    synchronize(device=device)
    start = time.perf_counter()
    result = fn()
    synchronize(device=device)
    return (time.perf_counter() - start) * 1000, result


@torch.no_grad()
def time_sentences(
    model: TransformerTranslator,
    device: torch.device,
    sentences: list[str],
    tokenizers: tuple[Tokenizer, Tokenizer],
) -> list[dict]:
    """Time both translate functions on each sentence and record whether they agree.

    The two run in alternating order, so neither always gets a warmer device.
    """
    src_tokenizer, tgt_tokenizer = tokenizers
    rows = []
    for index, sentence in enumerate(sentences):  # noqa: NAR001
        arms = [("uncached", greedy_translate), ("cached", translate_with_cache)]
        results = {}
        for name, translate in arms if index % 2 == 0 else arms[::-1]:
            call = partial(  # noqa: NAR001
                translate,
                src_sentence=sentence,
                transformer_model=model,
                device=device,
                src_tokenizer=src_tokenizer,
                tgt_tokenizer=tgt_tokenizer,
            )
            results[name] = timed_ms(fn=call, device=device)
        rows.append(  # noqa: NAR001
            {
                "index": index,
                "source": sentence,
                "uncached_ms": round(results["uncached"][0], 3),  # noqa: NAR001
                "cached_ms": round(results["cached"][0], 3),  # noqa: NAR001
                "same_output": results["uncached"][1] == results["cached"][1],
                "translation": results["cached"][1],
            }
        )
    return rows


@torch.no_grad()
def time_positions(
    model: TransformerTranslator, device: torch.device, memory: torch.Tensor, mask: torch.Tensor
) -> list[dict]:
    """Return the median time of one decode step at each output position, both ways.

    Every step feeds BOS_ID: a step's cost depends on how many positions it runs, not
    on which ids they hold, and a fixed id keeps EOS from ending a run early. Each
    step reads its argmax back with `.item()`, as the translate functions do.
    """
    limit = model.tgt_seq_length
    ids = torch.full(size=(1, limit), fill_value=BOS_ID, dtype=torch.int64, device=device)
    causal = mask_later_positions(seq_len=limit, device=device)
    pe = model.positional_embedding.pe
    uncached = [[] for _ in range(limit)]  # noqa: NAR001
    cached = [[] for _ in range(limit)]  # noqa: NAR001
    for _ in range(POSITION_REPEATS):  # noqa: NAR001
        for t in range(1, limit + 1):  # noqa: NAR001
            ms, _ = timed_ms(
                fn=lambda t=t: (
                    model.decode(
                        tgt_in=ids[:, :t],
                        pos_encoding=pe[:t],
                        memory=memory,
                        decoder_mask=causal[:, :, :t, :t],
                        memory_mask=mask,
                    )[0][:, -1]
                    .argmax(dim=-1)
                    .item()
                ),
                device=device,
            )
            uncached[t - 1].append(ms)  # noqa: NAR001
        cache = init_decoder_cache(transformer_model=model, memory=memory)
        for t in range(limit):  # noqa: NAR001
            ms, _ = timed_ms(
                fn=lambda t=t: (
                    model.decode(
                        tgt_in=ids[:, t : t + 1],
                        pos_encoding=model.positional_embedding.at_pos(pos=t).unsqueeze(dim=0),
                        memory=memory,
                        decoder_mask=None,
                        memory_mask=mask,
                        decoder_cache=cache,
                    )[0][:, -1]
                    .argmax(dim=-1)
                    .item()
                ),
                device=device,
            )
            cached[t].append(ms)  # noqa: NAR001
    return [
        {
            "position": t + 1,
            "uncached_ms": round(statistics.median(uncached[t]), 3),  # noqa: NAR001
            "cached_ms": round(statistics.median(cached[t]), 3),  # noqa: NAR001
        }
        for t in range(limit)  # noqa: NAR001
    ]


def cache_bytes(model: TransformerTranslator, memory: torch.Tensor) -> dict[str, int]:
    """Return the bytes one decode's caches allocate, measured and by formula.

    The formula is `2 (K and V) * blocks * d_model * capacity * bytes per element`,
    with capacity `tgt_seq_length` for self-attention and S for cross-attention.
    """
    caches = init_decoder_cache(transformer_model=model, memory=memory)
    measured = {
        kind: sum(  # noqa: NAR001
            t.numel() * t.element_size()
            for c in caches
            for t in (getattr(c, kind).k, getattr(c, kind).v)  # noqa: NAR001
        )
        for kind in ("self_attn", "cross_attn")
    }
    per_row = 2 * len(caches) * memory.shape[2] * memory.element_size()  # noqa: NAR001
    return {
        "self_measured": measured["self_attn"],
        "self_formula": per_row * model.tgt_seq_length,
        "cross_measured": measured["cross_attn"],
        "cross_formula": per_row * memory.shape[1],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write `rows` to `path` with a header taken from the first row's keys."""
    with path.open(mode="w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))  # noqa: NAR001
        writer.writeheader()
        writer.writerows(rowdicts=rows)


def main() -> None:
    """Load the best checkpoint, run all three measurements, and write the CSVs.

    Side effects:
        Writes `sentences.csv` and `positions.csv` under ASSETS, overwriting the
        report's checked-in data.
    """
    train_state, epoch_spec, config, tokenizers = load_transformer(
        config_path=BEST_TRANSLATOR_CONFIG_PATH,
        checkpoint_path=BEST_TRANSLATOR_CHECKPOINT_PATH,
    )
    model = cast(typ=TransformerTranslator, val=train_state.model)
    model.eval()
    device = epoch_spec.device
    with SqliteDao(path=CACHE_PATH) as cache:
        test = get_multi30k_texts(lang=config.data.src_lang, cache=cache).test
    time_sentences(model=model, device=device, sentences=test[:WARMUP], tokenizers=tokenizers)
    sentences = time_sentences(
        model=model, device=device, sentences=test[:SENTENCES], tokenizers=tokenizers
    )
    memory, mask = encode_source(
        src_sentence=test[0],
        transformer_model=model,
        device=device,
        src_tokenizer=tokenizers[0],
    )
    positions = time_positions(model=model, device=device, memory=memory, mask=mask)
    sizes = cache_bytes(model=model, memory=memory)

    ASSETS.mkdir(parents=True, exist_ok=True)
    write_csv(path=ASSETS / "sentences.csv", rows=sentences)
    write_csv(path=ASSETS / "positions.csv", rows=positions)

    uncached = [row["uncached_ms"] for row in sentences]
    cached = [row["cached_ms"] for row in sentences]
    print(f"device {device}, {len(sentences)} sentences")  # noqa: NAR001
    print(f"identical outputs: {sum(row['same_output'] for row in sentences)}")  # noqa: NAR001
    print(  # noqa: NAR001
        f"ms/sentence median: uncached {statistics.median(uncached):.1f}, "  # noqa: NAR001
        f"cached {statistics.median(cached):.1f}"  # noqa: NAR001
    )
    print(f"total s: uncached {sum(uncached) / 1000:.2f}, cached {sum(cached) / 1000:.2f}")  # noqa: NAR001
    print(f"position 1: {positions[0]}, position {len(positions)}: {positions[-1]}")  # noqa: NAR001
    print(f"source length S={memory.shape[1]}, cache bytes {sizes}")  # noqa: NAR001


if __name__ == "__main__":
    main()
