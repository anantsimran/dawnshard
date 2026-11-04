"""
shards.py
Offline tokenization into flat uint32 token shards with a document index and a manifest.

A corpus is tokenized once, before training, so the training loop never sees text.
Documents are concatenated, each followed by an EOS id, and cut into shards of about
`tokens_per_shard` tokens; a document may straddle two shards. Each shard is two files:

- `<prefix>_<n>.bin`: the token ids, uint32, readable with a memmap or a byte-range GET.
- `<prefix>_<n>.idx`: uint64 offsets, within the shard, where each document starts.

`manifest.json` lists every shard and its token count in stream order; a reader turns
a global token offset into (shard, offset) from it. Writing the manifest last means a
directory without one is an unfinished write.

`pack` turns any token stream into fixed `seq_len + 1` windows. Documents are packed
back to back, so a window can hold the end of one and the start of the next; the EOS
between them is what `transformer.mask.document_keep_mask` uses to keep them apart.

Tokenize one file from the command line (the script `scripts/tokenize_shards.sh` runs
this over many files in parallel):

    uv run python -m training.dataload.shards --input docs.txt --out shards/ \\
        --prefix docs --merges merges.txt --eos-id 261
"""

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from loguru import logger
from training.common import bpe
from training.dataload.tokens import TOKEN_DTYPE

MANIFEST = "manifest.json"


def write_shards(
    documents: Iterable[list[int]],
    eos_id: int,
    out_dir: Path,
    prefix: str,
    tokens_per_shard: int,
) -> list[dict]:
    """Write tokenized `documents` into shards under `out_dir`; return their manifest entries.

    Args:
        documents: Token ids per document, from any tokenizer.
        prefix: Shard file prefix, so several inputs can share one directory.

    Side effects:
        Writes `<prefix>_<n>.bin` and `.idx` files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    buffer: list[int] = []
    starts: list[int] = []
    for document in documents:
        starts.append(len(buffer))  # noqa: NAR001
        buffer.extend(document)  # noqa: NAR001
        buffer.append(eos_id)  # noqa: NAR001
        while len(buffer) >= tokens_per_shard:  # noqa: NAR001
            entries.append(  # noqa: NAR001
                _flush(
                    tokens=buffer[:tokens_per_shard],
                    starts=[s for s in starts if s < tokens_per_shard],
                    out_dir=out_dir,
                    name=f"{prefix}_{len(entries):05d}",  # noqa: NAR001
                )
            )  # noqa: NAR001
            buffer = buffer[tokens_per_shard:]
            starts = [s - tokens_per_shard for s in starts if s >= tokens_per_shard]
    if buffer:
        entries.append(  # noqa: NAR001
            _flush(
                tokens=buffer,
                starts=starts,
                out_dir=out_dir,
                name=f"{prefix}_{len(entries):05d}",  # noqa: NAR001
            )
        )  # noqa: NAR001
    return entries


def _flush(tokens: list[int], starts: list[int], out_dir: Path, name: str) -> dict:
    """Write one shard's token and index files; return its manifest entry."""
    np.asarray(tokens, dtype=TOKEN_DTYPE).tofile(out_dir / f"{name}.bin")  # noqa: NAR001
    np.asarray(starts, dtype=np.uint64).tofile(out_dir / f"{name}.idx")  # noqa: NAR001
    return {"name": f"{name}.bin", "num_tokens": len(tokens), "num_documents": len(starts)}  # noqa: NAR001


def write_manifest(out_dir: Path) -> dict:
    """Collect every shard's entry in `out_dir`, in name order, into `manifest.json`.

    Run once after every tokenize job has finished; readers treat the manifest as
    the list of shards.

    Side effects:
        Writes `out_dir/manifest.json`.
    """
    shards = []
    for path in sorted(out_dir.glob(pattern="*.bin")):  # noqa: NAR001
        num_documents = (out_dir / f"{path.stem}.idx").stat().st_size // 8
        shards.append(  # noqa: NAR001
            {
                "name": path.name,
                "num_tokens": path.stat().st_size // 4,
                "num_documents": num_documents,
            }
        )  # noqa: NAR001
    manifest = {
        "dtype": "uint32",
        "shards": shards,
        "num_tokens": sum(s["num_tokens"] for s in shards),  # noqa: NAR001
    }  # noqa: NAR001
    (out_dir / MANIFEST).write_text(data=json.dumps(obj=manifest, indent=2))
    return manifest


def pack(stream: Iterable[int], seq_len: int) -> Iterator[np.ndarray]:
    """Yield consecutive non-overlapping `seq_len + 1` windows of `stream`, EOS ids kept.

    The last partial window is dropped; every window is full, so no padding exists.
    """
    window: list[int] = []
    for token in stream:
        window.append(token)  # noqa: NAR001
        if len(window) == seq_len + 1:  # noqa: NAR001
            yield np.asarray(window, dtype=np.int64)  # noqa: NAR001
            window = window[seq_len:]


def main() -> None:
    """Tokenize one text file (one document per non-empty line) with BPE merges into shards."""
    parser = argparse.ArgumentParser(description="Tokenize a text file into uint32 shards.")
    parser.add_argument("--input", type=Path, required=True)  # noqa: NAR001
    parser.add_argument("--out", type=Path, required=True)  # noqa: NAR001
    parser.add_argument("--prefix", required=True)  # noqa: NAR001
    parser.add_argument(  # noqa: NAR001
        "--merges", type=Path, required=True, help="BPE merges: 'first second' per line."
    )  # noqa: NAR001
    parser.add_argument("--eos-id", type=int, required=True)  # noqa: NAR001
    parser.add_argument("--tokens-per-shard", type=int, default=2**24)  # noqa: NAR001
    parser.add_argument("--manifest", action="store_true", help="Only write manifest.json.")  # noqa: NAR001
    args = parser.parse_args()
    if args.manifest:
        logger.info("manifest: {} tokens", write_manifest(out_dir=args.out)["num_tokens"])  # noqa: NAR001
        return
    lines = args.merges.read_text(encoding="utf-8").splitlines()
    merges, _ = bpe.tokenizer_from_pairs(
        pairs=[(int(a), int(b)) for a, b in (line.split() for line in lines)]  # noqa: NAR001
    )  # noqa: NAR001
    documents = [
        line for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    entries = write_shards(
        documents=bpe.encode_many(texts=documents, merges=merges),
        eos_id=args.eos_id,
        out_dir=args.out,
        prefix=args.prefix,
        tokens_per_shard=args.tokens_per_shard,
    )
    logger.info("{}: {} documents into {} shards", args.input, len(documents), len(entries))  # noqa: NAR001


if __name__ == "__main__":
    main()
