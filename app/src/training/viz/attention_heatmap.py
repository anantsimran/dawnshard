"""
attention_heatmap.py
Plot per-head attention weights as a grid of heatmaps, one row per sentence.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from beartype import beartype
from training.common import bpe
from jaxtyping import Float, Int, jaxtyped
from loguru import logger
from training.transformer.constants import B, H, L

MAX_TOKENS = 24


def _token_labels(ids: list[int], vocab: dict[int, str]) -> list[str]:
    """Tick labels for `ids`, with spaces shown as '·' so word boundaries stay visible."""
    return [vocab[i].replace(" ", "·") for i in ids]  # noqa: NAR001


@jaxtyped(typechecker=beartype)
def plot_attention_heads(
    weights: Float[torch.Tensor, f"{B} {H} {L} {L}"],
    token_ids: Int[torch.Tensor, f"{B} {L}"],
    vocab: dict[int, str],
    title: str,
    path: Path,
    max_tokens: int = MAX_TOKENS,
) -> None:
    """Save a grid of attention heatmaps: rows are sentences, columns are heads.

    Within a cell, rows are query tokens and columns are the key tokens they attend
    to. Every cell shares one colour scale so heads can be compared. Pads are
    dropped and each sentence is cropped to its first `max_tokens` tokens to keep
    the labels legible, so a cropped row no longer sums to 1.

    Args:
        weights: Attention weights from one layer, one batch row per sentence.
        token_ids: The ids those weights came from, right-padded with bpe.PAD_ID.
        vocab: Maps ids to token strings, as returned by get_ag_news_dataloader.
        title: Figure title, e.g. which layer the weights came from.
        path: Where to write the PNG; parent directories are created.
        max_tokens: Keep at most this many leading tokens per sentence.
    """
    weights = weights.detach().float().cpu()
    lengths = (token_ids != bpe.PAD_ID).sum(dim=1).clamp(max=max_tokens).tolist()
    n_sentences, n_heads = weights.shape[0], weights.shape[1]
    vmax = max(weights[s, :, :n, :n].max().item() for s, n in enumerate(lengths))  # noqa: NAR001
    fig, axes = plt.subplots(
        nrows=n_sentences,
        ncols=n_heads,
        figsize=(2.4 * n_heads, 2.8 * n_sentences),
        squeeze=False,
        layout="constrained",
    )
    for s, n in enumerate(lengths):  # noqa: NAR001
        labels = _token_labels(ids=token_ids[s, :n].tolist(), vocab=vocab)
        for head in range(n_heads):  # noqa: NAR001
            ax = axes[s][head]
            # aspect="auto" fills the axes box that constrained layout reserved, so
            # tick labels can't spill into the row below.
            ax.imshow(
                X=weights[s, head, :n, :n].numpy(),
                cmap="Blues",
                vmin=0.0,
                vmax=vmax,
                aspect="auto",
            )
            ax.set_xticks(ticks=range(n), labels=labels, rotation=90, fontsize=5)  # noqa: NAR001
            ax.set_yticks(ticks=range(n), labels=labels, fontsize=5)  # noqa: NAR001
            ax.tick_params(labelleft=head == 0)
            ax.set_title(label=f"sentence {s}, head {head}", fontsize=7)
    fig.colorbar(mappable=axes[0][0].images[0], ax=axes, shrink=0.6, label="attention weight")
    fig.suptitle(t=title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fname=path, dpi=150)
    plt.close(fig=fig)
    logger.info("Saved plot to {}", path)  # noqa: NAR001


def plot_attention_probes(probe_dir: Path, vocab: dict[int, str]) -> None:
    """Plot every capture `save_attention_maps` wrote to `probe_dir`.

    Each `epoch_<NNN>.pt` file holds `{"token_ids": (n, L), "weights": [(n, h, L, L)
    per block]}`. Writes one `epoch_<NNN>_block_<i>.png` per block beside it.

    Args:
        probe_dir: Directory the probe wrote to.
        vocab: Maps ids to token strings, as returned by get_ag_news_dataloader.
    """
    for capture_path in sorted(probe_dir.glob(pattern="*.pt")):  # noqa: NAR001
        capture = torch.load(f=capture_path)
        for block_index, weights in enumerate(capture["weights"]):  # noqa: NAR001
            plot_attention_heads(
                weights=weights,
                token_ids=capture["token_ids"],
                vocab=vocab,
                title=f"{capture_path.stem}, block {block_index}",
                path=probe_dir / f"{capture_path.stem}_block_{block_index}.png",
            )
