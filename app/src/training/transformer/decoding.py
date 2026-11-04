"""Autoregressive decoding: one loop, with greedy and beam search as strategies.

`generate` owns the loop: call the model on the tokens so far, turn the logits into
log-probs, ask the strategy which token each row emits and which row it continues,
append, and reorder the cache to match. The strategy owns everything else: how many
rows a sentence takes, when a hypothesis is finished, and which one wins. Caching is
not a strategy concern: `generate` takes an optional `KVCache`, and the model's step
function decides from it whether to feed the whole prefix or only the newest token.

Rows are flattened sentence-major: row `b * beam_size + j` is beam `j` of sentence
`b`, so a batch of sentences decodes in one loop whatever the beam width.
"""

from collections.abc import Callable
from typing import Optional, Protocol

import torch
from training.transformer.kv_cache import KVCache

# (tokens so far (N, t), cache or None) -> next-token logits (N, vocab). With a cache,
# the function feeds only tokens[:, cache.length:]; without one, all of `tokens`.
type StepFn = Callable[[torch.Tensor, Optional[KVCache]], torch.Tensor]


def length_penalty(length: int, alpha: float) -> float:
    """GNMT length penalty, `((5 + length) / 6) ** alpha`; divide a summed log-prob by it.

    Every token adds a negative log-prob, so unnormalized beam scores favor short
    outputs. alpha = 0 turns the penalty off; alpha = 1 is close to a per-token mean.
    """
    return ((5 + length) / 6) ** alpha


class Strategy(Protocol):
    """What `generate` asks of a decoding strategy."""

    beam_size: int

    def start(self, batch_size: int, device: torch.device) -> None:
        """Reset for `batch_size` sentences of `beam_size` rows each."""
        ...

    def step(
        self, log_probs: torch.Tensor, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return `(next_tokens (N,), parent_rows (N,) or None for identity)`."""
        ...

    def done(self) -> bool:
        """Whether every sentence is finished, so the loop can stop early."""
        ...

    def results(self, tokens: torch.Tensor) -> list[list[int]]:
        """Return each sentence's best output ids, without BOS or EOS."""
        ...


def generate(
    step_fn: StepFn,
    strategy: Strategy,
    batch_size: int,
    bos_id: int,
    max_len: int,
    device: torch.device,
    cache: Optional[KVCache] = None,
) -> list[list[int]]:
    """Decode `batch_size` sentences, one token per step, until done or `max_len`.

    Args:
        step_fn: Model step; see `StepFn`. Expand any per-sentence input such as
            encoder memory to `batch_size * strategy.beam_size` rows before building it.
        bos_id: First input token of every row.
        max_len: Most output tokens per sentence, EOS included.
        cache: Built for `batch_size * strategy.beam_size` rows, or None to recompute
            the whole prefix every step.

    Returns:
        One id list per sentence, in input order.

    Side effects:
        Fills and reorders `cache`; calls `strategy.start` first.
    """
    strategy.start(batch_size=batch_size, device=device)
    rows = batch_size * strategy.beam_size
    tokens = torch.full(size=(rows, 1), fill_value=bos_id, dtype=torch.int64, device=device)
    for _ in range(max_len):  # noqa: NAR001
        logits = step_fn(tokens, cache)  # noqa: NAR001
        log_probs = torch.log_softmax(input=logits.float(), dim=-1)
        next_tokens, parents = strategy.step(log_probs=log_probs, tokens=tokens)
        if parents is not None:
            tokens = tokens.index_select(dim=0, index=parents)
            if cache is not None:
                cache.reorder(index=parents)
        tokens = torch.cat(tensors=[tokens, next_tokens.unsqueeze(dim=1)], dim=1)
        if strategy.done():
            break
    return strategy.results(tokens=tokens)


def _strip(ids: list[int], eos_id: int) -> list[int]:
    """Return `ids` up to, not including, the first `eos_id`."""
    return ids[: ids.index(eos_id)] if eos_id in ids else ids  # noqa: NAR001


class Greedy:
    """Emit each row's most likely token; a row that emitted EOS emits padding after."""

    beam_size = 1

    def __init__(self, eos_id: int, pad_id: int) -> None:
        """Remember the ids that end a row and fill it afterwards."""
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.finished = torch.zeros(size=(0,), dtype=torch.bool)

    def start(self, batch_size: int, device: torch.device) -> None:
        """Mark every row unfinished.

        Side effects:
            Resets `finished`.
        """
        self.finished = torch.zeros(size=(batch_size,), dtype=torch.bool, device=device)

    def step(self, log_probs: torch.Tensor, tokens: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Return the argmax per row and no reorder.

        Side effects:
            Marks rows that just emitted EOS as finished.
        """
        next_tokens = log_probs.argmax(dim=-1)
        next_tokens = next_tokens.masked_fill(mask=self.finished, value=self.pad_id)
        self.finished |= next_tokens == self.eos_id
        return next_tokens, None

    def done(self) -> bool:
        """Whether every row has emitted EOS."""
        return bool(self.finished.all())  # noqa: NAR001

    def results(self, tokens: torch.Tensor) -> list[list[int]]:
        """Each row without its BOS, cut at EOS."""
        return [_strip(ids=row[1:], eos_id=self.eos_id) for row in tokens.tolist()]


class BeamSearch:
    """Keep the `beam_size` best live hypotheses per sentence by summed log-prob.

    Each step scores every (beam, token) extension, takes the top `2 * beam_size`,
    and walks them best first: an EOS among the top `beam_size` moves its hypothesis
    to the sentence's finished set, scored with the length penalty; any other token
    fills the next live slot, until `beam_size` are live. Taking twice the width
    guarantees enough non-EOS candidates. A sentence is done once it has `beam_size`
    finished hypotheses. At `max_len`, live hypotheses join the finished set as they
    are, and the best normalized score wins.

    Finished hypotheses are copied out of the token tensor when they finish, so they
    stop growing; the rows they leave behind keep decoding live hypotheses.
    """

    def __init__(self, beam_size: int, eos_id: int, pad_id: int, alpha: float = 0.0) -> None:
        """Set the width and the length penalty.

        Args:
            alpha: GNMT length-penalty exponent; see `length_penalty`.
        """
        self.beam_size = beam_size
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.alpha = alpha
        self.scores = torch.zeros(size=(0, beam_size))
        self.finished: list[list[tuple[float, list[int]]]] = []

    def start(self, batch_size: int, device: torch.device) -> None:
        """Give each sentence one live beam; the others start at -inf so step 0 doesn't
        expand `beam_size` copies of BOS.

        Side effects:
            Resets `scores` and `finished`.
        """
        self.scores = torch.full(size=(batch_size, self.beam_size), fill_value=-torch.inf)
        self.scores[:, 0] = 0.0
        self.scores = self.scores.to(device=device)
        self.finished = [[] for _ in range(batch_size)]  # noqa: NAR001

    def _sentence_done(self, sentence: int) -> bool:
        """Whether `sentence` has `beam_size` finished hypotheses."""
        return len(self.finished[sentence]) >= self.beam_size  # noqa: NAR001

    def step(
        self, log_probs: torch.Tensor, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extend every live beam by one token and return the new rows' tokens and parents.

        Side effects:
            Updates `scores` and appends to `finished`.
        """
        batch_size, k = self.scores.shape
        vocab = log_probs.shape[-1]
        candidates = self.scores.unsqueeze(dim=-1) + log_probs.view(batch_size, k, vocab)  # noqa: NAR001
        top_scores, top_index = candidates.view(batch_size, k * vocab).topk(k=2 * k, dim=-1)  # noqa: NAR001
        rows = torch.arange(end=batch_size * k, device=log_probs.device)
        next_tokens = torch.full_like(input=rows, fill_value=self.pad_id)
        parents = rows.clone()
        new_scores = torch.full_like(input=self.scores, fill_value=-torch.inf)
        for sentence in range(batch_size):  # noqa: NAR001
            if self._sentence_done(sentence=sentence):
                continue
            live = 0
            for rank, (score, index) in enumerate(  # noqa: NAR001
                zip(top_scores[sentence].tolist(), top_index[sentence].tolist())  # noqa: NAR001
            ):
                parent, token = sentence * k + index // vocab, index % vocab
                if token == self.eos_id:
                    if rank < k:
                        self._finish(
                            sentence=sentence, score=score, ids=tokens[parent, 1:].tolist()
                        )
                    continue
                parents[sentence * k + live] = parent
                next_tokens[sentence * k + live] = token
                new_scores[sentence, live] = score
                live += 1
                if live == k:
                    break
        self.scores = new_scores
        return next_tokens, parents

    def _finish(self, sentence: int, score: float, ids: list[int]) -> None:
        """Add `ids` + EOS to `sentence`'s finished set, length-normalized.

        Side effects:
            Appends to `finished[sentence]`.
        """
        penalty = length_penalty(length=len(ids) + 1, alpha=self.alpha)  # noqa: NAR001
        self.finished[sentence].append((score / penalty, ids))  # noqa: NAR001

    def done(self) -> bool:
        """Whether every sentence has `beam_size` finished hypotheses."""
        return all(self._sentence_done(sentence=s) for s in range(len(self.finished)))  # noqa: NAR001

    def results(self, tokens: torch.Tensor) -> list[list[int]]:
        """Each sentence's best finished hypothesis, after adding unfinished live ones.

        Side effects:
            Appends each unfinished sentence's live beams to `finished`.
        """
        k = self.beam_size
        for sentence, hypotheses in enumerate(self.finished):  # noqa: NAR001
            if self._sentence_done(sentence=sentence):
                continue
            for beam, score in enumerate(self.scores[sentence].tolist()):  # noqa: NAR001
                if score > -torch.inf:
                    ids = tokens[sentence * k + beam, 1:].tolist()
                    penalty = length_penalty(length=len(ids), alpha=self.alpha)  # noqa: NAR001
                    hypotheses.append((score / penalty, ids))  # noqa: NAR001
        return [
            max(hypotheses, key=lambda scored: scored[0])[1]  # noqa: NAR001
            for hypotheses in self.finished
        ]
