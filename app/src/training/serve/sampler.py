"""Per-request sampling, incremental detokenization, and stop strings.

Each request carries its own `SamplingParams` and, when seeded, its own generator, so
one request's draws never depend on which others share its batch. Greedy is
temperature 0.

Text is produced byte-level: a token can end in the middle of a multi-byte UTF-8
character, so `IncrementalDetokenizer` holds incomplete bytes until the character is
whole. A stop string can arrive split across two tokens, so `StopChecker` holds back
the last `len(longest stop) - 1` characters until it knows they are not the start of
a stop.
"""

import codecs
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass(frozen=True)
class SamplingParams:
    """How one request picks tokens and when it stops."""

    max_tokens: int = 64
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: Optional[int] = None
    stop: tuple[str, ...] = ()
    ignore_eos: bool = False


def sample(
    logits: torch.Tensor, params: SamplingParams, generator: Optional[torch.Generator]
) -> int:
    """Pick one token from `(vocab,)` logits: greedy, or temperature with top-k and top-p."""
    if params.temperature == 0.0:
        return int(logits.argmax())  # noqa: NAR001
    logits = logits.float() / params.temperature
    if params.top_k:
        kth = logits.topk(k=params.top_k).values[-1]
        logits = logits.masked_fill(mask=logits < kth, value=-torch.inf)
    if params.top_p < 1.0:
        sorted_logits, order = logits.sort(descending=True)
        cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        drop = (
            cumulative - sorted_logits.softmax(dim=-1) > params.top_p
        )  # keep the token that crosses p
        logits = logits.scatter(
            dim=0, index=order, src=sorted_logits.masked_fill(mask=drop, value=-torch.inf)
        )
    return int(torch.multinomial(input=logits.softmax(dim=-1), num_samples=1, generator=generator))  # noqa: NAR001


class IncrementalDetokenizer:
    """Turn a token stream into text without splitting a UTF-8 character."""

    def __init__(self, token_bytes: Callable[[int], bytes]) -> None:
        """`token_bytes` maps one id to its raw bytes."""
        self.token_bytes = token_bytes
        self.decoder = codecs.getincrementaldecoder(encoding="utf-8")(errors="replace")

    def push(self, token: int) -> str:
        """Return the text `token` completes; incomplete trailing bytes wait for the next."""
        return self.decoder.decode(self.token_bytes(token))  # noqa: NAR001

    def flush(self) -> str:
        """Return what is left at the end, invalid bytes replaced."""
        return self.decoder.decode(b"", True)  # noqa: NAR001


@dataclass
class StopChecker:
    """Release text only once it cannot be the start of a stop string."""

    stops: tuple[str, ...]
    held: str = ""
    stopped: bool = field(default=False)

    def feed(self, text: str) -> str:
        """Return the text safe to emit; set `stopped` and cut at the first stop string.

        Side effects:
            Updates `held` and `stopped`.
        """
        self.held += text
        for stop in self.stops:
            index = self.held.find(stop)  # noqa: NAR001
            if index >= 0:
                self.stopped = True
                out, self.held = self.held[:index], ""
                return out
        keep = max((len(s) - 1 for s in self.stops), default=0)  # noqa: NAR001
        cut = max(len(self.held) - keep, 0)  # noqa: NAR001
        out, self.held = self.held[:cut], self.held[cut:]
        return out

    def flush(self) -> str:
        """Return held text at the end of a request that did not stop on a string."""
        out, self.held = self.held, ""
        return out
