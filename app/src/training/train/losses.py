"""losses.py -- loss functions that are not in torch.nn."""

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


def _chunk_loss_sum(
    hidden: torch.Tensor, weight: torch.Tensor, target: torch.Tensor, ignore_index: int
) -> torch.Tensor:
    """Summed cross-entropy of one chunk; its logits live only inside this call."""
    logits = (hidden @ weight.T).float()
    return F.cross_entropy(input=logits, target=target, ignore_index=ignore_index, reduction="sum")


class ChunkedCrossEntropy(nn.Module):
    """LM cross-entropy from hidden states, one chunk of positions at a time.

    Takes the final hidden states and the LM head weight instead of logits, so the
    full `(N, vocab)` logits tensor never exists. Each chunk computes its logits,
    upcasts them to fp32, and reduces them to a summed loss inside
    `torch.utils.checkpoint`, which saves only the chunk's inputs; backward
    recomputes the chunk's logits, uses them, and frees them. Peak memory for the
    loss is one chunk of logits instead of all of them, twice (forward and grad).

    ```
    hidden (N, d_model)                    weight (vocab, d_model)   target (N,)
        │                                       │                        │
        ▼ split into chunks of chunk_size rows  │                        │
    ┌───────────────── for each chunk ─────────────────────────────────────────┐
    │   hidden_chunk (C, d_model) ──► @ weight.T ──► (C, vocab) ──► .float()   │
    │                                                       │                  │
    │                         F.cross_entropy(reduction="sum") ◄── target_chunk│
    │                                                       │  inside checkpoint
    └───────────────────────────────────────────────────────┼──────────────────┘
                                                            ▼
                                        sum over chunks / count of real targets
                                                            ▼  loss ()
    ```

    The cost is one extra head matmul per chunk in backward. With a 150K vocabulary
    the logits are the largest activation of a small model, so trading that matmul
    for their memory usually buys a larger micro-batch.

    The mean is over targets that are not `ignore_index`, matching
    `F.cross_entropy`'s default reduction. `weight` is read from the head on every
    call, not copied, so a tied embedding keeps receiving its gradient.

    `forward` holds no state between calls.
    """

    def __init__(self, head: nn.Module, chunk_size: int = 4096, ignore_index: int = -100) -> None:
        """Keep a reference to the module whose `lm_head_weight` is the output projection.

        Args:
            head: Any module with an `lm_head_weight` property, `(vocab, d_model)`.
            chunk_size: Rows per chunk; peak loss memory is `chunk_size * vocab * 4` bytes.
        """
        super().__init__()
        self.head = [head]  # a list, so the head is not registered as a submodule twice
        self.chunk_size = chunk_size
        self.ignore_index = ignore_index

    def forward(self, hidden: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the mean cross-entropy of `hidden @ weight.T` against `target`.

        Args:
            hidden: `(N, d_model)` final hidden states, one row per target.
            target: `(N,)` class ids; `ignore_index` rows are left out of the mean.
        """
        weight = self.head[0].lm_head_weight
        total = hidden.new_zeros(size=(), dtype=torch.float32)
        for start in range(0, hidden.shape[0], self.chunk_size):  # noqa: NAR001
            end = start + self.chunk_size
            chunk_sum = checkpoint(  # noqa: NAR001
                _chunk_loss_sum,
                hidden[start:end],
                weight,
                target[start:end],
                self.ignore_index,
                use_reentrant=False,
            )
            total = total + cast(torch.Tensor, chunk_sum)  # noqa: NAR001
        return total / (target != self.ignore_index).sum().clamp(min=1)
