"""ZeRO stage 2 by hand: each rank keeps the gradients, fp32 master weights, and Adam
state only for the parameters it owns.

Every parameter has exactly one owner rank, chosen to balance element counts. During
backward, each parameter's gradient is reduced to its owner as soon as it is ready,
and every other rank frees its copy, so a rank holds full gradients only for what it
owns. At `step`, each owner updates its parameters with AdamW on an fp32 master copy,
then broadcasts them, so every rank ends the step with the full, equal weights.

```
backward: grad(p) ready ──► reduce(grad, dst=owner(p)) ──► non-owners: p.grad = None
step:     owner: grad / world ──► AdamW on master fp32 ──► p.data = master
          every p: broadcast(p, src=owner(p))
```

Reduce plus broadcast moves the same bytes as one all-reduce, which is why ZeRO-2
communicates no more than plain data parallelism. Owning whole parameters, not
slices of one flat buffer, is the simplification: the shards are uneven by up to one
parameter, and the collectives are per parameter rather than bucketed. The reduce is
synchronous in the hook so a non-owner's gradient can be freed immediately; this
version demonstrates the memory, and `BucketedDDP` the overlap.
"""

import math

import torch
import torch.distributed as dist
from torch import nn


def assign_owners(numels: list[int], world_size: int) -> list[int]:
    """Return an owner rank per parameter: largest first, each to the least-loaded rank."""
    load = [0] * world_size
    owners = [0] * len(numels)  # noqa: NAR001
    for index in sorted(range(len(numels)), key=lambda i: -numels[i]):  # noqa: NAR001
        rank = load.index(min(load))  # noqa: NAR001
        owners[index] = rank
        load[rank] += numels[index]
    return owners


class Zero2AdamW(torch.optim.Optimizer):
    """AdamW whose gradients, master weights, and moments are sharded by owner rank.

    Build it on the unwrapped model after `broadcast_parameters`, and do not also wrap
    the model in a DDP: this optimizer does the gradient reduction itself, from hooks it
    registers here.
    """

    def __init__(
        self,
        params: list[nn.Parameter],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        """Assign owners, keep fp32 masters and moments for owned parameters, hook every grad."""
        super().__init__(
            params=params,
            defaults={"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay},
        )
        self.flat = [p for group in self.param_groups for p in group["params"]]
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.owners = assign_owners(numels=[p.numel() for p in self.flat], world_size=self.world)
        self.index = {id(p): i for i, p in enumerate(self.flat)}  # noqa: NAR001
        self.master = {
            i: p.detach().float().clone()
            for i, p in enumerate(self.flat)  # noqa: NAR001
            if self.owners[i] == self.rank
        }  # noqa: NAR001
        self.moments = {
            i: (torch.zeros_like(input=m), torch.zeros_like(input=m))
            for i, m in self.master.items()
        }
        self.steps = 0
        for param in self.flat:
            param.register_post_accumulate_grad_hook(hook=self._reduce_to_owner)

    def _reduce_to_owner(self, param: torch.Tensor) -> None:
        """Sum `param.grad` onto its owner; non-owners drop theirs."""
        owner = self.owners[self.index[id(param)]]  # noqa: NAR001
        assert param.grad is not None
        dist.reduce(tensor=param.grad, dst=owner)
        if owner != self.rank:
            param.grad = None

    @torch.no_grad()
    def step(self, closure=None) -> None:  # noqa: NAR002
        self.steps += 1
        group = self.param_groups[0]
        beta1, beta2 = group["betas"]
        for i, master in self.master.items():
            param = self.flat[i]
            if param.grad is None:
                continue
            grad = param.grad.float() / self.world
            m, v = self.moments[i]
            m.mul_(beta1).add_(other=grad, alpha=1 - beta1)  # noqa: NAR001
            v.mul_(beta2).addcmul_(tensor1=grad, tensor2=grad, value=1 - beta2)  # noqa: NAR001
            step_size = group["lr"] * math.sqrt(1 - beta2**self.steps) / (1 - beta1**self.steps)  # noqa: NAR001
            master.mul_(1 - group["lr"] * group["weight_decay"])  # noqa: NAR001
            master.addcdiv_(
                tensor1=m,
                tensor2=v.sqrt().add_(other=group["eps"] * math.sqrt(1 - beta2**self.steps)),  # noqa: NAR001
                value=-step_size,
            )  # noqa: NAR001
            param.copy_(master)  # noqa: NAR001
        for i, param in enumerate(self.flat):  # noqa: NAR001
            dist.broadcast(tensor=param.data, src=self.owners[i])
