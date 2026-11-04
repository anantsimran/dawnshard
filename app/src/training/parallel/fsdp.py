"""Apply FSDP2 (`fully_shard`) to any model, block by block, then the root.

Each block of `block_type` becomes one FSDP unit: its parameters are sharded on dim 0
across the shard mesh dimension, all-gathered just before the block runs, and
resharded after (unless `reshard_after_forward=False`, which keeps them gathered
until backward to save one all-gather). The root unit takes whatever is left, here
the embedding, final norm, and head. Gradients are reduce-scattered back to shards.

```
model ──► for module in model.modules(): isinstance(module, block_type) ──► fully_shard(module)
      └─► fully_shard(model)                                            (the root, last)
then:     optimizer = AdamW(model.parameters())                         built after sharding
```

The optimizer must be built after `apply_fsdp`: `fully_shard` replaces each
parameter with a sharded DTensor, and an optimizer built earlier would hold the old,
unsharded tensors.

With `replicate > 1` the mesh is 2-D, (replicate, shard): HSDP. Parameters are
sharded within a group and replicated across groups, so the cross-group traffic is
an all-reduce of shards only, which is what keeps it off the slow inter-node links.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard


@dataclass(frozen=True)
class ParallelismSpec:
    """How to shard: the mesh, mixed precision, and when to reshard.

    Attributes:
        replicate: Size of the replicate mesh dimension; 1 is plain FSDP, more is HSDP.
        param_dtype: Dtype parameters are all-gathered and computed in; None keeps
            their own.
        reduce_dtype: Dtype gradients are reduce-scattered in.
    """

    replicate: int = 1
    reshard_after_forward: bool = True
    param_dtype: Optional[torch.dtype] = torch.bfloat16
    reduce_dtype: Optional[torch.dtype] = torch.float32


def apply_fsdp(
    model: nn.Module, block_type: type[nn.Module], spec: ParallelismSpec, device_type: str
) -> nn.Module:
    """Shard every `block_type` module of `model`, then `model` itself; return `model`.

    Raises:
        ValueError: If the world size is not divisible by `spec.replicate`.

    Side effects:
        Replaces `model`'s parameters with sharded DTensors in place.
    """
    world = dist.get_world_size()
    if world % spec.replicate:
        raise ValueError(f"world size {world} is not divisible by replicate={spec.replicate}")  # noqa: NAR001
    if spec.replicate > 1:
        mesh = init_device_mesh(
            device_type=device_type,
            mesh_shape=(spec.replicate, world // spec.replicate),
            mesh_dim_names=("replicate", "shard"),
        )
    else:
        mesh = init_device_mesh(device_type=device_type, mesh_shape=(world,))
    policy = MixedPrecisionPolicy(param_dtype=spec.param_dtype, reduce_dtype=spec.reduce_dtype)
    for module in model.modules():
        if isinstance(module, block_type):  # noqa: NAR001
            fully_shard(
                module=module,
                mesh=mesh,
                mp_policy=policy,
                reshard_after_forward=spec.reshard_after_forward,
            )
    fully_shard(
        module=model, mesh=mesh, mp_policy=policy, reshard_after_forward=spec.reshard_after_forward
    )
    return model
