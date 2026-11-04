"""One `ParallelismSpec` → one `DeviceMesh`, with the dimensions in bandwidth order.

```
world = pp × dp_replicate × dp_shard × tp

mesh dims, outermost → innermost:

("pp", "dp_replicate", "dp_shard", "tp")
   │         │              │        └─ consecutive ranks, one node, NVLink:
   │         │              │           communicates inside every layer
   │         │              └─ FSDP shards: per-layer all-gathers
   │         └─ HSDP replicas: one all-reduce of shards per step
   └─ pipeline stages: point-to-point, across nodes
```

Ranks are numbered row-major over the mesh, so the innermost dimension holds
consecutive ranks, which on a GPU cluster share a node. Tensor parallelism goes there
because it communicates inside every layer, forward and backward; pipeline
parallelism goes outermost because it only sends activations between neighbouring
stages. A TP group that crossed nodes would run about 10x slower and raise no error,
which is why the order is fixed here rather than left to each caller.
"""

from dataclasses import dataclass

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

MESH_DIMS = ("pp", "dp_replicate", "dp_shard", "tp")


@dataclass(frozen=True)
class ParallelismSpec:
    """Degrees of each parallelism; their product is the world size."""

    dp_replicate: int = 1
    dp_shard: int = 1
    tp: int = 1
    pp: int = 1

    @property
    def world_size(self) -> int:
        """Processes the spec needs."""
        return self.pp * self.dp_replicate * self.dp_shard * self.tp

    def mesh_shape(self) -> tuple[int, int, int, int]:
        """Sizes in `MESH_DIMS` order."""
        return (self.pp, self.dp_replicate, self.dp_shard, self.tp)

    def validate(
        self, world_size: int, num_heads: int, num_kv_heads: int, ffn: int, vocab: int
    ) -> None:
        """Raise if the spec does not cover the world or TP does not divide the model.

        Raises:
            ValueError: Naming the first mismatch.
        """
        if self.world_size != world_size:
            raise ValueError(f"spec covers {self.world_size} ranks, world has {world_size}")  # noqa: NAR001
        for name, size in (
            ("heads", num_heads),
            ("kv_heads", num_kv_heads),
            ("ffn", ffn),
            ("vocab", vocab),
        ):
            if size % self.tp:
                raise ValueError(f"tp={self.tp} does not divide {name}={size}")  # noqa: NAR001


def build_mesh(spec: ParallelismSpec, device_type: str) -> DeviceMesh:
    """Return the 4-D mesh over every rank, dims named `MESH_DIMS`, TP innermost."""
    return init_device_mesh(
        device_type=device_type, mesh_shape=spec.mesh_shape(), mesh_dim_names=MESH_DIMS
    )
