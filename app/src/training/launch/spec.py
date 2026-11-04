"""One immutable description of a training job, and its validation.

A `JobSpec` says what to run (an image pinned by digest, a Python module and its
arguments), how wide (nodes, processes or GPUs per node), and where data and
checkpoints live. Every backend renders the same spec into its own submission; none
of them adds a field. Because the image is pinned by digest, the same spec runs the
same bits on every backend; a tag could move between submissions.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# Kubernetes object names: DNS-1123 labels. Kept short so backends can add suffixes.
NAME_PATTERN = re.compile(pattern=r"^[a-z0-9]([-a-z0-9]{0,38}[a-z0-9])?$")
DIGEST_PATTERN = re.compile(pattern=r"^[^@\s]+@sha256:[0-9a-f]{64}$")
URI_SCHEMES = ("s3://", "gs://", "file://")


@dataclass(frozen=True)
class JobSpec:
    """What to run, how wide, and where its data and checkpoints live.

    Attributes:
        module: Python module run as `torchrun ... -m <module> <args>`.
        procs_per_node: Processes per node when `gpus_per_node` is 0; with GPUs, one
            process per GPU.
        queue: Kueue LocalQueue name; empty submits outside Kueue.
    """

    name: str
    image: str
    module: str
    args: tuple[str, ...] = ()
    nodes: int = 1
    gpus_per_node: int = 0
    procs_per_node: int = 1
    cpus_per_node: int = 2
    memory_gb: int = 8
    env: dict[str, str] = field(default_factory=dict)
    data_uri: str = ""
    checkpoint_uri: str = ""
    max_restarts: int = 1
    queue: str = ""

    @property
    def nproc_per_node(self) -> int:
        """Processes `torchrun` starts on each node."""
        return self.gpus_per_node or self.procs_per_node

    @property
    def world_size(self) -> int:
        """Total processes across the job."""
        return self.nodes * self.nproc_per_node

    def validate(self, require_digest: bool = True) -> None:
        """Raise ValueError on the first field a backend could not use.

        Args:
            require_digest: Demand `image@sha256:<64 hex>`. The local backend runs on
                the host and ignores the image, so it passes False.

        Raises:
            ValueError: Naming the bad field.
        """
        problems = [
            (
                not NAME_PATTERN.match(string=self.name),
                f"name {self.name!r} must be a DNS label of at most 40 chars",
            ),
            (
                require_digest and not DIGEST_PATTERN.match(string=self.image),
                f"image {self.image!r} must be pinned by sha256 digest",
            ),
            (not self.module, "module must be set"),
            (
                self.nodes < 1 or self.procs_per_node < 1,
                "nodes and procs_per_node must be at least 1",
            ),
            (
                self.gpus_per_node < 0 or self.max_restarts < 0,
                "gpus_per_node and max_restarts must be non-negative",
            ),
        ]
        problems += [
            (
                bool(uri) and not uri.startswith(URI_SCHEMES),  # noqa: NAR001
                f"uri {uri!r} must start with one of {URI_SCHEMES}",
            )
            for uri in (self.data_uri, self.checkpoint_uri)
        ]
        for failed, message in problems:
            if failed:
                raise ValueError(message)  # noqa: NAR001

    @classmethod
    def from_json(cls, path: Path) -> "JobSpec":
        """Load a spec from a JSON file of its fields; `args` may be a list."""
        fields = json.loads(s=path.read_text(encoding="utf-8"))
        fields["args"] = tuple(fields.get("args", ()))  # noqa: NAR001
        return cls(**fields)
