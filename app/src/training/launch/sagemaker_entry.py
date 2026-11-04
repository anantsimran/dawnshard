"""Container entrypoint on SageMaker: turn its resource config into a `torchrun` call.

SageMaker starts this on every host with `nproc_per_node module args...`. It reads
`/opt/ml/input/config/resourceconfig.json` for the host list, uses the first host as
the rendezvous endpoint, and replaces itself with `torchrun`, so the training module
runs exactly as it does on the other backends.
"""

import json
import os
import sys
from pathlib import Path

from training.launch.backend import RDZV_PORT

RESOURCE_CONFIG = Path("/opt/ml/input/config/resourceconfig.json")  # noqa: NAR001


def torchrun_argv(
    resource_config: dict, nproc_per_node: str, module: str, args: list[str]
) -> list[str]:
    """Return the `torchrun` argv for this host from SageMaker's resource config."""
    hosts = resource_config["hosts"]
    return [
        "torchrun",
        f"--nnodes={len(hosts)}",  # noqa: NAR001
        f"--nproc-per-node={nproc_per_node}",
        "--rdzv-backend=c10d",
        f"--rdzv-endpoint={hosts[0]}:{RDZV_PORT}",
        "--rdzv-id=sagemaker",
        "-m",
        module,
        *args,
    ]


def main() -> None:
    """Exec `torchrun` for this host; never returns."""
    config = json.loads(s=RESOURCE_CONFIG.read_text())
    argv = torchrun_argv(
        resource_config=config, nproc_per_node=sys.argv[1], module=sys.argv[2], args=sys.argv[3:]
    )
    os.execvp(argv[0], argv)  # noqa: NAR001


if __name__ == "__main__":
    main()
