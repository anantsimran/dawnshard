"""Ray Train entrypoint: run a training module on every `TorchTrainer` worker.

Runs inside the Ray cluster a RayJob created. Ray Train starts `--num-workers`
workers, sets up the torch process group across them, and calls `train_loop` on each;
`train_loop` runs the module as `__main__` with the given arguments, so the module is
the same one `torchrun` launches elsewhere. `ray` is imported lazily: it is installed
in the Ray image, not in this project's environment.

    python -m training.launch.ray_train --num-workers=2 --use-gpu=0 \\
        --storage-path=s3://bucket/ckpt --module training.model.qwen3 -- --steps 200
"""

import argparse
import importlib
import runpy
import sys


def train_loop(config: dict) -> None:
    """Run `config["module"]` as `__main__` with `config["args"]` as its argv."""
    sys.argv = [config["module"], *config["args"]]
    runpy.run_module(mod_name=config["module"], run_name="__main__")


def main() -> None:
    """Parse the entrypoint's arguments and run a `TorchTrainer`."""
    parser = argparse.ArgumentParser(description="Run a module under Ray Train.")
    parser.add_argument("--num-workers", type=int, required=True)  # noqa: NAR001
    parser.add_argument("--use-gpu", type=int, default=0)  # noqa: NAR001
    parser.add_argument("--storage-path", required=True)  # noqa: NAR001
    parser.add_argument("--max-failures", type=int, default=1)  # noqa: NAR001
    parser.add_argument("--module", required=True)  # noqa: NAR001
    parser.add_argument("args", nargs=argparse.REMAINDER)  # noqa: NAR001
    args = parser.parse_args()
    train = importlib.import_module(name="ray.train")
    torch_train = importlib.import_module(name="ray.train.torch")
    trainer = torch_train.TorchTrainer(
        train_loop_per_worker=train_loop,
        train_loop_config={"module": args.module, "args": [a for a in args.args if a != "--"]},
        scaling_config=train.ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=bool(args.use_gpu),  # noqa: NAR001
        ),  # noqa: NAR001
        run_config=train.RunConfig(
            storage_path=args.storage_path,
            failure_config=train.FailureConfig(max_failures=args.max_failures),
        ),
    )
    trainer.fit()


if __name__ == "__main__":
    main()
