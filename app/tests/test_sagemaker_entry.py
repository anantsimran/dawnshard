from training.launch.sagemaker_entry import torchrun_argv


def test_first_host_is_the_rendezvous_endpoint():
    argv = torchrun_argv(
        resource_config={"current_host": "algo-2", "hosts": ["algo-1", "algo-2"]},
        nproc_per_node="4",
        module="training.model.qwen3",
        args=["--steps", "5"],
    )
    assert "--nnodes=2" in argv and "--nproc-per-node=4" in argv
    assert "--rdzv-endpoint=algo-1:29500" in argv
    assert argv[-4:] == ["-m", "training.model.qwen3", "--steps", "5"]
