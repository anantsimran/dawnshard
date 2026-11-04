import yaml
from training.serve.autoscale import ScalingPolicy, Signals, decide, render_manifests, scale_up_legs, to_yaml

POLICY = ScalingPolicy(min_replicas=2, max_replicas=10, queue_up=4, queue_down=0.5, cooldown_up_s=60, cooldown_down_s=300)


def test_scales_up_on_queue_or_kv_only_after_the_up_cooldown():
    busy = Signals(replicas=2, waiting=20, kv_utilization=0.5)
    assert decide(policy=POLICY, signals=busy, now=100, last_change=0) == 5  # enough to clear the queue
    assert decide(policy=POLICY, signals=busy, now=30, last_change=0) == 2  # still cooling down
    full_kv = Signals(replicas=2, waiting=0, kv_utilization=0.9)
    assert decide(policy=POLICY, signals=full_kv, now=100, last_change=0) == 3


def test_hysteresis_and_the_longer_down_cooldown():
    middling = Signals(replicas=4, waiting=4, kv_utilization=0.6)  # between thresholds: hold
    assert decide(policy=POLICY, signals=middling, now=1000, last_change=0) == 4
    idle = Signals(replicas=4, waiting=0, kv_utilization=0.1)
    assert decide(policy=POLICY, signals=idle, now=200, last_change=0) == 4
    assert decide(policy=POLICY, signals=idle, now=400, last_change=0) == 3
    assert decide(policy=POLICY, signals=Signals(replicas=2, waiting=0, kv_utilization=0.0), now=1e6, last_change=0) == 2
    assert decide(policy=POLICY, signals=Signals(replicas=10, waiting=500, kv_utilization=1.0), now=1e6, last_change=0) == 10


def test_scale_up_legs_split_the_timeline_and_skip_missing_stamps():
    legs = scale_up_legs(stamps={"decided": 0, "node_ready": 150, "image_pulled": 200, "weights_loaded": 207, "ready": 240})
    assert legs == {"node": 150, "image": 50, "weights": 7, "warmup": 33}
    warm = scale_up_legs(stamps={"decided": 0, "weights_loaded": 9, "ready": 40})
    assert warm == {"node": 0.0, "image": 0.0, "weights": 9, "warmup": 31}


def test_manifests_render_a_ready_gated_deployment_and_its_scaler():
    deployment, scaled = yaml.safe_load_all(to_yaml(manifests=render_manifests(name="qwen", image="vllm/vllm-openai@sha256:" + "a" * 64, model="Qwen/Qwen3-1.7B", policy=POLICY)))
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["readinessProbe"]["httpGet"]["path"] == "/health"
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert scaled["spec"]["maxReplicaCount"] == 10 and scaled["spec"]["triggers"][0]["metadata"]["threshold"] == "4"
