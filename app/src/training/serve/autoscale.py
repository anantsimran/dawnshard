"""Scale replicas on queue depth and KV utilization, with hysteresis and cooldown.

`decide` is a pure function of the current signals and the scaler's last action, so
it is tested on synthetic timelines and the cluster side only wires it up: the same
thresholds are rendered into a KEDA `ScaledObject` on Prometheus queries.

LLM serving scales on how much work is waiting and how full the KV cache is, not on
CPU: a GPU replica's CPU is nearly idle while its queue grows. Two thresholds per
signal (up above the high one, down below the low one) stop the scaler oscillating
around one value, and a cooldown after each action lets the new replica's effect show
up in the metrics before the next decision. Scale-up is slow (node, image, weights,
warmup, each of which `scale_up_legs` separates), so the up threshold must fire
before the SLO breaks, not when it does.
"""

import math
from dataclasses import dataclass
from typing import Any

import yaml


@dataclass(frozen=True)
class ScalingPolicy:
    """Thresholds, bounds, and timing for `decide`."""

    min_replicas: int = 1
    max_replicas: int = 8
    queue_up: float = 4.0  # waiting requests per replica that trigger a scale-up
    queue_down: float = 0.5
    kv_up: float = 0.85
    kv_down: float = 0.4
    cooldown_up_s: float = 60.0
    cooldown_down_s: float = 300.0


@dataclass(frozen=True)
class Signals:
    """What the scaler reads, summed or averaged over replicas."""

    replicas: int
    waiting: float
    kv_utilization: float


def decide(policy: ScalingPolicy, signals: Signals, now: float, last_change: float) -> int:
    """Return the desired replica count.

    Up when waiting per replica or KV utilization crosses its high threshold and the up
    cooldown has passed (by enough replicas to bring the queue under the threshold);
    down by one when both are under their low thresholds and the longer down cooldown
    has passed; otherwise unchanged.
    """
    per_replica = signals.waiting / max(signals.replicas, 1)  # noqa: NAR001
    since = now - last_change
    if (
        per_replica > policy.queue_up or signals.kv_utilization > policy.kv_up
    ) and since >= policy.cooldown_up_s:
        needed = (
            math.ceil(signals.waiting / policy.queue_up)  # noqa: NAR001
            if per_replica > policy.queue_up
            else signals.replicas + 1
        )  # noqa: NAR001
        return min(policy.max_replicas, max(signals.replicas + 1, needed))  # noqa: NAR001
    if (
        per_replica < policy.queue_down
        and signals.kv_utilization < policy.kv_down
        and since >= policy.cooldown_down_s
    ):
        return max(policy.min_replicas, signals.replicas - 1)  # noqa: NAR001
    return signals.replicas


def scale_up_legs(stamps: dict[str, float]) -> dict[str, float]:
    """Seconds spent in each leg of a scale-up, from event timestamps.

    Args:
        stamps: `decided`, `node_ready`, `image_pulled`, `weights_loaded`, `ready`,
            in seconds; a leg whose stamp is missing (a warm node, a cached image) is 0.
    """
    order = ["decided", "node_ready", "image_pulled", "weights_loaded", "ready"]
    known = {k: stamps[k] for k in order if k in stamps}
    legs, previous = {}, known["decided"]
    for start, end, leg in zip(order, order[1:], ["node", "image", "weights", "warmup"]):  # noqa: NAR001
        if end in known:
            legs[leg], previous = known[end] - previous, known[end]
        else:
            legs[leg] = 0.0
    return legs


def render_manifests(
    name: str, image: str, model: str, policy: ScalingPolicy, gpus: int = 1
) -> list[dict[str, Any]]:
    """Render a vLLM Deployment and its KEDA ScaledObject from one policy.

    The readiness probe gates traffic until weights are loaded and `/health` answers;
    the ScaledObject scales on the same queue threshold `decide` uses.
    """
    labels = {"app": name}
    container = {
        "name": "vllm",
        "image": image,
        "args": ["vllm", "serve", model, "--enable-prefix-caching"],
        "ports": [{"containerPort": 8000}],
        "resources": {"limits": {"nvidia.com/gpu": gpus}},
        "readinessProbe": {"httpGet": {"path": "/health", "port": 8000}, "periodSeconds": 5},
    }
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "labels": labels},
        "spec": {
            "replicas": policy.min_replicas,
            "selector": {"matchLabels": labels},
            "template": {
                "metadata": {"labels": labels},
                "spec": {"containers": [container], "terminationGracePeriodSeconds": 120},
            },
        },
    }
    trigger = {
        "type": "prometheus",
        "metadata": {
            "query": f'sum(vllm:num_requests_waiting{{app="{name}"}})',
            "threshold": str(policy.queue_up),  # noqa: NAR001
        },
    }  # noqa: NAR001
    scaled = {
        "apiVersion": "keda.sh/v1alpha1",
        "kind": "ScaledObject",
        "metadata": {"name": name},
        "spec": {
            "scaleTargetRef": {"name": name},
            "minReplicaCount": policy.min_replicas,
            "maxReplicaCount": policy.max_replicas,
            "cooldownPeriod": int(policy.cooldown_down_s),  # noqa: NAR001
            "triggers": [trigger],
        },  # noqa: NAR001
    }
    return [deployment, scaled]


def to_yaml(manifests: list[dict[str, Any]]) -> str:
    """Dump manifests as one multi-document YAML string."""
    return yaml.safe_dump_all(documents=manifests, sort_keys=False)
