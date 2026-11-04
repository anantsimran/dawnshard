import json
import random
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from training.serve.router import (
    Replica,
    RouteRequest,
    Router,
    adapter_aware,
    admit,
    bounded_load_cap,
    power_of_two,
    predicted_wait,
    prefix_bounded_load,
    prefix_key,
    read_metrics,
    round_robin,
)


def _replicas(loads):
    return [Replica(name=f"r{i}", outstanding=load) for i, load in enumerate(loads)]


def test_power_of_two_never_picks_the_more_loaded_of_its_two_samples():
    rng = random.Random(0)
    for _ in range(2000):
        replicas = _replicas([rng.randrange(20) for _ in range(6)])
        seed = rng.randrange(10**9)
        a, b = random.Random(seed).sample(range(6), k=2)
        picked = power_of_two(replicas, RouteRequest(prompt=""), random.Random(seed))
        assert picked in (a, b)
        assert replicas[picked].outstanding == min(replicas[a].outstanding, replicas[b].outstanding)


def test_bounded_load_cap_is_never_exceeded_and_never_refuses():
    rng = random.Random(1)
    policy = prefix_bounded_load(epsilon=0.25)
    replicas = _replicas([0] * 4)
    prompts = ["system-0"] * 7 + ["system-1", "user"]  # one hot prefix
    for _ in range(5000):
        if rng.random() < 0.45 and any(r.outstanding for r in replicas):
            rng.choice([r for r in replicas if r.outstanding]).outstanding -= 1
            continue
        cap = bounded_load_cap(replicas=replicas, epsilon=0.25)
        index = policy(replicas, RouteRequest(prompt=rng.choice(prompts)), rng)
        assert index is not None
        replicas[index].outstanding += 1
        assert replicas[index].outstanding <= cap


def test_prefix_hash_is_stable_and_diverges_on_the_prefix_only():
    assert prefix_key(prompt="system prompt A" + "x" * 300) == prefix_key(prompt="system prompt A" + "x" * 300 + " tail")
    assert prefix_key(prompt="system prompt A") != prefix_key(prompt="system prompt B")
    policy, replicas = prefix_bounded_load(), _replicas([0, 0, 0, 0])
    picks = {policy(replicas, RouteRequest(prompt="same system prompt"), random.Random(s)) for s in range(10)}
    assert len(picks) == 1


def test_admission_rejects_exactly_when_the_predicted_wait_exceeds_the_slo():
    for outstanding in range(0, 40):
        replica = Replica(name="r", outstanding=outstanding)
        wait = predicted_wait(replica=replica, service_time_s=2.0, slots=8)
        assert admit(replica=replica, service_time_s=2.0, slots=8, ttft_slo_s=1.0) == (wait <= 1.0)
    assert admit(replica=Replica(name="r", outstanding=7), service_time_s=2.0, slots=8, ttft_slo_s=0.0)
    assert not admit(replica=Replica(name="r", outstanding=12), service_time_s=2.0, slots=8, ttft_slo_s=1.0)


def test_adapter_routing_prefers_residency_under_the_bound():
    replicas = _replicas([2, 1, 2])  # cap = ceil(1.25 x 6 / 3) = 3
    replicas[0].adapters = {"sql"}
    assert adapter_aware()(replicas, RouteRequest(prompt="", adapter="sql"), random.Random(0)) == 0
    replicas[0].outstanding = 9  # over the cap: go where it is cheapest instead
    assert adapter_aware()(replicas, RouteRequest(prompt="", adapter="sql"), random.Random(0)) == 1


def test_round_robin_cycles():
    policy = round_robin()
    assert [policy(_replicas([0, 0, 0]), RouteRequest(prompt=""), random.Random(0)) for _ in range(4)] == [0, 1, 2, 0]


def test_reads_vllm_metrics():
    text = "# HELP x\nvllm:num_requests_running{model=\"m\"} 3.0\nvllm:num_requests_waiting{model=\"m\"} 2.0\nvllm:kv_cache_usage_perc{model=\"m\"} 0.42\n"
    assert read_metrics(text=text) == (5, 0.42)


class _FakeReplica(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n"
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b"data: {\"port\": %d}\n\ndata: [DONE]\n\n" % self.server.server_port)

    def log_message(self, *args):
        pass


def test_proxy_streams_from_a_replica_and_sheds_with_429():
    replicas = [ThreadingHTTPServer(("127.0.0.1", 0), _FakeReplica) for _ in range(2)]
    for server in replicas:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    router = Router(urls=[f"http://127.0.0.1:{s.server_port}" for s in replicas], policy=round_robin(), service_time_s=1.0, slots=1, ttft_slo_s=0.5)
    router.refresh()
    proxy = router.serve(host="127.0.0.1", port=0)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{proxy.server_port}/v1/completions"
    body = json.dumps({"prompt": "hi"}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url=url, data=body), timeout=10) as response:
            assert b"[DONE]" in response.read()
        router.replicas[1].outstanding = 5  # round robin's next pick now looks overloaded
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(url=url, data=body), timeout=10)
        assert error.value.code == 429
    finally:
        proxy.shutdown()
        for server in replicas:
            server.shutdown()
