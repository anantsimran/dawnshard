import random

from training.serve.canary import CanaryController, Decision


def _run(extra_ttft, seed=0):
    rng, controller = random.Random(seed), CanaryController(min_requests=30)
    for i in range(20000):
        version = controller.route(request_id=f"r{i}")
        ttft = rng.expovariate(1 / 0.3) + (extra_ttft if version == "canary" else 0.0)
        decision = controller.observe(version=version, met_slo=ttft < 1.0)
        if decision in (Decision.ROLLBACK, Decision.DONE):
            return controller, i + 1
    return controller, None


def test_a_degraded_version_is_rolled_back_at_the_first_stage():
    controller, requests = _run(extra_ttft=1.5)
    assert controller.rolled_back and controller.canary_share == 0.0
    assert controller.log[0].startswith("rollback at 5%")
    assert requests < 2000  # about min_requests / 5% canary requests


def test_a_healthy_version_advances_to_full_traffic():
    controller, requests = _run(extra_ttft=0.0)
    assert not controller.rolled_back and controller.canary_share == 1.0
    assert controller.log == ["advance to 25%", "advance to 100%"]


def test_routing_is_stable_per_request_id_and_matches_the_share():
    controller = CanaryController()
    assert all(controller.route(request_id=f"x{i}") == controller.route(request_id=f"x{i}") for i in range(100))
    share = sum(controller.route(request_id=f"y{i}") == "canary" for i in range(20000)) / 20000
    assert abs(share - 0.05) < 0.01
