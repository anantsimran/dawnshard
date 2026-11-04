# T9: serving platform, design notes

This ticket adds a router with pure, tested policies and admission control
([router.py](../../app/src/training/serve/router.py)), an autoscaler decision function
with rendered manifests ([autoscale.py](../../app/src/training/serve/autoscale.py),
[infra/k8s/serving/](../../infra/k8s/serving/)), a canary controller
([canary.py](../../app/src/training/serve/canary.py)), and a discrete-event study of
all of them on one seeded open-loop trace
([serving_platform.py](../../app/src/training/experiments/serving_platform.py)).
Numbers are in [RESULTS.md](RESULTS.md).

______________________________________________________________________

## Policies are pure functions over a snapshot

`pick(replicas, request, rng) -> index | None`. The proxy only gathers replica state
(polled from vLLM's `/metrics`) and forwards bytes; everything that decides is a pure
function. That is what makes the ticket's invariants testable without a cluster
(power-of-two never picks the more loaded of its pair; the bounded-load cap is never
exceeded) and lets the same functions run inside the simulator.

The router keeps nothing it cannot rebuild from the replicas: the outstanding counts
are re-read on `refresh`, and an unreachable replica reads as full, so a router
restart or a second router instance needs no shared store.

## Prefix routing needs a load bound, and the bound needs a ceiling

Consistent hashing on the prompt's first 256 characters sends requests with the same
system prompt to the same replica, where its KV blocks are cached. A popular prefix
would then pile onto one replica, so each replica is capped at
`ceil((1 + epsilon) x mean load)` and overflow walks the ring.

The first version used the cap without the ceiling. With four replicas at load 1 the
cap was 1.56, every placement needed 2, and the policy refused 94% of traffic in the
simulation. The ceiling guarantees the least-loaded replica is always under the cap,
which is the property the bounded-loads paper relies on; a test now places 5,000
requests and asserts the policy never refuses and never exceeds the cap.

## Admission control sheds before the SLO breaks

A request is admitted only if the chosen replica's predicted wait (requests ahead of
it beyond its free slots, times the mean service time over the slots) leaves the TTFT
SLO reachable; otherwise the router returns 429. Under a 3x burst this turns a queue
that never recovers into a 25% rejection rate and 3.5x the goodput.

## Autoscaling: a pure decision, then KEDA

`decide(policy, signals, now, last_change)` scales on waiting requests per replica and
KV utilization, with separate up and down thresholds (hysteresis) and a long down
cooldown. The cluster side is a KEDA `ScaledObject` on the same Prometheus query and
threshold, rendered from the same `ScalingPolicy` with the vLLM Deployment and its
readiness probe.

## Canary: stable hash split, stage by stage, rollback on SLO misses

Requests are split by a hash of their id (a retry sees the same version). Each stage
waits for `min_requests` canary requests, then compares SLO-miss rates against the
baseline with a tolerance. A version 1.5 s slower is rolled back at 5%.

## Routes not to take

- **Round robin, or any policy blind to load.** Fine at low load; under a burst one
  replica's slow requests back up its queue and the tail never recovers.
- **Least outstanding from a shared, stale view.** Every router sends the next burst to
  the same "least loaded" replica (herding). Power of two needs only two reads and
  almost never herds.
- **Prefix affinity without a load bound.** The hit rate goes up and one replica melts
  under the popular prompt.
- **A fractional bounded-load cap.** Refuses traffic the cluster has room for (above).
- **No admission control.** Accepting every request during overload makes every
  request miss the SLO; shedding some keeps the rest inside it.
- **Autoscaling on CPU utilization.** A GPU replica's CPU is idle while its queue grows.
- **A single threshold.** The scaler oscillates around it; use two, plus cooldowns.
- **Expecting the autoscaler to save a burst.** Scale-up takes minutes (node, image,
  weights, warmup); a 60-second burst is over before the capacity arrives. Pre-pulled
  images and warm nodes do more than a smarter scaler, and admission control covers
  the gap.
- **Closed-loop load tests.** They slow down with the system and hide the tail this
  ticket is about.

## Not built

The live run on four GPU replicas, Prometheus, and KEDA; multi-LoRA serving with real
adapters (the routing policy is built and tested); the llm-d comparison. The simulator
models replicas as slot-limited servers with an LRU prefix cache; it is calibrated to
nothing, so its absolute numbers mean less than the differences between policies.
