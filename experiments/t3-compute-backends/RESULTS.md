# T3: results

Measured on the development container: 4 vCPU Linux, no GPU, no Docker daemon, no
cloud credentials, and Hugging Face blocked, so no TinyStories. Only the local backend
could run here. The Kubernetes, Ray and SageMaker adapters are implemented and tested
against fake `kubectl` and boto3 clients; their rows need the ticket's GPU quota and
accounts.

## Qwen3 decoder

| Config | Closed form | Meta-device count | Published |
|---|---|---|---|
| `QWEN3_0_6B` | 596,049,920 | 596,049,920 | 0.596B |
| `QWEN3_1_7B` | 1,720,574,976 | 1,720,574,976 | 1.72B |

The tiny config overfits one batch (loss halves in 30 steps,
`test_small_model_overfits_one_batch`). Loss falling on TinyStories is not checked:
the dataset could not be downloaded here.

## v0: local backend, 2 CPU processes, gloo

`make launch BACKEND=local` with [qwen3_tiny_local.json](../../infra/jobs/qwen3_tiny_local.json):
tiny config, random tokens, 20 steps, batch 4 x 64 tokens per rank. Both ranks ran
while a CPU training job held all four cores, so read the times as an upper bound.

| Phase | Seconds after submit |
|---|---|
| process started (after `uv run`, `torchrun`, and `import torch`) | 6.86 |
| process group ready | 7.07 |
| first step finished | 9.13 |
| job succeeded (wall clock of `make launch`) | 11.9 |

Prediction: under 10 s to first step. Measured 9.1 s. Almost all of it is interpreter
and import start-up; process-group init over gloo is 0.2 s.

## Not run here

| Version | Needs |
|---|---|
| v1: image size and cold pull time | a container registry and Docker daemon |
| v2-v3: kind, JobSet rendezvous, pod kill and group restart | Docker daemon for kind |
| v4: GKE L4 node pool, cold vs warm node breakdown | GCP project with GPU quota |
| v5: TrainJob + Kueue gang admission demo ([kueue.yaml](../../infra/k8s/kueue.yaml)) | a cluster with Kueue |
| v6: KubeRay RayJob, worker kill and resume | a cluster with KubeRay |
| v7: SageMaker managed spot, stop and restart on one checkpoint location | AWS account and role |
| Spend per session | the above; `scripts/teardown.sh` deletes what they create |

Commands, once credentials exist:

```bash
make launch BACKEND=k8s SPEC=infra/jobs/qwen3_small_ddp.json DRY_RUN=1
```

```bash
make launch BACKEND=k8s SPEC=infra/jobs/qwen3_small_ddp.json
```

```bash
GKE_CLUSTER=dawnshard GKE_ZONE=us-central1-a GCP_PROJECT=<project> S3_BUCKET=<bucket> scripts/teardown.sh
```
