# T3: compute backends, design notes

This ticket adds a Qwen3-style decoder
([qwen3.py](../../app/src/training/model/qwen3.py)), a token-window dataset
([tokens.py](../../app/src/training/dataload/tokens.py)), and the `launch` package
([app/src/training/launch/](../../app/src/training/launch/)): one `JobSpec`, one
`Backend` interface, and four adapters (local, Kubernetes JobSet, KubeRay RayJob,
SageMaker). `make launch BACKEND=<name>` submits a spec and follows it. Measured
numbers are in [RESULTS.md](RESULTS.md).

______________________________________________________________________

## The Qwen3 decoder

Built from a config with the official field names, and module attributes named as in
the released checkpoints (`q_proj`, `input_layernorm`, `embed_tokens`), so T8 can load
weights by name. Parameter counts are computed in closed form and checked against a
meta-device instantiation: 596,049,920 for 0.6B and 1,720,574,976 for 1.7B.

- **SDPA, not the repo's own `attention`.** `F.scaled_dot_product_attention` with
  `is_causal=True, enable_gqa=True` picks flash or memory-efficient kernels and
  groups KV heads without a `repeat_interleave` copy. The from-scratch `attention` in
  `transformer/` stays the teaching implementation; T4 needs to force SDPA backends,
  which only works through SDPA.
- **Its own modules, not `transformer/`'s blocks.** The existing blocks are pre-norm
  LayerNorm/ReLU encoder-decoder blocks with per-head weight tensors; Qwen3 needs
  RMSNorm, SwiGLU, RoPE, GQA, and fused `Linear` projections that match checkpoint
  shapes. Sharing would have meant flags on every existing block.
- **Flattened `(B*L, vocab)` logits**, like the repo's other sequence models, so
  `fit` and `CrossEntropyLoss` need nothing new. `hidden_states` stops before the head
  for losses that should never build full logits (T4).

## `JobSpec`: image digest, module, width; nothing backend-specific

The spec says what to run (`image@sha256:...`, `module`, `args`), how wide (`nodes`,
`gpus_per_node` or `procs_per_node`), and where data and checkpoints live. Validation
refuses a tag: a tag can move between two submissions of the same spec.

- **Pros.** One spec runs on all four backends; every backend-specific field (instance
  type, IAM role, namespace) lives in the backend's constructor, not the spec.
- **Cons.** Some backend features have no spec field (node selectors, priority
  classes, SageMaker warm pools). Adding one means deciding what it means on the
  other three.

**Alternative considered.** A YAML template per backend with placeholders. Easier to
read, but four templates drift apart, and conditional blocks (GPU limits only when
`gpus_per_node > 0`) become template logic. Manifests are built as dicts in Python
and checked against golden files instead.

## The same `torchrun` everywhere, with c10d rendezvous

Every backend runs `torchrun --nnodes=N --rdzv-backend=c10d --rdzv-endpoint=<node 0> -m <module>`. c10d assigns node ranks, so no backend numbers its nodes; the endpoint is
the only thing each backend has to provide (JobSet's stable pod-0 DNS name, SageMaker's
first host). Ray is the exception: Ray Train creates the process group itself, and the
training module skips `init_process_group` when one exists.

## Kubernetes: JobSet, DNS wait, memory-backed `/dev/shm`

One replicated Job of `nodes` indexed pods with `enableDNSHostnames`, failure policy
"restart the whole group" up to `max_restarts`. Each pod waits for pod 0's DNS record
before exec-ing `torchrun`, because rendezvous that starts before the headless
Service has published it fails rather than retries. `/dev/shm` is a memory-backed
emptyDir (the default is 64 MB). GPU jobs get the `nvidia.com/gpu` limit and the GPU
taint toleration. With a `queue`, the JobSet carries the Kueue label and starts
suspended, so Kueue admits all pods or none.

**Alternatives considered.** A bare Indexed Job plus a hand-made headless Service:
works, but one failed pod does not restart the others, and a half-dead process
group hangs. Kubeflow Trainer's TrainJob: renders a JobSet too, with runtimes managed
cluster-side; worth it for a platform team, one more CRD to install for this ticket.
Rendering a TrainJob instead of a JobSet is a contained change in `k8s.py`.

All cluster access is `kubectl` through an injected runner, not the Python client:
the commands in the tests are the commands a person would type to debug.

## Ray: a RayJob per run, Ray Train's `TorchTrainer` inside

The RayJob creates a cluster from the spec's image and tears it down after
(`shutdownAfterJobFinishes`). The entrypoint, `training.launch.ray_train`, runs the
same training module on every worker. Ray Train needs shared storage to resume a
restarted worker, so rendering refuses a spec without `checkpoint_uri`.

- **Pros.** Worker restarts from a checkpoint are Ray's job, not ours.
- **Cons.** Two schedulers (Kubernetes places pods, Ray places workers) and a version
  that must match between head and workers. `ray` is imported lazily, because it is
  in the image, not in this project's environment.

## SageMaker: boto3 `CreateTrainingJob`, not the SDK's `ModelTrainer`

The ticket names the SDK v3 `ModelTrainer`. The adapter builds the low-level
`CreateTrainingJob` request instead: the spec's digest-pinned image as a
bring-your-own container, an entrypoint (`training.launch.sagemaker_entry`) that
reads SageMaker's resource config and execs the same `torchrun`, S3 input channel,
checkpoint sync to `/opt/ml/checkpoints`, managed spot with a wait time twice the run.

- **Pros.** The request is plain data, so the tests check exactly what is sent, and
  the unit of portability stays "image + command" as on the other backends.
  `ModelTrainer` packages source code into the job instead, which would make
  SageMaker the one backend not running the pinned image as built.
- **Cons.** No SDK conveniences: local mode, automatic source upload, and the SDK's
  distributed-launcher presets.

## Launch telemetry

Each backend sets `DAWNSHARD_SUBMIT_TIME`, `DAWNSHARD_BACKEND`, `DAWNSHARD_IMAGE` and
`DAWNSHARD_RUN_ID`. The training process records when it started, when the process
group was ready, and (through a train probe) when the first step finished, and rank 0
writes them to `app/history/<run_id>.launch.json`. Scheduling and image-pull times
come from the orchestrator: `K8sBackend.cold_start` reads them from pod events. The
telemetry is a separate file rather than a field in the history JSON, so `fit`'s API
and the history format stay unchanged.

## Not built

Slurm and SkyPilot (v8, optional). The Docker image is the existing
[Dockerfile](../../Dockerfile): the PyPI torch wheels carry their CUDA libraries, so a
CUDA base image is not needed for the GPU stages as long as the node has a driver.

## Routes not to take

- **Image tags in job specs.** A tag can move between two submissions of the same spec;
  pin the digest.
- **Hand-written YAML per backend.** Four copies drift; render from one spec and check
  the output against golden files.
- **`torchrun` in a Kubernetes Deployment.** A Deployment restarts pods one at a time
  and never completes; a training job needs all-or-nothing gang semantics (JobSet,
  Kueue) and a defined end.
- **Starting rendezvous before the headless Service has DNS.** It fails instead of
  retrying; wait for pod 0's record.
- **The default 64 MB `/dev/shm`.** DataLoader workers and NCCL crash on it; mount a
  memory-backed emptyDir.
- **Numbering node ranks yourself.** c10d rendezvous assigns them; one less thing per
  backend to get wrong.
- **Spot instances without a checkpoint location.** A preempted job restarts from
  scratch, and rework can exceed the spot discount.
- **Credentials in the image or the repo.** Environment, Kubernetes Secrets, or the
  cloud credential chain only.
