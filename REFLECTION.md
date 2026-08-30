# Reflection

**Author:** Sumedh (DA25G523)

> Draft below — written from the actual engineering issues hit while building
> this pipeline. Edit freely to match your own voice and add anything I
> couldn't see from the code/manifests alone.

## 1. What was the most challenging part of this assignment?

The most persistent issue was simply speed: CIFAR-10 downloads fresh into
`data_dir` on every from-scratch run, and training itself was slow because
everything here targets CPU-only PyTorch wheels (`--index-url
https://download.pytorch.org/whl/cpu`) to keep the Docker images small — there's
no CUDA in the loop. The `resnet18` adaptation for 32×32 input makes this
worse in a subtle way: swapping the 7×7/stride-2 stem for a 3×3/stride-1 one
and replacing `maxpool` with `Identity` is exactly what an ImageNet-sized
stem needs for tiny images, but it also means the early conv blocks keep
operating on full-resolution feature maps for longer, so each epoch costs
noticeably more compute than the stock architecture would suggest. Tuning
`num_workers` in the `DataLoader` and batching sensibly helped, but the
honest fix would be either a GPU node or a lighter `simplecnn` for fast
iteration, falling back to `resnet18` for the final run.

The second big source of friction was configuration drift. The training
config's schema changed shape more than once over the course of the build —
from a richer nested format (`conv_channels`, `kernel_size`, a `normalize`
block) to the current flat one (`model.architecture`, top-level
`early_stopping_patience`, an `output` section for checkpoint paths). Every
change like that has a blast radius: `model.py`, `dataset.py`, `train.py`,
`serve.py`, the ConfigMap, and the Job/Deployment manifests all encode
assumptions about that shape, and it was easy for one of them to go stale
silently — e.g. a Kubernetes Job still passing `--config`/`--checkpoint-dir`
CLI flags after `train.py` dropped `argparse` entirely in favor of reading
paths straight out of the config file. Nothing errors loudly when that
happens; the flag is just ignored.

Related to that: the config hardcodes container-native absolute paths
(`/app/data`, `/app/checkpoints`), which is clean for Docker/Kubernetes but
means local, non-containerized runs need either root-owned directories at
those exact absolute paths or a manual, uncommitted edit to the config —
there's no environment-aware fallback the way `serve.py`'s `CHECKPOINT_PATH`
env var provides. Getting the checkpoint to actually hand off between the
training Job and the serving Deployment — two separate Pods, no shared
filesystem by default — was the other real "aha": it only works because both
mount the same `ReadWriteOnce` PVC, and that only works because nothing
schedules them onto different nodes simultaneously, which is a constraint
worth being honest about rather than glossing over.

## 2. What would you do differently if deploying this to production?

- **Compute**: train on GPU nodes rather than CPU wheels; the current setup
  is deliberately CPU-only to keep images small for local grading, not for
  real training throughput.
- **Storage**: replace the `data-pvc`/`checkpoints-pvc` pair with object
  storage (S3/GCS/etc.). `ReadWriteOnce` PVCs work for a single-node local
  cluster but don't scale to multi-node clusters or let you fan the
  checkpoint out to many serving replicas cleanly.
- **Model versioning**: `classifier_v1.pt` is a single mutable filename.
  A real deployment needs an actual model registry (MLflow, a versioned
  bucket path, etc.) so rollbacks and A/B comparisons are possible.
- **Secrets**: `model-registry`'s token is a committed placeholder by
  design. In production that becomes a real secrets manager (Vault, Sealed
  Secrets, cloud KMS-backed secrets) with rotation, never a value that ever
  touches git history even as a dummy.
- **Rollout safety**: the current `maxSurge: 1`/`maxUnavailable: 0` rolling
  update is a reasonable default, but production would add canary or
  blue/green rollout plus automated rollback tied to error-rate/latency
  SLOs, not just the `/health` probe.
- **Observability**: structured request logging, prediction-latency and
  confidence-distribution metrics, and drift detection — right now the only
  signal is whether the process is alive and the model loaded.
- **CI/CD**: extend `ci.yml` beyond lint/test to build and push the images,
  and ideally trigger retraining on a schedule or on new labeled data rather
  than only on manual `docker build`.

## 3. What did you learn about the MLOps lifecycle going from local development to Docker to Kubernetes?

The same Python code behaves differently at each stage, and the interesting
work is in the seams between them rather than in the model itself. Locally,
paths, environment variables, and "is the model loaded yet" are all things
you control by hand. In Docker, that control shrinks to whatever you baked
into the image plus what you mount or pass as `-e`/`-v` — which is what
pushed `serve.py` toward environment-variable overrides (`CHECKPOINT_PATH`)
rather than CLI flags, since nothing runs `serve.py` interactively once it's
`CMD`. In Kubernetes, it shrinks further and gets more explicit at the same
time: a `ConfigMap` and `Secret` decouple hyperparameters and credentials
from the image entirely, so it's genuinely possible to reconfigure training
without a rebuild — but you also inherit real orchestration concerns that
don't exist on a laptop, like the training `Job` needing to *complete*
before the serving `Deployment`'s readiness probe has any chance of
succeeding, or a liveness probe misconfigured to fire before a slow model
load finishes and get the Pod killed for no real reason. Readiness vs.
liveness semantics, `imagePullPolicy: IfNotPresent` for locally-built images,
and `ReadWriteOnce` PVC constraints all felt like abstract Kubernetes trivia
until this pipeline forced me to reason about them for a service that
actually had to come up in the right order and stay up.
