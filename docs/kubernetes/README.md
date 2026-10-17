# Strata on Kubernetes

Plain manifests for one Strata server on one GPU node. They run the image built from this repository's `Dockerfile`
through its `docker-entrypoint.sh`, so every setting is one of the entrypoint's env vars, as in
[INSTALL.md, Docker](../INSTALL.md#docker-linux). They work with `kubectl apply -k`, Kustomize, Argo CD / Flux, or
kapp.

| File | What it is |
|---|---|
| `kustomization.yaml` | The namespace (`strata`) and your image name. |
| `namespace.yaml`, `pvc.yaml` | The namespace and the data volume (model, pack, configs: about 70 GB for IQ3_S). |
| `deployment.yaml` | The server: one replica, `Recreate`, the NVIDIA runtime class, `IPC_LOCK`, long start-up probe. |
| `service.yaml` | Port `http` (8080), label `app: strata`, which is what [../monitoring/servicemonitor.yaml](../monitoring/servicemonitor.yaml) selects. |
| `alerts.yaml` | Optional Prometheus Operator rules (not in the kustomization). |

## Before you start

- A node with NVIDIA GPUs and driver 580 or newer, exposed to Kubernetes: the NVIDIA GPU Operator, or k3s with the
  NVIDIA Container Toolkit installed on the node (k3s then registers a `nvidia` runtime class).
- 64 GB of RAM or more on that node (Strata keeps 32-62 GB of the model in page-locked RAM).
- The image, built once:
  `docker build -t registry.example.com/strata:0.1.40.3 --build-arg CUDA_ARCHITECTURES=120 .` (your cards'
  architecture; the default builds them all), then either `docker push` it, or on a single-node k3s import it:
  `docker save registry.example.com/strata:0.1.40.3 | sudo k3s ctr -n k8s.io images import -`.
  Check it is there with `sudo k3s ctr -n k8s.io images ls -q | grep strata` before you deploy. A pod whose image is
  only in Docker's store, not in containerd's, stays in `ErrImageNeverPull` / `ImagePullBackOff`.

## Install

```sh
kubectl create namespace strata
kubectl -n strata create secret generic strata-api-key --from-literal=key="$(openssl rand -hex 24)"
# edit kustomization.yaml: images[0].newName / newTag = the image above
kubectl apply -k docs/kubernetes
kubectl -n strata logs -f deploy/strata
```

The first start downloads the model into the volume (~70 GB) and prepares it, then serves. The start-up probe allows
two hours for that. Later starts take 1-3 minutes. The server is then at `http://strata.strata.svc:8080/v1`
(OpenAI and Anthropic APIs, bearer or `x-api-key` with the secret's key). `GET /health` answers without the key.

With kapp (Carvel), the same files:

```sh
kapp deploy -a strata -f <(kubectl kustomize docs/kubernetes) --wait-timeout 2h
```

kapp waits for the Deployment to be ready, hence the timeout for the first start's download. For the same reason
`deployment.yaml` sets `progressDeadlineSeconds: 7200`: with the default 10 minutes, `kubectl rollout status` reports a
first start that is still downloading as failed.

## Choosing the model and the config

The env vars in `deployment.yaml` are the entrypoint's: `FAMILY`, `MODEL`, `CONTEXT`, `VISION`, `KV`, `GPUS`,
`LAYER_SPLIT`, `LOW_RAM`... (INSTALL.md lists them). The first start records the setup in
`/data/config/strata-<model>.json`. `REINSTALL=1` redoes it after you change one of those settings.

To run a config you edited yourself (extra engine args such as `--batch 8 --batch-groups 4`, a conversation cache
size, `"tool_call_recovery": true`), copy it into the volume under its own name and point `CONFIG` at it:

```yaml
- { name: CONFIG, value: /data/config/strata-iq3_s.batch.json }
```

The config's `"host"` must be `"0.0.0.0"`: a config written with `HOST=127.0.0.1` (a server behind a sidecar proxy)
listens on the pod's loopback only, so the probes fail and the Service gets "connection refused". `HOST` only
applies when setup writes a config, not to an existing one. The entrypoint prints one `Config:` line naming the file it starts. Look for it after every change. Since 0.1.40.2
the entrypoint makes its own link to the config, so a pod command that links a file before it is honoured only when
the link points into `/data/config/` (0.1.40.3). `CONFIG` is the explicit way. A server that quietly starts the
wrong config still answers, only slower. The `StrataBatchSlotsMissing` alert catches the batch case.

## Several GPUs

One model across several cards of the node (docs/MULTI_GPU.md): request them all and name them.

```yaml
env:
  - { name: GPUS, value: "0,1,2,3" }
  - { name: LAYER_SPLIT, value: "12,24,36" }   # optional: setup picks a split otherwise
resources:
  limits:
    nvidia.com/gpu: 4
```

Inside the pod the cards are numbered from 0 whichever physical ones the device plugin gave it.

## Memory, stopping, updating

- **Page-locked RAM.** The pod adds `IPC_LOCK`. The node's runtime must not cap `RLIMIT_MEMLOCK` (containerd's and
  k3s's systemd units leave it unlimited). There is no memory limit, because setup.py sizes the RAM it uses from the
  host, not from a cgroup limit. Under a limit, set `LOW_RAM=on`.
- **Stopping** releases tens of GB of page-locked RAM, which takes longer than Kubernetes' default 30 s:
  `terminationGracePeriodSeconds: 90`.
- **One pod at a time.** `strategy: Recreate`, since a second server cannot fit next to the first on the same GPUs.
  An update stops the old pod before the new one loads: plan for the 1-3 minute gap.
- **Updating** is a new image tag in `kustomization.yaml` (after `docker build` and push or import) and
  `kubectl apply -k` again. The volume is kept: no new download. A setup change needs `REINSTALL=1` once.

## Monitoring

`GET /metrics` serves Prometheus's text format with vLLM's metric names, so the scrape config, the `ServiceMonitor`
and the Grafana dashboard in [../monitoring/](../monitoring/) apply as they are. The `ServiceMonitor` reads the API
key from the same `strata-api-key` secret, which must then exist in its namespace too. `alerts.yaml` adds four rules
(slow decode, a queue that never empties, a first token over a minute, missing batch slots). Their thresholds come
from several days of a 4-GPU server's history.
