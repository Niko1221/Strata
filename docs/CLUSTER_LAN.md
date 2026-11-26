# LAN cluster groundwork (opt-in, draft)

This page describes the **groundwork** for an opt-in LAN cluster mode. It does not run a distributed model yet: it registers machines, reads their real resources, and watches their state. The full expert-sharding mode is a scope proposal in #7071 and the earlier #6938; maintainers have not scheduled it.

## What works today

- Create a named cluster and register nodes from the app's **More → Cluster** screen, or with `strata cluster` in a Terminal checkout.
- When a node is added, the host contacts its agent and records the machine's hostname, operating system, CPU cores, GPU name, VRAM and RAM.
- **Refresh** re-probes every node and updates its state and `last_seen`.
- A node that cannot be reached is shown as **Not reachable**; this is a real connection result, not a placeholder.
- Arabic is available throughout the screen; right-to-left layout follows the app language.

## Running the agent on a machine

The agent is a small authenticated JSON endpoint. It is **not** an inference server and does not load a model. Run it on every machine you want to register, from the Terminal folder of this checkout:

```sh
# Linux / macOS
export STRATA_CLUSTER_TOKEN='same-secret-on-every-node'
python -m strata_control.cluster.agent --host 0.0.0.0 --port 8899
```

```powershell
# Windows PowerShell
$env:STRATA_CLUSTER_TOKEN='same-secret-on-every-node'
python -m strata_control.cluster.agent --host 0.0.0.0 --port 8899
```

Use the **same token** on the app host and on every agent. The token is never sent from the browser: the Node host reads it from `STRATA_CLUSTER_TOKEN` and sends it as an `X-Cluster-Token` header.

**Use port 8899 for the agent.** Port 8791 is already used by the Strata host itself.

In the app, add a node with that machine's IP and port 8899. For a local test, use `127.0.0.1:8899` while the agent runs on the same computer.

## What the probe reads

`GET /cluster/info` returns:

| Field | Source |
| --- | --- |
| `hostname`, `os` | Python `platform` |
| `cpu_cores` | `os.cpu_count()` |
| `gpu`, `vram_gib` | `nvidia-smi` when present, otherwise the `STRATA_GPU_NAME` environment variable or `none` |
| `ram_gib` | Windows `GlobalMemoryStatusEx` or `/proc/meminfo` |
| `status` | `healthy` when the token is accepted and the endpoint answers |

No measurement is invented: when a source is unavailable the field is empty or zero, and the screen shows a dash.

## Degradation

If a node disappears, the next **Refresh** marks it **Not reachable**. The cluster file remains readable and the other nodes keep their state. This is a control-plane guarantee only: no generation is in flight in this draft, so there is no session to corrupt. A full implementation must add an explicit drain/rebalance path before any expert traffic crosses the network; that work is part of the #7071 scope.

## Security

- Keep the LAN private. Bind the agent to a trusted interface where possible.
- The shared token is a secret: do not paste it into chat, screenshots or committed files.
- This groundwork uses plain HTTP with a shared token. TLS or a private network (Tailscale/WireGuard) is required before any real deployment; it is part of the scope discussion.
- See [SAFETY.md](SAFETY.md) for Strata's general network and privacy rules.

## Limits

- No model weights are sharded, no experts are moved, and no inference runs across machines.
- The launch button on the screen reports that execution is not implemented yet.
- Discovery is manual (IP entry). mDNS/UDP discovery is a later phase.
- Heterogeneous backends (NVIDIA + AMD + Apple Silicon) are not covered; the existing single-machine rules in [MULTI_GPU.md](MULTI_GPU.md) and [DGX_SPARK_LINUX.md](DGX_SPARK_LINUX.md) still apply.

## Validation

```sh
# From app/host
node --test tests/cluster-ui.test.mjs

# From terminal/
python -m pytest tests/terminal/test_cluster_agent.py tests/terminal/test_cluster_registry.py tests/terminal/test_cluster_cmd.py -q
```

The UI test starts a local HTTP endpoint and verifies that adding a node records real GPU, VRAM and RAM values; a second check stops that endpoint and verifies that a refresh marks the node offline without breaking the stored cluster.
