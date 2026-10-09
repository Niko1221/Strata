# Supervisor resource leases

Experimental, opt-in coordination for a local agent that is about to run a
resource-heavy tool. A trusted supervisor asks Strata to drain existing model
work, unload the model and vision encoder, and report when the requested free
capacity is observed. The supervisor runs the tool only after `ready`, renews
the lease while the tool runs, and releases it when the tool returns.

This supplements [adaptive pressure handling](ADAPTIVE_STRATA.md). It does not
change weights, quantization, reasoning, context length, or tool approval.
The first implementation supports **full unload only**. It does not select a
cheaper partial eviction or move CPU computation onto another GPU.

## Requirements and configuration

This needs the experimental live-memory engine and active coadaptive mode, with
`idle_parking` enabled. Its existing requirements still apply: one supported
CUDA engine, one request at a time, measured reload admission, and no
`before_load` hook. It is not supported by an unmodified official release.

Add the following inside the existing `coadaptive` configuration:

```json
"tool_leases": {
  "enabled": true,
  "token_env": "STRATA_RESOURCE_LEASE_TOKEN",
  "max_ttl_seconds": 300
}
```

The feature is disabled when omitted. `max_ttl_seconds` must be an integer from
30 through 300. Put a randomly generated ASCII secret of at least 32 characters,
without whitespace, in the named environment variable for the Strata server and
trusted supervisor. Use a separate secret from the normal model API key. The
JSON contains the environment variable's name, not its value.

Do not put this credential or an acquired lease token in a prompt, model tool
arguments, public logs, or worker subprocess environment. An agent running shell
commands as the same OS user is not a security isolation boundary; this protocol
coordinates trusted cooperating clients rather than sandboxing hostile code.

The [standalone Hermes integration](../integrations/hermes-resource-feedback/README.md)
routes authorized supervisor and worker tool work through one supervisor broker.
It requires the companion [generic Hermes execution middleware, PR #135418](https://github.com/NousResearch/hermes-agent/pull/135418). Strata itself
has no Hermes dependency, and other supervisors can implement the same protocol.

## HTTP protocol

`POST /v1/resource-lease` accepts small JSON messages with an explicit
`Content-Length` of 1–4096 bytes. Chunked transfer is not accepted. Every POST
requires `Authorization: Bearer <control secret>`, including status polling.
The ordinary model API key and `x-api-key` do not authorize this endpoint.

Only a loopback peer is accepted. Requests carrying `Origin`, or a
`Sec-Fetch-Site` value other than empty or `none`, are rejected. This control
endpoint does not inherit wildcard CORS. Remote tunnels and browser pages are
not supported control clients.

Acquire a lease with a fresh UUID for this one tool operation:

```json
{
  "action": "acquire",
  "request_id": "61b34c2f-7e44-4b49-94f0-22b7338f7237",
  "mode": "unload",
  "ram_headroom_gib": 6,
  "vram_headroom_mib": 1024,
  "ttl_seconds": 60
}
```

These are **total free capacity targets**, not additional allocations or bytes
to evict. Strata checks them against fresh hardware totals and preserves higher
configured memory-policy, idle-parking and startup floors. Windows commit
headroom is checked separately from physical RAM. Values exceeding the host's
capacity are refused. These example targets are not a recommendation for every
compiler or GPU tool.

An accepted request returns HTTP 200 with:

```json
{
  "schema": "strata.resource-lease.v1",
  "enabled": true,
  "supported_modes": ["unload"],
  "lease_id": "<stable identifier>",
  "lease_token": "<opaque owner capability>",
  "state": "pending",
  "reason": "draining_requests",
  "expires_in_seconds": 60,
  "mode": "unload",
  "ram_headroom_gib": 6,
  "vram_headroom_mib": 1024,
  "availability_is_reservation": false
}
```

`pending` acknowledges the intention; it is not permission to run the tool.
Poll using the control secret plus the returned owner capability:

```json
{"action": "status", "lease_token": "<opaque owner capability>"}
```

Renew and release use the same authority:

```json
{"action": "renew", "lease_token": "<opaque owner capability>", "ttl_seconds": 60}
```

```json
{"action": "release", "lease_token": "<opaque owner capability>"}
```

Status, renew and release responses omit `lease_token`. Renewal starts a new
bounded interval from the monotonic clock. It cannot revive an expired,
released, failed or superseded lease. Release is idempotent. An old owner's
token never releases a successor's lease.

Only one lease can be active. Another acquisition receives HTTP 409. Repeating
the same UUID and identical parameters returns the same capability without
extending its deadline; changed parameters receive 409. Retry history is
in-memory, bounded to the most recent 128 acquisitions, and does not survive
server restart. A supervisor must not use old IDs as durable job storage.

`GET /v1/resource-lease` exposes only local, redacted state and capabilities. It
returns no request ID, lease ID, token or resource targets. It is not an owner
readiness acknowledgment; owners must use authenticated POST status.

## Lifecycle and cancellation

| State | Meaning |
| --- | --- |
| `pending` | Existing generation/preparation is draining, or released capacity is still below a target or cannot be measured. |
| `unloading` | One lifecycle owner is ending the native model and vision processes. |
| `ready` | Both processes are confirmed stopped and a fresh post-exit sample meets the requested targets and existing floors. |
| `released` | The owner withdrew the lease. No model reload is triggered. |
| `expired` | The monotonic lease deadline passed. No model reload is triggered. |
| `failed` | Safe memory release could not be established. The tool must not start. |

An existing model request or preparation owner may finish. New requests wait
outside the native FIFO and do not start image encoding or model loading.
Streaming clients receive keep-alives while waiting; disconnect and shutdown
cancel their wait. A lease never kills another active generation to obtain RAM.

The background controller takes the same FIFO and lifecycle lock used by idle
parking. It records the full reload footprint before unloading. `ready` requires
confirmed native **and** vision process death, not only an unload request or a
stale low-memory counter. If expiry/release happens during teardown, the barrier
remains until teardown finishes. An expired pending request is not unloaded
later merely because the engine becomes idle.

After release or expiry, the next ordinary model request still waits for the
recorded full startup footprint and stable fresh capacity. Unknown telemetry
blocks admission. A failed teardown retains process handles and measured
admission state, so an unverified survivor cannot be bypassed by starting a
second engine. Operator recovery may be required if a process cannot stop.

## Scope and costs

- `ready` means observed availability, **not an OS memory reservation**. Another
  application can consume memory immediately afterward. Status rechecks may
  return to `pending`. A tool must still handle its own allocation failures.
- Full unload releases fixed model allocations but loses warm state. Loading
  weights and reading the next prompt can cost more than a small compile saves.
  Use this for bounded heavy work, not each file read or trivial command.
- The lease does not estimate arbitrary programs, CPU requirements, child
  daemons or tool duration. A tool must not request model inference while holding
  its own lease: it would wait behind itself. Explicit background work needs a
  separate owner/lifetime contract and is not covered here.
- Renewal failure or expiry does not prove that a foreground process has
  stopped. Reactive memory admission remains necessary before reloading.
- There is no multi-GPU placement, GPU affinity change, CPU thread reservation,
  automatic choice among SSDs, or migration of all allocations between tiers.
  Existing adaptive cache controls remain separate. A finite working RAM/VRAM
  requirement still exists, and unknown or insufficient capacity can pause work.
- Preserve matched quality and completion evidence when evaluating it. Neither
  successful unit tests nor a shorter foreground tool time establishes a faster
  complete agent task, and no universal OOM-free progress guarantee is made.

Implementation provenance and related work are recorded in
[ADAPTIVE_PROVENANCE.md](ADAPTIVE_PROVENANCE.md). The protocol and state machine
are an original implementation using Strata's existing lifecycle and admission
code; no external allocator or scheduling dependency is imported.
