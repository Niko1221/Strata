# Supervisor resource leases

Experimental, opt-in coordination for a local agent that is about to run a
resource-heavy tool. A trusted supervisor asks Strata to drain existing model
work and report when the requested free capacity is observed. The default mode
unloads the model and vision encoder. Explicit resident modes can instead keep
the model loaded, release live cache capacity, or escalate when their contract
permits unload. The supervisor acknowledges execution through the owner-bound
`start` handshake, runs the tool once, renews the lease and releases it after the
bounded foreground command and its owned children finish.

This supplements [adaptive pressure handling](ADAPTIVE_STRATA.md). It does not
change weights, quantization, reasoning, context length, or tool approval.
Resident action selection is an experimental extension of the earlier unload-only
implementation. It uses a conservative feasibility order, not a learned scheduler
or a demonstrated universally cheapest action. It does not move arbitrary CPU
computation onto another GPU. Earlier unload-only measurements do not qualify
the new modes or establish an end-to-end speedup.

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
  "max_ttl_seconds": 300,
  "resume_ram_working_gib": 2,
  "resume_vram_working_mib": 256,
  "resume_timeout_seconds": 30
}
```

The feature is disabled when omitted. `max_ttl_seconds` must be an integer from
30 through 300. Put a randomly generated ASCII secret of at least 32 characters,
without whitespace, in the named environment variable for the Strata server and
trusted supervisor. Use a separate secret from the normal model API key. The
JSON contains the environment variable's name, not its value.

Resident relief also requires the native capability `memory_hold=1`. This adds
the optional `hold` suffix to the existing `MEMORY` command: a lease-owned resize
can release caches without immediately refilling them into the newly freed
capacity. An engine without that capability retains legacy unload support;
`auto` may fall back to verified unload and strict `relieve` fails. Do not forge
the INFO capability or send the extension to a binary that does not advertise it.

The resume allowances are additional working space above the effective base
RAM/commit and native VRAM floors, not another charge for already-held weights.
They default to **2 GiB RAM** and **256 MiB VRAM**, with a **30-second** bounded
return-admission wait. Configurable ranges are 0–128 GiB, 0–65,536 MiB and
1–300 seconds respectively. These are planning allowances that require
workload-specific qualification, not a guarantee for every 64K prompt, MTP,
vision or decode peak. A configured 64K context ceiling is not proof that the
allowance covers fully occupied 64K inference.

An operator can explicitly set `resume_vram_working_mib: 0` for a bounded,
qualified workload whose working buffers are already allocated. This removes
only the additional return allowance; it does not lower the configured native
VRAM floor or the actual-free-VRAM guard. For example, a qualification profile
with a 320 MiB native floor and a 250 MiB actual-free guard retains both limits.
That profile is not a recommendation for arbitrary prompts or a general 64K
memory guarantee. Record the override in both sides of any matched comparison.

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

The mode has the following meaning:

| Mode | Allowed result after drain |
| --- | --- |
| `unload` | Preserve full native-and-vision unload behavior. Existing broker profiles default to it. |
| `auto` | Select no residency change (`none`), acknowledged live relief (`relieve`), or verified unload. |
| `relieve` | Permit `none` or acknowledged live relief only. Fail explicitly if unload is needed. |

`none` is an action, not an acquisition mode. It still owns the inference barrier:
existing inference/preparation must drain and fresh native capacity must qualify
the same current process. It never means that inference can run alongside the tool.

An extended acquisition can also include `execution_ram_floor_gib` and
`execution_vram_floor_mib`. These are explicit **total free-capacity floors during
tool execution**, not additional allocations. Each must be finite, nonnegative
and no greater than the matching admission target. If omitted, it defaults to
that target, conservatively retaining the old interpretation. For example, a
trusted 12 GiB admission target and 4 GiB execution floor are distinct quantities;
the implementation does not infer an 8 GiB workload estimate from a single
legacy target. Effective safety floors can be higher than requested values.

An accepted request returns HTTP 200 with:

```json
{
  "schema": "strata.resource-lease.v1",
  "enabled": true,
  "supported_modes": ["unload", "auto", "relieve"],
  "supports_start": true,
  "lease_id": "<stable identifier>",
  "lease_token": "<opaque owner capability>",
  "state": "pending",
  "reason": "draining_requests",
  "expires_in_seconds": 60,
  "mode": "unload",
  "phase": "admission",
  "selected_action": "unload",
  "ram_headroom_gib": 6,
  "vram_headroom_mib": 1024,
  "execution_ram_floor_gib": 6,
  "execution_vram_floor_mib": 1024,
  "execution_hold": false,
  "availability_is_reservation": false
}
```

`pending` acknowledges the intention; it is not permission to run the tool.
Poll using the control secret plus the returned owner capability:

```json
{"action": "status", "lease_token": "<opaque owner capability>"}
```

After owner READY, commit the transition immediately before synchronous
command dispatch:

```json
{"action": "start", "lease_token": "<opaque owner capability>"}
```

Require the same `lease_id` and requested mode, `state: "ready"`,
`phase: "execution"`, a permitted selected action, the expected targets and a
positive remaining TTL. `start` is idempotent for the same already-executing owner;
it does not renew TTL, execute a tool, or replay a model request. It rejects a
terminal, expired, non-ready or stale-ready lease. The service rechecks current
capacity and ownership before accepting the transition.

The standalone broker does not retry a lost or ambiguous start reply. It leaves
the continuation uncalled and attempts owner release. New modes therefore need
the advertised handshake capability; they cannot be simulated against an older
server by treating public discovery or an acquisition receipt as READY.

Renew and release use the same authority:

```json
{"action": "renew", "lease_token": "<opaque owner capability>", "ttl_seconds": 60}
```

```json
{"action": "release", "lease_token": "<opaque owner capability>"}
```

Status, start, renew and release responses omit `lease_token`. Renewal starts a new
bounded interval from the monotonic clock. It cannot revive an expired,
released, failed or superseded lease. Release is idempotent. An old owner's
token never releases a successor's lease.

Only one lease or unresolved execution/native-operation hold can be active.
Another acquisition receives HTTP 409. Repeating
the same UUID and identical parameters returns the same capability without
extending its deadline; changed parameters receive 409. Retry history is
in-memory, bounded to the most recent 128 acquisitions, and does not survive
server restart. A supervisor must not use old IDs as durable job storage.

`GET /v1/resource-lease` exposes only local, redacted state and capabilities. It
returns no request ID, lease ID, token or resource targets. It is not an owner
readiness acknowledgment; owners must use authenticated POST status.

Compatibility is explicit. The broker's unchanged unload-only profiles can use
an older server's original acquisition/status/release sequence. A new owner view
advertising `supports_start: true` requires the handshake even for `unload`.
Explicit `auto`, `relieve` or execution-floor options require advertised support
before acquisition; an older endpoint is rejected rather than silently changing
the requested contract.

## Lifecycle and cancellation

| State | Meaning |
| --- | --- |
| `pending` | Existing generation/preparation is draining, or released capacity is still below a target or cannot be measured. |
| `relieving` | An owned native memory operation is pending; its acknowledgement and fresh capacity must be reconciled. |
| `unloading` | One lifecycle owner is ending the native model and vision processes. |
| `ready` | The selected action has completed and fresh capacity meets the current phase's targets and effective floors. An unload requires both processes dead; a resident action requires the same live processes, acknowledged ceilings and no pending native resize. |
| `released` | The owner withdrew the lease. No model reload is triggered. |
| `expired` | The monotonic lease deadline passed. An extended execution hold or unresolved native operation can still retain the inference barrier. |
| `failed` | Safe memory release could not be established. The tool must not start. |

An existing model request or preparation owner may finish. New requests wait
outside the native FIFO and do not start image encoding or model loading.
Streaming clients receive keep-alives while waiting; disconnect and shutdown
cancel their wait. A lease never kills another active generation to obtain RAM.

The background controller takes the same FIFO and lifecycle lock used by idle
parking. Existing in-flight cache growth must drain first. For a resident lease,
it seals RAM/GPU residency ceilings to the acknowledged current process; a later
phase may reduce those ceilings but cannot silently grow them. The memory
observer remains the sole sender of live `MEMORY` operations. A hold-bearing
acknowledgement prevents immediate refill from consuming capacity intended for
the tool. The execution free-capacity floor is separate from this residency
ceiling: lowering a requested floor does not authorize cache growth.

Each continuous capacity-deficit or observed unresolved-control episode is bounded
to 30 seconds before `auto` may escalate to owned unload; strict `relieve` fails
instead. Missing telemetry can wait until the lease or broker admission deadline;
it cannot grant readiness. Lease expiry is also authoritative for
whether the tool may start. A timeout, release or failed admission is not proof
that native work ended: an unresolved resize/drain retains the barrier until
its own acknowledgement or verified teardown. A stale acknowledgement cannot
clear a newer operation.

For unload, the controller records the full reload footprint first. Completion
requires confirmed native **and** vision process death, not only an unload
request or a stale low-memory counter. If expiry/release happens during teardown,
the barrier remains until teardown finishes. An expired pending request is not
unloaded later merely because the engine becomes idle.

Once an `auto` or `relieve` owner has successfully called `start`, expiry or
renewal failure does **not** reopen inference. Its `execution_hold` remains until
explicit owner release, since TTL expiry does not establish that the compiler
or its children stopped. The broker attempts release, including an expired lease,
only after recognized synchronous terminal completion from its trusted bounded
wrapper. Callback exceptions, terminal errors, timeout/interruption, background
metadata and unknown result shapes retain the extended hold without release.
Hermes may omit backend flags, so reserved timeout and signal-style exit codes
and standard output markers are treated conservatively. An ordinary completed
compiler failure such as exit code 1 can still release. The actual result or
exception is preserved without retry. Uncertain completion or a failed release
requires operator recovery after establishing that owned processes exited.
Legacy `unload` retains its earlier release-on-return/error and TTL behavior.
Background/detached work requires a different lifetime contract and is not
supported by this broker; a normal shell exit is not general daemon containment.

After resident release, the next request first waits for the same native process identity,
no pending resize, and fresh native/global readings taken after release and any
required acknowledgement. Configured vision must also be healthy before admission.
RAM and commit must meet the effective base floors
plus the configured RAM working allowance; native CUDA/DXGI headroom must meet
its base floor plus the VRAM allowance. Admission remains bounded, cancellable
and heartbeat-aware. Strict `relieve` fails without unloading when its bounded
wait expires. With `auto`, a freshly confirmed non-evictable VRAM floor or the
return-admission deadline can instead trigger the existing owned full-unload
path. It records a measured reload footprint before unloading, confirms death
of both native and vision processes, reconciles pending native controls and
clears the old lease targets only after verified teardown. The new request then
uses ordinary fresh, stable full-load admission; it is not dispatched on a stale
resident sample. Missing footprint data, a changed process or uncertain teardown
cannot authorize an unsafe reload. Cancellation still joins owned teardown.

This fallback occurs after the tool has completed and its lease was released.
It does not rerun the compiler or replay an accepted model stream. It can lose
the warm model state and add reload/prefill latency. Normal adaptive growth can recover only
after the lease ceilings have been cleared and existing stability/rate limits
permit it; release is not a request for eager full refill.

If the selected action unloaded the processes, ordinary requests instead use
the recorded full startup footprint and stable fresh capacity. A failed teardown
retains process handles and measured admission state, so an unverified survivor
cannot be bypassed by starting a second engine. These checks still cannot reserve
memory against an unrelated application that allocates immediately afterward.

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
  stopped. Extended execution holds require explicit release; reactive admission
  remains necessary before returning to inference.
- There is no multi-GPU placement, GPU affinity change, CPU thread reservation,
  automatic choice among SSDs, or migration of all allocations between tiers.
  Resident leases compose with existing adaptive cache controls under one
  acknowledged operation owner. A finite working RAM/VRAM requirement still
  exists, and unknown or insufficient capacity can pause work.
- Preserve matched quality and completion evidence when evaluating it. Neither
  successful unit tests nor a shorter foreground tool time establishes a faster
  complete agent task, and no universal OOM-free progress guarantee is made.

## Qualification boundary

The standalone broker has 54 focused tests passing on Python 3.11 and 3.14,
including nine tests against the real `ToolLeases` state machine with simulated
native lifecycle acknowledgements. These cover capability negotiation, legacy
wire compatibility, owner/phase/target validation, exactly-once dispatch, lost
start replies, strict relief, FIFO ordering, expiry holds, uncertain terminal
completion and release failure.
They establish protocol behavior, not real RAM/VRAM release or a performance
result. Native/service tests and live resident-mode acceptance must be tied to
their own final candidate revision before claiming C1 task completion or gains.

Implementation provenance and related work are recorded in
[ADAPTIVE_PROVENANCE.md](ADAPTIVE_PROVENANCE.md). The protocol and state machine
are an original implementation using Strata's existing lifecycle and admission
code; no external allocator or scheduling dependency is imported.
