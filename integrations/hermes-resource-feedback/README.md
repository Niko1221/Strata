# Supervisor resource feedback for Hermes

Experimental, opt-in integration. It supplements Strata's reactive system-pressure
controller with notice **before** a known local tool consumes resources. It does
not change model weights, quantization, context length, reasoning or tool approval.

This plugin is installed separately from Hermes. It requires the proposed generic
`authorized_tool_execution` middleware, trusted tool-lineage context and
`register_private_env_keys` plugin API in
[Hermes PR #135418](https://github.com/NousResearch/hermes-agent/pull/135418); older
Hermes versions must refuse to load it rather than simulate the boundary with a
pre-tool hook. Strata requires the resource-lease implementation in adaptive
[PR #1493](https://github.com/Niko1221/Strata/pull/1493), not just an official release.

## One supervisor owns the plan

1. An operator configures named tool classes and their command prefixes. These
   are declarations of expected resource use, not a shell parser or an automatic
   understanding of every command.
2. The supervisor calls `resource_plan` with the classes its task needs. A worker
   cannot change that plan. If a worker needs an ungranted class, it reports the
   need to the supervisor through ordinary delegation feedback.
3. After Hermes's command approval checks, eligible tool work enters one FIFO
   broker. It uses runtime parent/root identities; the model does not supply an
   owner ID, engine URL or control credential.
4. The broker asks Strata for a bounded lease and waits for a fresh `ready`
   acknowledgment. Strata drains existing work, unloads its native and vision
   processes, and checks physical RAM, commit and GPU headroom. Acknowledging the
   request alone is not permission to start the command.
5. The approved foreground command executes once. A renewal thread keeps its
   lease alive. The broker releases the lease when the command returns, including
   exceptions. Failed cleanup does not replace a successful command result or
   cause it to run twice; the engine's expiry remains the fallback.
6. The next normal model request waits for measured reload capacity. Ordinary
   pressure monitoring still covers unrelated apps and inaccurate estimates.

Sequential delegated agents are the initial acceptance target. Multiple eligible
workers are serialized by the broker; they never send competing engine commands.
Unrelated lightweight tools may run normally. This does not provide parallel LLM
execution on a single engine.

## Configuration

Keep the control credential in a profile-scoped secret, not a model prompt, skill,
tool argument or published JSON. Both processes need the same randomly generated
ASCII secret of at least 32 characters with no whitespace. The default secret name is
`STRATA_RESOURCE_LEASE_TOKEN`.
The plugin registers this name as private so Hermes's existing child-environment
sanitizer removes it from terminal subprocesses, including explicit forwarding
extras. Only the name is registered, never its value. Changing `token_env`
requires reloading the plugin so configuration and sanitization cannot disagree.

Strata's existing experimental live-memory/coadaptive/idle-parking prerequisites
must already be satisfied. Add this opt-in block inside `coadaptive`:

```json
"tool_leases": {
  "enabled": true,
  "token_env": "STRATA_RESOURCE_LEASE_TOKEN"
}
```

Install this directory through Hermes's normal local plugin workflow, enable the
plugin and its `resource_feedback` toolset, then configure its settings:

```yaml
plugins:
  enabled: [resource-feedback]
  isolation: in_process
  entries:
    resource-feedback:
      settings:
        enabled: true
        base_url: http://127.0.0.1:8080
        token_env: STRATA_RESOURCE_LEASE_TOKEN
        wait_seconds: 180
        default_profiles: []
        profiles:
          compile:
            command_prefixes: ["cmake --build "]
            ram_headroom_gib: 6
            vram_headroom_mib: 250
            ttl_seconds: 60
          gpu_test:
            command_prefixes: ["python tools/render_acceptance.py"]
            ram_headroom_gib: 4
            vram_headroom_mib: 4096
            ttl_seconds: 60
```

The numbers are **total free capacity requested before the tool**, not predicted
bytes Strata must evict. They are examples, not universal safe budgets. Use exact,
nonoverlapping command prefixes appropriate to the project; a wrapper script can
make one bounded command unambiguous. A general `python` or shell prefix is too
broad. A tool that launches another model call would deadlock behind its own lease
and is outside this initial contract.

The integration requires trusted **in-process** plugin execution: its runtime
lineage and synchronous continuation do not cross Hermes's plugin-host RPC
boundary. Host-process plugin isolation is not supported by this prototype.

`default_profiles` can contain operator-preauthorized classes. An explicit
supervisor plan overrides those defaults for that root session. Neither option
bypasses Hermes approval or allows workers to rewrite the supervisor plan.
Plans are process-local and bounded to 4,096 roots per broker configuration; a
full table refuses new plans instead of silently evicting an explicit denial.

The supervisor can start with:

```json
{"profiles": ["compile", "gpu_test"]}
```

as arguments to `resource_plan`. It should then delegate one worker at a time.
Workers use the ordinary terminal tool with a bounded foreground timeout. They
must not directly call Strata's control endpoint. No new model tool appears unless
this integration is explicitly enabled.

## Limits

- The initial actuator is **full unload**, which releases fixed allocations too,
  but loses warm model state and can add considerable reload/prefill time. Do not
  use it for every file read or tiny build. Live selective cache relief remains a
  separate existing controller; automatic cheapest-action selection is not claimed.
- A lease coordinates cooperating clients. It cannot reserve Windows memory
  against a new browser/game allocation, guarantee frame rates, prevent every OOM,
  or make a tool fit when the tool itself exceeds the machine's capacity.
- Readiness is sampled. Memory can change immediately afterward. TTL expiry and
  renewal failure are not guarantees that a foreground subprocess has exited;
  reactive capacity admission remains necessary before inference reloads.
- Only local foreground commands are covered. Explicit background processes and
  foreground-to-background promotion are refused for matched resource work.
  Protected foreground continuations do not detach through Hermes's yield path.
  Arbitrary shell scripts can still launch their own daemons: this is not a
  sandbox for untrusted code.
- The control credential is stripped from tool environments. Identity isolation
  is still cooperative runtime scoping. Code running as the same OS
  user can read that user's secrets; child agents are not separate security users.
- Active inference is allowed to reach its existing boundary. A lease does not
  kill another request, steal another owner's lease, or replay a tool command.
- No GPU affinity is changed. A game configured for an iGPU stays on the iGPU.
  An explicitly GPU-hungry validation tool can ask for headroom, but still needs
  its own device selection and correctness checks.

## Sources and implementation boundary

- [MARS](https://arxiv.org/abs/2604.26963) and its
  [OpenHands integration](https://github.com/Afterglow231/MARS_preview) inspired
  sharing tool intent with inference scheduling and separating admission from
  execution. No source, full scheduler or published performance result is imported.
- [vLLM sleep/wake](https://docs.vllm.ai/en/latest/features/sleep_mode/) is prior
  art for an explicit engine release/return control boundary. This integration
  uses Strata's own lifecycle; it does not copy vLLM's allocator or promise its
  wake-up times.
- Hermes's existing plugin, dispatch, profile scope and parent/child runtime are
  the integration foundation. The companion Hermes change is a generic execution
  boundary, not a Strata dependency in the agent core.
- [InferCept](https://arxiv.org/abs/2402.01869) is related cache-retention work.
  Its algorithms were reviewed but are not implemented here. Similarly, Ray's
  logical resource admission is related work, not a new dependency.

Tests must include real terminal approval/dispatch, supervisor/worker lineage,
FIFO ownership, failed readiness, cancellation, release/expiry, and a real
compiler-feedback cycle. Unit-test success alone is not an end-to-end resource
or performance claim.
