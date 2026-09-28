# Preserving a primary conversation around auxiliary calls (Windows, opt-in)

An agent may send an approval check or summary to the same single-sequence server it uses for its primary
conversation. Ordinary prompt checkpoints preserve the recurrent state and indexer tails, but the positional
KV storage is overwritten by the unrelated request. Returning to the primary can therefore require rereading
its entire prompt.

`--conversation-cache-mib N` enables one bounded, pageable host-RAM snapshot around **explicitly marked**
auxiliary calls. It defaults to zero/off. It requires `--prompt-cache` greater than zero. This implementation
uses Windows physical-memory telemetry for admission; other platforms safely decline admission.

## Client and engine settings

Add the following to the engine arguments in the server's JSON, keeping the rest of the existing configuration:

```text
--prompt-cache 2 --conversation-cache-mib 2048
```

Send `"strata_auxiliary": true` as a top-level field of an OpenAI-compatible `/v1/chat/completions` or
Anthropic-compatible `/v1/messages` request **only for the side call**. With an SDK exposing `extra_body`, use:

```python
extra_body={"strata_auxiliary": True}
```

The Python adapter forwards that boolean as `aux=1` on both `GEN` and `GENI`. Missing, false, numeric and string
values do not opt in. No classification based on prompt text or reasoning level is performed. Main and ordinary
subagent requests should remain unmarked. The patched adapter and engine must be used together.

For Hermes versions that support auxiliary `extra_body`, merge this into the existing configuration and restart
the client. Preserve the existing providers, timeouts and reasoning choices; do not duplicate the `auxiliary` key.

```yaml
auxiliary:
  approval:
    extra_body:
      strata_auxiliary: true
  compression:
    extra_body:
      strata_auxiliary: true
```

## Lifecycle and memory accounting

1. The first marked side call after a valid primary parks its complete mutable sequence state and checkpoints.
2. Further marked side calls leave that snapshot intact, including when their prompts are longer than the primary.
3. The next primary restores the snapshot only when its token prefix, image identities and control-vector setting
   match and offer more reuse than the live state. Otherwise the parked state is discarded.

The snapshot includes GDN recurrent/convolution state, PLE history and token metadata, the complete QSA device
state arenas and authoritative host KV allocations, and MTP device/host state. Weights and expert placement are
not copied. Device pointers remain unchanged for CUDA graph reuse. Image positions are reconstructed from the
incoming request as usual. This does not create another model instance or allow parallel decoding.

Admission checks the byte cap and currently available physical RAM, leaving 2,560 MiB immediately after the
snapshot allocation. Ordinary retained checkpoints are additional to the cap. Other processes can consume RAM
later; this is not a continuous reservation. Allocation/copy failure clears the snapshot and falls back to cold
prefill. A failed partial restore terminates the engine rather than continuing with inconsistent state.

Snapshots are temporary, disappear on restart, and protect only one primary around marked calls. Arbitrary
project/subagent switches and unmarked requests can still miss. Compression usually rewrites the primary
transcript, which still requires a fresh read. The auxiliary inference itself is not eliminated. This is a
prompt-reuse improvement, not a promised increase in decode throughput.

## Validation

The adapter tests run without a model or GPU:

```text
python -m unittest serve.test_server -v
```

`src/program/conversation_state_test.cpp` provides a synthetic real-CUDA + host-memory round trip, replacement,
empty/invalid ranges, overflow-safe admission, memory-floor checks and failed-admission invalidation. Its checks
remain active in Release builds. It needs a CUDA device but no model weights. Build the standalone target with
`cmake --build build --target conversation_state_test` and run the resulting executable. This target also works
with `STRATA_BUILD_TESTS=OFF` in published source snapshots that omit the upstream `tests/` directory.

For model acceptance, reserve the server for the test and retain the request JSON and engine logs:

- Primary lookup, two different marked auxiliaries, then exact-prefix primary continuation: verify a known answer
  and substantial `RESUME`/`reused` count with `strata conversation: parked` and `restored` log entries.
- An auxiliary longer than its parent must not evict it.
- Rewrite the primary history, then repeat the auxiliary/primary sequence: the obsolete snapshot must be discarded
  and the new primary must subsequently restore.
- Check same-image restoration and changed-image invalidation.
- Exercise host-backed KV positions beyond the resident window with MTP enabled, plus a known tool call and
  independently tested generated code. Record peak RAM and dedicated/shared GPU memory separately.

Initial local measurements used Strata 0.1.12 plus this patch, Swift 1.5 GSQ-RCO IQ3_XXS, a Ryzen 9 7940HS,
64 GB RAM and RTX 4070 Laptop 8 GB, 65,536 context, int8 KV, 32,768 resident KV and MTP spec 5. These historical
numbers are **not measurements of newer upstream prompt kernels**:

| Case | Snapshot off | Snapshot on |
|---|---:|---:|
| 5,585-token continuation after auxiliary calls: prompt time | 21.775 s | 0.334 s |
| Same continuation: reused tokens | 0 | 5,578 |
| 41,570-token continuation after auxiliaries | not measured in paired control | 41,563 reused; 0.439 s prompt time |

The known lookup answers were correct. The snapshot occupied approximately 1,514 MiB. Reported timings include
the restore within prompt processing; they exclude unrelated tool execution. They do not establish bitwise parity
under every model/configuration or guarantee arbitrary agent-task success.

To disable the feature, remove `--conversation-cache-mib` (or set it to zero) and stop marking auxiliary calls.
No model conversion or project-history change is required.
