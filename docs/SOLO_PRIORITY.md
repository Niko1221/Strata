# Single-engine request priority (experimental)

This opt-in Python scheduler lets one request give the engine to a waiting higher-priority
request, then continue. Only one native `GEN` runs at a time, with its normal speculative
decoding and expert cache. It does not allocate batch slots.

## Enable

Add `"solo_priority": true` to the server config and restart it. Keep a conversation RAM
cache enabled, with at least two slots. For example, on a machine with enough free RAM:

```json
{
  "solo_priority": true,
  "args": [
    "--conversation-cache-mib", "32768",
    "--conversation-cache-slots", "8",
    "--conversation-cache-min-free-mib", "16384"
  ]
}
```

These are additions to an existing config, not a complete config. Memory requirements
depend on the model, context and other applications. `parallel`, `--batch`, `--slots`
and replicas cannot be combined with this option. Remove the option to restore the
original scheduler; no model, engine, cache format or tokenizer changes are needed.

## Requests

An integer `priority` in the JSON request body controls admission and preemption. Higher
goes first, following the convention proposed in [PR #2109](https://github.com/Niko1221/Strata/pull/2109).
Non-integers count as zero. Equal priorities are first come, first served.

With no priority field, requests with at most 4,096 input tokens and an output budget of
at most 512 tokens get priority 1; others get 0. A client must explicitly identify a
quick interactive request if its tool schemas, conversation or output limit exceed these
thresholds. The scheduler cannot know how long an unrestricted answer will actually be.
`priority: 0` disables the automatic short-request boost for that request.

```json
{
  "messages": [{"role": "user", "content": "Return a brief explanation of this error."}],
  "priority": 1,
  "max_tokens": 256,
  "stream": true
}
```

The running request gets at least five seconds before a pause, including after each
resume. After thirty seconds **waiting**, a request takes precedence over younger
waiters and gets the next eligible pause of the running request regardless of priority. Queue age resets when it starts
running again. These rules keep long work moving while letting later small requests
interrupt it too. A request that is almost finished
(32 tokens or fewer left in its output budget) is not paused.

## Continuation and limits

- The server sends `STOP` and consumes **all** remaining accepted `T` tokens through
  `DONE` before releasing the engine. A pause is not an API cancellation or a completed
  tool call. The original parser, UTF-8 decoder, stop matcher and tool-call IDs stay alive.
- The continuation sends the original token IDs plus accepted output, with the remaining
  output budget. The native conversation cache parks/restores state. No text decoding,
  prompt rewriting or summarization is involved.
- Sampling uses the same seed across continuations. Without an explicit seed the server
  chooses one once. Native Philox draws are indexed by absolute token position. This does
  not promise identical sampled answers across hardware or floating-point execution paths.
- Cold prefill stops at an engine boundary, so a large prefill chunk can still delay the
  short request. Cache capture, restore and any eviction also take time. This is not a
  hard real-time latency guarantee. A resumed request must produce a token before it
  can yield again, so repeated arrivals cannot repeatedly discard the same unfinished
  cold chunk. A request that has restored any cached prefix also finishes its remaining
  prefill and produces a token before yielding: STOP during that prefill can discard
  the restored prefix in the native engine. If a snapshot was evicted, its reread also
  gets to finish first.
- Image-bearing `GENI` requests and `strata_checkpoint: false` requests are not preempted.
  Vision encoding, cache control, VRAM resizing and session-file operations wait until
  suspended requests finish. They cannot alter the engine under a parked request.
- Cancellation while parked removes the waiter without sending `STOP` to someone else's
  request. Engine death/restart fails the parked stream explicitly; it is not silently
  resumed against a different engine process.
- `/metrics` includes `live.solo_priority`: pause count, waiting/paused counts, and
  `resume_prompt_read` (native tokens actually read on continuation) and
  `resume_prefix_missed` (previously processed prefix tokens not restored). The former
  includes the unfinished part of a cold prompt; the latter helps detect cache eviction
  or inadequate cache capacity.

## Tests and contribution scope

```sh
python -m unittest serve.test_solo_priority -v
python -m unittest discover -s serve -p 'test_*.py'
```

The fake-engine tests exercise the real protocol reader and streaming parser, including
tokens emitted after STOP, split UTF-8, preemption inside tool arguments, cold prefill,
disconnect/cancellation, engine failure and queue fairness. Real hardware measurements
must separately check latency, total throughput, cache reuse and output correctness.

The option defaults off. The existing single-engine protocol loop is retained in
`_generate_single`; its added STOP hook runs only when passed a pause callback. The batch
path is unchanged. The policy is isolated in `serve/solo_priority.py`; there are no native
engine or on-disk format changes. The measured rollout is a controlled opt-in trial;
see the results and remaining validation limits below.

## Deployment and upgrades

The primary Windows comparison uses Strata's default, non-abliterated Qwen model
family in ISTA-DASLab IQ3_S quantization; see the
[one-hour multi-conversation soak, lifecycle checks and final comparison](../bench/results/2026-10-16-solo-priority-qwen-soak/README.md)
and the [earlier original-model comparison](../bench/results/2026-10-16-solo-priority-qwen-original/README.md).
[SC117 deployment results](../bench/results/2026-10-16-solo-priority-rtx3090/README.md)
are supplementary and are not pooled with those measurements. Model provenance,
quantization, runtime revision and matched settings must accompany contribution
benchmarks. Repeat the original-model comparison after relevant runtime changes.
Start with a controlled opt-in deployment: watch failed streams, short TTFT,
`resume_prefix_missed`, memory and throughput under the actual client workload.
The hour completed 94 tool round trips and 28 exact fact audits, with 301 pauses
and no processed-prefix losses. It also exposed failure paths that were fixed
and retested. Short text TTFT was 0.669 s median and 7.054 s p95, but reached
46.503 s during a partial-cache-hit prefill. One hour is not overnight reliability
or a guarantee that every small request will start immediately.

Keep the change as a Git commit on a branch based on a known upstream release.
Before upgrading, save the current commit and a private copy of the server config;
bring the commit onto the candidate upstream revision in a separate checkout.
Run the protocol tests, the full Python suite and a same-hardware comparison
before switching a live server. Native `STOP`/`DONE`, cache retention, sampling and
parser changes all warrant repeating the continuation checks even when Git merges
cleanly. Do not automatically hard-reset a deployment with local changes.

Rollback is a server restart with `solo_priority` removed or false. The unmodified
native engine and original model/cache formats still work. Returning to the known
upstream commit also removes the Python feature. Save configs outside Git: they can
contain API keys. Client-side aliases/priority forwarding are a separate integration
and must not be required to start a normal Strata server.
