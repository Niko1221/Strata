# Single-engine preemption: RTX 3090 / Windows, 2026-10-16

## Scope

Compare the unchanged Python server at `296b59d2f73c8f04a805df29c2fa97655adace78`
(v0.1.44.1) with `solo_priority: true`. The same official native engine, model and
settings are used. This is a latency/continuation experiment, not a model-quality
benchmark or a claim that every GPU/backend is production-validated.

Hardware: RTX 3090 24 GiB, Ryzen 9 5900X, 128 GiB installed RAM, Windows 11.
PCIe observed at Gen4 x16, GPU power limit 350 W. One sampled active reading was
79 °C, 345.5 W; this is not a measured thermal peak. The PC was dedicated to these
requests. Storage model and model repository revision were not recorded.

Model: `SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF`, two-part IQ3_S;
existing local pack, expert profile and MTP vocabulary. No model was modified.

Relevant launch settings (paths omitted):

```text
--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5
--mtp <existing vocabulary> --kv int8 --kv-resident 32768 --vision
--vram-reserve-mib 1536 --max-context 262144 --ple-io direct --adapt-async 1
--conversation-cache-mib 32768 --conversation-cache-slots 8
--conversation-cache-min-free-mib 16384
```

Effective engine settings: spec 6, mtp_max 4, lookup 3, 11 pool workers,
pcie_frac 0.55, expert cache 14,230 MiB / 7,329 slots, arena 47,962 MiB,
experimental speed projection off. No parallel/batch slots.

## Warm 128K coding request + short interactive request

The archive has 20 facts: exact nonce retrieval, revision precedence, joins,
integer sums, code-contract answers, citations and two deliberately absent values.
An initial audit warms the cache. The long follow-up emits that audit and then a
large Python module, capped at 3,072 generated tokens. Seven seconds after starting
it, another client requests a 21-token JSON answer with a 128-token output cap.
The cap intentionally cuts off the generated module; no claim is made that this
synthetic module is runnable. It keeps decode busy for a matched interval.

Three pairs per arm; client wall time includes queueing, cache operations and the
interruption. Aggregate throughput means **both outputs / pair wall time**, not
native decode speed. Results below use the final scheduler source, with hashes in
`results.json`. Baseline and final runs were on the same day, in that order, not an
ABBA sweep; small throughput changes can include thermal/workload variance.

| Median | Baseline | Final patch |
|---|---:|---:|
| Short time to first content | 19.562 s | 0.560 s |
| Pair completion time | 26.754 s | 27.760 s |
| Aggregate output throughput | 115.608 tok/s | 111.418 tok/s |

Short TTFT range with the final patch: **0.552–0.910 s**. Pair throughput is about
**3.6% lower**, pair duration about **3.8% longer**, in these three pairs. All three
long audits were 20/20 and all three short JSON answers were 3/3 in each arm.
Loading time is excluded. Timings were measured on the PC loopback API.

## Continuation and failure checks

- Final source: one 8,192-token long response with three small requests submitted
  seven seconds apart. Small TTFTs **0.686 / 0.399 / 0.398 s**; long completed in
  **69.688 s**. Its initial audit and a separate post-resume audit each scored 20/20.
- Final source: disconnect a parked client while a higher-priority cold 128K
  request runs. The other request completed, 20/20, and the next health request
  succeeded. No STOP was sent to that other request.
- Eight final-source preemptions: `resume_prompt_read=7`,
  `resume_prefix_missed=0`, no remaining waiters or parked requests. The seven
  tokens read are continuation boundary work, not full-prefix rereads.
- Greedy tool stream in the initial paired experiment: one call, all 450 ordered
  item strings and the boolean `false` correct in each arm. Short TTFT
  **19.125 → 0.142 s**, pair wall **26.318 → 26.628 s** (one pair each).
- **Known failure, retained:** temperature 0.6 / top_p 0.9 / seed 42, final source
  and an additional unchanged-server control. Both produced exactly the same
  tool arguments: 450 correct item strings but `ready: "False"` as a string,
  violating the requested boolean schema. Parsing produced one call in both
  arms. This is not a successful schema-correctness test; preemption did not
  introduce the failure. No tool was executed.

## Cold prefill tradeoff (initial prototype, one pair only)

A separate uncached ~128K prompt, short request arriving at +2 s:
short TTFT **52.563 → 4.507 s**; pair wall **54.715 → 60.683 s** (+10.9%).
Both 20-field audits and three-field short answers passed. Interrupted cold work
can need rereading, so this path has a larger cost and no fixed latency bound.
The final scheduler prevents repeated interruption of a resumed read until it
produces a token. Do not extrapolate this one prototype measurement to all cold
loads, to 256K prompts, or to the final policy under continuous arrivals.

The prototype allowed two pauses per API call. A subsequent multi-arrival test
caught a third short request waiting 48.9 s. The final policy removes that cap,
grants each resume at least five seconds, and ages waiters by queued time. Initial
prototype results are preserved separately from final-source results.

## Reproduce the warm comparison

The generator contains only synthetic data. Its archive text hash matches the
measured fixture. Use the same model/tokenizer and settings. The client reads its
credential from `STRATA_API_KEY`; never include it in a published command or file.

```sh
python bench/results/2026-10-16-solo-priority-rtx3090/make_fixture.py /tmp/solo-fixture.json
# Against the unchanged server:
python tools/bench_solo_priority.py --base http://127.0.0.1:8080 --fixture /tmp/solo-fixture.json --rounds 3 --out /tmp/baseline.json
# Restart the patched server with solo_priority=true, same engine/config:
python tools/bench_solo_priority.py --base http://127.0.0.1:8080 --fixture /tmp/solo-fixture.json --rounds 3 --out /tmp/priority.json
```

The client warms an archive fixture before measuring, ignores empty role/keepalive
headers for TTFT and keeps complete outputs and per-pair measurements. Tool-output
headers are not model tokens. Do not compare the initial tool-name-header timing
in the legacy captured results to first text-token timings.

Protocol/streaming tests run through the real Python pipe reader with a fake
engine; real-engine validation used the Windows release. CUDA/HIP/SYCL source was
not changed or rebuilt. AMD, Intel, vision preemption, cache exhaustion at 256K and
an overnight multi-client soak were not tested. The feature remains opt-in.

## Automated validation

- macOS, Python 3.11.7: `python -m unittest discover -s serve -p 'test_*.py'`
  — **796 tests in 289.866 s, OK (8 skipped)**.
- Windows real deployment environment: scheduler's **11 tests passed in 10.550 s**.
- The three-interruption test also passed eight repeated runs after synchronizing
  on actual resumption. A prior full-suite run exposed a test race: submitting the
  next short job before the long job resumed correctly needed fewer pauses than
  the test expected. The assertion now observes an actual resume before enqueueing.
- Default protocol comparison: after removing only the opt-in pause callback and
  its local flag, `_generate_single`'s AST equals the original single-engine loop.
  The existing protocol tests passed; the flag defaults off and the batch branch
  is unchanged. This is a source/control-path comparison, not a GPU bitwise test.
