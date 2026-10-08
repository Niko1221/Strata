# Adaptive background control: bounded Windows acceptance

This packet describes measurements from **2026-10-08** on one consumer laptop,
using an experimental branch based on **Strata v0.1.40.3**. It is a correctness
and resource-control report, not a general speedup claim. Production settings
were not changed by these isolated campaigns.

The successful gates were five targeted native tests, the Python suite, four
bounded generated-code tasks with real overlapping CPU work, active text-request
parking/resume and cancellation, and ordinary HTTP vision/tool-call handling.
An actual Hermes compiler-feedback session also completed after an explicit
continuation of its earlier turn-limit failure; both attempts' costs are retained.
**The pacing comparison did not make the foreground job faster.** Different
generated answer lengths prevent attributing shorter model completion times to
pacing. A previous output-limit failure is retained below.

## Hardware, software and identity

| Item | Tested configuration |
|---|---|
| CPU | AMD Ryzen 9 7940HS, 8 cores / 16 threads; AVX-512-capable |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU, nominal 8 GiB; native CUDA reported 8,187 MiB total; compute capability 8.9 |
| Host RAM | 64 GiB installed DDR5-5600, dual channel; Windows exposed about 59.72 GiB after hardware reservations |
| Storage | Model files on a dedicated 4 TB BIWIN X570 NVMe drive, operating through the laptop's PCIe 4.0 connection |
| OS | Windows x64, build 26300.9457, reported display version 26H2 |
| Model | [ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO IQ3_S GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), two shards, PLE enabled; not Swift |
| Native build | Release, CUDA architecture 89; MSVC 19.51.36260.0, CUDA compiler 13.4.59 |
| Foreground fixture | Original C++20 code in `foreground_work.cpp`, GCC `-O2 -pthread -static`; no AVX-512-only flags |
| Native binary SHA-256 | `b17c3ef7a55ff1b00e217fb5a5251a23583734b2ce6e2d57d1423012215d83aa` |

No power-plan, overclocking, driver, or external-application priority changes
were part of these campaigns. CPU capability is not evidence that every kernel
or the scalar foreground fixture executed AVX-512 instructions. These are not
large-page-allocation, bandwidth-ceiling, or PCIe-saturation measurements.

`results.json` includes the frozen Python source hashes used by parking and HTTP
tests. The pinned binary and supplied fixture identity distinguish these runs
from older branch revisions. Model filenames and settings identify the tested
variant; complete model, MTP and expert-profile content hashes are not supplied
in this packet, so it is not a fully self-contained model artifact bundle.

## Settings and fixtures

All model campaigns used a 65,536-token configured context, high reasoning,
IQ3_S, native MTP, int8 KV, PLE and the same configured experimental projection
vector: scale 1.0, project mode, per-layer direction, layers 4–44. This vector
was held constant; its quality/performance effect was not evaluated here.

| Setting | Requested | Effective native INFO where different |
|---|---:|---:|
| Resident expert RAM budget | 32 GiB | 32,767 MiB |
| Expert-cache sizing argument | 264 | 338 variable-size entries / 672 MiB |
| KV resident cells | 16,384 | 20,480 |
| Speculation argument | 4 | Window up to 6; MTP maximum 4 |
| Minimum draft probability | 0.5 | 0.50 |
| Expert pool workers | 7 | 8 participants including host thread |
| PCIe expert fraction | 0.37 | 0.37 |
| Prefill | auto | Runtime-selected chunks |
| Startup VRAM reserve | 448 MiB | Startup free 474 MiB |
| Conversation cache | 2,048 MiB / 2 slots | Minimum host free 3,072 MiB |

Other common options were `--mmap-experts`, `--live-memory` and configured
vision support. `STRATA_PREFILL_CPU_SHARE=auto`, `STRATA_STAGER_THREADS=8` and
`STRATA_STAGER_RING=128` were held constant. The optional vision encoder used
CPU execution with eight threads and the BF16 projector. No image was encoded
in the code-generation or parking cases.

The isolated launch guard required at least 42 GiB available before loading.
Campaign observers guarded 3 GiB available RAM and 250 MiB GPU headroom. These
are distinct from the policy's preferred 4 GiB RAM / 320 MiB GPU headroom and
the native startup fitting reserve. Sampled minima are not continuous guarantees.

[`request-fixtures.json`](request-fixtures.json) provides prompts, sampling,
expected answers and the native prompt wrapper. [`runtime-settings.json`](runtime-settings.json)
lists settings using explicit asset placeholders instead of local paths. These
files are fixtures, not an automatically runnable machine-specific config.
The native fairness prompt was 94 tokens, SHA-256 of the ASCII Python token-ID
list `a9c4d3648a4ffb5f97e4974ef94c27539da14b635afc0e8dffc73a4102fccda4`.

## Real foreground overlap and pacing

The foreground program ran eight independent workers, **16 billion iterations
per worker**, with a start gate and deterministic checksums. It was launched
as an independent application, not as a descendant that Strata's own-process
exclusion would hide. Each model response implemented `unique_sorted`; the
generated function was compiled and tested on four fixed plus 100 deterministic
random cases. All eight foreground checksums matched between runs.

Both arms kept expert-cache capacities fixed. Only background pacing differed;
this isolates pacing from live cache resizing. Order was off/on/on/off, with
fresh model processes and the same prompt, temperature 0, seed 42 and a
4,096-token output ceiling.

| Arm | Actual concurrent interval | Foreground wall | Model wall | Generated tokens | Decode tok/s | Code checks |
|---|---:|---:|---:|---:|---:|---|
| off 1 | 43.53 s | 45.52 s | 114.27 s | 1,742 | 15.59 | 104/104 |
| on 1 | 43.85 s | 45.84 s | 64.59 s | 785 | 12.64 | 104/104 |
| on 2 | 43.79 s | 45.79 s | 66.60 s | 873 | 13.62 | 104/104 |
| off 2 | 43.00 s | 44.99 s | 126.76 s | 1,927 | 15.58 | 104/104 |

Overlap is the intersection of independently recorded foreground and model
intervals. Foreground median time was **45.25 s off versus 45.81 s on**: no
foreground benefit was demonstrated. Model median was 120.51 s versus 65.60 s,
but on-arm answers were substantially shorter and decode throughput was lower.
**Do not call this a 46% model speedup.** Two runs per arm are insufficient to
estimate a reliable performance distribution.

Every sampled point during actual overlap had a complete external CPU sample:
43/43, 43/43, 43/43 and 42/42 respectively. Both on arms recorded positive
controller delays at all 43 overlap observations, up to 10 and 14 ms. Neither
off arm recorded a positive delay. No send error was recorded. These records
show controller decisions through its send path, not a trace of every native
sleep or its duration. Final telemetry has 45 positive observations per on arm;
the pre-shutdown summary had 44. Repeated observations are not unique commands.

Available RAM minima were 10.830, 10.989, 11.011 and **10.658 GiB**. Fresh native
CUDA-free minima were 326, 326, **324** and 326 MiB; DXGI budget-minus-usage
minima matched. Raw CUDA capacity and DXGI budget remain distinct measurements.
RAM was sampled about once per second. Native capacity ages were at most 2.63 s
and accepted only when at most 5 s old. Shorter unobserved excursions cannot be
excluded. This campaign performed no MEMORY cache resizing.

### Earlier failed gate is retained

Before the bulk Windows process sampler, bounded per-process scanning repeatedly
became incomplete under the eight-worker load. The four earlier arms had only
1/1/1/0 attributed external CPU observations and 0/1/1/0 positive-delay
observations. That was not a valid sustained pacing comparison. One on-arm
answer also reached 4,096 tokens with `finish=length` and no final answer; only
three of those four tasks passed. These failures are included in `results.json`
and are not folded into the final successful-only timing table.

The replacement bounded Windows x64 snapshot saw all 315 processes in a
separate host probe. Ten snapshots had median **6.603 ms**, range 5.635–7.043 ms.
PID/PPID matched psutil; creation-time difference was below 1 microsecond, RSS
difference 8 KiB, and user-CPU difference one 15.625 ms accounting tick between
sequential reads. The integrated sampler deliberately returned an incomplete
initial baseline, then complete samples in 6.7–8.1 ms. This supports better
measurement coverage; it is not a measured model throughput gain.

## Active text-request parking

Three real-model cases ran through `Service.run`: baseline, one park/resume,
and cancellation while suspended. The trigger and recovery hold were **synthetic
policy decisions**. Physical RAM/GPU capacity readings stayed real; no attempt
was made to exhaust memory. STOP/drain, journal creation, engine unload, admission,
reload, prefix re-prefill and resumed generation were real.

| Case | Request wall | Accepted output IDs | Result |
|---|---:|---:|---|
| Baseline | 64.781 s | 1,094 | `stop`; compiled; 104 checks; Unicode marker once |
| Park/resume | 116.098 s | 740 | `stop`; compiled; 104 checks; Unicode marker once |
| Cancel while suspended | 28.861 s | 49 | `cancel`; no replacement generation |

The resume case verified the next segment's exact input prefix, retained seed,
remaining output budget and new process identity. Future output was **not**
bit-identical to baseline. The two answer lengths differ, so subtracting their
wall times does not isolate parking overhead. The recovery hold itself was
intentional and contributes to elapsed time.

During resume parking, Windows available RAM increased by 37.348 GiB and the
global GPU-free view by 6,802.7 MiB after the old process exited. The analogous
cancel case released 37.442 GiB and 6,836.7 MiB. These are measured whole-process
unload effects, not evidence that all tensors migrated into RAM or SSD.
Suspended requests emitted heartbeats. Cancellation completed about 37 ms after
the cancel flag in its one observed trial. Sampled minimum RAM across parking
cases was 10.591 GiB; native CUDA and DXGI headroom minima were 326 MiB.

## HTTP compatibility checks

On the same binary and frozen Python, ordinary HTTP vision identified the
provided synthetic red-square PNG in 12.972 s. Parking correctly reported the
image path unsupported; normal vision still worked. A two-turn tool cycle
called `add` with 17 and 25, executed the supplied addition, then returned `42`
in 13.566 s total. This was actual tool-result continuation, not a fabricated
assistant/tool transcript. Sampled RAM stayed above 9.465 GiB and native free
VRAM above 340 MiB. Neither check is a broad visual or agent benchmark.

## Near-64K active-prefill pressure and reuse

The final campaign used the same **60,270-token** distributed-key prompt as the
earlier long-context acceptance run. It introduced 2 GiB of real, touched RAM
after 256 tokens had been processed. A fixed preferred RAM target of 10.737 GiB
was selected before allocation so the controller would release resources while
the laptop still had safe physical headroom. This is actual allocation pressure
against a deliberately elevated target, not a physical-exhaustion test.

Resident expert RAM fell from **32,767 to 30,404 MiB in 7.156 s**, by prompt
position **768/60,270**. The model process stayed the same. All four distributed
keys were answered correctly after the resize. The identical repeat, issued
while the 2 GiB allocation was still held, also returned all four keys correctly.

| Request | Total wall | Prompt processing | Reused / newly read tokens | Generated tokens |
|---|---:|---:|---:|---:|
| Cold, with live relief | 839.035 s | 829.525 s | 0 / 60,270 | 140 |
| Immediate repeat, pressure still held | 10.940 s | 0.586 s | 60,265 / 5 | 140 |

After releasing pressure, resident RAM recovered to 32,707 MiB, within the
64 MiB recovery tolerance. A separate conversation then returned its own new
key without carrying over the earlier keys. Across 982 observations, available
RAM stayed at least **8.805 GiB**, and fresh native CUDA/DXGI headroom at least
**294 MiB**. The separate observer CUDA context reported much more free memory;
under WDDM that view is not interchangeable with the native process's capacity.

The cold run was slower than an older 663.17 s campaign with no mid-prompt
relief and different resident behavior. This is **responsiveness, correctness
and cache-reuse evidence**, not a matched before/after speed improvement. The
native log recorded two cooperative prompt returns, at 768 and 4,608 tokens,
with 3,022.5 ms and 100.4 ms control pauses. It did not return/rebind its loan
at every 256-token chunk. Source inspection likewise shows ordinary CAPACITY
reporting and non-waiting BACKGROUND renewal do not request loan turnover.
The roughly 3.123 s of logged pauses cannot by themselves explain the older
run's timing difference; the workloads must be matched before attributing it.

[`near64k_fixture.py`](near64k_fixture.py) reproduces the synthetic reference
and exact native prompt through a caller-supplied tokenizer. It does not start
an engine or induce pressure. The prompt-ID checksum is
`5ec64b0c7e58d274b0b652888cea66fcff15608b1d4f30a611ad4f1ff35a8399`.

## Actual Hermes compiler-feedback continuation

A separate real Hermes session exercised code, terminal/compiler feedback and
verification against the running Strata server. Its first bounded attempt is
retained as a **failure**: after an initial compilation failure, a subsequent
build reached tests and failed. The agent repaired the candidate, which then
passed the independent 104-case checker, but it exhausted its six-turn limit
before obtaining its own visible passing build and final response. That attempt
exited with code 1 after 311.219 s and eight recorded API requests. Independent
checker success alone was not accepted as successful agent completion.

The same session and full transcript were then resumed with an isolated
configuration allowing twelve turns. The continuation supplied no answer or
code repair assistance. It made no further candidate edit: it completed the
missing build/verification work and final response. The visible compiler tool
used **four concurrent compiler jobs**, returned exit 0 and reported **2,096
deterministic cases passed**. A separate 104-case check also passed. The original
project test files, prior evidence, and production configuration were unchanged.

| Interval | Agent wall | Recorded API requests | Outcome |
|---|---:|---:|---|
| Initial six-turn attempt | 311.219 s | 8 | Failed; no agent-visible final passing build |
| Same-session continuation | 203.183 s | 3 | Passed; agent exit 0, visible build and final answer |
| Combined active agent intervals | **514.401 s** | **11** | Completed after continuation |

These are sums of active agent intervals, not intervening time or all model
loading. API requests are not the same as Hermes turns. The effective turn cap
was set in the isolated home's configuration; this is a test-budget change,
not a Strata speed fix or a claim about every Hermes CLI entry point.

The passing compiler tool took 2.102 s wall time, including 1.877 s compilation;
GCC 16.2.0 used C++20, `-O2 -g0 -march=znver4 -Wall -Wextra`. Three tool-time
samples saw the server loaded and idle, with a measured **0.03125 CPU-seconds**
spent by the native engine. This is consistent with sleeping while the agent
waits for its sequential compiler tool. There was no matched fairness-off
Hermes run, so the observation does not establish an additional CPU-saving or
completion-time benefit. No artificial pressure was injected in this campaign.

Sampled available RAM stayed at least **8.352 GiB** and native CUDA/DXGI
headroom at least **276 MiB**. This validates one bounded resumed agent workflow;
it does not demonstrate uninterrupted autonomy, general successful completion,
or a speedup over ordinary Strata/Hermes. `results.json` preserves the prior
failure, unchanged continuation candidate, API counts and combined elapsed cost.

## Reproducing the code-level gates

From a configured CUDA build of this branch, using the project's documented
compiler/toolkit environment, the native commands used these five targets:

```text
cmake --build build --parallel 6 --target strata live_memory_test background_control_test pool_background_test request_stop_test platform_memory_test
ctest --test-dir build -R "^(live_memory_test|background_control_test|pool_background_test|request_stop_test|platform_memory_test)$" --output-on-failure
```

All five passed on the tested Windows build. `build` is a generic example
directory; use your existing configured directory. The CUDA tests require the
normal Strata CUDA development environment. This packet does not claim that a
new Linux/HIP build or an arbitrary toolchain was validated.

With the project's Python dependencies installed, focused no-model regression
modules are runnable from repository root:

```text
python -m unittest serve.test_memory_policy serve.test_live_memory serve.test_coadaptive serve.test_coadaptive_reserve serve.test_resource_presets serve.test_windows_process_snapshot serve.test_routing_costs serve.test_request_parking serve.test_request_parking_service serve.test_service_resources serve.test_memory_policy_http
python -m unittest discover -s serve -p "test_*.py"
```

The recorded full suite ran **851 tests in 226.481 s: 844 passed, 7 skipped**.
Some tests use fakes; synthetic token-rate output from unit tests is not a model
speed measurement. Skipped tests are not counted as passes. Preserve platform
and environment requirements rather than removing skips to inflate coverage.

To repeat the foreground fixture, compile `foreground_work.cpp` with C++20,
`-O2 -pthread` (the recorded Windows build also used `-static`). Arguments are
`START_GATE OUTPUT_JSON ITERATIONS_PER_WORKER WORKERS`. The fixture waits up to
60 seconds for the gate file, writes a `.ready` marker, then starts its workers.
It bounds workers to 16 and iterations to 16 billion per worker. Supply fresh
output/gate paths and an external watchdog. Do not mistake a child excluded by
the resource sampler for an independently detected foreground application.

The fixtures and checker here are sanitized components, **not a validated
portable end-to-end campaign runner**. Repeating the full measurement requires
an isolated server configured for local model assets, timestamped sampling,
ownership-aware process cleanup and explicit admission guards. Preserve failed
answers, seed, actual output lengths, overlap, binary/source hashes and cold
versus reused prompt timing. Do not expose a benchmark server publicly.

## Boundaries and remaining validation

- Completed: the named bounded gates above on one Windows laptop and one model.
- The final near-64K real-pressure/reuse campaign passed its bounded gates;
  no matched throughput improvement was established.
- Actual Hermes compiler-feedback continuation passed, with the earlier failed
  six-turn attempt and its cost retained; uninterrupted autonomy was not shown.
- No demonstrated faster foreground application, general agent speedup,
  bit-identical continuation, multi-GPU behavior, arbitrary live worker resizing,
  general tensor migration, disk selection, image-request parking, crash/reconnect
  recovery, or pressure-triggered idle whole-engine unload. Upstream's existing
  timed idle-unload/autoload feature is separate from this controller.
- Safe points are cooperative. An in-flight chunk/kernel can delay reaction;
  another application can allocate faster than sampled control reacts. Finite
  running/reload footprints, device loss, disk errors and client timeouts remain.
  There is no unconditional “never OOM” or “never stop” guarantee.

See [Adaptive Strata scope](../../../docs/ADAPTIVE_STRATA.md) and
[implementation provenance](../../../docs/ADAPTIVE_PROVENANCE.md) for inherited
Strata PR #726 code, original extensions, API references, and the specific
ATSInfer/StarPU design inspiration. The foreground/checker fixtures are original
contributions under the repository's MIT license. No third-party benchmark code
was copied into this packet. Only measured aggregates and synthetic fixtures
are public; local paths, task databases, journals and raw reasoning are omitted.
