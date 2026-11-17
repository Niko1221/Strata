# Community measurements on RTX 3070 8 GiB

Measured locally on 16–17 November 2026; prepared by TJ-WANG-OK from saved records.
Flash-Next Q2_0 short-prompt decode medians were **45.00 tok/s before calibration and
44.25 tok/s after restoring the same settings**. No calibration candidate met the
3% retention threshold. Supplementary Qwen3.6 Q4 deployment checks show that it
can answer short requests on this card with a reduced-memory configuration.

These are custom historical measurements under the community guide, not a run of
the draft standard protocol, an approved result, or a controlled comparison of
the two models. No new inference was performed while preparing the submission.

## Hardware and software

- One NVIDIA GeForce RTX 3070, 8192 MiB VRAM; Intel Core i9-10850K,
  10 physical cores / 20 logical processors, AVX2; 64 GB RAM reported by calibration.
- Windows; exact edition/build was not recorded with the measurements. NVIDIA
  driver 610.60 from the saved deployment inventory. CUDA 13.0 release build.
- Control/Terminal bundle 0.1.61; engine 0.1.60, portable Windows CUDA binary.
  Engine BUILD.json identifies commit `09b603886925`; architectures 75/86/89/120,
  PTX enabled. Installed Terminal COMMIT collected during preparation:
  `c77572806996349b49995e4f98d41b5294d36a21` (not independently captured at run time).
- Storage model, RAM speed, PCIe negotiated link, power limit and background
  workloads during measurement: not recorded. A startup probe is in the log;
  it is not a PCIe link-width or storage benchmark. No peak-memory sampling.
- See [provenance.json](provenance.json) for sources and collection limits.

## Model and configuration

### Flash-Next Q2_0 (main measurement, 16 November)

ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO Q2_0, downloaded through ModelScope.
Exact model repository revision was not saved. Native files:
`Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` and
`Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf`.
32,768-token configured context, one slot, vision off, int8 KV, automatic expert
cache, prefill 512, VRAM reserve 1600 MiB. DFlash2 is selected by the launcher:
effective flags include `--draft dflash2 --spec 6 --draft-conf 0.8
--draft-fallback none`, Q4_0 selector pack and draft vocabulary. The saved input
configuration contains MTP flags; the effective startup command in
[flash-engine-excerpt.txt](flash-engine-excerpt.txt) records their replacement.
The 9-worker pool is the automatic default. No experimental speed projection is
present in the retained command. Calibration retained no overrides.

Saved launch invocation (replace placeholders with local paths):

```powershell
strata launch --family qwen --model q2_0 --context 32k --parallel 1 --vision off --access local --data-dir "<DATA>" --set vram-reserve-mib=1600 --set prefill=512 --save-preset Qwen38-Q2-32K --restart --yes --no-tui
```

### Qwen3.6 Q4 (supplementary deployment checks, 17 November)

Unsloth Qwen3.6-35B-A3B UD-Q4_K_XL, native
`Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, configured context 8192, one slot, vision off.
Exact repository revision was not recorded. Effective engine arguments:

```text
strata.exe --serve --native <DATA>/models/qwen36-UD-Q4_K_XL/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --max-context 8192 --mtp 2 --prefill 128
STRATA_Q8_PACKED=0
CUDA_MODULE_LOADING=LAZY
```

The log reports 16/256 experts per layer in VRAM (1.17 GiB), 18.75 GiB pinned host
memory, 10 CPU threads and 1.10 GiB staging. These are startup allocations, not
measured peaks. Although MTP was configured, all three saved responses report
`draft_n=0`; they provide no evidence of an MTP speedup. KV format was not
explicitly set in the saved Qwen3.6 command.

[config.json](config.json) preserves sanitized launch inputs for both models.

## Method

The main API check used [benchmark.py](benchmark.py): three fixed English prompts
(merge sorted lists, refrigerator explanation, European capitals), two sequential
rounds per phase, `max_tokens=256`, temperature 0, seed 42, thinking disabled,
non-streaming HTTP on loopback. All twelve outputs hit the 256-token cap; this
does not establish task completion. Exact prompts, output text, usage and timing
fields are preserved in [flash-baseline.json](flash-baseline.json) and
[flash-restored.json](flash-restored.json). The script's settings summary omits
seed; the request source records 42.

The existing service was used before calibration, then calibration restarted
engines during its scans, and the service was restored for the second phase.
There was no explicit cold-cache reset or controlled warm-up in the API script.
Repeated prompts can reuse prefixes; the table reports the saved cache counts.
Expert-cache state and background activity were not controlled. Loading is
excluded from API elapsed time. Throughput comes from the engine's separate
prompt/decode timings. Elapsed seconds are whole HTTP request times, not TTFT.
Only two repetitions per prompt were saved, below the guide's preferred three.

The automatic calibration used the installed tool's `legacy-decode-v1` suite,
took 908 s, and retained `{}`. Its complete measurements are in
[calibration.json](calibration.json), console output in
[calibration.log](calibration.log). The exact top-level shell command was not
saved; no reconstructed command is presented as having been executed.

Qwen3.6 used [verify-qwen36.py](verify-qwen36.py): three distinct short Chinese
prompts, one request each, `max_tokens=384`, temperature 0, thinking off,
non-streaming, no seed. See [qwen36-verification.json](qwen36-verification.json).
No distribution or median over repeated Qwen3.6 workloads is claimed.

## Results

Flash-Next aggregate medians and min–max ranges combine three different prompts;
they are descriptive summaries, not six repetitions of the same workload.

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Before calibration | 30–43 | 0–36 | 256 each | 6 (2 per prompt) | 36.55 (28.70–54.40) | 45.00 (43.30–50.90) | not measured |
| After restoring unchanged settings | 30–43 | 0–36 | 256 each | 6 (2 per prompt) | 44.55 (32.30–58.40) | 44.25 (41.50–51.20) | not measured |

Whole-request elapsed seconds, median (min–max): before 6.175 (5.405–6.577);
restored 6.267 (5.332–6.492). Generation-time-weighted decode: 46.04 and
45.50 tok/s, respectively. Both phases used the same retained parameters;
the difference is not a tuning gain or a controlled regression finding.

Calibration's confirmation stage compared default PCIe share / draft floor
0.15 / 0.5 against candidate 0.35 / 0.7, three runs each: medians 41.96 and
42.88 tok/s (about 2.19%). This was below the 3% retention threshold. The worker
and expert-adaptation sweeps are retained in the JSON, including slower results;
they are not recommendations to change upstream defaults.

Qwen3.6 one-off deployment checks:

| Prompt | Prompt / reused tokens | Output tokens | Prompt tok/s | Decode tok/s | Whole request s | Runs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chinese self-introduction | 18 / 0 | 141 | 38.5 | 30.1 | 5.176 | 1 |
| 17 × 23 | 27 / 0 | 4 | 79.9 | 36.7 | 0.452 | 1 |
| Eight planets, one sentence each | 26 / 0 | 202 | 80.4 | 27.2 | 7.747 | 1 |

The four-token arithmetic answer is too short for a useful throughput comparison.
Do not compare these single checks against the Flash-Next aggregate as a model
ranking: workloads, dates, sampling and cache conditions differ.

## Correctness, failures and limitations

Passed: both saved API suites returned nonempty text. Qwen3.6 health and model
listing succeeded; the arithmetic check contains the expected `391`. Planet and
self-introduction outputs are available for inspection; no general quality score
was computed. The Flash-Next generated code was not executed or tested.

Failed: the saved Qwen3.6 engine log records two 8K startup attempts failing to
allocate hybrid expert buffers: automatic prefill needed 1159 MiB with 314 MiB
free, and prefill 128 needed 1135 MiB with 214 MiB free. The later reduced-memory
configuration started successfully. The log does not record the environment of
each failed attempt, so it does not isolate which setting caused recovery.
See [qwen36-engine-excerpt.txt](qwen36-engine-excerpt.txt). A local deployment note
also mentions a 32K startup failure, but its original detailed log is not included
and it is not treated as a measured configuration here.

Not run: draft standard protocol, 4K/16K or full-context benchmark sweep,
long-context recall, concurrency, soak tests, peak RAM/VRAM sampling, independent
reproduction, matched model comparison, model quality evaluation, or engine builds.
No source code or runtime defaults are changed by this results-only submission.

## Evidence and reproduction

Run `python benchmark.py results.json` only against a running Flash-Next Q2_0
server at `http://127.0.0.1:8080` with the configuration above. It writes new
measurements and does not reproduce the historical cache or background state.
Run `python verify-qwen36.py` against a running Qwen3.6 server for the supplementary
checks; it writes `qwen36-deployment-verification.json` next to the script.
These are reproduction instructions, not additional runs performed for this PR.

`rows[].usage` uses token counts, `rows[].timings` uses milliseconds and tok/s,
and `elapsed_seconds` is client wall time. `round` and `prompt` are one-based.
Calibration JSON uses its original suite schema. Its embedded integrity hashes
refer to the original report: path sanitization invalidates them for applying
this public copy. Do not feed this sanitized report back into calibration apply.
[summary.json](summary.json) is recomputed from the saved numeric evidence.

Local paths are placeholders; GPU UUID and unrelated conversations are omitted.
Original prompt/output text for the included synthetic requests is preserved.
[SHA256SUMS.txt](SHA256SUMS.txt) hashes the submitted sanitized files.
