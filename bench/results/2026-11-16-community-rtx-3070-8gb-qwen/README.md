# Community Flash-Next Q2_0 measurements on RTX 3070 8 GiB

Measured locally on 16 November 2026; prepared by TJ-WANG-OK from saved records.
Flash-Next Q2_0 short-prompt decode medians were **45.00 tok/s before calibration and
44.25 tok/s after restoring the same settings**. No calibration candidate met the
3% retention threshold.

These are custom historical measurements under the community guide, not a run of
the draft standard protocol or an approved result. The historical measurements below are separate from the new standard-protocol run linked below.

## Additional standard-protocol measurements

A separate [17 November draft-standard run](../2026-11-17-standard-rtx-3070-7c6e2e69/README.md) now supplies
three repetitions each for short, 4K and 16K performance prompts, twelve sanity
checks, and the optional four-client stage against the one-slot server. See that
report for status, scores, raw evidence and limitations. The historical results
below are not mixed with those thinking-on protocol measurements.

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

### Flash-Next Q2_0 (16 November)

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

[config.json](config.json) preserves sanitized Flash-Next launch inputs.

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

## Correctness, failures and limitations

Passed: all twelve saved API requests returned nonempty text and reached the
256-token output cap. No general quality score was computed. The generated code
was not executed or tested, so task completion and code correctness are unverified.

Failed: no failed API requests are present in these twelve saved measurements.
Calibration found no candidate that met the 3% retention threshold. The engine
excerpt retains its low-VRAM warning; startup allocations are not measured peaks.

Not run in this historical series: draft standard protocol, 4K/16K or full-context benchmark sweep,
long-context recall, concurrency, soak tests, peak RAM/VRAM sampling, independent
reproduction, model quality evaluation, or engine builds.
No source code or runtime defaults are changed by this results-only submission.

## Evidence and reproduction

Run `python benchmark.py results.json` only against a running Flash-Next Q2_0
server at `http://127.0.0.1:8080` with the configuration above. It writes new
measurements and does not reproduce the historical cache or background state.
These instructions describe the historical custom workload; the new protocol run uses its own frozen fixtures.

`rows[].usage` uses token counts, `rows[].timings` uses milliseconds and tok/s,
and `elapsed_seconds` is client wall time. `round` and `prompt` are one-based.
Calibration JSON uses its original suite schema. Its embedded integrity hashes
refer to the original report: path sanitization invalidates them for applying
this public copy. Do not feed this sanitized report back into calibration apply.
[summary.json](summary.json) is recomputed from the saved numeric evidence.

Local paths are placeholders; GPU UUID and unrelated conversations are omitted.
Original prompt/output text for the included synthetic requests is preserved.
[SHA256SUMS.txt](SHA256SUMS.txt) hashes the submitted sanitized files.
