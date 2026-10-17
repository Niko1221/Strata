# Community benchmark: two V100-SXM2-16GB + RTX PRO 4000 Blackwell, Windows

Measured **2026-10-08**: mixed **sm_70/sm_120** using the installed **CUDA 12.9 engine 0.1.40**, IQ2_XS, **262144 context and Vision**. Four configurations, 48 fresh-prompt trials. This is a compatibility/performance observation on one workstation, not a general mixed-GPU support claim.

## Hardware and software

- **P:** RTX PRO 4000 Blackwell, nominal 24 GB (**24467 MiB** reported), physical index 0, WDDM desktop GPU, PCIe ×16 / host maximum Gen4 (Gen1 at idle), **145 W** limit. **V1/V2:** V100-SXM2-16GB, indices 1/2, **16384 MiB** each, TCC, PCIe Gen3 ×8, **300 W** each. Six V100 NVLinks are present; layer hand-off uses host RAM. P↔V peer access is unsupported; V1↔V2 supported ([enumeration](topology.json), not a bandwidth test).
- EPYC **7A23, 48C/96T**; nominal 64 GB RAM, **63.86 GiB** Windows-visible; KINGSTON SNV3S1000G NVMe. Desktop, browser/Codex and animated wallpaper remained active; their timing contribution was not isolated.
- Windows 10 Home 22H2 **19045.6466**; driver **582.16**; Python **3.12.10**; CUDA runtime **12.9.79**. Driver-advertised CUDA 13.0 is not the engine runtime.
- Frontend **`1735d6471df29b42c26170efaac1f1446a58640f`**; portable experimental release engine **0.1.40**, sm_70 and PTX included, native sm_120 not listed. Exact release compiler invocation/binary source commit unavailable. [Provenance](provenance.json) and per-arm [manifests](manifests/) retain hashes. No source/driver upgrade was part of this comparison.

## Model and configuration

- `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, **IQ2_XS**, two `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-0000{1,2}-of-00002.gguf` shards. Exact filenames/sizes/receipts are in provenance.
- Vision: same-revision `mmproj-Qwen3.8-Flash-Next-BF16.gguf`, GPU enabled, **1024 image tokens**, 16 threads. Setup-built **Q2_0 MTP** from `Qwen/Qwen3.8-Flash-Next` revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`, **106299-token CJK** draft vocabulary; profile/pack hashes included.
- **262144 context, INT8 KV, 32768 resident**; prefill auto→**8192**, ring **96**, CPU **47 workers + host**, concurrency **1**, low-RAM off. Profile-ranked cache, **no eviction**, arena pin limit **8 GiB**. MTP **4/min-p 0.5**, suffix draft default **3**, lookup-chain off. No calibration override or experimental speed projection; auto split's planning estimates are not measurements.

| Arm | Language order / resolved layers | Vision | Reserve first/later MiB | Actual cache slots |
|---|---|---|---|---:|
| A: dual-reference | V1,V2 / 23,25 | V1 | 700/default | 14876 |
| B: tri-first | P,V1,V2 / 24,11,13 | V1 | 3072/700 | 24310 |
| C: tri-last | V1,V2,P / 11,14,23 | V1 | 700/2304 | 24159 |
| D: dual-vision-pro | V1,V2 / 24,24 | P | 700/default | 15960 |

The last language stage holds head/MTP. C's later reserve applies to **both V2 and P**; this is not an order-only comparison. Moving Vision in D also changes auto split/cache, so D does not isolate encoder speed. [Configs](configs/) preserve requested settings; [resolved.json](resolved.json) records actual allocations.

```powershell
.\.venv\Scripts\python.exe -u -m serve.server --engine strata --config <MATERIALIZED_CONFIG> --host 127.0.0.1 --port 8081
```

Replace path/UUID placeholders as described in [REPRODUCE.md](REPRODUCE.md); keep GPU order and numerical settings. Run only one server at a time.

## Method

Arms ran serially A→B→C→D, restarting between arms. Each excluded one approximately 3400-input/256-output warm-up, then three runs at each ascending size. The [harness](scripts/benchmark.py) generates synthetic Python functions and an early per-trial nonce; measured request hashes match across all four arms. **Zero prefix reuse**, warm expert residency retained. Sampling **temperature 0**, reasoning **none**, **256-token** output cap; every measured output reached 256.

Prefill/decode use separate engine durations. Streaming TTFT runs from client send to first nonempty answer delta, ignoring keep-alives; total client latency includes frontend/prefill/decode, excludes loading. Speed prompts contain no images, with Vision loaded. [results.csv](results.csv) has every duration/rate and sampled resource peak; [summary.csv](summary.csv) has medians/ranges including total latency; [run-details.json](run-details.json) preserves outputs, warm-ups and counters. [Logs](logs/) retain startup and all speed timing blocks. RAM/GPU readings are approximately one-second whole-system/device samples, not instantaneous or model-exclusive peaks.

## Results

Median [minimum–maximum]. All **48/48** speed trials succeeded; no cancelled/failed speed requests or OOM observed. Earlier one-run screening is excluded. This fresh A baseline is independent of the earlier dual-V100 submission.

| Arm | Actual prompt tokens | Reused | Generated | Runs | Prefill tok/s | Decode tok/s | TTFT seconds |
|---|---:|---:|---:|---:|---|---|---|
| A | 4096 | 0 | 256 | 3 | 881.1 [879.2–884.6] | 85.8 [70.9–95.6] | 4.70 [4.67–4.71] |
| A | 32768 | 0 | 256 | 3 | 1803.3 [1799.0–1804.5] | 87.4 [87.0–93.0] | 18.29 [18.26–18.33] |
| A | 128000 | 0 | 256 | 3 | 1966.3 [1963.2–1973.6] | 86.8 [79.7–87.8] | 65.45 [65.19–65.55] |
| A | 258000 | 0 | 256 | 3 | 1630.8 [1615.9–1691.1] | 82.2 [78.7–83.6] | 158.86 [153.23–160.31] |
| B | 4096 | 0 | 256 | 3 | 1074.6 [1071.9–1086.7] | 109.5 [105.3–115.6] | 3.85 [3.80–3.87] |
| B | 32768 | 0 | 256 | 3 | 2612.1 [2611.7–2624.4] | 108.6 [106.8–113.5] | 12.65 [12.60–12.66] |
| B | 128000 | 0 | 256 | 3 | 3372.4 [3361.4–3393.2] | 100.0 [93.8–102.9] | 38.29 [38.10–38.43] |
| B | 258000 | 0 | 256 | 3 | 2980.5 [2952.7–3028.5] | 95.9 [94.8–97.3] | 87.24 [85.84–88.03] |
| C | 4096 | 0 | 256 | 3 | 1132.2 [1124.0–1138.0] | 109.3 [108.5–117.3] | 3.66 [3.65–3.69] |
| C | 32768 | 0 | 256 | 3 | 2706.5 [2686.2–2708.1] | 104.7 [104.2–110.5] | 12.22 [12.21–12.31] |
| C | 128000 | 0 | 256 | 3 | 3483.5 [3455.3–3524.4] | 100.7 [100.7–112.5] | 37.09 [36.69–37.39] |
| C | 258000 | 0 | 256 | 3 | 2956.6 [2920.5–3010.4] | 97.0 [95.8–98.7] | 87.91 [86.37–89.01] |
| D | 4096 | 0 | 256 | 3 | 908.3 [905.2–909.1] | 94.6 [92.6–101.8] | 4.56 [4.55–4.57] |
| D | 32768 | 0 | 256 | 3 | 1887.4 [1882.1–1889.4] | 94.8 [92.8–100.8] | 17.46 [17.46–17.51] |
| D | 128000 | 0 | 256 | 3 | 2064.7 [2036.2–2096.3] | 86.9 [85.8–87.3] | 62.33 [61.42–63.21] |
| D | 258000 | 0 | 256 | 3 | 1752.0 [1750.1–1760.6] | 79.1 [76.1–81.2] | 147.92 [147.20–148.07] |

B reduced 258000-token median TTFT **158.86→87.24 s**, with decode **82.2→95.9 tok/s**. B/C long-input performance was similar; B left more desktop VRAM. These are workload-specific observations, not predicted speedups.

Whole-arm observations, including startup and correctness; only B additionally includes needles. GPU columns are **P / V1 / V2**; peaks are not necessarily simultaneous.

| Arm | RAM peak / minimum available GiB | VRAM peak MiB | Temperature peak °C |
|---|---|---|---|
| A | 52.64 / 11.22 | 915 / 15874 / 15884 | 45 / 63 / 77 |
| B | 54.95 / 8.91 | 21294 / 13259 / 14834 | 87 / 63 / 78 |
| C | 54.89 / 8.97 | 22233 / 13030 / 13841 | 85 / 62 / 77 |
| D | 52.92 / 10.94 | 2217 / 15874 / 15904 | 58 / 65 / 78 |

B's minimum free P VRAM was **2778 MiB**, C's **1839 MiB**. Software thermal slowdown was observed on V2; power-cap events also occurred. [Resolved data](resolved.json) separates text-window and whole-arm samples, clocks/power/throttle counts. No power/clock changes were made. Per-request paging was not measured; Windows pagefile use was 167 MiB before / 131 MiB after, not evidence of zero paging during requests.

## Correctness and limitations

- All arms: health, OpenAI and Anthropic API smoke checks; **9/9** tool/image/queue checks each (two tool roundtrips, three single-card images, three four-card batches, two-client serial queue). Expected/actual answers and timings: [correctness.json](correctness.json); [card/check generator](scripts/api_checks.py). This is not concurrent-throughput scaling.
- B: unmodified official needle script, **32k/128k/250k × 10/50/90%**, **9/9 found**, no misses/errors/skips. Actual lengths **32308–32309 / 125703–125705 / 243257–243258**; all answers in [needles.json](needles.json).
- One Claude Code repair per arm: **A 28/29, B 29/29, C 29/29, D 27/29**. A/D reject empty strings; D also rejects padded ` 007 `. Original task wording (“whitespace-only”, “stripped”) leaves boundary ambiguity; failures and [repaired fixtures](coding/) are retained. No claim of GPU-caused quality changes or broad Agent superiority.

Limits: one prompt family, three repetitions, fixed arm order, short outputs, background activity and thermal effects. CUDA 12 mixed-architecture success is limited to this binary/hardware. No 24-hour soak, large-image-cache stress or private translation replay. API draft counters combine MTP/suffix; CSV also provides **log-derived MTP-only** counters by subtraction, documented in REPRODUCE. Full SSE/telemetry and private CLI transcripts remain local; model files/packs and operational launchers are omitted.
