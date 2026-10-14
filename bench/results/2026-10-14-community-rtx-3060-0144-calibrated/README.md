# RTX 3060 / Windows: Strata 0.1.44.1 with calibrated settings

Measured on 2026-10-14: Qwen3.8-Flash-Next-GSQ-RCO Q2_0, IQ2_XS and IQ3_XXS on one RTX 3060 12 GiB, Core i7-12700 and 64 GB RAM. This is a reduced rerun of the [0.1.43 report](https://github.com/libratechw/Strata/tree/0e97d25d9bf29af6b108f6fdce6f3a0dfb7e1c25/bench/results/2026-10-12-community-rtx-3060-0143/README.md), with the official Windows CUDA 13.0 engine 0.1.44.1 and reused model-specific calibration results. No older engine was run. The linked historical publication retains the wider 0.1.43 checks; they are not fresh 0.1.44.1 observations.

## Hardware and artifacts

- Model storage: NVMe SSD, Transcend TS2TMTE250S; 63.78 GiB installed physical RAM (marketed as 64 GB). No storage-throughput test was run.
- Windows-11-10.0.26300-SP0; Python 3.13.11; GPU/VRAM/driver/power limit/max PCIe link: `NVIDIA GeForce RTX 3060, 12288 MiB, 617.42, 170.00 W, 4, 16`. Background desktop applications and OS file cache were not controlled.
- Official release engine SHA-256 `bd18453f65b4d03b15de6100b1a9226426266560779b51b9265de176ae8f34d4`. The Python runtime checkout is `296b59d2f73c8f04a805df29c2fa97655adace78`; it is recorded separately from the prebuilt engine.
- Model repository/revision and freshly checked shard hashes: [model-provenance.json](model-provenance.json). Packs, tokenizer, Expert profile and shared MTP hashes: [runtime-artifacts.json](runtime-artifacts.json). No models were downloaded or changed for this run.
- The compared runtime-artifact hashes differ only for the engine executable; the original manifests remain in the complete records.
- Shared MTP: 512 Q2_0 routed draft experts and a 106,299-token shortlist. Its identity and hashes are retained in the complete records. Its upstream BF16 revision was not revalidated here.

## Fixed configurations and scope

The official `tools/calibrate.py` results were reused; calibration was not repeated during this suite. They tune short non-thinking, temperature-0 decode. The measurements below use thinking xhigh, temperature 1, top_p 0.95, top_k 20, min_p 0, presence penalty 0 and repetition penalty 1. The official tuner keeps a change only when its median decode rate exceeds the default by more than 3%; its selected outputs are used unchanged here. Selected settings are calibration outputs, not a proof of long-context optimality. The same post-calibration selection policy is used for all three models.

| Model | PCIe fraction | MTP floor | CPU workers argument | Adapt every/swaps/decay |
|---|---:|---:|---|---|
| Q2_0 | 0.00 | 0.70 | default | 1/80/0.97 |
| IQ2_XS | 0.20 | 0.70 | default | default/default/default |
| IQ3_XXS | 0.35 | 0.70 | default | default/default/default |

The default worker count actually loaded was 11. Q2_0 and IQ2_XS use full native Expert arenas; IQ3_XXS uses a 16 GiB resident budget plus OS file cache and buffered reads (`STRATA_RESIDENT_HEADROOM_GIB=1`, `STRATA_UNBUFFERED_LOAD=0`). Comparisons include these placement differences. All use INT8 KV, a 32,768-token VRAM KV window, automatic Expert-cache/prefill sizing and `--spec 4`; logs retain the effective verify-window size. No vision or speed projection was tested.

IQ2_XS uses the official 0.1.44.1 calibration outputs, with worker/adaptation defaults, as shown above.

Each of the six model/context configurations starts a fresh HTTP server on loopback. A short warmup and a 4K warmup each allow 128 output tokens. Speed requests have unique synthetic Python-module prompts, matched seeds and zero reused input tokens, and finish normally with a 512-token output cap including reasoning. There are three repetitions in ascending/descending/ascending input order, with Expert-cache state retained. Loading and warmups are excluded. Throughput uses engine timings; TTFT and whole-request latency use the streaming client. TTFT starts before the HTTP request and ends at the first non-empty reasoning or answer delta; keep-alives and empty deltas are ignored.

The reduced suite keeps 4K/32K/120K inputs at a 131,072 context limit and 260K input at 262,144. It omits the 64K-limit matrix, short-input runs at 262K, placement trials and historical-stop replays. It reduces recall to one middle-position 260K case per model, conversation to one IQ2_XS 128K sequence, and continuous operation to 15 minutes. These checks are functional smoke tests, not general quality or daily-use certification.

## Fresh speed

36/36 speed requests completed. PP is input-processing throughput; decode is generation throughput. Medians (minimum–maximum), tok/s; TTFT in seconds. Every row has three runs with the displayed actual input count and 512 generated tokens per run, including reasoning. Individual counts and timings remain in the per-request data.

| Model | Context limit | Actual input tokens | Generated tokens | PP | Decode | TTFT |
|---|---:|---:|---:|---|---|---|
| Q2_0 | 131,072 | 4,096 | 512 | 920.1 (914.7–966.4) | 65.7 (61.5–66.3) | 4.5 (4.3–4.5) |
| Q2_0 | 131,072 | 32,768 | 512 | 1040.1 (1027.3–1084.0) | 64.0 (61.7–65.2) | 31.6 (30.3–32.0) |
| Q2_0 | 131,072 | 120,000 | 512 | 981.3 (979.2–986.0) | 57.6 (56.3–59.8) | 122.4 (121.8–122.7) |
| Q2_0 | 262,144 | 260,000 | 512 | 889.9 (888.7–890.9) | 54.2 (53.3–55.3) | 292.4 (292.0–292.8) |
| IQ2_XS | 131,072 | 4,096 | 512 | 857.4 (853.1–879.6) | 60.4 (58.7–64.5) | 4.8 (4.7–4.9) |
| IQ2_XS | 131,072 | 32,768 | 512 | 974.8 (968.6–976.7) | 60.7 (56.5–61.6) | 33.7 (33.6–33.9) |
| IQ2_XS | 131,072 | 120,000 | 512 | 923.8 (922.5–931.7) | 58.0 (55.2–58.1) | 130.0 (128.9–130.2) |
| IQ2_XS | 262,144 | 260,000 | 512 | 851.2 (851.1–852.1) | 52.6 (48.7–54.9) | 305.7 (305.3–305.7) |
| IQ3_XXS | 131,072 | 4,096 | 512 | 732.8 (726.7–748.5) | 37.0 (35.0–39.2) | 5.7 (5.5–5.7) |
| IQ3_XXS | 131,072 | 32,768 | 512 | 940.4 (929.5–944.5) | 37.1 (35.9–37.4) | 34.9 (34.8–35.3) |
| IQ3_XXS | 131,072 | 120,000 | 512 | 891.0 (888.1–893.3) | 35.9 (32.6–36.7) | 134.8 (134.5–135.3) |
| IQ3_XXS | 262,144 | 260,000 | 512 | 827.4 (826.8–828.5) | 33.8 (33.6–35.2) | 314.5 (314.0–314.7) |

## Recorded 0.1.43 comparison

This reuses saved 0.1.43 observations; no old-engine rerun. 12/12 speed groups have matching request payload hashes for all three repetitions. Engine version, calibration settings, run date, OS cache and background load differ, so these changes do not isolate an engine-version effect. The 0.1.43 report used MTP floor 0.5 and default PCIe/worker/adaptation settings.

| Model | Limit | Target input | Decode 0.1.43 → 0.1.44.1 | PP 0.1.43 → 0.1.44.1 |
|---|---:|---:|---|---|
| Q2_0 | 131,072 | 4,096 | 58.5 → 65.7 (+12.3%) | 906.4 → 920.1 |
| Q2_0 | 131,072 | 32,768 | 60.7 → 64.0 (+5.4%) | 1017.6 → 1040.1 |
| Q2_0 | 131,072 | 120,000 | 53.9 → 57.6 (+6.9%) | 965.3 → 981.3 |
| Q2_0 | 262,144 | 260,000 | 50.6 → 54.2 (+7.1%) | 877.3 → 889.9 |
| IQ2_XS | 131,072 | 4,096 | 57.6 → 60.4 (+4.9%) | 845.8 → 857.4 |
| IQ2_XS | 131,072 | 32,768 | 56.6 → 60.7 (+7.2%) | 965.2 → 974.8 |
| IQ2_XS | 131,072 | 120,000 | 54.9 → 58.0 (+5.6%) | 917.2 → 923.8 |
| IQ2_XS | 262,144 | 260,000 | 47.4 → 52.6 (+11.0%) | 826.3 → 851.2 |
| IQ3_XXS | 131,072 | 4,096 | 34.0 → 37.0 (+8.8%) | 693.5 → 732.8 |
| IQ3_XXS | 131,072 | 32,768 | 34.1 → 37.1 (+8.8%) | 903.9 → 940.4 |
| IQ3_XXS | 131,072 | 120,000 | 33.2 → 35.9 (+8.1%) | 862.1 → 891.0 |
| IQ3_XXS | 262,144 | 260,000 | 32.2 → 33.8 (+5.0%) | 795.1 → 827.4 |

## Representative functional checks

One code word is placed at 50% depth of each 260K input. A pass requires it in the final answer, not reasoning alone; the output cap is 2,048 tokens. 3/3 passed.

| Model | Actual input | Position | Result |
|---|---:|---:|---|
| Q2_0 | 260000 | 50% | found |
| IQ2_XS | 260000 | 50% | found |
| IQ3_XXS | 260000 | 50% | found |

IQ2_XS at the 128K limit receives a roughly 32K module and three follow-up turns using its actual prior final answers. All four requests completed, with input reuse on the follow-ups. Output cap 1,024. This checks API continuity/reuse, not coding correctness.

| Turn | Input tokens | Reused | Fresh | TTFT seconds |
|---:|---:|---:|---:|---:|
| 0 | 32757 | 0 | 32757 | 33.49 |
| 1 | 33818 | 32752 | 1066 | 2.25 |
| 2 | 34913 | 33813 | 1100 | 2.28 |
| 3 | 36007 | 34908 | 1099 | 2.27 |

IQ2_XS then completed 16.4 minutes of back-to-back fresh 4K/32K/120K requests (15 requests, 512-token cap, no idle delays). This is a short continuous-operation check; the previous one-hour soak was not repeated. The completion record is retained in the complete records.

## Sampled resources and limits

No request-duration deadline. Protection stops an owned run if available RAM remains below 1 GiB for 15 seconds or Windows commit is at least 95% for 30 seconds. No reduced-residency fallback was used. About one-second samples include model loading, warmup and inference; the IQ2_XS 128K row also includes the short soak. They record whole-machine available RAM, commit, VRAM, GPU clocks/power/temperature, system-wide disk I/O and owned-process-tree RSS. Peaks are sampled observations, not allocation maxima, and disk I/O is not model-attributed.

| Configuration | Min available RAM GiB | Peak commit | Peak total VRAM GiB | Peak GPU °C |
|---|---:|---:|---:|---:|
| q2_0-131072 | 15.79 | 79.7% | 11.01 | 88 |
| q2_0-262144 | 15.56 | 82.2% | 10.96 | 89 |
| iq2_xs-131072 | 15.96 | 81.6% | 10.91 | 89 |
| iq2_xs-262144 | 14.02 | 84.5% | 10.91 | 89 |
| iq3_xxs-131072 | 2.45 | 60.3% | 10.92 | 89 |
| iq3_xxs-262144 | 2.31 | 62.5% | 10.92 | 90 |

## Evidence and reproduction

The compact layout follows the accepted [R9700 report](../2026-10-08-community-r9700-linux/README.md): CSVs for inspection, complete records in gzip, and offline verification. The measurements have not been rerun or changed.

| File | Contents |
|---|---|
| [requests.csv](requests.csv) | All 70 requests, including warmups and slow samples: counts, prompt/decode durations, throughput, TTFT, whole-request latency, draft counts, finish status and payload hashes |
| [speed.csv](speed.csv) | All 12 fresh-speed groups, with three-repeat medians and ranges |
| [comparison-0143.csv](comparison-0143.csv), [historical-requests.csv](historical-requests.csv) | The saved 0.1.43 comparison and its 36 corresponding observations |
| [resources.csv](resources.csv) | The six configurations' sampled memory and temperature limits |
| [fixed-configs/](fixed-configs/), [calibration-settings.json](calibration-settings.json), [protocol.json](protocol.json) | Fixed launch arguments, selected official calibration outputs and timing/sampling/guard policy |
| [measurements.json.gz](measurements.json.gz) | Complete original current publication: all bodies, reasoning/answers, metrics, logs, telemetry, calibration results and frozen sources; historical source files needed for the comparison |
| [harness/](harness/) | Byte-identical copies of the published measurement, replay, aggregation and validation scripts for code review |
| [coverage.json](coverage.json), [SHA256SUMS](SHA256SUMS) | Retained scope and publication-file hashes |

The gzip file is schema 1: `files` maps original 0.1.44.1 report-relative paths to byte count, SHA-256, Git blob SHA and base64-encoded bytes; `historical_files` does the same for the 0.1.43 comparison sources. Compression preserves their original bytes, including the existing gzip request files. Source commits are pinned inside the bundle. The historical source files include other original observations; only the 36 matching speed requests are used in the comparison CSV. Historical placement trials, retries and the one-hour soak are available in the linked fixed-commit publication and are not current-version checks. Model weights, executable binaries and generated packs are excluded.

Verify without a GPU, local engine, downloads or network access:

```text
python verify_report.py
python verify_report.py --extract NEW_EVIDENCE_DIR
```

The checker validates publication and source-byte hashes, all 70 saved requests/outputs/metrics, calibration settings, and CSVs against a fresh aggregation. It also checks all 36 old/new payload hashes and recomputes the comparison medians. `--extract` requires a new destination and restores `current/` and `historical/`; it preserves existing files. The original snapshot README is retained as provenance; use this README for the compact publication's commands.

To repeat the suite on Windows, use the recorded Python runtime checkout, engine and model assets, with an existing IQ2_XS installation containing `PATH_TO_STRATA/strata-iq2_xs.json`. The runner and validator hash this local file to check that the normal configuration remains unchanged; its contents need not match the original machine's configuration. Benchmark launches use the extracted `fixed-configs/*.json`. Extract first, then use the Strata Python environment. No downloads or updates are performed:

```text
PATH_TO_VENV_PYTHON NEW_EVIDENCE_DIR/current/reproduce.py --runtime PATH_TO_STRATA --data PATH_TO_STRATA_DATA --out NEW_OUTPUT_DIR
PATH_TO_VENV_PYTHON NEW_EVIDENCE_DIR/current/analyze.py NEW_OUTPUT_DIR --out NEW_ANALYSIS_DIR
PATH_TO_VENV_PYTHON NEW_EVIDENCE_DIR/current/validate.py NEW_OUTPUT_DIR --runtime PATH_TO_STRATA
```

The replay/validation scripts are frozen to this Windows configuration, including engine 0.1.44.1 and the observed 11 workers; they are not a general hardware benchmark framework. The measured-source snapshot and published runner have separate hashes. Their only difference is the corrected usage filename and soak example in the module documentation; execution logic is unchanged.

Local paths are replaced with `<STRATA>`, `<DATA>` and `<LOCAL_ROOT>`. Normal-use configuration was unchanged and all owned benchmark processes were stopped. Speed and synthetic recall do not establish general agent quality.
