# RTX 5070 Ti + RTX 3060 versus RTX 5070 Ti: Flash Next UD-Q4_K_XL

**Measured 2026-10-14 by @holgin • Strata 0.1.44.1 • two reversed blocks • eight fresh services • 360 measured requests**

On this machine, the default dual-GPU profile improved median short-chat decode by **13.57%** and simulated-agent follow-up decode by **15.32%** over the 5070 Ti alone. The single-GPU baseline processed fresh 16K and 32K prompts **13.34%** and **5.40%** faster. The saved dual calibration settings did not provide a general improvement and were not promoted.

This is a same-build performance report with the existing OC held constant. It supplies evidence for the default automatic split on an unequal NVIDIA pair; it does not establish coding quality, an optimal split, a stock-versus-OC gain, or an engine regression.

## Results

Values are engine tok/s: **median (minimum–maximum)** pooled over two fresh services per arm. Chat/probes use n=18 requests per arm; each fresh prompt size n=6; agent follow-ups n=30. These ranges are descriptive, not confidence intervals. The number of independent service restarts is two per arm.

| Arm | Short-chat decode | Fresh 4K prefill | Fresh 16K prefill | Fresh 32K prefill | Agent follow-up decode | Calibration-prompt decode |
|---|---:|---:|---:|---:|---:|---:|
| DUAL-BASE | 28.45 (23.50–33.90) | 351.15 (350.10–352.90) | 598.60 (596.40–599.70) | 646.40 (645.60–647.50) | 29.35 (21.30–40.10) | 28.75 (25.60–31.20) |
| SINGLE-BASE | 25.05 (21.30–29.80) | 348.30 (343.60–350.50) | 678.45 (674.20–679.60) | 681.30 (679.70–682.60) | 25.45 (15.50–35.30) | 25.15 (23.00–28.20) |
| DUAL-TUNED | 27.00 (24.80–32.40) | 351.15 (347.80–352.10) | 599.00 (597.40–599.50) | 647.10 (645.00–648.20) | 30.25 (23.40–39.80) | 27.70 (25.30–29.70) |
| SINGLE-TUNED | 21.80 (19.70–28.60) | 347.60 (342.00–350.20) | 679.35 (676.00–680.80) | 681.40 (680.00–682.40) | 24.05 (17.70–32.00) | 21.85 (19.90–23.60) |

| Candidate / reference | Chat | 4K prefill | 16K prefill | 32K prefill | Agent | Calibration prompts |
|---|---:|---:|---:|---:|---:|---:|
| DUAL-BASE / SINGLE-BASE | +13.57% | +0.82% | -11.77% | -5.12% | +15.32% | +14.31% |
| DUAL-TUNED / DUAL-BASE | -5.10% | +0.00% | +0.07% | +0.11% | +3.07% | -3.65% |
| SINGLE-TUNED / SINGLE-BASE | -12.97% | -0.20% | +0.13% | +0.01% | -5.50% | -13.12% |

A positive percentage means the candidate is faster. The saved tuning settings reduce dual chat/probe decode by 5.10%/3.65% while increasing the agent median by 3.07%. Token-weighted agent decode changes only +0.61% (29.68 → 29.86 tok/s); generated follow-up tokens change from 1,659 to 1,522. The same settings transferred to single GPU reduce chat/probe medians by 12.97%/13.12%. Later agent contexts depend on each arm’s own answers, so these are not byte-identical trajectories.

**Operational choice:** retain the measured dual baseline for sustained decode and this simulated-agent workload. The single baseline is useful for fresh long prompt ingestion. Neither choice implies better answers.

## Hardware, software and model

| Item | Measured configuration |
|---|---|
| CPU / board | Intel Core i9-9900K at 4.7 GHz (owner-reported), 8 cores / 16 threads, AVX2; MSI MEG Z390 ACE |
| RAM / swap | 128 GiB nominal, DDR4-2800 (2800 MT/s, owner-reported); Linux MemTotal 127,365,092 KiB (121.46 GiB); 8 GiB swap |
| Physical GPU 0 | RTX 5070 Ti, 16,303 MiB, sm_120; no display |
| Physical GPU 1 | RTX 3060, 12,288 MiB, sm_86; drives desktop |
| Interconnect | Both cards physically installed in every arm; observed under-load PCIe 3.0 x8, no NVLink |
| Model storage | Samsung SSD 980 PRO NVMe, ext4, model/root mount /dev/nvme0n1p2; counters nvme0n1 |
| OS / driver | Ubuntu 26.04.1 LTS; Linux 7.0.0-38-generic; NVIDIA 595.91.07 |
| Build | Local Release CUDA build, toolkit 13.4, architectures 86;120; Python 3.14.4 |
| Strata | 0.1.44.1, commit `296b59d2f73c8f04a805df29c2fa97655adace78`; tracked source clean |
| Model | Unsloth Qwen3.8-Flash-Next UD-Q4_K_XL, qwen4exp, 48 layers / 512 experts per layer |
| Model revision | `38bb39ee97821de2c9009abb7e93950eec396e66`; four SHA-256-verified shards, 111,334,654,784 bytes (103.68 GiB) |
| Pack | Native GGUF expert references; `iq_pack.py --compat-bf16`; no experts.bin; conversion metadata included |

Model source: [pinned Hugging Face revision](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66). The NVIDIA utility’s “CUDA 13.2” field is a driver capability display, distinct from the 13.4 build toolkit. Build flags, binary/library hashes, every shard name/hash, MTP/tokenizer/expert-profile hashes and upstream script hashes are in [provenance.json](provenance.json). BF16 compatibility conversion is not a claim of lossless recovery of pre-quantization weights. On 2026-10-15 the owner confirmed that the benchmark ran with DDR4-2800 (2800 MT/s) and the i9-9900K at 4.7 GHz. These are owner-reported operating settings; the original telemetry did not independently sample DDR4 rate or CPU frequency. Memory timings were not recorded.

A graphical session remained active (GNOME and Chromium/ChatGPT). The 3060 continued driving the desktop even when excluded from inference. No other model or training job ran concurrently. The report includes this background activity; it is not an idle laboratory measurement.

## Configuration and calibration

Common settings: 65,536 context, INT8 KV, 32,768-token resident KV window, 72 GiB resident expert budget, expert cache/prefill auto, 1,800 MiB VRAM reserve per stage, one slot, fit_max_tokens true, English draft vocabulary, MTP draft cap 4, `STRATA_HC_Q8=1`. Temperature 0, thinking/reasoning off, text only. API bound to 127.0.0.1:8081. Arena async and optional speed projection were not enabled. CUDA K-quant prompt support was compiled in; the runtime opt-in MMQ path was not enabled.

| Arm | GPU selection and placement | Changed settings |
|---|---|---|
| [DUAL-BASE](configs/DUAL-BASE.json) | gpu [0,1], layer_split auto; engine orders physical 3060 → 5070 Ti, layers [0,3) → [3,48), head/MTP on 5070 Ti | Baseline, readback workers 7 / PCIe fraction 0.17 / draft floor 0.50 |
| [SINGLE-BASE](configs/SINGLE-BASE.json) | gpu 0; all 48 layers on 5070 Ti | Baseline, readback workers 7 / PCIe fraction 0.18 / draft floor 0.50 |
| [DUAL-TUNED](configs/DUAL-TUNED.json) | Same default dual placement | Saved calibration settings below |
| [SINGLE-TUNED](configs/SINGLE-TUNED.json) | Same single selection | Dual-derived settings transferred without single-GPU recalibration |

Saved settings: `--pcie-frac 0.00 --spec-min-p 0.70 --pool-workers 5 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97`. The earlier official dual calibration search and numeric sweeps are retained in [calibration-source.json](calibration-source.json). Its 33.9 tok/s result came from a narrow direct-engine, 512-token adaptive-tier probe. It is a different protocol from these API measurements and should not be quoted as general model speed. These four arms compare the combined settings; they do not isolate the effect of each setting.

## Measurement protocol

Two sequential blocks, one fresh isolated service per arm:

1. DUAL-BASE → SINGLE-BASE → SINGLE-TUNED → DUAL-TUNED.
2. DUAL-TUNED → SINGLE-TUNED → SINGLE-BASE → DUAL-BASE.

Each service passed streamed/non-streamed API smoke, then ran the pinned upstream `tools/strata_bench.py` workload with `--runs 3 --long`. A small wrapper fixes only the Suite nonce to `match-20261014T162442Z`, matching initial prompts across servers while preserving unique tags within each suite. Agent follow-ups contain generated answers and therefore can diverge. [Reproduction instructions and helpers](REPRODUCE.md) show the portable equivalent.

| Workload per fresh service | Actual tokens / output cap | Measured requests |
|---|---|---:|
| Three short questions × 3 repeats | 50–58 prompt tokens; cap 100 | 9 |
| Fresh 4K / 16K / 32K, each × 3 repeats | 4,020 / 16,058 / 32,049 prompt tokens; cap 24; prefix reuse zero | 9 |
| Three six-turn simulated agent sessions | Initial turn plus five tool follow-ups; follow-up cap 64; actual input/output/reuse in request records | 18 |
| Three calibrate.py prompts × 3 repeats | Same tagged public prompts, cap 128, streamed API | 9 |

Total: 288 official measured requests (including initial agent turns) + 72 extra calibration-prompt probes = **360**. Smoke and warm-up calls are excluded. The official suite discards a short chat, two token-count slope probes and a 1K warm-up prompt. The extra probes run after the official suite. No generated code is executed; simulated tools do not measure an autonomous coding agent.

Each new service resets engine state. OS page cache is retained; no cache-drop is performed. Load timing is reported separately. Fresh prompts never reuse a conversation prefix, but can benefit from expert RAM/GPU cache warmed by preceding requests. Agent reuse and freshly read tokens remain separate columns. Engine-reported prompt and decode timings drive throughput; token-weighted decode is sum(generated tokens)/sum(decode_ms) × 1,000. Client TTFT starts at request send and ignores keep-alives/empty deltas; reasoning is off. Total request latency excludes model loading and includes prompt + decode + API overhead.

### TTFT and latency

Median seconds across two services; agent column excludes initial turns. Full ranges and all request latencies remain in [summary.json](summary.json) and [requests.csv](requests.csv).

| Arm | Chat TTFT | 4K TTFT | 16K TTFT | 32K TTFT | Agent TTFT | Chat total | Agent total |
|---|---:|---:|---:|---:|---:|---:|---:|
| DUAL-BASE | 1.204 | 11.487 | 26.858 | 49.631 | 11.425 | 4.685 | 13.284 |
| SINGLE-BASE | 1.212 | 11.578 | 23.705 | 47.090 | 11.582 | 5.141 | 13.677 |
| DUAL-TUNED | 1.223 | 11.476 | 26.836 | 49.573 | 11.423 | 4.908 | 13.102 |
| SINGLE-TUNED | 1.212 | 11.598 | 23.666 | 47.080 | 11.579 | 5.729 | 13.744 |

## Loading, memory, storage and restart variation

Every fresh service used a separate transient user systemd cgroup: MemoryHigh=100G, MemoryMax=106G, MemorySwapMax=0, RuntimeMaxSec=7200. Preflight required ≥100 GiB MemAvailable and exclusive model access. The monitor sampled approximately every two seconds and stopped on service swap/OOM, <12 GiB global available RAM, >512 MiB additional global swap, GPU ≥85 °C or repeated CPU ≥95 °C / telemetry failure. These limits and MemoryHigh reclaim are part of the measured setup.

| Block / arm | Load s | Official suite s | Chat decode | Agent decode | Min available GiB | Root NVMe reads GiB | MemoryHigh events | Global swap peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 / DUAL-BASE | 98.03 | 554.66 | 28.00 | 29.70 | 35.77 | 93.40 | 11685 | 0.000 |
| 1 / SINGLE-BASE | 72.02 | 530.80 | 25.40 | 25.20 | 43.31 | 91.21 | 7055 | 0.320 |
| 1 / SINGLE-TUNED | 84.03 | 540.80 | 21.40 | 24.70 | 43.37 | 92.30 | 10752 | 0.320 |
| 1 / DUAL-TUNED | 96.02 | 555.99 | 27.10 | 30.40 | 36.47 | 91.29 | 12379 | 0.320 |
| 2 / DUAL-TUNED | 96.03 | 551.11 | 26.90 | 30.10 | 36.36 | 93.40 | 11596 | 0.320 |
| 2 / SINGLE-TUNED | 64.02 | 538.37 | 22.40 | 23.70 | 43.33 | 85.94 | 5145 | 0.320 |
| 2 / SINGLE-BASE | 84.02 | 529.07 | 24.70 | 25.70 | 43.29 | 93.66 | 10727 | 0.637 |
| 2 / DUAL-BASE | 98.03 | 555.36 | 28.90 | 29.00 | 36.35 | 91.56 | 12288 | 313.020 |

All eight services completed with zero failed/skipped measured requests, zero service swap, zero cgroup max/OOM/OOM-kill events and no observed kernel GPU/engine error matches. Peak cgroup memory was approximately **100 GiB** in every run; minimum global available RAM was **35.77 GiB**. **MemoryHigh reclaim occurred in every run. Global swap peaked at 313.02 MiB and remained about 295.18 MiB after shutdown.** This is not a zero-global-swap or zero-memory-pressure result; no process-level causal attribution was made.

Physical read deltas use the underlying current model/root device nvme0n1 and span each entire service, including loading, smoke, suite and probes: 85.94–93.66 GiB per service, 732.75 GiB total. They include all device activity and are not expert-only decode reads. Device identities were resolved against the model mount; historical pre-reboot names are not used. Warm expert-file counters in the engine log are a separate measurement. The suite times above exclude loading, smoke and the nine extra probes; generated answer lengths vary, so service wall time is not a pure fixed-output speed comparison.

## Existing OC and GPU telemetry

Existing settings remained unchanged throughout: 5070 Ti power limit 330 W (default 300 W), LACT max_core_clock 3090 and memory offset 5,000; 3060 power limit 187 W (default 170 W), memory offset 2,000. Offsets are reported in raw LACT configuration units, without guessing effective MHz. Every before/after device-limit snapshot and the LACT configuration hash matched. No stock arm, fixed-clock lock or controlled cooldown was measured.

| Block / arm | GPU0 / GPU1 peak VRAM MiB | GPU0 / GPU1 max °C | GPU0 / GPU1 busy SM median MHz | GPU0 / GPU1 peak sampled W |
|---|---:|---:|---:|---:|
| 1 / DUAL-BASE | 14911.0 / 10742.0 | 68.0 / 66.0 | 2842.0 / 2032.0 | 102.16 / 100.26 |
| 1 / SINGLE-BASE | 14647.0 / 442.0 | 67.0 / 64.0 | 2827.0 / — | 98.75 / 39.99 |
| 1 / SINGLE-TUNED | 14647.0 / 424.0 | 65.0 / 64.0 | 2827.0 / — | 98.63 / 40.31 |
| 1 / DUAL-TUNED | 14671.0 / 10593.0 | 67.0 / 64.0 | 2827.0 / 2032.5 | 99.04 / 97.06 |
| 2 / DUAL-TUNED | 14669.0 / 10589.0 | 63.0 / 65.0 | 2827.0 / 2032.0 | 104.75 / 100.72 |
| 2 / SINGLE-TUNED | 14647.0 / 427.0 | 66.0 / 64.0 | 2827.0 / — | 98.74 / 40.12 |
| 2 / SINGLE-BASE | 14647.0 / 427.0 | 65.0 / 64.0 | 2827.0 / — | 98.18 / 39.99 |
| 2 / DUAL-BASE | 14669.0 / 10597.0 | 66.0 / 65.0 | 2827.0 / 2040.0 | 100.79 / 99.96 |

GPU0 is the 5070 Ti, GPU1 the 3060. Busy-clock medians include only samples with GPU utilization ≥10%; “—” means no such samples for the desktop-only 3060. Temperatures/power are observed two-second sample peaks, not electrical transient limits. Maximum sampled GPU temperature was 68 °C. Startup host-to-device probes are recorded in logs; they do not establish an end-to-end PCIe bottleneck. There is no stock-versus-OC causal speedup claim or general OC stability certification. The unchanged LACT SHA-256 was `df79b6a0f22417fe45eec2adfeac6677d549cb9f9cb8b92194519860d36a733b`.

## Scope and remaining tests

- Measured: default automatic dual placement versus single 5070 Ti, saved combined calibration versus baseline, public API throughput/TTFT/reuse, loading and numeric hardware telemetry.
- Not measured here: coding correctness, needle recall, concurrency, 128K/native 262K prompts, manual splits, both explicit GPU orders, independently calibrated single-GPU optimum, stock clocks, or an uninterrupted one-hour soak. Eight shorter services must not be described as a soak.
- No previous Strata-version arm: this report cannot quantify a 0.1.43 → 0.1.44.1 improvement or regression. Two service restarts per arm and active desktop background limit generalization. No statistical significance is claimed.
- Pooled mixed-prompt medians are workload descriptions. Output lengths and draft acceptance vary; official benchmark records contain timings/tokens, not complete decoded texts. The 72 calibration-probe texts/hashes are retained. Bit-identical answers were not established.

All models were stopped after measurement; Qwen remained disabled/inactive and model ports 8080–8083 were closed. No operating profile was promoted. A useful next test would isolate explicit GPU orders/manual splits or evaluate calibration with a longer fixed workload and independent correctness checks; these are suggestions, not completed measurements.

## Evidence and reproduction

- [DATA.md](DATA.md): fields, units, GPU-selection caveat and compressed telemetry format.
- [requests.csv](requests.csv) / [requests.json](requests.json): all 360 measured requests, actual tokens, prefix reuse, engine timings, draft counts, client TTFT and latency.
- [summary.json](summary.json): pooled distributions, weighted rates, per-service readback, placement and safety counters.
- [provenance.json](provenance.json): exact model/build/assets/scripts; [pack conversion metadata](pack-metadata/conversions.json). Original pack hashes identify the operational files; sanitized metadata has separate package checksums.
- [Validation record](VALIDATION.json), [final integrity check](audits/final-integrity.json), [post-campaign service state](audits/final-operating-state.json).
- [REPRODUCE.md](REPRODUCE.md): portable configuration rendering, bounded service launch and workload helpers. Helpers were checked offline; no additional GPU measurements are implied.
- [verify.py](verify.py): recomputes the reported results from exported requests and checks package integrity. Run `python3 verify.py` from this folder; it starts no model.

*Per-service records (in execution order):*

- [block1-DUAL-BASE](runs/block1-DUAL-BASE/summary.json): [official requests](runs/block1-DUAL-BASE/official.json), [calibration probes and answers](runs/block1-DUAL-BASE/calibration-probes.json), [engine log](runs/block1-DUAL-BASE/engine.log), [telemetry](runs/block1-DUAL-BASE/telemetry.jsonl.gz).
- [block1-SINGLE-BASE](runs/block1-SINGLE-BASE/summary.json): [official requests](runs/block1-SINGLE-BASE/official.json), [calibration probes and answers](runs/block1-SINGLE-BASE/calibration-probes.json), [engine log](runs/block1-SINGLE-BASE/engine.log), [telemetry](runs/block1-SINGLE-BASE/telemetry.jsonl.gz).
- [block1-SINGLE-TUNED](runs/block1-SINGLE-TUNED/summary.json): [official requests](runs/block1-SINGLE-TUNED/official.json), [calibration probes and answers](runs/block1-SINGLE-TUNED/calibration-probes.json), [engine log](runs/block1-SINGLE-TUNED/engine.log), [telemetry](runs/block1-SINGLE-TUNED/telemetry.jsonl.gz).
- [block1-DUAL-TUNED](runs/block1-DUAL-TUNED/summary.json): [official requests](runs/block1-DUAL-TUNED/official.json), [calibration probes and answers](runs/block1-DUAL-TUNED/calibration-probes.json), [engine log](runs/block1-DUAL-TUNED/engine.log), [telemetry](runs/block1-DUAL-TUNED/telemetry.jsonl.gz).
- [block2-DUAL-TUNED](runs/block2-DUAL-TUNED/summary.json): [official requests](runs/block2-DUAL-TUNED/official.json), [calibration probes and answers](runs/block2-DUAL-TUNED/calibration-probes.json), [engine log](runs/block2-DUAL-TUNED/engine.log), [telemetry](runs/block2-DUAL-TUNED/telemetry.jsonl.gz).
- [block2-SINGLE-TUNED](runs/block2-SINGLE-TUNED/summary.json): [official requests](runs/block2-SINGLE-TUNED/official.json), [calibration probes and answers](runs/block2-SINGLE-TUNED/calibration-probes.json), [engine log](runs/block2-SINGLE-TUNED/engine.log), [telemetry](runs/block2-SINGLE-TUNED/telemetry.jsonl.gz).
- [block2-SINGLE-BASE](runs/block2-SINGLE-BASE/summary.json): [official requests](runs/block2-SINGLE-BASE/official.json), [calibration probes and answers](runs/block2-SINGLE-BASE/calibration-probes.json), [engine log](runs/block2-SINGLE-BASE/engine.log), [telemetry](runs/block2-SINGLE-BASE/telemetry.jsonl.gz).
- [block2-DUAL-BASE](runs/block2-DUAL-BASE/summary.json): [official requests](runs/block2-DUAL-BASE/official.json), [calibration probes and answers](runs/block2-DUAL-BASE/calibration-probes.json), [engine log](runs/block2-DUAL-BASE/engine.log), [telemetry](runs/block2-DUAL-BASE/telemetry.jsonl.gz).

This results-only folder follows the pinned [community benchmark guide](https://github.com/Niko1221/Strata/blob/296b59d2f73c8f04a805df29c2fa97655adace78/docs/COMMUNITY_BENCHMARKS.md). Private paths and device UUIDs are replaced; models, packs and credentials are not included. Related heterogeneous-GPU planning discussion: [#2393](https://github.com/Niko1221/Strata/issues/2393) and [PR #2385](https://github.com/Niko1221/Strata/pull/2385). This report measures the pinned main build, not the unmerged planner PR.
