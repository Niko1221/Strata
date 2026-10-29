# Community benchmark: RTX 4090 + RTX 5060 Ti, UD-IQ4_XS

Measured 2026-10-29 by @ayvan. One custom Windows configuration, a standard
short suite and a separate long-speed/recall suite. No engine changes or claim
of a two-GPU speedup. All per-request measurements are in [results.json](results.json).

## Hardware and software

| Component | Recorded configuration |
| --- | --- |
| CPU | Ryzen 7 9800X3D, 8 cores / 16 threads |
| RAM | **128 GiB, 4 × 32 GiB A-DATA DDR5, configured 4800 MT/s**; 125.6 GiB OS-visible. Windows SMBIOS plus owner confirmation. DRAM timings and measured RAM bandwidth not obtained |
| Motherboard | Gigabyte X870 AORUS STEALTH ICE; BIOS F10c, 2026-07-21; PCB revision not read |
| GPU 0 | RTX 4090, 24564 MiB VRAM, observed PCIe **Gen4 x16**, 550 W power limit |
| GPU 1 | RTX 5060 Ti, 16311 MiB VRAM, observed PCIe **Gen4 x4**, 180 W power limit |
| Model drive | XPG GAMMIX S70 BLADE NVMe SSD |
| OS / driver | Windows 10 build 19045; NVIDIA 616.56 |
| Engine | Prebuilt 0.1.52, build commit a42cc3d7df70, CUDA 13.0 |
| Terminal/control | Bundle 0.1.53, runner commit 55a557bb6d77facc73d291c3f9894289496c770c |

The 4090 uses the CPU-attached primary slot; the 5060 Ti uses the chipset-attached
slot, consistent with Windows PnP paths and the manufacturer's
[board specifications](https://www.gigabyte.com/Motherboard/X870-AORUS-STEALTH-ICE-rev-10/sp).
Gen4 x16/x4 was sampled after the long suite; idle Gen1 readings are also retained.
The engine's startup host-to-device probes were **26.8 GB/s (CUDA0)** and
**6.5 GB/s (CUDA1)**. Chipset uplink traffic was not measured. `nvidia-smi topo -m`
failed on Windows; no P2P/NVLink claim. See [hardware.json](hardware.json) for
module part numbers, CIM sources and link snapshots. Power limits are not
measured consumption. Background application load was not recorded.

## Model and settings

`unsloth/Qwen3.8-Flash-Next-GGUF`, UD-IQ4_XS, three GGUF shards and native pack.
Full GGUF SHA-256 values are in results.json; auxiliary asset hashes are in
[configuration.json](configuration.json). Immutable source/quant/tokenizer
repository revisions were not recorded.

- GPU order `[0,1]`, layer split **38**: 4090 layers 0–37; 5060 Ti layers 38–47.
  One inference slot; this is a layer split, not helper-cache-only mode.
- Context **262144**, KV **int8**, resident KV **32768**; expert cache auto;
  prefill auto; MTP on, spec **4**, spec-min-p **0.5**; pool-workers **11**;
  host-core last; pipeline-windows **2**; trim-stage-weights; remote-expert-opt.
- GPU vision configured (max 1024 image tokens), but images were not tested.
  VRAM reserve **300 MiB**. Startup reports 6922/5010 GPU cache slots,
  8192-token auto chunks, route-tail skip on and layer-major prefill for the
  long suite. The saved tuner overlay is empty. No control-vector options.
- Recorded start: `strata launch --preset "UD-IQ4_XS"`. Full run/effective/launch
  configs are in configuration.json with private paths and credentials removed.
  Later-collected presets are labelled separately from test-time snapshots.

[engine-log.txt](engine-log.txt) retains startup/cache/page-lock details, low-VRAM
warnings and the long suite's timing lines. Snapshot memory usage is not a peak
measurement and does not rule out WDDM paging. No config was changed for testing.

## Standard short suite — draft protocol

Installed `strata bench run --profile launch-ud-iq4_xs --no-start --track custom
--quantization UD-IQ4_XS --out NEW_DIRECTORY`, frozen standard-v1 corpus.
Thinking on/preserved, effort xhigh, temperature 1, top_p .95, top_k 20,
min_p 0, presence penalty 0, repetition penalty 1. Streaming; one excluded
warm-up, three speed repeats, twelve sanity cases. Output caps: 512 for speed,
2048 for sanity, including reasoning. No retries. Model load/hash time excluded.

| Input case | Prompt tok/s median (range) | Decode tok/s median (range) | Streaming TTFT, seconds median (range) |
| --- | --- | --- | --- |
| Short | not reported: fixed overhead dominates | 95.6 (91.4–98.1) | 0.83 (0.69–0.88) |
| 4K | 1387.0 (1385.3–1393.8) | 95.4 (90.0–108.2) | 3.00 (2.95–3.00) |
| 16K | 2504.4 (2498.4–2510.0) | 101.4 (84.4–108.7) | 6.59 (6.56–6.61) |

**21/21 measured requests, 0 failed/skipped; sanity 12/12** (instructions 4/4,
retrieval 4/4, tool calls 4/4). About **1 min 51 s** including warm-up.
All six longer speed requests reached the 512-token cap. The model had been
restarted; one existing request had 53 input tokens and 1 output token, consistent
with a startup check (origin unverified). The suite itself warms the expert
cache. The protocol is draft; this is a custom-track community report, not an
approved standard-schema submission. Raw counts, metrics and sanity outputs are
retained; full client-event/reasoning transcripts are omitted from this packet.

## Historical long-speed ladder

After another restart (again one existing 53-input/1-output request), reused
the repository's [TITAN benchmark](https://github.com/Niko1221/Strata/blob/46c152a1310f77cd2f0ce2c7cd213962df14fe83/bench/results/2026-10-08-community-2x-titan-rtx-0.1.40.3/benchmark.py).
Only output paths, incremental saving and raw evidence capture were changed;
prompts, targets, sampling and output cap were preserved. Three repetitions,
temperature 0, nonstreaming, cap 256, thinking at server default. First request
included; no extra warm-up. Repetitive MoE prose, not a coding workload.

| Nominal target | Actual input | Reused | Newly read | Prompt tok/s median (range) | Decode tok/s median (range) |
| --- | --- | --- | --- | --- | --- |
| 4096 | 3040 | 0 | 3040 | 1151.1 (874.9–1152.2) | 108.5 (92.5–110.0) |
| 32768 | 24537 | 0 | 24537 | 2813.5 (2802.9–2827.1) | 112.8 (98.6–114.1) |
| 128000 | 95751 | 16384 | 79367 | 3293.3 (3267.9–3340.0) | 109.1 (104.5–112.4) |

Counts are identical across each row's three repeats; all outputs reached 256.
Prompt speed counts newly read tokens. **The last row is not a fresh-prefix
128K test**: the script estimates length by characters and reuse occurred.
Expert cache evolves. Raw `ttft_s` here is wall minus decode time, only an
estimate; it is not the standard suite's observed streaming TTFT.

## Long-context recall

After the ladder, ran the unchanged [tools/needle_bench.py](https://github.com/Niko1221/Strata/blob/46c152a1310f77cd2f0ce2c7cd213962df14fe83/tools/needle_bench.py):

```sh
python tools/needle_bench.py --url http://127.0.0.1:8080 --lengths 32k,128k,240k --depths 10,50,90 --out needles.json
```

Thinking off, temperature 0, cap 40; seed 7; one code word at each character
depth 10%, 50%, 90%; one trial per combination. Repository text supplies filler.

| Target | Actual input tokens | Depths | Found |
| --- | --- | --- | --- |
| 32k | 31706–31707 | 10%, 50%, 90% | 3/3 |
| 128k | 122904–122906 | 10%, 50%, 90% | 3/3 |
| 240k | 232807–232808 | 10%, 50%, 90% | 3/3 |

**9/9 found, no request errors.** Per-case words, answers and wall times are
in results.json. These test extraction on these inputs, not general quality.
Filler changes with the checkout; source revision and script hashes are recorded,
but original needle request bytes were not captured. Recall latency is not a
fresh-prefix 233K speed measurement.

## Validation and limits

**Passed:** both original saved standard runs validated with `strata bench report`
(only the restarted run is included here). Published JSON parses; per-request
counts, cache accounting, output caps and summary values checked. Long-suite
server count rose by exactly 18 (nine speed + nine recall). Links, hardware
consistency and private-path/credential scan checked before submission.

**Failed:** no inference request failures; topology matrix unavailable. Historical
speed targets were shorter than labelled and its longest point reused a prefix;
these limitations are kept visible above, not treated as clean-prefix results.

**Not run:** single-GPU baseline, alternate split/reserve, actual fresh-prefix
128K/240K throughput, concurrency, images, coding-quality suite, repeated needle
seeds, RAM/VRAM/power peaks, RAM/SSD bandwidth, DRAM timings, chipset uplink
measurement, another-machine reproduction. Unknown immutable model revisions
remain unknown. No claim that the second GPU or a restart makes inference faster.

Prepared with agent assistance from this machine's measurements. This five-file
packet follows the community guide: human report, raw per-run data, sanitized
configuration, hardware evidence and relevant engine logs. No models, generated
packs, full conversations or reconstructed corpus dumps are included.
