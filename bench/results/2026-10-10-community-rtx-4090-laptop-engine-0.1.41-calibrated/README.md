# Community benchmark follow-up: RTX 4090 Laptop GPU, engine 0.1.41, defaults and `--calibrate` settings

Measured on 2026-10-10 by [30crows](https://github.com/30crows) on the Lenovo Legion 82WQ of the
[2026-10-04 report](../2026-10-04-community-rtx-4090-laptop/README.md), now on Gentoo with Strata 0.1.41 and NVIDIA
615.78.08. Same model, workload and scripts as that report, at both of its context limits, once with the defaults
and once with the settings setup's calibration (`tools/calibrate.py`) picked on this laptop: a PCIe share of 0.20
instead of 0.55, a draft floor of 0.30 (131,072-token context) or 0.70 (262,144), and the adaptive expert tier
(`--adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97`).

Against the same day's runs with the defaults, **the calibrated settings raised decode throughput by 25-33% at the
131,072-token limit and by 32-39% at the 262,144-token limit**, to 75.9 / 81.9 / 76.6 tok/s at 4,096 / 32,768 /
128,000 prompt tokens and 74.6 / 71.8 / 73.4 / 70.6 tok/s at 4,096 / 32,768 / 128,000 / 250,000. Prompt throughput
stayed within 1.2%. One session per setting.

## Hardware and software

- NVIDIA GeForce RTX 4090 Laptop GPU (AD103), 16,376 MiB; PCIe Gen 4 x16 (16 GT/s x16 read in sysfs). Power limit
  175 W under load (Dynamic Boost, `nvidia-powerd`, platform profile `performance`): the enforced limit was 175 W in
  89-96% of each session's busy power samples (the others 150-173.7 W while Dynamic Boost ramped up), busy median
  draw 174 W, peak 80-83 °C, no thermal slowdown.
- Intel Core i9-13900HX (32 logical CPUs, no AVX-512; 23 expert-pool workers + the host thread); 64 GB RAM,
  62.52 GiB usable. Model files on btrfs (zstd:3) on LUKS on a WD_BLACK SN850X 2 TB. Swap (8 GiB zram, 64 GiB file)
  stayed unused.
- Gentoo Linux, kernel 6.18.54-gentoo-dist-bin; the IOMMU did not translate the GPU's DMA (group type `identity`,
  the kernel's default here). NVIDIA 615.78.08 (open kernel modules, built locally), CUDA 13.4 (V13.4.59), GCC 16.2.1,
  Python 3.14.7.
- Strata v0.1.41 (`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`), engine 0.1.41 built from source for sm_89 (CUDA
  architecture 89); engine sha256 `de1ee0eb6517fa0f2c76f7c988d2e3a12b11305eb983e711df2bd0bae7261cc8`, bundled
  expert profile sha256 `8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf`.
- Headless: console login, no desktop session, no browser.

## Model and configuration

The model files of the 2026-10-04 report (`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, IQ3_S, the installer's
pinned revision), `--kv int8`, `--expert-cache auto`, `--prefill auto`, MTP `--spec 4`. The defaults add
`--spec-min-p 0.5` and take the PCIe share from the startup probe (24.0 GB/s, so 0.55). The calibrated configs
replace and add, exactly as `./setup.sh --calibrate` writes them:

| Context | `--pcie-frac` | `--spec-min-p` | `--adapt-every` | `--adapt-swaps` | `--adapt-decay` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 131,072 | 0.20 | 0.30 | 1 | 80 | 0.97 |
| 262,144 | 0.20 | 0.70 | 1 | 80 | 0.97 |

The engine laid out the same in both settings: expert cache 3,897 slots (7.44 GiB) at 131,072 tokens and 2,886
(5.53 GiB) at 262,144; prompt chunks of 8,192 tokens with a 384-slot ring; 2,170 or 2,156 cache slots (4.12 GiB)
borrowed by the prompt path; 46.84 GiB of experts in RAM. The configs are in each folder (paths are the author's).
The server ran on loopback without a key and without a browser:

```bash
.venv/bin/python serve/server.py --engine strata --config strata-iq3_s-128k.json --port 8080
```

| Folder | Started | Boot | PCIe probe (best of four bursts) | Experts into RAM | Ready |
| --- | --- | --- | --- | ---: | ---: |
| [128k-default](128k-default/) | 11:00 | fresh | 24.0 GB/s (24.0 24.0 24.0 24.0) | 6.08 GiB/s | 20 s |
| [256k-default](256k-default/) | 12:09 | fresh | 24.0 GB/s (23.9 24.0 24.0 24.0) | 6.10 GiB/s | 20 s |
| [128k-calibrated](128k-calibrated/) | 14:30 | booted 13:10 (see below) | skipped (`--pcie-frac` given) | 6.16 GiB/s | 20 s |
| [256k-calibrated](256k-calibrated/) | 14:51 | the one before | skipped (`--pcie-frac` given) | 6.18 GiB/s | 20 s |

Before `128k-calibrated`, its boot ran one 131,072-token benchmark session with `--pcie-frac 0.47` (not part of
this report), host-to-device copy tests and the 131,072-token calibration; before `256k-calibrated`, the
262,144-token calibration. Each session started with the GPU at or below 45 °C.

## Calibration

`tools/calibrate.py` run by hand on each config, which measures and prints without saving (`./setup.sh --calibrate`
runs the same measurement and writes the result into the config). It decodes its own prompts on one engine, visits
each value three times in alternating order, and restarts the engine for the worker count and the expert tier; 669
and 703 seconds. Its outputs are in [`calibration/`](calibration/). Decode tok/s, medians of three (worker counts
and tier settings: mean of two):

| Value | 131,072 | 262,144 |
| --- | ---: | ---: |
| PCIe share 0 | 69.1 | 63.0 |
| PCIe share 0.2 | **73.5** | **66.8** |
| PCIe share 0.35 | 66.1 | 61.1 |
| PCIe share 0.55 (default) | 54.2 | 49.4 |
| PCIe share 0.75 / 0.9 / 1.0 | 46.3 / 44.2 / 39.8 | 42.1 / 39.0 / 36.6 |
| Draft floor 0.3 / 0.5 (default) / 0.7 | **75.5** / 72.1 / 75.1 | 63.7 / 66.5 / **68.3** |
| Re-measured: defaults against the picks | 54.4-54.8 against 69.8-75.1 | 48.9-49.9 against 67.5-69.2 |
| Workers 23 (default) / 15 / 12 / 7 / 6 | **70.3** / 66.4 / 63.1 / 60.6 / 57.3 | **69.1** / 64.8 / 61.6 / 59.1 / 56.0 |
| Expert tier: default / every 1, 80 swaps / every 1, 160 swaps | 76.1 / **79.8** / 78.9 | 76.3 / **79.1** / 78.6 |

At 131,072 tokens the ranges of the draft floors 0.3 and 0.7 overlapped (68.5-76.8 and 74.2-75.4); the tool kept
0.3. The PCIe share falls toward the CPU: with 0.20, 3-5% of the routed experts crossed PCIe during the
benchmark's decode, against 13-22% with the default.

## Method

The unchanged `benchmark.py` and `monitor.py` of the
[RTX 5090 report](../2026-09-30-community-rtx-5090/README.md), then the unchanged `tools/needle_bench.py`, as in the
2026-10-04 report:

```bash
B=bench/results/2026-09-30-community-rtx-5090
.venv/bin/python $B/monitor.py OUT/telemetry.jsonl &
.venv/bin/python $B/benchmark.py --root . --pack ../Strata-data/packs/iq3_s --url http://127.0.0.1:8080 --out OUT
kill %1
.venv/bin/python tools/needle_bench.py --url http://127.0.0.1:8080 --lengths 32k,128k --depths 10,50,90 --out OUT/needles.json
```

At 262,144 tokens: `--targets 4096,32768,128000,250000` and `--lengths 32k,128k,256k`.
[`run-session.sh`](run-session.sh) runs these steps for one session and also saves `sysinfo.txt`, `power.csv` and
the session's engine log (`engine-timing.txt` keeps its timing lines). One warm-up request (excluded), then three
runs at each length in increasing order, greedy, reasoning off, 256 generated tokens. All 42 speed requests read
their whole prompt (zero reused tokens), generated 256 tokens and stopped at the output limit; none failed. No
request preceded the warm-up. Beside the monitor, `nvidia-smi` logged the enforced power limit every 2 s
(`power.csv`). Prompt throughput is the engine's freshly read tokens divided by its prompt time; decode is
generated tokens divided by decode time; TTFT and total are measured by the client over loopback.

## Results

Median **[minimum-maximum]** of three runs. Per-run records: `results.json`; aggregates: `summary.json`.

### 131,072-token context

| Settings | Prompt tokens | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| defaults | 4,096 | 1,865.5 [1,786.2-1,866.4] | 60.9 [55.3-63.6] | 2.229 [2.226-2.322] | 6.403 [6.224-6.927] |
| calibrated | 4,096 | 1,869.6 [1,817.7-1,883.1] | 75.9 [71.8-79.2] | 2.218 [2.203-2.277] | 5.569 [5.414-5.823] |
| defaults | 32,768 | 2,693.3 [2,674.5-2,700.0] | 61.7 [61.2-64.3] | 12.217 [12.188-12.300] | 16.313 [16.256-16.375] |
| calibrated | 32,768 | 2,672.5 [2,665.2-2,694.6] | 81.9 [78.2-86.8] | 12.301 [12.207-12.339] | 15.444 [15.136-15.553] |
| defaults | 128,000 | 2,577.5 [2,574.6-2,584.2] | 58.7 [57.5-59.7] | 49.760 [49.620-49.804] | 54.093 [53.885-54.233] |
| calibrated | 128,000 | 2,567.9 [2,566.8-2,577.2] | 76.6 [71.9-80.7] | 49.936 [49.753-49.955] | 53.087 [53.073-53.494] |

Calibrated against defaults: decode +24.6% / +32.8% / +30.5%, prompt +0.2% / -0.8% / -0.4%, total time -13.0% /
-5.3% / -1.9%.

### 262,144-token context

| Settings | Prompt tokens | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| defaults | 4,096 | 1,783.0 [1,725.2-1,790.6] | 53.7 [51.5-55.1] | 2.332 [2.323-2.407] | 7.059 [6.953-7.352] |
| calibrated | 4,096 | 1,804.5 [1,751.3-1,806.9] | 74.6 [71.0-75.2] | 2.299 [2.296-2.365] | 5.704 [5.679-5.953] |
| defaults | 32,768 | 2,681.4 [2,668.3-2,692.4] | 54.5 [53.8-54.6] | 12.271 [12.225-12.333] | 16.944 [16.888-17.065] |
| calibrated | 32,768 | 2,671.6 [2,662.1-2,681.3] | 71.8 [70.3-73.4] | 12.311 [12.264-12.354] | 15.817 [15.808-15.932] |
| defaults | 128,000 | 2,560.1 [2,556.5-2,578.4] | 53.9 [51.1-54.7] | 50.093 [49.740-50.176] | 54.893 [54.389-55.080] |
| calibrated | 128,000 | 2,551.1 [2,539.7-2,553.1] | 73.4 [71.1-74.8] | 50.261 [50.223-50.488] | 53.839 [53.690-53.889] |
| defaults | 250,000 | 2,402.4 [2,398.7-2,406.7] | 53.1 [52.4-53.2] | 104.232 [104.047-104.384] | 109.014 [108.836-109.237] |
| calibrated | 250,000 | 2,402.8 [2,400.7-2,407.0] | 70.6 [68.0-73.4] | 104.196 [104.022-104.301] | 107.901 [107.489-107.939] |

Calibrated against defaults: decode +38.9% / +31.8% / +36.0% / +32.9%, prompt +1.2% / -0.4% / -0.4% / +0.0%, total
time -19.2% / -6.7% / -1.9% / -1.0%. In both contexts every calibrated decode run was faster than every default
run at the same length. With a 256-token answer, a long prompt's total time hardly moves: the prompt path streams
every expert over PCIe whatever these settings say.

| Folder | Decode cache hit rate | Routed experts over PCIe | MTP drafts accepted | GPU power, busy median |
| --- | --- | --- | --- | ---: |
| 128k-default | 74.5-80.6% | 13.2-16.7% | 137-156 of 205-230 | 174 W |
| 128k-calibrated | 69.9-74.6% | 3.0-3.7% | 146-163 of 237-287 | 174 W |
| 256k-default | 65.8-74.0% | 17.2-21.7% | 141-157 of 209-238 | 174 W |
| 256k-calibrated | 60.5-66.8% | 3.8-4.8% | 131-148 of 160-177 | 174 W |

Memory (system-wide, peaks of the speed suite): VRAM 15,764 / 15,750 / 15,766 / 15,750 MiB of 16,376; RAM used
53.18 / 53.18 / 53.16 / 53.17 GiB; swap 0.1 MiB.

## Against the 2026-10-04 report

Medians; the 2026-10-04 numbers are from that report.

| Context, prompt tokens | Prompt tok/s: 2026-10-04 / defaults / calibrated | Decode tok/s: 2026-10-04 / defaults / calibrated |
| --- | --- | --- |
| 131,072, 4,096 | 1,345.0 / 1,865.5 / 1,869.6 | 51.2 / 60.9 / 75.9 |
| 131,072, 32,768 | 2,505.5 / 2,693.3 / 2,672.5 | 49.9 / 61.7 / 81.9 |
| 131,072, 128,000 | 2,385.8 / 2,577.5 / 2,567.9 | 48.1 / 58.7 / 76.6 |
| 262,144, 4,096 | 1,343.0 / 1,783.0 / 1,804.5 | 43.4 / 53.7 / 74.6 |
| 262,144, 32,768 | 2,457.5 / 2,681.4 / 2,671.6 | 43.3 / 54.5 / 71.8 |
| 262,144, 128,000 | 2,291.6 / 2,560.1 / 2,551.1 | 39.7 / 53.9 / 73.4 |
| 262,144, 250,000 | 2,116.4 / 2,402.4 / 2,402.8 | 40.3 / 53.1 / 70.6 |

This is not a controlled comparison: between the two dates the operating system (Ubuntu 26.04.1 to Gentoo), kernel
(7.0.0-38-generic to 6.18.54), driver (610.57.04 to 615.78.08), engine (0.1.38 to 0.1.41), filesystem (ext4 to
btrfs) and session (desktop to console) all changed. One measured factor: in separate runs on this laptop (not part of
this report), booting with `intel_iommu=on` cost up to 31% prompt throughput with driver 595.104.02 and nothing
measurable with 615.78.08; the Ubuntu install's IOMMU setting was not recorded.

## Correctness and limitations

- All needles were found in every session: six of six at 131,072 tokens (prompt lengths 33,286-33,287 and
  126,638-126,640) and nine of nine at 262,144 (also 251,816-251,817); see `needles.json`.
- The greedy outputs of the calibrated runs were not identical to the default runs' (0 of 21), but neither were
  those of two default sessions with the same driver and configuration (0 of 9; one of them not part of this
  report): this engine's greedy text varies between sessions. Output quality was not evaluated beyond the needles.
- One session per setting; the calibrated sessions shared a boot with other measurements (see above).
- The calibration measures on its own short prompts; its 0.30 / 0.70 draft-floor picks rest on small differences.
- Not tested: other packs, long answers, sampled decoding, thinking, coding-task correctness, a desktop session,
  other PCIe shares between 0 and 0.35 on the benchmark itself.
