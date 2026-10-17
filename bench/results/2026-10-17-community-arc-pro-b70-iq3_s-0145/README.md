# Community benchmark on Intel Arc Pro B70 32 GB + GSQ-RCO IQ3_S

Measured on 2026-10-17 by kgmkm. Single-GPU serve of the official
ISTA-DASLab GSQ-RCO IQ3_S on engine 0.1.45 (SYCL AOT build for
Battlemage), with experimental speed projection (ESP) enabled.
Main limitations: one run per condition (no median/range), TTFT and
needle recall not measured, VRAM reserve below the engine's own
recommendation (LOW warning active during the runs).

## Hardware and software

- GPU and VRAM: Intel Arc Pro B70 32 GB (Battlemage G31), single-GPU
  serve. A second card (Intel Arc B580) is present and drives the
  desktop only; the engine is pinned to the B70 with
  `ONEAPI_DEVICE_SELECTOR=level_zero:1`. Engine PCIe probe
  12.4 GB/s host->device, so pcie_frac 0.34. lspci link
  capabilities need root on this box, so the negotiated link
  width is not verified.
- CPU: AMD Ryzen 7 3700X, 8 cores / 16 threads, no AVX-512 (AVX2 expert
  kernels); 7 pool workers on logical processors 1-7, host thread on 0.
- Installed RAM: 121 GiB. Engine arena ~46.8 GiB (47962 MiB);
  experts stream from the GGUF on demand (`--stream-experts`, no host
  arena for experts).
- Storage: NVMe SSD (594 GB partition, 398 GB free). MTP draft files
  read at 2286 MiB/s; token embedding mapped in 0.3 s.
- OS: Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic, Intel `xe` kernel
  driver, Intel oneAPI 2026.2. Desktop is XFCE with the display on
  the B580, so the B70 carries no display load.
- Strata commit 477ac13c (v0.1.45), SYCL AOT build (`AOT=bmg-g31`,
  `BUILD_DIR=build-sycl-aot`, `BUILD EXIT 0`, `errors: 0`); engine
  0.1.45 as reported by `/metrics` and the engine log.
- Background workloads: XFCE desktop on the B580 plus normal desktop
  processes; the server had served requests earlier the same day
  (expert cache warm), then idled before the benchmark.

## Model and configuration

- Model repository: ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, IQ3_S/
  (3.50 bpw). Shard sizes: 54,817,524,224 B + 28,800,138,432 B.
  Repository revision was not recorded at download time. Vision
  encoder disabled.
- Custom pack: Strata-data/packs/iq3_s (dense 1.5 GB, native serve,
  302 canonical tensors skipped), MTP rt/ runtime, 24576-pair expert
  profile (`ja-en-mixed.bin`); prepared with
  `setup.py --setup --family qwen --model IQ3_S --experimental-speed-projection on`.
- Context 32768; KV int8; expert cache 13536 slots, 25.80 GiB VRAM
  (282 per layer, per-layer blobs); prefill 4096; VRAM reserve 500 MiB;
  spec 4 on the command line (adapted to spec 6 / mtp_max 4 at
  runtime); spec_min_p 0.50.
- Reasoning: `reasoning_effort: none` per request; sampling and
  temperature left at defaults.
- ESP enabled: control vector `project`, `per-layer`, layers 4-44
  (41 steered), scale 1.0. A refusal A/B with ESP on/off is included below.
- `fit_max_tokens: true` (top-level config) so over-long requests are
  clamped instead of rejected with 400.

```text
serve/server.py --engine strata --config strata-iq3_s.json --port 8080
  --host 127.0.0.1 --api-key <redacted>
```

Full sanitized config: `config-stock.json`. Engine start lines:
`engine-start-stock.txt`. The sanitized files replace the home
directory with `<home>` and the API key with `<redacted>`; no other
values were changed.

## Method

- `benchmark-iq3s.sh` (packaged, run 2026-10-17 23:15 JST, `KEY`
  passed via environment, never stored): a warm-up ("Say hello.",
  8 tokens), then each condition once with a unique prompt
  (0 reused tokens every time): two 1100-word generations (story about a
  lighthouse keeper, factual article on tides; max_tokens 1400), two
  prompt reads (unique nonsense-word documents of 4,021 and 19,880 actual
  tokens; max_tokens 24, reply limited to the first word), and one
  refusal probe (dark-fantasy execution scene, Japanese) sent twice with
  `experimental_speed_projection` true and false (max_tokens 500).
- Timing boundaries: the engine log's own lines
  (`prompt N tokens = R reused + M read in X ms (Y tok/s),
  G generated in Z ms (W tok/s)`); client-side wall clock is also listed
  and includes HTTP/python overhead versus the engine sums.
- Memory: a `/metrics` snapshot after the runs
  (`metrics-stock.json`). Note the snapshot's host-side GPU fields name
  the display card (Arc B580); the serving card is the B70, as the
  engine log's 25.80 GiB cache and PCIe probe show.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| decode-story (1100-word lighthouse story) | 48 | 0 | 1400 (finish=length) | 1 | 56.6 (single run) | 27.8 (single run) | not measured |
| decode-explan (1100-word tides article) | 47 | 0 | 1299 (finish=stop) | 1 | 68.8 (single run) | 38.0 (single run) | not measured |
| prompt-4k (unique nonsense-word document) | 4021 | 0 | 5 | 1 | 421.3 (single run) | n/a (5 output tokens) | not measured |
| prompt-20k (same, longer document) | 19880 | 0 | 4 | 1 | 1004.5 (single run) | n/a (4 output tokens) | not measured |

Per-request rows: `requests.csv`. Per-condition medians (each of one run):
`measurements.csv`. Engine timing lines are transcribed there; engine
start lines: `engine-start-stock.txt`.

- Draft acceptance: story 617/1263 (48.9%), explan 621/1208 (51.4%).
- Decode expert cache hit rate: not recorded in this run's log lines.
- ESP refusal A/B on the same dark-fantasy prompt: both ESP on and ESP
  off generated 500 tokens (finish=length, scene written to the cap);
  no refusal on either side this time, unlike the 0.1.41 RTX 5060 Ti
  report where ESP off refused at 106 tokens.
- VRAM free: 211 MiB at measure time with a LOW warning active
  (`requests may stall`, engine suggests `--vram-reserve-mib 801`).
- No failures, no skipped cases, no client-side retries in these runs.

## Correctness and limitations

- No needle recall test (not measured); vision not tested; no soak run;
  TTFT not measured; no temperature sampling (one 54 C / 41 W idle
  snapshot in `/metrics` only).
- Single run per condition: no median/range and no repeatability
  assessment.
- VRAM reserve 500 MiB is below the engine's own recommendation; the
  LOW warning was active, so these numbers describe that pressured
  configuration, not a headroom-safe one.
- The story decode (27.8 tok/s) is slower than the explanation decode
  (38.0 tok/s) on lower draft acceptance (48.9% vs 51.4%); output text
  differs, so this is not a controlled decode comparison.
- Untested: other quantizations, contexts above 32768, a layer split
  across the B580, `--batch`, calibration sweeps, concurrent requests.
