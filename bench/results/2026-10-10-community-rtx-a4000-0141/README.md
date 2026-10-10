# Community benchmark on RTX A4000 16 GB (WSL2) + GSQ-RCO IQ3_S

Measured on 2026-10-10 by kgmkm. Single-GPU serve of the official
ISTA-DASLab GSQ-RCO IQ3_S on engine 0.1.41 under WSL2, with experimental
speed projection (ESP) enabled. Companion to the RTX 5060 Ti report in the
same pull request, measured with the same script and model on the same
machine, one day later. Main limitations: one run per condition
(no median/range), TTFT and needle recall not measured, WSL2 + virtiofs
stack rather than native Linux.

## Hardware and software

- GPU and VRAM: NVIDIA RTX A4000 16 GB (16376 MiB), single-GPU serve,
  Ampere (compute capability 8.6). The GeForce RTX 5060 Ti 16 GB in the
  same box was idle (0 MiB) during these runs.
  PCIe Gen4 x4 link: probed 6.5 GB/s host->device, so pcie_frac 0.18
  (the 5060 Ti sibling probed 14.1 GB/s on its x8 link).
- CPU: AMD Ryzen 7 3700X, 8 cores / 16 threads, no AVX-512 (AVX2 expert
  kernels); 7 pool workers.
- Installed RAM: 128 GB host; WSL2 VM 78 GiB. Engine arena ~48 GB.
- Storage: model served from a Windows NVMe drive (KLEVV CRAS C720 2 TB)
  mounted in WSL2 via virtiofs from a Windows partition on the same NVMe
  drive. Engine load 46.84 GiB at 1.92 GiB/s.
- OS: Windows 11 26H2 (build 26300); WSL components 3.0.1.0 with the
  Ubuntu distro running in WSL2 mode
  (kernel 6.18.40.1-microsoft-standard-WSL2), Ubuntu 24.04.2 LTS.
  Driver 616.92, nvidia-smi 615.71.08, CUDA UMD 13.4; CUDA toolkit 13.0
  for the build.
- Strata commit fb58e0db (v0.1.41), source build via wsl-build.py
  (135/135 objects); engine 0.1.41 (src 62eb4907c92a8cb7).
- Background workloads: none; the regular server on the other GPU was
  stopped for this run and restarted afterwards.

## Model and configuration

- Model repository: ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, IQ3_S/
  (3.50 bpw). Shard sizes: 54,817,524,224 B + 28,800,138,432 B;
  mmproj BF16 907,543,008 B. Repository revision was not recorded at
  download time. Vision encoder disabled.
- Custom pack: Strata-data/packs/iq3_s (dense 1.5 GB), MTP rt/ runtime,
  24576-pair expert profile; prepared with
  `setup.py --setup --family qwen --model IQ3_S --experimental-speed-projection on`.
  Same pack, same files as the 5060 Ti sibling run.
- Context 131072; KV int8; expert cache auto (3668 slots, 7.01 GiB VRAM,
  PROFILE-ranked, prefilled in 1.3 s); prefill auto; spec 4 on the
  command line (adapted to spec 6 / mtp_max 4 at runtime); spec_min_p 0.50.
  The MTP draft head read took 4.54 s on this card.
- Reasoning: `reasoning_effort: none` per request; sampling and
  temperature left at defaults.
- ESP enabled: control vector `project`, `per-layer`, layers 4-44
  (41 steered), scale 1.0. A refusal A/B with ESP on/off is included below.
- No manual pool-worker, VRAM-reserve, or calibration tuning.

```text
serve/server.py --engine strata --config strata-iq3_s.json --port 8084
  --host 0.0.0.0 --gpu 0 --fit-max-tokens --api-key <redacted>
```

Full sanitized config: `config-stock.json` (identical to the sibling
report; only the launcher's `--port 8084` / `--gpu 0` differ, so the
regular :8080 server on the 5060 Ti stayed untouched in configuration).

## Method

- `benchmark-iq3s.sh` (packaged, run 2026-10-10 17:59 JST): a warm-up
  ("Say hello.", 8 tokens) plus a token-calibration request, then each
  condition once with a unique prompt (0 reused tokens every time): two
  1100-word generations (story about a lighthouse keeper, factual article
  on tides; max_tokens 1400), two prompt reads (unique nonsense-word
  documents of 3,657 and 18,052 actual tokens; max_tokens 24, reply
  limited to the first word), and one refusal probe (dark-fantasy
  execution scene) sent twice with `experimental_speed_projection` true
  and false (max_tokens 500).
- Timing boundaries: the engine log's own lines
  (`prompt N tokens = R reused + M read in X ms (Y tok/s),
  G generated in Z ms (W tok/s)`); client-side wall clock is also listed
  and includes ~2-4 s of HTTP/python overhead versus the engine sums.
- Engine 0.1.41 caveat: prompt chunks under 1024 tokens partly run on
  the CPU (`STRATA_PREFILL_CPU_SHARE` default), so the 47-48-token prompt
  reads below carry that caveat; long prompts and decode are unaffected.
- Memory: nvidia-smi plus a `/metrics` snapshot after the runs
  (`metrics-stock.json`).

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| decode-story (1100-word lighthouse story) | 48 | 0 | 1400 (finish=length) | 1 | 42.7 (single run) | 41.4 (single run) | not measured |
| decode-explan (1100-word tides article) | 47 | 0 | 1337 (finish=stop) | 1 | 37.5 (single run) | 42.1 (single run) | not measured |
| prompt-4k (unique nonsense-word document) | 3657 | 0 | 3 | 1 | 261.7 (single run) | n/a (3 output tokens) | not measured |
| prompt-20k (same, longer document) | 18052 | 0 | 5 | 1 | 486.9 (single run) | n/a (5 output tokens) | not measured |

Per-request rows: `requests.csv`. Per-condition medians (each of one run):
`measurements.csv`. Engine timing excerpts: `engine-logs/`.

- Draft acceptance: story 712/1263 (56.4%), explan 771/1258 (61.3%).
- Decode expert cache hit rate: 79.0% / 75.1% on the two main runs
  (best 83.0% on the 500-token ESP run).
- ESP refusal A/B on the same dark-fantasy prompt: ESP on generated
  500 tokens (finish=length, scene written to the cap); ESP off stopped
  at 107 tokens (finish=stop) with a refusal text offering a milder
  rewrite. Same qualitative outcome as on the 5060 Ti.
- VRAM free: 457 MiB at measure time.
- No failures, no skipped cases, no client-side retries in these runs.

Sibling comparison (same PR, same model/config, same script): the A4000
decodes about 13% slower than the 5060 Ti (41.4/42.1 vs 48.2/47.7 tok/s)
and reads long prompts about 19-24% slower (261.7/486.9 vs 343.0/598.3
tok/s), consistent with the older architecture and the x4 PCIe link
(6.5 vs 14.1 GB/s probed). Draft acceptance and cache hit rates are
close (56-61% vs 59-62%; 79/75% vs 86/77%).

## Correctness and limitations

- No needle recall test (not measured); vision not tested; no soak run;
  TTFT not measured; no temperature sampling.
- Single run per condition: no median/range and no repeatability
  assessment (in particular not for short-prompt reads under the 0.1.41
  CPU-share default).
- WSL2 + virtiofs: these numbers describe that stack, not native Linux
  on the same card.
- Untested: other quantizations, contexts above 131072, a layer split
  across both cards, `--batch`, calibration sweeps, concurrent requests.
