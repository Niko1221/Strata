# STRATA_DENSE_F32=1 on a GTX 1080 Ti: cuBLAS 12.8.4.1 passes gemm_f32_parity bitwise; prompt +7% / +10%

Measured on 2026-10-16. An A/B of the opt-in `STRATA_DENSE_F32=1` (#2178, in 0.1.44) on a second sm_61 card,
on a cuBLAS the route had not been checked against, plus the parity run the docs ask for before enabling it.
Speed only; no quality suite. The main limitation: short and 7K prompts only, one card, one cuBLAS.

## Hardware and software

- GPU: NVIDIA GeForce GTX 1080 Ti 11 GiB (GP102, sm_61, PCIe Gen3 x16, on NUMA node 1); ~0.7 GiB of it held by the
  desktop (Wayland session) during the runs
- CPU: 2x Intel Xeon E5-2696 v4 (44c/88t, no AVX-512), 125.8 GiB RAM; the 1080 Ti also drives the desktop
- OS/driver/toolkit: CachyOS Linux LTS 6.18.55, driver 580.178.04, CUDA 12.8 toolkit at /opt/cuda-12.8
  (**cuBLAS 12.8.4.1**, the library the engine loads)
- Strata: `main` at `296b59d2` family, engine 0.1.44.1 source build, `-DCMAKE_CUDA_ARCHITECTURES=61
  -DSTRATA_EXPERIMENTAL_SM60=ON`; the same binary in both arms
- Power/clocks: the card runs a −200 MHz clock offset (1733 MHz under load, 1278 idle); every arm started at 57-58 C

## Model and configuration

Qwen3.8-Flash-Next IQ3_S (setup's pack, native experts, the shipped expert profile and MTP head), both arms
identical except `env.STRATA_DENSE_F32`: `0` (arm A, the default) and `1` (arm B). Server config
(`strata-iq3_s-bench.json`): `--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.70 --mtp
~/Strata-data/mtp/rt --max-context 262144 --kv int8 --kv-resident 32768 --conversation-cache-mib 8192
--vram-reserve-mib 1024 --prompt-cache-tail --prompt-cache-every 4096 --pcie-frac 0.20`, `env` also has
`STRATA_MTP_ADAPTIVE=1`. The engine picked a **4,608-token prompt chunk** (the 11 GiB card; the P40 report's 24 GiB
card ran 8,192). `--spec 4` and the conversation cache were on, so decode numbers include MTP verification.

Both arms were bound to the GPU's NUMA node the way [#2684](https://github.com/Niko1221/Strata/issues/2684)
describes (`numactl --preferred=1 --cpunodebind=1`); the desktop stayed on node 0. One instance at a time,
alternated A B A B (two rounds per arm), each instance a fresh server start and model load (untimed), one warm-up
request before the bench, 3 requests per case per round with a random nonce so nothing was reused.

## Method

`bench.py` of the P40 report ([2026-10-12-p40-sm61-kernels](../2026-10-12-p40-sm61-kernels/README.md)):
a ~565-token prompt with 200 new tokens and a ~7,047-token prompt with 60, greedy, `reasoning_effort` none,
temperature 0. Speeds are the engine's own log lines; the runner script and the four per-arm transcripts are in
this folder (`ab-arm.sh`, `arm-*.log`). Before the A/B, `gemm_f32_parity --bench` was run with the engine's own
toolkit ([gemm-f32-parity-cuda12.8.txt](gemm-f32-parity-cuda12.8.txt)).

## Results

**Parity, cuBLAS 12.8.4.1**: all **5,201 chunk sizes bitwise equal**, exit 0 — the same count as the P40's
cuBLAS 12.4.5.8 and the maintainers' cuBLAS 12.9 run on a P100 (#2178). The tool's kernel timings on this card
(FP16 operands → FP32 copies, conversions included): qkv T2048 26.1→12.7 ms (2.05x), T7040 92.7→48.1 (1.93x);
qsa query T4096 70.6→36.0 (1.96x), T7041 121.7→60.7 (2.00x); gdn gate T2048 15.3→8.9 (1.72x);
gdn out T7040 63.9→29.9 (2.14x). The 7,040/7,041 kernel cliff the P40 hit is there too (query, gate).

**Engine, medians of 6 requests per arm** (per-request values in the `arm-*.log` transcripts):

| Arm | short ~565 tok: prompt tok/s | long ~7,047 tok: prompt tok/s | short decode tok/s | long decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| A: `STRATA_DENSE_F32=0` | 240.4 (232-244) | 341.4 (331-356) | 29.6 (27-32) | 30.9 (27-34) |
| B: `STRATA_DENSE_F32=1` | 257.0 (249-260) | 375.2 (359-404) | 30.6 (28-34) | 29.9 (28-32) |
| delta | **+6.9%** | **+9.9%** | noise | noise |

Per round, long prompt: A 338 / 344 → B 360 / 383 tok/s. The route's startup notice appears once in each B engine
log and in no A log. Decode is unchanged within the run-to-run spread, as expected: the switch replaces the prompt
GEMMs, not the decode GEMVs.

Why the gain is smaller than the P40's 360-432 → 464-474: this card's prompt chunk is 4,608 tokens (not 8,192),
and cuBLAS 12.8's FP16-operand kernels sit on the fast side at several of the sizes a 4,608-token chunk runs
(query T2048 0.98x, gate T4096 1.10x, qsa out ~1.1x in the parity bench), so less GEMM time is recoverable here.
VRAM: ~10.3 GiB used under load in both arms; the FP32 operand copies fit inside the existing reserve.

## Correctness and limitations

- `gemm_f32_parity` is bitwise on this cuBLAS; **no answer-identity check was run** in this A/B (the P40 report's
  byte-identical `equal_check.py` comparison was not repeated here; with the parity bitwise, the route's outputs
  are the same bits as the FP16 call at every chunk size it takes).
- Not tested: prompts above 7K tokens, the 32K/128K sweep, needle or quality checks, Windows, CUDA 12.9 or 13
  toolkits on this card, `--batch`, other models or quants.
- Decode carries MTP (`--spec 4`, adaptive); its spread (27-34 tok/s) is wider than the arm difference.
- The background desktop (Wayland compositor, a browser) ran unchanged throughout; no other GPU compute process.
