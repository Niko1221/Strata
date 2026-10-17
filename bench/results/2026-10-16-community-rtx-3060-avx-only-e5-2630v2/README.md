# Community benchmark on RTX 3060 12 GB with an AVX-only Xeon E5-2630 v2 (no AVX2/FMA), DDR3, PCIe Gen2 x4

Measured on 2026-10-16 by @Norhiro. Strata built with `STRATA_ISA_FLOOR=avx` (docs/INSTALL.md, "Older CPUs (experimental)"), run on a real AVX-only CPU, which the docs list as not yet tested. Main limitation: one machine, one quantization (IQ2_XS) and contexts up to 40,960 tokens; the GPU sits behind a PCIe Gen2 x4 link, so expert misses and long prompts are bound by that link and by DDR3, not by the GPU. Headline: decode about 17 tok/s on NUMA node 1 alone and 24-26 tok/s with both sockets, HumanEval 0-39 37/40 base and 35/40 plus, 9/9 needles to 32K, 65.7 minutes of soak without a failure, and no SIGILL anywhere.

## Hardware and software

- GPU: ['NVIDIA GeForce RTX 3060'], 12288 MiB; driver 610.57.04
- PCIe: 0000:81:00.0: max 16.0 GT/s PCIe x16, current 2.5 GT/s PCIe x4, numa_node 1; 0000:80:00.0: max 5.0 GT/s PCIe x4, current 2.5 GT/s PCIe x4, numa_node 1
- GPU power limit: see `T0-env/environment.txt` (nvidia-smi query)
- CPU: 2 x Intel Xeon E5-2630 v2 (Ivy Bridge, 6 cores / 12 threads each, AVX + F16C, no AVX2/FMA), 2 NUMA nodes; the engine bound to node 1 (the GPU's node) unless a test says otherwise
- RAM: 251.8 GiB DDR3; read bandwidth measured 30.5 GB/s on node 1 (OpenMP stream read, 12 threads)
- Storage: Solidigm SSDPFKNU512GZ NVMe (models and the n-gram table, btrfs), 8 GT/s x4 behind a PLX PEX 8747 switch card; 0 PCIe errors this boot (see `T0-env/environment.txt`)
- OS: Linux 7.2.5-200.fc44.x86_64
- Strata: commit 296b59d2, engine 0.1.44.1, source build (CUDA 13.3 from pip wheels, g++-14, sm_86, `-DSTRATA_ISA_FLOOR=avx`); engine SHA-256 in `T0-env/environment.txt`
- Background workloads: the GPU's other services were released for every test (a GPU lease); otherwise an idle GNOME desktop (gnome-shell, gnome-remote-desktop, a terminal; about 0.4 GB of VRAM before the engine started, load average 0.9). GPU power limit 170 W (default).
- Network: disconnected during the tests

## Model and configuration

- ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, revision ed59f92, IQ2_XS (`Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-0000{1,2}-of-00002.gguf`); SHA-256 of every file in `manifest/`
- Context 40,960; KV int8; expert cache auto; prefill auto; MTP spec 4 (min p 0.5); thinking off; temperature 0; setup's settings otherwise

```text
{
 "exe": "<WORKDIR>/experiments/03_run/Strata/engine/strata",
 "args": [
  "--pack",
  "<WORKDIR>/experiments/03_run/Strata-data/packs/iq2_xs",
  "--native",
  "<WORKDIR>/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf",
  "--ple-gguf",
  "<WORKDIR>/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf",
  "--expert-profile",
  "<WORKDIR>/experiments/03_run/Strata/data/expert-profile.bin",
  "--expert-cache",
  "auto",
  "--prefill",
  "auto",
  "--spec",
  "4",
  "--spec-min-p",
  "0.5",
  "--mtp",
  "<WORKDIR>/experiments/03_run/Strata-data/mtp/rt",
  "--max-context",
  "40960",
  "--kv",
  "int8"
 ],
 "cwd": "<WORKDIR>/experiments/03_run/Strata",
 "tokenizer": "<WORKDIR>/experiments/03_run/Strata-data/packs/iq2_xs/tokenizer",
 "model_name": "qwen3.8-flash-next-iq2_xs",
 "log": "<WORKDIR>/experiments/06_test/results/2026-10-16/T2-speed/engine.log",
 "lib_dirs": [
  "<WORKDIR>/toolchain/.venv/lib/python3.12/site-packages/nvidia/cu13/bin",
  "<WORKDIR>/toolchain/.venv/lib/python3.12/site-packages/nvidia/cu13/lib64"
 ],
 "port": 8080,
 "gpu": 0,
 "gpus_asked": true
}
```

## Method

- T1: `tools/test_*.py` one module at a time (`python -m unittest tools.<module>`), and the CTest suite of a separate build of the same source with `-DSTRATA_BUILD_TESTS=ON`
- T2: `tools/strata_bench.py --long` (3 runs per test), twice: cold (model files dropped from the page cache with posix_fadvise DONTNEED, no sudo) and warm (started again right after)
- T3: `tools/ab_engine.py`, arms both sockets / NUMA node 1, 3 rounds alternating
- T4: `tools/needle_bench.py --lengths 4k,16k,32k --depths 10,50,90`
- T5: the checks of the 2026-10-09 RTX 3060 report (HumanEval 0-39 with EvalPlus 0.3.1, 12 Japanese questions, 8 tool calls), same prompts and limits, non-streamed requests
- T6: soak, back-to-back requests (every 4th a 17x19 check, others fresh ~512/4K/32K prompts, 512 tokens), GPU and RAM sampled every 10 s
- T7 (optional): `tools/quant_quality.py` twice on IQ2_XS and once on UD-IQ4_XS

## Results

### Speed (T2)

**cold**

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 17.1 (15.1-19.4) | 2.18 (1.94-2.56) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 178.3 (177.7-179.0) | not measured | 22.59 (22.50-22.66) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 342.1 (341.8-342.3) | not measured | 46.99 (46.97-47.03) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 352.4 (351.8-352.4) | not measured | 91.03 (91.03-91.18) |
| agent session, follow-up turns | 14561 (14559-14565) at the last turn | see below | up to 64 | 3 | see below | 20.5 (15.1-26.6) | 21.89 (20.91-22.36) |

**warm**

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 17.7 (15.6-19.0) | 2.19 (1.88-2.56) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 177.8 (177.2-177.9) | not measured | 22.64 (22.64-22.73) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 341.9 (341.9-342.4) | not measured | 47.02 (46.95-47.02) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 351.0 (350.1-351.8) | not measured | 91.40 (91.19-91.61) |
| agent session, follow-up turns | 14581 (14560-14594) at the last turn | see below | up to 64 | 3 | see below | 20.1 (15.2-27.5) | 21.88 (20.89-22.24) |

Page cache before the cold run: page cache dropped for 57 files: Cached 164.8 GiB -> 4.9 GiB

### NUMA placement A/B (T3)

| Request | Arm | Rounds | Decode tok/s median (range) | Prompt tok/s median (range) |
| --- | --- | ---: | --- | --- |
| chat0 | both | 3 | 26.1 (25.5-26.4) | 23.8 (22.5-24.1) |
| chat0 | node1 | 3 | 17.3 (16.9-17.5) | 15.8 (15.6-16.4) |
| chat1 | both | 3 | 24.2 (24.2-26.5) | 23.9 (23.3-24.0) |
| chat1 | node1 | 3 | 18.3 (17.6-19.2) | 16.5 (15.8-17.1) |
| chat2 | both | 3 | 25.7 (25.7-26.5) | 22.4 (22.0-22.7) |
| chat2 | node1 | 3 | 19.0 (19.0-20.0) | 15.2 (15.0-15.9) |
| long | both | 3 | 25.0 (25.0-26.3) | 257.5 (257.4-258.1) |
| long | node1 | 3 | 18.1 (18.0-18.1) | 258.4 (258.2-258.5) |

## Correctness and limitations

### Unit tests (T1)

- Python: 104 of 111 modules passed. `tools.test_setup_choices` (errors=4) and `tools.test_setup_engine_hash` (failures=2) fail the same way on the unmodified upstream source at 296b59d on this host. Five setup modules hit the runner's 600 s timeout in this offline run: `tools.test_setup_golden`, `tools.test_setup_lowram`, `tools.test_setup_rerun_size`, `tools.test_setup_risk`, `tools.test_setup_unsloth`. Run again on 2026-10-17 with the same runner, the last four passed in 0-17 s (also inside `unshare -n`) and `test_setup_golden` failed in 8 s with failures=8, as it does on upstream here. I could not reproduce the hang, so its cause is unknown; these modules are mocked setup tests and do not touch the engine.
- CTest: 117 of 119 passed (98%). All AVX paths passed (`router_dot_parity_avx`, `iq_avx2_parity`, `iq_avx1_parity`, `q8k_quant_parity_avx`, `isa_guard_check`); no SIGILL. The two failures are environmental: `ple_parity` needs the Q2_0 model (not on this machine; Q2_0 needs AVX-512 VNNI), and `platform_memory_test` fails only `lock_resident` because `ulimit -l` is 8 MB.

```text
	 62 - ple_parity (Failed)
	 70 - platform_memory_test (Failed)
```

### Long-context recall (T4)

| Length | Depth % | Prompt tokens | Found | Seconds |
| --- | ---: | ---: | --- | ---: |
| 4k | 10 | 3327 | True | 23.3 |
| 4k | 50 | 3327 | True | 22.9 |
| 4k | 90 | 3328 | True | 23.1 |
| 16k | 10 | 15157 | True | 48.3 |
| 16k | 50 | 15155 | True | 47.6 |
| 16k | 90 | 15155 | True | 47.7 |
| 32k | 10 | 32778 | True | 92.2 |
| 32k | 50 | 32778 | True | 92.1 |
| 32k | 90 | 32779 | True | 92.1 |

Found 9 of 9. A needle test measures recall on these inputs, not overall quality.

### Code, Japanese and tool checks (T5)

| Machine | Code evaluated | Base pass | Base+plus pass | Japanese | Tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| this machine (E5-2630 v2, AVX only) | 40 | 37/40 | 35/40 | 11/12 | 8/8 |
| 2026-10-09 RTX 3060 report (i7-12700, AVX2) | 40 | 36/40 | 35/40 | 11/12 | 8/8 |

Code answers that reached the 1,024-token cap: 0. Request errors: 0. Differences from the 3060 report: context 40,960 here (131,072 for its code tasks, 8,192 for its small tasks), non-streamed requests, another CPU's rounding (IQ answers are not byte-reproducible across machines). HumanEval IDs 0-39 only, not the full 164-task score.

### Soak (T6)

- 65.7 minutes, 116 requests, 0 failed; known-answer checks 29/29 correct
- GPU max temperature 67.0 C, max power 115.49 W, max VRAM 11555.0 MiB; engine RSS first/last/max [36015.9, 36572.9, 36671.8] MiB; lowest MemAvailable 210.24 GiB

| Kind | Requests | Prompt tokens | Decode tok/s 1st half | 2nd half | Prompt tok/s 1st half | 2nd half |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| check | 29 | 29 | 5.4 | 5.8 | 14.6 | 14.4 |
| p32000 | 29 | 31932 | 24.9 | 24.6 | 349.5 | 349.6 |
| p4096 | 29 | 4039 | 26.5 | 25.75 | 177.3 | 177.45 |
| p512 | 29 | 451 | 26.0 | 25.7 | 116.4 | 118.1 |

### Quant yardstick (T7, optional)

```text
reference: <WORKDIR>/experiments/06_test/results/2026-10-16/T7-quant-quality/iq2_xs-run1.tsv
run                                      segment         pos    top1       KL   KL p99     PPL PPL ref
iq2_xs-run2.tsv                          code            999 100.00%  0.00000      nan   2.087   2.087
iq2_xs-run2.tsv                          docs            999 100.00%  0.00000      nan  10.724  10.724
iq2_xs-run2.tsv                          english         999 100.00%  0.00000      nan   4.635   4.635
iq2_xs-run2.tsv                          multilingual    999 100.00%  0.00000      nan   2.785   2.785
iq2_xs-run2.tsv                          agent           999 100.00%  0.00000      nan   6.664   6.664
iq2_xs-run2.tsv                          all            4995 100.00%  0.00000   0.0000   4.539   4.539
run1 and run2: byte-identical
```

IQ2_XS against UD-IQ4_XS (not the official Q8_0 reference):

```text
reference: <WORKDIR>/experiments/06_test/results/2026-10-16/T7-quant-quality/ud-iq4_xs-run1.tsv
run                                      segment         pos    top1       KL   KL p99     PPL PPL ref
iq2_xs-run1.tsv                          code            999  90.29%  0.20423      nan   2.087   2.068
iq2_xs-run1.tsv                          docs            999  67.87%  0.39906      nan  10.724  10.128
iq2_xs-run1.tsv                          english         999  75.48%  0.29378      nan   4.635   4.309
iq2_xs-run1.tsv                          multilingual    999  87.29%  0.15097      nan   2.785   2.736
iq2_xs-run1.tsv                          agent           999  80.68%  0.49630      nan   6.664   4.980
iq2_xs-run1.tsv                          all            4995  80.32%  0.30887   3.4172   4.539   4.149
```

### Files in this folder

- `T0-env/`: machine facts, SHA-256 of the 57 model, pack and engine files
- `T1-unit/`: per-module Python test logs and `summary.tsv`, CTest log and JUnit XML
- `T2-speed/`: `strata_bench.py` output (README, `results.json`) for cold and warm, engine start-up excerpt
- `T3-ab-numa/`, `T4-needle/`, `T5-quality/` (answers, EvalPlus grades), `T6-soak/` (every request, 10 s GPU/RAM samples), `T7-quant-quality/` (comparison tables; the 5 MB per-token TSVs are left out)
- `scripts/`: the test scripts used (`t0_env.sh` ... `t7_quant_quality.sh`, `summarize.py`, `soak.py`, ...)

Local paths are replaced with `<WORKDIR>`, the user name, GPU UUID and link-local addresses are removed.

### Not tested

- Other quantizations than IQ2_XS (UD-IQ4_XS only in T7); Q2_0 cannot run without AVX-512 VNNI
- Contexts above 40,960; images; `--batch` / concurrent clients; Windows
- llama.cpp on the same machine
- Nothing was stopped. `--clients`, the 4-bit (UD-IQ4_XS) speed and quality runs, and the full HumanEval 164 were skipped.
