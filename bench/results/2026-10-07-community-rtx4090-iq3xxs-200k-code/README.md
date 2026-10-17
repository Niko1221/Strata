# Community benchmark: NVIDIA GeForce RTX 4090

Measured on 2026-10-07 by [Dmitry-B](https://github.com/Dmitry-B). This tests Strata 0.1.40.2 with qwen3.8-flash-next-iq3_xxs and a 204800-token context. Prompts are code-explanation text; greedy decoding, a 256-token output cap, three runs per configuration; TTFT measured over streaming. These are synthetic workloads; they do not establish general answer quality. The same sweep with Russian-language prompts is in [2026-10-07-community-rtx4090-iq3xxs-200k-ru](../2026-10-07-community-rtx4090-iq3xxs-200k-ru/README.md).

## Hardware and software

- GPU: NVIDIA GeForce RTX 4090; 23028 MiB reported VRAM; 480.00 W power limit; PCIe bus 00000000:01:00.0; PCIe link speed and width: not measured. GPU clocks were not fixed.
- CPU: AMD Ryzen 9 7950X 16-Core Processor (32 logical CPUs).
- RAM: 46464 MiB installed.
- Storage: models, packs and the expert profile are on a 1.9 TB NVMe drive (ADATA LEGEND 960, ext4, mounted at `/`). No network storage is in the measured path.
- Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic; NVIDIA driver 610.57.04; release 13.4, V13.4.92.
- Strata commit `e8ca9afd03d839d4f8dbbe82dffce7f8a3bafd7a` (tag `v0.1.40.2`), branch `main`; engine 0.1.40.2 **compiled from source on this machine** (`engine/BUILD.json` records `"source": "local"`, `archs: [89]`, nvcc from `/usr/local/cuda`). No ready-made Linux engine is published for this tag, so a source build is the only path here; the same build command is what `./update.sh` runs.
- Background workloads: a dsh/Authentik/Caddy web stack and stock Ubuntu services; the GPU was dedicated to Strata but the operating system was not isolated. This server also serves an interactive agent session; no request from it was issued while the measured runs were running (see *Correctness and limitations*).
- PCIe link speed and width: not measured - `nvidia-smi -q -d BUS` answers "Failed to parse --display/-d flags" on this driver, and `lspci` link status was not collected.
- Background workloads: a dsh/Authentik/Caddy web stack and stock Ubuntu services; the GPU was dedicated to Strata but the operating system was not isolated. This server also serves an interactive agent session; no request from it was issued while the measured runs were running (see *Correctness and limitations*).

## Model and configuration

- Model: qwen3.8-flash-next-iq3_xxs (the Strata server's model name); GGUF filenames, sizes, and modification times are in env.json (no hashes).
- Context 204800; INT8 KV; `--mmap-experts` with the expert cache in `auto` mode; GPU vision - see the config copy below.
- Expert profile: a profile learned online on this model (`--expert-profile … --expert-profile-save … --expert-profile-save-every 10`), reused between restarts. It is a persistent artifact of this machine, not a fresh calibration.
- Experimental speed projection is enabled: `--control-vector-scaled …/Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer`. It was enabled in every earlier report from this PC.
- `--vram-reserve-mib 989`, `--pcie-frac 0.00`, `--pool-workers 10`, `--spec 4` with an MTP draft pack.
- Draft vocabulary subset: `draft_vocab=cyrillic` (the English/code subset plus the whole Cyrillic script, ~106k rows). This is wider than the default `en` subset; draft acceptance on English text was unaffected (72-79% in this run), but it can cost a few percent of decode speed.
- The engine's auto prompt chunk is `prompt chunk auto: 8192 tokens, a 96-slot ring` in the server log. It is unchanged from 0.1.38 through 0.1.40.1, so these numbers are comparable with the earlier reports from this PC.

```text
/home/dgbox/Strata/engine/strata --serve --pack /home/dgbox/Strata-data/packs/iq3_xxs --native /home/dgbox/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf /home/dgbox/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile /home/dgbox/Strata/data/expert-profile-learned.bin --expert-cache auto --prefill auto --spec 4 --mtp /home/dgbox/Strata-data/mtp/rt --max-context 204800 --kv int8 --mmap-experts --vision --vram-reserve-mib 989 --control-vector-scaled /home/dgbox/Strata/data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer --pcie-frac 0.00 --spec-min-p 0.50 --pool-workers 10 --expert-profile-save /home/dgbox/Strata/data/expert-profile-learned.bin --expert-profile-save-every 10
```

Full server config: [config.json](config.json) (was at /home/dgbox/Strata/strata-200k.json; the bearer token in its `mcp_servers` block is replaced with `removed`), environment details: [env.json](env.json).

## Method

- Warm-up: a **full sweep** (4K, 32K, 128K, gen-only) was run first and discarded, then the benchmark's own 4K warm-up. The full warm-up is needed on this machine: the first pass after a service restart reads 4K at 2059-2125 tok/s and reached a 30.9 s TTFT on its first 32K repeat, while the measured pass read 2178-2185 tok/s and 9.38-9.54 s. That difference is the cold expert cache and page cache, not the version.
- Every measured prompt carries a random marker, so the prompt-prefix cache is not reused; the table's reused column reports the actual reused token counts from the engine (0 in every run here).
- Throughput comes from the engine's timing fields (prompt_per_second / predicted_per_second). TTFT is the time to the first non-empty streaming delta, ignoring keep-alives. Total latency (wall) is the whole request time measured at the client.
- temperature=0, reasoning_effort=none, a 256-token output cap (1024 for the generation-only case). The model often stopped early on the repetitive text; actual generated lengths are in the table.
- The expert cache was warmed by the warm-up sweep and earlier sessions; the expert profile state is in the config copy.
- Memory: peak VRAM/RAM sampled every 2 seconds during the measured runs, plus a start snapshot.
- The measurement script is a local script (not part of this repository); it issues the requests, reads the engine's timing lines, and writes `runs.json` / `needles.json` in the format used by the earlier reports from this PC. Available on request.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT s median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 4096-prompt | 3971 | 0 | 256 | 3 | 2180.0 (range 2178.0-2185.2, n=3) | 133.5 (range 132.3-136.8, n=3) | 1.84 s (range 1.84-1.85, n=3) |
| 32768-prompt | 31307 | 0 | 256 | 3 | 3367.3 (range 3310.4-3369.2, n=3) | 133.7 (range 131.8-139.2, n=3) | 9.38 s (range 9.38-9.54, n=3) |
| 131072-prompt | 125011 | 0 | 142 | 3 | 3312.6 (range 3274.6-3321.6, n=3) | 128.1 (range 122.8-131.3, n=3) | 38.03 s (range 37.95-38.49, n=3) |
| gen-only | 163 | 0 | 1024 | 3 | 224.6 (range 222.9-246.4, n=3) | 141.7 (range 140.0-145.1, n=3) | 0.74 s (range 0.67-0.75, n=3) |

- Total latency (client, wall): 4096-prompt - 3.75 s (range 3.7-3.77, n=3); 32768-prompt - 11.31 s (range 11.28-11.37, n=3); 131072-prompt - 39.1 s (range 39.04-39.8, n=3); gen-only - 7.89 s (range 7.79-8.04, n=3).
- Draft acceptance (accepted/total, all runs of a case): 4096-prompt 495/690; 32768-prompt 501/661; 131072-prompt 285/381; gen-only 2063/2629.
- Memory: {"start_snapshot": {"vram_used_mib": 22026, "ram_used_kib": 5816984}, "peak_vram_used_mib": 22026, "peak_ram_used_kib": 6570944, "note": "peak = the maximum of the 2-second samples taken during the measured runs"}
- Every run with its draft statistics (accepted/total): [runs.json](runs.json).
- Recall check (needle): [needles.json](needles.json).

## Comparison with engine 0.1.40 on the same PC

Same PC, same model, same server config, same launch command, same `expert-profile-learned.bin`, same prompt chunk (8192). Only the Strata version changed: 0.1.40 (commit `1735d64`) -> 0.1.40.2, both compiled from source with the same CUDA 13.4 for `sm_89`. The 0.1.40 numbers are in [runs-0.1.40-baseline.json](runs-0.1.40-baseline.json) (measured 2026-10-06).

A control pair is included as well: [runs-0.1.40.1-control.json](runs-0.1.40.1-control.json) is a full repeat of the same sweep on **the identical engine binary** (0.1.40.1 = commit `82f46a8`, which changed only the Python server; `engine/strata` was not rebuilt, its md5 is the same as in the 0.1.40 baseline). It measures the noise band of this method on this machine: prompt -1.1…-2.0%, decode -3.1…-5.6%. Anything inside that band is not a change.

| Configuration | Prompt tok/s 0.1.40 -> 0.1.40.2 | Decode tok/s 0.1.40 -> 0.1.40.2 | TTFT s 0.1.40 -> 0.1.40.2 | Control (same binary) prompt / decode |
| --- | --- | --- | --- | --- |
| 4096-prompt | 1978.6 -> 2180.0 (+10.2%) | 130.1 -> 133.5 (+2.6%) | 2.04 -> 1.84 | -1.8% / -5.0% |
| 32768-prompt | 3136.3 -> 3367.3 (+7.4%) | 136.4 -> 133.7 (-2.0%) | 10.07 -> 9.38 | -2.0% / -3.4% |
| 131072-prompt | 3118.5 -> 3312.6 (+6.2%) | 124.1 -> 128.1 (+3.2%) | 40.38 -> 38.03 | -1.1% / -5.6% |
| gen-only | 212.5 -> 224.6 (+5.7%) | 143.2 -> 141.7 (-1.0%) | 0.79 -> 0.74 | -6.6% / -3.1% |

- **Prompt throughput is the finding here: +5.7…10.2% at every length, with TTFT lower by 0.05-2.35 s.** The prompt column is the comparable one - it comes from the engine's own timing and its ranges are tight in both versions (3082-3123 and 3275-3322 tok/s at 131072-prompt), while the identical-binary control moved it by at most 2%. The gain is uniform across lengths; its cause is not identified by these measurements. An earlier draft of this report attributed it to the prompt stager's wait change (#1057), but the 0.1.40.3 docs measure the opposite: sleeping stager waits read prompts 5-6% slower on a Ryzen 9 7940HS + RTX 4070 laptop (while whole-machine CPU use falls from 77-90% to 21-25%), and on a desktop RTX 5070 spinning is 1.2% faster. #1057 is a CPU-load feature, not a speed feature, so it is removed here as an explanation.
- **Decode: no measurable change.** All four cases stay inside the control band (-5.6…-3.1% for the same binary), so the release's +1.3…2.2% on an RTX 5070 is not resolvable on this card. The F4 verify windows (`STRATA_MMVQ_IL`, on by default for RTX 30 and newer, bit-identical output) did not change decode here.
- Draft acceptance is unchanged in substance (75.2% -> 71.7% at 4096-prompt, 77.1% -> 78.5% at gen-only); the 131072-prompt totals differ because the answers ended early at different points (256/256/142 tokens in 0.1.40, 163/142/142 here).
- Recall: 6/6 at 32K and 128K in both versions.
- Longer trend on this PC: [2026-10-03](../2026-10-03-community-rtx4090-iq3xxs-200k-code/README.md) (0.1.38) -> [2026-10-04](../2026-10-04-community-rtx4090-iq3xxs-200k-code/README.md) (0.1.39) -> this report (0.1.40.2). Prompt throughput at 131072 tokens: 2962 -> 3085 -> 3313 tok/s; the 0.1.40 step in between (3119 tok/s) is in runs-0.1.40-baseline.json.

## Correctness and limitations

- Speed measurements do not establish general answer quality. The recall check passed 6/6 at 32K and 128K across depths 10/50/90; see needles.json.
- The GPU was not fully isolated: background services may have added small noise.
- Prompts are synthetic (repeated text with a random marker); real workloads will show different prefix reuse and draft acceptance.
- This server also serves an interactive agent session. No request from it was issued during the measured runs. The discarded warm-up sweep is not comparable in any case: its first 32K repeat shows a 30.9 s TTFT against 9.4 s in the measured pass.
- The engine is compiled from source with CUDA 13.4, while the release binaries are built with CUDA 13.0. Comparisons inside this report are unaffected (both versions were built the same way on this machine), but they are not a comparison against the published Linux binaries - there are none for this tag.
- Opt-ins from this release were **not** tested here and are not part of these numbers: `STRATA_PREFILL_CPU_SHARE`, `STRATA_IO_PREFETCH` / `STRATA_IO_PF_STAGE`, `pin=N` (docs/RESEARCH_RUNS.md), `STRATA_SPEC_GUMBEL`, `STRATA_MMVQ_IL=0`.
