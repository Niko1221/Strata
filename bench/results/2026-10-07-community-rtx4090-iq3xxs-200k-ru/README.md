# Community benchmark: NVIDIA GeForce RTX 4090 (Russian-language prompts)

Measured on 2026-10-07 by [Dmitry-B](https://github.com/Dmitry-B). This tests Strata 0.1.40.2 with qwen3.8-flash-next-iq3_xxs and a 204800-token context. Prompts are Russian prose; greedy decoding, a 256-token output cap, three runs per configuration; TTFT measured over streaming. These are synthetic workloads; they do not establish general answer quality. The same sweep with code-explanation prompts is in [2026-10-07-community-rtx4090-iq3xxs-200k-code](../2026-10-07-community-rtx4090-iq3xxs-200k-code/README.md), which also carries the full method description.

This report exists because the Russian variant behaves differently from the English/code one on the same machine: the same prompts are tokenized ~1.4x denser, draft acceptance is much lower, and decode is ~25-30% slower. Those differences are the point of publishing the two variants side by side.

## Hardware and software

- GPU: NVIDIA GeForce RTX 4090; 23028 MiB reported VRAM; 480.00 W power limit; PCIe bus 00000000:01:00.0; PCIe link speed and width: not measured - `nvidia-smi -q -d BUS` answers "Failed to parse --display/-d flags" on this driver, and `lspci` link status was not collected. GPU clocks were not fixed.
- CPU: AMD Ryzen 9 7950X 16-Core Processor (32 logical CPUs).
- RAM: 46464 MiB installed.
- Storage: models, packs and the expert profile are on a 1.9 TB NVMe drive (ADATA LEGEND 960, ext4, mounted at `/`). No network storage is in the measured path.
- Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic; NVIDIA driver 610.57.04; release 13.4, V13.4.92.
- Strata commit `e8ca9afd03d839d4f8dbbe82dffce7f8a3bafd7a` (tag `v0.1.40.2`), branch `main`; engine 0.1.40.2 **compiled from source on this machine** (`engine/BUILD.json` records `"source": "local"`, `archs: [89]`, nvcc from `/usr/local/cuda`). No ready-made Linux engine is published for this tag, so a source build is the only path here.
- Background workloads: a dsh/Authentik/Caddy web stack and stock Ubuntu services; the GPU was dedicated to Strata but the operating system was not isolated. This server also serves an interactive agent session; no request from it was issued while the measured runs were running (see *Correctness and limitations*).

## Model and configuration

- Model: qwen3.8-flash-next-iq3_xxs (the Strata server's model name); GGUF filenames, sizes, and modification times are in env.json (no hashes).
- Context 204800; INT8 KV; `--mmap-experts` with the expert cache in `auto` mode; GPU vision - see the config copy below.
- Expert profile: a profile learned online on this model (`--expert-profile … --expert-profile-save … --expert-profile-save-every 10`), reused between restarts. It is a persistent artifact of this machine, not a fresh calibration.
- Experimental speed projection is enabled: `--control-vector-scaled …/Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer`. It was enabled in every earlier report from this PC.
- `--vram-reserve-mib 989`, `--pcie-frac 0.00`, `--pool-workers 10`, `--spec 4` with an MTP draft pack.
- Draft vocabulary subset: `draft_vocab=cyrillic` (the English/code subset plus the whole Cyrillic script, ~106k rows). This report is the case where that setting matters: without it the MTP draft head covers almost no Cyrillic continuation, and decode would be far slower than the numbers below.
- The engine's auto prompt chunk is `prompt chunk auto: 8192 tokens, a 96-slot ring` in the server log. It is unchanged from 0.1.38 through 0.1.40.1, so these numbers are comparable with the earlier Russian-variant reports from this PC.

```text
/home/dgbox/Strata/engine/strata --serve --pack /home/dgbox/Strata-data/packs/iq3_xxs --native /home/dgbox/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf /home/dgbox/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile /home/dgbox/Strata/data/expert-profile-learned.bin --expert-cache auto --prefill auto --spec 4 --mtp /home/dgbox/Strata-data/mtp/rt --max-context 204800 --kv int8 --mmap-experts --vision --vram-reserve-mib 989 --control-vector-scaled /home/dgbox/Strata/data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer --pcie-frac 0.00 --spec-min-p 0.50 --pool-workers 10 --expert-profile-save /home/dgbox/Strata/data/expert-profile-learned.bin --expert-profile-save-every 10
```

Full server config: [config.json](config.json) (was at /home/dgbox/Strata/strata-200k.json; the bearer token in its `mcp_servers` block is replaced with `removed`), environment details: [env.json](env.json).

## Method

Same as in the code-variant report, with two differences worth stating:

- The prompt text is repeated Russian prose with a random marker inside. The tokenizer needs ~0.279 tokens per character here, against ~0.196 for the code text, so the same nominal length becomes a longer prompt: 4028 tokens for a 4K target (3971 in the code variant), 31828 for 32K (31307), 127228 for 128K (125011).
- Warm-up: a **full sweep** (4K, 32K, 128K, gen-only) was run first and discarded, then the benchmark's own 4K warm-up. In this run the discarded pass read 4K at 2108-2125 tok/s and had one 4K repeat delayed to a 32.4 s TTFT by a request from the interactive session this server also serves; the measured pass read 2197-2210 tok/s with TTFT 1.85-1.86 s.
- Every measured prompt carries a random marker, so the prompt-prefix cache is not reused (the reused column is 0 in every run). Throughput comes from the engine's timing fields; TTFT is the time to the first non-empty streaming delta, ignoring keep-alives; wall is measured at the client.
- temperature=0, reasoning_effort=none, a 256-token output cap (1024 for the generation-only case). On Russian prose the model stops early very often - the generated lengths vary from run to run inside one case (for example 102 / 256 / 123 tokens at 32768-prompt), which is why the decode column of this report is much noisier than the prompt column.
- Memory: peak VRAM/RAM sampled every 2 seconds during the measured runs, plus a start snapshot.
- The measurement script is a local script (not part of this repository); it issues the requests, reads the engine's timing lines, and writes `runs.json` / `needles.json` in the format used by the earlier reports from this PC. Available on request.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT s median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 4096-prompt | 4028 | 0 | 142 | 3 | 2204.7 (range 2196.8-2209.9, n=3) | 108.1 (range 106.7-108.6, n=3) | 1.86 s (range 1.85-1.86, n=3) |
| 32768-prompt | 31828 | 0 | 123 | 3 | 3356.1 (range 3351.1-3383.4, n=3) | 93.7 (range 91.2-98.7, n=3) | 9.61 s (range 9.54-9.63, n=3) |
| 131072-prompt | 127228 | 0 | 123 | 3 | 3317.3 (range 3258.7-3343.9, n=3) | 89.9 (range 83.0-108.3, n=3) | 38.82 s (range 38.52-39.52, n=3) |
| gen-only | 228 | 0 | 640 | 3 | 309.2 (range 296.4-320.4, n=3) | 94.0 (range 85.9-96.2, n=3) | 0.75 s (range 0.72-0.78, n=3) |

- Total latency (client, wall): 4096-prompt - 3.17 s (range 3.15-3.18, n=3); 32768-prompt - 10.85 s (range 10.64-12.35, n=3); 131072-prompt - 40.18 s (range 39.64-42.56, n=3); gen-only - 7.52 s (range 4.07-8.26, n=3).
- Draft acceptance (accepted/total, all runs of a case): 4096-prompt 157/283; 32768-prompt 215/349; 131072-prompt 209/360; gen-only 580/1335. Accepted share is 43-62% here, against 72-79% for the same model on code text on the same card.
- Memory: {"start_snapshot": {"vram_used_mib": 22026, "ram_used_kib": 5918028}, "peak_vram_used_mib": 22026, "peak_ram_used_kib": 6610492, "note": "peak = the maximum of the 2-second samples taken during the measured runs"}
- Every run with its draft statistics (accepted/total): [runs.json](runs.json).
- Recall check (needle): [needles.json](needles.json). The needle prompts are English; the recall result is therefore about long-context retrieval, not about Russian text.

## Comparison with engine 0.1.40 on the same PC

Same PC, same model, same server config, same launch command, same `expert-profile-learned.bin`, same prompt chunk (8192). Only the Strata version changed: 0.1.40 (commit `1735d64`) -> 0.1.40.2, both compiled from source with the same CUDA 13.4 for `sm_89`. The 0.1.40 numbers are in [runs-0.1.40-baseline.json](runs-0.1.40-baseline.json) (measured 2026-10-06).

A control pair is included as well: [runs-0.1.40.1-control.json](runs-0.1.40.1-control.json) is a full repeat of the same sweep on **the identical engine binary** (0.1.40.1 = commit `82f46a8`, which changed only the Python server; `engine/strata` was not rebuilt, its md5 is the same as in the 0.1.40 baseline). On this variant the noise band is wider than on the code variant: prompt -0.7…+0.8%, decode -0.1…+8.6%.

| Configuration | Prompt tok/s 0.1.40 -> 0.1.40.2 | Decode tok/s 0.1.40 -> 0.1.40.2 | TTFT s 0.1.40 -> 0.1.40.2 | Control (same binary) prompt / decode |
| --- | --- | --- | --- | --- |
| 4096-prompt | 1893.8 -> 2204.7 (+16.4%) | 85.8 -> 108.1 (+26.0%) | 2.16 -> 1.86 | +0.8% / +8.6% |
| 32768-prompt | 3125.6 -> 3356.1 (+7.4%) | 91.4 -> 93.7 (+2.5%) | 10.31 -> 9.61 | -0.7% / -0.1% |
| 131072-prompt | 3136.4 -> 3317.3 (+5.8%) | 89.9 -> 89.9 (0.0%) | 41.06 -> 38.82 | -0.5% / +5.7% |
| gen-only | 298.5 -> 309.2 (+3.6%) | 87.8 -> 94.0 (+7.1%) | 0.78 -> 0.75 | -3.8% / +8.3% |

- **Prompt throughput moved the same way as in the code variant: +3.6…16.4%, with TTFT 0.03-2.24 s lower.** The prompt column is tight in both versions (3092-3139 and 3259-3344 tok/s at 131072-prompt) and the identical-binary control moved it by at most 0.8%, so this part is a real change. The +16.4% at 4096-prompt is the largest number in either variant; the code variant measured +10.2% on the same case, so treat the exact percentage as partly workload-dependent and the direction as solid.
- **Decode: not established.** The 4096-prompt +26% is confounded - the answers in 0.1.40 ran to the 256-token cap in two of three runs and here all three stopped at 142-144 tokens, and the accepted-draft share changed from 190/481 to 157/283. Different answers, different draft acceptance, different decode speed. The other three cases are inside the control band. The `gen-only` case, which runs longest, moved +7.1% against +8.3% for the same binary: no change.
- Recall: 6/6 at 32K and 128K in both versions.
- Longer trend on this PC, Russian variant, 131072-prompt: [2026-10-03](../2026-10-03-community-rtx4090-iq3xxs-200k-ru/README.md) 2984 tok/s (0.1.38) -> [2026-10-04](../2026-10-04-community-rtx4090-iq3xxs-200k-ru/README.md) 3126 (0.1.39) -> 3136 (0.1.40, in runs-0.1.40-baseline.json) -> 3317 here (0.1.40.2).

## What the Russian variant adds to the picture

- Prompt throughput is essentially language-independent: 3313-3367 tok/s at 32K for code text, 3351-3383 for Russian prose. The engine reads the same amount regardless of the script.
- Decode is not: 133.5-141.7 tok/s on code text, 89.9-108.1 on Russian prose at the same lengths - about 25-30% slower, and the gap tracks draft acceptance (72-79% accepted on code text, 43-62% here). On a Cyrillic-heavy workload the draft vocabulary is the setting that matters most on this model.
- The same nominal prompt length is a longer prompt in Cyrillic (4028 vs 3971 tokens at the 4K target, 127228 vs 125011 at 128K), so a context budget planned from English numbers is ~2% short for Russian text.

## Correctness and limitations

- Speed measurements do not establish general answer quality. The recall check passed 6/6 at 32K and 128K across depths 10/50/90; see needles.json.
- The GPU was not fully isolated: background services may have added small noise.
- Prompts are synthetic (repeated Russian prose with a random marker); real workloads will show different prefix reuse, answer lengths, and draft acceptance.
- This server also serves an interactive agent session. No request from it was issued during the measured runs; one request from it landed in the discarded warm-up sweep (its third 4K run shows a 32.4 s TTFT).
- The engine is compiled from source with CUDA 13.4, while the release binaries are built with CUDA 13.0. Comparisons inside this report are unaffected (both versions were built the same way on this machine), but they are not a comparison against the published Linux binaries - there are none for this tag.
- Opt-ins from this release were **not** tested here and are not part of these numbers: `STRATA_PREFILL_CPU_SHARE`, `STRATA_IO_PREFETCH` / `STRATA_IO_PF_STAGE`, `pin=N` (docs/RESEARCH_RUNS.md), `STRATA_SPEC_GUMBEL`, `STRATA_MMVQ_IL=0`.
