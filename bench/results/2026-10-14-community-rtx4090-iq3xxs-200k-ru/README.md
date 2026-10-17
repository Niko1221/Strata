# Community benchmark: NVIDIA GeForce RTX 4090 (Russian prompts)

Measured on 2026-10-14 by [Dmitry-B](https://github.com/Dmitry-B). This tests Strata 0.1.44.1 with qwen3.8-flash-next-iq3_xxs and a 204800-token context. Prompts are Russian-language text; greedy decoding, a 256-token output cap, three runs per configuration; TTFT measured over streaming. These are synthetic workloads; they do not establish general answer quality. The same sweep with code-explanation prompts, the version comparison and the two opt-out attributions are in [2026-10-14-community-rtx4090-iq3xxs-200k-code](../2026-10-14-community-rtx4090-iq3xxs-200k-code/README.md).

This is the Russian half of the 0.1.40.3 -> 0.1.44.1 comparison, and it follows [2026-10-08-community-rtx4090-iq3xxs-200k-ru](../2026-10-08-community-rtx4090-iq3xxs-200k-ru/README.md). Short answer:

- **Reading long prompts: no change** (-0.8…-1.1%, inside the same-binary band of +0.7…+2.1%).
- **Reading a short prompt: +30.8%** (228-token prompt, 314.8 -> 411.9 tok/s, TTFT 0.74 -> 0.57 s), reproduced by the control run on the same binary (-3.8% off the new value). The code-style report shows the same +31% and attributes it to the 0.1.41 CPU-prefill default.
- **Decode: the direction is the same (+8.9…+20.2%), but on Russian prompts this column is not usable as evidence** - the model stopped at different lengths in every run of every configuration, including the control. The lengths and draft totals are reported instead.
- Recall 6/6 at 32K and 128K in both versions.
- The sweep out to the context limit (156K / 185K / 200K tokens) was measured with code-style prompts only; on Russian text it was not repeated. Its result - +1.5…+1.7% prompt throughput at 156K and above, outside the same-binary band - is in the [code-style report](../2026-10-14-community-rtx4090-iq3xxs-200k-code/README.md).

## Hardware and software

- GPU: NVIDIA GeForce RTX 4090; 23028 MiB reported VRAM; 480.00 W power limit; PCIe bus 00000000:01:00.0; PCIe link speed and width: not measured (`nvidia-smi -q -d BUS` answers "Failed to parse --display/-d flags" on this driver). GPU clocks were not fixed.
- CPU: AMD Ryzen 9 7950X 16-Core Processor (32 logical CPUs).
- RAM: 46464 MiB installed.
- Storage: models, packs and the expert profile are on a 1.9 TB NVMe drive (ADATA LEGEND 960, ext4, mounted at `/`). No network storage is in the measured path.
- Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic; NVIDIA driver 610.57.04; nvcc release 13.4, V13.4.92.
- Strata commit `296b59d2f73c8f04a805df29c2fa97655adace78` (tag `v0.1.44.1`), branch `main`; engine 0.1.44.1 **compiled from source on this machine** (`engine/BUILD.json` records `"source": "local"`, `archs: [89]`, nvcc from `/usr/local/cuda`). The release publishes Windows archives only, so a source build is the only path here; it is the same build `./update.sh` runs. The previous version used in the comparison was built the same way on this machine.
- Background workloads: a dsh/Authentik/Caddy web stack and stock Ubuntu services; the GPU was dedicated to Strata but the operating system was not isolated.
- **Observer:** unlike the earlier reports from this PC, this machine does not serve the agent that ran the measurements - the agent runs against a different host (4x Tesla V100). No request from the agent could reach this server during the runs, and the whole session (warm-up, both styles, the control, the previous version, the two opt-outs) ran as one detached process from start to finish.

## Model and configuration

- Model: qwen3.8-flash-next-iq3_xxs (the Strata server's model name); GGUF filenames, sizes and modification times are in env.json (no hashes).
- Context 204800; INT8 KV; `--mmap-experts` (the file tier is active: experts are streamed from NVMe into an `auto` expert cache of 6986 slots / 11.24 GiB); GPU vision - see the config copy below.
- Expert profile: a profile learned online on this model (`--expert-profile … --expert-profile-save … --expert-profile-save-every 10`), reused between restarts. It is a persistent artifact of this machine, not a fresh calibration, and the same file was in use for the 0.1.38 … 0.1.40.3 reports from this PC.
- Experimental speed projection is enabled: `--control-vector-scaled …/Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer`. It was enabled in every earlier report from this PC.
- `--vram-reserve-mib 989`, `--pcie-frac 0.00`, `--pool-workers 10`, `--spec 4` with an MTP draft pack, `draft_vocab=cyrillic` (the English/code subset plus the whole Cyrillic script, ~106k rows).
- The server config also sets `env: {"STRATA_PF_FUSED": "1"}`; that key is present in the working config and was therefore present in every run below, including the previous version.
- **Two defaults of the new engine are ON and were ON in every measured run** (the engine prints both at start): `prefill: the CPU share is ON by default for prompt chunks below 1024 tokens` (0.1.41) and `STRATA_ROUTE_TAIL_SKIP=7 is ON (default)` (0.1.42). Both change answers, so this build is **not bit-identical to 0.1.40.3**. Their separate cost/benefit is measured in the code-style report.
- The engine's auto prompt chunk is `prompt chunk auto: 8192 tokens, a 96-slot ring` in the server log **in both 0.1.44.1 and 0.1.40.3** - unchanged from 0.1.38 through 0.1.44.1, so prompt-length numbers stay comparable with the earlier reports from this PC.
- The launch command is byte-identical to the one in the 0.1.40.3 report (compared line by line in the server log); the update did not change the server config files.

```text
/home/dgbox/Strata/engine/strata --serve --pack /home/dgbox/Strata-data/packs/iq3_xxs --native /home/dgbox/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf /home/dgbox/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile /home/dgbox/Strata/data/expert-profile-learned.bin --expert-cache auto --prefill auto --spec 4 --mtp /home/dgbox/Strata-data/mtp/rt --max-context 204800 --kv int8 --mmap-experts --vision --vram-reserve-mib 989 --control-vector-scaled /home/dgbox/Strata/data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer --pcie-frac 0.00 --spec-min-p 0.50 --pool-workers 10 --expert-profile-save /home/dgbox/Strata/data/expert-profile-learned.bin --expert-profile-save-every 10
```

Full server config: [config.json](config.json) (was at /home/dgbox/Strata/strata-200k.json; the bearer token in its `mcp_servers` block is replaced with `removed`), environment details: [env.json](env.json).

## Method

- Warm-up: a **full sweep** (4K, 32K, 128K, gen-only) was run first and discarded (the code-style warm-up sweep of the same session; the engine was not restarted between the two sweeps). Required on this machine: in a discarded sweep the first 4K read ran at 1522.7 tok/s with a 2.62 s TTFT, while the measured pass read 1970.3-2151.8 tok/s with 1.89-2.06 s TTFT. That difference is the cold expert cache and page cache after a service restart, not the version.
- Every measured prompt carries a random marker, so the prompt-prefix cache is not reused; the table's reused column reports the actual reused token counts from the engine (0 in every run here).
- Throughput comes from the engine's timing fields (prompt_per_second / predicted_per_second). TTFT is the time to the first non-empty streaming delta, ignoring keep-alives. Total latency (wall) is the whole request time measured at the client.
- temperature=0, reasoning_effort=none, a 256-token output cap (1024 for the generation-only case). On Russian text the model stopped early and at different points in nearly every run; actual generated lengths are listed per case below.
- Memory: peak VRAM/RAM sampled every 2 seconds during the measured runs, plus a start snapshot.
- The measurement script is a local script (not part of this repository); it issues the requests, reads the engine's timing lines, and writes `runs.json` / `needles.json` in the format used by the earlier reports from this PC. Available on request.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT s median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 4096-prompt | 4028 | 0 | 256/102/256 | 3 | 2060.5 (range 1970.3-2151.8, n=3) | 134.3 (range 114.9-176.1, n=3) | 1.97 s (range 1.89-2.06, n=3) |
| 32768-prompt | 31828 | 0 | 238/256/123 | 3 | 3296.5 (range 3260.9-3317.6, n=3) | 149.0 (range 148.2-179.4, n=3) | 9.68 s (range 9.62-9.78, n=3) |
| 131072-prompt | 127228 | 0 | 132/161/162 | 3 | 3305.9 (range 3293.5-3331.1, n=3) | 161.7 (range 148.9-184.4, n=3) | 38.55 s (range 38.25-38.68, n=3) |
| gen-only | 228 | 0 | 652/799/768 | 3 | 411.9 (range 380.9-424.0, n=3) | 142.8 (range 137.6-143.8, n=3) | 0.57 s (range 0.55-0.61, n=3) |

- Total latency (client, wall): 4096-prompt - 3.86 s (range 2.63-4.1, n=3); 32768-prompt - 11.21 s (range 10.35-11.49, n=3); 131072-prompt - 39.39 s (range 39.24-39.62, n=3); gen-only - 5.93 s (range 5.28-6.15, n=3).
- Draft acceptance (accepted/total, all runs of a case): 4096-prompt 364/470; 32768-prompt 421/479; 131072-prompt 320/397; gen-only 1274/1784.
- Memory: peak VRAM 22024 MiB (unchanged from the start snapshot), peak RAM 6654892 KiB.
- Every run with its draft statistics (accepted/total): [runs.json](runs.json).
- Recall check (needle): [needles.json](needles.json) - 6/6 found at 32K and 128K, depths 10/50/90.

## Comparison with engine 0.1.40.3 on the same PC

Same PC, same model, same server config, same launch command, same `expert-profile-learned.bin`, same prompt chunk (8192), the same session as the code-style sweep. Only the Strata version changed: 0.1.40.3 (commit `d5ea713`) -> 0.1.44.1 (commit `296b59d`), both compiled from source on this machine with CUDA 13.4 for `sm_89`.

Reference runs included so the comparison is self-contained:

- [runs-0.1.40.3-previous.json](runs-0.1.40.3-previous.json) - the previous version, **measured in this same session** (its binary was swapped back in, the service restarted, the warm-up sweep run and discarded, then the same sweep measured);
- [runs-0.1.44.1-control.json](runs-0.1.44.1-control.json) - a full repeat of the sweep on **the identical 0.1.44.1 binary**: the noise band of this method on this machine;
- [needles-0.1.40.3.json](needles-0.1.40.3.json) - the recall check of the previous version (6/6).

| Configuration | Prompt tok/s 0.1.40.3 -> 0.1.44.1 | Decode tok/s 0.1.40.3 -> 0.1.44.1 | TTFT s 0.1.40.3 -> 0.1.44.1 | Noise band (same binary): prompt / decode |
| --- | --- | --- | --- | --- |
| 4096-prompt | 2080.9 -> 2060.5 (-1.0%) | 121.5 -> 134.3 (+10.5%) | 1.95 -> 1.97 | +1.6% / +0.1% |
| 32768-prompt | 3334.1 -> 3296.5 (-1.1%) | 136.8 -> 149.0 (+8.9%) | 9.57 -> 9.68 | +2.1% / -0.9% |
| 131072-prompt | 3331.9 -> 3305.9 (-0.8%) | 140.6 -> 161.7 (+15.0%) | 38.24 -> 38.55 | +0.7% / +8.9% |
| gen-only (228-token prompt) | 314.8 -> 411.9 (**+30.8%**) | 118.8 -> 142.8 (+20.2%) | 0.74 -> 0.57 | -3.8% / -1.8% |

### What is a change and what is not

- **Long-prompt reading: no change.** -0.8…-1.1% sits inside the same-binary band (+0.7…+2.1%). TTFT moved by 0.02-0.31 s.
- **Short-prompt reading: +30.8%, a real change.** The control on the same binary lands 3.8% off the new value, and the code-style sweep shows the same +31.2% on a 163-token prompt. The [code-style report](../2026-10-14-community-rtx4090-iq3xxs-200k-code/README.md) attributes it: with `STRATA_PREFILL_CPU_SHARE=0` the same build reads 213.8 tok/s with a 0.78 s TTFT, i.e. back to the 0.1.40.3 level.
- **Decode: reported, not concluded.** On Russian text the answer length differs in every run of every configuration - 256/102/256 here against 256/231/143 on 0.1.40.3 at 4096-prompt, 652/799/768 against 345/302/790 at gen-only - and the control run differs from the measured run in the same way. Medians over different lengths are not evidence, so the honest statement is: **decode did not get slower; the direction of the change matches the code-style measurement (+8.9…+20.2%), where the clean case (identical lengths, identical draft acceptance) gave +13.8% and `STRATA_ROUTE_TAIL_SKIP=0` brought it back to the 0.1.40.3 number.**
- Draft acceptance did not drop (4096: 73.9% -> 77.4%; 32768: 84.0% -> 87.9%; 131072: 83.8% -> 80.6%; gen-only: 65.1% -> 71.4%); the spread between the measured run and the control on the same binary is of the same size (70.5%, 85.5%, 76.6%, 70.8%), so these differences are not attributed to the version.
- Recall: 6/6 at 32K and 128K in both versions ([needles.json](needles.json), [needles-0.1.40.3.json](needles-0.1.40.3.json)).
- Answers are **not** bit-identical to 0.1.40.3: the span crosses the 0.1.41 and 0.1.42 default changes listed above.

## Longer trend on this PC

Same config, same method, prompt throughput at 131072 tokens (Russian / code):

[2026-10-03](../2026-10-03-community-rtx4090-iq3xxs-200k-ru/README.md) 0.1.38: 2984 / 2962 -> [2026-10-04](../2026-10-04-community-rtx4090-iq3xxs-200k-ru/README.md) 0.1.39: 3126 / 3085 -> 0.1.40: 3136.4 / 3118.5, in [runs-0.1.40-baseline.json](../2026-10-08-community-rtx4090-iq3xxs-200k-ru/runs-0.1.40-baseline.json) of the 2026-10-08 report -> 0.1.40.2: 3317.3 / 3312.6, measured on 2026-10-07 and published as PR #1377 (that branch is not in `main`, so the numbers are quoted here rather than linked) -> [2026-10-08](../2026-10-08-community-rtx4090-iq3xxs-200k-ru/README.md) 0.1.40.3: 3336.2 / 3292.3 -> this report 0.1.44.1: 3305.9 / 3276.6 (with 0.1.40.3 re-measured in the same session at 3331.9 / 3307.6).

Long-prompt reading has been flat since 0.1.40.2. What 0.1.41-0.1.44 changed on this machine is at the other end of the range: short prompts read ~30% faster, and decode is ~14% faster where it can be measured cleanly (see the code-style report).

## Correctness and limitations

- Speed measurements do not establish general answer quality. The recall check passed 6/6 at 32K and 128K across depths 10/50/90 in both versions; see needles.json.
- The GPU was not fully isolated: background services may have added small noise.
- Prompts are synthetic (repeated text with a random marker); real workloads will show different prefix reuse and draft acceptance.
- On Russian prompts the decode column of this method is unusable as evidence: the model stops at different lengths in nearly every run, in both versions and in the control. Only the prompt column and TTFT are compared.
- The engine is compiled from source with CUDA 13.4, while the release binaries are built with CUDA 13.0. Comparisons inside this report are unaffected (every version here was built the same way on this machine), but they are not a comparison against published Linux binaries - there are none for this tag.
- The opt-out runs (`STRATA_ROUTE_TAIL_SKIP=0`, `STRATA_PREFILL_CPU_SHARE=0`) and the long-context sweep (156K-200K tokens) were made with code-style prompts only; they are linked from the code-style report and not repeated here.
- Opt-outs from this release that were **not** tested here: `STRATA_IO_PREFETCH` / `STRATA_IO_PF_STAGE`, `STRATA_UNBUFFERED_LOAD`, `pin=N` (docs/RESEARCH_RUNS.md), `STRATA_SPEC_GUMBEL`, `STRATA_MMVQ_IL=0`, `STRATA_STAGER_SLEEP=0`, `--conversation-cache-mib`, `--batch` / `--batch-groups`, `--kv k8v4`. Multi-GPU, AMD/ROCm and Intel Arc paths do not apply to this PC.
- The 0.1.40.3 reference run reuses the binary that served this machine before the update (built 2026-10-08 with the same toolchain), not a fresh build of that tag.
