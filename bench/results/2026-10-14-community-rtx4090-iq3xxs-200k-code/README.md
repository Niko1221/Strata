# Community benchmark: NVIDIA GeForce RTX 4090

Measured on 2026-10-14 by [Dmitry-B](https://github.com/Dmitry-B). This tests Strata 0.1.44.1 with qwen3.8-flash-next-iq3_xxs and a 204800-token context. Prompts are code-explanation text; greedy decoding, a 256-token output cap, three runs per configuration; TTFT measured over streaming. These are synthetic workloads; they do not establish general answer quality. The same sweep with Russian-language prompts is in [2026-10-14-community-rtx4090-iq3xxs-200k-ru](../2026-10-14-community-rtx4090-iq3xxs-200k-ru/README.md), and the same method on a different machine (4x Tesla V100, IQ3_S, 262144 context) is in [2026-10-14-community-4x-tesla-v100-iq3-s-262k-code](../2026-10-14-community-4x-tesla-v100-iq3-s-262k-code/README.md). Read together, the two reports are the positive and the negative half of one claim: the two defaults that turned on in 0.1.41 and 0.1.42 are adaptive, and on a machine where the experts are resident the engine switches both of them off and neither gain appears.

This is the follow-up of [2026-10-08-community-rtx4090-iq3xxs-200k-code](../2026-10-08-community-rtx4090-iq3xxs-200k-code/README.md) (0.1.40.3 on the same PC and the same config), and it covers four releases at once (0.1.41, 0.1.42, 0.1.43/0.1.43.1, 0.1.44/0.1.44.1). Short answer, with the previous version re-measured in the same session:

- **Reading long prompts up to 128K: no change.** 4K/32K/128K move by -0.6…-1.1%, inside the same-binary band of this method.
- **Reading prompts at the top of the context: +1.5…+1.7%, and that one is outside the noise band.** At 156291 / 185531 / 200015 tokens the same-binary control moves by only ±0.1-0.2%, while 0.1.44.1 reads 3283.5 / 3292.5 / 3266.4 tok/s against 3235.1 / 3238.3 / 3213.8 on 0.1.40.3; TTFT at 200015 tokens falls from 62.32 s to 61.32 s. Recall at 192K is 3/3 in both versions.
- **Reading a short prompt: +31%** (163-token prompt, 221.6 -> 290.8 tok/s, TTFT 0.75 -> 0.58 s). Reproduced by a second run of the same binary, and it disappears completely with `STRATA_PREFILL_CPU_SHARE=0` (213.8 tok/s, TTFT 0.78 s) - i.e. it is the 0.1.41 CPU-prefill default, measured here for the first time on this PC.
- **Decode: +13.8% on the one case where the column is comparable** (generation-only, 1024 tokens in every run, draft acceptance unchanged). With `STRATA_ROUTE_TAIL_SKIP=0` the same build decodes at 142.0 tok/s, which is exactly the 0.1.40.3 number (141.1) - so the gain is the 0.1.42 default, and its size matches the release's "+10-20%".
- Recall 6/6 at 32K and 128K in both versions; draft acceptance did not drop.

## Hardware and software

- GPU: NVIDIA GeForce RTX 4090; 23028 MiB reported VRAM; 480.00 W power limit; PCIe bus 00000000:01:00.0; PCIe link speed and width: not measured (`nvidia-smi -q -d BUS` answers "Failed to parse --display/-d flags" on this driver). GPU clocks were not fixed.
- CPU: AMD Ryzen 9 7950X 16-Core Processor (32 logical CPUs).
- RAM: 46464 MiB installed.
- Storage: models, packs and the expert profile are on a 1.9 TB NVMe drive (ADATA LEGEND 960, ext4, mounted at `/`). No network storage is in the measured path.
- Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic; NVIDIA driver 610.57.04; nvcc release 13.4, V13.4.92.
- Strata commit `296b59d2f73c8f04a805df29c2fa97655adace78` (tag `v0.1.44.1`), branch `main`; engine 0.1.44.1 **compiled from source on this machine** (`engine/BUILD.json` records `"source": "local"`, `archs: [89]`, nvcc from `/usr/local/cuda`). The release publishes Windows archives only, so a source build is the only path here; it is the same build `./update.sh` runs. The previous version used in the comparison was built the same way on this machine (its binary was kept from 2026-10-08 and swapped back in - see *Comparison*).
- Background workloads: a dsh/Authentik/Caddy web stack and stock Ubuntu services; the GPU was dedicated to Strata but the operating system was not isolated.
- **Observer:** unlike the earlier reports from this PC, this machine does not serve the agent that ran the measurements - the agent runs against a different host (4x Tesla V100). No request from the agent could reach this server during the runs, and the whole session (warm-up, both styles, the control, the previous version, the two opt-outs) ran as one detached process from start to finish.

## Model and configuration

- Model: qwen3.8-flash-next-iq3_xxs (the Strata server's model name); GGUF filenames, sizes and modification times are in env.json (no hashes).
- Context 204800; INT8 KV; `--mmap-experts` (the file tier is active: experts are streamed from NVMe into an `auto` expert cache of 6986 slots / 11.24 GiB); GPU vision - see the config copy below.
- Expert profile: a profile learned online on this model (`--expert-profile … --expert-profile-save … --expert-profile-save-every 10`), reused between restarts. It is a persistent artifact of this machine, not a fresh calibration, and the same file was in use for the 0.1.38 … 0.1.40.3 reports from this PC.
- Experimental speed projection is enabled: `--control-vector-scaled …/Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer`. It was enabled in every earlier report from this PC.
- `--vram-reserve-mib 989`, `--pcie-frac 0.00`, `--pool-workers 10`, `--spec 4` with an MTP draft pack, `draft_vocab=cyrillic` (the English/code subset plus the whole Cyrillic script, ~106k rows).
- The server config also sets `env: {"STRATA_PF_FUSED": "1"}`; that key is present in the working config and was therefore present in every run below, including the previous version and the opt-outs.
- **Two defaults of the new engine are ON and were ON in every measured run** (the engine prints both at start):
  - `prefill: the CPU share is ON by default for prompt chunks below 1024 tokens` (0.1.41);
  - `STRATA_ROUTE_TAIL_SKIP=7 is ON (default)` (0.1.42).
  Both change answers, so this build is **not bit-identical to 0.1.40.3**; the two opt-out runs below isolate their cost/benefit separately.
- The engine's auto prompt chunk is `prompt chunk auto: 8192 tokens, a 96-slot ring` in the server log **in both 0.1.44.1 and 0.1.40.3** - unchanged from 0.1.38 through 0.1.44.1, so prompt-length numbers stay comparable with the earlier reports from this PC.
- The launch command is byte-identical to the one in the 0.1.40.3 report (compared line by line in the server log); the update did not change the server config files.

```text
/home/dgbox/Strata/engine/strata --serve --pack /home/dgbox/Strata-data/packs/iq3_xxs --native /home/dgbox/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf /home/dgbox/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile /home/dgbox/Strata/data/expert-profile-learned.bin --expert-cache auto --prefill auto --spec 4 --mtp /home/dgbox/Strata-data/mtp/rt --max-context 204800 --kv int8 --mmap-experts --vision --vram-reserve-mib 989 --control-vector-scaled /home/dgbox/Strata/data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer --pcie-frac 0.00 --spec-min-p 0.50 --pool-workers 10 --expert-profile-save /home/dgbox/Strata/data/expert-profile-learned.bin --expert-profile-save-every 10
```

Full server config: [config.json](config.json) (was at /home/dgbox/Strata/strata-200k.json; the bearer token in its `mcp_servers` block is replaced with `removed`), environment details: [env.json](env.json).

## Method

- Warm-up: a **full sweep** (4K, 32K, 128K, gen-only) was run first and discarded, then the benchmark's own 4K warm-up. Required on this machine: in the discarded sweep the first 4K read ran at 1522.7 tok/s with a 2.62 s TTFT, while the measured pass read 2155.7-2166.5 tok/s with 1.85 s TTFT. That difference is the cold expert cache and page cache after a service restart, not the version.
- Every measured prompt carries a random marker, so the prompt-prefix cache is not reused; the table's reused column reports the actual reused token counts from the engine (0 in every run here).
- Throughput comes from the engine's timing fields (prompt_per_second / predicted_per_second). TTFT is the time to the first non-empty streaming delta, ignoring keep-alives. Total latency (wall) is the whole request time measured at the client.
- temperature=0, reasoning_effort=none, a 256-token output cap (1024 for the generation-only case). The model often stopped early on the repetitive text; actual generated lengths are in the table.
- Memory: peak VRAM/RAM sampled every 2 seconds during the measured runs, plus a start snapshot.
- The measurement script is a local script (not part of this repository); it issues the requests, reads the engine's timing lines, and writes `runs.json` / `needles.json` in the format used by the earlier reports from this PC. Available on request.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT s median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 4096-prompt | 3971 | 0 | 256 | 3 | 2157.2 (range 2155.7-2166.5, n=3) | 151.6 (range 145.9-156.0, n=3) | 1.85 s (range 1.85-1.86, n=3) |
| 32768-prompt | 31307 | 0 | 256 | 3 | 3324.5 (range 3314.0-3334.9, n=3) | 143.7 (range 143.7-149.3, n=3) | 9.44 s (range 9.41-9.47, n=3) |
| 131072-prompt | 125011 | 0 | 142/142/256 | 3 | 3276.6 (range 3264.9-3289.9, n=3) | 127.4 (range 122.4-137.5, n=3) | 38.21 s (range 38.06-38.35, n=3) |
| gen-only | 163 | 0 | 1024 | 3 | 290.8 (range 274.1-312.3, n=3) | 160.6 (range 155.3-167.5, n=3) | 0.58 s (range 0.54-0.61, n=3) |

- Total latency (client, wall): 4096-prompt - 3.52 s (range 3.49-3.6, n=3); 32768-prompt - 11.18 s (range 11.17-11.21, n=3); 131072-prompt - 39.5 s (range 39.08-40.21, n=3); gen-only - 6.94 s (range 6.64-7.19, n=3).
- Draft acceptance (accepted/total, all runs of a case): 4096-prompt 500/670; 32768-prompt 496/666; 131072-prompt 334/473; gen-only 2071/2622.
- Memory: peak VRAM 22024 MiB (unchanged from the start snapshot), peak RAM 6564952 KiB. The expert cache was sized 6986 slots / 11.24 GiB, 702 MiB of VRAM free with everything loaded.
- Every run with its draft statistics (accepted/total): [runs.json](runs.json).
- Recall check (needle): [needles.json](needles.json) - 6/6 found at 32K and 128K, depths 10/50/90.
- The sweep was also run out to the context limit (156291 / 185531 / 200015 tokens) in a separate session of the same day: see *Long context* below.

## Comparison with engine 0.1.40.3 on the same PC

Same PC, same model, same server config, same launch command, same `expert-profile-learned.bin`, same prompt chunk (8192). Only the Strata version changed: 0.1.40.3 (commit `d5ea713`) -> 0.1.44.1 (commit `296b59d`), both compiled from source on this machine with CUDA 13.4 for `sm_89`.

Three reference runs are included so the comparison is self-contained:

- [runs-0.1.40.3-previous.json](runs-0.1.40.3-previous.json) - the previous version, **measured in this same session**: its engine binary (`engine/strata.01403.bak`, built on this PC on 2026-10-08) was swapped in, the service restarted, the same warm-up sweep run and discarded, then the same sweep measured;
- [runs-0.1.44.1-control.json](runs-0.1.44.1-control.json) - a full repeat of the sweep on **the identical 0.1.44.1 binary**. This is the noise band of the method on this machine, and it is what decides whether a difference is a change;
- [needles-0.1.40.3.json](needles-0.1.40.3.json) - the recall check of the previous version (6/6).

| Configuration | Prompt tok/s 0.1.40.3 -> 0.1.44.1 | Decode tok/s 0.1.40.3 -> 0.1.44.1 | TTFT s 0.1.40.3 -> 0.1.44.1 | Noise band (same binary): prompt / decode |
| --- | --- | --- | --- | --- |
| 4096-prompt | 2169.2 -> 2157.2 (-0.6%) | 136.9 -> 151.6 (+10.7%) | 1.84 -> 1.85 | -5.8% / -6.7% |
| 32768-prompt | 3350.5 -> 3324.5 (-0.8%) | 140.7 -> 143.7 (+2.1%) | 9.37 -> 9.44 | -0.4% / +0.8% |
| 131072-prompt | 3307.6 -> 3276.6 (-0.9%) | 129.5 -> 127.4 (-1.6%) | 37.85 -> 38.21 | +0.7% / +6.9% |
| gen-only (163-token prompt) | 221.6 -> 290.8 (**+31.2%**) | 141.1 -> 160.6 (**+13.8%**) | 0.75 -> 0.58 | -0.7% / +1.1% |

### What is a change and what is not

- **Long-prompt reading: no change.** -0.6…-1.1% at 4K/32K/128K sits inside the same-binary band (-5.8…+0.7%). The honest reading is "the same speed", not "-1%". TTFT moved by 0.01-0.36 s.
- **Short-prompt reading: +31%, and it is a real change.** Two independent runs of the same binary agree (290.8 and 288.7 tok/s, 0.58 s TTFT) against 221.6 tok/s and 0.75 s on 0.1.40.3. The prompt is 163 tokens, i.e. below the 1024-token threshold of the 0.1.41 CPU-prefill default. See the attribution below.
- **Decode: the only case where the column is comparable is gen-only** - every run of every configuration generated exactly 1024 tokens there, and draft acceptance is unchanged (77.8% on 0.1.40.3, 79.0% on 0.1.44.1, 79.5% in the control). There the gain is **+13.8%**, outside the same-binary band (+1.1%). At 4096-prompt the lengths are also identical (256 in every run) and the direction is the same (+10.7%), but the control on that case spans -6.7%, so the defensible statement there is "not slower, plausibly a few percent faster", not "+10.7%". At 32768 and 131072 the answers ended at different lengths between runs, so no decode conclusion is drawn from those rows; the lengths and draft totals are given above instead.

### Attribution: each new default measured separately

Two opt-in-turned-default features were switched off one at a time, with the same config and the same method (test configs `strata-test-200k-no-tail-skip.json` and `strata-test-200k-no-cpu-share.json`, which differ from the working config only by one `env` key; the working config was not touched). Raw runs: [runs-0.1.44.1-no-tail-skip.json](runs-0.1.44.1-no-tail-skip.json), [runs-0.1.44.1-no-cpu-share.json](runs-0.1.44.1-no-cpu-share.json). Code-style prompts only; the Russian sweep was not repeated for the opt-outs.

| Configuration | Decode tok/s: default / `ROUTE_TAIL_SKIP=0` / `PREFILL_CPU_SHARE=0` | Prompt tok/s: default / `TAIL_SKIP=0` / `CPU_SHARE=0` | TTFT s: default / `TAIL_SKIP=0` / `CPU_SHARE=0` |
| --- | --- | --- | --- |
| gen-only (1024 tokens in every run) | 160.6 / **142.0** / 161.1 | 290.8 / 290.4 / **213.8** | 0.58 / 0.58 / **0.78** |
| 4096-prompt (256 tokens in every run) | 151.6 / **139.4** / 153.0 | 2157.2 / 2133.9 / 2145.2 | 1.85 / 1.88 / 1.86 |
| 32768-prompt (256 tokens in every run) | 143.7 / **139.2** / 147.4 | 3324.5 / 3303.8 / 3317.1 | 9.44 / 9.50 / 9.46 |
| 131072-prompt (lengths differ) | 127.4 / 116.5 / 135.3 | 3276.6 / 3256.1 / 3275.1 | 38.21 / 38.45 / 38.23 |

- **The decode gain is `STRATA_ROUTE_TAIL_SKIP=7`.** With it off, the same 0.1.44.1 binary decodes gen-only at 142.0 tok/s - the 0.1.40.3 number is 141.1 - and at 4096-prompt at 139.4 against 136.9 on 0.1.40.3. Grouping the runs by the setting rather than by version: decode on = 160.6/162.4 (gen-only), 151.6/141.5/153.0 (4K), 143.7/144.9/147.4 (32K); decode off = 141.1/142.0, 136.9/139.4, 140.7/139.2. That is +13.8% at gen-only, +8.8% at 4096, +3.5% at 32768 by medians of the groups. The release's "+10-20% decode on 12-16 GB cards that miss experts" is the right order of magnitude here, and this card is not a small one - the effect on a 24 GB card with a 6986-slot expert cache is real but smaller at long prompts.
- **The short-prompt gain is the CPU prefill share.** With `STRATA_PREFILL_CPU_SHARE=0` the 163-token prompt reads at 213.8 tok/s with a 0.78 s TTFT - back to 0.1.40.3's 221.6 / 0.75 - while decode stays at 161.1. Turning only `ROUTE_TAIL_SKIP` off leaves the short-prompt reading at 290.4. So the two features account for the two columns separately, and neither one is a side effect of the other.
- **Neither default costs anything measurable at long prompts**: with either one off, 4K/32K/128K reading stays within 1.1% of the default build.
- **Both defaults are adaptive, and the negative half of that is measured on another machine.** On a 4x Tesla V100 machine where 98% of the experts are resident in the GPU cache, the engine's own start log turns `STRATA_ROUTE_TAIL_SKIP` off by itself (`98.0% of the experts are in the GPU cache, so few are missed and skipping them costs more draft acceptance than it saves`) and does not arm the CPU prefill share at all; forcing `STRATA_PREFILL_CPU_SHARE=1` there changes the 163-token case by ~2% instead of +31%. See *Why the two machines differ* in [the 4xV100 report](../2026-10-14-community-4x-tesla-v100-iq3-s-262k-code/README.md).
- Draft acceptance is unchanged in substance in every configuration (4096: 75.2% -> 74.6%; 32768: 79.4% -> 74.5%; gen-only: 77.8% -> 79.0%). The 32768 drop is inside the run-to-run spread of this case (the control gives 76.1% on the same binary).
- Recall: 6/6 at 32K and 128K in both versions ([needles.json](needles.json), [needles-0.1.40.3.json](needles-0.1.40.3.json)).
- Answers are **not** bit-identical to 0.1.40.3: the span crosses the 0.1.41 and 0.1.42 default changes listed above. The release notes' byte-identity claim is 0.1.43.1 -> 0.1.44.1, which is not what this comparison spans.

## Long context: 156K / 185K / 200K tokens, up to the 204800 limit

The earlier reports from this PC stopped at 131072 tokens. The configured context is 204800, so the sweep was extended to the limit: targets 160k / 190k / 205k produce **156291 / 185531 / 200015 actual prompt tokens** (97.7% of the context) with the same 256-token output cap, three runs each, same warm-up protocol, same random marker per prompt. The largest request (200015 prompt + 256 generated tokens) was served without errors and without resizing.

Raw runs: [runs-long-context.json](runs-long-context.json) (0.1.44.1), [runs-long-context-control.json](runs-long-context-control.json) (the same binary, second pass), [runs-long-context-0.1.40.3.json](runs-long-context-0.1.40.3.json) (the previous version, same session, binary swapped back in). Recall: [needles-long-context.json](needles-long-context.json) and [needles-long-context-0.1.40.3.json](needles-long-context-0.1.40.3.json).

| Configuration (actual prompt tokens) | Prompt tok/s 0.1.40.3 -> 0.1.44.1 | Decode tok/s 0.1.40.3 -> 0.1.44.1 | TTFT s 0.1.40.3 -> 0.1.44.1 | Noise band (same binary): prompt / decode |
| --- | --- | --- | --- | --- |
| 156291 | 3235.1 -> **3283.5 (+1.5%)** | 139.1 -> 162.3 (+16.7%) | 48.38 -> 47.67 | +0.1% / -8.8% |
| 185531 | 3238.3 -> **3292.5 (+1.7%)** | 118.0 -> 125.7 (+6.5%) | 57.38 -> 56.43 | -0.2% / +7.1% |
| 200015 | 3213.8 -> **3266.4 (+1.6%)** | 126.2 -> 135.3 (+7.2%) | 62.32 -> 61.32 | +0.0% / -3.3% |

- **The prompt gain at the top of the range is real.** Three lengths, three independent runs each, ranges of 1-3 tok/s inside a case (200015 tokens: 3266.0-3268.1 on 0.1.44.1, 3208.4-3214.5 on 0.1.40.3), and the control on the same binary lands within ±0.2% of the measured value. The gain is +1.5…+1.7% at all three lengths, and TTFT is 0.7-1.0 s lower.
- **It has the opposite sign to the 4K-128K rows, and both are within their own bands.** At 4K-128K the deltas are -0.6…-1.1% against a band of -5.8…+0.7%; at 156K-200K they are +1.5…+1.7% against a band of ±0.2%. The bands differ because a 50-60 s request is dominated by steady-state expert streaming, while a 2-4 s request carries much more relative setup noise. So the defensible statement is: nothing measurable up to 128K, a small consistent gain of about 1.5% at 156K and above. What in 0.1.41-0.1.44 causes it is **not identified here** - the candidates that touch long prompts (`STRATA_ROUTE_TAIL_SKIP`, the adaptive-tier copy overlap of 0.1.43 and its 0.1.43.1 re-tuning, the measured PCIe share) are not separated by these runs, and `ROUTE_TAIL_SKIP=0` at 128K changed prompt speed by only -0.6%.
- **Decode at these lengths is reported, not concluded**: the answers ended at different points in most runs (at 156291: 163/98/163 on 0.1.40.3, 103/97/163 on 0.1.44.1, 102/163/102 in the control), and the control spans -8.8…+7.1%. The 200015-token row is the only one where every run of both versions reached the 256-token cap; there the delta is +7.2% against a control of -3.3%, i.e. the same direction as the shorter cases but not separated from their noise.
- **Memory does not move**: peak VRAM 22024 MiB (0.1.44.1) and 22042 MiB (0.1.40.3) - the same as at 128K; peak RAM 6895344 KiB and 6981676 KiB. Nothing in the 200K request path spilled.
- Recall at 192K, depths 10/50/90: **3/3 found on both versions** (191970-191971 prompt tokens per check, 56-61 s each).
- Not measured at these lengths: the Russian prompt variant, and anything above 204800 - the config's context is 204800, so 200015 tokens is the practical ceiling for this setup.
- Provenance caveat for the reference run: its `env.strata_commit` records the git working tree, which stayed at `v0.1.44.1` while the old binary was swapped in; `env.engine_version` records the binary actually used (`0.1.40.3`), and the binary's md5 is the one kept as `engine/strata.01403.bak`.

## Longer trend on this PC

Same config, same method, prompt throughput at 131072 tokens (code / Russian):

[2026-10-03](../2026-10-03-community-rtx4090-iq3xxs-200k-code/README.md) 0.1.38: 2962 / 2984 -> [2026-10-04](../2026-10-04-community-rtx4090-iq3xxs-200k-code/README.md) 0.1.39: 3085 / 3126 -> 0.1.40: 3118.5 / 3136.4, in [runs-0.1.40-baseline.json](../2026-10-08-community-rtx4090-iq3xxs-200k-code/runs-0.1.40-baseline.json) of the 2026-10-08 report -> 0.1.40.2: 3312.6 / 3317.3, measured on 2026-10-07 and published as PR #1377 (that branch is not in `main`, so the numbers are quoted here rather than linked) -> [2026-10-08](../2026-10-08-community-rtx4090-iq3xxs-200k-code/README.md) 0.1.40.3: 3292.3 / 3336.2 -> this report 0.1.44.1: 3276.6 / 3305.9 (with 0.1.40.3 re-measured in the same session at 3307.6 / 3331.9).

At the longer lengths there is no earlier series to compare with - the 156K/185K/200K rows above are the first measurements at the top of this context on this PC.

Long-prompt reading at 128K has been flat since 0.1.40.2. What 0.1.41-0.1.44 changed on this machine is at the other end of the range: short prompts read ~30% faster, and decode is ~14% faster where it can be measured cleanly.

## Correctness and limitations

- Speed measurements do not establish general answer quality. The recall check passed 6/6 at 32K and 128K across depths 10/50/90 in both versions, and 3/3 at 192K (see needles.json and needles-long-context.json).
- The GPU was not fully isolated: background services may have added small noise.
- Prompts are synthetic (repeated text with a random marker); real workloads will show different prefix reuse and draft acceptance.
- The decode column of this method is noisy: on the cases where the model stopped at different lengths between runs, no decode conclusion is drawn, and the lengths and draft totals are reported instead.
- The engine is compiled from source with CUDA 13.4, while the release binaries are built with CUDA 13.0. Comparisons inside this report are unaffected (every version here was built the same way on this machine), but they are not a comparison against published Linux binaries - there are none for this tag.
- Opt-outs from this release that were **not** tested here: `STRATA_IO_PREFETCH` / `STRATA_IO_PF_STAGE`, `STRATA_UNBUFFERED_LOAD`, `pin=N` (docs/RESEARCH_RUNS.md), `STRATA_SPEC_GUMBEL`, `STRATA_MMVQ_IL=0`, `STRATA_STAGER_SLEEP=0`, `--conversation-cache-mib`, `--batch` / `--batch-groups`, `--kv k8v4`. AMD/ROCm and Intel Arc paths do not apply to this PC; the multi-GPU path is covered by the [4x Tesla V100 report](../2026-10-14-community-4x-tesla-v100-iq3-s-262k-code/README.md) of the same day.
- The opt-out runs were made with code-style prompts only; the Russian sweep was not repeated under the opt-outs. The long-context sweep (156K-200K) was also run with code prompts only.
- The 0.1.40.3 reference run reuses the binary that served this machine before the update (built 2026-10-08 with the same toolchain), not a fresh build of that tag.
