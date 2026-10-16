# Community benchmark on 2x NVIDIA GeForce RTX 3090 24 GB (sm_86), i7-11700K, 62.6 GiB RAM

Measured on 2026-10-16 by [saadharis](https://github.com/saadharis). Strata **0.1.44.1**
(source build, project `296b59d2`, engine 0.1.44.1), `qwen3.8-flash-next-iq3_xxs`
(GSQ-RCO IQ3_XXS), **two GPUs with an automatic layer split**, **262,144-token context**,
`--kv int8`, vision on, and **`"parallel": 2`** (two batch slots). Suite 1 of
`tools/strata_bench.py`, 3 runs per test, medians with ranges; plus a two-client
concurrency burst. Main limitation: the engine logged a **low-VRAM warning** during these
runs (see [Limitations](#limitations)) — the numbers below are unaffected, but that
configuration is over-subscribed on this machine.

This box is a 24/7 serving host for agent clients (Hermes, Claude Code, a local harness)
over one OpenAI/Anthropic-compatible endpoint, which is why the concurrency burst and the
`parallel` setting matter more here than the single-stream peak.

## Hardware and software

- **GPUs:** 2x NVIDIA GeForce RTX 3090, 24,576 MiB each, driver **580.178.04**,
  layer split automatic (CUDA0 dense + cache, **CUDA1 layers 23-47**). Both cards at
  their stock **420 W** power limits at the time of the run; clocks not captured.
- **CPU / RAM:** 11th Gen Intel Core i7-11700K, 8 cores / 16 threads; 62.6 GiB installed.
- **Storage:** the pack, GGUF shards and the MTP draft head all live on NVMe
  (`/mnt/fast`); models are not on the SATA volume.
- **PCIe:** the engine probes **6.2 GB/s** host->device on the split card -> `pcie_frac 0.17`
  (default 0.55), and 1.6 GB/s on the primary -> `pcie_frac 0.04`. `nvidia-smi` reports a
  narrow link at idle (gen1 x2 / x4); combined with those probes that is worth checking on
  a similar box, but it was not investigated here.
- **OS:** Linux `7.0.0-34-generic`. Engine built against **CUDA 12.6** for `sm_86`
  (`engine/BUILD.json`: source local, archs [86], vision gpu). Note the driver exposes
  CUDA 13.0 while the local toolkit is 12.6.
- **Background:** the host runs Docker stacks (Immich, Nextcloud, open-webui) and a
  persistent local-engine bridge; no other GPU work ran during the suite, but nothing was
  isolated or pinned.

## Model and configuration

- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` (the ISTA-DASLab GSQ-RCO family by filename),
  two GGUF shards in `/mnt/fast/flash-next/IQ3_XXS/`, pack
  `/mnt/fast/strata-data/packs/iq3_xxs`. Shard sizes and hashes are **not recorded** here.
- Expert profile `/mnt/fast/Strata/data/expert-profile.bin`; the automatic split's startup
  line reports it holds **19,942 of 24,576 profiled pairs (~99.4% of the routed mass)**,
  chosen as the best of 46 placements, `K=23`, predicted **24.4 ms per decode window**.
- Expert cache `auto` -> **6,018 slots (13.7 GiB level)** on CUDA1; the split card gets
  5,846 slots (10.56 GiB).
- MTP draft head `/mnt/fast/strata-data/mtp/rt`, `--spec 4 --spec-min-p 0.5`.
- Abliteration control vector applied in `--cvec-mode project`, layers 4-44.
- `--max-context 262144`, `--kv int8`, `--prefill auto`, `--vram-reserve-mib 700`,
  `--remote-expert-opt`, `--vision` (separate `strata-vision` binary, mmproj BF16).
- **`"parallel": 2`** -> the engine starts with `--batch 2`; `/v1/status` reports
  `concurrency: {serving: 2, requested: 2}`.
- The suite was pointed at the **already-running** server (`--url http://127.0.0.1:8080`),
  so model loading is not timed. Full config and an engine log excerpt are in this folder.

## Method

`tools/strata_bench.py --url http://127.0.0.1:8080 --clients 2 --long --runs 3`, thinking
off for the suite, temperature 0. One warm-up chat and a 1K prompt are discarded first, and
every prompt carries a unique tag so the conversation cache cannot help the fresh-prompt
tests. Speeds are the engine's own timings; first-token times are measured by the client
over HTTP.

## Results

| Configuration | Actual prompt tokens | Reused | Generated | Runs | Prompt tok/s (median, range) | Decode tok/s (median, range) | TTFT s (median, range) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 | 9 | not measured | **98.3** (81.3-112.5) | 0.33 (0.30-0.36) |
| 4K prompt, fresh | 4019 | 0 | 24 cap | 3 | **440.7** (439.8-440.7) | not measured | 9.14 (9.14-9.16) |
| 16K prompt, fresh | 16058 | 0 | 24 cap | 3 | **917.2** (916.8-917.2) | not measured | 17.53 (17.53-17.54) |
| 32K prompt, fresh | 32049 | 0 | 24 cap | 3 | **1104.9** (1104.7-1105.1) | not measured | 29.04 (29.04-29.05) |
| agent session, last turn | 14594 | see below | up to 64 | 3 | see below | **111.9** (96.6-149.0) | 8.46 (7.52-8.85) |

Prompt throughput **rises with length** here (440 -> 917 -> 1,105 tok/s across 4K -> 16K -> 32K)
because fixed per-request overhead is amortised, so the 4K figure should not be read as this
machine's long-context prefill rate. Cross-checking the engine's own `strata serve: prompt`
lines for fresh requests of the same prose text gives **1,221.5 tok/s at 56,755 tokens**,
and historic lines on the previous engine version (0.1.39) show **1,220.9-1,246.9 tok/s at
118K-133K** — i.e. prefill is unchanged across those versions.

**Decode caveat:** the suite caps generation at **24 tokens** for the prompt tests, and a
~24-token answer understates decode by design (it samples only the first window). The
trustworthy rows above are short chat (100 tokens) and the agent session (64/turn). A
300-token run on the same box measured **120.2 tok/s** where 24-token rows read 69-96.

### Agent session (conversation cache)

Cache hit over the follow-up turns **68.7%** (68.5-68.7), with only **73 tokens** (73-143)
of earlier turns read again per session; the 13,729 new file tokens are read once. First
turn, nothing cached: first token 4.29 s (4.26-4.31).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token s | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 143.4 | 4.262 | 81.6 |
| 1 | 3761 | 640 | 3121 | 363.1 | 8.616 | 97.2 |
| 2 | 5294 | 3754 | 1540 | 204.9 | 7.534 | 104.6 |
| 3 | 8738 | 5358 | 3380 | 386.0 | 8.78 | 106.2 |
| 4 | 11664 | 8801 | 2863 | 339.5 | 8.457 | 110.2 |
| 5 | 14617 | 11718 | 2899 | 343.7 | 8.46 | 96.6 |

### Concurrency: 2 clients

**71.4 tok/s total** (63.7-77.4); TTFT p50 0.799 s, p95 1.166 s; request latency p50
3.574 s, p95 4.017 s; **errors 0**. With `parallel: 2` the two requests decode together
rather than queueing; a request beyond the two slots waits its turn.

Worth noting for anyone considering `parallel`: with two clients busy, per-request decode
sits around **40-51 tok/s** each (batch slots decode without MTP drafts) while the total is
preserved — the gain is that nobody waits for a whole answer. A request **alone** still
takes the solo path at full speed: measured **107.6 / 119.4 tok/s** on this box with
`parallel: 2` active, against 98.3 tok/s in the suite.

## Limitations

- **The configuration is over-subscribed on VRAM, and the engine says so.** With
  `parallel: 2` at 262,144 context (each batch slot costs ~3.49 GiB of expert cache at this
  context), the engine logged:

  ```
  strata serve: CUDA1: 356 MiB of VRAM free with everything loaded - LOW: its window graphs
  are captured on first use and need room too, and a full card can stall a request;
  add --vram-reserve-later-mib 1112 to the config's args (or lower --max-context)

  strata serve: WARNING: only 446 MiB of VRAM free at the start of this request (below
  512 MiB): when the card is this full the driver can page GPU memory to system RAM, decode
  falls to a few tok/s and a window may never finish (#2065). Add --vram-reserve-mib 768 to
  the config's args, or lower --max-context / --batch
  ```

  **No degradation or failure was observed in these runs** — decode, prefill and the
  concurrency burst were all healthy — but the warning is real and is the most useful thing
  in this report for a similar 2x24 GB box: at 256K, two batch slots plus the default
  expert cache does not leave the 512 MiB the engine wants free. `--vram-reserve-mib 768`
  and `--vram-reserve-later-mib 1112` (the engine's own advice), or `--batch 1`, or a lower
  context, are the fixes.
- **One machine, one run of the suite**, desktop in normal use, power limits and clocks not
  captured. Wall-clock jitter on a machine that also serves agent traffic.
- The suite's prompt tests **cap generation at 24 tokens**, so decode is not measured there.
- This is a **speed suite only**: no correctness or recall check was run. An earlier
  `--clients 2` burst of the same shape returned 0 errors, but a longer soak was not done.
- Shard sizes/hashes, pack hash and clock/power capture are absent; treat the configuration
  as described rather than as fully reproducible.

## Files

| File | What it is |
| --- | --- |
| `README.md` | this report |
| `results.json` | per-request data for every test in the suite |
| `strata-iq3_xxs.json` | the exact model config used |
| `engine-BUILD.json` | how the engine was built (source, archs, CUDA dirs) |
| `engine-log-excerpt.txt` | engine log lines: per-request timings, expert-cache hit rates, the split placement and the VRAM warning |
