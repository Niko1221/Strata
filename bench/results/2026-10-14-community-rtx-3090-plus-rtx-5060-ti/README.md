# Community benchmark on an RTX 3090 + RTX 5060 Ti pair (expert tier on the second card)

Measured on 2026-10-14 by [txiki739](https://github.com/txiki739). Strata **0.1.43.1** engine
(source build of the `eddoursul/Strata` `custom` line) with the upstream **0.1.44.1** server,
Unsloth **UD-IQ4_XS** with a native-experts pack, 200,192-token context: an RTX 3090 as the
main card and an RTX 5060 Ti as the expert tier. `tools/strata_bench.py --long` (3 runs per
test, medians) plus a 32K/128K needle sweep (found 6 of 6). Main limitation: one machine and
one engine session with warm caches, and both cards' power limits set below their defaults.

## Hardware and software

- **GPUs:** NVIDIA GeForce RTX 3090, 24 GiB (main card, engine GPU 1) + NVIDIA GeForce
  RTX 5060 Ti, 16 GiB (expert tier, `--second-gpu 0`); PCIe 4.0 (3090 at x16, 5060 Ti at x8);
  driver 615.78.08. **Power limits 250 W (3090; default 350, max 375) and 150 W
  (5060 Ti; default 180)**, set below the defaults by the author for noise and thermals;
  clocks not captured.
- **CPU / RAM:** AMD Ryzen 7 5700X, 8 cores / 16 threads, **no AVX-512** (the engine logs
  `this CPU has no AVX-512: the expert kernels run on AVX2`), 7 pool workers; 125.7 GiB
  installed RAM. Most experts live in RAM (the engine reports a 56,762 MiB arena).
- **Storage:** Crucial P3 Plus 2 TB NVMe; the OS, the model shards, the pack and the engine
  all on that one drive.
- **OS:** Linux 7.2.9 (CachyOS); CUDA 13.4 (`V13.4.92`, the build toolkit).
- **Strata:** engine built from [eddoursul/Strata](https://github.com/eddoursul/Strata)
  `custom` at `c4bcb6e8` (the upstream 0.1.43.1 engine line plus that fork's work), with
  local commits on top that touch configuration and measurement files only (the tree the
  tool recorded is `0e81b4db`; no engine code differs from `c4bcb6e8`). Engine version
  0.1.43.1, binary md5 `182402f55f2ddc5a1a8a3d2459aa4a81`. Server: the upstream 0.1.44.1
  `serve/server.py` (an HTTP layer only: the speeds below are the engine's own timings).
- **Background:** nothing else on the two cards during the run; the desktop session of the
  same box was idle; no build or other benchmark ran.

## Model and configuration

- **Model:** Unsloth `Qwen3.8-Flash-Next-UD-IQ4_XS` (three GGUF shards, `...-00001` to
  `...-00003-of-00003.gguf`) ran as a native-experts pack `ud-iq4_xs` with an expert profile
  (`expert-profile.bin`); MTP draft head at `mtp/rt-es`. The served model id is a local
  alias, shown here as `ud-iq4_xs-dual` (the private label is withheld).
- **Context** 200,192; **KV** int8 with the elastic grow (`--kv-grow`); **expert cache**
  auto -> 7,265 slots / 16,384 MiB (16.38 GiB) on the second card, the experts themselves in
  RAM; 8 GiB pinned per card; **prefill** 32,768; **speculation** `--spec 4 --spec-min-p 0.5
  --spec-lookup 16`; **adaptive tier** `--adapt-every 1 --adapt-swaps 192`; PLE IO from RAM.
  `STRATA_IQ256_GATHER=1` and `STRATA_POOL_FUSED=1` in the config's `env`. No calibration,
  no experimental speed projection, vision not built.

The configuration, with local names removed, is in
[strata-ud-iq4_xs-dual.json](strata-ud-iq4_xs-dual.json); the launch line as recorded:

```text
strata --pack ud-iq4_xs --native Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill 32768 --feed-max 52 --spec 4 --spec-min-p 0.5 --spec-lookup 16 --adapt-every 1 --adapt-swaps 192 --mtp rt-es --max-context 200192 --kv int8 --pcie-mode kernel --pcie-frac 0.25 --second-gpu 0 --main-gpu 1 --head-split 0.45 --pool-workers 7 --second-gpu-reserve-mib 512 --second-gpu-prefetch-mb 6 --kv-grow --vram-pin-gib 8 --second-gpu-pin-gib 8 --ple-io ram
```

## Method

- `python tools/strata_bench.py --long` against a server already running (started a few
  minutes earlier with the author's usual launcher): model loading is **not** timed and the
  expert cache was warm. The suite discards one warm-up chat and a 1K prompt first, and every
  prompt carries a unique tag, so the fresh-prompt tests cannot reuse a cached prefix.
- short chat: three questions, up to 100 new tokens, 3 runs; 4K / 16K / 32K prompts:
  generated word lists (24 new tokens), 3 fresh prompts each; agent session: a system prompt
  and four tools, then 5 turns that each add about 1,500 words of a file and continue from
  the model's own earlier answers, 3 sessions. Thinking off for the whole suite.
- needles: `python tools/needle_bench.py --url http://127.0.0.1:<port> --lengths 32k,128k
  --depths 10,50,90`, haystack built from the repository's own text files, 40 max tokens,
  thinking off. Run right after the suite, one request after another against the same
  engine, which reuses whatever prompt prefix is already cached (noted below).
- `results.json` holds every request and the machine facts; the machine block was completed
  and scrubbed by hand where the tool records nothing (both cards listed, CUDA, storage and
  power limits added, a local host name and a private DNS name removed).

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 100.1 (84.2-123.8) | 0.27 (0.26-0.37) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1722.8 (1690.6-1789.6) | not measured | 2.35 (2.26-2.39) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2424.3 (2388.6-2426.3) | not measured | 6.65 (6.64-6.75) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 2417.6 (2393.1-2424.4) | not measured | 13.29 (13.26-13.43) |
| agent session, follow-up turns | 14595 (14567-14634) at the last turn | 68.8 % hit | up to 64 | 3 | see below | 124.1 (101.8-146.3) | 2.48 (1.81-2.63) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.
`not measured` and `24 cap` are as the tool reports them (the prompt tests cap the answer at
24 tokens). Total time of the suite: 130 s.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.8-68.8); earlier-turn tokens read again per
session: 4 (4-4) (earlier turns that the cache did not hold; the 13,744 (13,743 - 13,744) new
tokens of the files are read once); first turn (nothing cached) first token: 1.79 s
(0.42-2.01). Per-turn data of run 1:

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 343.8 | 1.787 | 119.0 |
| 1 | 3719 | 634 | 3085 | 1223.8 | 2.54 | 123.6 |
| 2 | 5252 | 3782 | 1470 | 806.5 | 1.841 | 132.0 |
| 3 | 8801 | 5315 | 3486 | 1335.8 | 2.633 | 101.8 |
| 4 | 11685 | 8825 | 2860 | 1157.7 | 2.494 | 108.8 |
| 5 | 14595 | 11748 | 2847 | 1160.3 | 2.477 | 124.1 |

Per-request data: [results.json](results.json).

### Needle recall (32K and 128K)

| Length | Depth | Actual prompt tokens | Seconds | Found |
| --- | ---: | ---: | ---: | --- |
| 32k | 10 % | 32,789 | 12.6 | yes |
| 32k | 50 % | 32,789 | 14.0 | yes |
| 32k | 90 % | 32,790 | 12.2 | yes |
| 128k | 10 % | 128,486 | 49.9 | yes |
| 128k | 50 % | 128,484 | 45.3 | yes |
| 128k | 90 % | 128,484 | 33.0 | yes |

**6 of 6 found** (the code word was quoted back exactly). The six runs went to the same
engine one after another, and it reuses cached prefixes: at 128K the first run (depth 10 %)
is the closest to a fresh read (128,486 tokens in 49.9 s, about 2.6K tok/s wall), and the
later 128K runs re-read less of the prompt (hence 45.3 s and 33.0 s). The 32K runs
(12.2-14.0 s each) follow the same suite at 32K. Per-check data: [needles.json](needles.json).

Engine start-up lines from the console, with paths trimmed, are in
[engine-log-excerpt.txt](engine-log-excerpt.txt).

## Correctness and limitations

- This is a speed suite plus the recall check above; no quality benchmark was run, and a
  needle test measures recall on those inputs only. (In daily use the same engine also
  passes tool calls and literal-copy checks; that is not part of this report.)
- One machine, one engine session with warm caches: model loading is not timed and the
  expert cache was already filled (7,265 experts, 16.38 GiB). Cold-start numbers would be
  lower; on other days a warm restart of this pair took 25-65 s including the expert load.
- Both cards run below their default power limits (250 W / 150 W), so absolute speeds are
  below a stock pair; no clocks were captured and no telemetry was recorded.
- The 4K prompt reading slower than 16K/32K (1723 against 2418 tok/s) is as measured; it is
  one sweep family here, not a repeated finding.
- The agent session's token counts differ a little between runs because the conversation
  includes the model's own replies; the reuse numbers come from the answer's usage.
- Concurrency (`--clients`) was not tested: the server runs with one parallel slot.
- Storage and PCIe widths come from `nvidia-smi` and the OS (the widths are the cards'
  maximums): the NVMe is a consumer drive shared with the OS, and no PCIe probe figure was
  recorded.
