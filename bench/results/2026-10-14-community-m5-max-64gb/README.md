# Community benchmark: Mac Studio M5 Max 64 GB, experts streamed from the SSD

Measured on 2026-10-14 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges) on the release
engine. Four runs on the same day, one after the other, each from an emptied file cache (`sudo purge`) with nothing else
running: the setup's own configuration and a tuned one, for IQ3_XXS and for Unsloth's UD-Q4_K_XL. Each subfolder is the
tool's own report (`README.md`, `results.json`, `engine-log-excerpt.txt`). The tool labels the machine "CPU only, VRAM
not read": it does not read Apple GPUs; the runs are on the Metal engine.

## Hardware and software

- Mac Studio (Mac17,14), Apple M5 Max: 18-core CPU, 40-core GPU, 64 GB unified memory, internal 1 TB SSD
- macOS 27.0.1; `iogpu.wired_limit_mb=58000` (raised from the default for the 40 GB arena of the tuned UD-Q4_K_XL run)
- Strata v0.1.44 (ed4773eb), built from source by `./setup.sh` (Metal engine 0.1.44-metal), engine unchanged
- Packs from setup: IQ3_XXS (GSQ-RCO) and UD-Q4_K_XL (Unsloth). Both are larger than this Mac keeps resident, so setup
  configured both to stream their experts from the SSD (`STRATA_METAL_STREAM=1`) without the MTP draft layer.
- The tuned runs use an MTP draft layer with Q8_0 experts, made from setup's fetched MTP tensors with
  `tools/mtp_pack.py --src <Strata-data>/mtp --experts q8_0 --out <Strata-data>/mtp/mtp-q8_0.gguf`.

## Results (setup's configuration vs tuned, same day)

| | IQ3_XXS setup | IQ3_XXS tuned | UD-Q4_K_XL setup | UD-Q4_K_XL tuned |
|---|---:|---:|---:|---:|
| short chat, decode tok/s | 45.4 (32.8-47.0) | 48.7 (42.4-68.7) | 29.7 (27.4-30.4) | 42.8 (29.6-46.0) |
| 4K prompt, prompt tok/s | 1434.5 | 1677.0 | 678.3 | 1048.4 |
| 16K prompt, prompt tok/s | 1434.3 | 1663.6 | 700.8 | 1049.6 |
| 16K prompt, first token (s) | 11.22 | 9.68 | 22.95 | 15.33 |
| agent session, decode tok/s | 42.4 (33.9-46.0) | 59.9 (36.0-65.6) | 26.4 (17.4-32.0) | 44.8 (38.7-53.4) |
| agent session, first token (s) | 2.76 | 2.06 | 10.55 | 5.19 |
| agent session, cache hit | 68.7 % | 68.6 % | 68.7 % | 68.6 % |
| suite time (s) | 140 | 119 | 339 | 204 |

Tuned settings (the rest is the setup's configuration):

| | IQ3_XXS | UD-Q4_K_XL |
|---|---|---|
| `--mtp <Strata-data>/mtp/rt --spec` | 4 | 3 |
| `STRATA_METAL_MTP_GGUF` | `mtp-q8_0.gguf` | `mtp-q8_0.gguf` |
| `STRATA_METAL_ARENA_GB` | 30 | 40 |
| `STRATA_METAL_PROMPT_CHUNK` | 4096 | 2048 |
| `STRATA_ROUTE_TAIL_SKIP` | 0 | 0 |

## How the tuned settings were chosen

From our own sweep on this Mac the same day (our client, greedy, 256 new tokens, three prompts, median of 5, every
configuration from an emptied file cache, the arena it got recorded, pinned with `STRATA_METAL_ARENA_GB` where a size is given):

- **MTP pays while streaming on 64 GB.** IQ3_XXS with a 27 GB arena: 53.2 tok/s with one token per step, 81.4 / 68.3 / 85.6
  with `--spec 4` and the Q2_0 draft (code / prose / structured). `--spec 5` was not faster. UD-Q4_K_XL is SSD-read bound: `--spec 3` was best
  for it, `--spec 4` slower at the same arena.
- **A Q8_0 draft layer accepts more than Q2_0.** IQ3_XXS `--spec 4`, 26 GB arena: 81.5 / 68.3 / 74.1 with Q2_0, 84.5 / 74.2 /
  81.8 with Q8_0. A Q4_0 draft layer stops the Metal engine (#2504).
- **The arena sets the speed of a streamed pack.** UD-Q4_K_XL `--spec 3` with the Q8_0 draft: 25.9 / 27.7 / 25.8 tok/s at
  20 GB, 34.2 / 36.1 / 36.7 at 28 GB, 50.5 / 40.2 / 46.5 at 36 GB, 56.7 / 51.8 / 52.3 at 40 GB, no swap, about 16 % of RAM
  free at the end; three repeats at 40 GB agree within 0.3 %. IQ3_XXS levels off from 26 GB (84 / 74 / 82-89 at 26-34 GB).
- **A longer prompt chunk reads prompts faster when streaming.** UD-Q4_K_XL at 38 GB, 6,959-token prompt: 504 tok/s at
  1024 (the default below 96 GB), 708 at 2048 with decode unchanged, 1,105 at 4096 with decode on code down from 53.9 to
  43.0 (its 4.25 GB of prompt buffers sit in the arena).
- **Same answers.** In every run whose output we hashed, the greedy text of 256 tokens on the three prompts was
  byte-identical with and without MTP, at every `--spec` and arena, 5 of 5 runs each (IQ3_XXS, IQ3_S, UD-IQ4_XS, UD-Q4_K_XL,
  UD-Q5_K_XL).
- `STRATA_ROUTE_TAIL_SKIP` on or off gave the same speed in our sweep (for example UD-Q4_K_XL without MTP, skip off 26.4 / 28.4 / 26.6
  vs the setup's configuration 25.8 / 27.6 / 26.5 tok/s), so the tuned runs keep every expert.

## Not tested

Other chips or memory sizes; 0.1.44.1 (its Mac fix is for chips other than M4 / M5); contexts over 32,768 tokens;
concurrent requests; any quality check beyond the byte-identical outputs above. A short request from another
conversation drops the conversation cache on Metal (#2502); the agent sessions here are one conversation each.
