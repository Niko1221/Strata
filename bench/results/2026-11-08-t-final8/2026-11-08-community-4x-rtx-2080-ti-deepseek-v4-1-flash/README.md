# Community benchmark on NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti

Measured on 2026-11-08 with `tools/strata_bench.py` (suite 1, 5 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti, 88 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz (14 cores, 28 threads); installed RAM: 121.4 GiB; storage: nvme0n1 Lexar SSD NM620 2TB nvme. The CPU has **no AVX-512** (Broadwell-EP: AVX2 and BMI2 only), so the engine's AVX-512 dequantise-and-dot paths for the CPU lane do not run here and the lane's 14-thread AVX2 path is the one measured below.
- OS: Linux 7.0.0-42-generic; GPU driver: 595.91.07; CUDA (driver supports up to): 13.2
- Strata: project 0.1.55, commit c62d5e3a; engine ds4-0.1
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=102.91, pcie=gen3 x16, pstate=P2, temp_c=64
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=133.48, pcie=gen3 x16, pstate=P2, temp_c=78
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=129.65, pcie=gen3 x16, pstate=P2, temp_c=71
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=148.20, pcie=gen3 x16, pstate=P2, temp_c=64
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: deepseek-v4.1-flash
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0; three engine environment variables on top of the arguments (this folder's `env.txt` and `config.json`):
  - `DS4_ASYNC_T=8` - keep the drafter's draft-verify window asynchronous. The window is T=6, so a value below 6 runs every verify round synchronously; this is also the engine's own default, so it only overrides an older preset.
  - `DS4_DEFAULTS=075` - the measured 2.00 bpw speed profile of the expert path. Because it changes the model's numbers, it is an explicit opt-in token, never a default (see *Correctness and limitations*).
  - `DS4_ATTN_SPLIT=0` - the default attention split; set explicitly only so the run cannot pick up an experimental value from the environment.

```text
strata-dsv41 --pack deepseek-v4.1-flash-ist-das-2.00bpw --max-context 131072 --eos 1 --vision on --dspark auto --dspark-conf 0.5 --tp 4 --experts 5 --threads on
```

### Model identity

The pack is ISTA-DASLab's 2.00 bits-per-weight build of **DeepSeek-V4.1-Flash-1017** (GSQ-RCO, IQ2_XXS for the routed
experts). What was measured is pinned by these files and hashes:

- weights, four shards: `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf` … `-00004-of-00004.gguf`;
- pack descriptor `ds4.txt` (the model's dimensions), sha256 `d10cf9e340d23bfa147c768a57101d01a39cb67928b66c915e02ecb17ad62ff1`;
- pack tensor index `tensors.txt`, sha256 `46197620c7113b7d661761ca29ef4ebf0a7aa03cda5cf7c2e5665e3c6d097395`;
- drafter `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf`, sha256
  `6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a` (the file the installer pins);
- Engram `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` and vision `mmproj-BF16.gguf`.

`--pack-info` on this pack reports 132.1 GB of weights at 1.94 bits a weight (544 G parameters), 8.60 MB a routed expert,
7 files, and expert tensors by type IQ1_M 60, IQ2_XS 18, IQ2_XXS 42. The pack is metadata that points at those GGUF files;
it needs the same four shards and the two shared files to run.

## Method

Run by `python tools/strata_bench.py --long --runs 5` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 5 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 5 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 5 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 32 (29-38) | 0 | 100 (100-100) | 15 | not measured | 39.1 (26.4-51.8) | 0.43 (0.35-0.53) |
| 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 5 | 539.4 (467.5-556.4) | not measured | 7.66 (7.43-8.84) |
| 16K prompt, fresh | 16562 (16562-16562) | 0 (0-0) | 24 cap | 5 | 905.4 (871.3-917.3) | not measured | 18.34 (18.10-19.05) |
| 32K prompt, fresh | 33126 (33126-33126) | 0 (0-0) | 24 cap | 5 | 1035.0 (984.6-1052.0) | not measured | 32.09 (31.57-33.73) |
| agent session, follow-up turns | 12672 (12669-12682) at the last turn | see below | up to 64 | 5 | see below | 31.4 (25.5-39.3) | 11.93 (8.26-15.27) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token. The suite ran with no failed
request and no skipped test (`failures: []` in `results.json`).

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt`. The issue/pull-request body for
this report is `PUBLISH.md`.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.6 % (69.6-69.6); earlier-turn tokens read again per session: 1 (0-2) (earlier turns that the cache did not hold; the 11796 (11796-11798) new tokens of the files are read once); first turn (nothing cached) first token: 5.60 s (5.56-6.79).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 602 | 0 | 602 | 88.9 | 6.79 | 22.5 |
| 1 | 3336 | 637 | 2699 | 199.6 | 13.554 | 29.2 |
| 2 | 4850 | 3397 | 1453 | 150.1 | 9.714 | 25.5 |
| 3 | 7959 | 4912 | 3047 | 200.0 | 15.271 | 31.4 |
| 4 | 10037 | 8017 | 2020 | 175.8 | 11.531 | 28.3 |
| 5 | 12671 | 10094 | 2577 | 188.6 | 13.71 | 35.4 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

**This is not a standard-protocol result.** It was produced by `tools/strata_bench.py` (the guide's *quick look* path), not by
`strata bench run` from [BENCHMARK_PROTOCOL.md](../../../../docs/BENCHMARK_PROTOCOL.md). It is published as a community report,
which the guide allows for measurements made before or outside the protocol; the numbers are the engine's own timings and are
reproducible with the commands below.

Not measured: this is a speed suite, and it checks no answer's correctness. Two things are known about the 075 profile from
separate sessions and are recorded here rather than implied: on `data/quant_quality/v1` (filtered to this pack's vocabulary)
it agrees with the reference on the model's top token at **87.07 %** of 4,995 positions with mean KL **0.0371** (p99 0.3045),
and a needle test at 32K and 128K found the hidden code word **6 of 6 in both profiles** (075 and the 026 reference), with the
code word itself as the answer. So the numeric difference is real and the recall cost is not measurable by that test.

Also not tested: the HIP/ROCm build of this engine (no HIP toolchain on this machine), the Windows packaged build (no Windows
machine), and any second graphics card family. The conversation uses the model's own replies, so token counts differ a little
by the decoded words between runs. Machine details come from the server, `nvidia-smi` and the OS; background load is not
recorded. Total time of the suite: 736 s.

## How this compares with the same build and the same model, tuning off

Same box, same pack, same engine commit and the same suite; the `async 8` arms and their baselines alternate in time on
2026-11-08, and the "tuning off" row is the day before (2026-11-07) because that is when the untuned baseline was measured —
it is quoted for context, not as a same-day control. "Tuning off" is the same engine with `--threads off` and no
`DS4_DEFAULTS` (the FAT/slab path off).

| Configuration | chat decode | agent decode | 4K prompt | 16K prompt | 32K prompt |
| --- | --- | --- | --- | --- | --- |
| tuning off, 2026-11-07 (`--threads off`, no 075) | 31.0 | 28.3 | 426.0 | 673.2 | 764.8 |
| this report (075 + `--threads on`, `DS4_ASYNC_T=4`) | 34.0 | 29.9 | 540.8 | 913.5 | 1040.5 |
| **this report, adopted config (`DS4_ASYNC_T=8`)** | **39.1** | **31.4** | 539.4 | 905.4 | 1035.0 |

The last row is the file set published here (`results.json`, `config.json`). The `async 8` step is +15 % chat and +5 % agent
decode over the row above it, with prompts unchanged, measured across four `=8` arms and four baselines that alternate in time
(`bench/results/2026-11-08-t-async8*` and `-t-final*`). The untuned row is `bench/results/2026-11-07-t-075base`; every
number in this table is a median in that folder's `results.json`.

## Decode at a long context, in milliseconds

A separate 128-token decode at 32K on this configuration (drafter loaded, `--ids` path, so the per-token split is printed):
**34.96 ms a token, i.e. 28.4 tokens/s**, of which the CPU lane is **12.5-15.3 ms** (36-44 %), the GPU expert matvec 7.7 ms,
and the final device synchronisation 26.25 ms, which overlaps the rest rather than adding to it. The lane moves 223 MB of
expert weights a token at 15-17 GB/s, against 22.3-25.7 GB/s for the same kernels at the same thread count. An earlier
"39.5 ms of CPU-lane wait" figure came from an 8-token decode right after a 32K prompt (12.93 tok/s) and does not describe
steady state.

## Reproducing this report

```sh
cd <strata-checkout>
export DS4_ASYNC_T=8 DS4_DEFAULTS=075 DS4_ATTN_SPLIT=0
.venv/bin/python strata.py serve --profile <this folder>/config.json   # same args and env as the run; HTTP API on :8080
.venv/bin/python tools/strata_bench.py --url http://127.0.0.1:8080 --long --runs 5 \
    --out bench/results/$(date +%F)-reproduce
```

`tools/strata_bench.py --config <this folder>/config.json --long --runs 5` also works: it starts the server itself, as it did
for this report. The config's `exe`, `cwd`, `lib_dirs`, `log` and `--pack` paths in this folder are **placeholders** (`<strata-checkout>`,
`<packs>`), not this machine's own paths: put your own checkout and pack location there. The arguments, the environment and
the suite are the parts that transfer unchanged.

The pack is ISTA-DASLab's `deepseek-v4.1-flash-ist-das-2.00bpw` (2.00 bits per weight, 132.1 GB on disk, 8.60 MB a routed
expert), and the drafter is DSpark `Q8_0` (`dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf`, the file the installer pins).

## Limits

This box's chat decode drifts with page-cache warmth: identical baselines read 34.0, 35.5, 37.5 and 36.3 chat across one
afternoon, so chat is quoted with its full range and every comparison above is made against an arm measured within the same
hour. Agent decode is the stabler metric (28.3-30.5 in every baseline, 31.0-32.0 in every `=8` arm). Prompt medians moved
less than 1 % between arms of the same length.
