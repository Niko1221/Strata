# Community benchmark on AMD Radeon RX 7900 XTX

Measured on 2026-11-08 with `tools/strata_bench.py --long` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. This is a pre-protocol quick report against an already-running server, not a conforming standard-protocol run (the protocol counts a second run against the same engine as cache reuse). Needle recall was run against the same server afterwards.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XTX (gfx1100), 24 GiB VRAM (24560 MiB reported; 23.34 GiB used by the resident model before and after the suite)
- CPU: AMD Ryzen 9 3900X 12-Core Processor (12 cores, 24 threads, no AVX-512); installed RAM: 31.2 GiB (28 GiB used at the run's memory snapshot — the model is resident-expert cached in RAM); storage: NVMe SSD (holds the model pack and expert cache)
- PCIe link: 16 GT/s, width x16 (Gen4, `lspci` LnkSta)
- GPU power limit: cap set to 321 W immediately before the run (stock default 261 W), unchanged during the run; average draw read 10.0 W idle before and 69.0 W after the run
- OS: Linux 7.2.15-1-omarchy; GPU driver: amdgpu 7.2.15-1-omarchy on ROCm 7.2.4 (HIP; CUDA not applicable)
- Strata: project 0.1.46.2, commit cbfd5b92 (source checkout for the tools); engine 0.1.57 prebuilt release binary, component `0.1.57-linux-x86_64-hip-rx7900`
- Background workloads: none. The suite ran detached during an idle window with no interactive sessions; the engine log during the run contains only the benchmark's requests, so client first-token times are not inflated by other traffic. No other program used the GPU.

## Model and configuration

- Model and quantization: qwen3.8-flash-next-gyro-gyro-m (custom Gyro pack `packs/gyro-gyro-m`, single GGUF `Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf`, TQ2_0, used as both `--native` and `--ple-gguf`)
- Context: 131072 tokens; parallel slots: 1; KV: int8
- Launch command of the already-running server (from the engine process argv):
  `strata --serve --pack packs/gyro-gyro-m --native <models>/gyro-Gyro-M/Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf --ple-gguf <same file> --expert-profile <engine component data>/expert-profile.bin --expert-cache auto --prefill auto --max-context 131072 --kv int8 --resident-experts --session-min-free-mib 2048 --draft dflash2 --draft-model <packs>/dflash2/qwen3.8-flash-next-dflash2.q4_0-selq4.pack --draft-fallback none --mtp-draft-vocab <packs>/mtp/rt/draft_vocab.bin --spec 6 --draft-conf 0.5`
- Engine-reported settings (`engine_info` in `results.json`): drafting dflash2, spec 8, mtp_max 6, lookup 3, expert cache 15351 MiB (12329 slots, primary tier), prefill cap 8192, conversation cache 2304 MiB / 4 slots. No calibration or experimental speed projection was enabled beyond the above.
- Thinking off for the suite (tool default); temperature 0; the server had no API key at run time, so the needle call ran keyless.
- Quant repository revision not recorded (custom local pack).

## Method

Run by `python tools/strata_bench.py --long --url http://127.0.0.1:8081` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency: not run (the server was not launched with `--batch`)

Draft acceptance over all 36 successful requests: 1112 of 1782 drafts accepted (62.4 %). All 36 requests returned tokens; none failed or was skipped.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 75.0 (56.9-93.4) | 0.38 (0.31-0.54) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1184.1 (1180.6-1184.4) | not measured | 3.42 (3.42-3.43) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1566.0 (1563.8-1568.8) | not measured | 10.29 (10.27-10.30) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1548.9 (1547.8-1557.4) | not measured | 20.74 (20.62-20.75) |
| agent session, follow-up turns | 14670 (14658-14686) at the last turn | see below | up to 64 | 3 | see below | 81.1 (55.7-106.1) | 3.03 (2.50-3.22) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.5 % (69.4-69.5); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 13827 (13827-13828) new tokens of the files are read once); first turn (nothing cached) first token: 1.74 s (1.73-1.75).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 353.1 | 1.748 | 48.8 |
| 1 | 4092 | 627 | 3465 | 1099.7 | 3.176 | 62.5 |
| 2 | 5625 | 4155 | 1470 | 579.0 | 2.564 | 65.1 |
| 3 | 9058 | 5688 | 3370 | 1055.7 | 3.218 | 65.6 |
| 4 | 11815 | 9107 | 2708 | 914.3 | 2.989 | 102.8 |
| 5 | 14658 | 11839 | 2819 | 940.4 | 3.027 | 55.7 |

Per-request data: `results.json`. This run did not start the server, so no engine start-up log is attached.

## Correctness check

`tools/needle_bench.py --lengths 32k,128k --depths 10,50,90` against the same server, afterwards: 6 of 6 found (`needles.json`). Times 19.4-20.5 s at 32k prompts (31663-31664 tokens) and 42.9-83.5 s at 128k prompts (125020-125022 tokens). This is recall only; no other quality check was run.

## Comparison with the earlier IQ2_XS run on this machine

The same machine ran the same suite on qwen3.8-flash-next IQ2_XS on 2026-11-06 (`2026-11-06-community-rx7900xtx-iq2_xs/`) with the same power cap (321 W) and the same draft configuration (dflash2 pack, `--spec 6 --draft-conf 0.5`). IQ2_XS measured prompt 1051/1385/1381 tok/s and chat decode median 80.3 tok/s at fresh 4K/16K/32K; this Gyro-M run measures prompt 1184/1566/1549 tok/s and chat decode median 75.0 tok/s.

## Limits

Speed only: no correctness check beyond the needle recall was run. The conversation uses the model's own replies, so token counts differ a little between runs. Power cap was raised from the 261 W stock default to 321 W before the run, so compute-bound prompt throughput is not comparable with stock-cap runs; decode is memory-bandwidth-bound, so the cap change buys little there. Single machine, single engine session. Total time of the suite: 187 s; the needle run took about 5 further minutes.
