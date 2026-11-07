# Community benchmark on AMD Radeon RX 7900 XTX (gfx1100)

Measured on 2026-11-06 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed suite plus a 6-case needle recall check.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XTX (gfx1100), 24 GiB VRAM; PCIe Gen4 x16 (LnkSta 16GT/s x16); liquid-cooled (shared loop)
- CPU: AMD Ryzen 9 3900X 12-Core Processor (12 cores, 24 threads, no AVX-512); installed RAM: 31.2 GiB; storage: NVMe SSD (Btrfs on LUKS)
- OS: Linux 7.2.15-1-omarchy; amdgpu driver from the kernel; ROCm 7.2.4; headless boot, no compositor or other GPU consumers
- Strata: project 0.1.46.2, commit cbfd5b92; engine 0.1.57 — prebuilt HIP rx7900 release binary from the signed component feed, not a source build
- Power limits: GPU package power cap raised from 261 W to 321 W immediately before the run; observed average package power during the run: 200 W
- Background workloads: none besides the agent session driving this benchmark (see Method)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq2_xs — GSQ-RCO IQ2_XS requant of Qwen3.8-Flash-Next, two GGUF shards (`Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf` + shard 2, the shared PLE/expert file); quant repository revision not recorded
- Draft/spec: dflash2 draft pack `qwen3.8-flash-next-dflash2.q4_0-selq4.pack` with `--draft-fallback none`, MTP draft vocab `mtp/rt/draft_vocab.bin`, `--spec 6 --draft-conf 0.5`
- Expert profile: `expert-profile.bin` from control 0.1.57 data (generic profile, not coder-tuned)
- Vision encoder: not enabled
- Context: 131072 tokens; parallel slots: 1; KV int8; `--expert-cache auto --prefill auto --resident-experts`
- Settings: the launch profile's own configuration, unchanged; thinking off for the suite; temperature 0

```text
Engine launch (run config launch/iq2_xs-128k.json, credentials removed):
strata --serve --pack <packs>/iq2_xs --native <models>/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
  --ple-gguf <models>/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf \
  --expert-profile <control 0.1.57 data>/expert-profile.bin --expert-cache auto --prefill auto \
  --max-context 131072 --kv int8 --resident-experts --draft dflash2 \
  --draft-model <packs>/dflash2/qwen3.8-flash-next-dflash2.q4_0-selq4.pack --draft-fallback none \
  --mtp-draft-vocab <packs>/mtp/rt/draft_vocab.bin --spec 6 --draft-conf 0.5
Measured through the HTTP API of the already-running server; memory: VRAM ~25.2 of 25.75 GB used at the resident snapshot, engine RSS ~14.3 GB (weights are mmap'd), MemAvailable ~3-6 GiB during inference.
```

## Method

Run by `python tools/strata_bench.py --url http://127.0.0.1:8081 --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

Two agent-session TTFT outliers are client-side contention, not model behavior: run 1 turn 0 (45.71 s) and run 3 turn 5 (46.96 s) overlapped concurrent chat requests from the agent session driving the benchmark on the same single-slot engine. All engine-side throughput timings are unaffected; the other TTFTs show no such gaps. The chat decode range (50.4–104.8 tok/s) reflects warm-up: run 1 measured the cold expert cache / draft acceptance, runs 2–3 the warmed state.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 80.3 (50.4-104.8) | 0.34 (0.29-0.50) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1051.0 (1027.2-1071.1) | not measured | 3.85 (3.77-3.93) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1385.5 (1369.5-1396.6) | not measured | 11.62 (11.53-11.75) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1381.4 (1376.8-1381.7) | not measured | 23.24 (23.24-23.32) |
| agent session, follow-up turns | 14683 (14683-14702) at the last turn | see below | up to 64 | 3 | see below | 84.7 (65.2-110.8) | 3.47 (2.72-46.96) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.4 % (69.4-69.5); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 13827 (13827-13828) new tokens of the files are read once); first turn (nothing cached) first token: 2.05 s (2.00-45.71).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 320.5 | 45.712 | 66.6 |
| 1 | 4105 | 640 | 3465 | 928.3 | 3.756 | 65.2 |
| 2 | 5638 | 4168 | 1470 | 519.5 | 2.851 | 75.9 |
| 3 | 9071 | 5701 | 3370 | 902.6 | 3.758 | 71.3 |
| 4 | 11826 | 9118 | 2708 | 793.6 | 3.436 | 82.9 |
| 5 | 14702 | 11883 | 2819 | 819.1 | 3.466 | 84.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Correctness check: `tools/needle_bench.py --lengths 32k,128k --depths 10,50,90` against the same server — 6 of 6 found: 32k (31,663–31,664 prompt tokens, 22–34 s per request) and 128k (125,020–125,022 prompt tokens, 142–145 s per request, request time includes generation). Per-case data: `needles.json`. A needle test measures recall on those inputs, not overall model quality.

Limitations: speed suite only plus the needle check above. The conversation uses the model's own replies, so token counts differ a little between runs. Total time of the speed suite: 337 s. Power cap was 321 W at the time of measurement (261 W stock, raised immediately before the run); results are not comparable with runs at the stock cap for compute-bound prompt throughput. Decode is memory-bandwidth-bound, so the cap change buys little there.
