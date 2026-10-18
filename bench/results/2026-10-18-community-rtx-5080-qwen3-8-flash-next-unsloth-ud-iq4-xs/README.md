# Community benchmark on NVIDIA GeForce RTX 5080

Measured on 2026-10-18 by @ZanThredFolt with `tools/strata_bench.py` (suite 1, 5 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured over HTTP. This table is speed-only; needle-recall correctness was run separately on the same build and passed 6/6 (Appendix B).

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5080, 16 GiB VRAM
- CPU: AMD Ryzen 9 9950X 16-Core Processor (16 cores, 32 threads); installed RAM: 60.2 GiB; storage: NVMe, CT4000T705SSD3 (root device)
- OS: Linux 7.2.6-76070206-generic; GPU driver: 595.104.02; CUDA: 13.2
- Strata: project 0.1.45, commit 477ac13c; engine 0.1.45
- Background workloads and power limits: power limit 360 W; no browser, downloads, or other GPU tasks during the suite window; the host also runs a local agent client that was idle for the whole suite (Appendix C says how this was verified and discards one window that failed it)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq4_xs (Unsloth UD-IQ4_XS, 3 GGUF shards `Qwen3.8-Flash-Next-UD-IQ4_XS-0000{1,2,3}-of-00003.gguf`, pack `unsloth-ud-iq4_xs`)
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0
- Host-specific features enabled (the main reason to look at this report; full launch line below):
  - `STRATA_HC_Q8=1` (head/context path in Q8)
  - FP8 PLE sidecar: `--ple-gguf .../Strata-data/ple/ple-fp8.gguf`
  - MTP drafts: `--mtp .../Strata-data/mtp/rt`, `--spec 4 --spec-min-p 0.50`
  - `--kv int8 --kv-resident 32768 --resident-budget-gib 36 --pcie-frac 0.35`
  - adaptive expert routing: `--expert-profile .../data/expert-profile.bin --expert-cache auto --prefill auto --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97 --adapt-async 1 --prompt-cache-spread`
  - conversation cache: `--conversation-cache-mib 2048 --conversation-cache-slots 2 --conversation-cache-min-free-mib 4096 --conversation-cache-min-tokens 8192 --conversation-cache-park-gain 4096`
  - `--vision` (vision tower comes from the pack; no separate mmproj file)

Launch line (home paths replaced; no credentials involved):

```
<home>/Strata/engine/strata --serve --pack <home>/Strata-data/packs/unsloth-ud-iq4_xs \
  --native <home>/Strata-data/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  --ple-gguf <home>/Strata-data/ple/ple-fp8.gguf \
  --expert-profile <home>/Strata/data/expert-profile.bin \
  --expert-cache auto --prefill auto --spec 4 --mtp <home>/Strata-data/mtp/rt \
  --max-context 131072 --kv int8 --kv-resident 32768 --resident-budget-gib 36 \
  --vision --pcie-frac 0.35 --spec-min-p 0.50 --adapt-every 1 --adapt-swaps 160 \
  --adapt-decay 0.97 --prompt-cache-spread --adapt-async 1 \
  --conversation-cache-mib 2048 --conversation-cache-slots 2 \
  --conversation-cache-min-free-mib 4096 --conversation-cache-min-tokens 8192 \
  --conversation-cache-park-gain 4096 \
  --expert-profile-save <home>/Strata/data/expert-profile.bin
```

## Method

Run by `python tools/strata_bench.py --long --runs 5` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

Before the suite, an external orchestrator additionally: waited for GPU utilization under 5 % across five consecutive samples; issued fresh 4K probes until two consecutive prompt-speed reports agreed within 10 % (they had last been 510–881 tok/s just before); issued fresh 32K probes until two agreed within 15 % (1236/1284 tok/s); then two short decode chats and a 15 s settle. GPU utilization, temperature, throttle-reason bits, `MemAvailable`, `SwapFree` and `pswpout` were logged every 2 s/5 s across the whole window (Appendix C).

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 5 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 5 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 5 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency: not run — the server holds one slot and no `--batch`, so simultaneous requests would measure queueing, not concurrent throughput

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 15 | not measured | 72.9 (57.2-83.5) | 0.40 (0.35-0.44) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 5 | 572.8 (450.7-628.3) | not measured | 7.07 (6.45-8.96) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 5 | 966.8 (928.6-1173.0) | not measured | 16.66 (13.74-17.35) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 5 | 568.4 (264.5-664.2) | not measured | 56.46 (48.33-121.27) |
| agent session, follow-up turns | 14601 (14575-14605) at the last turn | see below | up to 64 | 5 | see below | 69.7 (43.4-99.8) | 11.49 (7.92-50.01) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.5-68.8); earlier-turn tokens read again per session: 3 (3-143) (earlier turns that the cache did not hold; the 13718 (13717-13720) new tokens of the files are read once); first turn (nothing cached) first token: 6.89 s (4.84-10.58).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 59.8 | 10.236 | 52.5 |
| 1 | 3759 | 638 | 3121 | 260.6 | 12.044 | 67.8 |
| 2 | 5292 | 3822 | 1470 | 187.0 | 7.916 | 43.4 |
| 3 | 8712 | 5343 | 3369 | 187.7 | 18.006 | 54.1 |
| 4 | 11638 | 8776 | 2862 | 215.4 | 13.356 | 67.6 |
| 5 | 14601 | 11631 | 2970 | 228.9 | 13.029 | 68.2 |

Per-request data: `results.json`. The suite did not start the server, so no engine-log excerpt is included; the launch line above is the live `/proc` cmdline captured at run end.

## Correctness and limitations

The suite itself measures speed only; needle recall on the same build passed 6/6 (Appendix B). The conversation uses the model's own replies, so token counts differ a little between runs. Fresh-prompt prefill speed is sensitive to expert/page-cache state on this 16 GiB-VRAM expert-streaming box: four windows were measured the same morning (Appendix C) and prefill medians moved more than 2x while decode, cache-hit and TTFT stayed within a few percent. Treat the prefill medians as "this configuration in a warm, quiet window" and the appendix as the honest envelope. On a 60 GiB-RAM host with `--resident-budget-gib 36`, the engine legitimately writes tens of GiB to zram during long-prompt streaming (73.8 GiB over this window, worst 60 s burst 7.0 GiB); no swap thrash coincided with a timed run in the headline window. Total time of the headline suite: 902 s.

---

## Appendix A — Provenance (headline window)

| Field | Value |
| --- | --- |
| Suite window | 2026-10-18 09:26:53–09:45:00 local (14:26–14:45 UTC) |
| Engine process started | 09:23:15 local (3.5 min before the suite; system had been rebooted) |
| Strata commit | 477ac13cbe8e582c20484871ce459d20d82df09d |
| Engine / project version | 0.1.45 / 0.1.45 |
| GPU / driver / CUDA | RTX 5080 16 GiB / 595.104.02 / 13.2 |
| PCIe link (current) | gen 4, x16 |
| Power limit / idle draw | 360 W / 47–54 W between stages |
| CPU / RAM | Ryzen 9 9950X (16C/32T) / 60.2 GiB |
| Storage | CT4000T705SSD3 NVMe (rotational=0), root |
| GPU samples (2 s) | 489; util min 0 % max 100 %; temp max 63 °C; clock-throttle-reason bits set in 0 samples |
| Memory during suite | MemAvailable min 9.03 GiB; SwapFree min 11.1 GiB; pswpout delta 73.8 GiB (zram), worst 60 s 7.0 GiB |
| Background load | user apps closed; co-resident agent client made zero requests during the window (Appendix C method) |
| Swap config | 16 GiB zram + 4 GiB disk swap (disk swap untouched during the window) |

Machine-readable companion: `provenance.json` (same folder).

## Appendix B — Correctness (needle recall, same build and config)

Run 08:46–09:00 local the same morning on the same engine version, commit and launch config (before an unrelated restart; process restart does not alter these results). `tools/needle_bench.py --lengths 32k,128k --depths 10,50,90`. **6/6 found**; raw records in `needles.json`.

| Context | Depth | Prompt tokens | Needle | Result | Seconds |
| --- | --- | ---: | --- | --- | ---: |
| 32k | 10 % | 32,792 | meadow-copper-504 | FOUND | 63.7 |
| 32k | 50 % | 32,792 | falcon-quartz-940 | FOUND | 59.6 |
| 32k | 90 % | 32,793 | willow-cobalt-696 | FOUND | 62.4 |
| 128k | 10 % | 127,285 | falcon-saffron-138 | FOUND | 241.5 |
| 128k | 50 % | 127,283 | quartz-tundra-528 | FOUND | 226.6 |
| 128k | 90 % | 127,283 | quartz-glacier-192 | FOUND | 183.7 |

## Appendix C — Windows measured, kept and discarded (same morning, same box)

The box hosts the benchmark server and the user's agent client on one GPU. Method used to keep them apart: the orchestrator ran only while the GPU idled under 5 %, agent turns were confirmed absent from the window, and GPU/swap logs were kept. Windows, earliest first:

| # | Window (local) | Content | Verdict |
| --- | --- | --- | --- |
| 1 | 07:18 / 07:30 / 07:53 | three early suites (one custom harness, two official-tool), back-to-back with live agent traffic, no needles, incomplete provenance; 4K medians 524 and 998 | discarded → attic; superseded by #3/#4 |
| 2 | 08:16–08:46 + needles 08:46–09:00 | full run after an engine restart (25 min uptime); a reclaim storm wrote ~109 GiB to swap during it: 4K median 377, 16K 691, 32K 313 (104–415); its needle arm (08:46–09:00) ran clean and is kept as Appendix B | speed table discarded (storm); needles kept |
| 3 | **09:26–09:45 — headline** | fresh system reboot at ~09:20, engine up 09:23:15; warm-up passed both paths (4K last probes 583/644, 32K 1236/1284 tok/s); tables above | **kept** |
| 4 | 10:01–10:21 — reproducibility | second suite ≥5 min after all agent traffic, but its warm-up window had been crossed by two ~100 K-token agent prefills; system ran this whole window in a slower cache state | kept as lower bound below |

Appendix-D suite (window 4) medians, for reproducibility:

| Config | Prompt tok/s | Decode tok/s | TTFT s | vs headline |
| --- | --- | --- | --- | --- |
| short chat | — | 68.2 (50.2-80.4) | 0.43 | decode −6 %, TTFT +7 % |
| 4K fresh | 323.3 (249.3-461.9) | — | 12.48 | prefill −44 % |
| 16K fresh | 440.5 (400.5-695.4) | — | 36.51 | prefill −54 % |
| 32K fresh | 474.7 (263.1-559.9) | — | 67.59 | prefill −16 % |
| agent session | cache hit 68.8 % | 68.8 (42.6-87.5) | 14.78 | decode −1 %, cache hit identical |

Read: decode, first-token for short chats and conversation-cache behaviour are stable to within noise across windows (±7 %, cache hit identical to one decimal); fresh-prompt prefill on this expert-streaming, 16 GiB-VRAM box depends strongly on transient cache state (4K medians seen all morning: 377, 323, 573, 998 tok/s). If you want prefill numbers from this class of box, ask for the warm-up record — several windows are the honest picture, one number is not.

## Appendix D — What this config is for

Two host-specific features are enabled and are not in the default install: `STRATA_HC_Q8=1` and the FP8 PLE sidecar (`ple-fp8.gguf`). An informal same-box A/B earlier this month measured roughly +21 % decode from `HC_Q8` alone; the headline tables above show the full config (both features on) reaching 69–73 tok/s decode with 68.8 % conversation-cache hits at 14.6 K-token agent turns on an RTX 5080. Decode at this level from 16 GiB VRAM against a 93.7 GB model is the point of the box; the prefill numbers show where the remaining pressure is.
