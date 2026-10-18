# Community benchmark on NVIDIA GeForce RTX 5090

Measured on 2026-10-18 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. The official table is the speed suite. The needle check is in the long-context section below.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5090, 32 GiB VRAM
- CPU: AMD Ryzen 7 9700X 8-Core Processor (8 cores, 16 threads); installed RAM: 93.7 GiB; storage: not measured
- OS: Windows 11 (Python reports Windows 10.0.26300); GPU driver: 617.42; CUDA: 13.0 in the release engine package
- Strata: project 0.1.45, commit 477ac13c; engine 0.1.45
- Background workloads: no other model port in [5310, 6310, 6311, 9810, 9812] was accepting connections at start. Power limit about 402.50 W. PCIe during the run is in the long-context section.

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 524288 tokens; parallel slots: 2
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --batch-mtp --max-context 524288 --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700 --conversation-cache-mib 16384 --conversation-cache-slots 4 --pcie-frac 0.35 --spec-min-p 0.50 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
```

Environment of the config: `STRATA_PF_FUSED=1`

The config's `parallel` value is 2. The server adds `--batch 2`, which is why that flag is not in the line above. The engine log for this start shows two slot sessions, 1.39 GiB each, and an expert cache of 10038 slots / 19.05 GiB after that reservation. KV is int8: 32768 of 524288 cells per QSA layer stay in VRAM, and the log puts the K/V in 6.19 GiB of pinned RAM. YaRN factor 2 (mscale 1.069315) is on for the whole process.

## Method

Run by `python tools/strata_bench.py --clients 2 --long --runs 3` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K / 32K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 122.5 (104.8-142.6) | 0.28 (0.24-0.31) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 2865.8 (2863.4-2867.4) | not measured | 1.42 (1.42-1.43) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 5624.5 (5617.4-5629.3) | not measured | 2.88 (2.88-2.91) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 5731.6 (5715.5-5831.1) | not measured | 5.63 (5.53-5.64) |
| agent session, follow-up turns | 14538 (14524-14554) at the last turn | see below | up to 64 | 3 | see below | 170.3 (110.8-238.1) | 1.36 (1.31-1.45) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.8); earlier-turn tokens read again per session: 74 (3-74) (earlier turns that the cache did not hold; the 13718 (13718-13719) new tokens of the files are read once); first turn (nothing cached) first token: 1.23 s (1.11-1.24).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 563.2 | 1.108 | 96.2 |
| 1 | 3752 | 631 | 3121 | 2261.6 | 1.4 | 110.8 |
| 2 | 5285 | 3815 | 1470 | 1136.8 | 1.317 | 140.9 |
| 3 | 8718 | 5349 | 3369 | 2475.4 | 1.394 | 165.3 |
| 4 | 11627 | 8765 | 2862 | 2125.4 | 1.365 | 152.6 |
| 5 | 14554 | 11655 | 2899 | 2163.8 | 1.359 | 128.3 |

### Concurrency: 2 clients

Total 117.2 tok/s (91.5-120.5); first token p50 0.671 s, p95 1.005 s; request latency p50 2.155 s, p95 2.763 s; errors 0. Server parallel slots: 2 (requests beyond that wait their turn).

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

The official table is speed. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi` and the OS. The power limit and the PCIe link measured during the run are in the long-context section. Total time of the official suite: 80 s.

## Long-context extension

Same server process as the official suite above, after that suite had already run. Thinking was off (`reasoning_effort` none and `enable_thinking` false). Temperature was 0. Each new answer was capped at 256 tokens. A fresh prompt is a unique word list. A cached resend is the same user message after a short request was sent to every slot, and it counts only when the engine log recorded a restore and at least 90% of the prompt tokens were reused. A park the engine skipped is not averaged into the cached rows. The fresh concurrent rows are two different prompts sent together. Their time to first token includes queueing. On a long prefill the 256-token answer often finishes before the other prefill does, so that row does not by itself show batched decode. The official two-client burst above is the short-prompt concurrency figure, and its total tok/s includes prompt tokens.

YaRN factor 2 is process-wide. The 64K, 128K and 256K rows use that scaled RoPE. They are not native-context measurements. One request on this server still pays for the second slot's reserved memory. There is no parallel-1 arm in this run.

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 65536, fresh single | 63911 | 0 | 256 | 3 | 5834.9 (5753.2-5841.7) | 277.9 (193.4-283.2) | 11.01 (10.99-11.19) |
| 65536, cached single | 63911 | 63904 | 256 cap | 3 | 52.3 (52.0-52.5) | 278.9 (253.4-279.8) | 0.18 (0.16-0.19) |
| 65536, fresh concurrent | 63886 | 0 | 256 | 6 | 5253.8 (4719.1-5786.2) | 154.5 (32.0-280.6) | 18.07 (11.14-25.11) |
| 65536, cached concurrent | 63911 | 63904 | 256 | 1 | 53.0 (53.0-53.0) | 33.5 (33.5-33.5) | 0.18 (0.18-0.18) |
| 131072, fresh single | 127867 | 0 | 256 | 3 | 5748.3 (5673.3-5749.6) | 270.3 (269.8-276.0) | 22.36 (22.36-22.64) |
| 131072, cached single | 127867 | 127860 | 256 cap | 3 | 38.0 (34.9-39.9) | 270.0 (262.5-272.3) | 0.26 (0.24-0.27) |
| 131072, fresh concurrent | 127802 | 0 | 256 | 6 | 5403.0 (5118.4-5754.0) | 143.8 (31.1-262.9) | 35.35 (22.39-48.35) |
| 131072, cached concurrent | 127929 | 127922 | 256 | 1 | 36.9 (36.9-36.9) | 33.1 (33.1-33.1) | 0.27 (0.27-0.27) |
| 262144, fresh single | 255693 | 0 | 256 | 3 | 5438.9 (3624.8-5467.3) | 262.5 (67.5-264.0) | 47.20 (46.96-113.14) |
| 262144, cached single | 255718 | 255710 | 256 cap | 2 | 23.8 (22.4-25.2) | 258.6 (256.3-261.0) | 0.39 (0.39-0.40) |
| 262144, fresh concurrent | 255646 | 0 | 256 | 6 | 4686.8 (3628.6-5402.5) | 40.6 (23.8-72.9) | 113.01 (60.52-167.06) |
| 262144, cached concurrent | 255742 | 255735 | 256 | 1 | 24.3 (24.3-24.3) | 32.5 (32.5-32.5) | 0.39 (0.39-0.39) |
| full window (target 523976), fresh single | 511045 | 0 | 256 | 3 | 4769.7 (4747.2-4828.5) | 25.8 (21.2-39.8) | 110.38 (108.37-152.27) |
| full window (target 523976), cached single | not measured | not measured | 256 cap | 0 | not measured | not measured | not measured |
| full window (target 523976), fresh concurrent | 510926 | 0 | 256 | 2 | 4611.3 (4480.6-4742.0) | 24.3 (22.1-26.5) | 172.06 (116.70-227.42) |
| full window (target 523976), cached concurrent | not measured | not measured | not measured | 0 | not measured | not measured | not measured |
| full window, fresh process, one prompt | 511089 | 0 | 256 | 1 | 4986.7 | 179.8 | 102.83 |
| full window, fresh process, resend (park skipped) | 511089 | 0 | 256 | 1 | 4960.6 | 235.5 | 103.22 |

Per-request rows are in `long_matrix.json`. The fresh-process resend is in `cold-cache.json`. Decode medians use only answers of at least 32 tokens. Prompt tok/s stays in the table when the answer is short. On a cache hit the engine reads only the uncached tail (about 7 tokens). The prompt tok/s cell on those rows is that short read. Compare time to first token instead: about 0.18 s, 0.26 s and 0.39 s against 11 s, 22 s and 47 s fresh.

The word-list estimate for the full window aimed at 523976 tokens and landed at about 511000. That is the measured prompt. It leaves room for the 256-token answer inside 524288.

A fresh pair's time to first token includes queueing. Every measured pair was judged `prefill looks serial; decode did not overlap`: the slower request's first token arrived only after the faster request had finished its 256 tokens, which on these lengths means the second prefill was still running. That row is not batched decode. The official two-client figure above is the short-prompt concurrency number, and its total tok/s includes prompt tokens.

- 64K and 128K single resends: 3 of 3 restored after a short request was sent to both slots. Draft accepts were about 191 to 193 of 256, and decode stayed near the fresh decode.
- 64K and 128K cached concurrent: one pair only. One of the two prompts restored and the other reread the whole prompt (0 tokens reused). The restored request decoded at about 33 tok/s with 0 draft accepts while that reread was still in its prefill. The reread then decoded at about 265 to 277 tok/s. The table's cached-concurrent decode cell is that one restored request, not two cache hits.
- 256K: 2 of 3 single resends restored. The third reread (0 reused, 0 draft accepts, 30.6 tok/s). One of the three fresh prompts also accepted 0 drafts; fresh decode on that length runs from 67.5 to 264.0 tok/s, median 262.5. The cached pair restored one side only.
- Full window in the same long process: 3 fresh prompts, reused 0, draft accepts 0, decode median 25.8 tok/s (21.2 to 39.8). Three resends also reused 0, so they are not in the cached row. The log during those requests shows parked snapshots of about 264K tokens (about 4 GiB) and eviction counts past 100, not a restore of the 511K prompt. One fresh pair was measured (wall 242 s). The second and third pairs were skipped because available RAM was under 4 GiB. Available RAM was already about 3.5 GiB at the end of the 64K block; the process stayed up.
- Full window on a new process, before any shorter benchmark rows: one prompt, 511089 tokens, 193 drafts accepted, decode 179.8 tok/s, time to first token 102.8 s. The engine did not keep a snapshot. Its log says `skip parking (physical RAM admission; need 8243 MiB plus 2560 MiB floor, 9377 MiB available)`. The resend therefore read the prompt again (192 drafts accepted, decode 235.5 tok/s, time to first token 103.2 s). The flush between them was two short requests with `strata_checkpoint` false, so the miss is the admission skip, not a new snapshot from the flush. The 16 GiB conversation-cache setting is not the check that refused this snapshot.

### Needle check

One code word in repository text, two depths. A miss is a miss on that input, not a quality score. `tools/needle_bench.py` aims 2% under the named length. The 128k rows ran at the end of the long process. The 256k and 480k rows ran on the fresh process, after the cold full-window prompt. The 480k depth-90 request was faster because the engine borrowed 229376 tokens from the parked 256k snapshot of the same source text. The 98.4 s row is the cold 476354-token read. All six were found. `needles.json` and `needles-long.json` have the words.

| Length | Depth | Prompt tokens | Seconds | Result |
| --- | ---: | ---: | ---: | --- |
| 128k | 10 | 127534 | 28.0 | found |
| 128k | 90 | 127534 | 27.5 | found |
| 256k | 10 | 261213 | 50.2 | found |
| 256k | 90 | 261213 | 47.9 | found |
| 480k | 10 | 476354 | 98.4 | found |
| 480k | 90 | 476354 | 57.5 | found, borrowed 229376 tokens from the parked 256k snapshot |

### Memory and the bus

- GPU sample while this run was the workload: busiest sample util 100.0%, PCIe gen 4.0 x16.0, power 210.7 W, power limit 402.5 W, peak VRAM used in the samples 31693.0 MiB.
- nvidia-smi at start: NVIDIA GeForce RTX 5090, driver 617.42, display_active Disabled, link capability gen 4 x16.
- Available RAM is recorded on each length in `long_matrix.json`. At the end of the long process it was 3.42 GiB. The engine's own startup lines say 119 MiB of VRAM free and suggest `--vram-reserve-mib` 1093, then 1664. The reserve was left at 700. nvidia-smi's highest sample during the run was 31693 MiB used of 32607. KV-streaming hit rates in the early log stayed above 85 percent. This run does not separate driver paging from the draft accepting 0 tokens on the late full-window rows.

### Correctness and limitations

- The official suite's long prompts generate 24 tokens and this extension generates up to 256. Those decode columns are not interchangeable.
- An immediate resend was not used as the cached condition. The slot can still be holding the prompt; the flush plus a restore line is the condition.
- Cached concurrent is one pair, taken once two shorter snapshots had parked, not three rounds. At the full window the park was skipped, so that cell is empty.
- Expert cache stays warm from the official suite. Later lengths are not a cold load. Model load is outside the request timings.
- KV for the whole context is reserved at startup for both slots. A second request does not allocate that KV again.
- `--vram-reserve-mib` stayed 700 even if the engine suggested more.
- There is no parallel-1 arm. One request on this server still pays for the second slot's reserved memory. A short-prompt calibration harness is not a row here.
- Draft accepts move decode. Rows with about 190 accepts land near 180 to 280 tok/s. Rows with 0 accepts on the long process land near 26 to 40 tok/s. The per-run counts are in `long_matrix.json` and `cold-cache.json`.
- This file does not change the engine. The suite was started by calling the benchmark's server and suite functions with the port passed in. `tools/strata_bench.py`'s command-line start path reads a port name it never assigns, so that command was not what launched the server.
- The long process set its stop flag to low RAM after one full-window pair and did not run the 256k or 480k needles. Those needles were run on the next process and are in the table above.
