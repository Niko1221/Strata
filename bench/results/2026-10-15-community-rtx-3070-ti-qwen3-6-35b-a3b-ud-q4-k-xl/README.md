# Community benchmark on NVIDIA GeForce RTX 3070 Ti

Measured on 2026-10-15 by jkir4n with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed suite; a small correctness battery was run separately (see below).

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3070 Ti, 8 GiB VRAM
- CPU: AMD Ryzen 5 7600X 6-Core Processor (6 cores, 12 threads); installed RAM: 31.1 GiB; storage: XPG GAMMIX S70 BLADE 2 TB NVMe (PCIe 4.0; the model lives on this drive)
- OS: Windows 11 (10.0.26200); GPU driver: 617.14; CUDA: 13.0 (the ready-made engine's build)
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1 (official Windows CUDA 13 release binary)
- Background workloads: desktop apps open during the run (three Edge and two Brave processes, WhatsApp, Discord, CrossDeviceService); GPU power limit: 290 W (default, unchanged)

## Model and configuration

- Model: abliterated community fine-tune — `llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GGUF`, Q5_K_S (sha256 `137ea38c8d8761443dce77dd63ced173d6c18c1b3c88a6e836957c266191531a`), substituted into the UD-Q4_K_XL install slot; the server's label `qwen3.6-35b-a3b-ud-q4_k_xl` comes from that slot. The MTP block is present and active.
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
engine:  strata.exe --native <model>.gguf --max-context 32768 --prefill auto --mtp 2
server:  serve/server.py --engine strata --config strata-qwen36-ud-q4_k_xl.json --port 8080
```

(The config also sets `"parallel": 4`, which this engine path ignores — see the notes.)

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 50.6 (46.6-60.7) | 0.52 (0.49-0.59) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1189.1 (1177.3-1193.4) | not measured | 3.40 (3.39-3.45) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1100.2 (1097.3-1100.7) | not measured | 14.62 (14.62-14.67) |
| agent session, follow-up turns | 14532 (14531-14534) at the last turn | see below | up to 64 | 3 | see below | 35.5 (22.4-47.3) | 8.28 (3.31-14.05) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 0.0 % (0.0-0.0); earlier-turn tokens read again per session: 30094 (30088-30100) (earlier turns that the cache did not hold; the 13708 (13708-13708) new tokens of the files are read once); first turn (nothing cached) first token: 1.35 s (1.32-1.40).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 459.9 | 1.395 | 60.4 |
| 1 | 3738 | 0 | 3738 | 1133.3 | 3.384 | 47.3 |
| 2 | 5253 | 0 | 5253 | 1070.9 | 4.944 | 45.3 |
| 3 | 8687 | 0 | 8687 | 1043.7 | 8.364 | 35.5 |
| 4 | 11587 | 0 | 11587 | 1096.2 | 10.61 | 31.0 |
| 5 | 14531 | 0 | 14531 | 1037.9 | 14.038 | 22.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server; this run used a server that was already running, so the file is absent).

### Additional measurements (same machine, same day)

- Second suite run (same settings, during a memory-tight stretch with a hardware sampler active): chat 39.4 tok/s median (14.4-50.1), 4K 1187.7 (695.8-1189.0), 16K 1099.2 (1097.1-1102.4), agent session 34.6 (23.3-48.5).
- Concurrency (`--clients 4`): 36.5 tok/s total over the burst (36.1-37.6 across 3 rounds); first-token p50 6.09 s / p95 11.58 s; request latency p50 8.78 s / p95 14.10 s; 0 errors. The qwen36 engine path has no batch slots in 0.1.44.1 (its log: `ignoring --batch 4 (not an option of this engine)`), so requests beyond the first wait their turn.
- Conversation cache: 0 % reuse even for an identical repeated prompt — no checkpoints on this engine path; every request re-reads its prompt from token 0.
- 25K-token prompt (near the 32K context): 25,298 tokens at 376.2 tok/s prompt, 14.7 tok/s decode, ~67 s to first token, ~75 s total; MTP drafts 65/108 accepted.
- Thinking on (default, 3 short prompts): decode 49.2-54.1 tok/s — no decode-rate penalty vs thinking off; wall 8-20 s per prompt.
- Memory: VRAM steady at 7,666 of 8,192 MiB; free RAM ranged 0.2-3.8 GiB during the runs. All routed experts run from pageable host RAM (0 of 256 per layer fit in VRAM at this quant); the box sits at its memory edge, and chat decode dips (to ~14 tok/s) when Windows trims/compresses model pages.
- Small correctness battery after the suite (not part of it): 5/5 — `27*43` = 1161; the five-machines/five-widgets puzzle = "5 minutes"; a fizzbuzz function whose outputs were executed and verified; capital of Australia = Canberra; an exact 3-bullet, ≤5-word instruction format.

## Correctness and limitations

The suite itself runs no correctness check (the battery above was run separately). The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi` and the OS. Total time of the first suite: 242 s. Numbers are single-session medians on a box at its RAM edge — expect variance with different background load. The model is a community abliterated fine-tune substituted into the install slot; its tensor types (Q5_K gate/up/down, MTP present) are accepted by the engine as-is.
