# Community benchmark on NVIDIA GeForce RTX 3060 (USB4 eGPU), Qwen3.6-35B-A3B UD-Q4_K_XL

Measured on 2026-11-15 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3060, 12 GiB VRAM
- CPU: AMD Ryzen 7 PRO 6850U with Radeon Graphics (8 cores, 16 threads); installed RAM: 28.2 GiB; storage: nvme0n1 ADATA LEGEND 700 nvme
- OS: Linux 7.0.0-42-generic; GPU driver: 595.99.02; CUDA (driver supports up to): 13.2
- Strata: project 0.1.59.1, commit 13eedf55; engine 0.1.59
- GPU detail (after the run, near idle): power_limit_w=170.00, power_draw_w=109.31, pcie=gen4 x4, pstate=P2, temp_c=51
- Background workloads: an idle desktop session; `nvidia-smi` listed no other compute process before the server started, and the engine log has only the suite's 41 requests (`engine-log-excerpt.txt`)

## Model and configuration

- Model and quantization: qwen3.6-35b-a3b-ud-q4_k_xl
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 40.8 (40.2-40.9) | 0.42 (0.40-0.48) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 663.3 (663.3-663.3) | not measured | 6.06 (6.06-6.06) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 662.6 (662.4-662.7) | not measured | 24.25 (24.25-24.26) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 647.6 (647.4-647.6) | not measured | 49.51 (49.51-49.53) |
| agent session, follow-up turns | 13818 (13816-13822) at the last turn | see below | up to 64 | 3 | see below | 25.1 (20.3-31.4) | 15.16 (6.06-21.23) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 0.0 % (0.0-0.0); earlier-turn tokens read again per session: 30428 (30421-30445) (earlier turns that the cache did not hold; the 12968 (12968-12968) new tokens of the files are read once); first turn (nothing cached) first token: 3.03 s (3.03-3.12).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 201.2 | 3.119 | 38.0 |
| 1 | 4083 | 0 | 4083 | 673.4 | 6.236 | 31.3 |
| 2 | 5600 | 0 | 5600 | 615.8 | 9.144 | 29.2 |
| 3 | 8836 | 0 | 8836 | 583.2 | 15.197 | 25.0 |
| 4 | 11072 | 0 | 11072 | 608.8 | 18.217 | 23.0 |
| 5 | 13822 | 0 | 13822 | 651.5 | 21.233 | 20.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 541 s.

## Notes added by hand

These are not detected by the tool. They come from `lspci`, `nvidia-smi`, the run config and `engine-log-excerpt.txt`.

**The card is an external GPU over USB4.** The RTX 3060 sits in an enclosure on the USB4 port of a mini PC with a 15 W-class
laptop CPU (Ryzen 7 PRO 6850U, 8 cores). `lspci` shows it behind the AMD "USB4/Thunderbolt PCIe tunnel" bridge;
`nvidia-smi` reports the link as gen4 x4. Install: Linux release 0.2.13.1 (Control 0.1.59.1, engine 0.1.59
`linux-x86_64-cuda-cuda13`), updated with `strata update` before the run and started fresh with
`strata launch --preset qwen36 --port 8083 --no-tui --yes`. Model files from setup's `qwen36` path (Unsloth UD-Q4_K_XL with
the MTP layer); quantization repository revision not recorded.

**Engine arguments** (setup's run config, unchanged): `--native <model> --max-context 65536 --prefill auto --mtp 2 --vision`
(vision on the CPU, not used by this suite).

**Expert placement.** At this 64K context the engine kept **0 of 256 experts per layer in VRAM**; all 18.75 GiB of experts
are in pinned host memory (log: `hybrid experts: 0 of 256 per layer in VRAM ... bigger prompt chunks streamed`, 1.77 GiB of
VRAM free after the weights and the context). The engine's own test start of the same files at a 4,096 context placed 38 of
256 per layer (2.78 GiB) in VRAM. This report does not measure a shorter context.

**Prompt time comes in steps of one chunk.** The prompt chunk is 2,048 tokens. Every prompt of more than 128 tokens took
about 3.02 s per started chunk, however full the chunk was: 260, 609, 906 and 1,028 tokens each took 3,023-3,027 ms, 4,019
tokens (2 chunks) 6,064 ms, 16,058 tokens (8 chunks) 24.25 s. Prompts of up to 128 tokens run on the CPU (43-51 tokens:
0.40-0.47 s). This is why the 4K, 16K and agent first-token times repeat to the millisecond. A plausible reading, not
measured: each chunk that streams the host-resident experts to the card pays one pass over them through the USB4 link
(18.75 GiB in 3.02 s would be about 6.2 GiB/s). No bandwidth test was run, and no internal-slot run on the same card is
available for comparison.

**Next to the numbers in `docs/QWEN36.md`.** That page lists 64.0 tok/s for one request on an RTX 3060 12 GB in the hybrid
mode (Strata 0.1.45, `--batch 8 --mtp 0`, greedy, 128 new tokens, a different machine and a different workload). This run
decoded 40.8 tok/s (median of 9 short chats). The two are not the same measurement; the differences that are known are the
USB4 link, 0 experts in VRAM at 64K context, the CPU, and the engine version.

**Two things this run did not show, recorded as observed:**

- **The conversation cache reused nothing.** Every agent follow-up turn logged `0 reused`, so each turn read the whole
  conversation again (30,428 tokens per session). `docs/QWEN36.md` says this model continues from the held state when the
  same conversation is sent again. Not investigated here.
- **No MTP drafts.** With `--mtp 2` in the arguments and `"drafting": "mtp"` reported by the server, every request logged
  `drafts accepted 0 of 0`. Thinking was off and the temperature 0 (the suite's defaults). Not investigated here.
