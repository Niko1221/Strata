# Community benchmark on AMD Instinct MI100 (3x)

Measured on 2026-11-05 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Run against a Strata server already running on `xnet-ai-01.xnet.com:8000` (remote server, so the run is `nonconforming` per the protocol — the standard protocol targets a freshly loaded local engine). Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: 3x AMD Instinct MI100 (Arcturus, gfx908), 32 GiB VRAM each (a 4th MI100 is present in the system but idle); selected GPUs: 1, 2, 3; PCIe link not recorded
- CPU: AMD EPYC 7402 24-Core Processor (24 cores, 48 threads); installed RAM: 251.6 GiB; storage: NVMe WDC 1024 GB plus SK hynix SATA 512 GB
- OS: Linux 6.12.115+deb13.1-amd64 (Debian 13 trixie); GPU driver/backend: ROCm HIP (`backend: hip`)
- Strata: project 0.1.56, commit ffa80d73; engine 0.1.56 (build-908)
- Background workloads and any power limits: not recorded

## Model and configuration

- Model repository and revision: Qwen3.8-Flash-Next-Uncensored, Q4_K_S quantization, GGUF in 3 shards (`...-00001-of-00003.gguf`)
- Vision encoder: enabled (strata-vision, mmproj BF16, GPU offload disabled)
- Custom packs or profiles: pack `orca-q4_k_s`, expert profile `expert-profile.bin`, expert-cache `auto`
- Context: 262144 tokens; KV type and streaming: KV int8, prefill `auto`; low-RAM mode: not set
- MTP: spec 3, draft-conf 0.35; reasoning: effort `high`, budget 8192 tokens; sampling: temperature 0; calibration/experimental speed projection: not enabled
- Drafting: MTP only (DFlash2 disabled via `drafting: "mtp"` in config)

```text
Launch (config strata-orca-q4_k_s.json, port 8000):
server.py --engine strata --config strata-orca-q4_k_s.json --port 8000
engine: strata --serve --pack .../packs/orca-q4_k_s --native <q4_k_s shard 1> --ple-gguf <q4_k_s shard 1> --ple-io mmap --expert-profile .../expert-profile.bin --expert-cache auto --prefill auto --max-context 262144 --kv int8 --vision --spec-min-p 0.5 --mtp .../mtp-orca/rt --spec 3 --layer-split auto
```

## Method

Run by `python tools/strata_bench.py --url http://xnet-ai-01.xnet.com:8000` against the HTTP API of a Strata server already running; model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs (9 requests)
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency: not run (single client)

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 78.4 (63.2-92.9) | 0.57 (0.54-1.17) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1185.2 (1177.4-1185.4) | not measured | 3.50 (3.50-3.51) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1964.4 (1958.8-1965.8) | not measured | 8.29 (8.28-8.31) |
| agent session, follow-up turns | 13149 (13143-13149) at the last turn | see below | up to 64 | 3 | see below | 76.8 (58.7-89.0) | 3.47 (2.59-4.00) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.2-69.3); earlier-turn tokens read again per session: 3 (3-4) (earlier turns that the cache did not hold; the 12278 (12278-12280) new tokens of the files are read once); first turn (nothing cached) first token: 2.17 s (2.03-2.21).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 314.8 | 2.033 | 72.6 |
| 1 | 3388 | 631 | 2757 | 814.4 | 3.501 | 58.7 |
| 2 | 4921 | 3452 | 1469 | 582.3 | 2.622 | 60.6 |
| 3 | 8165 | 4984 | 3181 | 819.8 | 3.998 | 70.3 |
| 4 | 10384 | 8220 | 2164 | 713.7 | 3.139 | 76.3 |
| 5 | 13143 | 10433 | 2710 | 759.1 | 3.68 | 83.5 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (not produced — the benchmark used an already-running server, it did not start one).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `rocm-smi` and the OS; background load is not recorded. Total time of the suite: 131 s. The run is against a remote server and is marked `nonconforming`; it is valid for self-comparison but not directly comparable to a freshly loaded local-engine run.
