# Community benchmark on AMD Instinct MI100 (3×)

Measured on 2026-11-05 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Run against a Strata server already running on `xnet-ai-01.xnet.com:8000` (remote server, so the run is `nonconforming` per the protocol — the standard protocol targets a freshly loaded local engine). Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: 3× AMD Instinct MI100 (Arcturus, gfx908), 32 GiB VRAM each (a 4th MI100 is present in the system but idle); selected GPUs: 1, 2, 3; PCIe link not recorded
- CPU: AMD EPYC 7402 24-Core Processor (24 cores, 48 threads); installed RAM: 251.6 GiB; storage: NVMe WDC 1024 GB plus SK hynix SATA 512 GB
- OS: Linux 6.12.115+deb13.1-amd64 (Debian 13 trixie); GPU driver/backend: ROCm HIP (`backend: hip`)
- Strata: project 0.1.56, commit ffa80d73; engine 0.1.56 (build-908)
- Background workloads and any power limits: not recorded

## Model and configuration

- Model repository and revision: Qwen3.8-Flash-Next-Uncensored, Q4_K_S quantization, GGUF in 3 shards (`...-00001-of-00003.gguf`)
- Vision encoder: enabled (strata-vision, mmproj BF16, GPU offload disabled)
- Custom packs or profiles: pack `orca-q4_k_s`, expert profile `expert-profile.bin`, expert-cache `auto`, drafting `dflash2` with `dflash2.q4_0` pack and MTP draft vocab
- Context: 262144 tokens; KV type and streaming: KV int8, prefill `auto`; low-RAM mode: not set
- MTP: spec 6 (running process; config lists spec 2), draft-conf 0.35; reasoning: effort `high`, budget 8192 tokens; sampling: temperature 0; calibration/experimental speed projection: not enabled

```text
Launch (config strata-orca-q4_k_s.json, port 8000):
server.py --engine strata --config strata-orca-q4_k_s.json --port 8000
engine: strata --serve --pack .../packs/orca-q4_k_s --native <q4_k_s shard 1> --ple-gguf <q4_k_s shard 1> --ple-io mmap --expert-profile .../expert-profile.bin --expert-cache auto --prefill auto --max-context 262144 --kv int8 --vision --draft dflash2 --draft-model <dflash2 pack> --draft-fallback none --mtp-draft-vocab <draft_vocab.bin> --spec 6 --draft-conf 0.35 --layer-split auto
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 64.9 (62.0-117.3) | 0.48 (0.45-0.51) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1301.0 (1277.9-1302.3) | not measured | 3.20 (3.20-3.25) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2168.8 (2168.8-2172.4) | not measured | 7.51 (7.51-7.52) |
| agent session, follow-up turns | 13150 (13146-13156) at the last turn | see below | up to 64 | 3 | see below | 63.4 (50.7-80.7) | 3.28 (2.49-3.84) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.1-69.3); earlier-turn tokens read again per session: 4 (3-74) (earlier turns that the cache did not hold; the 12279 (12279-12279) new tokens of the files are read once); first turn (nothing cached) first token: 1.95 s (1.89-1.97).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 338.1 | 1.89 | 69.2 |
| 1 | 3388 | 630 | 2758 | 844.2 | 3.368 | 59.7 |
| 2 | 4921 | 3381 | 1540 | 606.1 | 2.635 | 63.8 |
| 3 | 8165 | 4984 | 3181 | 852.0 | 3.839 | 62.0 |
| 4 | 10393 | 8229 | 2164 | 739.7 | 3.03 | 50.7 |
| 5 | 13150 | 10440 | 2710 | 801.2 | 3.496 | 80.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (not produced — the benchmark used an already-running server, it did not start one).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `rocm-smi` and the OS; background load is not recorded. Total time of the suite: 125 s. The run is against a remote server and is marked `nonconforming`; it is valid for self-comparison but not directly comparable to a freshly loaded local-engine run.
