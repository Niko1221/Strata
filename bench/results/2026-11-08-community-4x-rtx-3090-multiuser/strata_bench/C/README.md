# Community benchmark on NVIDIA GeForce RTX 3090 + NVIDIA GeForce RTX 3090 + NVIDIA GeForce RTX 3090 + NVIDIA GeForce RTX 3090

Measured on 2026-11-08 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3090 + NVIDIA GeForce RTX 3090 + NVIDIA GeForce RTX 3090 + NVIDIA GeForce RTX 3090, 96 GiB VRAM
- CPU: AMD Ryzen Threadripper 3990X 64-Core Processor (64 cores, 128 threads); installed RAM: 251.5 GiB; storage: sda     Samsung SSD 860 EVO 500GB sata; nvme1n1 CT1000P310SSD8            nvme; nvme2n1 CT1000P310SSD8            nvme; nvme0n1 CT1000P310SSD8            nvme
- OS: Linux 7.0.0-38-generic; GPU driver: 580.196.07; CUDA (driver supports up to): 13.0
- Strata: project 0.1.58, commit 146df3c4; engine 0.1.58
- GPU detail (after the run, near idle): power_limit_w=320.00, power_draw_w=293.41, pcie=gen3 x16, pstate=P2, temp_c=66
- GPU detail (after the run, near idle): power_limit_w=320.00, power_draw_w=275.73, pcie=gen3 x16, pstate=P2, temp_c=66
- GPU detail (after the run, near idle): power_limit_w=320.00, power_draw_w=287.46, pcie=gen4 x8, pstate=P2, temp_c=59
- GPU detail (after the run, near idle): power_limit_w=320.00, power_draw_w=287.46, pcie=gen4 x8, pstate=P2, temp_c=68
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq4_xs
- Context: 262144 tokens; parallel slots: 16
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-212-stocktab-f83fd81691a5b053 --pack unsloth-ud-iq4_xs --native Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill 512 --spec 4 --spec-min-p 0.5 --mtp rt --max-context 262144 --kv int8 --resident-experts --trim-stage-weights --ple-io ram --aux-cpus 65-127 --conversation-cache-mib 126976 --conversation-cache-slots 448 --kv-resident 28672 --conversation-cache-min-free-mib 16384 --conversation-cache-min-tokens 1024 --vision --prompt-cache-every 8192 --prompt-cache 12
```

Environment of the config: `STRATA_MMQ_DIRECT=1`, `STRATA_SLOT_COPY_DIRECT=1`, `STRATA_PIPE_BYIELD_NODRAIN=1`, `STRATA_ADMIT_MERGE_FIRST=1`, `STRATA_CKPT_POOL=1`, `STRATA_CKPT_THREADS=1`, `STRATA_SLOT_CKPT_MOVE=1`, `STRATA_SLOT_COPY_BATCHSYNC=1`, `STRATA_SPLIT_MTP_BATCH=1`, `STRATA_MMQ_GROUP_SORT=1`, `STRATA_BATCH_SELF_COMMIT=1`, `STRATA_NAN_GUARD=0`, `STRATA_QSA_SLOTS=1`, `STRATA_HANDOFF_PIPE=8`, `STRATA_KV_PREFETCH=1`, `STRATA_GR_FAST=1`, `STRATA_GR_NORM_FAST=1`, `STRATA_MMVF_ROWS=1`, `STRATA_GDN_TC=0`, `STRATA_GDN_CONVL2=0`, `STRATA_PROMPT_ATTN_IMMA=0`, `STRATA_AUTO_NOBORROW=0`, `STRATA_PF_W16_CACHE_MIB=0`, `STRATA_MMQ_KQ=0`, `STRATA_BATCH_GDN_GROUPS=1`, `STRATA_BATCH_ATTN_ROWS=1`, `STRATA_CKPT_POOL_READY=4`

## Method

Run by `python tools/strata_bench.py --clients 16 --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 126.8 (108.2-177.2) | 0.20 (0.17-0.33) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 4750.0 (4736.6-5709.6) | not measured | 0.86 (0.72-0.87) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 6225.5 (6217.3-6592.5) | not measured | 2.61 (2.46-2.61) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 6401.1 (6365.4-7152.5) | not measured | 5.05 (4.52-5.08) |
| agent session, follow-up turns | 13604 (13595-13610) at the last turn | see below | up to 64 | 3 | see below | 138.0 (126.6-173.2) | 0.77 (0.48-1.22) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.7 % (69.2-69.9); earlier-turn tokens read again per session: 72 (1-284) (earlier turns that the cache did not hold; the 12726 (12725-12727) new tokens of the files are read once); first turn (nothing cached) first token: 0.91 s (0.65-0.92).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 957.5 | 0.653 | 140.4 |
| 1 | 3838 | 634 | 3204 | 4258.4 | 0.777 | 144.9 |
| 2 | 5371 | 3902 | 1469 | 3174.2 | 0.484 | 127.1 |
| 3 | 8615 | 5364 | 3251 | 3117.9 | 1.067 | 138.0 |
| 4 | 10821 | 8657 | 2164 | 3576.9 | 0.629 | 131.1 |
| 5 | 13595 | 10885 | 2710 | 3848.9 | 0.728 | 166.6 |

### Concurrency: 16 clients

Total 275.3 tok/s (247.8-275.3); first token p50 2.468 s, p95 4.609 s; request latency p50 7.416 s, p95 8.245 s; errors 0. Server parallel slots: 16 (requests beyond that wait their turn).

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 81 s.
