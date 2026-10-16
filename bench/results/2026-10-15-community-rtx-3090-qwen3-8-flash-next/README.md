# Community benchmark on 4× NVIDIA GeForce RTX 3090 (Unsloth UD-IQ4_XS, 262,144-token context)

Measured on 2026-10-15 by philglau with `tools/strata_bench.py` (suite 1, `--clients 1 --long`, 3 runs per test, medians with ranges). Speeds are the engine's own timings (`prompt_per_second`, `predicted_per_second`); first-token times are measured by the client over HTTP. Speed only: no correctness check was run in this suite.

## Hardware and software

- GPU and VRAM: **4× NVIDIA GeForce RTX 3090, 24 GiB each** (96 GiB total), all four selected for the run; PCIe: GPU0 x16, GPU1/GPU2/GPU3 x4 electrical (gen4 capable), no NVLink (PCIe hops only)
- CPU: 13th Gen Intel(R) Core(TM) i7-13700K (16 cores, 22 threads); installed RAM: 44.0 GiB
- Storage: model, packs, and MTP files on a **CIFS network share** (`//<nas>/llm_models` mounted at `/mnt/srv10_llm_models`); engine on local NVMe. 35.4 GiB of cold experts are read from the model files over the network through the file cache; the rest are resident (see below)
- OS: Debian 12 (bookworm) LXC container on Proxmox VE, kernel 7.0.14-6-pve; GPU driver: 595.84; CUDA: 13.2
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1 (release binary)
- Background workloads and power limits: none running on the container; **each GPU power-limited to 250 W** on the Proxmox host (factory default 350 W; GPU1 default 420 W), enforced power limit verified via `nvidia-smi`; host-side details otherwise not recorded

## Model and configuration

- Model: **Unsloth Dynamic UD-IQ4_XS** (`Qwen3.8-Flash-Next-UD-IQ4_XS-0000{1,2,3}-of-00003.gguf`, ~94 GB) from `unsloth-UD-IQ4_XS`; served under the generic id `qwen3.8-flash-next` (the install's configs all use the same `model_name` so clients survive config switches)
- Vision encoder: BF16 mmproj (`mmproj-Qwen3.8-Flash-Next-BF16.gguf`), enabled but not exercised by this suite
- Custom packs/profiles: native-experts pack `packs/unsloth-ud-iq4_xs`; expert profile `data/expert-profile.bin` (24,576 ranked pairs); MTP draft vocabulary `mtp/rt`
- Context: **262,144 tokens** (the model's native window); KV: int8, streaming off; parallel slots: 2 (`--batch 2`)
- Expert cache: `auto` with `--resident-budget-gib 20` (pinned RAM tier) + per-GPU VRAM expert caches (~12–14 GiB each) → **89% of experts resident at load**; remaining ~35.4 GiB mmap-read from the CIFS mount through the OS file cache (decode hit rate during this suite: **99.7–99.8%**, ≤0.1% cross-GPU PCIe reads)
- Prefill: `auto`; MTP speculative decode: on (`--spec 4 --spec-min-p 0.5`); low-RAM mode: not used
- Reasoning: thinking off for the suite; temperature 0; calibration / experimental speed projection: not used

```text
/root/Strata/engine/strata --serve \
  --pack /mnt/srv10_llm_models/strata/packs/unsloth-ud-iq4_xs \
  --native /mnt/srv10_llm_models/strata/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  --expert-profile /root/Strata/data/expert-profile.bin \
  --expert-cache auto --prefill auto \
  --spec 4 --spec-min-p 0.5 --mtp /mnt/srv10_llm_models/strata/mtp/rt \
  --max-context 262144 --kv int8 --resident-budget-gib 20 --vision \
  --vram-reserve-mib 896 --vram-reserve-later-mib 1175 \
  --layer-split auto --batch 2
```

Layer split (auto, weights + KV by layer): CUDA0 layers 0–11, CUDA1 12–26, CUDA2 27–35, CUDA3 36–47 and the head.

## Method

Run by `python tools/strata_bench.py --url http://127.0.0.1:8080 --clients 1 --long` against the HTTP API of an already-running server; model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests. Server had been up and serving for ~3 hours (expert caches warm, RAM tier populated) before the run.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K / 32K prompts (`--long`): generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (`--clients 1`): 1 request sent at a moment, total tokens over the wall time of the burst, latency over all requests. Note: with `--batch 2`, MTP speculative decoding is disabled inside shared batch windows, but a single request occupies one slot alone and keeps MTP

Total suite time: 131 s.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 113.3 (88.8-153.1) | 0.24 (0.22-0.26) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1313.3 (1249.7-1313.8) | not measured | 3.08 (3.08-3.23) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2393.8 (2359.3-2398.1) | not measured | 6.73 (6.72-6.83) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 3240.1 (3234.1-3248.9) | not measured | 9.92 (9.89-9.94) |
| agent session, follow-up turns | 14585 (14553-14597) at the last turn | see below | up to 64 | 3 | see below | 127.1 (81.5-139.5) | 2.76 (2.05-3.19) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.8); earlier-turn tokens read again per session: 73 (2-73) (earlier turns that the cache did not hold; the 13728 (13728-13729) new tokens of the files are read once); first turn (nothing cached) first token: 1.21 s (1.21-1.24).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 496.9 | 1.241 | 80.3 |
| 1 | 3757 | 637 | 3120 | 1038.6 | 3.022 | 129.3 |
| 2 | 5290 | 3821 | 1469 | 684.1 | 2.167 | 130.5 |
| 3 | 8734 | 5354 | 3380 | 1066.4 | 3.189 | 94.4 |
| 4 | 11660 | 8797 | 2863 | 967.5 | 2.984 | 127.1 |
| 5 | 14597 | 11698 | 2899 | 1055.8 | 2.768 | 99.0 |

### Concurrency: 1 clients

Total 75.5 tok/s (69.9-79.4); first token p50 0.321 s, p95 0.335 s; request latency p50 1.694 s, p95 1.818 s; errors 0. Server parallel slots: 2 (requests beyond that wait their turn).

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server; not present here — an existing server was benchmarked).

### Memory observations (snapshot during the suite, while idle between requests)

- VRAM used per GPU: 23.6 / 23.3 / 19.8 / 23.5 GiB (GPU0–3; GPU2 carries the smaller 9-layer session)
- RAM: 30 GiB used of 44 GiB (20 GiB pinned expert RAM tier included); no paging, no OOM
- Network expert reads stayed in the file cache; no SSD/NAS streaming collapse was observed at any prompt length

## Correctness and limitations

- Not measured: this is a speed suite. A separate needle/recall check (`tools/needle_bench.py`) was not run for this report.
- The chat decode range is wide (88.8–153.1) because short chats with MTP vary with draft acceptance; the agent-session median (127.1) is a more stable solo-decode figure.
- The conversation uses the model's own replies, so token counts differ a little between runs.
- Machine details come from the server, `nvidia-smi` and the OS; background load on the Proxmox host is not recorded. Each GPU runs at a **250 W enforced power limit** (factory default 350 W / 420 W for GPU1), which caps boost clocks under sustained load (the split-search log showed 1.73–1.80 GHz).
- With `--batch 2`, concurrent requests beyond one shared window lose MTP; per-request decode in shared windows is lower (measured separately at ~66 tok/s per stream for 2 concurrent streams). The `--clients 1` arm here is effectively solo.
- Vision was enabled on the server but not exercised by this suite (a separate image-description check produced an accurate description of a test image).
- Model files on CIFS: first-load warm-up and the 35.4 GiB non-resident expert tier read over the network; repeated expert reads are absorbed by the file cache. A machine with all experts resident (more VRAM) would likely post better short-prompt prefill numbers.
