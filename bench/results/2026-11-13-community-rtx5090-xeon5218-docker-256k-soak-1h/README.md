# An hour of back-to-back requests on a 5090 in Docker: 4,025 requests, 0 failed, decode flat to 0.7% (2026-11-13)

One hour (3,601 s) of continuous requests against Strata **0.2.13** (official `0.2.13-cuda13` Docker image, engine
**0.1.59**) on an **RTX 5090 32 GB**, one request at a time, no pause between them. **4,025 requests, 4,025 ok,
0 failed, 0 timeouts, `/health` ok in all 336 samples.** This is a stability report, not a protocol benchmark: the
prompts are mine, not `bench/prompts/standard-v1`.

The point of sending it is the shape of the hour: **decode did not drift**. First-half median 264.8 tok/s,
second-half median 264.7 tok/s — a 0.04% difference over 4,025 requests — and VRAM was flat within 2 MiB the whole
time.

## Rig

- **GPU:** NVIDIA GeForce RTX 5090, 31.8 GiB (sm_120, driver 580.142, CUDA 13.0), one card, **no layer split**,
  `--parallel 1`.
- **CPU:** 2x Intel Xeon Gold 5218 @ 2.30 GHz, 32 cores / 64 threads, 2 NUMA nodes, no AVX-512.
  **RAM:** 125.6 GiB. **OS:** Ubuntu 24.04.5, kernel 5.15.0-58.
- **Docker:** official image, `--gpus device=1`, `--security-opt seccomp=unconfined`, `--shm-size 2g`.
- **Model:** Qwen3.8-Flash-Next **Q2_0**, **256K context**, `--kv int8 --kv-resident 16384`, `--vision gpu`,
  MTP `--spec 4`, tuning skipped. Expert cache at start: **14,570 slots** (21.19 GiB), `--vram-reserve-mib 2048`.
- Sampling: the model card's defaults, `temperature 0` in the harness so the answers are comparable run to run.

## What ran

Three prompt shapes, rotated in order, so the hour is a mix rather than one repeated request:

| Kind | Prompt | Max out | Requests |
| --- | --- | ---: | ---: |
| short | a one-line question ("what does HTTP 429 mean") | 120 | 2,013 |
| medium | an explanation request (~30-token prompt) | 320 | 1,342 |
| long | a ~2,400-token passage plus a question about it | 160 | 670 |

The harness (`soak.py`, standard library only) writes one row per request and one row per 10-second sample of
`nvidia-smi` and `/proc/meminfo`, so the timeline is in the data, not reconstructed afterwards.

## Results

| | value |
| --- | --- |
| Requests / ok / failed | **4,025 / 4,025 / 0** |
| Decode, all requests | median **264.8** tok/s (p10 108.7, p90 318.2, min 47.3, max 367.1) |
| Decode, short | median **281.4** (187.0-367.1), n=2,013 |
| Decode, medium | median **259.8** (199.1-284.9), n=1,342 |
| Decode, long (~2.4K prompt) | median **107.2** (47.3-113.0), n=670 |
| Wall time per request | median 0.61 s, max 1.70 s |
| `/health` | ok in **336 / 336** samples |

### No drift over the hour

Split into six equal windows of 670 requests:

| Window | Requests | decode median |
| --- | --- | ---: |
| 1 | 1–670 | 264.1 tok/s |
| 2 | 671–1,340 | 265.7 tok/s |
| 3 | 1,341–2,010 | 264.8 tok/s |
| 4 | 2,011–2,680 | 266.1 tok/s |
| 5 | 2,681–3,350 | 262.1 tok/s |
| 6 | 3,351–4,020 | 265.6 tok/s |

Spread across the windows: **262.1 to 266.1 tok/s, 1.5%**. There is no warm-up ramp and no fall-off: the first
window is the same as the last. The long-prompt third of the mix is why the overall p10 is 108.7 — those requests
carry a ~2.4K prompt and decode at a different rate for a reason (below), not because the card degraded.

### The card stayed where it started

| Metric (336 samples, every 10 s) | median | min | max |
| --- | ---: | ---: | ---: |
| VRAM used | **30,386 MiB** | 30,384 | 30,386 |
| GPU utilization | 100 % | 3 % | 100 % |
| GPU temperature | 48 °C | 30 °C | **51 °C** |
| Host memory available | 58.6 GiB | 58.4 GiB | 59.1 GiB |

VRAM varied by **2 MiB** across the whole hour, and the hottest the card got was **51 °C**. The 3% utilization
minimums are the gaps between requests, not a stall.

## Why the long prompts decode at 107 tok/s instead of 265

The three kinds differ in prompt length, and on this box prompt length is what moves decode:

- short/medium prompts are ~25-35 tokens, so almost the whole request is decode and the expert cache is warm from the
  previous request.
- the long kind is a ~2,400-token prompt. Past that prompt the engine is reading experts for a much wider context, and
  the decode rate settles near 107 tok/s. It is the same behaviour the standard benchmark shows going from 4K
  (242.7 tok/s) to 16K (214.4 tok/s) prompts, only further.

That is a property of the workload, and it is why I would not quote "265 tok/s" without saying which prompt shape it
came from.

## Failures, exclusions and limitations

- **Not the standard protocol.** These are my prompts, not `bench/prompts/standard-v1`; the numbers are not comparable
  with a `strata bench run` report. See [#5875](https://github.com/Niko1221/Strata/pull/5875) for that.
- **`temperature 0`, one request at a time, `--parallel 1`.** No batching, no concurrency, no sampling effects.
- **One card, one model.** No layer split, no second 5090, no IQ3_S.
- **The card was shared.** Another tenant's job was using ~4.7 GB of the 5090 for the whole hour; the engine had
  14,570 expert-cache slots of the ~30.4 GB it was holding. This is a normal contended state, not an empty card.
- The harness measures wall time with the client's clock and takes decode/prompt rates from the server's own
  `timings`, so client and server time are not mixed.
- Only one hour, once. No 24-hour run, no memory-leak sweep beyond what the flat VRAM says.

## Reproduce

```sh
docker run -d --name strata --gpus '"device=1"' --security-opt seccomp=unconfined --shm-size 2g \
  -p 8899:8080 -v <models>:/data \
  -e MODEL=Q2_0 -e CONTEXT=256k -e KV=int8 -e VISION=gpu -e PARALLEL=1 -e TUNE=skip \
  -e STRATA_SET="kv-resident=16384" -e API_KEY=<key> \
  ghcr.io/niko1221/strata:0.2.13-cuda13 run

SOAK_KEY=<key> SOAK_SECONDS=3600 SOAK_OUT=./soak python3 soak.py
```

## Files

| File | What |
| --- | --- |
| `requests.csv` | One row per request: kind, wall time, prompt/decode tokens and rates, draft acceptance, status, timestamp. |
| `samples.csv` | One row per 10 s: VRAM used/total, GPU utilization, temperature, host MemTotal/MemAvailable, `/health`. |
| `summary.json` | The aggregates above, as the harness wrote them. |
| `soak.py` | The harness itself (standard library only). |
