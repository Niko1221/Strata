# A100 80 GB PCIe community benchmark — round 2

The long-output matrix measured 105.6–113.0 tok/s aggregate at ~6.5K context and 37.5–52.1 tok/s at 64K. The 64K x8 attempt completed 7/8 requests and delivered 41.7 tok/s aggregate.

## Environment

- NVIDIA A100 80 GB PCIe (sm_80, Gen4), Intel Xeon Platinum 8358P (AVX-512), approximately 504 GiB system RAM.
- Strata software v0.2.13.1 (operator-reported), internal engine v0.1.59.1, Qwen3.8-Flash-Next IQ3_S, native 262,144-token window, parallel 4. Exact engine build commit for this round was not captured.
- The operator reported `STRATA_BATCH_MTP=1` (batch MTP enabled), `KV=int8`, `CONTEXT=256k`, `memlock=-1`, and `PARALLEL=4` after Compose tuning. The engine was freshly restarted before testing and the harness reportedly generated no model traffic during this window. The client logs do not independently attest to the launch settings.

## Method

The client used OpenAI-compatible SSE streaming, temperature 0, and streamed usage. Synthetic prompts used randomized content and headers designed to avoid prefix reuse. Actual context sizes are approximate. The main matrix used an output limit of 1,024 tokens and generally stopped at around 90 tokens. The long-output matrix used a 1,600-token limit.

TTFT is time to the first content or reasoning delta. Per-request effective prefill divides prompt tokens by TTFT. Approximate per-stream decode divides reported completion tokens by time from the first to the last nonempty content or reasoning delta (the first-token boundary can bias the rate upward). The short-output matrix reports aggregate **input-token / full cell-wall-time** (prompt-token delivery proxy); the long-output matrix reports aggregate **completion-token / full cell-wall-time**. Both include queueing, scheduling, and failed attempts in the wall time. Neither is an isolated engine prefill/decode kernel rate. Medians are recalculated from request rows, averaging the two middle values for even sample sizes. Null metrics are excluded and recorded zero rates are included.

## Results

### Short-output matrix

| Context | Concurrency | Client OK | TTFT samples | TTFT median (s) | TTFT range (s) | Prompt/TTFT proxy median (tok/s) | Prompt/TTFT proxy range (tok/s) | Aggregate input/wall (tok/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64K | 1 | 1/1 | 1 | 18.2 | 18.2–18.2 | 3501.5 | 3501.5–3501.5 | 2869.7 |
| 64K | 4 | 4/4 | 4 | 55.0 | 21.8–95.6 | 1234.7 | 665.0–2915.5 | 2599.7 |
| 64K | 8 | 8/8 | 8 | 105.1 | 27.2–174.4 | 610.9 | 364.8–2340.2 | 2870.4 |
| 64K | 16 | 16/16 | 16 | 178.5 | 30.7–326.3 | 357.1 | 194.8–2072.2 | 3092.1 |
| 128K | 1 | 1/1 | 1 | 36.3 | 36.3–36.3 | 3503.7 | 3503.7–3503.7 | 3432.5 |
| 128K | 4 | 4/4 | 4 | 98.7 | 39.9–157.7 | 1340.3 | 805.9–3190.9 | 3139.8 |
| 128K | 8 | 8/8 | 8 | 185.3 | 47.0–323.2 | 693.5 | 393.3–2702.8 | 3109.7 |
| 128K | 16 | 16/16 | 16 | 354.9 | 59.1–650.4 | 359.3 | 195.5–2149.0 | 3109.5 |
| 250K | 1 | 1/1 | 1 | 76.8 | 76.8–76.8 | 3226.0 | 3226.0–3226.0 | 3192.5 |
| 250K | 4 | 4/4 | 4 | 203.8 | 82.2–325.4 | 1265.9 | 761.5–3014.8 | 2997.0 |
| 250K | 8 | 8/8 | 8 | 371.5 | 83.1–660.4 | 675.7 | 375.4–2982.7 | 2973.8 |
| 250K | 16 | 16/16 | 13 | 624.7 | 129.1–1117.9 | 396.5 | 221.7–1918.7 | 2865.3 |

### Long-output matrix

| Context | Concurrency | Client OK | TTFT samples | TTFT median (s) | TTFT range (s) | Decode proxy median (tok/s) | Decode proxy range (tok/s) | Aggregate output/wall (tok/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ~6.5K | 1 | 1/1 | 1 | 1.9 | 1.9–1.9 | 126.9 | 126.9–126.9 | 110.3 |
| ~6.5K | 4 | 4/4 | 4 | 8.7 | 2.0–15.2 | 31.4 | 27.6–35.3 | 105.6 |
| ~6.5K | 8 | 8/8 | 8 | 23.6 | 2.1–57.5 | 25.1 | 14.4–31.1 | 113.0 |
| ~6.5K | 16 | 16/16 | 16 | 78.1 | 3.0–191.7 | 21.7 | 7.9–40.9 | 110.9 |
| 64K | 1 | 1/1 | 1 | 17.3 | 17.3–17.3 | 119.4 | 119.4–119.4 | 52.1 |
| 64K | 4 | 4/4 | 4 | 58.9 | 19.6–99.8 | 23.8 | 19.0–43.7 | 46.9 |
| 64K | 8 | 7/8 | 7 | 97.2 | 20.9–189.9 | 18.3 | 0.0–39.7 | 41.7 |
| 64K | 16 | 16/16 | 16 | 254.4 | 30.3–612.8 | 10.0 | 3.5–22.7 | 37.5 |

## Failed requests and incomplete timing

- decode, 64K × 8: 7/8 client OK.

The long-output 64K x8 cell had 7/8 client-OK responses, but only 6/8 returned nonzero completion tokens (one of the seven OK rows recorded zero prompt and completion tokens). In the short-output 250K x16 cell, the client counted 16 requests as OK, while three had no first-content timing and no recorded completion tokens (13/16 with valid output). The available data does not establish a cause.

## Reproduction and limits

The accompanying Python clients are reference implementations for the two workloads. Set STRATA_BASE to the full chat completions URL and supply STRATA_API_KEY through the environment when required. Run python benchmark-matrix.py and python benchmark-decode.py from the same directory, using a clean output directory or moving existing output files first because the clients append results.

This is a follow-up under different deployment and background-load conditions. The operator confirms that round 1 had batch MTP disabled and round 2 enabled. However, round 1 used a shared engine while round 2 followed Compose tuning and an engine restart; the difference between rounds cannot be attributed to batch MTP alone. Standard strata bench run, recall, warm-prefix, repeatability and 1M-context testing were outside this run.

The reference clients were prepared for publication. They report exception classes instead of full exception messages, and the decode client can summarize cells with error rows. The bundled measurements are unchanged. The JSONL files preserve aggregate failure counts but do not retain individual exception text for failed requests; the Python clients are publication-prepared reference versions, not byte-identical copies of the scripts used during measurement.
