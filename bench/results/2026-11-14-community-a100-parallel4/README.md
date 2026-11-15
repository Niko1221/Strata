# A100 80 GB PCIe community benchmark — round 1

The long-output matrix measured 46.5–97.1 tok/s aggregate at ~6.5K context and 30.8–51.1 tok/s at 64K. At 64K x16, the conventional per-stream decode median was 8.2 tok/s.

## Environment

- NVIDIA A100 80 GB PCIe (sm_80, Gen4), Intel Xeon Platinum 8358P (AVX-512), approximately 504 GiB system RAM.
- Strata software v0.2.13.1 (operator-reported), internal engine v0.1.59.1, Qwen3.8-Flash-Next IQ3_S, native 262,144-token window, parallel 4. Exact engine build commit for this round was not captured.
- Another client shared the engine during measurement. The operator confirms `--batch-mtp` was disabled, with `--parallel 4`. No preserved engine launch log independently verifies these historical settings. Single-request speculative decoding was not isolated by this benchmark; model pack provenance was unavailable.

## Method

The client used OpenAI-compatible SSE streaming, temperature 0, and streamed usage. Synthetic prompts used randomized content and headers designed to avoid prefix reuse. Actual context sizes are approximate. The main matrix used an output limit of 1,024 tokens and generally stopped at around 90 tokens. The long-output matrix used a 1,600-token limit.

TTFT is time to the first content or reasoning delta. Per-request effective prefill divides prompt tokens by TTFT. Approximate per-stream decode divides reported completion tokens by time from the first to the last nonempty content or reasoning delta (the first-token boundary can bias the rate upward). The short-output matrix reports aggregate **input-token / full cell-wall-time** (prompt-token delivery proxy); the long-output matrix reports aggregate **completion-token / full cell-wall-time**. Both include queueing, scheduling, and failed attempts in the wall time. Neither is an isolated engine prefill/decode kernel rate. Medians are recalculated from request rows, averaging the two middle values for even sample sizes. Null metrics are excluded and recorded zero rates are included.

## Results

### Short-output matrix

| Context | Concurrency | Client OK | TTFT samples | TTFT median (s) | TTFT range (s) | Prompt/TTFT proxy median (tok/s) | Prompt/TTFT proxy range (tok/s) | Aggregate input/wall (tok/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64K | 1 | 1/1 | 1 | 20.5 | 20.5–20.5 | 3105.4 | 3105.4–3105.4 | 2658.0 |
| 64K | 4 | 4/4 | 4 | 49.9 | 21.3–78.4 | 1320.7 | 811.5–2991.5 | 3169.8 |
| 64K | 8 | 0/8 | 0 | — | — | — | — | 0.0 |
| 64K | 16 | 16/16 | 16 | 231.0 | 36.8–374.6 | 275.9 | 170.0–1727.4 | 2684.8 |
| 128K | 1 | 1/1 | 1 | 81.9 | 81.9–81.9 | 1552.3 | 1552.3–1552.3 | 1501.5 |
| 128K | 4 | 4/4 | 4 | 98.2 | 40.5–155.7 | 1345.7 | 816.3–3139.3 | 3211.0 |
| 128K | 8 | 8/8 | 8 | 182.0 | 47.8–316.7 | 706.1 | 401.2–2658.2 | 3182.4 |
| 128K | 16 | 16/16 | 15 | 378.7 | 64.7–712.9 | 335.7 | 178.2–1962.3 | 2645.8 |
| 250K | 1 | 1/1 | 1 | 115.8 | 115.8–115.8 | 2139.5 | 2139.5–2139.5 | 2038.9 |
| 250K | 4 | 4/4 | 4 | 214.0 | 92.9–354.3 | 1204.4 | 699.8–2667.4 | 2757.0 |
| 250K | 8 | 8/8 | 8 | 368.7 | 89.6–647.0 | 679.6 | 382.9–2766.0 | 3044.5 |
| 250K | 16 | 16/16 | 16 | 728.0 | 131.1–1324.6 | 341.4 | 187.1–1890.7 | 2983.9 |
| 64K | 8 | 8/8 | 8 | 87.5 | 20.5–154.0 | 735.5 | 413.0–3111.0 | 3263.8 |

### Long-output matrix

| Context | Concurrency | Client OK | TTFT samples | TTFT median (s) | TTFT range (s) | Decode proxy median (tok/s) | Decode proxy range (tok/s) | Aggregate output/wall (tok/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ~6.5K | 1 | 1/1 | 1 | 2.0 | 2.0–2.0 | 49.4 | 49.4–49.4 | 46.5 |
| ~6.5K | 4 | 4/4 | 4 | 6.8 | 4.0–14.7 | 33.1 | 25.8–35.4 | 97.1 |
| ~6.5K | 8 | 8/8 | 8 | 40.3 | 3.9–134.2 | 13.1 | 7.0–17.7 | 54.9 |
| ~6.5K | 16 | 16/16 | 16 | 133.2 | 5.1–269.5 | 18.1 | 6.8–82.9 | 88.6 |
| 64K | 1 | 1/1 | 1 | 17.3 | 17.3–17.3 | 114.5 | 114.5–114.5 | 51.1 |
| 64K | 4 | 2/4 | 2 | 28.5 | 18.4–38.6 | 12.0 | 0.0–24.0 | 9.3 |
| 64K | 8 | 8/8 | 8 | 111.9 | 22.2–298.5 | 14.1 | 6.8–71.1 | 39.9 |
| 64K | 16 | 15/16 | 15 | 211.5 | 30.5–669.9 | 6.8 | 0.0–60.0 | 32.2 |
| 64K | 4 | 4/4 | 4 | 57.2 | 18.8–95.4 | 23.8 | 17.0–41.8 | 47.9 |
| 64K | 16 | 16/16 | 16 | 311.5 | 31.3–645.4 | 8.2 | 2.2–16.3 | 30.8 |

## Failed requests and incomplete timing

- matrix, 64K × 8: 0/8 client OK.
- decode, 64K × 4: 2/4 client OK.
- decode, 64K × 16: 15/16 client OK.

The failed or partial cells were rerun successfully. The first matrix 64K x8 attempt failed at the connection level.

## Reproduction and limits

The accompanying Python clients are reference implementations for the two workloads. Set STRATA_BASE to the full chat completions URL and supply STRATA_API_KEY through the environment when required. Run python benchmark-matrix.py and python benchmark-decode.py from the same directory, using a clean output directory or moving existing output files first because the clients append results.

This is a shared-load baseline. The report contains no controlled idle run. Standard strata bench run, recall, warm-prefix, repeatability and 1M-context testing were outside this run.

The reference clients were prepared for publication. They report exception classes instead of full exception messages, and the decode client can summarize cells with error rows. The bundled measurements are unchanged. The JSONL files preserve aggregate failure counts but do not retain individual exception text for failed requests; the Python clients are publication-prepared reference versions, not byte-identical copies of the scripts used during measurement.
