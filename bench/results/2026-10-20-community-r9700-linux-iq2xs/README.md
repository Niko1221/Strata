# One R9700 with IQ2_XS: one client, two clients, and four levers that did not help

Measured on 2026-10-20 on one 32 GB Radeon AI PRO R9700 (gfx1201) with the original
ISTA-DASLab Flash-Next **IQ2_XS** pack and engine **0.1.46.1** (upstream `v0.2.1.1`,
`9ae93f7b`). It complements the earlier R9700 report
([2026-10-08-community-r9700-linux](../2026-10-08-community-r9700-linux/README.md)),
which used IQ3_S on an experimental 0.1.40.3 build: this one uses the current engine,
the IQ2_XS quantization, and it looks at what happens with **two** clients.

The results in one line: one client decodes at **74-78 tok/s** at 32K and at a **178K
context**, two clients at 32K share the GPU and get 22-46 tok/s each, and two clients at
178K do not share it at all - each ends up at 8-10 tok/s end to end. Four Strata-specific
levers were measured against that deep case and **none of them moved it**.

## Hardware, source and model

| Item | Recorded environment |
| --- | --- |
| GPU | Radeon AI PRO R9700, 32 GB, gfx1201, BDF `0000:09:00.0`; PCIe 4.0 x16, 28.3 GB/s probed host to device |
| Host | AMD Ryzen 9 5950X, 16 cores / 32 threads, 125 GiB RAM, Debian 13, kernel 6.12.107+deb13-amd64 |
| Engine | 0.1.46.1, HIP, gfx1201, built from source (`src 85733a7e5e473b25`), llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d` |
| ROCm | venv ROCm SDK 7.10.0a20251120 (gfx120X wheels); hipBLASLt **100200** |
| GPU state | 43 C idle, 300 W cap, no throttle; Strata was the only GPU process on the card (30.6 of 34.2 GB) |
| Model | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` IQ2_XS, 68 GB; shard 2 is the PLE table |
| MTP | the original Q2_0 draft runtime, 6.5 GB, from the model pack |
| Server | `serve/server.py` on `127.0.0.1:9797`, `parallel 2`, `--batch 2 --batch-mtp`, `--kv int8`, `--kv-resident 32768`, `--prefill auto:32768`, `--max-context 262144`, `--vram-reserve-mib 3072`, `--conversation-cache-mib 8192`, `--vision` (CPU encoder) |
| Attention | `STRATA_HIP_WMMA=1` (the gfx12 default) and an own hipBLASLt table for 100200 (43 rows) |
| Draft window | `--spec 6`, `--spec-min-p 0.35` (see "flags" below) |

The host is shared with other services; CPU interference is not excluded. The GPU was
dedicated to Strata for the whole run.

## Method

Measurements come from `llama-benchy` (the maintained Strata fork of
[eugr/llama-benchy](https://github.com/eugr/llama-benchy)) with `--strata`, which sets
`reasoning_effort=none`, `--no-cache` and filters cache artifacts:

```sh
llama-benchy --base-url http://127.0.0.1:9797/v1 --api-key "$KEY" \
  --model qwen3.8-flash-next --served-model-name qwen3.8-flash-next-iq2_xs \
  --tokenizer .../packs/iq2_xs/tokenizer --strata \
  --depth 131072 --pp 65536 --tg 64 --concurrency 2 --runs 2
```

- Fresh prompts every request; the tokenizer in the pack does not load in the tool, so it
  falls back to gpt2 and **token counts are approximate**. Depth and prompt sizes were
  chosen with margin, and the engine's own log reports the real token counts.
- `--depth` builds a prior conversation, so the depth is the same for every request in a
  run; `--concurrency 2` is two simultaneous clients.
- Client clocks (`t/s`) and the server's own `timings` (`srv t/s`) are reported separately.
  They agree within a few percent except where a note says otherwise.
- The engine's per-request log line (`generated in X ms (Y tok/s)`) and its batch summary
  (`strata batch: N windows, avg R rows, M ms/window`) are the mechanism evidence; they
  come from the same runs.

## Results

### One client

| Case | Prompt | Decode tok/s | Prompt tok/s | End to end |
| --- | --- | --- | --- | --- |
| 32K context | 32,768 | 74.2 ± 7.8 | 2,303 ± 152 | 12.9 s to first token |
| 32K context (run 2) | 32,768 | 75.7 ± 4.7 | 2,368 ± 190 | 12.6 s |
| 32K context (run 3) | 32,768 | 77.8 ± 5.7 | 2,291 ± 76 | 13.0 s |
| 178K context (depth 131,072 + prompt 65,536) | 178,205 tokens read | 75.4 ± 8.8 | 2,222 ± 6 | 80.2 s to first token |

The 178K read is the engine's own number: 178,205 tokens in 79,998 ms (2,228 tok/s).

### Two clients

| Case | Aggregate srv tok/s | Per request tok/s | Peak per request | Engine window |
| --- | --- | --- | --- | --- |
| 32K context | 24.4 - 25.9 | 22.4 ± 18.3 | 46.0 ± 4.9 | 3.05 - 3.29 rows, 36 ms |
| 178K context | 26.7 - 29.7 | 8.3 - 10.2 | 13.0 - 17.0 | 1.50 - 1.82 rows, 30 - 41 ms |

At 32K the two slots **overlap**: a window carries 3.05-3.29 rows and both clients make
progress. At 178K they **take turns**: a window carries 1.50-1.82 rows, and the MTP
drafter still works there (63-82% of proposals accepted), so this is not a draft failure.
A slot decodes at about 50 tok/s whenever it holds the GPU, but end to end a client sees
8-10 tok/s, because the other client's fresh 178K read (about 80 s of GPU-serial prompt
reading at 2,228-2,403 tok/s) sits in front of it.

Two clients at 256K each, both above 30 tok/s, do not fit on this card: reading 356K
tokens of prompt at about 2,300 tok/s is 155 s of exclusive GPU time, and the slots stop
overlapping windows at that depth.

### Flags measured, with the numbers

All measured with the same tool on the same engine build, same day:

| Change | Result |
| --- | --- |
| `--spec 6` instead of `--spec 4` | solo 32K 77.8 ± 5.7 against 73.3 ± 2.9 tok/s; two clients level (24.65 against 25.53 srv at 32K, 26.7 against 29.7 at 178K). The two arms were **not** interleaved, so the 6% solo difference is inside the run-to-run spread and is not claimed as a win. |
| `--spec-min-p 0.35` instead of 0.5 | 33 of 49 paired wins over five sweeps, +4-9% decode. Measured with the engine's per-request override (one engine, interleaved paired requests) and with thinking on, so the absolute rate is not comparable with the tables above; only the pairing is. |
| `--aux-cpus auto` | 4 of 8 paired reps, -0.6%. No effect; the 32 threads are shared with other services, so there is no spare CPU to move to. |
| `pcie_frac` | not a lever: the engine probes the link (28.3 GB/s) and already uses 0.55. |
| `--kv-resident 131072` (4x the resident cells) | two clients at 178K: 26.68 against 26.9 srv tok/s. No effect. |
| `--lookup-chain 4` | two clients at 178K: 8.86 ± 4.6 per request, 1.63-1.72 rows per window. No effect. |
| `--adapt-async 1` | two clients at 178K: 9.45 ± 5.2 per request, 1.68 rows per window. No effect. |
| `--mtp-window 4096` | two clients at 178K: 10.24 ± 5.9 per request, 1.82 rows per window; solo 32K interleaved with the base: 74.63 against 74.98 tok/s. No effect. |
| `--kv k8v4` | not usable here: a 4-bit V pool refuses the matrix-core prompt path (`qsa_prompt_attn.cu` needs `k_q4 == nullptr && v_q4 == nullptr`), the same gate as `q4_0`, and prompt reading is what starves the second client. |

An early reading of `--mtp-window 4096` looked like +13% prompt throughput. It was drift:
with the arms interleaved, the base gave 2,303.5 and 2,368.1 tok/s against the flag's
2,317.1 and 2,398.6. Single A/B runs are not enough on a shared box.

## What the two-client path costs

From the engine's own start-up lines on this build:

```
KV streaming: 32768 of 262144 cells per QSA layer in VRAM, the K/V in 3.09 GiB of pinned RAM
--batch: 2 slot sessions on CUDA0 (1.34 GiB each); 24.58 GiB free
--batch 2: the slot sessions take 1.90 GiB of VRAM on CUDA0 that the expert cache would otherwise hold
expert cache auto: 23.21 GiB free, 3072 MiB reserved -> 14251 slots
```

One 262,144-cell session costs **3.09 GiB of pinned RAM** (only the QSA layers store KV) and
a second batch slot costs **1.90 GiB of VRAM** taken out of the expert cache (15,858 slots
normally, 14,965 with `--kv-resident 65536`). RAM is not the limit on a 125 GiB host; VRAM
is. A usable prompt is `context - 8 - max_tokens`; beyond that the server answers HTTP 400
and never truncates.

## Two findings that may interest the engine

1. **This build is not bit-reproducible run to run.** Two identical greedy requests, same
   prompt and same settings, produced different text (the SHA-256 of the reply differed),
   and so did two requests with no override at all. The engine's own log says "the GPU
   computes the experts in the cache; it rounds differently from the CPU", and the
   adaptive tier moves experts between the GPU and CPU paths between requests. The
   identity gate described in the docs (10 prompts, 4K and 20K exact checks) does not
   reproduce here, so only speed claims are defensible on this build.
2. **The server shares engine batch slots across concurrent same-model requests.** A
   benchmark script that sends two clients the same template and the same word pool does
   not measure two fresh prompts: the second client reads one token and reports 6,876
   tok/s of prompt processing. The prompt has to differ per request, and the engine's
   log line `N reused` is the check.

## Measurement traps we hit (each cost hours)

- A repeat prompt hits the prompt and conversation cache: 4.3 tok/s artifacts, one-second
  walls, absurd prompt rates. `--strata` implies `--no-cache`, and the prompts still have
  to be fresh per request.
- Thinking defaults to high on Strata. A decode number without an explicit
  `reasoning_effort` measures thinking, not the engine.
- Interleave the arms. One A/B/B/A pair is not enough on a shared host.
- The API's own `timings` dilute a concurrent decode rate with the other client's prompt
  wait. The engine's per-request log line gives the decode phase on its own.

## What was not tested

- Not the `strata bench run` standard protocol, and no recall (needle) checks. The
  community guide asks for a same-day run of the build under test next to the candidate;
  we did not run upstream's build side by side, so these numbers are not a direct
  comparison with any other report.
- One card, one quantization (IQ2_XS), one engine build (0.1.46.1), one host with shared
  CPUs. Nothing here transfers to another card.
- The two-client numbers come from two concurrent requests of the same model. A mixed
  workload (one deep agent session plus one short interactive client) was not measured.
- The four depth levers were each measured once per configuration, not interleaved with
  the base. Their effect sizes are smaller than the spread, which is why the table above
  reports them as "no effect" rather than as deltas.
- `--spec-min-p 0.35` was measured with the engine's per-request override and with
  thinking on, not with this tool.

## Files

- [results.json](results.json): the numbers above, per run, with the configuration of each.

## Reproducing

The engine was run as a service (`serve/server.py`, `127.0.0.1:9797`) with the settings
in the first table. The commands are the `llama-benchy` invocations shown under "Method",
with `--depth 131072 --pp 65536 --tg 64 --concurrency 2` for the two-client case and
`--depth 0 --pp 32768 --tg 256 --concurrency 1` for the single-client case.

## Credits

The measurements were taken on the submitter's machine by an AI agent driving the engine
and the benchmark tool; the numbers are raw tool output, and the submitter reviewed them
before this report was filed.
