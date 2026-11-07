# Community benchmark: one Intel Arc Pro B70, three Strata builds (0.2.11, 0.2.12, 0.2.12 stock)

Measured on 2026-11-07 by [@rwkeyes](https://github.com/rwkeyes). One Intel Arc Pro B70, one model, one
configuration, three builds, run in six interleaved engine windows so machine drift lands on all three arms
instead of one.

This is a **community report in the format of [docs/COMMUNITY_BENCHMARKS.md](../../../docs/COMMUNITY_BENCHMARKS.md)**,
not a standard-protocol run: it carries no protocol ID, the sampling is greedy (`temperature 0`) rather than the
model card's, and there is no twelve-case sanity suite. The main limitation is at the bottom, together with the
two tuning observations that come with controls of their own.

The main result is small and was the reason to measure: **0.2.12 is about 1 % faster at decode than 0.2.11 on
this card and, within noise, identical to an unmodified stock 0.2.12 build** — i.e. the 0.2.12 engine change
costs nothing on SYCL and the ~1 % is the whole of the difference.

## Hardware and software

- **GPU:** Intel Arc Pro B70 (Battlemage G31), 32 GB, PCI `0000:03:00.0`, engine device 0; GuC firmware
  `xe/bmg_guc_70.bin 70.72.1`. A second GPU is present (AMD Raphael iGPU, `0000:1b:00.0`) and drives the display
  only; the engine never uses it.
- **CPU / RAM:** AMD Ryzen 7 7700X (8 cores, 16 threads) / 61 GiB.
- **Storage:** model shards and the pack on NVMe and a 12.7 TB data volume; experts are streamed from the GGUF at
  run time, so the model does not have to fit in VRAM.
- **PCIe:** the engine's own probe reports 7.0 GB/s host→device and derives `pcie_frac 0.19` (default 0.55).
- **OS / driver:** Ubuntu 24.04.5 LTS, kernel `7.0.0-42-generic`, in-kernel `xe` driver (no out-of-tree version).
- **Runtime:** Intel oneAPI 2026.2 (SYCL), AOT code for `bmg-g31`.
- **Builds:** source builds, `cmake -S sycl -B build-sycl-aot -DSTRATA_SYCL_AOT=bmg-g31` then `ninja -j10`, no
  patches. Commits, engine versions and full binary SHA-256 in [`provenance.json`](provenance.json).
- **Background workloads:** the machine's desktop only; no other inference client. The card's power limit was not
  read (not measured).

| Arm | Strata commit | Engine | Binary SHA-256 (first 12) |
| --- | --- | --- | --- |
| 0.2.11 | `3aa7fbfa7661` | 0.1.57 | `a97c282602cc` |
| 0.2.12 | `068bd337d062` | 0.1.58 | `8a62c7062284` |
| 0.2.12 stock | `146df3c46` (tag v0.2.12) | 0.1.58 | `510e9d6cd4e8` |

The 0.2.12 arm is a working tree with local commits on top of the tag — this report compares builds, it does not
describe a release. The stock arm is the unmodified tag with the same build options, which is what makes the
"0.2.12 ≈ stock" reading meaningful.

## Model and configuration

- **Model:** Qwen3.8-Flash-Next GSQ-RCO (**Coder**) IQ1_M, two shards, 29,608,446,496 + 28,800,138,432 bytes
  (54.5 GiB); pack `coder-iq1_m` (1.4 GB) with `expert-profile-mine.bin` (a profile produced by the engine's own
  calibration on this host) and `draft_vocab_en.bin`. File hashes: not measured (see limitations).
- **Context:** 131,072 tokens; `--kv int8`; `--kv-resident 32768`; conversation cache `--prompt-cache 6`,
  `--prompt-cache-every 4096`.
- **Drafting:** `--spec 6 --spec-min-p 0.5`, MTP draft layer `rt`, `--mtp-window 8192`.
- **Experts:** `--expert-cache auto --stream-experts --prefill auto`, `--vram-reserve-mib 700`, `--batch 0`
  (one slot).
- **No changes between the three arms:** the same config file and the same command line, only the binary differs.

```text
strata --serve --pack <nvme>/packs/coder-iq1_m \
  --native  <models>/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf \
  --ple-gguf <models>/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf \
  --expert-profile data/expert-profile-mine.bin --expert-cache auto --stream-experts --prefill auto \
  --prompt-cache 6 --prompt-cache-every 4096 --batch 0 \
  --spec 6 --spec-min-p 0.5 --mtp <nvram>/mtp/rt --mtp-draft-vocab data/draft_vocab_en.bin --mtp-window 8192 \
  --max-context 131072 --kv int8 --kv-resident 32768 --vram-reserve-mib 700
```

The full config with every argument is in [`provenance.json`](provenance.json); home and data paths are
placeholders there too.

## Method

- **Client:** loopback HTTP, `/v1/chat/completions` non-streaming, one request at a time, `temperature 0`,
  `top_k 1`, `top_p 1`, `min_p 0`, `seed 42`, `reasoning_effort: "none"` (thinking off, so the lanes measure the
  engine, not how long a scratchpad runs).
- **Order:** six engine windows — 0.2.11, 0.2.12, stock, 0.2.11, 0.2.12, stock — each a fresh engine start, so
  no arm inherits another arm's cache.
- **Lanes per window** (scripts in [`scripts/`](scripts)):
  - warm-up: `Reply READY.` (excluded from the tables; decoder warm-up cost shows up here);
  - `fresh` 4,210 and 8,830 prompt tokens: a generated file of 140 or 280 `export function ruleN(x) {...}`
    lines, asked to describe it; 128-token cap. Two trials each, and the script refuses a run whose
    `cached_tokens` is not 0, so these are genuinely fresh prompts;
  - `follow-up` 4,451 and 9,071 tokens: the same conversation plus the assistant's answer and a fixed
    follow-up question, which reuses the prompt prefix — this is the lane where the cache pays;
  - `long` 30,178 prompt tokens with a 1-token cap: prompt ingestion only;
  - `needle` 7,855 prompt tokens with a code line `// REMEMBER THIS CODE: ZQ-7731-KX` at record 125 of 250,
    asked to reply with only the code (32-token cap);
  - `code` a 32-token question with a 90-token cap: the short-request case.
- **Rates:** prompt and decode tokens/s are the engine's own `timings` (`prompt_n/prompt_ms`,
  `predicted_n/predicted_ms`), matched to each request through the engine log; `wall` is the client's monotonic
  clock around the whole request. Time to first token is not reported for the restart sweep (the client did not
  record it there) and is not the engine's number anywhere.
- **Repetitions:** 2 windows × 2 trials = 4 samples per fresh/follow-up lane per arm; the `long`, `needle` and
  `code` legs run once per window, so 2 samples. Medians with min–max in
  [`threeway-results.json`](threeway-results.json), which also carries every request record.

## Results

Prompt and decode tokens/s. Medians of 4 requests per lane for the fresh and follow-up lanes, 2 for the
single-request legs (`long`, `needle`, `code`); ranges are in the JSON.

| Lane | Prompt tokens | 0.2.11 prompt / decode | 0.2.12 prompt / decode | 0.2.12 stock prompt / decode | 0.2.12 vs 0.2.11 (decode) |
| --- | ---: | --- | --- | --- | ---: |
| fresh prompt, 4,210 tok | 4,210 | 1333.3 / 82.70 | 1332.7 / 83.00 | 1336.2 / 83.25 | +0.4 % |
| fresh prompt, 8,830 tok | 8,830 | 1396.8 / 80.85 | 1402.5 / 81.85 | 1400.2 / 81.85 | +1.2 % |
| follow-up, 4,451 tok | 4,451 | 326.6 / 74.25 | 329.8 / 74.75 | 330.5 / 74.55 | +0.7 % |
| follow-up, 9,071 tok | 9,071 | 272.9 / 72.45 | 274.6 / 73.05 | 273.4 / 72.95 | +0.8 % |
| prompt only, 30,178 tok | 30,178 | 1824.6 / 48.10 | 1826.8 / 48.60 | 1828.3 / 48.45 | +1.0 % |
| needle, 7,855 tok | 7,855 | 1432.8 / 96.25 | 1437.5 / 97.25 | 1433.0 / 97.05 | +1.0 % |
| code prompt, 32 tok | 32 | 59.4 / 113.20 | 60.0 / 114.45 | 59.0 / 114.15 | +1.1 % |

Reading it:

- **0.2.12 vs 0.2.11:** decode +0.4 % to +1.2 % depending on lane, prompt throughput within ±1 %. Consistent in
  sign across all six windows, but small enough that it should be read as "not a regression, a hair faster",
  not as an improvement to advertise.
- **0.2.12 vs stock 0.2.12:** −0.3 % to +0.3 %, i.e. the two builds are the same engine as far as this workload
  can tell. That is the point of the third arm: the local tree's changes did not move the speed either way.
- **Prompt throughput is the interesting number on this card:** ~1.4 k tok/s at 4–9 k, ~1.83 k tok/s at 30 k.
  Longer prompts get *faster* per token here, which is the AOT kernel plus the expert cache warming as the prompt
  is read, not a steady-state property.
- **The cached follow-up lane is ~270–330 tok/s of prompt** because only the tail is read again; the uncached part
  is what the number measures, and it is not comparable with the fresh lane.

### Tuning observation 1: draft depth (`--spec`)

Same build (0.2.12), same config, restart pair `A B B A`, the only changed setting is `--spec 6` (start-up value)
against `--spec 4`. 6 samples and 12 samples per lane, A and B windows alternating:

| Lane | spec 6 decode / prefill | spec 4 decode / prefill | change (decode) |
| --- | --- | --- | ---: |
| code (prompt ~3,907) | 85.85 / 1345.7 | 92.95 / 1340.8 | +8.3 % |
| extract24 (prompt ~36) | 59.02 / 153.5 | 75.27 / 149.6 | +27.5 % |
| extract96 (prompt ~45) | 65.17 / 97.4 | 84.95 / 95.6 | +30.4 % |
| reuse1 (prompt ~3,910) | 110.21 / 1347.0 | 129.37 / 1350.4 | +17.4 % |
| reuse2 (prompt ~3,910, cached) | 111.06 / 19830.5 | 130.17 / 20047.8 | +17.2 % |

Deeper speculation is not free: on short outputs and repeated contexts, every extra draft token that is rejected
costs more than it gains, so a smaller depth wins by 8–30 % *in these lanes*. Prompt throughput is unchanged
(±1 %) and time to first token is identical to the millisecond in all five lanes, so this is purely the acceptance
economics of the draft.

Two warnings. The `extract` lanes produce **empty answers on this build** — it has no thinking policy, so a
24- or 96-token cap goes into the reasoning block (the answer column in the JSON says so); their decode rates
compare reasoning tokens only. And this report does **not** measure the retrieval lane at depth 4, so it does not
support "use 4 everywhere". Depth should be chosen per request — that is a separate change with its own evidence,
not a constant to copy from this table.

### Tuning observation 2: `--kv-resident` changes a state, not a setting

Restart sweep on the same build, one engine window per arm, the same lanes as the main table:

| Arm (one engine window each) | fresh decode | fresh prefill | follow-up decode |
| --- | ---: | ---: | ---: |
| `--kv-resident 32768` (start-up value) | 82.40 | 1367.0 | 75.00 |
| `--kv-resident 65536` | 74.30 | 1470.7 | 79.90 |
| `--kv-resident 32768` (restored) | 82.10 | 1369.4 | 74.70 |
| `--conversation-cache-mib 4096` (cross-check) | 82.30 | 1381.1 | 74.90 |

At 65,536 resident KV cells this card trades **+7.6 % prompt throughput and +6.5 % cached-lane decode for
−9.8 % fresh decode**, and setting the flag back to 32,768 restores the baseline exactly (82.10 vs 82.40 decode,
1369.4 vs 1367.0 prefill). The effect follows the residency state the engine ends up in, not the flag that
produced it: a control arm that only changes the conversation cache lands on the same baseline numbers. Anyone
measuring a single arm of this flag is measuring whichever state their previous run left behind.

## Failures and skipped cases

- Every scheduled request completed; nothing failed, timed out or was skipped.
- The over-window request (a prompt of 165,779 tokens against `--max-context 131072`) returned **HTTP 400** with
  `request exceeds the context window … requests are never truncated`. That is the intended behaviour and it is
  excluded from the tables.

## Correctness and limitations

- **Recall:** the needle lane answered `ZQ-7731-KX` **6/6** across the three arms and both windows, exact text,
  `finish_reason: "stop"`. One needle, one depth (mid-document), at 7,855 tokens — this is a smoke test, not a
  long-context evaluation.
- **The comparison is greedy:** `temperature 0`, `top_k 1`. Generation length is fixed by the caps (128 tokens in
  the main lanes, 90 in `code`), so differences in when the model would have stopped do not leak into these
  rates. It also means these numbers say nothing about quality or about sampling behaviour.
- **Answers were empty in the `extract` lanes of the `--spec` pair** (see above); no quality claim is made there.
- **Not measured:** model file SHA-256 (54.5 GiB across two shards was not rehashed), GPU power draw and clock
  behaviour, VRAM high-water mark, TTFT in the restart sweep, any second GPU configuration, any concurrency
  (`--batch 0`, one slot), any other quantization.
- **One card, one model:** a single B70 with a Coder IQ1_M pack. Nothing here is a claim about the B60 pairs in
  the report index, about other Intel cards, or about 0.2.12 on CUDA/HIP.
- **This is not the standard protocol.** No `strata bench run`, no twelve sanity cases, no fixed corpus, so the
  numbers are not comparable with a standard-protocol result — only with another run of this battery.

## Files

| File | Contents |
| --- | --- |
| [`threeway-results.json`](threeway-results.json) | every request of the three-way comparison (72 records) and the lane summaries used above |
| [`spec-depth-results.json`](spec-depth-results.json) | the `--spec` restart pair: per-request records and lane summaries |
| [`kv-resident-results.json`](kv-resident-results.json) | the `--kv-resident` restart sweep |
| [`provenance.json`](provenance.json) | commits, engine versions, binary hashes, sanitized configuration, method |
| [`scripts/bench_prefill.py`](scripts/bench_prefill.py) | fresh and follow-up lanes, with the cache-reuse guard |
| [`scripts/strata-longprobe.py`](scripts/strata-longprobe.py) | code, 30 k prompt, needle and over-window legs |

Rates in every JSON are the engine's own timings; `wall_s` is the client's. Units are tokens/s and seconds.
