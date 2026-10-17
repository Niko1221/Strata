# The baseline profile of this release on this machine

```
measured:              2026-10-17T00:47:53Z … 00:56:19Z (UTC), a control point 00:58:58Z … 01:00:35Z
subject:               Strata 0.1.44.1 (engine 0.1.44.1), 4x Tesla V100-SXM2-16GB, --layer-split auto,
                       context 262144, KV int8, --spec 4, MTP draft, expert cache auto, vision on
service state:         running (no stop was needed), wrapper pid 63585, engine 63594, the pid file is current
tooling:               scripts/baseline-profile.py, sha256 01e2f4c38fcb0f8e87afe7edd2154519f2ceb68766512c108c30d5145aa00374
raw:                   raw/profiles/2026-10-17T004753Z/  (runs.json, baseline-points.csv, dmon-rxpci-txpci.csv,
                       gpu-samples.csv, conditions-before.json, conditions-after.json, engine-log-delta.txt,
                       engine-log-summary.txt, engine-log-cuda-errors.txt, run-notes.txt)
                       raw/profiles/2026-10-17T005858Z/  (the control point on an idle machine)
processed table:       baseline-profile.csv
comparison with PR #2525: comparisons/baseline-vs-pr2525.csv
units:                 tok/s = tokens/s; ms = 1e-3 s; MiB; MB = 1e6 bytes; degC; W
status labels:         measured / comparison / hypothesis / not measured / needs a service stop
change to the system:  none (the measurements went through the HTTP API of the running server; nothing was
                       installed or stopped, the config and the binaries were not changed)
secrets:               the api_key is not included; the config is identified by its SHA-256
```

---

## 0. The boundary: what was deliberately **not** re-measured

Reused without re-measuring:

| quantity | source (do not repeat) |
|---|---|
| the full long-context sweep 185 531 / 200 015 / 260 331 tokens, TTFT 72.6 / 78.5 / 102.8 s | `source/pr-2525/files/…/runs.json` (not included in this PR, see TRIMMED.md) |
| the previous 0.1.43 and the noise control on the same 0.1.44.1 binary | `runs-0.1.43-previous.json`, `runs-0.1.44.1-control.json` (same) |
| recall (needle) 6/6 at 32K and 128K, depths 10/50/90, both versions | `needles.json`, `needles-0.1.43.json` (same) |
| the host-to-card and card-to-card bands, pair competition, P2P latency | `p2p-results.md`, PR #2525 |
| the per-phase PCIe profile as a separate probe, uplink saturation, the `STRATA_PREFILL_HELP` A/B | PR #2525 `interconnect-experiments/` |

What **is taken fresh and is a new factor**: the output length is fixed (in PR #2525 it was not, so its decode
column was invalid). With it, first on this machine: ITL from the engine's timings, an estimate of the server's
overhead on TTFT, draft acceptance at a fixed length, GPU utilization / power draw / temperatures / clocks
during the measurements, and PCIe bytes per request window.

## 1. Conditions (measured)

| item | value |
|---|---|
| engine | `0.1.44.1` (`/v1/status.engine`), tag `v0.1.44.1` = `296b59d2f73c8f04a805df29c2fa97655adace78` |
| engine binary | `<HOME>/strata185/Strata/engine-cuda12/strata`, sha256 `bdfec9f553d0…b2280de6`, mtime 2026-10-14 16:01 local - **the same one** measured in PR #2525 (2026-10-14 22:11-22:53 local) |
| server config | `<HOME>/strata185/Strata/strata-185-iq3_s.json`, sha256 `f539e51b…4af88b`; **bit-identical** to the config copy in PR #2525 (16 keys, no differences; `api_key` excluded from the comparison) |
| model | `qwen3.8-flash-next-iq3_s`; `…-00001-of-00002.gguf` 54 817 524 224 B (mtime 2026-10-02T18:55:35Z), `…-00002-of-00002.gguf` 28 800 138 432 B (mtime 2026-09-30T14:44:33Z); pack `dense.bin` 1 538 035 200 B, `experts.bin` 50 292 326 400 B |
| model checksum | **partial** (SHA-256 of the first 256 MiB + the last 256 MiB of each file): shard 1 `63ae1b28a6bc1866751e2e28d7569731d3b7026bb8061aee935b42d9a4b9f9d4`, shard 2 `49795b4e0b3b78e70e99bb1b1e35a37c85cbdd76f12eea437e8e02e9be1039ca`. The full hash of 83.6 GB was not computed: it would evict the OS page cache the engine reads experts from and distort the following measurements. PR #2525 has no model hashes either - the comparison goes by size and mtime |
| GPU state before | 15246 / 15594 / 15182 / 16060 MiB used, 0 % utilization, 36-37 degC, 53-56 W, 1312 MHz (idle) |
| host background load | loadavg `0.72 0.69 0.66`; the machine up since `2026-10-15 18:25:56` local (= 15:25:56Z) |
| background load on the server | **recorded:** 4 foreign requests passed through the same port during the main run (about 20 585 / 20 882 / 21 361 / 21 726 prompt tokens) - see §7 |
| time zones | the machine's local time = UTC+3; everything here is UTC |

**Initialization time.** Not measured in this part (it would need a service stop). From existing records (a
stop/return window, `raw/logs/2026-10-17T001623Z/`): the engine process started at `2026-10-17 03:14:24` local
= **00:14:24Z** (the `engine started:` line in the working log), `/v1/status.started` = **00:15:11Z**, the first
successful `chat/completions` - **00:16:24Z**. That is **<= 120 s** from the engine process start to serving a
request; weight and cache loading is broken down in the log (1466 MiB of dense weights in 0.5 s, 300 native
matrices 2018.88 MiB in 0.5 s, MTP 1098 MiB in 0.42 s, filling the expert cache 5632 slots / 9.88 GiB,
capturing the verify windows). A precise initialization measurement is only possible in a stop window.
A later window measured it directly: **66 s** from the start command to the first served request
(`baseline-profile.md` §8.1).

## 2. Method, and how it differs from PR #2525

Matched to PR #2525's method so the numbers are comparable:

- the same prompt generator: `PARA_CODE` from the community bench script, the same formula
  `ratio = prompt_n / (len(PARA_CODE)*4 + 24)` -> the measured profile **0.1965 tokens per character**, and the
  actual prompt lengths matched the PR's: 3971, 15667, 31307, 125011, 163 tokens;
- a random marker at the start of the prompt -> the prefix cache is not reused (`cached = 0` in every measurement);
- `temperature = 0`, `reasoning_effort=none`, streaming; TTFT = the first non-empty delta;
- speeds come from the engine's timings (`prompt_per_second`, `predicted_per_second`, `predicted_per_token_ms`), not from client wall time;
- warm-up before the measurements (4k and 32k, 64 tokens each) - not part of the statistics; it also warms the expert cache and the window graphs after an engine restart;
- **3 repeats per point**, median + min/max + spread in %; inconvenient runs were not excluded.

**The changed factor (the reason this measurement was made): the output length is fixed.**

The engine does not support `ignore_eos`/`min_tokens` for text (the API only has `vision.min_tokens` for
images). So the EOS tokens `248044` (`<|endoftext|>`) and `248046` (`<|im_end|>`) - `serve/server.py:729`,
`--eos-ids` by default - were banned through `logit_bias = -100`. A pre-check of the method
(`raw/profiles/2026-10-17T004452Z/`): `max_tokens = 32` -> `gen_n = 32/32`, `max_tokens = 64` -> `gen_n = 64/64`.
In all 18 runs of the main sweep and all 6 runs of the control point `gen_n == usage_completion == max_tokens`
and `finish = "length"` - **24/24** (measured, `runs.json` -> `fixed_length_ok`).

A deliberate side effect of the method: generation runs to the limit, so the text is a forced continuation, not
a natural answer. That is correct for speed and ITL metrics, and is not suitable for judging answer coherence.

Background tools run alongside (read-only):

- `nvidia-smi dmon -s t -d 1` - `rxpci/txpci` for 4 cards, 1649 rows; summed per request window = the "PCIe reverse counter" (there are no NVLink byte counters on this driver, see `data-path-analysis.md`);
- `nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.current.sm,clocks_throttle_reasons.active` every 2 s - 944 samples;
- the growth of the engine's working log (33 342 -> 33 479 lines, 137 lines), parsed per request and searched for CUDA errors.

## 3. The baseline profile (measured, median of 3 repeats)

All runs: `gen_n == max_tokens`, `finish_reason = "length"`, zero reused tokens.

| point | prompt tok (actual) | prompt tok/s, median [min-max] (spread) | decode tok/s, median [min-max] | ITL, ms | TTFT, s | wall, s | drafts accepted |
|---|---:|---|---|---:|---:|---:|---|
| 4k | 3 971 | **868.7** [857.9-869.3] (1.31 %) | **99.0** [93.4-100.6] | 10.10 | 4.59 | 7.19 | 485/699 = 69.4 % |
| 16k | 15 667 | **1515.1** [1513.4-1516.3] (0.19 %) | **109.6** [101.2-110.7] | 9.12 | 10.38 | 12.71 | 517/645 = 80.2 % |
| 32k | 31 307 | **1892.6** [1887.7-1893.0] (0.28 %) | **104.2** [104.0-104.4] | 9.60 | 16.59 (1) | 19.03 | 520/638 = 81.5 % |
| 64k | 62 519 (2) | **2213.5** [2209.9-2216.3] (0.29 %) | **96.2** [94.2-97.1] | 10.40 | 46.50 (1) | 49.13 | 502/663 = 75.7 % |
| 128k | 125 011 | **2372.8** [2372.3-2377.1] (0.20 %) | **98.1** [95.6-99.1] | 10.20 | 52.78 | 55.35 | 515/679 = 75.8 % |
| gen-only (decode 1024 tokens) | 163 | 196.2 [192.6-196.4] | **109.0** [107.7-117.5] | 9.18 | 0.85 | 10.23 | 2097/2695 = 77.8 % |

(1) The third 32k run gave TTFT 33.81 s, and all three 64k runs gave 44.7 / 46.5 / 47.3 s while the engine's
own `prompt_ms` was 28.2 s: a queue for the single serving slot (`concurrency/serving = 1`) behind foreign
requests from the same client (§7). The runs were not excluded; the table shows medians.
(2) The target was 65 536 tokens; the actual length 62 519 is 1.6 % shorter than the PR's 63 539 point (tok/s
is normalized per token, but the row is marked).

The control point on an idle machine (`raw/profiles/2026-10-17T005858Z/`, no foreign requests in the window):
31 307 tokens -> **1889.1** [1884.7-1894.4] tok/s, decode 99.7, TTFT 16.61 s. The difference from the main run
is **0.19 %** -> the background foreign requests did not distort prompt throughput.

**The noise band of the method, measured again:** prompt tok/s - a spread of 0.19-0.29 % at 16k-128k, 1.31 % at
4k; decode tok/s - 0.38-8.99 %; TTFT - 0.2-0.5 % (without foreign requests). This matches PR #2525's estimate
(plus/minus 0.3 % on prompt) and confirms: **an effect of a change below 1 % on prompt tok/s cannot be shown by this method.**

### The length-to-speed curve

prompt tok/s: 868.7 -> 1515.1 -> 1892.6 -> 2213.5 -> 2372.8 at 3.9k -> 15.7k -> 31.3k -> 62.5k -> 125k. The
increase with length continues even at 125k (+7 % over 62.5k); within noise the curve is monotonic. The PR's
"plateau" (2560 at 185k, -1 % at 260k) was not re-measured here (reused from the PR).

decode tok/s at a fixed length: 99.0 (4k) -> 109.6 (16k) -> 104.2 (32k) -> 96.2 (62.5k) -> 98.1 (125k) ->
109.0 (pure decode of 1024 tokens). ITL 9.12-10.40 ms. The repeat spread on decode reaches 9 % - **decode needs
at least 3 repeats and a median; a single run carries no information.**

## 4. Resources by phase (measured)

### 4.1 PCIe per single-request window, median, GB per card (rxpci / txpci)

| point | RX GPU0 | RX GPU1 | RX GPU2 | RX GPU3 | TX GPU0 | TX GPU1 | TX GPU2 | TX GPU3 | RX per 1k prompt tokens, MB/card |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4k (3 971) | 2.6 | 6.7 | 5.6 | 4.4 | 0.56 | 0.71 | 0.81 | 0.59 | 659 / 1691 / 1420 / 1097 |
| 16k (15 667) | 12.3 | 7.1 | 13.5 | 8.7 | 1.76 | 0.78 | 1.52 | 1.07 | 782 / 456 / 865 / 553 |
| 32k (31 307) | 14.7 | 25.8 | 20.4 | 29.6 | 2.72 | 2.64 | 2.37 | 3.57 | 469 / 824 / 651 / 945 |
| 64k (62 519) | 45.7 | 56.6 | 53.9 | 68.8 | 9.53 | 8.82 | 7.51 | 8.12 | 732 / 905 / 863 / 1101 |
| 128k (125 011) | 59.4 | 65.2 | 62.1 | **85.7** | **21.1** | **20.3** | 10.3 | 9.2 | 475 / 521 / 497 / 685 |
| gen-only (163) | 0.03 | 0.09 | 0.09 | 0.28 | 0.07 | 0.08 | 0.08 | 0.04 | - |

Compared with PR #2525's per-phase profile (prefill 125 351 -> 61.9-78.4 GB RX and 6.9-11.9 GB TX per card):
**RX reproduces** (59.4-85.7 GB per card at 125 011 tokens). **At lengths >= 32k the busiest card by inbound
traffic is GPU3** (945 / 1101 / 685 MB per 1k prompt tokens, 1.4-2.0x the least loaded card of the same point);
at 4k and 16k the leaders were GPU1 and GPU2. **TX on GPU0/GPU1 is higher than in the PR** (21.1 / 20.3 GB
against 6.9-11.9 GB) - the difference matches the direction of a different layer split (§6). The direction
"host-to-card dominates" is confirmed again: RX/TX = 2.8-9.4x (minimum 2.8x on GPU0 at 125k, maximum 9.4x on GPU1 at 4k).

Method limits: a 1 s polling step and skipped seconds make these **lower bounds**; the request window includes
the HTTP exchange and the generation tail; NVLink traffic is not in these counters (the counters' unavailability
was proven). At idle `rxpci/txpci = 0` (confirmed by samples before and after the runs).

### 4.2 VRAM, utilization, power, clocks

| card | used before | peak during | delta | peak util | peak temp | peak power | SM clock |
|---|---:|---:|---:|---:|---:|---:|---|
| GPU0 | 15246 MiB | 15246 MiB | 0 | 100 % | 47 degC | 295 W | 1530 MHz |
| GPU1 | 15594 MiB | 15600 MiB | +6 | 100 % | 49 degC | 319 W | 1530 MHz |
| GPU2 | 15182 MiB | 15188 MiB | +6 | 100 % | 48 degC | 319 W | 1530 MHz |
| GPU3 | 16060 MiB | 16068 MiB | +8 | 100 % | 46 degC | 316 W | 1530 MHz |

- **VRAM does not grow with prompt length**: the increase over the whole sweep is <= 8 MiB per card. KV pools
  are allocated at engine start; the risk of a long context is not "out of VRAM" but exhausting the pool/window.
  Matches the PR (there 0-24 MiB).
- Utilization: 100 % on all four cards during prefill >= 16k; at 4k GPU0 reached only 22 %, the others 99-100 %; 35-38 % during pure decode.
- Power: in 499 of 944 samples `power.draw` exceeded the rated 300 W limit (a maximum of 319 W on GPU1/GPU2/GPU3) during prefill >= 32k; 115-180 W during pure decode.
- Clocks: 813 of 944 samples at the maximum 1530 MHz, **97 samples at 1425-1522 MHz**, 34 at 1312 MHz (idle). `clocks_throttle_reasons.active` came back empty for that query, so the cause of the clock reduction is **not established**; the coincidence in time with power peaks is an observation, not proof.
- CUDA errors, `no kernel image is available`, `invalid device function`, OOM in the log growth: **0 occurrences** (`engine-log-cuda-errors.txt`, empty).

## 5. Comparison with PR #2525's baseline (comparison)

The reference points are PR #2525's `runs.json` (0.1.44.1, same machine, same build, 2026-10-14),
`runs-0.1.44.1-control.json` (its noise band) and `runs-0.1.43-previous.json`. Full table: `comparisons/baseline-vs-pr2525.csv`.

| prompt tok (actual) | PR 0.1.44.1, tok/s | **now**, tok/s | delta | PR control (same binary) | delta vs 0.1.43 |
|---|---:|---:|---:|---:|---:|
| 3 971 | 881.6 | 868.7 | **-1.46 %** | 879.6 (noise 0.31 %) | -0.86 % |
| 15 667 | 1548.5 | 1515.1 | **-2.16 %** | 1546.2 (0.39 %) | -1.78 % |
| 31 307 | 1985.3 | 1892.6 | **-4.67 %** | 1982.3 (0.13 %) | -4.08 % |
| 63 539 / 62 519 | 2345.3 | 2213.5 | **-5.62 %** | 2348.0 (0.27 %) | -5.68 % |
| 125 011 | 2490.7 | 2372.8 | **-4.73 %** | 2495.8 (0.08 %) | -5.17 % |
| 163 (gen-only) | 203.7 | 196.2 | **-3.68 %** | 205.2 (0.34 %) | -2.29 % |

TTFT (comparable, the method is the same):

| prompt tok | PR, s | now, s | delta |
|---|---:|---:|---:|
| 3 971 | 4.53 | 4.59 | +1.3 % |
| 15 667 | 10.15 | 10.38 | +2.3 % |
| 31 307 | 15.81 | 16.59 | +4.9 % |
| 63 539 / 62 519 | 27.16 | 46.50 | **not comparable**: all three runs of this point queued behind foreign requests (`server_overhead about 18 s`, §7); without the queue it would be about 28.2 s (the engine's `prompt_ms`) |
| 125 011 | 50.29 | 52.78 | +4.9 % |
| 163 | 0.82 | 0.85 | +3.7 % |

Decode: **not comparable** - in the PR the model stopped at 93-182 tokens out of 256 (the column is invalid);
here the length is fixed (256/1024). The PR's numbers (94.6-119.5 tok/s) can be neither confirmed nor refuted
by this measurement; valid values are in §3.

VRAM peak per card: PR `[15980, 15386, 15034, 16128]` MiB, now `[15246, 15600, 15188, 16068]` MiB. In total
**426 MiB less** (62 102 against 62 528 MiB), but **the distribution shifted**: GPU0 is 734 MiB lighter,
GPU1/GPU2 heavier (+214 / +154 MiB), GPU3 60 MiB lighter. This is a direct consequence of a different layer split (§6).

**Conclusion about the delta (measured in the part "this is not the version and not background load"):** the
prompt tok/s drop of 1.5-5.7 % is stable across all 3 repeats, 5-20 times the noise band, and reproduced on an
idle machine (the control point, delta 0.19 %), with the **same binary** (sha256 matches), the **same config**
(sha256 matches the PR's copy) and the **same tag** `v0.1.44.1`. So the difference belongs to the **state of the
machine / engine session**, not to the Strata version and not to the hardware. The specific cause was established
by the experiment in §8: the `--layer-split auto` choice.

## 6. Why the profile is lower than the PR's: session-state facts plus a hypothesis

Two runs of the same binary compared through the working log (the PR's session from line 21677+, started
2026-10-14 16:02:18 local; the current one from line 33145+, started 2026-10-17 03:14:24 local = 00:14:24Z).
Free memory **before** layer placement is identical (11.26 / 14.79 / 14.79 / 13.81 GiB); from there the sessions diverge:

| session parameter | PR #2525 (14 Oct) | now (17 Oct) |
|---|---|---|
| layer split (`layer split auto`) | `K=12,25,37` -> CUDA0 0-11 (12 layers), CUDA1 12-24 (13), CUDA2 25-36 (12), CUDA3 37-47 (11) + head | `K=11,24,36` -> CUDA0 0-10 (**11**), CUDA1 11-23 (13), CUDA2 24-35 (12), CUDA3 36-47 (**12**) + head |
| the engine's prediction for a decode window | 28.4 ms | 28.4 ms (the same) |
| expert cache CUDA0 | 5846 slots (10.29 GiB) | **5632 slots (9.88 GiB)** |
| expert cache CUDA3 | 5441 slots, holds 5441 of **5632** profiled pairs (96.6 %) | 5469 slots, holds 5469 of **6144** pairs (**89.0 %**) |
| resident experts total | 24 087, "98 % of the experts resident" | **23 901, "97 %"** |
| slot borrowing by the prompt path | 1932 / 1841 / 1752 / 1625 | 1954 / 1841 / 1706 / 1631 |
| free after load (MiB) | 296 / 914 / 1262 / 172 (LOW on CUDA3) | 1018 / 632 / 1036 / 170 (LOW on CUDA3) |
| the expert profile file | `expert-profile-learned-185-iq3s.bin` | the same file, **but it had been overwritten between sessions** (`--expert-profile-save-every 10`; the log growth of this measurement contains `expert profile saved … (periodic)`) |

A fact (not a hypothesis): the split and the expert-cache placement in the current session are different, and
**the resident share of experts on CUDA3 fell from 96.6 % to 89.0 % while CUDA3 got one more layer and holds the
head**. This is consistent with §4.1 (at lengths >= 32k the PCIe inbound leader is GPU3, 685-1101 MB per 1k
tokens) and with the speed drop growing with prompt length (-1.5 % at 4k -> -4.7…-5.6 % at 62-125k): the longer
the prompt, the more accesses go to non-resident experts.

**Hypothesis (tested in §8 -> confirmed):** the main cause of the delta is the `layer split auto` choice and the
expert-cache placement that goes with it, which the engine makes at start from free VRAM and from the learned
expert-profile file; that file had been rewritten many times since 14 October by other sessions, so "the same
version on the same machine" got a different split (K=11,24,36 instead of K=12,25,37) and a different placement.

Secondary candidates (measured, not quantified): the power-limit exceedance (up to 319 W against 300 W) and 97
samples at 1425-1522 MHz; the state of the OS page cache (in this measurement `file tier I/O` was 0-15 MB of
real disk reads per request, so the disk was barely involved; the PR did not record this parameter).

## 7. What was in the window of the main measurement (no run was excluded)

- **4 foreign requests** passed through the same server in the same window (about 20.6k, 20.9k, 21.4k, 21.7k prompt tokens, generating 36-363 tokens). `concurrency/serving = 1`, so they queued ahead of my requests: hence TTFT 33.81 s on the third 32k run and `server_overhead about 18 s` on all three 64k runs. This is reflected in the `ttft_s_max` and `server_overhead_s_median` columns of `baseline-profile.csv` and in `runs.json` (`tail_s`, `server_overhead_s`).
- The control point on an idle machine showed the same prompt tok/s (1889.1 against 1892.6) -> foreign requests distorted only TTFT/wall, not throughput.
- `server_overhead_s` (estimated as TTFT minus the engine's `prompt_ms`) outside queues: 0.02-0.09 s. So the server's normal overhead (HTTP, tokenization, the slot) is small; every noticeable overhead in this measurement was a queue.
- `tail_s` is about -0.02 s at every point: there is no tail after the last token.
- Correctness checks: **24/24** runs (18 main + 6 control) with `gen_n == max_tokens` and `finish = length`; **0** CUDA error lines in the log growth; per-request expert-cache hit rate 98.3-100.0 %; greedy determinism confirmed (the same prompt twice -> identical text, SHA-256 `cf94ba5b9a530efd…` in both answers; the second call reused 224 prefix-cache tokens).
- Recall (needle) was not re-measured: 6/6 in PR #2525 at 32K/128K, depths 10/50/90, for both versions.

## 8. The delta attribution experiment (a stop window, 2026-10-17T01:51…02:21Z) - the cause found

Raw: `raw/restart-attribution/2026-10-17T015058Z/` (stop-start.log with timings, session-layout-before/after/restored.txt, processes-*.txt, first-request-check-*.json, engine-split12.log, server-split12.out), `raw/profiles/2026-10-17T015257Z/` (a fresh auto session), `raw/profiles/2026-10-17T020549Z/` (a pinned split).
Comparison table: **`comparisons/auto-vs-pinned-split.csv`**.

### 8.1 Step 1 - a full restart by the normal scripts (the "transient state" hypothesis) - **refuted**

`stop-strata-iq3_s.sh` (01:51:19Z) -> VRAM freed to 7 MiB on every card -> `start-strata-iq3_s.sh` (01:51:44Z).
**Initialization time: 66 s** from the start command to the first served request (`/health` 200 after 60 s - that
is the wrapper; VRAM >= 14 GiB and the first real `chat/completions` after 65-66 s). The return was verified by
three signs.

The new session chose **exactly the same layout** as the previous one: `K=11,24,36`, CUDA0 cache 5632 slots,
CUDA3 5469 of 6144 pairs, 23901 resident experts (97 %), the same slot borrowing. The full sweep (6 points x 3
repeats, 01:52:57…02:00:09Z) gave the same speeds as the warmed session:

| point | warmed session (00:48Z) | fresh session (01:53Z) | delta |
|---|---:|---:|---:|
| 3 971 | 868.7 | 865.9 | -0.32 % |
| 15 667 | 1515.1 | 1517.5 | +0.16 % |
| 31 307 | 1892.6 | 1893.2 | +0.03 % |
| 62 519 | 2213.5 | 2204.5 | -0.41 % |
| 125 011 | 2372.8 | 2372.4 | -0.02 % |

Conclusion (measured): the profile **does not depend** on session uptime, warm-up or background requests; the
layout choice is **deterministic** (two independent starts produced an identical set of numbers). This also
corrects §6: by the number of held pairs the difference from the PR is minimal (24270 against 24279 of 24576,
about 100 % of "routed mass" in both sessions) - what diverge are the split boundaries and the CUDA0 cache size
(5632 against 5846 slots), not the volume of held experts.

A side observation: the TTFT of the 62 519 point in the fresh session was **28.4 s** (against 46.5 s in the main
run), i.e. the earlier difference was entirely a queue for the single serving slot; relative to the PR (27.16 s)
that is +4.6 %, as at other lengths.

### 8.2 Step 2 - pinning the PR's split (the "auto-split choice" hypothesis) - **confirmed**

The same binary, the same config, but `--layer-split 12,25,37` instead of `auto` (the config variant
`configs/strata-185-iq3_s.split12.json`, sha256 `43d44198…`, a separate log file in the measurement folder). The
engine reproduced the PR session's layout: CUDA1 6656/6656 pairs, CUDA2 6144/6144, **CUDA3 5443 of 5632 pairs
(96.6 %)**, 24387 resident experts (99 %), CUDA0 cache 6144 slots.

| point | auto (K=11,24,36) | **pinned 12,25,37** | delta | PR #2525 (auto -> 12,25,37) |
|---|---:|---:|---:|---:|
| 31 307 tok/s | 1893.2 | **1978.0** | **+4.48 %** | 1985.3 (a 0.4 % hit) |
| 125 011 tok/s | 2372.4 | **2546.2** | **+7.33 %** | 2490.7 (2.2 % above the PR) |
| 163 (gen-only) tok/s | 202.5 | 207.5 | +2.47 % | 203.7 |
| decode 256 tokens, tok/s | 99.3 / 97.5 | 101.1 / 96.5 | within noise | invalid in the PR |
| ITL, ms | 9.79-10.26 | 9.61-10.37 | unchanged | - |
| TTFT 31 307, s | 16.57 | **15.87** | -4.2 % | 15.81 (a 0.4 % hit) |
| TTFT 125 011, s | 52.80 | **49.19** | -6.8 % | 50.29 |

**Conclusion (measured):** the 1.5-5.7 % delta against PR #2525 is explained by **the `--layer-split auto`
choice**. With the PR's split pinned, the current build does not merely catch the PR baseline - it exceeds it on
long prompts (+2.2 % at 125 011 tokens). The decode column and ITL do not depend on the split.

The mechanism (a log observation, not a measurement): the auto placement search takes a layer away from CUDA0 in
favour of CUDA3 (11 instead of 12 layers), CUDA0 gets 214 fewer expert-cache slots, CUDA3 holds 89.0 % of its
profiled pairs instead of 96.6 % - and the loss is largest exactly at the lengths with the most expert accesses.
The engine's own estimate ("predicted 28.4 ms per decode window") is identical for both splits, i.e. the auto
search's prediction does not reflect the real cost of prefill.

### 8.3 What this implies for future measurements

1. **Compare the layout, not only the version.** Each session's log prints it (`layer split auto: K=…`, `expert
   cache N slots`, `N of its M profiled pairs`) - that is a mandatory part of any measurement protocol on this machine.
2. The working configuration (`auto`) is **slower** than pinned `12,25,37` by 4.5 % (32k) and 7.3 % (128k) with
   unchanged decode. Moving to a pinned split is a change of the working configuration, so it was raised as a
   question; **the decision was not to move, see §8.5.**
3. The profile in §3 remains a correct description of the **working** configuration (`auto`); the pinned-split
   profile is a separate line in `comparisons/auto-vs-pinned-split.csv`.

### 8.4 An incident in the stop window (read it before the next window)

A `serve/server.py` wrapper started manually with a relative cmdline (`.venv/bin/python serve/server.py …`) **is
not stopped** by the normal `stop-strata-iq3_s.sh`: its cmdline does not match the pgrep pattern, and SIGINT to
the pid from the pid file does not terminate it - it survives the engine stop, holds port 8080, and after 15 s
restarts its own engine (`serve/server.py: restart(), RESTART_RETRY_S = 15`). Because of that the standard start
could not take the port (`server.py: error: port 8080 is already in use`) and died, leaving a stale pid file.
The fix: `kill -TERM <the actual wrapper pid>` (from `ss -ltnp | grep ':8080 '`), then check free VRAM, then the
normal start. **Rule:** start the experiment's service variant the same way as the main one (an absolute path to
python), otherwise it falls out of the stop script's field of view.

Another observation: the first `chat/completions` right after VRAM was OK (46 s after start) returned an empty
body; a retry a few seconds later succeeded (3/3). Do the return check after the `engine started` line appears in
the engine log, not from a single `/health`.

### 8.5 The decision: the working configuration stays `--layer-split auto` (a note for the future)

**The operator's decision (2026-10-17T02:55Z):** do not switch. The reason is the risk for a request whose topic
has experts outside the learned profile: without auto a card cannot get more slots than the start gave it, and
such a request could become noticeably slower.

**What exactly is fixed at start and does not change at runtime** (from the initialization lines,
`raw/restart-attribution/2026-10-17T015058Z/session-layout-*.txt`):
1. which layers are on which card (`CUDA3 runs layers 36-47`);
2. how many expert-cache slots each card has (`expert cache 5469 slots (11.29 GiB), 5469 of its 6144 profiled pairs`);
3. which pairs count as resident (`23901 resident experts`).

**What adapts at runtime:** the cache contents inside the allocated slots (the resident hit path plus the
streaming ring, `the prompt path's streamed ring 96 slots`) and the expert profile file itself
(`--expert-profile-save-every 10`). But the profile influences the **next** start, not the current one.

**So the concern is exactly right in this sense:** for an unprofiled topic the narrow card (CUDA3, the only one
short of slots for all its pairs - 5469 of 6144) cannot get more slots than the start allocated. `auto` would
re-allocate layers at the next restart to match the updated profile; a pinned split would not. The pinned
variant's 4.5-7.3 % loss on long prompts is bought precisely with that rigidity.

**How large the risk is, from already measured data (an upper bound):** the worst observed values in real
traffic are a hit rate of **98.3 %** (the baseline sweep) and **99.1 %** (the correctness check,
`raw/correctness/2026-10-17T022637Z/engine-log-delta.txt`). That request still gave 91.3 tok/s decode and 18.6
MiB of OS reads against 2070.7 MiB of expert accesses (0 major faults - the data came from the page cache). So
the penalty for unprofiled experts is measurable but not catastrophic in the observed range; there was no failure
or refusal.

**Effect on the answer:** only as a change in the order of floating-point operations. The correctness work
established the mechanism (`correctness-report.md` §5, matrix row B17): a different expert service path changes
values on close logits, so isolated text differences are possible (the same mechanism produces the first-request
non-determinism). That is not a "wrong answer", but the text can differ.

**The condition to revisit this decision (signals in the engine log):** a sustained per-request expert-cache hit
rate below 98 %, file-tier accesses growing beyond the observed about 30 MiB of OS reads per request, or a change
in the topic/length of requests. Then reconsider in both directions: pin a new split (for the new profile) or keep `auto`.

**A cheap risk test without changing the configuration (experiment E1, about 4 minutes of a stop window):** take a
"cold" topic whose experts are clearly outside the profile's top rank, run it x3 under `auto` and x3 under
`configs/strata-185-iq3_s.split12.json` in one window; compare hit rate, prompt/decode tok/s, file-tier I/O
volume and `text_sha` (with a fixed output length and warm-up). Not run yet.

There is a second, more important part of that experiment: **testing the adaptivity of auto itself.** After
running the cold topic, record the checksum of the profile file
(`<HOME>/strata185/data/expert-profile-learned-185-iq3s.bin`, `--expert-profile-save-every 10`), then restart the
service on `auto` and compare the chosen split (`layer split auto: K=…`, `N of its M profiled pairs`) with the
previous one. If the split changed to match the new profile, the adaptation mechanism works and the concern about
the pinned variant is immaterial for auto; if it did not, auto is choosing a layout from stale data, which is a
separate defect.

#### 8.5.1 Why the risk is small on this model: the measured data-tier hierarchy

A survey of the engine log lines (all requests of the baseline sweep and the correctness check):

- **The engine's own RAM tier is not used:** every line reads `expert tiers: … RAM 0 blobs, files 0 blobs`. The
  intermediate tier in practice is the **OS page cache** through `--mmap-experts`; the indicator of real disk
  access is `major faults` in the `file tier I/O` line.
- **The model almost entirely fits in the page cache:** the model files are 52 GiB + 27 GiB = 79 GiB
  (`--mmap-experts`), the machine has 91 GiB of RAM, of which 84 GiB is in buffers/cache and 81 GiB available
  (`system-inventory.md`). So file-tier accesses are mostly served from RAM.
- **Real NVMe accesses happened, but small ones that did not break speed.** The worst observed case: `the OS read
  1026.1 MB from storage (3118 major faults) for 16995.3 MB of expert reads`
  (`raw/profiles/2026-10-17T004452Z/engine-log-delta.txt`): that same request gave **848.2 tok/s prompt and
  103.1 tok/s decode** - 1 GiB of real disk read caused no collapse (comparing with the warmed baseline of 868.7
  at 3971 tokens is not valid, since this was the first request after a start, but the upper bound of the effect
  is visible: there is no order-of-magnitude drop).
- Other observed values: `17.6 MB (252 major faults)`, `493.5 MB (900 major faults)`, `2.1 MB (2 major faults)`
  against gigabytes of expert accesses; most requests show `(0 major faults)`.

Conclusion about the scenario: for an unprofiled topic what actually happens is **additional reading (from RAM,
rarely from NVMe)**, i.e. a uniform slowdown of prefill/decode, not a failure and not a change in answer
quality. On this model it is unlikely precisely because almost all experts are resident (23901 of 24576 pairs,
97 %) and the remainder is served from the page cache.

#### 8.5.2 Why auto stays the main line

A second, stronger argument against pinning: **the project is young, and most users will be in automatic mode. If
everything here is pinned by hand, we lose the ability to measure how well the engine's own mechanisms work** -
the layer split, the expert cache, the prefill path choice. A pinned config would turn version and patch
comparisons into a comparison of two hand-tuned layouts.

So the protocol is:
1. the working configuration and all baselines are **on auto**;
2. pinned values are only a **diagnostic probe** (as in §8.2: it was needed to prove the cause of the delta), and after use the configuration is restored;
3. the requirements and expected effects of any placement change are phrased as **"improve auto's decisions"**
   (in particular, add prefill cost and per-card residency to the split search criterion), not as "the user
   should set N, M, K".

## 9. What this profile does not cover

| scenario | status | reason |
|---|---|---|
| one GPU, a minimal configuration | needs a service stop | needs an engine restart with a different device set; also absent from the PR |
| four GPUs, the current multi-GPU mode | covered | the main run |
| prefill of short and medium contexts | covered | 3 971 / 15 667 / 31 307 / 62 519 tokens |
| decode at a fixed token count | covered | 256 tokens at every prompt point + 1024 tokens gen-only |
| long context from existing results | reused | 125 011 tokens re-measured (with valid decode); 185 531 / 200 015 / 260 331 from the PR |
| a layer split with other distributions | needs a service stop | the split is chosen at start; the actual current one is recorded (§6) |
| speculative decoding | partial | the build runs with `--spec 4` + MTP; acceptance measured at a fixed length (69.4-81.5 %). An A/B "spec off" needs a stop |
| inter-card transfers: count and sizes | not possible with this driver | no NVLink counters (DCGM 4.x without the profiling module, NVML Sampling & Reporting absent); the indirect method is §4.1 |
| duration of synchronizations/waits | indirect only | `server_overhead_s`, `tail_s`, the queue at `concurrency/serving = 1`; the engine exposes no internal synchronization counters |

## 10. What to treat as the baseline going forward

1. **For comparing future changes on this machine, use the §3 table (and `baseline-profile.csv`), not PR #2525's
   numbers** - at the same `auto` split. The PR-to-now difference (-1.5…-5.7 %) is real and explained by a
   different layer split (§8), so the two lines must not be mixed into one delta. **A mandatory part of any
   measurement protocol:** record `K=` and `expert cache N slots` from the new engine session's log.
2. **Treat the decode column as valid only with a fixed output length** (the method: banning EOS `248044`/`248046`
   through `logit_bias`, checking `gen_n == max_tokens` and `finish_reason == "length"`). Without that the column
   carries no information on this machine (natural stopping at about 95-110 tokens).
3. The effect-detection threshold: **>= 1 %** on prompt tok/s (a noise band of 0.2-0.3 %, the worst case 1.3 % at
   4k); for decode, at least 3 repeats and a median, with single-run spread up to 9 %.
4. The tool for repetition: `scripts/baseline-profile.py` (a ready call `python3 scripts/baseline-profile.py` -
   the same lengths, 3 repeats, a fixed output length, dmon and `nvidia-smi` samples, engine log growth parsing).
   Points and repeats are set by `--lengths`, `--reps`, `--max-tokens`, `--gen-max`, `--warmup-lengths`,
   `--no-eos-ban` (for a "as in the PR" comparison).

## 11. Limits of this report

- The absence of CUDA errors is not proof of numeric correctness of the SM70 paths (that is `correctness-report.md`).
- The §3 numbers describe synthetic prompts with a marker; they transfer to real workloads (different prefix reuse, different drafts) only as an order of magnitude.
- The PCIe bytes in §4.1 are a lower bound because of the 1 s step and skipped seconds; NVLink traffic is not included.
- The cause of the SM clock reduction in 97 samples is not established (the `clocks_throttle_reasons.active` field did not come back in that query).
- The delta against the PR was explained by a hypothesis (§6) and confirmed by one targeted experiment with a service stop (§8).
