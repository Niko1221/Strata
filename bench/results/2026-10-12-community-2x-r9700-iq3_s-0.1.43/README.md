# Community benchmark on 2x AMD Radeon AI PRO R9700 32 GB (gfx1201), Ryzen 9 5900X, 125 GiB RAM: v0.1.43 on a two-card layer split, with and without the decode switches

Measured on 2026-10-13 by justincinmd, all five runs in one session (B, E, C, D, B2 in that order, a fresh server for
each run). Strata v0.1.43 (run E: v0.1.43 with the nine decode switches of the companion PR, see below), Flash-Next IQ3_S,
`--layer-split 26` across two R9700s with every expert resident in VRAM, 262,144-token context.

The question for upstream: v0.1.43's gfx12 table (`src/core/arch_defaults.cpp`) turns on only the prompt half of the
gfx1151 switches, because "the decode half measured slower" on one R9700 (#1478). On this two-card split the decode
switches are a gain, and on four greedy prompts they gave identical output: the same reasoning text, answer text and
token counts as stock in all three runs (see "Greedy output hashes"):

- **B** stock v0.1.43, its gfx12 defaults only.
- **C** v0.1.43 with E's nine decode switches and `STRATA_Q8_PACKED=1` (inert on this pack, see E) set by hand,
  **plus `STRATA_HIP_WMMA=1` and `STRATA_VERIFY_QDEDUP=1`**.
  C is therefore not "upstream's exact decode set" alone. WMMA is not bitwise (the engine says so at start), and C's
  greedy output differs from B's on three of four prompts.
- **D** = C + `--batch 2` (two pipelined batch slots).
- **E** v0.1.43 with nine decode switches of the gfx1151 table on top of its five gfx12 prompt switches (14 in
  all): `Q6_PACKED MMVF_ROWS ATTN_LANECELL EXPERT_V2 TSUM LFUSE GDN_SPLIT QFUSE PLE_BATCH`. Upstream's table calls
  them "exact" (meant to leave the output unchanged); what this report checked is under "Greedy output hashes".
  No WMMA, no QDEDUP. E has the same 14 switches as v0.1.43 + `STRATA_GFX12_SPLIT_DECODE=1` of the companion PR,
  offered separately (branch
  [`pr/gfx12-split-decode-optin`](https://github.com/justincinmd/Strata/tree/pr/gfx12-split-decode-optin)).
  This is configuration equivalence: the opt-in revision of that PR was not measured here.
  In that PR the nine are opt-in: by default the engine behaves as v0.1.43, and `STRATA_GFX12_SPLIT_DECODE=1` turns
  them on for a layer split over two or more gfx12 GPUs. E also has the same switches as v0.1.43 with the nine set by
  hand, which any user can do today.
  How E was measured: with an earlier revision of that branch, commit `b52a14a0` (three commits on v0.1.43), which
  turned the same nine on by default on such a split (hence "on by default" in E's start log). That commit is no
  longer on a published branch, so its source is included as `source-E-b52a14a0.patch` (`git format-patch` of
  v0.1.43..b52a14a0; `git am` or `git apply` on v0.1.43 gives tree `15de36cd`). That revision set the same 14
  switches at start, and E's start line names exactly these 14. Its other differences from the current opt-in PR
  only moved the point where the table is applied (after the split is resolved) and changed the handling of
  `STRATA_Q8_PACKED`, which E did not set (see "Hardware and software"). The patch is kept verbatim: its commit messages and docs text carry that draft's
  earlier figures and claims ("+6.5%", "output unchanged", all twelve sampled answers identical); the figures in this
  report (+5.8% / +4.9%, 11 of 12 sampled runs matching) replace them. A later session on the same machine (2026-10-14) found that seeded sampled output is not always repeatable with stock v0.1.43 itself: one seeded request (prose, seed 1001) gave one answer in two stock runs and a different one in a third (with one-token decode windows, which do not change greedy output), and another build compared with stock gave both answers too, so E's one differing sampled run (11 of 12) is consistent with that run-to-run variation and neither proves nor disproves bit-exactness. The same 2026-10-14 session also saw stock's greedy prose answer change between sessions (the other three greedy prompts kept their values), which matches upstream's documented note that about one HIP start in ten gives a different greedy answer (docs/AMD_HIP.md, "Known"); within each session here, stock and the switches agreed on all greedy requests. `STRATA_Q8_PACKED` is not one of the nine
  because it is inert on this pack: the only Q8_0 dense matrix, `blk.47.ffn_down_shexp`, has 640 inputs, below the
  packed kernel's minimum width, so nothing is packed (no "packed ... Q8_0" line in any start log here, also not in C
  and D, which set it by hand).
- **B2** stock v0.1.43 again, after E, C and D: a same-session control for E.

The gfx1151 decode half has two more entries that E does not set: `STRATA_SH_STREAM` (no run set it: 0.1.40 turned the
shared-expert stream off on HIP, #826) and `STRATA_Q8_PACKED` (inert here, above). "The nine switches" below means
the decode half without them.

Main limitation: one machine, one quant. Decode figures are medians of three runs per prompt; prompt reads are one
fresh request per size. Timings are client-side (see Method), cross-checked against the engine's own timing lines.

## Hardware and software

- GPUs: 2x AMD Radeon AI PRO R9700 32 GB (Navi 48, gfx1201, 31.86 GiB as the engine reports it). GPU 0 is
  headless and runs layers 0-25. GPU 1 drives the desktop (`--vram-reserve-later-mib 1800` leaves it room),
  runs layers 26-47, the head and the MTP draft layer. Power cap 300 W each (the default; not changed). Stock clocks.
- PCIe: MSI MEG X570 Unify, the two CPU slots split x8/x8, so each card links PCIe 4.0 x8 to the CPU (lspci shows Gen5
  x16 for the card's own internal bridge). The engine's host-to-device probe read 14.5 GB/s on each card.
- CPU: AMD Ryzen 9 5900X, 12 cores / 24 threads, AVX2, no AVX-512. RAM: DDR4, MemTotal 125.5 GiB.
- Storage: the IQ3_S shard 1 and the pack on a SATA SSD (~560 MB/s); shard 2 (the n-gram table) on a PCIe 4.0 NVMe.
  The expert arena loaded at 0.89-1.71 GiB/s across the five runs (engine start logs), mostly from the page cache.
- OS: Ubuntu 24.04.5 LTS, kernel 6.8.0-146-generic, amdgpu-dkms 6.16.13 (1:6.16.13.30300400), ROCm 7.2.4 (HIP 7.2.53211).
- Strata: tag v0.1.43 (`7695859a`), source build: Release, Ninja, `-DSTRATA_ENABLE_HIP=ON -DCMAKE_PREFIX_PATH=/opt/rocm
  -DCMAKE_HIP_ARCHITECTURES=gfx1201`, ggml from llama.cpp `3cf03257` (the commit `setup.py` pins). Note: without
  `-DCMAKE_HIP_ARCHITECTURES=gfx1201` this build produced gfx1100 code and the engine refused the card.
  Server: v0.1.43's `serve/server.py`.
  Run E: the same build options on v0.1.43 plus an earlier revision of the companion PR branch
  ([`pr/gfx12-split-decode-optin`](https://github.com/justincinmd/Strata/tree/pr/gfx12-split-decode-optin)),
  commit `b52a14a0`, included as `source-E-b52a14a0.patch`.
  That revision changed `src/core/arch_defaults.cpp` (with its header and test: the switch table),
  `src/program/generate.cpp` (where the table is applied: after the split is resolved), `src/core/native_dense.cpp`
  (with its header: `STRATA_Q8_PACKED` split pricing and an out-of-memory fallback, active only with
  `STRATA_Q8_PACKED=1`, which E did not set) and docs. E was served by that tree's `serve/server.py`, which
  that revision did not touch.
- Background: a GNOME desktop on GPU 1; no other GPU work. Each run waited for a load average below 3 (it was 2.76-2.97
  when each benchmark started).

## Model and configuration

- Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, IQ3_S, two shards (54,817,524,224 + 28,800,138,432 bytes; the
  same sizes as in the 2026-10-09 RTX 5080 report; revision not recorded, downloaded 2026-10-03). Routed experts 46.84 GiB.
- Pack: `tools/iq_pack.py`, no conversions (`conversions.json` has no tensors), built 2026-10-03 by an earlier setup;
  v0.1.43 loaded it unchanged. Expert profile `data/expert-profile.bin` and draft vocabulary `data/draft_vocab_en.bin`
  are byte-identical to v0.1.43's (sha256 `8f59b4aa…` and `36915152…`). MTP draft layer (`mtp/rt`) from an earlier setup run (revision not recorded).
- Vision encoder on (CPU, 12 threads); no image requests in this benchmark.
- Context 262,144; `--kv int8 --kv-resident 65536` (the engine: "65536 of 262144 cells per QSA layer in VRAM, the K/V
  in 1.55 GiB of pinned RAM"); `--expert-cache auto`: GPU 0 13,312 slots (24.19 GiB), GPU 1 11,264 slots (22.64 GiB),
  "100% of the experts resident" (24,576 of 24,576); `--prefill auto` (the engine picked 16,384-token chunks);
  `--trim-stage-weights`; conversation cache 16 GiB / 6 slots.
- MTP `--spec 4 --spec-min-p 0.5` with the English draft vocabulary. No calibration; experimental speed projection off.
- `STRATA_HIPBLASLT_TUNING` set to v0.1.43's own `tools/hip/gfx1201-hipblaslt-100202.txt` in every run (the engine:
  "hipBLASLt tuning enabled (24 rows, gfx1201, version 100202)").
- Thinking on, `reasoning_budget_tokens` 8192. With `max_tokens` 1024 the server capped thinking at 512 tokens ("to
  leave room for the answer") and then ran a wrap-up pass, so some requests are two engine passes.

```text
strata --serve --pack <pack> --native <IQ3_S shard 1> --ple-gguf <IQ3_S shard 2> --expert-profile data/expert-profile.bin
  --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp <mtp/rt> --max-context 262144 --kv int8
  --kv-resident 65536 --conversation-cache-mib 16384 --conversation-cache-slots 6 --vision --vram-reserve-mib 700
  --vram-reserve-later-mib 1800 --mtp-draft-vocab data/draft_vocab_en.bin --trim-stage-weights --layer-split 26
  [D only: --batch 2]
config: "gpu": [0, 1], "layer_split": "26"
env, all runs: STRATA_HIPBLASLT_TUNING=tools/hip/gfx1201-hipblaslt-100202.txt
env, C and D:  STRATA_Q8_PACKED=1 STRATA_Q6_PACKED=1 STRATA_MMVF_ROWS=1 STRATA_ATTN_LANECELL=1 STRATA_EXPERT_V2=1
               STRATA_TSUM=1 STRATA_LFUSE=1 STRATA_GDN_SPLIT=1 STRATA_QFUSE=1 STRATA_PLE_BATCH=1
               STRATA_VERIFY_QDEDUP=1 STRATA_HIP_WMMA=1
               (+ STRATA_GDN_HEAD=1 GDN_PP=2 GDN_CONVL2=1 GDN_NOY=1, the gfx12 defaults set by hand, same values)
env, B2:       as B
env, E:        as B; the engine (v0.1.43 + b52a14a0, an earlier revision of the companion PR) set the five prompt and nine
               decode switches itself. Same switches: v0.1.43 + STRATA_GFX12_SPLIT_DECODE=1 with the current PR,
               or v0.1.43 with the nine set by hand
served by v0.1.43's serve/server.py on 127.0.0.1
```

The full configs with paths replaced: `config-B-stock.json`, `config-C-env.json`, `config-D-batch2.json`,
`config-E-exact.json`, `config-B2-stock.json`. The engine's start lines: `engine-start-*.txt` (B and B2 log "5 exact
speed switches on by default: GDN_HEAD GDN_PP GDN_CONVL2 GDN_NOY CVEC_FUSE"; C and D log only CVEC_FUSE, because the
other four were set by hand; E logs "gfx12 (RDNA4), layer-split table: 14 exact speed switches on by default" and names
all 14; "on by default" comes from the earlier, default-on revision of the PR that E ran). E kept the stock expert
placement: GPU 0 13,312 slots, GPU 1 11,264, 100% resident. Its GPU 1 has 25.58 GiB of room for experts against
stock's 26.07 GiB: `STRATA_Q6_PACKED` packs the output head (+497.3 MiB) and the MTP draft
head (+81.2 MiB) on the last stage. C shows the same figures; D has 24.18 GiB of room (its batch slot sessions) and
is still 100% resident.

## Method

- `tune_bench.py --runs 3` (included; it imports the prompts from `decode_bench.py` and reads `longctx.txt.gz`, sha256 `85b38b0c…` unzipped: Strata's own `docs/*.md` concatenated in name order, 434,992 characters;
  tune_bench reads at most the first 230,400). The scripts in this folder produced the data; after the runs they got
  stricter failure handling only (see the end of Scripts). They read the .gz directly: `python3 tune_bench.py --url http://127.0.0.1:PORT/v1 --model MODEL --name X --runs 3 --out results-X.json`.
- Each arm is a fresh server on the same model files (page cache warm). Startup is timed separately (server start to its
  `ready:` line). One 8-token warm-up request, not counted.
- Decode: four prompts (prose essay, Python LRU cache with tests, a rename edit of a ~280-token module, a train-timetable
  reasoning question), `max_tokens` 1024, three runs each, greedy (temperature 0) and sampled (temperature 1.0, top-p
  0.95, top-k 20, seeds 1000/1001/1002). Decode tok/s = (completion tokens - 1) / (time of the last streamed token - time
  of the first), measured by the client; reasoning tokens count. Headline figures are the **geometric mean of the four
  per-prompt medians**. Each run also records separate hashes of the reasoning text and the answer text, the
  completion token count, the finish reason, and one combined hash over all four (`"format": 2` in the JSON).
- Cold prompt reads: one request each of about 4.6K, 19K and 69K tokens, each a different-length slice of `longctx.txt`
  behind a fresh UUID (the engine reported 0 reused tokens), 16 new tokens. Prompt tok/s = prompt tokens / client TTFT.
  The engine's own figure is given beside it.
- Warm turns: a 36K-token conversation, then three follow-ups that each add about 3K tokens of "tool output" and resend
  the exact earlier answers. The engine reused 35.9-41.9K tokens and read 2.8-3.1K new tokens per turn. Reported: median
  TTFT and decode of the three follow-ups.
- Concurrency (D only): `conc_bench.py --conc 1,2,4 --rounds 2 --max-tokens 512 --cache warm`. Every level sends the
  same fixed request set (the four prompts, seeds 100-103, temperature 0.7, 512 tokens) with at most C requests in
  flight, so C=1 runs the set one request after another. One unmeasured priming pass comes first, so the measured
  prompts are repeats (the server reported a median of 95.5 cached prompt tokens per request). Aggregate = all
  generated tokens / wall time of the round; the table gives the median of two rounds.
- `decode_bench.py` (the stand-alone decode test) was not run in this session; no `decode-*.json` files.
- Not recorded: power draw, clocks, VRAM peaks during the run, RAM peaks (the server ran in a cgroup with MemoryMax 100 GiB).

## Results

### Summary

<!-- table:summary -->
| Run | Greedy tok/s (geo-mean) | Temp 1.0 tok/s (geo-mean) | Prompt tok/s at 4.6K / 19K / 69K | Warm turn TTFT / decode | Ready | Failed requests |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| B stock v0.1.43 | 108.9 | 101.5 | 1,367 / 1,584 / 2,483 | 2.9 s / 93.5 tok/s | 75 s | 0 |
| C + decode switches + WMMA + QDEDUP | 121.1 | 109.1 | 1,759 / 2,188 / 3,628 | 2.2 s / 94.3 tok/s | 69 s | 0 |
| D = C + `--batch 2` | 117.1 | 107.8 | 1,750 / 2,213 / 3,502 | 2.1 s / 67.1 tok/s | 75 s | 0 |
| E v0.1.43 + the nine decode switches (`STRATA_GFX12_SPLIT_DECODE=1`) | 115.0 | 106.2 | 1,515 / 1,756 / 2,761 | 2.6 s / 102.2 tok/s | 84 s | 0 |
| B2 stock v0.1.43 again (after E) | 108.5 | 101.0 | 1,368 / 1,552 / 2,441 | 2.8 s / 85.7 tok/s | 78 s | 0 |
<!-- /table:summary -->

B to C: greedy +11.2%, sampled +7.5%, prompt reads +29% / +38% / +46%, warm-turn TTFT 2.9 s to 2.2 s. Most of C's
prompt gain should come from WMMA (the engine's own hint at start: "STRATA_HIP_WMMA=1 reads prompts about 30% faster on
this card with --kv int8"). An earlier side check (first version of the scripts, E's first build) points the same
way: E + `STRATA_HIP_WMMA=1` alone read prompts at 1,759 / 2,376 / 3,556 tok/s, close to C (per-run data not included).

E against the two stock sessions of the same session, one before E (B) and one after (B2). The two stock sessions
agree within 0.5% on decode (B2 against B: greedy -0.4%, sampled -0.5%) and within 2% on prompt reads.

| E against | Greedy | Temp 1.0 | Prompt reads 4.6K / 19K / 69K |
| --- | ---: | ---: | --- |
| B, before E (108.9 / 101.5) | +5.6% | +4.6% | +11% / +11% / +11% |
| B2, after E (108.5 / 101.0) | +6.0% | +5.1% | +11% / +13% / +13% |
| mean of B and B2 (108.7 / 101.3) | +5.8% | +4.9% | +11% / +12% / +12% |

E's warm-turn decode (102.2 tok/s, the median of 94 / 104 / 102) is above both stock sessions (93.5 and 85.7). Warm-turn
decode swings widely from turn to turn here (B2: 86 / 61 / 94; D: 61 / 104 / 67), so we do not read a change into it.
A first version of this report (same arms, the first version of the scripts, 2026-10-12 and 2026-10-13; not
included) gave E +6.5% greedy and +6.1% sampled against the mean of two same-day stock sessions. Those scripts did not
compare the reasoning text and did not validate the streams, and part of that baseline's per-run data was not kept, so
those figures are replaced by the ones above.

### Decode per prompt

Median (lowest-highest) of three runs; generated tokens per run in brackets. Prompt tokens: prose 72, code 81, edit
282, reasoning 108.

<!-- table:decode -->
| Prompt | Mode | B stock | C env | D batch 2 | E nine switches | B2 stock |
| --- | --- | --- | --- | --- | --- | --- |
| prose | greedy | 101.4 (101.0-101.6) [906 x3] | 119.1 (118.3-119.2) [901 x3] | 119.2 (117.4-119.3) [901 x3] | 106.7 (106.2-106.8) [906 x3] | 100.4 (97.2-101.3) [906 x3] |
| code | greedy | 107.2 (106.7-107.4) [1024 x3] | 118.7 (113.6-118.9) [1024 x3] | 118.5 (113.9-118.8) [1024 x3] | 113.0 (108.0-113.1) [1024 x3] | 106.9 (106.4-107.2) [1024 x3] |
| edit | greedy | 108.4 (108.3-108.4) [342 x3] | 114.9 (113.3-115.5) [342 x3] | 115.1 (114.9-115.5) [342 x3] | 114.3 (100.4-114.4) [342 x3] | 108.2 (98.1-108.2) [342 x3] |
| reasoning | greedy | 119.4 (119.3-119.8) [327 x3] | 132.4 (131.0-132.7) [303 x3] | 115.6 (114.9-133.1) [303 x3] | 126.7 (125.9-127.1) [327 x3] | 119.2 (119.1-119.6) [327 x3] |
| prose | temp 1.0 | 93.5 (90.3-97.7) [1016, 1024, 1017] | 104.7 (100.4-105.0) [1024, 1024, 964] | 98.9 (97.5-104.8) [1024, 974, 964] | 97.7 (94.9-102.6) [1016, 1024, 1017] | 93.5 (87.2-97.5) [1016, 1024, 1017] |
| code | temp 1.0 | 92.7 (90.0-95.9) [1024 x3] | 104.7 (96.4-119.0) [1024 x3] | 106.2 (96.5-119.6) [1024 x3] | 95.4 (89.6-101.3) [1024 x3] | 92.6 (88.3-95.8) [1024 x3] |
| edit | temp 1.0 | 105.2 (102.6-109.0) [417, 321, 371] | 105.7 (102.0-125.8) [473, 351, 303] | 105.3 (101.8-125.8) [473, 351, 303] | 110.4 (108.5-115.4) [417, 321, 371] | 102.6 (101.5-105.1) [417, 321, 371] |
| reasoning | temp 1.0 | 116.5 (115.1-117.9) [307, 266, 261] | 122.4 (119.0-127.5) [293, 324, 259] | 122.1 (119.2-127.5) [293, 324, 259] | 123.4 (122.1-124.7) [307, 266, 261] | 117.1 (114.9-118.2) [307, 266, 261] |
<!-- /table:decode -->

The client figures match the engine's timing lines for single-pass requests (B code greedy: client 107.2 / 107.4 /
106.7, engine 107.1 / 107.3 / 106.6 tok/s). C's and D's sampled runs differ in length and text from B's, so their
sampled speeds compare less directly than the greedy ones. E's sampled runs have the same token counts as B's; 11 of
the 12 also have the same text (see the hash section).

D's greedy reasoning prompt ran at 115.6 and 114.9 tok/s in two of its three runs (133.1 in the third; C: 131.0-132.7),
with the same output as C. The engine's own timing lines agree (113.7 / 114.4 / 132.5). We did not find the cause. It
puts D's greedy geo-mean 3.3% below C's; on the other three greedy prompts D and C are within 0.3%.

### Prompt reads (one fresh request each)

<!-- table:cold -->
| Run | Actual prompt tokens | Reused | TTFT (client) | Prompt tok/s, client (engine) |
| --- | --- | ---: | --- | --- |
| B stock | 4,646 / 19,003 / 68,924 | 0 | 3.40 / 11.99 / 27.75 s | 1,367 (1,380) / 1,584 (1,590) / 2,483 (2,493) |
| C env | 4,647 / 19,004 / 68,929 | 0 | 2.64 / 8.68 / 19.00 s | 1,759 (1,781) / 2,188 (2,199) / 3,628 (3,647) |
| D batch 2 | 4,643 / 19,000 / 68,926 | 0 | 2.65 / 8.59 / 19.68 s | 1,750 (1,771) / 2,213 (2,224) / 3,502 (3,520) |
| E nine switches | 4,646 / 19,003 / 68,927 | 0 | 3.07 / 10.82 / 24.96 s | 1,515 (1,530) / 1,756 (1,763) / 2,761 (2,773) |
| B2 stock | 4,644 / 19,002 / 68,927 | 0 | 3.39 / 12.24 / 28.23 s | 1,368 (1,381) / 1,552 (1,591) / 2,441 (2,450) |
<!-- /table:cold -->

Our earlier noise estimate on this machine: about 1% for decode and the 16K / 64K reads, about 7% for the 4K read.

### Greedy output hashes (reasoning text, answer text, token count and finish reason)

<!-- table:hashes -->
| Prompt | B stock | C env | D batch 2 | E nine switches | B2 stock |
| --- | --- | --- | --- | --- | --- |
| prose | `da08875e22` x3 | `a2abefd8e1` x3 | `a2abefd8e1` x3 | `da08875e22` x3 | `da08875e22` x3 |
| code | `a5dcdf993f` x3 | `c97c19e7aa` x3 | `c97c19e7aa` x3 | `a5dcdf993f` x3 | `a5dcdf993f` x3 |
| edit | `74eb807af0` x3 | `74eb807af0` x3 | `74eb807af0` x3 | `74eb807af0` x3 | `74eb807af0` x3 |
| reasoning | `34ff6e23f4` x3 | `f2a54462a0` x3 | `f2a54462a0` x3 | `34ff6e23f4` x3 | `34ff6e23f4` x3 |
<!-- /table:hashes -->

- The table shows the first 10 hex digits of the combined hash in `results-*.json` (`hash_combined`: one hash over the
  reasoning text, the answer text, the completion token count and the finish reason; the JSON also has the separate
  reasoning and answer hashes). Strata's OpenAI endpoint returns no token ids or logprobs, so a match means the same
  text and the same number of generated tokens, not identical internal arithmetic.
- Each arm repeats its own greedy hashes in all three runs.
- C differs from B on prose, code and reasoning (reasoning and answer hashes both differ; edit matches). We attribute
  this to `STRATA_HIP_WMMA`, which the engine itself labels "the output bits change (last-place rounding)"; C did not
  separate WMMA from QDEDUP. An earlier side check (first version of the scripts, answer hash only) did: E +
  `STRATA_HIP_WMMA=1` alone gave C's four answer hashes, and E + `STRATA_VERIFY_QDEDUP=1` alone gave B's.
- C and D match on all four prompts: `--batch 2` changed no solo output.
- E matches B and B2 on all four prompts in all three runs: with the nine decode switches the greedy reasoning text,
  answer text and token counts are identical to stock v0.1.43 on this split.
- B2 repeats B on all four greedy prompts and on all twelve sampled runs (temperature 1.0, seeds 1000-1002).
- At temperature 1.0, E matches B on 11 of the 12 sampled runs (combined hash). The exception is the code prompt with
  seed 1002: the same reasoning text, the same token count (1024, finish `length`), but a different answer text
  (answer hash `29d9749188` against stock's `7af92f0820`). So E's sampled output is not identical to stock in every
  case. We did not look for the cause. (The first version of the scripts compared answer hashes only and saw 12 of 12.)

### Two conversations at once (D, `--batch 2`)

<!-- table:conc -->
| Concurrent requests | Aggregate tok/s (median of rounds) | Per stream tok/s | TTFT (median) | Tokens per round | Cached prompt tokens (median) | Failed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 99.5 (98.8, 100.1) | 104.4 | 0.23 s | 1,821 | 95.5 | 0 |
| 2 | 103.3 (103.8, 102.7) | 53.9 | 0.12 s | 1,821 | 95.5 | 0 |
| 4 | 106.9 (105.9, 107.9) | 53.5 | 2.66 s | 1,821 | 95.5 | 0 |
<!-- /table:conc -->

With the same request set at every level, two requests at a time gave 103.3 tok/s in total against 99.5 for one
request at a time (+3.8%), and four gave 106.9 (+7.4%). Per stream, each of two concurrent requests ran at about half the
solo rate (53.9 against 104.4 tok/s). The engine ran the batch slots as pipelined groups (`--batch-groups auto: 2
groups of 1 slot`) without `--batch-mtp`, so (as docs/BATCHING.md describes) they ran no MTP drafts; a request alone
keeps the solo path with its drafts. With `--batch 2`, solo speed was greedy 117.1 against C's 121.1 (-3.3%, almost all
from the reasoning prompt, see above) and sampled 107.8 against 109.1 (-1.2%). The batched answers were not checked for
correctness (the #1139 QFUSE batch fix, `0a22c467`, is in v0.1.43). `conc-D-batch2.json` keeps every request (TTFT,
tokens, cached prompt tokens, finish reason) and every round's wall time. The first version of the script sent a
different prompt mix at each level and showed 94.9 against 82.3 tok/s; that figure is replaced by the table above.

### Startup

Server start to `ready:` with the model files in the page cache: B 75 s, E 84 s, C 69 s, D 75 s, B2 78 s. Earlier stock
v0.1.43 starts on this machine ranged from 54 to 120 s, so these figures do not rank the arms. For comparison, stock
v0.1.40.1 on 2026-10-06 took about 95 s to load the same split.

### Earlier measurement: `--pipeline-windows 2` on this split (v0.1.40.1, 2026-10-06)

Stock v0.1.40.1, the same model and `--layer-split 26` with all experts resident, short prompts of `decode_bench.py`,
three runs each (per-run data was not kept). tok/s with `--pipeline-windows 2` / without:

| Mode | prose | code | edit | reasoning |
| --- | --- | --- | --- | --- |
| greedy | 102 / 106 | 104 / 105 | 102 / 104 | 111 / 114 |
| temp 1.0 | 96 / 102 | 96 / 104 | 92 / 104 | 106 / 108 |

No gain on a two-stage split with every expert resident on two equal cards; it was slightly slower in every cell. This
agrees with docs/MULTI_GPU.md (2x R9700: code decode -3%) and with v0.1.43 keeping it opt-in for two stages. We did not
repeat it on v0.1.43.

Per-run data (the file names keep the run label "E-exact", after upstream's name for the switches): `results-B-stock.json`, `results-C-env.json`, `results-D-batch2.json`, `results-E-exact.json`, `results-B2-stock.json` (the
tune_bench output plus each request's generated tokens and finish reason from the server log); the engine's timing line
for every pass: `engine-requests-*.txt`. Run E's engine source: `source-E-b52a14a0.patch` (see the E description at the
top).

## Scripts

Included: `tune_bench.py` (decode, cold prompt reads, warm turns), `decode_bench.py` (the four prompts, also a
stand-alone decode test), `conc_bench.py` (concurrent requests), `bench_stream.py` (the streaming client the three
share) and `test_bench_stream.py` (an offline self-test: `python3 tune_bench.py --selftest`, no GPU or server needed).
Standard library only. Each takes `--url http://127.0.0.1:PORT/v1 --model MODEL`; `--out` (or `--json` for
decode_bench) names the JSON file.

The scripts were hardened after an external review, and every figure above was measured with the hardened version.
The first version's results are not included. What changed from the first version:

- Stream validation. The first version ignored SSE `error` events, did not check `finish_reason`, accepted a stream
  that ended without `[DONE]`, and counted chunks when the server sent no `usage`. Now a request counts only if the
  stream has no error event, ends with `finish_reason` `stop` or `length` and `[DONE]`, and reports
  `usage.completion_tokens`. HTTP and network errors become records too. Failed requests are listed under `failures`
  in the JSON and are left out of every median and summary (a summary whose inputs all failed is `null`); the exit
  status is 1. In the runs above no request failed: all 32 tune_bench requests of each arm and D's 24 measured
  concurrency requests (plus 4 priming requests) passed these checks.
- Output hashes. The first version hashed `sha1(answer text + completion token count)` and dropped the reasoning
  text. The new JSON (`"format": 2`) keeps, for every request, separate hashes of the reasoning text and the answer
  text, the completion token count, the finish reason, and one combined hash over all four. Strata's OpenAI endpoint
  returns no token ids or logprobs, so token ids are not hashed (`"token_ids": null`). Matching hashes still show
  matching text, not identical internal arithmetic.
- `conc_bench.py`: every level now sends the same fixed request set (the four prompts, fixed seeds 100-103,
  temperature 0.7, the same `max_tokens`), with at most C requests in flight, so C=1 runs the set one request after
  another. `--cache warm` (default) keeps the prompt texts identical and sends one unmeasured priming pass first;
  `--cache cold` gives every request a unique fixed-length nonce instead. The JSON keeps every request (prompt id,
  round, level, slot, start and end time, TTFT, tokens, cached prompt tokens, finish reason, status) and every round's
  wall time, and the summary is computed from those records.
- The long-context fixture is read only for the cold and warm tests, straight from `longctx.txt.gz`; `--help`,
  `--quick`, `decode_bench.py` and `conc_bench.py` need no unpacked file.
- Every request record also keeps the server's cached prompt tokens and the engine's own `timings` speed, when sent.

A second review, after these runs, found failure cases that the scripts still handled badly. The scripts in this folder
include the fixes; they change only how failures are handled. In the runs above no request failed, the priming pass
had 4 of 4 requests ok, and no level was repeated, so the fixes would not change any figure. The fixes:

- Field types. `bench_stream.py` now checks the types of the fields it reads: `usage`, `timings` and
  `prompt_tokens_details` must be objects, token counts non-negative integers, `choices` a list of objects, each
  `delta` an object, content, reasoning and finish reason strings, timing speeds numbers. An event that breaks this
  gives a `bad_event` record. Before, such an event raised an exception, which could end a run before its JSON was
  written. `stream()` now never raises: any other exception, while building the request (for example an invalid
  `--url`) or while reading the stream, gives a `client_error` record.
- Priming. In `conc_bench.py --cache warm`, a failed priming request now stops the run before measurement: the JSON is
  written with the priming records under `failures`, no measured level runs, and the exit status is 1. Priming is not
  retried. Before, a priming failure was kept in the records but not listed, and measurement went on.
- Every concurrent request gets a record, also if the client code raises in a worker thread (`client_error`).
- `--conc` must list distinct positive integers; `--rounds`, `--repeat` and `--max-tokens` must be positive, and so
  must `--runs` of `tune_bench.py` and `decode_bench.py` (and decode_bench's `--max-tokens`); a bad count is a usage
  error (exit status 2) before any request is sent. Before, `--runs 0` gave an empty "successful" run. Every
  round (the priming pass included) gets a unique `round_id`, and requests join their round by it. Before, `--conc 1,1`
  was accepted and both runs of a level shared round numbers, which counted each round's tokens twice.
- `conc_bench.py` now writes `"format": 3` (adds `round_id`, a `phase` per failure and `stopped`). `conc-D-batch2.json`
  is the format 2 file from the run; `summarize()` still reads it and gives the same summary (the self-test checks this).

## Correctness and limitations

- Greedy output is repeatable per arm (identical combined hashes in all three runs). B to C changes it (WMMA); B to E
  keeps the reasoning text, answer text and token counts on all four greedy prompts. At temperature 1.0, 11 of E's 12
  runs match stock; one answer text differs (same reasoning and token count).
- The answers themselves were not graded. No needle or recall test was run. No image requests, although the vision
  encoder was loaded.
- One machine, one quant (IQ3_S), one split (26). Prompt reads are single requests. Sampled decode depends on the text
  each arm happened to generate.
- Not tested: `--batch-mtp`, `--pipeline-windows` on v0.1.43, a single R9700, other contexts, Windows. The WMMA and
  QDEDUP side checks on top of E were not repeated with the hardened scripts.
