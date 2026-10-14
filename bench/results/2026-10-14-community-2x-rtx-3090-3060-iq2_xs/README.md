# Community benchmark: an unequal pair (RTX 3090 24 GB + RTX 3060 12 GB), Qwen3.8-Flash-Next IQ2_XS

Measured on 2026-10-14 by **Turtle**. Answers the first item of
[`docs/TEST_REQUESTS.md`](../../../docs/TEST_REQUESTS.md): two NVIDIA cards of different sizes, both card
orders, `--layer-split auto`. This machine is such a pair, and it is a hard one: the 24 GB card is on PCIe 4.0
x16 and the 12 GB card on **PCIe 3.0 x4**.

Speed only, no quality suite beyond the needle checks below. One machine, one build, one day; every number here
comes from a quiet run, i.e. the instance that normally serves this box was stopped and each measurement started
its own server on port 8092.

## Hardware and software

- GPUs: **NVIDIA GeForce RTX 3090 24 GB** (PCIe 4.0 x16, power limit 250 W) + **NVIDIA GeForce RTX 3060 12 GB**
  (PCIe 3.0 **x4**, power limit 130 W); 36 GiB VRAM in total. Link widths from `nvidia-smi -q`; the engine's own
  probe reads 26.8 GB/s on the 3090 and 3.4 GB/s on the 3060, which is what Gen3 x4 gives, so the x4 is the real
  link and not a probe artifact.
- CPU: AMD Ryzen 7 7700X (8 cores, 16 threads); RAM: 122.6 GiB; storage: two NVMe, the packs on a different
  physical disk from the model files.
- OS: Linux 7.0.0-31-generic; driver 580.178.04; CUDA 13.0.
- Strata: project 0.1.43, commit `7695859a`; engine 0.1.43. Source build.
- Background workloads: none on the cards; four of the runs below overlapped a large file download on another
  physical disk (see Limitations).

## Model and configuration

- Model: `qwen3.8-flash-next-iq2_xs` (Qwen3.8-Flash-Next, GSQ-RCO **IQ2_XS**, a native pack), context 131072,
  1 parallel slot, MTP on, `--draft-vocab cyrillic`, vision encoder loaded.
- The install's own configuration, unchanged for the baseline run:

```text
strata --pack iq2_xs --native .../IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
  --ple-gguf .../Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf
  --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto
  --spec 4 --spec-min-p 0.5 --mtp Strata-data/mtp/rt
  --max-context 131072 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700
```

- `"gpu": "0,1"` in the config. The engine reorders the cards itself: `layer split: card order 1,0 (the faster
  card last; "gpu_order": "as_given" keeps 0,1, #1352)`. In the engine's own numbering **CUDA0 is the 3060
  (28 SMs, 6.70 GiB free before its carve, miss cost x5.95)** and **CUDA1 is the 3090 (82 SMs, 18.27 GiB free,
  miss cost x1.00)**; `--layer-split auto` chose **K=3** because K=2 would leave a stage with fewer than three
  layers, and the log says `K=2 would give a stage fewer than 3 layers (a tiny expert cache there clamps the
  prompt chunk of every card, #1616); chose K=3 instead (29.3 ms per decode window, not 28.5)`.
  So the 3060 runs layers 0-2 (1536 slots, 2.06 GiB) and the 3090 layers 3-47 (12927 slots, 17.32 GiB).

## Method

`tools/strata_bench.py --long --huge` (suite 1, 3 runs per test, medians with ranges), each run starting its
own server from the install configuration. `--huge` is a local, opt-in addition to the tool (it adds the same
prompt test at about 65,536 tokens; without the flag the suite is unchanged) - the patch is offered separately and
the 64K column can be reproduced with `--long` plus that flag, or by hand. Model loading is not timed. One warm-up
chat and a 1K prompt are discarded first. Every prompt carries a unique tag, so the conversation cache cannot help
the fresh-prompt tests; `reused_tokens` was 0 in every prompt test. Thinking off, temperature 0, 24 new tokens for
the prompt tests and 100 for the short chats, 64 per agent turn.

| Test | Actual prompt tokens | New tokens | Runs |
| --- | ---: | ---: | ---: |
| short chat | 609-700 | 100 | 3 x 3 questions |
| 4K prompt | 4019 | 24 | 3 |
| 16K prompt | 16058 | 24 | 3 |
| 32K prompt | 32049 | 24 | 3 |
| 64K prompt | 64040 | 24 | 3 |
| agent session | 609 -> 14470 over 5 turns | 64/turn | 3 sessions |

The long-context section uses `tools/ctx_sizes.py` instead: one server start per KV format at
`--max-context 262144`, two fresh prompts per size, 256 generated tokens each, `reused_tokens` 0 everywhere.

## Results: the standard suite, both cards (medians, range in brackets)

| Test | RTX 3090 + RTX 3060 split |
| --- | ---: |
| short chat decode, tok/s | 114.8 (99.6-149.1) |
| short chat first token, s | 0.28 |
| 4K prompt, tok/s | 1704.6 (1700.5-1704.7) |
| 4K first token, s | 2.37 |
| 16K prompt, tok/s | 1945.1 (1944.2-1946.9) |
| 16K first token, s | 8.28 |
| 32K prompt, tok/s | 1949.5 (1947.8-1950.0) |
| 32K first token, s | 16.46 |
| 64K prompt, tok/s | 1982.0 (1979.9-1982.7) |
| 64K first token, s | 32.35 |
| agent cache hit, % | 68.7 |
| agent re-read tokens per session | 72 |
| agent turn first token, s | 2.40 |
| agent decode, tok/s | 109.8 |
| suite time, s | 238 |

For the same machine, same build and same suite, one card at a time: **3090 alone** 103.3 decode, 2132.8 / 2441.4
/ 2449.3 / 2416.4 prompt at 4K / 16K / 32K / 64K; **3060 alone** 44.2 decode, 364.7 / 616.8 / 617.8 / 615.1.
Per-run data: `../../2026-10-14-rerun/`.

## Card order: both orders of the same pair

The pair is unequal, so the order matters here. Two distinct placements were measured, each one with its own
engine line quoted:

**3060 first, 3090 second** - the engine's default reorder, `layer split across GPUs [1, 0]` with `card order
1,0`: the 3060 (device 1) runs layers 0-2 and the 3090 (device 0) layers 3-47.

**3090 first, 3060 second** - the config's order kept, `layer split across GPUs [0, 1]`: the 3090 runs layers
0-44 and the 3060 the last three, 45-47. In this batch we reached it by putting `--layer-split auto` in the args;
the engine's own log names `"gpu_order": "as_given"` as the supported way to keep `0,1`.

| Placement | short chat decode | 4K prompt | 16K prompt | 32K prompt | 64K prompt |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3060 first (`[1, 0]`, 3060 layers 0-2) | **114.8** | **1704.6** | **1945.1** | **1949.5** | **1982.0** |
| 3090 first (`[0, 1]`, 3090 layers 0-44) | 109.3 | 1559.9 | 1774.0 | 1779.9 | 1746.3 |
| 3090 first, second independent run | 109.5 | 1557.5 | 1770.9 | 1779.4 | 1744.5 |

The two runs of the second placement agree within 0.2% on every row, so the difference is the placement and not
the run: **putting the small card first and the big one last is 5.0% faster in decode and 9.3-13.5% faster in
prompt reading on this pair** (4K +9.3%, 16K +9.6%, 32K +9.5%, 64K +13.5%). That is the opposite of what the
documented reorder rule optimizes for: the rule puts the faster card last because the last stage runs the head,
the draft layer and the verify and every prompt chunk waits on it, but on this pair the 12 GB card on a Gen3 x4
link should not be the one waiting at the end. The split planner here also chooses which card gets the bulk of the
layers - 3 of 48 either way, just at opposite ends - so this is a placement result, not only an order result.

One more run in the same batch, with an explicit `--layer-split 3`, also logged `[1, 0]` (3060 first, 1536 slots)
and measured 113.1 decode / 1577.5 prompt at 4K, i.e. 7.5% below the baseline at the same placement with the same
slot counts. We did not resolve that spread and it is not used in the table above. Per-run data:
`card-orders/`.

## Long context, 128K-256K, three KV formats (2 runs each, 256 generated tokens)

The question we actually had: to work at 128K-262K, does the KV format matter? `--max-context 262144`,
`--kv-resident 32768` in all three arms. Actual prompt tokens: 127,710 / 165,608 / 194,825 / 224,047 / 255,433.

| KV | 128K prompt / decode | 166K | 195K | 224K | 256K | 3090 slots |
| --- | --- | --- | --- | --- | --- | ---: |
| int8 (default) | **1902.0** / 96.1 | **1849.3** / 95.7 | **1810.2** / 95.0 | 1777.5 / 94.0 | 1745.7 / 88.9 | 12758 |
| k8v4 | 1837.5 / 93.0 | 1792.4 / 96.8 | 1758.5 / 97.4 | 1730.0 / 91.8 | 1704.0 / 91.5 | 12827 |
| q4_0 | 1876.1 / 95.3 | 1839.5 / 93.7 | 1808.5 / 93.4 | **1780.9** / 89.2 | **1755.8** / 84.2 | 12902 |

First token times at those five sizes: int8 67.2 / 89.6 / 107.7 / 126.1 / 146.4 s.

Against int8: k8v4 is 2.4-3.4% slower at reading the prompt and within noise on decode; q4_0 is 1.4% slower to 0.6%
faster on the prompt and **5.1-5.2% slower on decode at 224K and 256K**. The memory saving is real and matches
`DETAILS.md` - the engine's own start line says `the K/V in 0.26 / 0.20 / 0.14 GiB of pinned RAM` for int8 / k8v4
/ q4_0 at 262144 - but it buys only 69 and 144 expert slots out of ~12,800, and nothing in speed. **int8 stays
the default here.** Per-run data: `long-context/`.

A side note for other reporters: measuring decode at a long context with a 64-token cap understates it, because
the MTP/spec warm-up is inside the window. The same int8 configuration measured 87.6 tok/s at 195K with a 64-token
cap and 95.0 with 256; at 256K, 81.3 against 88.9. The numbers above use 256.

## Opt-in settings we tried

Eleven settings, each one run of the standard suite in the same batch, compared against the control measured in
the same batch at the same placement. None of them is worth keeping on this rig.

| Setting | Placement | decode | 4K | 16K | 32K | 64K | Note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `--trim-stage-weights` | `[1, 0]` | 114.5 | 1591.7 | 1663.3 | 1637.2 | 1639.8 | +1.2% decode against its own `--layer-split 3` control (113.1); freed 2.7 GiB on the 3060 but the cache stayed 1536 slots (the profile has only 1536 ranked pairs for layers 0-2), so the VRAM is not turned into cache |
| `--kv k8v4` | `[0, 1]` | 114.3 | 1603.9 | 1858.7 | 1882.9 | 1928.2 | prompt 4-6% below the int8 run at the same placement in the long-context table |
| `STRATA_PCIE_AUTO_SPLIT=1` | `[0, 1]` | 109.5 | 1557.5 | 1770.9 | 1779.4 | 1744.5 | no effect; also prints no line in the log. Matches the control (109.3 / 1559.9) |
| `--pipeline-windows 2` | `[0, 1]` | 92.3 | 1559.6 | 1767.5 | 1770.2 | 1737.5 | **15.5% slower decode**, prompt unchanged; the log says `two verifiers per stage` and it took 371 MiB out of the 3060's cache. Docs say a two-stage split stays opt-in |
| `STRATA_PF_FUSED=1` (3090 alone) | n/a | 102.3 | 1941.9 | 2114.0 | 2092.9 | 2045.4 | **9.0 / 13.4 / 14.6 / 15.4% slower prompt** than the 3090 alone (2132.8 / 2441.4 / 2449.3 / 2416.4). In `docs/MULTI_GPU.md` this flag is +6.4% prompt on 2x RTX 3090 - a different rig and a different card count, and we cannot reproduce the gain here |
| `--resident-experts` | `[1, 0]` | 109.2 | 1389.0 | 1630.4 | 1610.1 | 1636.1 | the low-RAM mode itself costs 4.9% decode and 17-18.5% prompt on a 122.6 GiB box that does not need it |
| `--resident-experts --adapt-async 1` | `[1, 0]` | 109.4 | 1379.6 | 1614.7 | 1609.3 | 1638.9 | nothing (+0.2% decode): the log says `43 rounds, 3609 experts swapped in, 67.5 ms per round`, and a round fits inside the 4 x 29.4 ms windows, so the blocking tier had nothing left to hide |

`--adapt-async 1` **does not run at all** in the default mode: with `--expert-cache auto` the engine prints
`--adapt-async 1 is off (it needs the resident RAM mode, --resident-experts)`. Two of the runs above are therefore
controls, not measurements of the flag. Per-run data: `options/`.

## KV streaming counters: this is the one thing we think is a bug

On the split the engine's per-request counter reads **`KV streaming: 100.00% of 0 block reads hit VRAM, 0.0 MiB
read from RAM`** in every run - at 131072 and at 262144 context, with 32K and with 256K prompts, and with all
three KV formats. The same machine, build, model and `--kv-resident 32768` on the **3090 alone** reports real
reads for the same residency line: `89.58% of 246456 block reads hit VRAM, 103.5 MiB read from RAM`.

Both start lines are identical in shape:

```text
split:   strata generate: KV streaming: 32768 of 131072 cells per QSA layer in VRAM, the K/V in 0.13 GiB of pinned RAM
one 3090: strata generate: KV streaming: 32768 of 131072 cells per QSA layer in VRAM, the K/V in 1.55 GiB of pinned RAM
```

With three quarters of each QSA layer's cells not in VRAM, a 64K-256K prompt has to touch the RAM copy, so either
the split does not stream K/V blocks at all or the counter is not wired for the split path. We could not decide
which from the outside, and it matters to us because it is where the time of a long prompt goes. Reported
separately as an issue with the full logs.

## Correctness checks

- `tools/needle_bench.py`, 8K / 16K / 28K at depths 10 / 50 / 90, on the split: **9 of 9 found for int8, for
  k8v4 and for q4_0** (27 requests). At 28K the prompt is about 28,177 tokens and answered in 17 s.
- The standard suite's prompt tests report `reused_tokens: 0` on every fresh prompt, so the numbers are first
  reads, not cache hits; the agent sessions report their own reuse (68.7% hit, 72 re-read tokens over 3 sessions).
- No coding, tool-calling or image result was checked. Nothing here is a quality claim.

## Limitations

- One machine, one build, one day. Nothing here is an A/B between engine versions.
- Four of the runs (the long-context arms and the KV sweep) ran while a large model file was downloading to
  another physical disk. The pack is read from the other NVMe, but the CPU and the RAM are shared.
- The 3060 sits on a **Gen3 x4** link. `--pcie-frac` cannot set the two cards' shares separately, and
  `STRATA_PCIE_AUTO_SPLIT=1` did nothing measurable here. A different slot is the only real fix, untested.
- Power limits were left at 250 W / 130 W; no power draw, temperature or clock was recorded, and no run was long
  enough to show heat.
- `parallel_slots` is 1: nothing here says anything about several requests at once, and no soak was run.
- The 4K prompt column is the least reproducible one on this rig: two runs of the identical placement differed
  by 7.5% there once (see Card order), while the 16K-64K columns agreed within 0.5% in the same pair.
- `--draft-vocab cyrillic` is in the config but the suite is English and code, so it is not exercised.
- Not tested: the low-RAM mode with less than 122 GiB of RAM, Windows/WSL, AMD, and `--kv-resident` raised above
  32768.

## Data

- `results.json`, `engine-log-excerpt.txt` - the 3060-first split run, the one in the tables above
- `card-orders/explicit-layer-split-3.json`, `card-orders/order-0-1-control.json` - the card-order runs
- `long-context/int8.json`, `k8v4.json`, `q4_0.json`, `int8-engine-log-excerpt.txt` - the 128K-256K runs
- `options/*.json` - the opt-in settings, one per row of that table

All files are `tools/strata_bench.py` / `tools/ctx_sizes.py` output; only local model paths were shortened.
The scripts are in the repository, the commands are the two tool names above plus the flags named in the tables.
