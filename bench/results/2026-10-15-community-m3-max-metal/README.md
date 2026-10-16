# Community benchmark: Apple M3 Max (Metal), Flash-Next IQ3_S

Measured 2026-10-15 by @alvarorsouza-arch. Engine 0.1.44.1-metal built from source on `main` at `296b59d`, macOS 26.6.
Hardware: Apple M3 Max, 40-core GPU, 128 GB unified memory, 8 TB SSD, on AC. Model: GSQ-RCO IQ3_S, context 65,536,
no vision, experts resident with no SSD streaming, setup defaults otherwise.

Result at defaults, with the Mac in high power: **55.0 / 58.8 / 58.8 tok/s decode** on my fixed 449 token harness
prompt with 256 out, and **504 tok/s** on a 16,388 token prompt. That decode figure has the MTP drafter on, which setup
writes by default; with no drafter at all the same build gives 48.2 / 48.6 / 48.8 tok/s, and that is the number I use
for bandwidth. docs/MAC.md reports 94 to 102 tok/s for IQ3_S on an M5 Max, so this chip lands at about 1.6x slower,
close to the published bandwidth ratio (400 against 546 GB/s). No setting I tried lifted decode above the defaults.

## Power mode was worth more than any setting

The same build, the same config and the same harness run gave **24.1 / 24.4 / 24.3 tok/s** in an earlier session,
with prompt ingestion at 223.9 tok/s for 10,908 tokens. After I put the machine in high power (`pmset -g` reports
`powermode 2`; I did not record the previous value) decode went to 57.4 / 57.6 / 58.1 and 15,748 token prompts
ingested at 488 / 482 / 473 tok/s. My nightly harness flagged it as `+136.1%` against its own nine prior rows.

So a Mac report can be off by more than 2x on power settings alone. I would have written a wrong conclusion about
kernels had I not switched it. Suggest that community reports for Macs quote the power mode, and that docs/MAC.md
says a laptop in low power mode runs at roughly half.

## What the engine picks on this GPU

`metal/probe/mtl_probe.cpp` here: `apple7 1, apple8 1, apple9 1, apple10 0, metal3 1, metal4 1`. The `coop_k4` probe
in `metal/tests/coop_probe.h` returns false, so `window.cpp` keeps `gu_v3`, `dn_v3` and `flash` off, and the prompt
path announces it:

```text
strata metal: prompt kernels (mac-kern's prompt chain) for chunks of 2048 rows,
              the older expert / attention kernels (no M5 layout), ready in 231 ms
```

Decode is GPU time rather than I/O or submission: a 256 token answer reports `files 0 blobs ... 169.9 MB read`, so
nothing streams, and the trace puts the time on the GPU.

The bytes per step are not a guess here. Joining the pack's `index.txt` with the GGUF tensor table, one token costs
5,159 MB of weights: 982 MB of routed experts (10 of 512 in each of 48 layers) and 4,177 MB of everything else, where
attention alone is 2,202 MB, the hyper connections 648 MB, the LM head 521 MB and the SSM path 468 MB. That is 9.4% of
the 54.8 GB pack per generated token, and the experts are only 19% of the traffic.

With no drafter at all (`--mtp` and `--spec` removed, `--suffix-draft 0`), decode is **48.2 / 48.6 / 48.8 tok/s** in
high power. `STRATA_METAL_TRACE=1` gives a median step of 20.26 ms, 19.23 ms of it on the GPU, about 1,110 dispatches.
20.26 ms over 5,159 MB is 255 GB/s. `metal/tests/rt_bench.cpp`, unmodified, measures what this GPU can read: **381.8
GB/s** best (five runs today gave 381.8 to 394.9) and 365.7 GB/s for chunks of 1 MB, which is the shape the experts
arrive in. Pure decode therefore runs at
70% of a plain streaming read of the same bytes, and the ceiling this pack allows is 70.9 tok/s. Of the 20.26 ms step,
14.1 ms is the byte time of those bytes and 6.2 ms is something else; `rt_bench dispatch` prices a barrier between
dispatches at 1.2 us, so 1,110 of them account for 1.3 ms, which is not where the 6.2 ms went.

## Paired runs, one boot each, all in high power

| Variant | Prompt tok/s (tokens) | Harness decode, three runs | Cold 449 token prompt tok/s |
| --- | ---: | --- | ---: |
| defaults | 504 (16,388) | 55.0 / 58.8 / 58.8 | 404.6 |
| `--expert-cache 1024` plus `--no-prefill-borrow` | 491 (18,948) | 52.5 / 56.5 / 57.7 | 394.1 |
| `--expert-cache 4096` plus `--no-prefill-borrow` | 480 (18,948) | 52.3 / 56.8 / 58.3 | 394.7 |
| `COOP_K4=1` | 597 (18,308) | 53.7 / 56.5 / 58.0 | 500.4 |
| `STRATA_METAL_MATH=fast` | 490 (17,028) | 54.3 / 58.2 / 60.0 | 398.0 |
| `STRATA_METAL_GEMV_NR=2` | 500 (18,308) | 51.1 / 53.7 / 55.0 | 394.9 |

Prompt token counts differ per row because each row used its own filler, which is why I give them. Nothing beats the
defaults on decode. `COOP_K4=1` buys roughly 20% on prompt ingestion and is the only variant worth mentioning, and it
should not be used, see below.

## Broken or missing on this backend

`--kv int8` and `--kv-resident 32768`, both fine on my CUDA box, stop the Mac engine at boot:

```text
strata metal: strata::kernels::kv_block_bytes has no Metal implementation yet
```

From `--expert-cache 1024` upward every prompt request fails with HTTP 400:

```text
lending the prompt path its slots failed:
prefill: relayout needs borrowed buffers and a chunk of at most 0
```

`--no-prefill-borrow` avoids it and is how the two cache rows above were measured. At the default 64 the engine only
warns that the prompt path allocates its own buffers. Raising the cache also does nothing for speed, which fits the
design: Metal reads experts in place from the mapped file, so there is no bus to save. In an earlier session
`--expert-cache 16384` made 40.62 GiB of cache, RSS 90.4 GB and 5.7 GB of swap, with decode unchanged.

`--batch 2`, `--batch 3` and `--batch 4` (also spelled `--slots`) start without a complaint: the engine builds the slot
sessions (1.73 GiB each, 6.91 GiB for four) and prints `strata verify: batch windows of up to 4 sequences`. The first
request that runs two sequences together stops the process:

```text
strata batch: slot 0 takes 67 tokens (copied in 12.7 ms)
strata metal: strata::kernels::pdl_supported has no Metal implementation yet
```

The exit code is 86, which `metal/src/engine/missing.cpp` documents as its default. The server restarts the engine,
which costs one to three minutes, and the requests behind it fail the same way. Through `tools/serve_load.py`,
`--clients 4 --rounds 3 --max-tokens 256`: four requests return 2 tokens each in about 5 s (0.4 tok/s total) where one
client in the same run gets 44.2 tok/s. `STRATA_METAL_STUBS=log` shows the distance to a working path: the batch route
reaches 17 entry points with no Metal implementation, `pdl_supported` and `pdl_scope` first, then the multi-row family
a batched window would need (`fused_gr_read_multi`, `gdn_conv_l2_multi`, `gdn_ab_multi`, `gdn_step_norm_multi`,
`shared_expert_multi`, `broadcast_streams`) and the pool side (`doorbell_publish_res`, `copy_from_mapped`,
`copy_i32_from_mapped`). That mode leaves outputs unwritten on purpose, so the answers are meaningless.
docs/MAC.md says `--parallel` is not ported; on 0.1.44.1 the flag is `--batch`, and the engine accepts it and dies
mid-request instead of stopping at boot the way `--kv int8` does.

`COOP_K4=1` forces the M5 cooperative tensor layout and the output goes wrong. Paired runs at temperature 0, taken in
the throttled session, since correctness does not depend on power: recall of a code word hidden at mid depth in 9,106
tokens returned correctly at defaults and produced no answer in 1,200 tokens with the flag set, and an arithmetic
prompt that defaults answer correctly turned into text about prompt injection. The probe is right about this GPU.

## Tensor units on the prompt path: the switches exist, they do not pay here

The boot line above says `the older expert / attention kernels (no M5 layout)`, and `window.cpp` backs it: on this
GPU the prompt GEMM runs the v2 experts and the fp32 `qsa_attn`, and matrix units only come in through
`STRATA_METAL_PROMPT_EXPERTS=v3` and `STRATA_METAL_PROMPT_QSA=flash`. #2609 tunes the decode GEMVs and tiles for the
M1 and M2 families only, with 35.3 to 39.1 tok/s on an M2 Max with IQ3_XXS and identical output, and it states that
M3 and newer build exactly what they build today. #2348 has `qsa_attn_mm` and `qsa_flash` failing on an M1 Max with
`kIOGPUCommandBufferCallbackErrorInnocentVictim`. So the M3 family sits between two tuned paths and inside neither.

I tried the two switches here, one boot each, all in high power, two prompts of about 16K tokens, the fixed harness
three times, and a needle recall in 9,106 tokens as a correctness gate.

| Variant | Prompt tok/s, two runs | Harness decode, three runs | Needle |
| --- | --- | --- | --- |
| defaults | 486.0 and 481.9 | 52.8 / 56.5 / 57.6 | pass |
| `STRATA_METAL_PROMPT_EXPERTS=v3` | 498.7 and 514.2 | 53.6 / 57.5 / 58.7 | pass |
| plus `STRATA_METAL_PROMPT_QSA=flash` | 480.4 and 469.3 | 52.2 / 55.9 / 56.6 | pass |

The v3 gain is about 4%, which is inside my own session to session variation, and flash costs about 4%. I kept the
defaults. What the M3 seems to be missing, in my reading, is a tuning pass for family 9, the way #2609 did for
families 7 and 8: pure decode here already runs at 70% of the read bandwidth this GPU reaches in `rt_bench`, so the
remaining 30% looks like per step scaling rather than a flag that is switched off.

## The drafter is on in every decode number here

Setup writes `--spec 2` with the MTP draft layer and the suffix drafter on, so 58 tok/s counts accepted tokens rather
than GPU steps. The 256 token harness run reports `drafts accepted 113 of 143`, which is 143 steps for 256 tokens, so
about 1.79 tokens per step and roughly 31 ms per step. One boot per row, same session, high power.

| Drafter | Harness decode, three runs | Tokens per step | Prompt tok/s, two runs near 16K |
| --- | --- | ---: | ---: |
| MTP `--spec 2` plus suffix (defaults) | 51.9 / 56.7 / 57.8 | 1.77 | 500.5 and 481.7 |
| MTP `--spec 2`, suffix off (`--suffix-draft 0`) | 53.6 / 57.1 / 57.8 | 1.79 | 479.3 and 476.3 |
| suffix only (no `--mtp`, no `--spec`) | 47.6 / 50.4 / 51.3 | 1.13 | 521.7 and 514.4 |
| MTP `--spec 3` plus suffix | 44.9 / 49.3 / 49.6 | 2.42 | 490.9 and 475.2 |

MTP is worth about 15% on this chip, against the 27 to 36% docs/MAC.md shows on an M5 Max: a step of two tokens costs
me 31 ms against the 20.3 ms of a one token step with no drafter, so the window saves bytes (the dense 4,177 MB gets
read once for 1.79 tokens) and pays for the proposal and the verify work. In bandwidth terms the defaults sit near 186
GB/s, under the 255 GB/s of the pure step, so with the drafter on this engine is not close to its read ceiling.
`--spec 3` is clearly worse (48.7 ms per step) and the suffix drafter does nothing on this prompt. Prompt ingestion
runs a few percent faster with no MTP at all. All four rows passed a needle recall in 9,106 tokens.

## Where the remaining time is not

Two measurements narrow this down, both in high power with no drafter.

`gridbar_bench` (in the repo) reads a chain of 48 phases of 107 MB, one dispatch per phase, at 379.4 GB/s, and 96
phases of 54 MB at 373 to 377 GB/s. So the layer chain and the dispatch boundaries cost nothing here: those same bytes
arrive in 14.1 ms, and the step takes 20.26 ms. `rt_bench dispatch` adds that 1,110 barriers cost 1.3 ms.

The dequant arithmetic is not the 6.2 ms either. `STRATA_HC_Q8=0` swaps the hyper connection read from the Q8_0 copy
(648 MB per step) to the BF16 original (about 1,260 MB), which is 11.9% more bytes with nearly free decoding. If the
step were only paying for bytes it would fall from 49.1 to 43.9 tok/s; it fell to 45.3 / 46.0 / 45.9, so the cheaper
arithmetic was worth about 0.2 ms of the step. Byte traffic scales, arithmetic barely moves.

So the shape of a fix, in the order I would try it:

- Fewer bytes on the dense path, which is 81% of the traffic: attention at 2,202 MB per token and the LM head at 521 MB
  are the fat, not the experts.
- Batching, which is missing on this backend: `qsa_decode_attn_batch`, `native_qsa_indexer_append_batch` and
  `native_ple_postops_batch` are `missing()` stubs in `metal/src/engine/stubs.cpp`. Reading 4,177 MB once for four
  streams is the only change that turns bandwidth into throughput without touching quality.
- The drafter's own cost: the second token of a `--spec 2` window adds 10.7 ms of GPU, while the extra bytes it reads
  are small (see the U(2) section, about 0.44 GB more of experts plus the draft layer, whose own drafting is measured
  at 1.848 ms per round in a one-shot run). `--mtp-q4` exists in `src/program/generate.cpp` and has no Metal path, and
  the draft layer here is 116 MB of dense plus 708 MB of experts.
- A stage level GPU profile for decode: this one already exists as `STRATA_METAL_KPROF`, used below, and it is not
  documented anywhere. What would help is a line in docs/MAC.md and a `rt_bench` style summary of the classes, since
  that is what made these numbers readable at all.

## How much of the matrix hardware is in play

`metal/tests/mm_bench.cpp` on this machine, no changes: fp16 `matmul2d` reaches 12.0 to 13.3 TFLOP/s on a square
4096 cubed and 12.4 to 13.1 TFLOP/s on the model's own dense shape (2048 tokens by 10240 by 2560). On the expert
shapes it is 9.5 to 11.1 TFLOP/s with 128 tokens in a batch, and only 2.0 to 3.1 TFLOP/s with 16 tokens.

Weight elements per token, from the same inventory as the byte count: 6.67 G, so 13.34 GFLOP per token of matmul.
That puts prefill at 504 tok/s at 6.7 TFLOP/s, about 51% of the measured matrix peak, and pure decode at 48.5 tok/s
at 0.65 TFLOP/s, about 5%. The decode figure is not waste: with one token per step there is nothing to multiply, and
`mm_bench` shows the units already fall to a fifth of peak at a batch of 16. Batching is what would let this GPU do
the arithmetic rather than only the reading, which is the same conclusion the byte count gave.

The Apple Neural Engine is not in the picture at all for this backend: `ANE`, `CoreML`, `MPSGraph`, `Espresso` and
`H16` do not appear anywhere in `metal/src`, `serve` or `docs/MAC.md`, and Strata is a Metal compute engine. I could
not read the ANE counters either, `powermetrics` wants root and sudo on this machine wants a password.

## Per kernel class in one decode step (`STRATA_METAL_KPROF=1`)

The engine already has this: it replays each kernel class of a step alone and prints the median. Window of one token,
1106 dispatches, GPU 18.611 ms, and the classes replayed in series sum to 18.547 ms, so nothing is being lost between
kernels. The step is the honest sum of its kernels, and the byte rate of each is:

| Class | ms | MB | GB/s |
| --- |---:|---:|---:|
| LM head gemv | 1.329 | 521 | 392 |
| `attn_qkv` gemv | 1.302 | 473 | 363 |
| `ssm_out` gemv | 1.089 | 468 | 430 |
| hyper connection read (`gr_down_q8` plus `gr_up_q8`) | 2.682 | 648 | 242 |
| expert GEMMs (`moe_gu` plus `moe_down`) | 3.565 | 982 | 275 |

Against the 365 to 385 GB/s that `rt_bench` measures here, the head, attention and SSM gemvs are already at the machine,
and the two laggards are the hyper connection read at 242 and the expert path at 275.

Going from a window of one token to the `--spec 2` window of two (GPU 26.184 ms, 1142 dispatches) shows where the
marginal token goes: `gdn_step` goes from 1.099 ms over 36 dispatches to 2.394 ms over 72, and the hyper connection
read from 2.682 to 4.938 ms, 1.84x for the same 648 MB of weights. Those kernels scale with rows, so a second token
does not ride on the weight read that is already in flight. `moe_down` goes 1.270 to 1.955 ms, which is about what the
extra routed experts weigh.

`--no-fused-gr` does nothing on this backend: `fused_gr_read`, `fused_gr_read_multi` and `fused_gr_supported` are
`missing()` stubs in `metal/src/engine/stubs.cpp`, and `setup_host.cpp` has an empty `fused_gr_check()` with the
comment that Metal has no variants. A paired run of pure decode agrees, 51.4 / 51.9 / 51.4 against 51.3 / 52.0 / 51.9
tok/s. Worth marking in the help text, since the flag reads like a general A/B.

Session note for anyone repeating this: pure decode across four boots today ran 48.2 to 52.0 tok/s, about 7%, so any
single knob under roughly 5% is not measurable in one session, mine included.

## U(2), the factor the byte model was missing

`tools/quant_bound.py` wants U(P), the distinct experts a pass of P tokens reads per expert a single token reads, from
the engine's window logs, and falls back to `--union` (default 1.0) without them. I tried to measure it with
`--dump-routing` on a one-shot run and with
`tools/collect_profile_traces.py --mode routing`, with and without `--mmap-experts`. Both runs generated correctly and
both wrote empty traces:

```text
routing dumped           /tmp/one.trace (0 records of layer, k, ids, weights)
verify window            wait for rings 0.000  pool 0.000  host 0.000  commit 0.000 ms/round;
                         CPU experts 0.00 distinct / 0.00 routed per layer
```

`generate.cpp:5929` prints that explanation when you pass `--no-pool`: the routed ids reach the host through the
doorbell the expert pool reads. The Metal engine has no host pool, it reads experts in place from the mapped file, and
the core does not notice, so on a Mac the flag accepts the path and produces nothing. `collect_profile_traces.py` does
not notice either, because the file exists.

What the KPROF numbers gave first was a ceiling: the two token over one token time ratio of the classes that read
expert weights is 1.54 (`moe_down_20_4_2_4`), 1.38, 1.43 and 1.39, a mean of 1.45, which put U(2) at 1.45 or below.
Then I made the measurement direct with a small opt-in patch, Metal only, sent as #2652: a `route_dump` kernel copies
each layer's `moe_route` output into its own slot, because `ids_` is reused by the next layer and a resident window has
no host turn between layers, and `finish()` writes the records in the `--dump-routing` format. The variable is
`STRATA_METAL_DUMP_ROUTING=path`. Measured on this machine over 48 layers and 123 window rows, greedy, one prompt:

| Pass | U(P) mean | across layers | experts repeated in the pass |
| --- |---:|---|---:|
| 2 tokens | 1.684 | 1.543 to 1.950 | 15.8% |
| 4 tokens | 2.822 | 2.288 to 3.765 | 29.5% |
| 8 tokens | 4.615 | 3.301 to 6.773 | 42.3% |

U(2) came out above the kernel time ratio, so the expert kernels already take some of the second token's weights for
less than their cost in bytes. With U(2)=1.684 and U(8)=4.615 and the 393.1 GB/s `rt_bench` gave in the same session,
`tools/quant_bound.py` puts the floor of a two token window at 135.1 tok/s, against the 26.18 ms verify pass I
measured (56% of that floor), and the floor of an eight token pass at 361.5 tok/s. That eight token figure wants
4.8 TFLOP/s of matmul, while `mm_bench` measures 2.0 to 3.1 TFLOP/s at batch shapes that small, and that is the kernel
work batching would need rather than the bandwidth. The trace is cheap and I cannot pin the cost down: one pair gave
61.31 against 63.11 tok/s (2.9%) and a later pair 60.97 against 61.21 (0.4%) on a 120 token greedy run, and pure decode
moves about 7% boot to boot here. A build of `main` and the patched build with the variable unset printed the same 120
output ids, byte for byte.

## Not tested

The guide's 4,096 / 32,768 / 128,000 sweep with three runs per size; vision; contexts above 65,536; `--spec` above
2; `--parallel` untested and the batched decode kernels are `missing()` stubs on Metal; soaks; a power and clock trace,
I only have `pmset` state and before and after medians. The pack and the MTP draft layer were copied from a 0.1.40.3
install, so a clean first install on 0.1.44.1 on this hardware is unverified.

## Doc asks

- docs/MAC.md, Limits: `--kv int8` and `--kv-resident` have no Metal implementation and abort the engine.
- docs/MAC.md: power settings on laptops, with the measured factor (24 tok/s in the previous mode against 58 in high
  power, same build and config). Community reports should quote the power mode and the drafter state (`--spec`, MTP on
  or off) beside every decode number, since both move the number by more than any tuning knob I could find.
- docs/MAC.md: `--expert-cache` on Metal is a blob count over weights that are already mapped, so raising it costs RAM
  and buys nothing, and past about 1,000 slots the prefill borrow fails with the 400 above.
- docs/MAC.md: mark `COOP_K4=1` as unsupported on `apple9` hardware rather than experimental.
- `--help`: the A/B flags that are CUDA-only. `--no-fused-gr` and the fused gr path have no Metal implementation, and
  on a Mac they read like a tuning knob that silently does nothing.
- `STRATA_METAL_KPROF` deserves a paragraph in docs/MAC.md: it is the fastest way to see what a Mac step is made of,
  and I only found it by reading `window.cpp`.
- `--dump-routing`, in the help and in `tools/collect_profile_traces.py`: it needs the host expert pool, so on Metal it
  writes a trace with 0 records. The collector should fail on an empty trace rather than accept the file.
- docs/MAC.md: no M3 row. If #2609 lands, the M3 family (9) still builds what it built before, so an M3 number for
  `STRATA_METAL_PROMPT_EXPERTS=v3` plus a family 9 tile pass would tell people whether there is anything to turn on.
- docs/MAC.md: `rt_bench bw` is already in the repo and it is what let me read my own numbers. A decode tok/s on its
  own does not say how close a Mac is to its read ceiling, and every table here is with a drafter on; a no-drafter row
  and the machine's `rt_bench bw` figure belong next to them.
- docs/MAC.md, and the engine: `--batch` on Metal. The doc line names `--parallel`, which no longer exists; the flag
  today is `--batch`, the engine accepts it, builds 1.73 GiB of slots per sequence and then exits with code 86 on the
  first request that runs two together. Refusing it at boot, the way `--kv int8` stops there, is kinder than a restart
  between requests.
