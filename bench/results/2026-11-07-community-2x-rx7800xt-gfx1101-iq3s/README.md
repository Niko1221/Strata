# Community benchmark: 2x RX 7800 XT (gfx1101), Ryzen 7 3700X, 61 GB

Two RDNA3 consumer cards on an x4/chipset + x16/CPU-direct pair, no AVX-512 on the CPU, the
stock recommended IQ3_S weights, and the model's native 262,144-token window with the whole
INT8 KV cache in VRAM. Reported per `docs/COMMUNITY_BENCHMARKS.md`: three runs per
configuration, medians and ranges, prompt and decode throughput kept apart, and the recall
check attached. Everything here was measured on 2026-11-07 on one machine, in one day, on the
released v0.2.12 package (engine 0.1.58) with no source build and no non-default build options.

The headline is not the throughput table. It is the decode-window anatomy in
[decode-timing-excerpt.txt](decode-timing-excerpt.txt): on this machine **67.3% of every decode
window is `GPU-reach wait`** — the two stage handshakes across the PCIe split — with 1.85
tokens accepted per window and a 26.0 ms window, which implies 74.7 tok/s and matches the
measured 79-96 tok/s at community depths and 55-65 tok/s at the 190-215K depths this box
actually serves. 26 configuration arms, the vendor's own exclusive decode calibration, and two
different drafters were measured against that ceiling; almost nothing moved it, and the one
thing that moved a lot was prompt speed, not decode.

## Hardware and software

| | |
| --- | --- |
| GPU | 2x AMD Radeon RX 7800 XT, 16 GB each (`gfx1101`, PCI ids `0x747e`, SKU `EXT98286`) |
| GPU placement | GPU0 on the **chipset** arm `06:00.0`, electrically **x4**; GPU1 **CPU-direct** in J10, `0d:00.0`, **x16**. Both links train 16 GT/s (Gen4) |
| Measured PCIe | pinned host-to-device: **6.42 GB/s** on GPU0, **19.45 GB/s** on GPU1. The chipset slot is wired x4 (`03:02.0` LnkCap x4) and the CPU<->X570 uplink stays x4, so a device swap can never widen GPU0 |
| GPU-GPU | no bridge, no P2P path in use: the layer split exchanges through host memory |
| CPU | AMD Ryzen 7 3700X, 8 cores / 16 threads, 1 socket, **no AVX-512**, one CCX, `numa_balancing=0` |
| RAM | 60.7 GiB |
| Storage | WD SN850X 8 TB (model, pack, engram table; LV `strata-data`) + Kingston NV2 1 TB in the CPU-direct M2A slot |
| Board | Gigabyte X570 GAMING X |
| OS | Ubuntu 26.04.1 LTS, kernel 7.0.0-42-generic |
| ROCm | HIP toolchain from the Ubuntu packages (`hipcc 7.1.1`), in-kernel `amdgpu` |
| Strata | **v0.2.12** — `strata 0.2.12 (build 0.1.58; python 3.12.15)`, official release package (`install.sh`), backend `hip`, devices `[0, 1]` |
| Engine | `/home/dom/strata-official/v0.2.12/engine/strata`, sha256 `74f27a82451cb039e466a72869ae88255dbb225cbabbb7f7ee3357ccc704b0eb`, **byte-identical to the release component manifest** (`engine-component.json`, `build_id 7be42963e6be+rocm`, `archs: gfx1100, gfx1101, gfx1102` - our cards are `gfx1101`, a declared target). The release `SHA256SUMS.txt` hashes the release archives, not the unpacked engine; `install.sh` verified the archive at install time |
| Vision encoder | `engine/strata-vision` + `mmproj-Qwen3.8-Flash-Next-BF16.gguf`, sha256 `b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0`, 907,543,008 bytes, running on the **CPU** with 8 threads |

Full machine/software record: [BUILD.json](BUILD.json). Model and weight provenance, including
the shard sizes and the mmproj note: [model-provenance.json](model-provenance.json).

## Model and configuration

Qwen3.8-Flash-Next **IQ3_S** (the recommended quality tier) from
`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, two GGUF shards (54,817,524,224 +
28,800,138,432 bytes), the pack `Strata-data/packs/iq3_s`, `data/expert-profile.bin`, and the
MTP draft layer in `Strata-data/mtp/rt`. These are the **stock** weights: the abliterated
(uncensored) IQ3_S copy is also present on this box and was measured earlier the same day;
the numbers below are the stock model, which is also what the standard protocol is defined on.

```
--pack /mnt/strata-data/Strata-data/packs/iq3_s
--native .../models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
--ple-gguf .../models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
--expert-profile data/expert-profile.bin --expert-cache auto
--prefill 8192 --spec 2 --spec-min-p 0.5 --draft-conf 0.7
--mtp /mnt/strata-data/Strata-data/mtp/rt
--max-context 262144 --kv int8 --kv-resident 0
--layer-split 24 --vram-reserve-mib 1024 --resident-budget-gib 40
--conversation-cache-mib 8192
--vision
```

Engine environment as read from the running process (`/proc/<pid>/environ`), which is the profile's
`env` plus the keys the systemd drop-in adds:

```
STRATA_DECODE_TIMING=1
STRATA_HIPBLASLT_TUNING=/home/dom/strata-pr-test-20261016/upstream-table.txt
STRATA_METRICS_HISTORY_S=240
STRATA_MTP_STAGE0=1
STRATA_PF_FUSED=1
STRATA_PREFILL_RING=512
STRATA_REPLICA_START=serial
STRATA_ROUTE_TAIL_SKIP=7
STRATA_STAGE_TRIM=1
```

The hipBLASLt table is the borrowed 43-row `gfx1100` table the engine loads for `gfx1101`
(`prefill gemm: hipBLASLt tuning enabled (43 rows, gfx1101, version 100200)`); the release ships
no `gfx1101` table. `--kv-resident 0` is written into both `args` and `user_args`:
without it the 0.1.58 launcher can silently choose streamed KV despite the `args` value, and
the point of this configuration is the whole INT8 KV in VRAM.

Serving `host 0.0.0.0`, `api_key` set (removed from [config-sanitized.json](config-sanitized.json)),
`parallel: 1` (one conversation at a time; the 2/4/8-slot batch attempts on this engine are a
different experiment and are recorded as a failure below).

Two engine-auto behaviours are worth knowing before reading the tables: the engine picks
`pcie_frac 0.18/0.54` itself for this split, and `--pipeline-windows` above 2 is **clamped back
to 2** on a two-stage split (the load line prints 2 even when 3 is in argv).

Observed after warm-up: **30,486 MiB VRAM across the two cards** (of 2 x 16 GiB), engine RSS
**59.4 GiB**, `MemAvailable` **19.2 GiB**. GPU0 ran at 80% utilisation, active sclk 2581 MHz,
188 W against a 212 W cap, junction 82 C; GPU1 47 W of a 220 W cap, HBM 86 C (the hottest
silicon on the box).

## Method and reproduction

`benchmark.py` in this folder. It is the house protocol: 4,096 / 32,768 / 128,000 target prompt
tokens, **256-token output**, `temperature=0`, non-streaming, three runs per depth, a distinct
`Document revision marker:` line per run so no run reuses another run's prefix, and one warm-up
request before the measured runs (warm-up and the ~2 min engine start are never timed). The
corpus is a repeated paragraph of MoE-routing prose, so the filler selects a narrow expert set;
the recall check below is the correctness evidence for long context, not these prompts.

The harness calibrates its own filler: after the warm-up request it takes the engine's reported
`prompt_tokens` and derives bytes-per-token (4.75 for this paragraph), so the target depths are
honest. In the first A/B rounds the constant was inherited from a different corpus (3.388), so
those rounds ran 2,946 / ~25K / 89,665 tokens; they are kept and labelled as measured, because
they are internally consistent and they are the rounds the A/B decision was taken on.

Decode throughput comes from the engine's own accounting line, not from wall-clock division:

```
strata serve: prompt 89665 tokens = 0 reused + 89665 read in 53192 ms (1685.5 tok/s),
  256 generated in 2664 ms (96.1 tok/s), drafts accepted 138 of 149, 1 checkpoints
```

so reasoning tokens are counted the way the vendor counts them, and the same line also gives
`reused` (prompt-cache reuse, which must be 0 for a cold run) and the draft-acceptance fraction.
`expert cache hit rate` is taken from the engine log per run, and VRAM / `MemAvailable` /
engine RSS are snapshotted per run (see the `-runs.csv` columns).

Three independent instruments were used, because single runs on this box drift by +-5%:

1. **A 26-arm configuration screen** on short prompts (4K / 32K / 64K, one warm-up, one request
   per depth per arm), one fresh engine start per arm, order fixed: `sweep-summary.jsonl`,
   `screen-arm-table.md`.
2. **The vendor's own exclusive tuner**: `strata engine calibrate --measure --exclusive`, which
   restarts the engine and sweeps `--pcie-frac`, `--spec-min-p`, `--pool-workers` and the
   adapt ladder, 3 rounds for the first two dimensions. Raw round lines:
   [calibration-rounds.txt](calibration-rounds.txt); the machine-readable report:
   [calib-01.json](calib-01.json).
3. **The interleaved A/B** described above: two configs, two rounds, order swapped between
   rounds, fresh engine per arm. This is the only instrument whose differences are trusted; the
   round effect it exposed was large (both configs gained ~12% at 32K and lost ~11% at 4K
   between rounds).

Correctness gates: `qa-fingerprints.md` (six arithmetic / date / capital / code / sum prompts at
`temperature=0`, byte-stable answers, 6/6 on every promoted arm), a needle-in-a-haystack recall
check ([needles.json](needles.json)) at 32K and 128K with needles at depth 10/50/90, and an
image probe on a synthetic PNG ([vision-probe.json](vision-probe.json)).

The standard protocol run completed on the promoted configuration: **22/22 requests, `status: complete`, suite quality gate 12/12**, and its own artifacts are kept in [standard-v1-run/](standard-v1-run/) (`results.json`, `outputs.jsonl`, `config.json`, `provenance/`).

## Results

### Standard protocol (`strata bench run --suite standard-v1`)

| case | runs | input tokens | cached | output tokens | prompt tok/s (median, range) | decode tok/s (median, range) | TTFT s | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `short` | 3 | 77 | 0 | 512 | 65.1 ([60.7, 73.4]) | 58.0 ([56.8, 68.0]) | 1.21 | completed,truncated |
| `prompt-4k` | 3 | 4093 | 0 | 512 | 770.0 ([768.6, 771.4]) | 57.9 ([56.3, 66.5]) | 5.35 | truncated |
| `prompt-16k` | 3 | 16387 | 0 | 512 | 1299.5 ([1295.6, 1301.5]) | 58.8 ([52.9, 61.1]) | 12.65 | truncated |

Protocol `strata-qwen38-thinking-v1-draft.1` (corpus sha256 `e75eff8b9feea04f745520b392c4c17d97aefc1ba185e222aa6740efa8f68407`, protocol git commit `9780a8c27d84f408240ed4b07062ab7275f02f09`), profile `qwen3.8-flash-next-thinking`, concurrency 1, 512-token output cap, **streaming**; the suite takes decode from the engine's own `timings.predicted_per_second` and prompt from `usage.prompt_tokens` over the stream.

All three rounds of both long-prompt cases hit the 512-token output cap (`finish_reason: length`) because the thinking model writes a long chain of thought first: that is why 7 of the 9 performance runs report `truncated`, and why the decode column here is a shorter burst than a 512-token answer suggests. The sanity pool ran separately and all 12 completed.

Suite quality gate: **12/12 passed** - instruction 4/4, retrieval 4/4, tool calls 4/4. Model files were re-hashed by the suite itself (shard 1 sha256 `4c1eb2ceb4915e11...`). Concurrency stages were not selected (recorded in the run's `missing` map, together with the draft protocol's unpinned model-card revision).

### House protocol, 4,096 / 32,768 / 128,000, 256-token output, 3 runs per depth

| target | measured prompt tokens | prompt tok/s (median) | decode tok/s (median, range) | draft accept | expert-cache hit | runs |
| --- | --- | --- | --- | --- | --- | --- |
| 4,096 | 4,188 | 777.2 | **76.8** (72.0-76.9) | 84.8% | 96.3% | 3 |
| 32,768 | 32,821 | 1,501.3 | **95.2** (91.3-96.1) | 92.9% | 96.2% | 3 |
| 128,000 | 127,940 | 1,695.7 | **90.3** (89.5-91.9) | 92.3% | 96.4% | 3 |

Across the three depths the decode medians average **87.4 tok/s** (median of the three depth medians 90.3). Every run was cold (`reused = 0`, `cached = 0`),
`temperature=0`, non-streaming, 256-token output, a different revision marker per run. The corpus is
repetitive MoE-routing prose, so expert-cache hit sits at 95-97% and draft acceptance at 85-93%; on the
frozen standard corpus (`strata bench run --suite standard-v1`, real prose) the same configuration decodes
57.9-58.8 tok/s at 4K and 16K. Quote the standard-suite numbers when comparing against other reports.

Resource envelope over the nine measured runs: VRAM peaked at 30 GiB across the two cards, engine RSS at 56 GiB, and `MemAvailable` fell to 10 GiB during the 128K runs - the box has about 10 GiB of slack at the deepest point, which is why the parked-conversation budget matters. Raw per-run data with VRAM, `MemAvailable`, engine RSS and cache-hit columns: [runs.csv](runs.csv),
[results.json](results.json).

### Production depth (the depths this box really serves: 193K and 238K prompt tokens)

| request | prompt tokens | reused | prompt tok/s | decode tok/s | draft accept |
| --- | --- | --- | --- | --- | --- |
| 193K cold, 512 out | 192,855 | 0 | 1,649.6 | 57.5 | 79.6% |
| 238K cold*, 512 out | 237,851 | 49,152 | 1,486.5 | 58.7 | 83.5% |
| 238K cached, 512 out | 237,857 | 229,376 | 577.4 (of 8,481 read) | 60.8 | 81.3% |
| 193K cached, 512 out | 192,861 | 180,224 | 574.1 (of 12,637 read) | 59.8 | 80.4% |

\* the 238K "cold" run reused the 49,152-token warm-up conversation as its prefix, so it is cold
only past the warm-up; reported as measured rather than relabelled.

Across eight repetitions of the *same* configuration at these depths, decode ranged
**53.6-60.8 tok/s**; that spread, not any setting, is what the "average decode" of a single run
hides.

### Long-context recall and images

- `needles.json`: **6/6** found, verbatim, at 31,412 and 133,046 prompt tokens with the needle at
  depth 10 / 50 / 90. (The first attempt scored 0/6 with empty answers because the 64-token
  budget was spent entirely in `reasoning_content`; the check was corrected to 512 tokens and to
  read both answer fields, and the raw first attempt is kept in `needles-promoted-first-attempt.json`.)
- `vision-probe.json`: a 320x256 synthetic PNG (sha256 in the file) over `/v1` came back
  *"A red circle and a green square sit side by side on a blue background."* in 5.58 s;
  `/health` reports `"images": true`.
- QA gate (`qa-fingerprints.md`): 6/6 at `temperature=0` with byte-stable content fingerprints
  (`1f7e13aed475, a0d04378f379, bd2af89bd428, 9166eeff1e50, 7719a1c782a1, 135debd48370`) —
  identical before and after the promotion.

### What moved the needle, and what did not

Full tables: [screen-arm-table.md](screen-arm-table.md) (21 arms under a warm protocol plus 5
under the first, un-warmed protocol, plus the levers rejected before measurement with reasons).

| lever | result on this box |
| --- | --- |
| `STRATA_PREFILL_RING=512` | **prompt +19.5%**: 1,393.7 -> 1,664.7 tok/s at 115,188 tokens, 1,373.0 -> 1,641.0 at 192,843, 1,350.9 -> 1,528.8/1,604.3 at 237,839; decode and acceptance unchanged; output path bit-identical upstream, answers identical in 16 arms; costs ~1.0 GiB pinned host RAM. R=256 is worse at 115K, R=1024 buys nothing further. Kept. |
| `STRATA_HIP_WMMA=1` | **prompt +18-23% at 32K/64K** (1,521.9 -> 1,795.2 and 1,621.0 -> 1,916.7), answers 6/6, decode -3%; and **no gain at all at 193K** (1,642.7 vs 1,649.6) - at production depth the prefill path is not GEMM-bound. Not kept. |
| parked-conversation cache (`--conversation-cache-slots`, `-min-tokens`, `-min-free-mib`, `-park-gain`) | best arm `slots 2 + floor 2560 + min-tokens 4096 + park-gain 8192` gained +3.6% decode on the 4K-64K screen (63.95 vs 61.73) with acceptance up ~4 points; at 193K/238K the same family measured *below* control. Not kept. |
| `--spec` / `--spec-min-p` | monotone loss above the default on this box: `spec 4` 58.38, `spec 6` 54.20 against 61.73 control; `--spec-min-p 0.35` 56.95. The vendor tuner's 3-round sweep prefers `--spec-min-p 0.7` (60.4/59.1/61.7 vs 57.0/60.0/57.9 for 0.5) on its own decode suite - unresolved here, see Not run. |
| DFlash2 drafter (stock-matched pack) | **acceptance up, speed down**: 92.4% at 4K and 88.5% at 32K (best of the sweep) but 56.98 decode mean against 61.73. The draft pass competes with the main pass on the same two cards. |
| `--draft auto` | 57.10 (the MTP path then defaults to spec 4, which this box dislikes). |
| `--mtp-q4` | acceptance up (94.9% at 32K), decode 61.50 = flat. |
| `--kv k8v4` | acceptance up (84.5% at 64K cold), decode 59.90 = down; also outside the INT8 constraint. |
| `--layer-split 26` | 60.38 and slightly slower prefill: the engine's equal 24/24 with its own `pcie_frac` is right. |
| `--pipeline-windows 3` | clamped back to 2 by the engine on a two-stage split (the load line prints 2), so the arm is a null test. |
| CPU placement (`--pool-workers`, `--aux-cpus auto`, one-sibling-per-core pinning) | 61.85 / 61.23 / 62.60 against 61.73: the built-in pool (one worker per physical core, host on the last) is already correct; the vendor sweep agrees (7 workers 60.7, 5 workers 60.9, 4 workers 56.6, 2 workers 51.5). |
| `--pcie-frac` (vendor exclusive sweep, 3 rounds) | 0 -> 59.4/59.3/57.9, 0.18 -> 60.0/60.3/59.2, 0.2 -> 60.3/60.1/59.4, 0.35 -> 58.2/58.6/59.1, 0.55 -> 53.3/52.8/53.8, 0.75 -> 50.4/50.8/52.0, 1.0 -> 42.4/42.3/43.2. The engine's automatic 0.18/0.54 is already on the optimum; the x4 card is why the far end collapses. |
| `--prefill-pipe R` (vendor sweep) | 826 off, 830 evened-out, 793 at R = 768/1536/3072/6144: nothing to take here. |
| `STRATA_SCORES_THREAD=1` | never engages: the HIP gate at `src/kernels/cuda/qsa_select.cu:1224-1236` accepts `gfx12` or `gfx1100` by name, and `gfx1101` resolves to 0 (`grep -c "qsa exact thread"` over the 86 MB engine log = 0). PR #5458's +10-13% is a `gfx1100` result. |
| `STRATA_PA_FAST=1` | +0.4%/+1.2% = noise on `gfx1101` (PR #5458 measured +13.82% on `gfx1100`). |
| concurrency (`--batch`/`--slots` 2/4/8, `--kv-resident` streaming) | measured the same day: 8 slots thrashed (solo probe 174 s to first token, `loaded=false`); two engines, one per card, decoded 41-43 tok/s per stream against 55-65 for one engine across both cards. |

### Why decode sits where it sits

`STRATA_DECODE_TIMING=1` prints the decode-window anatomy; 73 samples of >=100 windows
([decode-timing-excerpt.txt](decode-timing-excerpt.txt)):

```
strata decode timing: 318 windows, avg T 1.75, 1.61 tokens/window, 29.18 ms/window
  = verify 29.18 (GPU-reach wait 19.56 + per-layer host 4.08 + ...)
```

median **1.85 tokens per window**, median **26.0 ms per window**, of which **67.3% is
`GPU-reach wait`** and 12.5% is per-layer host work: an implied **74.7 tok/s** (range
23.5-96.6). Adding acceptance does not buy speed on this machine - DFlash2 and `--mtp-q4` both
raised acceptance and neither raised throughput - because the window, not the draft, is the
clock. What would change it is owning the fast PCIe path (both cards CPU-direct, or P2P), not a
flag.

## Limits and caveats

- Single-run arms on this box drift by +-5%, and between two rounds of the identical protocol the
  house medians moved +12% at 32K and -11% at 4K. Differences below ~8% from a single arm are not
  claims; the table marks which levers were re-tested by a second instrument.
- The filler corpus is repeated MoE-routing prose, so it selects a narrow expert set (expert-cache
  hit 92-97% during the house runs). The recall check, not these prompts, is the long-context
  correctness evidence.
- The first two A/B rounds inherited a wrong bytes-per-token constant (3.388 instead of the
  measured 4.75), so those runs sit at 2,946 / ~25K / 89,665 tokens; they are labelled as
  measured. The definitive run re-derives the constant from the warm-up reply.
- `--layer-split` is 24/24 with the engine's own `pcie_frac`; this box is asymmetric by design
  (x4 chipset arm vs x16 CPU-direct), so the numbers do not transfer to a symmetric pair.
- Reliability during this work: 0 `#267` pipelined-verify timeouts and 0 engine kills across the
  measured arms; across the whole 86 MB engine log there remain four historical kills, all
  `no progress for 60 s ... (reading the prompt)` at 65,991 / 97,230 / 120,133 tokens.
  No 48 h soak was run. `pp_dpm_mclk` tops at the active 1218 MHz and the amdgpu overdrive
  interface is absent (`pp_sclk_od`/`pp_mclk_od` missing, no `amdgpu.ppfeaturemask`), so GPU
  clocks were never raised; a lock needs a reboot.

## Validation

Commands, run on the box as user `dom` against `http://127.0.0.1:8080`:

```
python3.12 control/strata.py --version
python3.12 control/strata.py engine calibrate --profile <profile> --plan --json
python3.12 control/strata.py engine calibrate --profile <profile> --measure --exclusive \
        --out calib-01.json --json
python3.12 control/strata.py engine calibrate --profile <profile> --review-result calib-01.json
python3.12 control/strata.py bench run --profile <profile> --no-start --suite standard-v1 \
        --out /home/dom/strata-bench-20261107 --json
python3.12 benchmark.py <profile> <label> <out-prefix>          # house protocol, this folder
python3.12 qa.py <label> <profile>                              # six correctness prompts
python3.12 needles.py <profile> <label>                         # recall at 32K and 128K
python3.12 vision_probe.py <profile>                            # synthetic PNG over /v1
```

**Passed**

- Engine and control versions, engine/vision hashes, model shard sizes, mmproj hash.
- Engine argv at the promoted configuration: exactly one each of `--vision --kv --kv-resident
  --layer-split --max-context --spec --spec-min-p --prefill`, stock `IQ3_S-00001/00002` shards,
  no `abliterated` path anywhere, `--kv int8 --kv-resident 0 --max-context 262144` present,
  `STRATA_PREFILL_RING=512` in the live process environment.
- `/health`: `loaded: true`, `max_context: 262144`, `images: true`, model
  `qwen3.8-flash-next-iq3_s`; unit `active`, `NRestarts=0`.
- Correctness gates: QA 6/6 (byte-stable fingerprints), needle recall 6/6 at 31K/133K,
  vision probe correct, 0 `#267` and 0 kills in every measured window.
- `engine calibrate --plan` preflight: every required component `ok` (engine 0.1.58, tokenizer
  files, pack, native, ple-gguf, expert-profile, mtp, vision exe+mmproj+model, monitor_tui 8.2.8).

**Failed (and what happened)**

- Two harness bugs of mine, both disclosed and fixed: a literal `%` in a log-grep pattern raised
  `TypeError: not enough arguments for format string` (killed the first A/B), and a profile
  builder that filtered keys produced a config the launcher rejected with
  `run config needs a nonempty exe path`, which crash-looped the service for ~16 minutes until a
  full-copy profile was deployed. Neither affects the numbers above; both are why the first A/B
  round is reported as variance rather than as a comparison.
- The first needle attempt: 0/6 with empty answers (token budget spent in `reasoning_content`).

**Not run**

- A verified A/B of the vendor calibration candidate (`--spec-min-p 0.70`, `--pcie-frac 0.20`,
  `--adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97`) against the promoted configuration:
  the attempt that would have answered it deployed nothing (the `install` bug above), so its
  apparent -3% is drift, not a measurement. The candidate profile is kept on the box.
- A 48 h soak; GPU clock/power overdrive (needs a reboot); `iommu=pt`; NVMe NAND temperature
  under sustained load.
- Intel/SYCL routes (out of scope: a different machine and no oneAPI toolchain on it).
