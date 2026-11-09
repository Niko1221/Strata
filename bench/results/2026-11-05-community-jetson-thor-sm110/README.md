# Jetson AGX Thor (sm_110): the three documented models, compatibility and speed

Runs 2026-11-04 to 2026-11-08 on one **NVIDIA Jetson AGX Thor** (L4T R38.2.2, Ubuntu 24.04.4, `6.8.12-tegra`, aarch64,
122.8 GiB unified memory, ~117 GiB usable by the GPU; driver 580.00, CUDA 13.0.48; model files on the NVMe
WD PC SN5000S 1 TB). Engine: **0.1.56 built on this machine from `f-4559-thor-sm110`** - the arm64 fat CUDA
package carrying `sm_121 + sm_110` machine code and `compute_110` PTX ([docs/JETSON_THOR.md](../../../docs/JETSON_THOR.md),
issue #4559). Terminal: the repository's own (`strata.py`, dev checkout at `5819cc0b`).

Every standard-suite and dgx-style cell in this report has its runner-produced files in the report itself: the
`iq3s-32k*/`, `qwen36-32k/`, `maxn/*`, `normal/*` folders carry `results.json`, `outputs.jsonl` (or `base.jsonl`),
`config.json` and `provenance` (`base.summary.md`/`base.meta.json` for dgx runs), as the runners wrote them. Only the
engines' own verbose logs are left out (hundreds of MB of slot dumps); the numbers in every table are recomputable
from the included files.

## Compatibility, model by model

| Model | Pack | Runs on the Thor | Correctness | Speed |
| --- | --- | --- | --- | --- |
| Qwen3.8-Flash-Next | ISTA-DAS IQ3_S (84 GB) | **yes** - 4 slots, context 32768 | standard suite: **sanity 12/12** | below |
| Qwen3.6-35B-A3B | Unsloth MTP **UD-Q4_K_XL** (22.9 GB) | **yes** - 4 slots, context 32768 | standard suite: **sanity 12/12** | below |
| Qwen3.6-35B-A3B | UD-Q6_K_XL, Q8_0 | not installed: the same path and file layout as UD-Q4_K_XL, but each pack is a further 33/38 GB download and this machine's link runs ~1.3 MB/s | - | - |
| DeepSeek-V4.1-Flash-1017 | ISTA-DAS **Coder** set (`...-GSQ-RCO-Coder-IQ2_XXS`, 191 GB download incl. shared base) | **yes** - served at 8K and 64K context (its own engine package, experts split GPU/CPU lane) | smoke: arithmetic and long completions correct; the qwen-specific 12-check protocol is not its protocol | below (dgx-bench cells, not the standard suite) |

Details worth recording:

- Both served models loaded from their GGUF/pack and answered chat, tool-call and retrieval requests; the
  engine's K-quant prompt library `libstrata_kq.so` (in the fat package, `sm_110` cubin) loads automatically
  (`K-quant prompt kernels from ... (STRATA_MMQ_KQ=1)`, `prompt experts on MMQ (int8)`).
- **Qwen3.6 at context 128K with 4 slots does not fit this machine**: the engine itself refuses with exact
  numbers (`not enough VRAM for 4 slots at a context of 131072 (5.56 GiB each, 0.78 GiB free)`). 32K with
  4 slots runs. A smaller context or fewer slots is the same trade Strata states for other unified-memory PCs.
- Tegra counts reclaimable page cache against the GPU pool (`cudaMemGetInfo`): after a previous 78 GB model
  run the free figure stayed at ~3.4 GiB and the engine refused; `posix_fadvise(DONTNEED)` over the model
  files as their owner freed ~44 GB and it started (reboot or `drop_caches` need root; the fadvise trick does
  not). Same root cause as the `expert_cache_segmented_test` miss on this machine.
- DeepSeek-V4.1-Flash: its engine package (`strata-dsv41` 0.1.56, `sm_121 + sm_110`) builds, verifies
  (`package_engine.py --verify`, `binary_deps.py`), is selected by `strata update` against a feed carrying it,
  and prints its version inside the official CUDA 13.0 container on the Thor. The fit probe said the set fits
  and it did: the Coder set (191 GB incl. the shared base; the link had improved to tens of MB/s by 2026-11-07,
  download plus pack conversion in about three hours) served on the Thor with no trouble. Its own placement
  line at load: `ds4: CUDA0 layers [0, 39] + head: weights 5.70 GB, experts 35.29 GB (4054 of 7680 in VRAM),
  caches 0.01 GB`; the other 3,626 routed experts (31.3 GB) run in place on the CPU lane. Memory is not tight:
  about 41 GB of pack, roughly 41 GB resident beside the KV, and the CPU lane reads the model's files through
  the page cache. One note from the runs: the 32K cell needs a context above 32768 (the calibrated 32,774-token
  prompt overflows a 32768 context by six tokens and the server correctly answers 400), so the 16K/32K cells
  ran at 64K context. Speed cells, one protocol of its own (dgx_bench, T=0, medians of 2): 8K context cold
  expert cache `maxn/ds-coder-8k.md` (short chat 87.6 prompt / 8.8 decode, 4K prompt 182.7 / decode 10.2,
  TTFT 22.2 s); 64K context warmer cache `maxn/ds-coder-64k.md` (16K prompt 330.0 / decode 17.2, 32K prompt
  356.4 / decode 14.7, TTFT 92.2 s). The CPU-lane split is why its decode sits near 9-17 tok/s against the
  13-23 tok/s the Spark guide prints for its resident-heavy DeepSeek placement: this set keeps 53% of its
  experts on the GPU. Correctness smoke: `17 crates of 23 tools` answered `391`; the qwen-thinking sanity
  protocol is not this model's protocol, so its 12 checks were not applied.

## Speed: the standard protocol suite (`strata bench run`, protocol `strata-qwen38-thinking-v1-draft.1`)

Both runs used the frozen `standard-v1` corpus (SHA `e75eff8b...`), thinking on, the model card's sampling,
three repeats of short / 4K / 16K prompts, the twelve sanity checks and the optional four-at-once rounds.
Cells are median (min-max) n=3; prompt/decode rates are the engine's own timings, TTFT the client's.
Track: custom. Not cold-cache: the expert cache warms within the run, as the protocol notes.

**Qwen3.8-Flash-Next IQ3_S, context 32768, 4 slots** ([results.json](iq3s-32k/results.json), [report](iq3s-32k/README.md)):

| Case | Prompt tok/s | Decode tok/s | TTFT s |
| --- | ---: | ---: | ---: |
| short | n/a | 38.3 (36.4-40.3) | 0.96 |
| prompt-4k | 769.0 (732.6-834.6) | 34.0 (32.5-34.1) | 5.37 |
| prompt-16k | 1000.2 (910.9-1010.5) | 35.6 (33.9-39.2) | 16.44 |

Sanity **12/12**. Four requests at once: 51.0 / 52.9 / 54.4 tok/s in total per round.

**Qwen3.6-35B-A3B UD-Q4_K_XL, context 32768, 4 slots** ([results.json](qwen36-32k/results.json), [report](qwen36-32k/README.md)):

| Case | Prompt tok/s | Decode tok/s | TTFT s |
| --- | ---: | ---: | ---: |
| short | n/a | 50.7 (50.7-51.0) | 0.14 |
| prompt-4k | 1375.1 (1370.1-1378.4) | 37.4 (37.4-37.4) | 2.96 |
| prompt-16k | 1221.0 (1218.9-1227.0) | 20.2 (20.1-21.7) | 13.41 |

Sanity **12/12**. Four requests at once: 100.5 / 100.5 / 100.8 tok/s in total per round - the Thor serves four
Qwen3.6 requests at once at roughly the speed of one IQ3_S request.

A side note, not fitted away: Qwen3.6's decode at prompt-16k (20.2) sits well below its 4K decode (37.4) on
this machine, where IQ3_S stays flat (~34-36). One machine, custom track, three repeats each - worth watching
on other ARM64 boxes rather than presenting as a settled number.

## Arm key: what the A/B files' labels mean

`tools/ab_engine.py` names arms A, B, C... on its command line. Two of these sweeps still have their runner scripts
on the machine and the engine's own startup line confirms them; the Nov 4/5 switch sweeps ran from throwaway
scripts, so their mapping is reconstructed from the contemporaneous notes and stated as such. To close the gap for
the future, `ab_engine.py` on this branch now writes a manifest line naming the config and every arm's full spec,
and repeats the arm spec inside each record - new A/B files are self-describing.

| file | arms | basis |
| --- | --- | --- |
| `ab-q36-depth.jsonl` (120 W) and `ab-q36-depth-maxn.log` (locked clocks) | A=`--mtp 0` (drafting off), B=`--mtp 2` (the shipped default), C=`--mtp 3`, D=`4`, E=`5`, F=`6`, G=`7` | proven: runner script kept on the machine (`A:xargs=--mtp;0 ... G:xargs=--mtp;7`), and the engine startup line in the per-arm logs shows the appended argument (e.g. arm C's log: `... --prefill auto --mtp 2 --mtp 3`) |
| `ab-q36-dv.jsonl` | A/B/C: the shipped launch, the same with `STRATA_Q36_DRAFT_VOCAB` set, and it with the draft-vocab size varied | reconstructed from contemporaneous notes; the knob was found useless, which is why the mapping is thin |
| `ab-iq3s-dropone.jsonl` | five arms, each one of the five switches removed from the full table (`STRATA_SPEC_COST` / `STRATA_HC_Q8` / `STRATA_PF_FUSED` / `STRATA_HC_UPMIX` / `STRATA_SIDE_MIXER` set to `0` over the table-on baseline; IQ3_S at `--spec 4`) | reconstructed: the per-switch numbers each drop produced are exactly the rows of the table below; the per-arm launch specs themselves were not archived (the tool gap above) |
| `ab-iq3s-stack.jsonl` | five arms, the table built up switch by switch from `STRATA_SM110_PERF=0` | reconstructed, same basis |
| `ab-iq3s-spec.jsonl`, `ab-iq3s-decide.jsonl` | fixed draft-window depths (`--spec 2/4/6/8`) plus the cost-rule arm (`STRATA_SPEC_COST=1`), and the final keep/drop decision pair | reconstructed from contemporaneous notes |

Anyone reproducing these must use the engine from this branch's package (`sm_110` cubins); the `STRATA_SM110_PERF=0`
switch is the table-off baseline, and a switch set explicitly is kept by the engine ("a switch you set is kept").

## The sm_110 speed table, measured switch by switch (added on this branch)

Five switches went into `nv_defaults.cpp` for cc 11.0, each measured on this machine with
`tools/ab_engine.py`, balanced pairs, medians of n=2-3 (`ab-iq3s-stack.jsonl` build-up,
`ab-iq3s-dropone.jsonl` drop-one, `ab-iq3s-decide.jsonl` final decision; IQ3_S 32K, thinking off,
256-token caps - the pairs compare switches, not the suite):

| Switch | Measured here | In the table |
| --- | --- | --- |
| `STRATA_SPEC_COST=1` | at `--spec 4`: every case faster or equal (chats 1.041/1.000/1.014, 6K prompt 1.031); fixed windows lost at depths 2 and 6/8 | yes |
| `STRATA_HC_Q8=1` | drop-one: chat 47.5 to 44.4 tok/s, 6K prompt prefill 850 to 790 | yes |
| `STRATA_PF_FUSED=1` | drop-one: 6K decode 36.3 to 32.5, prefill 850 to 764 | yes |
| `STRATA_HC_UPMIX=1` | 3 switches to 4: chats +1.8/+2.7/+1.3, 6K prefill +56; bit-identical | yes |
| `STRATA_SIDE_MIXER=2` | decode inside the noise, 6K prefill +128 (851 vs 723) | yes (for the prompt) |

`STRATA_SM110_PERF=0` turns the set off; `STRATA_DEFAULTS=026` leaves it out, as for sm_121. The engine says
what it set: `strata: sm_110 (Jetson AGX Thor): 5 speed switches on by default (...)`. End to end against
the as-shipped run of the same suite above ([iq3s-32k-tuned/results.json](iq3s-32k-tuned/results.json),
sanity again **12/12**): prompt **+20.3% / +9.5%** (4K/16K), first answer at 4K **-27%**, decode **+7.6%**
at 4K and **-6.2%** at 16K (ranges barely overlap: a real small loss at 16K thinking-on, bought back with
interest by the prompt), four-at-once rounds 48.6 / 51.6 (pool partly stale at that run).

## Drafting depths on this part (Qwen3.6 and IQ3_S)

Speculation is the engine's built-in MTP draft + full verification; depths swept paired
(`ab-q36-depth.jsonl`, `ab-iq3s-spec.jsonl`, thinking off, 256 caps):

- **Qwen3.6 UD-Q4_K_XL**: the shipped `--mtp 2` is the slowest setting here - 53.0-53.9 tok/s against
  54.1-54.5 with drafting off; deeper drafts win up to **`--mtp 7`**: 55.7/56.4/56.7 (chats) and 34.7
  (6K prompt), about +5% over the default. `STRATA_Q36_DRAFT_VOCAB=8000/20000` did not help
  (52.4/51.9/51.4). `--batch-mtp` not swept at 1 slot.
- **IQ3_S**: `--spec` 2/4/6/8 peaks at 4 both ways, and the cost rule (in the table) beat plain 4 in every
  case - which is why the depth stays 4 and only the rule is shipped on.
- **DFlash2** (its drafter fetched and packed on this machine): runs correctly - standard suite sanity
  **12/12** ([iq3s-32k-dflash2/results.json](iq3s-32k-dflash2/results.json), model hash skipped) - but
  decodes slower than the model's own MTP layer: 35.1/30.9/31.6 against 38.3/34.0/35.6. So `launch` keeps
  MTP on sm_110 and `drafting._TESTED` stays without sm_110, now with numbers rather than an absence.

## The pool number again, with the leak counts

The `cudaMemGetInfo`-after-exit behavior above was chased further on 11-05: right after a big engine run it
reported 1.8-5.1 GiB free of 122.8 while `MemAvailable` was ~76 GB; a restart restores 117.4. 103 orphaned
`launch-demo` runtime hosts were found on the running machine, each holding a small CUDA reservation
(~3.3 GiB in total, released on kill). Engines size slots, expert caches and prompt chunks from that
number, so on this device the effects were: a second `strata launch` refused (`0.78 GiB free`, exact quote
above), and Qwen3.6's free-based chunk/expert sizing cut single-stream decode to 11-17 tok/s with the
whole model in page cache (the healthy 50.7/37.4 above were taken with a fresh pool). `--parallel 1` gets
past the slot precheck (it only fires for 2+ sessions) but not the sizing. Not fixed here; named so the
precheck/sizing can be judged against `MemAvailable` on Tegra by whoever owns that code.


## Clock modes: MAXN measured against the 120 W default (same branch, same tuned engine)

With `sudo nvpmodel -m 0 && sudo jetson_clocks` the GPU runs pinned at 1575 MHz (120 W mode caps the GPC rail at 1386
and idles at 315), the memory domain at 1692 instead of 1557, the CPUs at 2601. Against the 120 W DVFS runs of the same
protocol suite: iq3_s short decode 37.3 to **47.4 (+27%)**, 4K decode 36.6 to 40.2, 16K decode 33.4 to **40.5 (+21%)**,
prompt +13 to +16%, TTFT -14%; Qwen3.6 short 50.7 to 59.3 (+17%), 4K 37.4 to 43.5, 16K 20.2 to 25.2 (+25%), prefill
+15 to +16%. Decode moves because the memory-domain clock moves too - this is the cheapest Thor speed-up there is, and
it survives on 120 W hardware only as a DVFS gamble. The 120 W half ran 2026-11-07 14:26/14:37Z, the MAXN half
2026-11-05 22:56/23:06Z - a day and a recorded terminal commit apart, not the same night: same scripts, same
engine-package lineage, pool cleaned by fadvise before each phase; both suites sanity 12 of 12, both logged
**nonconforming** for one protocol point only (the running model file could not be hashed). The 11-07 half's
recorded commit `5819cc0b` predates the speed-table merge, so its run record alone does not establish its table
state; the engine binaries of those dates came from the bench tree, which carries the table
([normal/iq3s-64k.md](normal/iq3s-64k.md), [normal/q36-128k.md](normal/q36-128k.md)):

| | iq3_s short / 4K / 16K decode | iq3_s 4K / 16K prompt | iq3_s TTFT 4K | q36 short / 4K / 16K decode | q36 prefill 4K / 16K |
|---|---|---|---|---|---|
| 120 W DVFS | 41.3 / 35.9 / 36.5 | 973 / 1,128 | 4.25 s | 53.7 / 39.3 / 22.5 | 1,420 / 1,272 |
| MAXN pinned | 47.4 / 40.2 / 40.5 | 1,073 / 1,241 | 3.85 s | 59.3 / 43.5 / 25.2 | 1,586 / 1,412 |
| MAXN gain | +15 / +12 / +11% | +10 / +10% | -9% | +10 / +11 / +12% | +12 / +11% |

At the dgx cells the same shape holds for q36 (+10 / +10 / +9% prompt, +9 / +12 / 0% decode) and iq3_s
(+9 / +14% prompt at 4K / 16K, decode +6 to +12%); one honest exception: the 32K cell came out slower at
MAXN in this pair (950 against 1,051) while the 16K cell gained - a two-round cell at the machine's
largest size, likely page-cache state over the 78 GB pack rather than the clocks, and it is left stated
rather than averaged away. The earlier +27% headline was a cross-session comparison (a pre-table, partly
stale-pool run against a fresh one); this spaced pair is the one to quote, and it can only be cross-session -
a clock state changes the whole machine, no session can hold both halves. Quote it as "MAXN + jetson_clocks:
+10 to +15% across the board on this part, cross-session, same scripts and engine package" - the cheapest
Thor speed-up there is.

The same tool at the same cells as the Spark guide (`tools/dgx_bench.py`, 64K context, temperature 0, 256-token
answers; [maxn/iq3s-64k.md](maxn/iq3s-64k.md), [maxn/q36-128k.md](maxn/q36-128k.md), [maxn/nvfp4-64k.md](maxn/nvfp4-64k.md)):

| Cell (prompt tok/s) | Thor MAXN iq3_s | Thor MAXN q36 | Thor MAXN NVFP4 | Spark iq3_s (guide) |
|---|---:|---:|---:|---:|
| 4K | 921 | 1757 | 704 | 1,296 |
| 16K | 1,138 | 1,584 | 761 | 1,680 |
| 32K | 950 | 1,339 | 767 | 1,735 |
| decode 4K/16K/32K | 37.6 / 37.5 / 38.0 | 43.2 / 25.0 / 13.2 | 27.1 / 28.9 / 27.5 | 45.5 / 46.5 / 46.5 |

So: 0.71 / 0.68 / 0.55 of the Spark's prompt speed by 32K, 0.82 of its decode. The 32K cell is where Thor falls off
most (KV bytes on the same shared bandwidth).

## NVFP4 on Thor: works, but it is not a speed-up here

The Terminal's `--family nvfp4 --model NVFP4` pack (Strata's GPTQ conversion, 128 GB, downloaded whole on this machine)
loads and answers: standard-suite sanity **12 of 12** at context 64K. Speed is worse than IQ3_S on the same machine
(MAXN): prompt 704/761/767 against 921/1138/950, decode 27.1/28.9/27.5 against 37.6/37.5/38.0. The package does carry
the `quantize_mmq_nvfp4` kernels for sm_110, but MODELS.md's own sentence - RTX 50 cards run FP4, other cards run
NVFP4 at 8-bit - holds for the Thor in practice: no FP4 fast path measured here. Keep IQ3_S on Thor.

## Home-cooked packs: what a Thor-specific quantization recipe buys (and what it costs)

Every decode bound on this machine was computed from the pack's real per-layer formats
([quant/iq3s-bound-thor.txt](quant/iq3s-bound-thor.txt) at 273 GB/s unified, all experts resident): IQ3_S reads
982 MB of routed experts + 4,166 MB of dense weights per token -> **hard ceiling 53.0 tok/s**, of which the MAXN runs
already reach 38-40 (72-76%); Qwen3.6's ceiling is 106.7. Experts are only 19% of IQ3_S's per-token bytes - so the
community "shrink the experts" recipe cannot double anything, and a wholesale llama-quantize Q2_K pack even grows the
file (74.8 GiB vs 77.9) because Unsloth's per-layer expert mix (2.05 bpw) is leaner than stock Q2_K (2.6).

Measured with the repo's pinned llama.cpp (3cf0325) built for sm_110, one model at a time (the machine wedged its
network stack when two 78 GB loads overlapped - do not overlap loads on this platform):

| Pack | file | tg128 (tok/s) | PPL, same 250 KB corpus | verdict |
|---|---:|---:|---:|---|
| IQ3_S as-is | 77.9 GiB | 21.6 | 2.258 | baseline |
| thorA: stock Q2_K everywhere | 74.8 GiB | 28.8 (+33%) | 3.028 (+34%) | not worth it |
| thorB: Thor recipe ([quant/thorB-args.txt](quant/thorB-args.txt)) | 80.7 GiB | 26.2 (+21%) | 2.312 (+2.4%) | the shape that works |

thorB keeps routed experts at their stock formats (per-layer `--tensor-type blk\.(...)\.ffn_gate_exps\.weight=...`
lists - llama.cpp has no `keep` at this pin, so each format group is written back explicitly), keeps the 26.8 GiB
PLE table at IQ4_NL, and instead cuts the dense side: hyper-connections BF16 to Q8_0, attention and the SSM output
and the head to Q5_K. +21% decode for 2.4% perplexity, and it points where a Thor pack should go next (the dense
layers carry 81% of the bytes). Note llama.cpp's tg sits far below the Strata engine's decode (Strata's PLE and
expert-cache path is the reason); use ratios across packs, not the absolute.

Method notes for llama.cpp on Thor: `-ngl 99` needs the project built with GGML_CUDA on a CMake that knows CUDA 13
(the system cmake 3.28 worked once CUDACXX was set and the arch list was `110-real;110-virtual`); the perplexity
tool at this pin has no dataset download, so the corpus is the bench's own answers
(`quant/ppl-corpus.txt` is reproducible from the outputs.jsonl files).

## The two follow-ups, both clock-paired

Pack ratios at 120 W DVFS (same llama-bench cells as the MAXN runs): iq3_s 20.4 tg / 348 pp; thorA 26.5 (+30%) /
306; thorB 24.9 (+22%) / 331. The recipe gains (+30 / +22% decode) hold at both clock states - the lever is bytes
per token, which clocks cannot change. ([quant/FIN120-*-bench.txt](quant/))

Drafting depths re-swept for Qwen3.6 at MAXN ([ab-q36-depth-maxn.log](ab-q36-depth-maxn.log), paired arms, two
rounds): the picture changes with the clocks. At 120 W deeper drafts paid off to depth 7 (+5% over the default);
at MAXN every depth from 3 to 6 sits level with drafting **off** (chats 58.6-59.2 tok/s), depth 7 fades slightly,
and the shipped default 2 remains the worst cell by 5-7 tok/s - a 128 K-context engine spending its extra
compute on attention rather than on draft gains. The one setting that survives both clock states is the advice:
do not leave Qwen3.6 at the default 2 on a Thor; 3 is the small winner (best long-prompt decode too), off is
fine at MAXN.

## Reproducing

Same machine and package: the fat engine comes from `ARCHS="121;110" sh spark/build_engine.sh` (or
`packaging/build_cuda_linux_aarch64.sh`); serve with `strata launch --model <size> [--family qwen36]
--context 32k --vision off --drafting mtp --tune skip --yes --no-tui` and measure with `strata bench run
--url http://127.0.0.1:8080 --suite standard-v1 --optional-concurrency --out <dir>`. The container path
(official `nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04`, `--runtime nvidia --ulimit memlock=-1`) reproduces
the build and serves the same way; docs/JETSON_THOR.md has the recipe and the function-for-function match
between the container and host binaries.

## Long context: IQ3_S at 128K and 256K ([longctx/](longctx/README.md))

Added 2026-11-09, MAXN power mode, same package: **IQ3_S serves 128K context (2 or 4 slots) and 256K (1
slot)** with decode flat at 36-38 tok/s from 64K through 256K context - the 64K MAXN cells' decode carries
unchanged to a 131K-token prompt (256K cell: prefill 1117.8, decode 38.2, TTFT 119.3 s). Four slots at 128K
cost ~2 tok/s. The five sm_110 switches, `--spec 8`, `--kv-resident 131072` and the cost rule off all sit
inside noise at T=0 long-prompt cells - the defaults are the setting there (the first -30-50% reads were
cold-page-cache artifacts, caught and re-paired; see the section's trap note). The q36 128K refusal above is
that pack's own KV sizing, not a platform limit.

## Not covered here

Vision (this package ships no picture reader), DFlash2 as a default (measured slower than MTP here, above),
DeepSeek concurrency and Q6/Q8 packs for Qwen3.6 (bandwidth-limited), more than one module, a long soak, and

