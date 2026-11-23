# 2026-11-23: one release event per prompt batch — RX 7900 XT, cold prompt reads +22% to +32% (PR #6813)

[PR #6813](https://github.com/Niko1221/Strata/pull/6813) replaces the fused-native prompt walk's per-slot `used`
event with one event per batch. On this machine that turns 1,515 tok/s into 1,996 tok/s on a 123K cold prompt
(**+31.8%**) on the IQ3_S pack, and 1,789 into 2,188 (**+22.3%**) on the IQ3_XXS pack. Output is unchanged.
The win is the patch plus the two existing env switches; the switches alone on 0.1.66 buy +1.5% to +4.8%.

## Machine and revision

AMD Ryzen 9 7945HX (16c/32t), 64 GB RAM, Radeon RX 7900 XT 20 GB (gfx1100), Windows 11 build 26300, WDDM,
driver 32.0.32029.2010, HIP runtime `amdhip64_7.dll`. Base revision v0.2.20 (engine 0.1.66); the patched side is
that tag built from source against ROCm 10.2 (TheRock) wheels with commit `65284f37cf` on top.

## Method

Each arm is one fresh engine process started from a launcher profile, one cold unique prompt, `max_tokens=8`,
greedy. The number read out is the engine's own line, `prompt N tokens = R reused + F read in T ms (X tok/s)`,
never the wall clock (which carries tokenization and generation). Prompts are seeded synthetic paragraphs sized
with a chars-per-token probe, so each arm lands within ~0.5% of the target size but the prompts are not
byte-identical across arms. These are not the frozen `standard-v1` fixtures, so this is not a standard-protocol
result; it is a before/after on one machine.

Launcher profiles were kept byte-identical apart from the engine path and the env flags under test. The learned
expert table was redirected to a throwaway copy in every arm, so synthetic routing never wrote the real one.
Stock and patched arms alternate through the session, so drift between the halves shows up rather than favouring
one build.

## IQ3_S pack, ~123K cold prompt

| Run | Engine / env | Prompt | Read | tok/s | vs shipped |
|---|---|---:|---:|---:|---:|
| K119-default | 0.1.66 shipped component, stock defaults | 123,062 | 81.2 s | 1,514.9 | — |
| P0-parity | source build, `STRATA_PF_SLOT_EVENTS=1` (old per-slot path) | 123,452 | 83.8 s | 1,472.9 | -2.8% |
| P1-batch | patch on, `COPY_WAIT_DEDUPE=1` | 122,947 | 74.3 s | 1,655.5 | **+9.3%** |
| P2-stack | patch on, `+ COPY_EVENT_EVERY=128`, `GPU_MAX_HW_QUEUES=8` | 122,952 | 61.8 s | 1,990.4 | **+31.4%** |
| P3-nohwq | patch on, `+ COPY_EVENT_EVERY=128`, no `GPU_MAX_HW_QUEUES` | 123,210 | 61.7 s | **1,996.0** | **+31.8%** |

Against its own build (P0 → P3) the ladder is +12.4% for the batch event, then +35.1% once the copied side is
sparsified. `GPU_MAX_HW_QUEUES=8` changed nothing once both were on, so the deployed configuration leaves it out.
Note the 2.8% gap between the two stock references: run-to-run drift on this box at this prompt size is about
that big, so read differences under ~3% here as noise.

## IQ3_XXS pack, 124,305 tokens: does it carry to a second pack

The same prompt text, same config shape (`--prefill auto:16384`), different pack, and this profile runs with
`STRATA_NO_LARGEPAGES=1`, so the RAM tier is pinned in ordinary pages here.

| Run | Engine | tok/s | RAM pin |
|---|---|---:|---|
| X1 | 0.1.66 shipped, stock | 1,791.7 | 5,549 ranges / 29,694 MiB, cap reached |
| X3 | same again | 1,786.8 | same |
| X2 | patch on | (1,831.7) | **436 ranges / 4,356 MiB, driver refused** — thrown out |
| X4 | patch on, dedupe + `EVENT_EVERY=128` | 2,187.6 | 5,549 / 29,694 MiB |
| X5 | same again | 2,189.4 | same |
| X6 | same again | 2,188.1 | same |

Stock mean 1,789.3, patched mean **2,188.4 (+22.3%)**, and the three patched runs agree to 0.08%. The gain is
smaller than on IQ3_S, most likely because the streamed bytes are smaller here (39.97 GiB of expert blobs vs
46.84 GiB for IQ3_S, so ~1.7 MB vs ~2.0 MB a blob) and the submission overhead is a smaller share of each batch;
the direction and order of magnitude are the same. X2 is the run that first looked like "only +2%": the driver
refused its RAM pin, and the number measured a different machine state.

## 36K cold prompt

| Run | Engine | Prompt | tok/s |
|---|---|---:|---:|
| R1 / R2 / R3 | 0.1.66 shipped, stock (`STRATA_PF_F16` 0 / default / 1) | ~36.6K | 1,321.3 / 1,332.4 / 1,325.4 |
| F2 | patch on, dedupe + `EVENT_EVERY=128` | 36,507 | **1,870.4** |

+41.0% against the mean of the three stock runs. The kernel switch (`STRATA_PF_F16`) moved nothing on this pack,
which is why all three are treated as stock. For scale: the same stock engine reads a 2,391-token prompt at
540 tok/s, and short prompts stay on the verify windows, where this patch is not on the path. The first request
after load on the XXS pack (an 11,372-token calibration probe) went from ~1,550 to ~2,018 tok/s, so the effect is
there well below 100K.

## What the env flags buy without the patch

| Run | Env on the shipped 0.1.66 | tok/s | vs stock |
|---|---|---:|---:|
| K119-default | none | 1,514.9 | — |
| E1 | `COPY_EVENT_EVERY=128` | 1,537.8 | +1.5% |
| E2 | `GPU_MAX_HW_QUEUES=8` | 1,437.9 | -5.1% |
| E3 | both | 1,587.5 | +4.8% |

Sparse copied events alone barely move, because the issuer still waits on one event per slot. Batching the
release side is what lets the dedupe and the sparse copies collapse; that is why the patch and the flags are one
change, not two.

## Tried and not proven

* `STRATA_RESIDENT_HEADROOM_GIB=2` gave 1,340.1 (-11.5%) and `+ --prefill auto:65536` gave 1,332.6 (-12.0%).
  Both runs pinned nothing (`complement ready: resident 42.45 GiB, pinned 0.00 GiB`), so the loss cannot be
  attributed to either knob on its own. Do not read these as "a bigger chunk is slow".

## Check the RAM pin before trusting a number

The residency line the engine prints at startup is the validity gate for every run on this box:

* `complement ready: resident X GiB, pinned X GiB` with the two equal — the RAM tier is fully locked;
* `N ranges registered (M MiB), the cap was reached` — normal with `STRATA_NO_LARGEPAGES`;
* `pinned 0.00 GiB`, or `the driver refused the next one` — the machine is not in the state being measured.

`bench_prefill_ab.py` prints that line in `compare` mode next to each number for exactly this reason.

## Output

Greedy `temperature=0` runs on identical prompts give byte-identical final content between stock and patched.
The reasoning trace differs — it also differs between two runs of stock itself on the same prompt (one token flip
in the content, 3,396 vs 2,037 characters of reasoning), because MTP draft acceptance and non-associating
floating point make the decode path non-reproducible on its own. The patch moves event timing, never the
arithmetic or the bytes.

## Reproduce

```
python bench_prefill_ab.py run patched --config PROFILE.json \
       --exe /path/to/patched/strata.exe --tokens 123000 --seed 20261126 \
       --env STRATA_PREFILL_COPY_WAIT_DEDUPE=1 --env STRATA_PREFILL_COPY_EVENT_EVERY=128
python bench_prefill_ab.py run stock --config PROFILE.json --tokens 123000 --seed 20261126
python bench_prefill_ab.py compare stock patched --out DIR
```

`results.json` holds every run in this folder with its env, prompt size, milliseconds and RAM pin;
`log-excerpts.txt` holds the two engine log lines each number came from.

## Limits

* One machine, one GPU, Windows WDDM only. The change compiles for the other CUDA/HIP paths but nothing here
  measures them: no Linux run, no CUDA, no SYCL or Metal, no multi-GPU, no split-arena run.
* Both packs measured are Swift IQ3 prompt reads on one 7900 XT; a pack whose experts mostly fit in VRAM has
  little streaming left for this patch to help with.
* Stock runs vary by ~3% at 123K within a session, so single-digit percentages in the tables are not separable
  from noise. The +22% and +32% results are three and four runs wide.
* The two negative controls above are confounded (see "Tried and not proven") and should be re-run separately
  before anyone quotes them.
