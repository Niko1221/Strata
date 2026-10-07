# Community benchmark: 2x TITAN RTX (sm_75, NVLink), Xeon E5-2696 v4

Two consumer Turing cards, no AVX-512 on the CPU, and the model's native
262,144-token window. Reported per `docs/COMMUNITY_BENCHMARKS.md`: three runs
per configuration, medians and ranges, prompt and decode throughput kept apart,
and the recall check attached. Engine version 0.1.40.3.

> A note from the submitter, since numbers alone do not say why this was worth
> the weekend: I did not expect any of this to be possible. The whole point of
> the exercise was to see whether a 125B MoE could run at all on two consumer
> Turing cards, and it does — 71-77 tok/s of decode with the model's full
> 262,144-token window, on hardware that was never sold for this. Whatever you
> do with the engine, that part is genuinely surprising. Thank you.

## Hardware and software

| | |
| --- | --- |
| GPU | 2x NVIDIA TITAN RTX, 24 GB each (Turing, sm_75), driver 615.71.09 |
| GPU-GPU | `NV2` — 2 bonded NVLinks (`nvidia-smi topo -m`). **Not used**, see below |
| PCIe | x16, **gen 3** under load on both cards (gen 1 at idle), 220 W power limit each |
| CPU | Intel Xeon E5-2696 v4 @ 2.20 GHz, 22 cores / 44 threads, 1 socket, **no AVX-512** |
| RAM | 125 GiB |
| Storage | Samsung MZVLW1T0HMLH NVMe 1 TB (model), SATA SSD also present |
| OS | AlmaLinux 9.8, kernel 5.14.0-687.51.1.el9_8.x86_64 |
| CUDA | 12.9 (nvcc), source build |
| Strata | tag `v0.1.40.3` (commit `d5ea713`), built from source for `CMAKE_CUDA_ARCHITECTURES=75` |

Build used no non-default options beyond `-DSTRATA_ENABLE_CUDA=ON
-DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=75`. No prebuilt engine:
the release ships `strata-windows-x64.zip` only.

## Model and configuration

Qwen3.8-Flash-Next **IQ3_S** (3.5 bpw, the recommended quality tier),
`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, two GGUF shards
(54,817,524,224 + 28,800,138,432 bytes), plus the MTP draft layer packed by
`setup.sh` into `Strata-data/mtp/rt` (787 MB).

```
--pack Strata-data/packs/iq3_s --native ...IQ3_S-00001-of-00002.gguf
--ple-gguf ...IQ3_S-00002-of-00002.gguf --expert-profile data/expert-profile.bin
--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5
--mtp Strata-data/mtp/rt --max-context 262144 --kv int8 --kv-resident 65536
--vision --vram-reserve-mib 700 --ple-io ram --layer-split auto
--reasoning-budget-tokens 12000
```

`host 0.0.0.0`, `api_key` set (removed here). Calibration was not enabled and
the experimental speed projection was off. The vision encoder was compiled
locally for sm_75 (the ready-made encoder has no Turing code) and enabled in
the running configuration.

Observed after warm-up: 8,649 cached experts, 16.00 GiB of VRAM across
both cards' cache together. The 27.1 GiB n-gram table under `--ple-io ram`
is file-backed, so `free` reports it as page cache, not `used`.

## Method and reproduction

`benchmark.py` in this directory. Three runs at each of 4,096 / 32,768 /
128,000 target prompt tokens, 256-token output cap, `temperature=0`, non-
streaming. **Each run sends a distinct prompt** (a `Document revision marker`
differs per run) so no run reuses another one's prefix; a warm expert cache is
shared across runs by design, and the engine log's `expert cache NN% hit` is
recorded per run.

Prompts are a repeated paragraph of MoE-routing prose, so the filler itself
selects a narrow expert set; the recall check below is the correctness
evidence for long context, not these prompts.

Startup (~1 min 10 s, reading ~55 GB of experts into RAM) is **not** included
in any timing. Decode throughput comes from the engine's own
`timings.predicted_per_second`, which counts reasoning tokens; TTFT is
wall-clock minus generation time.

## Results

Prompt throughput rises with prompt length — the opposite of a degradation
curve, presumably from amortising expert-cache warm-up and larger prefill
chunks.

| Prompt tokens | Prompt t/s (3 runs) | Median | Decode t/s (3 runs) | Median | TTFT (3 runs) |
| --- | --- | --- | --- | --- | --- |
| 4,096 | 831.8 / 876.3 / 868.9 | 868.9 | 65.9 / 70.9 / 71.5 | 70.9 | 3.69 / 3.50 / 3.52 s |
| 32,768 | 1552.1 / 1548.9 / 1540.3 | 1548.9 | 77.3 / 79.2 / 77.0 | 77.3 | 15.97 / 16.02 / 16.10 s |
| 128,000 | 1636.7 / 1626.9 / 1621.7 | 1626.9 | 71.1 / 73.8 / 70.7 | 71.1 | 59.13 / 59.50 / 59.69 s |

Longer replies decode at the same rate as short ones: a separate 2,000-token
run at a 95-token prompt measured ~59-70 t/s with an expert cache hit rate of
95-99%, so decode is context-independent here.

Both GPUs work throughout: sampled `utilization.gpu` during decode was
**57% / 42%** with 195 W / 155 W and ~1900 MHz on both. The imbalance is the
auto layer split (one card carries slightly more layers plus the MTP draft).
Idle: both drop to 0% and ~1350 MHz.

## Recall and limitations

`needle_bench.py --lengths 32k,128k --depths 10,50,90`: **6 of 6 found**, no
misses or errors. Actual prompt lengths 32,343 (32k) and 125,918-125,920
(128k); wall-clock 13-21 s (32k) and 75-83 s (128k).

**NVLink is present but unused.** `nvidia-smi topo -m` reports `NV2`
between the two cards, but per `docs/MULTI_GPU.md` the engine deliberately
does not use NVLink or peer-to-peer access: activations cross cards through
pinned RAM once per verify window rather than twice per layer, so the same
numbers should be expected on cards with no bridge. Do not expect a gain from
adding a bridge on consumer boards.

Settings compared and a note on `--ple-io ram`: on this box the `--ple-io
ram` and `direct` routes measure the same prompt throughput (1596 vs 1595
t/s in the shared comparison we used for the earlier report), while the
`ram` route holds 27.1 GiB more resident page cache. Kept here only because
this machine has RAM to spare; on a smaller machine the default is the
better choice.

**Expert cache map.** 8,649 experts cached this run, 16.00 GiB of VRAM. The
config's `--kv-resident 65536` left room for it; the earlier 0.1.31 build on
this same box cached 10,080 experts at 18.17 GiB because the vision encoder
was not resident yet. Vision loaded: the text part of the 0.1.40.3 run ran
with the encoder resident, which costs a fixed slice of VRAM. During testing
the slot count settled at 8,649.

Not measured: the experimental speed projection and the low-RAM variant.
Vision is enabled in the graph but this report covers text-only runs.
