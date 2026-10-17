# Community benchmark: RX 6900 XT 16 GB (gfx1030) x1 and x2, Ryzen 5 5600X, 128 GB RAM - Strata 0.1.42: Unsloth UD-Q4_K_XL beside GSQ-RCO IQ3_S, one card and the layer split, stock and with the open gfx103x pull requests

Measured on 2026-10-10/11 on the Linux desktop of the [2026-10-09 (0.1.41)](../2026-10-09-community-2x-rx-6900xt-0.1.41/README.md)
report. It adds a second expert format to that machine's numbers and answers three questions:

1. What does **Unsloth UD-Q4_K_XL** (Q4_K / Q5_K / Q5_1 experts) do on RDNA2 next to **GSQ-RCO IQ3_S**, on one card
   and on the two-card layer split, with the engine built with `-DSTRATA_MMQ_KQUANTS=ON` - the flag PR #1897 asks
   setup to turn on for AMD?
2. What does **0.1.42 as released** do on this card family against 0.1.41, and what do the three open gfx103x pull
   requests (#1149, #1151, #1167) plus the two gfx103x switches add on top?
3. Do the llama.cpp RDNA2 MMQ patches that help the dense 27B carry over to Strata's expert GEMMs? (No - measured
   below, so nobody else has to try.)

The short answer: **stock 0.1.42 reads prompts at the same speed for both formats** (one card 471-476 / 504-505 /
488-489 tok/s at 4K / 32K / 128K, the split 462-464 / 763-766 / 873), because the stock prompt path is bound by the
dense FP16 route, not by the experts. The pull requests and switches double the one-card prompt (IQ3_S 959 / 1,153 /
1,129; UD-Q4_K_XL 676 / 1,117 / 1,087) and lift the split 2.1-2.3x (IQ3_S 985 / 1,792 / 2,046; UD-Q4_K_XL 756 /
1,748 / 1,981). **Decode is where the formats part**: UD-Q4_K_XL's experts are 53% larger (71.7 against 46.8 GiB), the caches hold about 40% fewer
of them, and it decodes 30% slower on one card (31-33 against 47-53 tok/s) and 35% slower on the split (45-51 against
75-82). 0.1.42 itself is speed-neutral against 0.1.41 on this machine (stock split 464 / 766 / 873 against 464 / 767 /
869), and under a fixed configuration its greedy output is byte-identical. No arm stalled.

For #1897's own numbers (one RX 6900 XT on PCIe 4.0 x16, UD-Q4_K_XL, 0.1.42 with the flag and
`STRATA_ROUTE_TAIL_SKIP=7`: 1,036 tok/s at 7.8K, 1,104 at 15.8K, decode 32-38): the same configuration on this x8
card reads 1,085-1,093 tok/s at 8K and 1,121-1,122 at 16K over three cold starts, decode 39-40 (the `e414/` folders).

## Hardware and software

- 2x AMD Radeon RX 6900 XT 16 GB (gfx1030, Navi 21), each on PCIe 4.0 x8 (the board splits one x16 slot). The
  one-card arms use the second card (no display attached, PCI 0f); the split arms use both with the vision encoder
  off (`NOVISION`), as the 0.1.41 report's split arms did.
- AMD Ryzen 5 5600X (6 cores / 12 threads, AVX2, no AVX-512), 128 GB DDR4-3200 (4 DIMMs, FCLK 1600).
- Ubuntu 26.04, kernel 7.0.0-38-generic, ROCm 10.0 (HIP clang from `/opt/rocm/core-10.0`), rocBLAS 5.6.0.
- Strata **0.1.42** (tag `v0.1.42`, `61b3fb5d`), two trees:
  - **stock**: the tag as is, built with `-DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1030
    -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ_KQUANTS=ON` (Release), the pinned upstream ggml
    (`3cf03257f`). Binary `91808c71...` (`BUILD.txt` in each `*-stock` folder).
  - **prs**: the same tag plus the three open gfx103x pull requests #1149 (FP16 prompt range), #1151 (the FP32 tiled
    QSA block scorer) and #1167 (rocBLAS solution table for the FP16 prompt GEMMs) and two small local commits
    (a per-device rocBLAS solution cache, its Windows path), same build options, same pinned ggml. Binary
    `90bf221c...`. This is the production tree of the machine except that production links a modified ggml, which
    E413/E414 below measure at +-1%.
- Server: `serve/server.py --engine strata` from each tree, Python 3.14 venv, port 8089, loopback.

## Model and configuration

- **GSQ-RCO IQ3_S**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S` (native pack `iq3_s`, PLE shard, MTP draft `rt`), 131,072
  context, `--kv int8 --kv-resident 32768 --ple-io ram`, `--expert-cache auto --prefill auto`.
- **Unsloth UD-Q4_K_XL**: `Qwen3.8-Flash-Next-UD-Q4_K_XL` (unsloth, revision `38bb39e`; native pack built with
  `tools/iq_pack.py --compat-bf16`; the PLE table found in shard 2; `--ple-io direct`), the same context and cache
  settings. The experts are 71.7 GiB in RAM (no `--resident-budget-gib`; the box has 128 GB), against 46.8 GiB for
  IQ3_S.
- One card: `--spec 4 --spec-min-p 0.5`, `gpu: 1`. Layer split: `gpu: [1, 0]`, `layer_split: auto` (the engine picked
  layers 0-23 / 24-47 for both formats), `gpu_order: as_given`; IQ3_S with the machine's production decode
  configuration `--spec 3 --spec-min-p 0.7 --pipeline-windows 2` (the 0.1.41 report's "decode window"), UD-Q4_K_XL
  with `--spec 4 --spec-min-p 0.5` and no pipeline windows (its only slot configuration on this machine).
- Expert cache slots (the same in the stock and prs arms): one card IQ3_S 4,861, UD-Q4_K_XL 2,831; split IQ3_S
  6,036 + 4,770, UD-Q4_K_XL 3,536 + 3,010. Decode expert-cache hit rates from the engine logs (4K / 128K requests):
  one card IQ3_S 72-79%, UD-Q4_K_XL 60-68%; split IQ3_S 91-97%, UD-Q4_K_XL 81-88%.
- **stock arms**: no environment variables at all (the engine's defaults; on HIP 0.1.42 leaves `STRATA_ROUTE_TAIL_SKIP`
  and the measured PCIe share off, so stock answers are the 0.1.41 answers).
- **prs arms**: `STRATA_HIP_PROMPT_F16=1 STRATA_SH_STREAM=1 STRATA_HIP_ADAPT_KERNEL_COPY=1 STRATA_SELECT_SGEMM=1
  STRATA_IO_THREADS=128 STRATA_PREFILL_CPU_SHARE=auto`, the rocBLAS table and cache of #1167, and on the split
  `STRATA_SPLIT_MTP_BATCH=1` (new in 0.1.42: the draft layer's prompt K/V batched through the prompt-path MMQ on the
  last stage; +4-7% prompt on this machine, measured in E413 with the production tree, decode and draft acceptance
  unchanged). The FP16 route and the batched draft K/V change the output bits; the pull requests do not beyond FP32
  summation order.

## Method

[benchmark.py](benchmark.py) is the script of the earlier reports (unchanged): deterministic synthetic Python filler
with a different nonce near the start of each request, counted with Strata's tokenizer. Every arm started its own
server from its tree and ran `benchmark.py --targets 4096,32768,128000 --runs 2` (two runs per length, serial,
increasing length, after one excluded warm-up). All speed requests read their whole prompt (zero reused tokens) and
generated 256 tokens to the output cap. Prompt throughput is freshly read tokens / `prompt_ms`, decode throughput the
engine's `decode_tok_s`; TTFT is streaming over loopback. `reasoning_effort: none`, temperature 0.

The eight servers ran one after the other between 00:20 and 01:30 local time on 2026-10-11 (one card first, then the
split); each folder has `config.json` (paths rewritten: `/work/stock`, `/work/prs`, `/m`, `/bench`), `summary.json`,
`results.json` (request hashes, every output text, the engine's per-request accounting) and `BUILD.txt` (tree, binary
SHA-256, environment, GFXOFF reading at start). The `e414/` folders are the 8K / 16K one-card UD-Q4_K_XL arms run the
evening before (`--targets 8192,16384 --runs 2`, `STRATA_ROUTE_TAIL_SKIP=7` added to match #1897).

## Results

Each cell is the median **[minimum-maximum]** of two runs; TTFT in seconds. No request failed, was cancelled or
stalled.

### One card (PCI 0f), stock 0.1.42 against the pull requests + switches

| arm | prompt 4K | 32K | 128K | decode 4K | 32K | 128K | TTFT 4K / 32K / 128K |
|---|---|---|---|---|---|---|---|
| IQ3_S, stock | 476 [471-481] | 504 [504-505] | 488 [488-488] | 45.1 [42.2-48.1] | 47.1 [46.2-48.1] | 46.2 [44.7-47.7] | 8.6 / 65.0 / 262 |
| IQ3_S, prs | **959** [943-975] | **1,153** [1,153-1,154] | **1,129** [1,128-1,130] | 47.7 [45.3-50.1] | 52.5 [51.0-54.0] | 47.4 [46.3-48.4] | 4.3 / 28.5 / 114 |
| UD-Q4_K_XL, stock | 471 [469-473] | 505 [505-505] | 489 [489-489] | 29.5 [28.9-30.0] | 31.3 [31.2-31.5] | 31.5 [31.4-31.5] | 8.8 / 64.9 / 262 |
| UD-Q4_K_XL, prs | **676** [670-682] | **1,117** [1,115-1,118] | **1,087** [1,086-1,087] | 31.6 [31.5-31.7] | 32.6 [32.4-32.9] | 32.9 [32.5-33.2] | 6.1 / 29.4 / 118 |

- Stock reads both formats at the same speed, within 1%: the stock prompt path's time is in the dense BF16 GEMMs
  (`STRATA_HIP_PROMPT_F16=1` is the switch the engine itself recommends at start), and the expert kernels are not
  the limit.
- With the pull requests the expert format shows: UD-Q4_K_XL's 4K prompt is 30% behind IQ3_S's (every prompt chunk
  streams the layer's whole expert set from RAM, and the larger set amortises worse over a short prompt), 32K and
  128K are 3-4% behind.
- UD-Q4_K_XL decodes 30% slower on one card: 2,831 slots against 4,861, a hit rate of 60-68% against 72-79%, and
  more bytes per expert.

### Layer split across both cards

| arm | prompt 4K | 32K | 128K | decode 4K | 32K | 128K | TTFT 4K / 32K / 128K |
|---|---|---|---|---|---|---|---|
| IQ3_S, stock | 464 [460-468] | 766 [763-768] | 873 [870-876] | 70.6 [64.2-77.0] | 72.4 [69.8-74.9] | 71.8 [71.5-72.1] | 8.9 / 42.8 / 147 |
| IQ3_S, prs (+ `STRATA_SPLIT_MTP_BATCH=1`) | **985** [962-1,009] | **1,792** [1,792-1,792] | **2,046** [2,040-2,052] | 76.6 [73.7-79.5] | 81.9 [81.0-82.9] | 74.5 [73.1-75.9] | 4.2 / 18.3 / 63 |
| UD-Q4_K_XL, stock | 462 [457-467] | 763 [762-765] | 873 [871-874] | 43.3 [40.3-46.3] | 45.6 [45.2-46.1] | 47.0 [46.8-47.2] | 8.9 / 43.0 / 147 |
| UD-Q4_K_XL, prs (+ `STRATA_SPLIT_MTP_BATCH=1`) | **756** [754-757] | **1,748** [1,745-1,751] | **1,981** [1,981-1,981] | 44.6 [41.3-47.9] | 51.1 [49.9-52.3] | 48.7 [48.5-48.9] | 5.5 / 18.8 / 65 |

- The stock split of 0.1.42 equals the stock split of 0.1.41 on this machine (464 / 767 / 869 in the 0.1.41 report):
  the release is speed-neutral here, as its notes say for AMD. The IQ3_S stock decode of 70-72 is the production
  decode configuration (`--spec 3 --spec-min-p 0.7 --pipeline-windows 2`) on the stock binary, not a 0.1.42 gain.
- The two formats read 32K / 128K prompts within 3% of each other once the pull requests are in; the 4K gap (756
  against 985) is the expert-stream amortisation again.
- UD-Q4_K_XL decodes 35% slower on the split (45-51 against 75-82); its arm also runs `--spec 4 --spec-min-p 0.5`
  without pipeline windows, which on IQ3_S is worth about 10% of the difference (the 0.1.41 report).

### Against #1897 and the llama.cpp RDNA2 MMQ patches (the `e414/` folders)

One card, UD-Q4_K_XL, the prs tree, `STRATA_ROUTE_TAIL_SKIP=7` added to match #1897's run, 8K and 16K prompts,
two runs each, three cold starts for the baseline:

| build | ggml | prompt 8K | 16K | decode 8K | 16K |
|---|---|---|---|---|---|
| prs, pinned upstream ggml (`e414/S`, three starts) | `3cf03257f` | 1,085 / 1,093 / 1,089 | 1,121 / 1,122 / 1,121 | 40.3-40.4 | 39.1 |
| prs, the modified ggml (`e414/O`) | `159c651f5` + RDNA2 MMQ tiles, mad24 scales | 1,082 | 1,113 | 40.4 | 39.0 |
| prs, the modified ggml with `GGML_CUDA_MMQ_Q8K` compiled into `strata_mmq` (`e414/Q`, two starts) | + Q8_K-style activations | 1,090 / 1,099 | 1,126 / 1,126 | 37.4-37.5 | 39.2-39.3 |
| #1897, one RX 6900 XT on PCIe 4.0 x16, Ryzen 9 5950X, ROCm 7.2.4 (the PR's table) | pinned | 1,036 (7.8K) | 1,104 (15.8K) | 32-38 | |

`STRATA_PREFILL_TIMING=1` on the 8K prompt of the baseline (GPU timeline 7.2 s): dequant 7.5%, gate/up GEMM 18.2%,
down GEMM 8.3%, wait-copy 1.6%; hc read 11.7%, GDN 25.0%, QSA attention 12.2%, QSA projections 6.5%, the rest small.
The expert GEMMs are a third of the prompt and the PCIe stream is hidden behind them. The llama.cpp patches that are
worth +5-11% on the dense 27B (the MMQ kernels are ggml's, so they compile into `strata_mmq`) move the gate/up GEMM by
-2% (tiles, mad24) and +5% (Q8_K activations) and the prompt by -1% / +0.5%, inside the run-to-run spread; the Q8_K
activations cost 7% of the 8K decode. Strata's expert GEMMs have far fewer rows per launch than the 27B's 512-row
micro-batches the patches were tuned on. The K-quant MMQ flag is the lever for this format on RDNA2; kernel tuning past
it is not.

## Stalls, memory, power

- Zero stalls in the eight arms and the e414 arms (`stall report` / `timed out at layer` / `no progress for`: 0 in
  every `engine.log`), including the stock split's 128K requests without `STRATA_HIP_ADAPT_KERNEL_COPY`. The GFXOFF
  knob of the card in use read 1 (the kernel default) at every start (the idle first card read 0 at one one-card
  start, `iq3s-single-stock`, a leftover of its previous use).
- VRAM at the end of each arm (`mem_info_vram_used`): one card 15.9-16.0 GiB on the card in use; split 15.9-16.2 GiB
  on the first card, 15.8-16.0 on the second (UD-Q4_K_XL's split arms the highest, 16,122 / 16,218 MiB).
- RAM: UD-Q4_K_XL's experts take 71.7 GiB pinned (`cudaHostRegister PORTABLE`), IQ3_S's 46.8 GiB; MemAvailable
  (sampled once a second) did not drop below 38 GiB in any arm.

## Limitations

- One machine, two quantizations, one context size (131,072), a small synthetic greedy workload with a 256-token
  output cap; no recall, vision, concurrency, sampling or thermal run. The 0.1.41 report's recall checks (54 of 54)
  were not repeated for UD-Q4_K_XL; its text and image answers were spot-checked when the slot was set up on
  2026-10-08, not scored.
- The two formats are not compared for quality here. The machine's own earlier check (teacher-forced KL against a bf16
  reference, on the dense 27B's quantizations, not these) is in the companion repository; for the two Flash-Next
  formats only speed was measured.
- The "prs" arm bundles the three pull requests, two local commits and the two gfx103x switches; only the bundle was
  measured. The one-card arms ran `--spec 4 --spec-min-p 0.5`, the IQ3_S split arms the production
  `--spec 3 --spec-min-p 0.7 --pipeline-windows 2`, the UD-Q4_K_XL split `--spec 4 --spec-min-p 0.5` without pipeline
  windows: the decode columns across formats on the split are not a pure format comparison.
- `--pipeline-windows 2` and `STRATA_SPLIT_MTP_BATCH=1` make the greedy output differ between server starts on this
  machine; users who need reproducible greedy output should leave both off.
- The #1897 numbers quoted are the PR author's, on a different host (x16, 5950X, ROCm 7.2.4, Arch) and slightly
  different prompt lengths; the comparison is for scale, not an A/B.

Developed with an AI coding assistant; every number above was measured on 2x RX 6900 XT (gfx1030, PCIe 4.0 x8 each) /
Ryzen 5 5600X, ROCm 10.0. The tuning history behind the "prs" arm, with the llama.cpp patches and the dense-27B
measurements the last section refers to, is at
[Vop3p/dual-6900xt-flash-next-tuning](https://github.com/Vop3p/dual-6900xt-flash-next-tuning).
