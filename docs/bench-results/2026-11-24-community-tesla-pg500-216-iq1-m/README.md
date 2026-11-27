# Community benchmark: Tesla PG500-216 (GV100GL, 32 GB) on a GMKtec UM780 — Linux, PCIe **Gen3 x4**, `STRATA_PF_WMMA` A/B

Measured on 2026-11-24 by Yonneh0. The protocol standard suite (`strata bench run --suite standard-v1`,
A/B interleaved) plus fresh ~51K-token 64K prefills and the `needle_bench.py` recall battery on a card the
existing [V100 document](../NVIDIA_V100.md) does not cover: the **PG500-216** is a passive GV100GL carrier
with 32 GB HBM2, and this box wires it at **PCIe Gen3 ×4** (`lspci` `LnkCap` x16 / `LnkSta` **x4 (downgraded)**,
≈3.4–3.9 GB/s) inside a 30.6 GiB mini-PC — the hardware class requested in [TEST_REQUESTS.md](../../TEST_REQUESTS.md)
(#1194: a Linux host with ≤32 GB RAM, where the resident mode + file tier decide how much of the model fits).

**Headline:** IQ1_M (2-shard GSQ-RCO, ~58 GB of experts) on 32 GB HBM2 + 30.6 GiB RAM reads a standard-v1
16K prompt at **1,519 t/s** and decodes at **64–78 tok/s** with the DFlash2 drafter; at fresh ~51K-token
prompts `STRATA_PF_WMMA=1` + `STRATA_PF_WMMA2=1` measured **1,703 t/s vs 1,477 t/s baseline (+15.3 %,
same-day B→A→B, the two B arms agreeing to 0.02 %)**. A single card at a **×4** link at its full 250 W boost
**beats the dual-V100-SXM2 ×16-NVLink pair locked at 200 W/1,020 MHz (1,644 t/s)** in NVIDIA_V100.md.
The ×4 link is not the limit: at 64K the phase profile attributes ~3 % to card↔host streaming.

## Hardware and software

- GPU and VRAM: Tesla PG500-216 (`10de:1df0`, GV100GL, sm_70/CC 7.0, 32 768 MiB HBM2, max SM 1 380 MHz);
  CPU: Ryzen 7 7840HS (Zen 4, 8C/16T, AVX-512 incl. `avx512_bf16`/`avx512_vnni`); RAM: 2×16 GiB DDR5-5600
  (30.6 GiB usable); storage: 937 GB NVMe (81 % used); PCIe: **Gen3 ×4** — `LnkCap: Speed 8GT/s, Width x16`
  vs `LnkSta: Speed 8GT/s, Width x4 (downgraded)`
- OS: Ubuntu 26.10 (kernel 7.3); driver 580.178.04; CUDA 12.9.2 (`nvcc V12.9.86`)
- Strata: main downloaded 2026-11-24 (no VCS metadata; docs state release 0.2.21, engine built from the tree
  reports **0.1.68**); **source build**: `cmake -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70
  -DSTRATA_EXPERIMENTAL_SM60=ON`, gcc 14.4.0, plus `tools/vision` (`-DSTRATA_VISION_CUDA=ON
  -DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_PORTABLE=OFF`)
- Background workloads: none during measurement (the production serve was stopped; suite arms ran alone).
  Power: factory 250 W cap, unlocked — idle 44–45 °C / 38 W. `gpu_samples.csv` (8 s `nvidia-smi` loop; the
  loop died partway through the run, so it covers 91 load samples — the 64K and needle windows — plus 50 idle
  ones, not every suite minute): SM clock at its **maximum 1 380 MHz in 73/91 load samples** and at
  1 312–1 372 MHz (97–99.5 % of max; normal boost transitions) in the other 18 — **nothing below 1 312 MHz**,
  up to 260 W instantaneous `power.draw`, 53–73 °C. No thermal-throttling signature (that pins clocks deep
  under max for sustained periods; idle never exceeded 45 °C at 38 W).

## Model and configuration

- `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001/2-of-00002.gguf` (29 608 446 496 + 28 800 138 432 B) via the
  native pack (`tools/iq_pack.py`, IQ1_M), 48 layers, 256 routed experts/layer (12 288), hidden 2 560
- Vision: `strata-vision` + `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (elastic, off the hot path); expert-profile
  `expert-profile-coder.bin`; speed-projection control vector layers 4–44 (`--cvec-mode project --cvec-dir per-layer`)
- Context 262 144; KV int8; resident experts (`--resident-experts`, ~16.3 GiB page-locked, rest file tier);
  `--expert-cache auto` → 10 198 slots / 19 878 MiB VRAM; `--prefill auto:32768`; conversation cache 8 192 MiB / 4 slots
- DFlash2 drafter `--draft dflash2 --spec 6 --draft-conf 0.5` (stock q4_0 pack; `--draft-fallback mtp`, MTP rt kept);
  thinking **off** for the suite per protocol; server default sampling otherwise

```text
strata --pack .../coder-iq1_m --native .../IQ1_M-00001-of-00002.gguf --ple-gguf .../00002 \
  --expert-profile expert-profile-coder.bin --expert-cache auto --prefill auto:32768 \
  --spec 6 --mtp .../mtp/rt --max-context 262144 --conversation-cache-mib 8192 --kv int8 \
  --resident-experts --vision --vram-reserve-mib 700 \
  --control-vector-scaled .../speed-projection.gguf:1.0 --control-vector-layer-range 4 44 \
  --cvec-mode project --cvec-dir per-layer \
  --draft dflash2 --draft-model .../qwen3.8-flash-next-dflash2.q4_0.pack --draft-fallback mtp \
  --draft-conf 0.5 --spec-min-p 0.5
```

Arms: **A** = production env `STRATA_PF_WMMA=1 STRATA_PF_WMMA2=1`; **B** = same build/config without them.

## Method

`strata bench run --url ... --suite standard-v1` per the draft protocol (frozen prompts, 22 requests, warm-up
excluded); arms interleaved A→B→A→B (rounds 1–2, one server start per arm) so arm order cannot bias the pair;
engine timings (`prompt_per_second`, `predicted_per_second`) from the final stream chunk; conversation cache
untouched by the suite prompts (unique tags). Supplementary: fresh word-list prompts of 50 935 tokens
(`scripts/p64.py`, seed 2026, `max_tokens 2`), run B→A→B the same day with one server start per arm;
`STRATA_PREFILL_TIMING=1` on a same-day same-build run of the same harness for the phase table
(`64k/phase_64k.md`, 50 835 tokens, seed 42); `tools/needle_bench.py --lengths 32k,128k --depths 10,50,90`
for recall; `gpu_samples.csv` for the clock/power trace (coverage caveat in §Hardware).

## Results

### Standard suite (`strata bench run --suite standard-v1`, protocol draft.1, medians of 3 runs per point)

| Workload | Metric | A: `PF_WMMA`+`PF_WMMA2` | B: neither | Δ |
| --- | --- | --- | --- | --- |
| short (~100 tok prompt) | prompt tok/s | 102.2 / 102.3 | 120.3 / 127.8 | **−17 %** (small prompts lose — see Limits) |
| 4K prompt (≈3.9K tok) | prompt tok/s | 1,274.5 / 1,272.1 | 1,199.8 / 1,199.8 | **+6.2 %** |
| 4K prompt | TTFT | 3.2 s | 3.4 s | |
| 16K prompt (≈15.4K tok) | prompt tok/s | 1,513.2 / 1,519.7 | 1,351.6 / 1,351.9 | **+12.0 %** |
| 16K prompt | TTFT | 10.9 s | 12.2 s | |
| decode (DFlash2 spec 6) | tok/s | 63.9–78.4 | 68.0–78.8 | the flags do not touch the decode path; run spread is drafter-acceptance noise |
| sanity I1–I4, R1–R4, T1–T4 | 12 checks | 12/12 ×2 runs | 12/12 ×2 runs | identical pass counts both arms |

`run.status` is **nonconforming** in all four runs: the protocol caps performance outputs at 512 tokens and
this coder model writes past it (8/9 performance requests "truncated"; the speed fields are the engine's own
`timings` and unaffected). All 12 sanity checks completed and passed. Arm medians repeat across rounds to
<0.5 % on prompt speed. Per-run documents: `results/std_{A,B}_r{1,2}/`.

### Fresh ~51K-token prompt, same day, B→A→B

| Arm | Actual prompt tokens | prompt tok/s |
| --- | --- | --- |
| B (neither flag) | 50,935 | 1,476.8 |
| A (`PF_WMMA=1 PF_WMMA2=1`) | 50,935 | **1,703.1** |
| B again | 50,935 | 1,477.0 |

**+15.3 %**; the two B arms agree to 0.02 %. Single card, ×4 link, full 250 W boost — above the dual
V100-SXM2 ×16-NVLink pair locked at 200 W/1,020 MHz (**1,644 t/s**) reported in NVIDIA_V100.md.
Fixture and script: `64k/` + `scripts/p64.py` (seeded 50 935-token word-list prompt, `stream=true`, speed
from the final SSE `timings` chunk). The phase profile for these runs: `64k/phase_64k.md`.

### Needle recall (arm A)

`tools/needle_bench.py --lengths 32k,128k --depths 10,50,90`: **6/6 found**, actual prompt lengths 27 761–27 762
and 107 361 tokens, per-case times 18–62 s (`needles.json`). Recall on those inputs only — not a quality claim.

## What limits prefill on this box (phase profile, arm A, 50 835-token prompt)

GPU timeline 26 292 ms against 26 429 ms wall (**99.5 % GPU-busy**): expert GEMM gate/up+down 30.5 %,
attention (qsa) 19.7 %, hyper-connections ~16 %, GDN ~11.6 %, card↔host streaming (gather + wait copy +
grouping) **3.2 %**, dequant 1.2 %. No dominant phase: this is Volta execution spread across GEMM/attention/HC/GDN
at max clock — the ×4 link costs ~3 %, not the headline. Full table: `results.json` + `phase_64k.txt`.

## Limits

- One machine, one model, one quantization; production serve config — a benchmark-tuned launch could differ.
- The `PF_WMMA` gain is prompt-length-dependent: **+12 % at 16K, +6 % at 4K, −17 % at ~100 tokens** — the
  crossover for this IQ1_M model is between ~100 and 4 000 tokens (a small grouped GEMM does not amortize the
  FP16 staging). Arm A costs ~20 t/s on trivial prompts (102 vs 120-128, both far below the prefill-critical
  region); long-prompt serving should keep the flags, very short-prompt chat may not.
- Standard suite is pre-protocol (draft); its numbers are comparable to other standard-v1 reports only.
- IQ1_M on Volta is the slow IQ family case (no DP4A); the PF_WMMA path carries it — do not extrapolate to IQ4.
- Conversation cache effects, concurrency and vision were **not** tested by the suite (`--clients` left for a
  follow-up). Needle test measures recall on those inputs, not quality.

## Files

| Path | What |
| --- | --- |
| `results/std_{A,B}_r{1,2}/` | the four `strata-standard-benchmark` runs as written by the runner: `results.json` (hardware, software, model, configuration, every request + timings, summaries; runner-sanitized) and `outputs.jsonl` (per-request raw stream records) |
| `64k/64k_B.txt`, `64k/64k_A.txt` | one line per fresh-prompt run: `prompt_n <tokens> prompt_per_second <tok/s>` |
| `64k/phase_64k.md` | raw + grouped `STRATA_PREFILL_TIMING=1` breakdown |
| `needles.json` | `needle_bench.py` per-case records: length label, depth, actual prompt tokens, seconds, found |
| `gpu_samples.csv` | `nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu` (plain, no header), 8 s apart |
| `scripts/p64.py` | the 20-line 64K harness (seeded word-list prompt, SSE `timings` reader) |

All measurements were produced by the reporter on the machine described, on 2026-11-24, on the production serve
config (§Model and configuration) started as a bare process (the usual systemd cap was not active) with no
other load.
