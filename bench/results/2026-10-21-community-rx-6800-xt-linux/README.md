# Community report: AMD Radeon RX 6800 XT 16 GB (gfx1030), Ryzen 9 5950X, 128 GB RAM, Linux — v0.2.1 and v0.2.2, original Flash-Next IQ3_S

Measured on 2026-10-20 and 2026-10-21 on one home-lab Linux box, the card alone serving the model. This covers a single
RX 6800 XT (the 72-CU sibling of the RX 6900 XT in the [2x RX 6900 XT reports](../2026-10-09-community-2x-rx-6900xt-0.1.41/README.md)),
ROCm 10.0.0, the gfx103x switches, a v0.2.1 -> v0.2.2 migration, a sweep of the prompt-speed switches the docs list for
RDNA2, a rocBLAS table calibrated on this card, ROCm 10.1.0, and the conversation cache. The standard benchmark of the
final configuration is in [../2026-10-21-standard-unknown-35dea42b](../2026-10-21-standard-unknown-35dea42b/README.md)
(the runner could not read the card's name and wrote "AMD Radeon"; it is this RX 6800 XT).

**Disclosure: this is vibe-coded.** The owner of the machine directed the work; an AI agent (Anthropic's Claude Code)
installed the stack, built the engines, ran every benchmark and wrote this report and the PR. The numbers below are the
measured outputs of those runs (per-run JSON in `data/`), not estimates; anything not measured is marked as such.

## Hardware and software

- **GPU:** AMD Radeon RX 6800 XT 16 GB (Navi 21, gfx1030, 72 CUs, wave32, no matrix cores), stock clocks, driver power
  cap 272 W (min 255 W). Resizable BAR on (16 GB BAR). No display attached.
- **Slot:** ASUS PRIME X570-PRO, top CPU slot. 2026-10-20: the only card, PCIe 4.0 x16. 2026-10-21: an RTX 4070 sat
  idle in the second CPU slot, so the RX 6800 XT ran at PCIe 4.0 x8 (x8/x8 split); the 4070 was not used by Strata.
- **CPU / RAM / storage:** Ryzen 9 5950X (16 cores, AVX2), 128 GB DDR4-3200 (4x32 GB), Crucial T500 1 TB NVMe.
- **OS:** Ubuntu 24.04.5, kernel 7.0.0-38-generic (the kernel of the 2x RX 6900 XT reports), the kernel's amdgpu driver.
- **ROCm 10.0.0** (HIP 7.15.26333, rocBLAS 5.6.0.8d1ae90e, AMD clang 23), the same version as the 2x RX 6900 XT box. AMD
  publishes 10.x for Ubuntu 24.04 only as the `rocm/dev-ubuntu-24.04:10.0.0-full` container image; we copied its
  `/opt/rocm` to the host. Note for others: the image wires `/opt/rocm/{bin,lib,llvm,...}` through `/etc/alternatives`
  symlinks; copy with links preserved and repoint those 9 top-level links at `core-10.0/...`, otherwise `clang++` cannot
  find `librocm_sysdeps_*.so` through its `$ORIGIN` rpath. ROCm 10.1.0 (HIP 7.16.26385, rocBLAS 5.7.0.39d8d603) the same way.
- **Strata:** v0.2.1-3-gc093ac26 (engine 0.1.46) on 10-20; v0.2.2 `011c9e60` (engine 0.1.47) on 10-21. Both built by
  `./setup.sh --backend hip --model IQ3_S --context 131072 --kv int8` against the system ROCm 10.0.0 (setup's flags:
  `STRATA_ENABLE_HIP=ON CMAKE_HIP_ARCHITECTURES=gfx1030 STRATA_PREFILL_MMQ=ON STRATA_Q6K_EXPERTS=ON STRATA_MMQ_KQUANTS=ON`).
- **Model:** ISTA-DASLab Qwen3.8-Flash-Next-GSQ-RCO IQ3_S, the same two files as the 2x RX 6900 XT reports
  (sha256 `4c1eb2ce...2aca3` and `316b46f3...e113`), setup's pack, MTP draft layer, `data/expert-profile.bin`.
- **Configuration (setup's):** context 131,072, `--kv int8 --kv-resident 32768`, `--expert-cache auto`, `--prefill auto`,
  `--spec 4 --spec-min-p 0.5`, `--ple-io ram`, rocBLAS table `gfx1030-rocblas-5.6.0.8d1ae90e.txt` (setup points the engine
  at it; log: `rocBLAS tuning enabled (112 rows, 76 with a solution, ...)`). Expert cache ~4,830-4,850 slots, ~9.2 GiB.
- **"Switches"** below = `STRATA_HIP_PROMPT_F16=1 STRATA_SH_STREAM=1 STRATA_HIP_ADAPT_KERNEL_COPY=1`.

## Method

`benchmark.py` from [2026-10-09-community-2x-rx-6900xt-0.1.41](../2026-10-09-community-2x-rx-6900xt-0.1.41/benchmark.py),
unchanged: serial, fresh prompts with a nonce, greedy, 256-token cap, the server's own prompt/decode timings, a fresh
server per arm. 2 runs per size, medians (min/max in `data/*-summary.json`). Prompt sizes 4,096 / 32,768 / 128,000
(the 128K run on 10-21 tokenized to ~125K with this tokenizer build). Nothing else used the GPU.

## Results

### v0.2.1, one card (2026-10-20, PCIe 4.0 x16)
| arm | 4K prompt t/s | 32K | 128K | decode t/s 4K / 32K / 128K | TTFT 4K / 32K / 128K s |
|---|---|---|---|---|---|
| stock (setup's config) | 521.1 | 545.8 | 525.2 | 47.1 / 44.0 / 48.5 | 7.9 / 60.1 / 243.8 |
| **switches** | **986.0** | **1,111.6** | **1,032.8** | 48.5 / 50.5 / 51.7 | 4.2 / 29.5 / 124.0 |
The single RX 6800 XT matches or beats the single-card RX 6900 XT numbers of the 0.1.41 report (stock 473 / 502 / 486,
switches 850 / 1,102 / 1,027), likely from the newer engine and the 16-core CPU (5 pool workers there).

### v0.2.2 and the prompt-speed sweep (2026-10-21, PCIe 4.0 x8, all with the switches)
| arm | 4K | 32K | 125K | decode t/s |
|---|---|---|---|---|
| v0.2.1 (same day) | 942 | 1,095 | - | 45.2 / 46.5 |
| v0.2.2 | 937 | 1,087 | 1,010 | 46.3 / 46.8 / 47.9 |
| **v0.2.2 + `--prefill auto:32768`** (setup's tip for 126 GB RAM) | 929 | **1,123** | **1,054** (TTFT 121.5 vs 126.9 s) | 46.3 / 46.0 / 45.6 |
| + `--resident-experts` | 738 | 983 | - | 41.4 / 45.3 |
| + `STRATA_SELECT_SIMT=1` | 938 | 1,106 | - | 44.6 / 44.9 |
| + `STRATA_DENSE_MMQ=1` | 935 | 1,129 | - | 46.0 / 43.9 |
| + `STRATA_HC_PERSIST=1` | 937 | 1,122 | - | 47.4 / 46.2 |
| + rocBLAS table calibrated on this card (ROCm 10.0) | 939 | 1,123 | - | 47.8 / 46.7 |
| ROCm 10.1.0 build + its own calibrated table | 937 | 1,121 | - | 47.3 / 45.0 |
| no rocBLAS table at all (a mis-named table file) | 823 | 1,106 | - | 47.1 / 44.0 |
Single runs per arm, two repetitions each: differences of 1-3% are inside this box's noise.

What we read from it:
- **v0.2.2 = v0.2.1 in speed on this card** (its speed work is Gyro / NVIDIA / Mac). We moved to it for the HIP first
  verify-window hang fix and the tool-call parsing fixes.
- **`--prefill auto:32768` is the only prompt switch that paid: +3% at 32K, +4.4% at 125K.** Kept.
- **`--resident-experts` was 21% slower at 4K** although it got everything it asked for: the log shows `resident 41.18 GiB,
  pinned 41.18 GiB`, `1874 of the prompt path's 1874 lendable slots keep their experts in RAM`, `file tier 0.0 GiB`, and
  `adaptive overlap: off: not in the resident RAM mode`. With 128 GB the default mode already keeps the whole expert arena
  in pinned RAM, so resident mode has nothing to save and loses the adaptive overlap (not re-tested with
  `STRATA_ADAPT_OVERLAP_RESIDENT=1`).
- **rocBLAS table: having one matters (+14% at 4K vs none); which one does not.** The shipped 6900 XT table and a table
  calibrated on this RX 6800 XT (`tools/hip/tune_rocblas`, 112 rows, 82 with a solution) measure the same. Both tables we
  made are in `data/rocblas-tables/` (ROCm 10.0.0 / rocBLAS 5.6.0.8d1ae90e and ROCm 10.1.0 / 5.7.0.39d8d603), in case they
  are useful as rows for other gfx1030 cards; we did not change `tools/hip`.
- **ROCm 10.1.0 = 10.0.0** in prompt and decode.
- `STRATA_DENSE_MMQ`, `STRATA_SELECT_SIMT`, `STRATA_HC_PERSIST`: within noise at 4K/32K on one card.

### Conversation cache (v0.2.2, the deployed config)
`--conversation-cache-mib 24576 --conversation-cache-slots 16 --conversation-cache-min-tokens 2048`: a 7,695-token chat,
then a different 3,084-token request, then the chat's next turn: **8.6 s -> 2.5 s**; engine log `conversation cache:
restored 7688 tokens (checkpoint) in 35.9 ms`; snapshots 338 MiB (7.7K tokens) and 270 MiB (3.1K tokens).

### Answer quality (not a Strata benchmark; for context)
The owner's own private agentic-memory benchmark (33 cases of a tool-calling "reflect" loop over a synthetic memory
store, thinking medium, graded by an LLM judge): **94.9** with v0.2.1 + switches on this card, against 95.9 / 95.1 for
the same model and quant on an RTX 4070 12 GB (CUDA, older engine). The F16 prompt route did not cost measurable quality
there. The prompts are private and not included.

### Standard benchmark (`strata bench run`, protocol strata-qwen38-thinking-v1-draft.1, track custom)
Final configuration (v0.2.2, switches, `--prefill auto:32768`, conversation cache on), freshly loaded: 21 of 21 required
requests, **12/12 sanity (instruction 4/4, retrieval 4/4, tool calls 4/4)**, 4 min 2 s after load. Prompt 4K 951.9 t/s,
16K 1,099.8 t/s; decode 44.6 (short) / 47.6 (4K) / 43.2 (16K) t/s; TTFT 1.72 / 4.34 / 14.94 s. See the standard folder.

## Limits
- One machine, one card, two repetitions per arm; the 10-21 runs were at PCIe x8 (another card idle beside it).
- No needle/recall runs in the community sweep (the standard benchmark's retrieval checks passed 4/4).
- Two-card (layer split) RX 6800 XT not measured. Windows not measured. The `rx6000` engine package was not tried
  (source builds only).
- The quality figure is from a private suite and is not reproducible by others.
