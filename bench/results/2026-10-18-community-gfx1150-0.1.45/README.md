# Radeon 890M (gfx1150) on 0.1.45: the shipped table A/B, and the 17 exact switches' identity check (2026-10-18)

Follow-up to PR #1395 (and #1217). Two things asked there:

1. **The shipped table, A/B** (`tools/hip/gfx1150-hipblaslt-100401.txt` as 0.1.41+ ship it, against plain hipBLAS, 3+
   interleaved rounds, the prompts of [2026-10-07-community-gfx1150](../2026-10-07-community-gfx1150/README.md)):
   **prompts read 1.54x / 1.72x / 1.74x faster** at 1K / 3.6K / 7K tokens (medians of 5 rounds, every round the same
   order), decode unchanged.
2. **The identity check of the 17 switches** (gfx1151's exact defaults less `STRATA_HCD_EXACT`) on this chip: all 17
   off against all 17 on, greedy token ids of 10 short prompts plus a 4K and a 20K prompt: **12 of 12 identical**, on two
   models (GSQ-RCO IQ3_XXS twice per side, OrcaRouter Uncensored IQ3_XXS once per side), and each side identical to its
   own repeat. The HIP parity tests pass with the 17 set as well.

## Rig

- MINISFORUM N5 PRO: Ryzen AI 9 HX PRO 370 (4 Zen 5 + 8 Zen 5c cores, 24 threads, AVX-512), Radeon 890M (gfx1150,
  RDNA3.5, 16 CUs), 96 GB DDR5-5600 (93.4 GiB usable), BIOS carve-out 512 MB. The GPU's memory is the GTT pool of the
  same RAM, 70 GiB (`ttm.pages_limit=18350080 ttm.page_pool_size=18350080`; dmesg: `71680M of GTT memory ready`).
- Proxmox VE 9.2 (Debian 13), kernel 7.0.14-24-pve with the in-tree amdgpu / amdkfd. The engine runs in an unprivileged
  LXC container (Debian 13, 8 vCPUs, 40 GiB) with `/dev/kfd` and `/dev/dri/renderD128` passed in.
- ROCm: AMD TheRock 7.14.1 for gfx1150 (hipBLASLt 1.4.1, version number 100401), as before.
- Engine **0.1.45** (`477ac13`), built by hand with GCC 14.2: `-DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF
  -DSTRATA_PREFILL_MMQ=ON -DSTRATA_PARITY_PROMPT_ATTN=ON -DCMAKE_HIP_ARCHITECTURES=gfx1150` (STRIX_HALO.md's steps with
  the architecture changed). ctest: 71 of 75 pass; `hip_prefill_hcd_exact_parity`, `hip_prefill_hipblaslt_gemm` (no
  table) and `hip_prefill_rocblas_gemm` skip, `ple_parity` fails (it needs `bench/micro`). With the table
  `hip_prefill_hipblaslt_gemm` passes and `hip_prefill_hcd_exact_parity` still skips (the gfx1150 table picks other
  solutions than 1176 / 1177); `iq_parity` by hand: all 10 formats ok.
- Models: Qwen3.8-Flash-Next **GSQ-RCO IQ3_XXS** (ISTA-DASLab, a native pack with `experts.bin`; the table A/B and the
  identity check) and **OrcaRouter Qwen3.8-Flash-Next-Uncensored IQ3_XXS** (`--compat-bf16` pack; the identity check).
  All 24,576 experts in the GTT pool (`--mmap-experts --expert-cache 24576`), `--prefill auto` (8,192-token chunks),
  `--spec 4 --spec-min-p 0.5` with the MTP draft layer, `--max-context 65536 --kv int8`, `--ple-io mmap`, `--vision`
  (CPU encoder). Config: [strata-gfx1150.json](strata-gfx1150.json).
- Nothing else used the GPU during the runs (the image and audio servers stopped, the chat bot that uses this server
  stopped).

## 1. The shipped table against plain hipBLAS

The method of the 2026-10-07 folder, with more rounds: the server restarted for every run (cold prompt cache), one request
at a time, the engine's own `strata serve: prompt` line. Each run: a short chat three times (256 tokens out, greedy),
then one fresh document prompt each of about 1K, 3.6K and 7K tokens (a random nonce first). Five rounds of the two arms,
the order alternating (H T, T H, ...); a third arm with `STRATA_HIP_WMMA=1` (opt-in, changes bits) in the first three
rounds for reference. Every run is in [runs.csv](runs.csv).

- `hipblas+switches`: no table, the 17 switches
- `table+switches`: the shipped `tools/hip/gfx1150-hipblaslt-100401.txt` (byte-identical to the one in #1395) and the 17
  switches, as `STRATA_EXPERIMENTAL_GFX1150=1` setup writes them
- `table+switches+HIP_WMMA`: the same plus `STRATA_HIP_WMMA=1`

| Prompt | hipblas+switches | table+switches | table+switches+HIP_WMMA |
|---|---:|---:|---:|
| 987-989 tokens | 95.4 tok/s (94.6-96.4) | 147.0 (145.6-148.7) | 152.2 (151.9-152.7) |
| 3,562-3,564 tokens | 128.5 (127.4-129.1) | 221.5 (220.4-222.2) | 254.3 (253.9-254.8) |
| 7,009-7,010 tokens | 132.8 (131.4-133.0) | 231.6 (229.0-233.4) | 277.2 (275.1-279.6) |
| decode, short chat (runs 2-3) | 15.2 tok/s (15.1-15.3) | 15.2 (15.1-15.3) | 15.2 (15.1-15.2) |

Medians (and ranges) over the rounds: 5, 5 and 3. The table is 1.54x / 1.72x / 1.74x; `STRATA_HIP_WMMA=1` adds 4% / 15% /
20% on top. `prefill gemm: hipBLASLt tuning enabled (144 rows, gfx1150, version 100401)` in every table run.

## 2. The 17 switches: identity check

`ATTN_LANECELL CVEC_FUSE EXPERT_V2 GDN_CONVL2 GDN_HEAD GDN_NOY GDN_PP=2 GDN_SPLIT LFUSE MMVF_ROWS PF_PAD PLE_BATCH
Q6_PACKED Q8_PACKED QFUSE SH_STREAM TSUM` (all `STRATA_`, `=1` unless noted), all off against all on. Both sides with the
shipped table, without `STRATA_HIP_WMMA`.

The engine is started directly (`serve.server.StrataEngine`, as `tools/calibrate.py` does) and the generated token ids
are compared, not the text: [scripts/strata_ident.py](scripts/strata_ident.py). Greedy, no thinking, one request at a
time, in a fixed order: 10 short prompts (English, Japanese, code, SQL, arithmetic, a translation; 128 tokens out), then a
4,024-token and a 20,024-token prompt (this checkout's `docs/*.md`; 64 tokens out). MTP on, as served.

| Model | Runs | Result |
|---|---|---|
| GSQ-RCO IQ3_XXS | off, on, off, on | 12 / 12 identical off against on (both pairs), off against off and on against on |
| OrcaRouter Uncensored IQ3_XXS | off, on | 12 / 12 identical |

Speed from the same runs (the 10 short prompts' decode in total, and the two long prompts): GSQ-RCO decode 18.3 / 18.8 tok/s
off, 18.3 / 19.0 on; 20K prompt 207.5 / 206.6 off, 212.4 / 215.5 on (+2-4%). Orca decode 18.4 -> 18.9, 20K 205.8 -> 214.9.
Per case: [identity.csv](identity.csv) (sha-256 of each case's id list, times, draft acceptance).

The HIP ctest suite with the 17 set in the environment (and the table): the same results as without them (only
`ple_parity` fails, `hip_prefill_hcd_exact_parity` and `hip_prefill_rocblas_gemm` skip).

## 3. `tools/calibrate.py` on this machine

Run directly (`.venv/bin/python tools/calibrate.py strata-orca.json`; setup does not offer it on AMD), on the Orca
config as it serves here (table, the 17 switches, `STRATA_HIP_WMMA=1`): 2,210 s. **Nothing beat the default by the 3%
it asks**, so it keeps every setting ([calibrate.json](calibrate.json)):

| Step | Values (median tok/s) |
|---|---|
| PCIe share | 0 / 0.2 / 0.35 / 0.55 / 0.75 / 0.9 / 1: 19.4-19.5 for all (every expert is in the GTT pool: no misses to place) |
| draft floor `--spec-min-p` | 0.3: 19.2, 0.5: 19.5, 0.7: 19.7; confirmed 0.7 against 0.5 in 3 interleaved rounds: 19.70 vs 19.47 (+1.2%) |
| `--pool-workers` | 6 (the engine's own): 19.2, 4: 19.2, 3: 19.3, 2: 19.2 |
| adaptive tier (every / swaps / decay) | default 18.8, 1 / 80 / 0.97: 18.9, 1 / 160 / 0.97: 18.8 |
| `STRATA_SH_STREAM` | 0: 18.9, 1: 19.3 (+2%) |

`STRATA_SH_STREAM=1` stays in this machine's config: it is exact (section 2) and +2% here, under the 3% bar.

## Notes

- **A long prompt.** On the Orca config with WMMA, a 50,408-token prompt read in 204 s (247 tok/s); the GTT pool stayed at
  56.6 GiB from start to end (the 64K K/V is allocated at start).
- **`--kv-grow` is off on HIP.** It says `--kv-grow is off (one GPU, a profile, the whole K/V in VRAM, every expert in
  RAM, ...)`, but the cause is `src/core/vmm.cpp`'s HIP branch (`vmm_available()` is false: "no virtual memory here").
  docs/DETAILS.md lists the conditions without saying it is CUDA only.
- **0.1.43 against 0.1.45** (the Orca config, same server, a restart each): the same greedy answers for four fixed prompts
  (to 14K tokens) with and without `STRATA_HIP_WMMA`, decode the same (15.0-15.2 tok/s), prompts 3-5% faster on 0.1.45
  (3.6K: 241 -> 253 tok/s, 7K: 268 -> 280 with WMMA).

## Not tested

- KL or task quality (the identity check is bitwise; `STRATA_HIP_WMMA` changes bits and is not part of it).
- Other ROCm versions (the table is for hipBLASLt 100401; #2147's `gfx1150-hipblaslt-100500.txt` was not tried here).
- setup itself: the configs are hand-written (the same `env` setup's `STRATA_EXPERIMENTAL_GFX1150=1` writes).
- Contexts past 20K tokens in the identity check.
