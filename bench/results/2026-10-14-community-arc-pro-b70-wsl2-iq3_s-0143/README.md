# Community benchmark on Intel Arc Pro B70 32 GB (WSL2) + GSQ-RCO IQ3_S — bonus historical snapshot

Measured on/before 2026-10-14 by kgmkm, transcribed from the Windows→Linux
migration memo of 2026-10-14. Serve of the official ISTA-DASLab GSQ-RCO
IQ3_S on engine 0.1.43.1 **plus four uncommitted local WSL2 patches**
(GPU-PV passthrough support: Level Zero fallback in `sycl/setup_intel.py`,
`/dev/dxg` + WSL lib mounts and env-filter narrowing in
`sycl/serve/strata-sycl.sh`, `STRATA_SYCL_FREE_MIB` override in
`sycl/src/core/expert_cache.cpp`, 1 GiB host-mirror chunks in
`sycl/src/core/gguf_expert_source.cpp`). This is a **bonus**: summary
values only — per-request engine lines, `/metrics` snapshot, and the
exact server config were not preserved. Main limitations: no median/range,
no TTFT, no needle recall, patched (not stock) engine, WSL2 GPU-PV stack
rather than native Linux.

## Hardware and software

- GPU and VRAM: Intel Arc Pro B70 32 GB, single-GPU serve under WSL2
  GPU-PV passthrough. Engine PCIe probe 6.3 GB/s host->device
  (vs 13.3 GB/s native reference).
- CPU: AMD Ryzen 7 3700X (same box as the native-Linux report).
- OS: Windows with WSL2 (Ubuntu 24.04); Intel Arc driver 32.0.101.8976;
  user-space compute stack from kobuk-team/intel-graphics PPA
  (`libze-intel-gpu1`, `libze1`, `intel-opencl-icd`, `clinfo`).
- Strata base v0.1.43.1 (`a0116b27`) plus the four local WSL2 patches
  above (92 insertions, never submitted upstream as code).
- Background workloads: not recorded.

## Model and configuration

- Model repository: ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, IQ3_S.
- Context 32768; KV int8; `--stream-experts`; expert cache 9196 slots,
  17.46 GiB VRAM (sized via `STRATA_SYCL_FREE_MIB=22000`, i.e.
  measured-free minus margin, because the free-VRAM query reads 0
  under GPU-PV); `--prefill 2048`; `--vram-reserve-mib 1024`;
  spec 4; spec_min_p 0.5; MTP rt/.
- Experimental speed projection: not recorded for these runs.
- Exact launch command and sanitized config: not preserved (see
  limitations). The memo records the setting values listed above.

## Method

- Values below are summary numbers transcribed from the 2026-10-14 memo
  (appendix B); the per-request engine timing lines were not preserved,
  so there is no `requests.csv`. Warm-up, repetitions, and cache state
  were not recorded.
- Decode conditions: a story prompt, a code prompt, and a summary
  prompt (exact prompts not preserved). Prompt conditions: a 3,744-token
  fresh prefill and short prompts of 10–15 tokens plus a 393-token summary.
- Timing boundaries: engine-reported tok/s as noted in the memo.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| prompt-3.7k (fresh prefill) | 3744 | not recorded | not recorded | 1 (summary) | 198.8 (single value) | n/a | not measured |
| decode-story | not recorded | not recorded | not recorded | 1 (summary) | n/a | 10.1 (single value) | not measured |
| decode-code | not recorded | not recorded | not recorded | 1 (summary) | n/a | 13.8 (single value) | not measured |
| decode-summary | not recorded | not recorded | not recorded | 1 (summary) | n/a | 9.3 (single value) | not measured |
| short prompts | 10–15 | not recorded | not recorded | noted as range | 10–15 (noted as range) | n/a | not measured |
| summary-393 | 393 | not recorded | not recorded | 1 (summary) | 55.5 (single value) | n/a | not measured |

Per-condition values: `measurements.csv`. No per-request data was
preserved, so there is no `requests.csv`.

- Draft acceptance: story 109/211 (51.7%), code 130/149 (87.2%);
  summary decode drafts not recorded. The memo calls drafts healthy on
  code and notes low single-digit hit rates on story-like prompts
  (profile corpus mismatch suspected).
- Output quality: short outputs and the 3.7K summary coherent in JP/EN.
  Not claimed bitwise-identical.
- Caveat (WSL-only, under investigation at memo time): `--prefill 4096`
  deterministically degraded long-prompt output to `!` repetition (100%
  drafts accepted, reproducible including cached-KV reruns) while short
  prompts stayed coherent; `--prefill 2048` restored good output on a
  fresh 3.7K-token prefill. Cause not identified — stated as under
  investigation, no conclusion claimed.
- Rough gap vs native: about 1/5 prompt and 1/3 decode of the native
  reference (prompt 4K 980–1000, decode 30.5/40.8, PCIe 13.3 GB/s);
  the memo attributes the main gap to the link (6.3 vs 13.3 GB/s).

## Correctness and limitations

- Summary transcription from a memo, not a logged benchmark run:
  no per-request rows, no `/metrics` snapshot, no sanitized config,
  no exact prompts, no repetition counts.
- Patched engine (four local WSL2 changes), so these numbers do not
  describe stock 0.1.43.1.
- No needle recall test; vision not tested; no soak run; TTFT not
  measured.
- Untested: everything not listed above, including whether the
  `--prefill 4096` issue reproduces on native Linux or newer engines
  (on the 0.1.45 native-Linux B70 run in the sibling folder,
  `--prefill 4096` served all conditions coherently).
