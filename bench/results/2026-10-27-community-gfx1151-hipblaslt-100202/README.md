# gfx1151 hipBLASLt table for a system ROCm 7.2.4 (hipBLASLt 1.2.2, 100202)

Setup takes a system ROCm 7 in `/opt/rocm` over the TheRock wheels. On Strix Halo with ROCm 7.2.4 installed
(for llama.cpp, say) no gfx1151 table matched its hipBLASLt 1.2.2, so the prompt's dense GEMMs ran on plain
hipBLAS and big prompts were about 2x slower than with the 7.14 wheels.

## PC

Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB, Ubuntu, kernel 7.0.0-38, system ROCm 7.2.4,
engine 0.1.51 (4b9e05a2) built against `/opt/rocm`.

## Table

The 15 shapes of `tools/hip/gfx1151-hipblaslt-100401.txt` (`shapes.json`) at the same token buckets:

```sh
LD_LIBRARY_PATH=/opt/rocm/lib build-hip/tune_hipblaslt --shapes-file shapes.json \
  --tokens 2047,2048,8191,8192,16383,16384 --workspace-mib 32 --tuning-out gfx1151-hipblaslt-100202.txt
```

`meta device_arch=gfx1151 hipblaslt_version=100202 hipblaslt_header_version=1.2.2`; 90 cases, none rejected
by the tuner's accuracy gate. Sum of the 15 GEMMs, plain hipBLAS -> table: T 2048 115.5 -> 19.1 ms,
T 8192 467.9 -> 77.2 ms, T 16384 945.4 -> 157.3 ms.

## End to end

Unsloth UD-Q4_K_XL, setup's config (`--expert-cache 9328 --prefill auto --kv int8 --ple-io ram
--resident-budget-gib 71 --max-context 262144`), no vision, a fresh server per run, page cache dropped,
`temperature 0`, 256 tokens out. "Prompt set": `STRATA_IGPU_PROMPT_SET=1 STRATA_PF_FUSED=1
STRATA_PF_FUSED_KQ=1 STRATA_PF_SWITCH_MIN_T=4096 STRATA_PREFILL_STREAM_MIN=128` (docs/STRIX_HALO.md, section 5).

| ROCm | Table | Prompt set | 59,281-token prompt | 232,322-token prompt |
| --- | --- | --- | --- | --- |
| system 7.2.4 | none | off | 247.0 tok/s, decode 29.0 | 228 tok/s, decode 26 |
| system 7.2.4 | none | on | 808.3, decode 28.5 | |
| system 7.2.4 | 100202 | off | 470.6, decode 28.4 | |
| system 7.2.4 | 100202 | on | 1,107.0, decode 29.6 | 1,033.7, decode 26.9 |
| wheels 7.14.0a20260608 | 100400 | off | 461.5, decode 28.7 | 417.1, decode 25.1 |
| wheels 7.14.0a20260608 | 100400 | on | 1,053.9, decode 29.5 | 1,025.5, decode 25.8 |

One run each. The engine log with the table: `prefill gemm: hipBLASLt tuning enabled (90 rows, gfx1151, version 100202)`.
