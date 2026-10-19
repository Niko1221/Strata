# Community benchmark on RX 6700 XT (gfx1031) run as gfx1030, and the FP16 prompt route

Measured on 2026-10-19 on one RX 6700 XT with system ROCm 7.1.1. Two things are in here:

- **the card needs two changes to run long prompts at all** - the engine built for `gfx1030;gfx1031` and
  `HSA_OVERRIDE_GFX_VERSION=10.3.0`, both in [docs/AMD_GFX1031.md](../../docs/AMD_GFX1031.md). A gfx1031-only build
  loads, answers short prompts, and dies on the first ~12K-token prompt with `rocBLAS error: Cannot read
  .../rocblas/library/TensileLibrary.dat: No such file or directory for GPU arch : gfx1031`, which reads as out of
  memory. It is not: 62.7 GiB RAM, 12,272 MiB VRAM, and the prompt's dense matrix products are the only path that
  calls rocBLAS, and ROCm 7.1.1's rocBLAS ships gfx1030 kernels only
  (`ls /opt/rocm/lib/rocblas/library | grep -c gfx1031` is 0).
- **`STRATA_HIP_PROMPT_F16=1` is worth 1.76x to 1.99x on this card** on fresh prompts of 4K-32K tokens, and it does
  not touch decode. Every number below is from this machine; ranges are the low-high of three runs.

The suite ran twice, one engine session per arm, the model reloaded from cold between arms, nothing else changed.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 6700 XT, 12,272 MiB (device id 0x73DF, which setup maps to gfx1031), one GPU. PCIe
  link: not measured (`lspci -vv` needs a password here). GTT total 31.4 GiB.
- CPU: Intel Xeon E5-1620 v3 @ 3.50 GHz, 4 cores / 8 threads (max 3.6 GHz). Installed RAM: 64 GB, 62.7 GiB usable;
  speed and timings not measured.
- Storage: OS on a WDC WDS120G2G0A 120 GB SATA SSD; the model's GGUF on a Samsung 850 EVO 250 GB SATA SSD, mounted
  `fuseblk` (a FUSE mount, not ext4) - not compared against a native mount.
- OS: Ubuntu 24.04.5 LTS, kernel 6.8.0-147-generic, amdgpu from the kernel; ROCm 7.1.1 in /opt/rocm, rocBLAS 5.1.1.
- Strata: project 0.1.45, commit `477ac13c`; engine 0.1.45, HIP, built from source on this machine for
  `gfx1030;gfx1031` (`engine/BUILD.json` "archs": `["gfx1030","gfx1031"]`); not a release binary.
- Background workloads: the desktop session and this terminal; the agent that normally uses this server was idle
  during both arms; nothing else used the GPU. Power limits, clocks and temperatures: not measured.

## Model and configuration

- Model: Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S (`Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` and
  its second part, IQ3_S), served as `swift-1.5-iq3_s` from a built pack. The engine loaded 46.84 GiB at 0.24 GiB/s.
- Vision encoder: `--vision` is off in this config (`images: false` in the engine's own info).
- Context 65,536 tokens, KV int8, one parallel slot, expert cache 2,645 slots / 5.07 GiB of VRAM, PROFILE policy
  with no eviction, `--expert-cache auto`, `--prefill auto` (`prefill_cap` 8192), MTP draft layer with `--spec 4`
  (the engine reports `spec: 6`, `mtp_max: 4`), `--pcie-frac 0.55`, `--spec-min-p 0.70`, adapt on, expert arena
  47,962 MiB in host RAM, 616 MiB VRAM free after load.
- The two env entries the card needs, and the arm switch:

```text
# strata-swift-iq3_s.json  "env"
default arm: {"HSA_OVERRIDE_GFX_VERSION": "10.3.0"}
f16 arm:     {"HSA_OVERRIDE_GFX_VERSION": "10.3.0", "STRATA_HIP_PROMPT_F16": "1"}

# the engine says which route it took, on its startup line:
strata prefill: HIP device 0 gfx1030 - tip: STRATA_HIP_PROMPT_F16=1 reads long prompts about 2x faster here ...
strata prefill: HIP device 0 gfx1030 - STRATA_HIP_PROMPT_F16=1: the prompt's 16-bit GEMMs run FP16 in and out ...

strata --pack <packs>/swift-iq3_s --native <models>/swift-IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
       --ple-gguf <the same GGUF> --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto
       --spec 4 --mtp <mtp>/rt --max-context 65536 --kv int8 --pcie-frac 0.55 --spec-min-p 0.70
       --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97
# serve/server.py --engine strata --config strata-swift-iq3_s.json --port 8080   (API key removed)
```

## Method

[METHODS.md](METHODS.md) - `python tools/strata_bench.py --url http://127.0.0.1:8080 --long`, the built-in suite
against the running server, three runs per test, thinking off, a unique tag per prompt so the fresh-prompt tests
cannot be helped by the cache, model loading not timed. `results-default.json` and `results-f16.json` are the tool's
own per-request data, field for field. `engine-log-default.txt` and `engine-log-f16.txt` are each arm's startup
lines: which arch the card reported, which GEMM route the engine took, the cache sizes.

`scripts/rocblas_gemm_probe.cpp` and `scripts/prompt_test.py` are the two extra measurements below.

## Results

Prompt tok/s is the engine's own `prompt_per_second` on fresh prompts (0 tokens reused); decode is
`predicted_per_second` on the chat suite and the agent turns.

| Configuration | Prompt tokens | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT s median (range) |
| --- | ---: | ---: | --- | --- | --- |
| 4K prompt, default route | 4,019 | 3 | 339.2 (294.1-339.5) | not measured | 11.90 (11.88-13.78) |
| 4K prompt, FP16 prompt GEMMs | 4,019 | 3 | 595.9 (448.2-596.1) = 1.76x | not measured | 6.80 (6.79-9.03) |
| 16K prompt, default route | 16,058 | 3 | 352.5 (343.2-355.7) | not measured | 45.61 (45.20-46.85) |
| 16K prompt, FP16 prompt GEMMs | 16,058 | 3 | 692.9 (674.9-714.0) = 1.97x | not measured | 23.23 (22.55-23.85) |
| 32K prompt, default route | 32,049 | 3 | 356.2 (353.8-357.4) | not measured | 90.05 (89.77-90.68) |
| 32K prompt, FP16 prompt GEMMs | 32,049 | 3 | 707.5 (701.7-711.1) = 1.99x | not measured | 45.39 (45.14-45.76) |
| short chat, default route | 46 (43-51) | 9 | not measured | 21.7 (20.2-26.8) | 1.69 (1.51-1.96) |
| short chat, FP16 prompt GEMMs | 46 (43-51) | 9 | not measured | 21.0 (19.7-24.9) | 1.71 (1.52-2.37) |

The whole suite (chat + 4K + 16K + agent + 32K, three runs each) took 731 s on the default route and 464 s with the
FP16 prompt GEMMs.

The tool's own `--compare` of the two folders (arm B against arm A): prompt +75.7% / +96.6% / +98.6% at 4K / 16K /
32K, TTFT -42.9% / -49.1% / -49.6%, agent-turn TTFT -35.8%, chat TTFT +1.1%, chat decode -3.2%, agent decode -5.5%,
agent cache hit -0.1%. The decode and cache-hit deltas are the run-to-run spread; the chat arm (46-token prompts,
which never reach rocBLAS) is the control that says so.

The FP16 route changes the prompt read and the time to first token, and not the decode: chat decode 21.7 against
21.0 tok/s, agent-turn decode 23.6 against 22.3, both inside the run-to-run spread. A 46-token prompt never reaches
the rocBLAS path, so the chat arm is a control, not a comparison.

The 4K FP16 median is dragged by its own first run (448.2 against 596.1 and 595.9); the two later runs sit on the
16K and 32K numbers.

### Agent session (the case a Hermes-sized client lives in)

Five turns that each add about 1,500 words of a project file and continue from the model's own earlier answers, 64
tokens out, three sessions per arm. Cache behaviour is identical in both arms (68.8% against 68.8% of the
follow-up-turn tokens reused, about 2 tokens of earlier turns read again), so the difference is the reading:

| | Cache hit | Turn TTFT s median (range) | First-turn TTFT s | Decode tok/s median (range) |
| --- | ---: | --- | --- | --- |
| default route | 68.8% (68.6-68.8) | 9.59 (5.33-14.52) | 4.79 (4.74-6.55) | 23.6 (15.5-27.7) |
| FP16 prompt GEMMs | 68.8% (68.6-68.8) | 6.16 (4.27-10.64) | 4.39 (4.35-5.63) | 22.3 (18.5-30.4) |

The turns that re-read about 2,900 tokens of uncached text ran at 207-263 tok/s on the default route and at
519-552 tok/s with the FP16 route, in the same sessions.

### Matched 11,103-token prompt, the same server session, every read from the log

`scripts/prompt_test.py`, a fixed code-like prompt of 11,103 tokens, 48 asked and 37 generated each time:

- default route, 3 reads: 33.9 s / 32.8 s / 34.4 s reading (327.6 / 338.1 / 322.4 tok/s), the 37 tokens in 1.09 s /
  1.64 s / 1.68 s (22.0-34.1 tok/s)
- FP16 route, 1 fresh read: 17.9 s reading (620.9 tok/s) = 1.90x, the 37 tokens in 1.26 s (29.3 tok/s)
- a 13,023-token prompt on the default route in the same session: 38.6 s (337.6 tok/s)
- the second copy of the same text reuses 11,098 of 11,103 tokens (5 read) in 102 ms, so the FP16 arm contributes one
  fresh read: the matched pair is one fresh read per arm, not two independent samples

### The GEMMs under the prompt path

`scripts/rocblas_gemm_probe.cpp`, `rocblas_gemm_ex` at `N = 10240 x T = 1024 x K = 2560` (a shape the prompt path
uses), each type checked against a CPU reference, run on this card under the override:

| route | TFLOPS | relative error |
| --- | ---: | ---: |
| f16 in, f32 out (the default) | 7.0 | 2.5e-4 |
| f16 in and out (`STRATA_HIP_PROMPT_F16=1`) | 21.3 | 4.3e-4 |
| bf16 in, f32 out | 6.6 | 1.9e-3 |
| f32 in and out | 11.8 | 6.6e-7 |

The raw GEMM is 3.0x faster in FP16-out; the end-to-end prompt read lands at 1.8-2.0x, which is what is left after
the attention, the MoE routing and the host work around it.

## Correctness and limitations

- Every arm answered coherently: the three chat questions were answered, the agent turns used the project file they
  were given, and the 11,103-token prompt returned a summary of it. No needle test, no recall suite, no
  golden-answer parity, no HumanEval here - this is a speed run.
- The FP16 route is **not bit-identical** to the default (an FP16 rounding of the prompt GEMMs, and the engine says
  so on its startup line), so answers can differ slightly on borderline prompts. The GPU expert cache rounds
  differently from the CPU as well, on both arms (the engine's own note points at
  `bench/results/2026-09-27-cache-parity`).
- One machine, one engine session per arm, one order (default first, FP16 second). Not interleaved, so drift
  between the two sessions - temperatures, the expert cache filling up - is not separated out; the chat control arm
  and the identical cache hit rates argue against a large drift.
- The card reports and runs as gfx1030 for the whole run, so this is a report about a 6700 XT running the gfx1030
  code path; there is no gfx1031 rocBLAS measurement here because ROCm 7.1.1 ships none.
- Not tested: 4K/32K token-id parity between the two arms, Windows, TheRock's venv ROCm, a layer split with a 6700
  XT in it, the vision encoder (`--vision` off), concurrency (`--seqs 1`, no `--batch`), a long soak, other
  quantisations than Swift 1.5 IQ3_S, the other gfx1031 cards (6750 XT, 6800M), ROCm older than 7.1.1, PCIe width,
  RAM speed, temperatures and clock behaviour under load.
