# Community benchmark: RX 6700 XT with the RX 6000 package

Measured 2026-10-22 on an AMD Radeon RX 6700 XT 12 GB (gfx1031), using the official RX 6000 engine package from Strata 0.2.2. This is a three-run `tools/strata_bench.py --long` result for the aligned package settings, collected after an amdgpu reset. It complements the private-session replay results in the parent report; it does not measure the package's launcher defaults after reset.

## Hardware and software

- GPU: AMD Radeon RX 6700 XT, Navi 22, gfx1031; 12,272 MiB VRAM; PCIe 16.0 GT/s x16; power cap 186 W (sysfs).
- CPU: AMD Ryzen 7 5700G, 8 cores / 16 threads; installed RAM: 62 GiB.
- Storage: NVMe. OS: Ubuntu 24.04.5 LTS, kernel 7.0.0-38-generic; kernel driver: amdgpu. ROCm is not installed on the host.
- Strata source revision: v0.2.2 (`011c9e60`); package engine 0.1.47, `strata-engine-0.1.47-linux-x86_64-hip-rx6000`, package runtime ROCm 7.13.0a20260515. Docker 29.9.1.
- The benchmark ran in a container. Background workloads were not recorded.

## Model and configuration

- Model source and revision were not recorded; the model was Qwen3.8-Flash-Next GSQ-RCO IQ2_XS. GGUF parts `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf` and `...00002-of-00002.gguf`; IQ2_XS pack and MTP `rt`.
- Context 196,608; one parallel slot; int8 KV with a 32,768-token resident window; expert cache `auto`; prefill `auto`; MTP 4; vision off; thinking off; temperature 0.
- Package settings: `--pcie-frac 0.2 --spec-min-p 0.7`, `STRATA_HIP_PROMPT_F16=1`, `STRATA_PREFILL_CPU_SHARE=auto`. These are the aligned settings from the replay comparison, passed to the package launcher as `STRATA_SET` and environment variables.
- Method: one warm-up, then three short chats (up to 100 output tokens each), three fresh prompts at each length (24 output tokens each), and three agent sessions (one initial turn plus five follow-ups, up to 64 output tokens per turn). Fresh prompts used unique tags; no prompt tokens were reused there. Thinking was off.
- Command: `python3 tools/strata_bench.py --url http://127.0.0.1:8081 --long --config ~/strata-pkg-data/install/strata-iq2_xs.json --out DIR` (`--runs 3` default). The API key was supplied through `STRATA_KEY`; its value is not recorded.

## Results

Speeds are medians with ranges from the three runs, except short chat which contains nine requests.

| Test | Prompt tokens | Prompt tok/s | Decode tok/s | First token |
|---|---:|---:|---:|---:|
| Short chat | 46 (43–51) | — | 32.9 (27.6–40.1) | 1.08 s (0.88–1.29) |
| Fresh 4K prompt | 4,019 | 567.8 (541.2–568.1) | — | 7.12 s (7.11–7.47) |
| Fresh 16K prompt | 16,058 | 684.2 (683.8–688.9) | — | 23.52 s (23.35–23.53) |
| Fresh 32K prompt | 32,049 | 674.4 (672.8–675.1) | — | 47.58 s (47.53–47.69) |
| Agent follow-up turns | 14,553 at last turn | see `results.json` | 40.6 (23.6–45.4) | 5.62 s (3.65–6.97) |

The sampler recorded 396 seconds. Of 387 samples with GPU busy at least 80%, 361 (93.3%) had a 1,000 MHz memory clock. This meets the approximately 90% busy-sample criterion used in the handoff. The test is a speed suite; no correctness check was run.

The round-one source build, run first after a separate reset, measured 40.7 tok/s chat decode and 579 / 693 / 660 tok/s at 4K / 16K / 32K. The aligned package results here are 32.9 tok/s and 568 / 684 / 674 tok/s. These runs had similar memory clocks while busy, but they were separate runs and differed in expert profile and vision setup; do not attribute the differences to the engine package alone.

## Files

- `raw/community-aligned-after-reset/results.json`: per-request results from the benchmark tool.
- `raw/community-aligned-after-reset/bench-output.txt`: benchmark stdout with the local output path removed.
- `raw/community-aligned-after-reset/samples.txt`: one-second GPU samples (epoch, memory and shader clocks, GPU busy, free VRAM, power, temperature).
- `raw/community-aligned-after-reset/engine-command.txt` and `engine.log`: actual package engine command, environment flags, and engine output.
- The parent report's `raw/` directory contains the earlier replay outputs and configurations.
