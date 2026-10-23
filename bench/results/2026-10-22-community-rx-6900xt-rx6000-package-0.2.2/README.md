# Community benchmark on 2x AMD Radeon RX 6900 XT: the 0.2.2 `rx6000` engine package, first run on a real RX 6000 card

Measured on 2026-10-22 by Vop3p. Two things were tested: (1) whether the Linux `rx6000` engine package of Strata 0.2.2 (engine 0.1.47, "built, not tested on a real RX 6000 card") runs on gfx1030 and how fast it is against source builds with the same options; (2) why its decode under `--pipeline-windows 2` reads 20% below a 0.1.44 source build. The second question is answered: it is the server's model-card `presence_penalty 1.5` default for non-thinking requests (issue #2940), which sends the request down the serial decode loop; with `presence_penalty 0` in the run config the package decodes as fast as 0.1.44. Main limitation: synthetic prompts, 256-token answers, two runs per prompt length, no answer-quality check.

## Hardware and software

- GPU: 2x AMD Radeon RX 6900 XT 16 GB (gfx1030; one reference card, one Sapphire Nitro+), each on PCIe 4.0 x8 (the probe reads 14.1 GB/s host to device per card); both cards at a 245 W power cap and a -81 mV core voltage offset (the production setting of this box, left in place for every arm); CPU Ryzen 5 5600X (6 cores, no AVX-512); 128 GB DDR4-3200; model files on SATA SSD, the expert arena in RAM.
- OS Ubuntu 26.04, kernel 7.0.0-38-generic, ROCm 10.0 (`/opt/rocm`, HIP 7.15) for the driver and the source builds. The package carries its own ROCm 7.1 user-space libraries (`libamdhip64.so.7`, rocBLAS 5, hipBLASLt 1) with `RUNPATH=$ORIGIN`; `ldd` resolves all 21 of them inside the package, none from `/opt/rocm`.
- Strata: the `strata-engine-0.1.47-linux-x86_64-hip-rx6000.tar.gz` asset of the v0.2.2 release (sha256 in `rx6000-package-sha256.txt`, `BUILD.json` copied to `rx6000-package-BUILD.json`: source `f-amd-vision 76f90e58436d`, archs gfx1030 + gfx1031, `strata --version` prints `strata 0.1.47`). Source builds for comparison, all with the same CMake options (`-DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1030 -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ_KQUANTS=ON`, ROCm 10.0 clang, upstream ggml): v0.2.2 (011c9e60, engine 0.1.47), v0.2.0 (e88ee39d, engine 0.1.45.3), v0.1.44 (ed4773eb), v0.1.43.1 (a0116b27). The API server was `serve/server.py` of the v0.2.2 checkout for every arm (the 0.1.4x source arms used their own checkout's server).
- Background: nothing else on the GPUs (the production llama-swap slots were parked for the duration); `monitor_amd.py` telemetry sampled every second.

## Model and configuration

- Model: Qwen3.8-Flash-Next, ISTA-DASLab GSQ-RCO **IQ3_S** (`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000{1,2}-of-00002.gguf`), a native expert pack built by Strata's setup, the `rt` MTP draft head, no vision encoder, no custom pack. The expert profile (`expert-profile.bin`) is the one setup wrote for this box on 2026-10-07.
- Context 131072, `--kv int8 --kv-resident 32768`, `--expert-cache auto --prefill auto --ple-io ram`, `--vram-reserve-mib 700`. One card: `"gpu": 1`. Two cards: `"gpu": [1, 0]`, `"layer_split": "auto"` (0-23 / 24-47), `"gpu_order": "as_given"`.
- Speculation: `--spec 4 --spec-min-p 0.5` (the setup default) or `--spec 3 --spec-min-p 0.7 --pipeline-windows 2`; stated per arm. No environment variables except `STRATA_DECODE_TIMING=1` and `STRATA_MTP_STAGE0=1` where stated. `"parallel"` unset (single stream).
- Sampling: the run configs `iq3s-single.json` and `iq3s-split.json` have **no** `sampling` block; `iq3s-split-pp0.json` adds `"sampling": {"temperature": 0, "presence_penalty": 0}`. The requests send no sampling fields and `chat_template_kwargs.enable_thinking: false`. Since 0.1.45 (a8091f4d) the server then applies the model card's instruct set, which includes `presence_penalty: 1.5`, unless the run config pins it.

```text
# one arm, e.g. the package on the split with pipeline windows (config.json in each arm folder has the exact argv)
python serve/server.py --engine strata --config /bench/split-pkg-s3pw2/config.json --port 8089
python benchmark.py --root /work/terminal-0.2.2 --pack /m/packs/iq3_s --url http://127.0.0.1:8089 --out /bench/split-pkg-s3pw2 --targets 4096,32768,128000 --runs 2
```

## Method

`benchmark.py` (copied here): a 19-token warmup, then prompts of 4,096 / 32,768 / 128,000 tokens (a repeated paragraph with a fresh nonce per request, so nothing is reused: `reused 0` on every request), 256 new tokens each, temperature 0 in the request, two runs per length. Speeds are the engine's own `prompt_per_second` and `predicted_per_second` (newly read tokens over prompt time; generated tokens over decode time, MTP tokens included). TTFT and elapsed are client-side over loopback. Each arm is a fresh engine process; loading is excluded. The order of the arms is the order of the table (not interleaved: each question compares arms run within the same hour, and the day-to-day drift of this box is below 1% on prompts, 2-5% on decode). `results.json` keeps every request with its draft counts; `engine-log-excerpt.txt` keeps the start-up lines, the per-request `strata serve: prompt ...` lines and, where enabled, the `decode timing` lines.

## Results

Medians of the two runs, prompt tok/s / decode tok/s; ranges in `summary.json`. "serial" = the engine log contains `--pipeline-windows: a request with repetition penalties (penalty_last_n) decodes serially`.

| arm folder | engine | cards | spec / pipeline | run-config sampling | 4K | 32K | 128K | stalls | serial |
|---|---|---|---|---|---|---|---|---|---|
| single-pkg-s4 | package 0.1.47 | 1 | 4 / 0.5, none | none | 479 / 43.1 | 505 / 45.3 | 490 / 45.4 | 0 | - |
| split-pkg-s4 | package 0.1.47 | 2 | 4 / 0.5, none | none | 465 / 60.9 | 768 / 62.3 | 873 / 63.0 | 0 | - |
| split-pkg-s3pw2 | package 0.1.47 | 2 | 3 / 0.7, pw 2 | none | 462 / 61.7 | 759 / 59.1 | 873 / 63.0 | 0 | yes |
| split-pkg-syslibs-s3pw2 | package, system ROCm 10 libs via LD_LIBRARY_PATH | 2 | 3 / 0.7, pw 2 | none | 464 / 61.8 | 762 / 59.6 | 875 / 61.4 | 0 | yes |
| split-src022-s3pw2 | source v0.2.2 | 2 | 3 / 0.7, pw 2 | none | 466 / 59.5 | 765 / 56.0 | 875 / 61.6 | 0 | yes |
| split-src020-s3pw2 | source v0.2.0 (0.1.45.3) | 2 | 3 / 0.7, pw 2 | none | 463 / 59.3 | 761 / 56.0 | 874 / 61.8 | 0 | yes |
| split-src1440-s3pw2 | source v0.1.44 | 2 | 3 / 0.7, pw 2 | none | 458 / 72.7 | 761 / 77.2 | 872 / 67.9 | 0 | no |
| split-src1431-s3pw2 | source v0.1.43.1 | 2 | 3 / 0.7, pw 2 | none | 460 / 74.0 | 761 / 74.6 | 871 / 73.4 | 0 | no |
| split-src1440-pw2-timing | source v0.1.44, `STRATA_DECODE_TIMING=1` | 2 | 3 / 0.7, pw 2 | none | 460 / 72.8 | 762 / 75.0 | 872 / 73.8 | 0 | no |
| split-src022-pw2-timing | source v0.2.2, `STRATA_DECODE_TIMING=1` | 2 | 3 / 0.7, pw 2 | none | 463 / 59.4 | 761 / 55.9 | 873 / 61.7 | 0 | yes |
| split-src022-pw2-stage0 | source v0.2.2, `STRATA_DECODE_TIMING=1 STRATA_MTP_STAGE0=1` | 2 | 3 / 0.7, pw 2 | none | 463 / 59.6 | 724 / 56.0 | 810 / 63.8 | 0 | yes |
| split-src022-pw2-pp0 | source v0.2.2, `STRATA_DECODE_TIMING=1` | 2 | 3 / 0.7, pw 2 | `presence_penalty 0` | 465 / **76.0** | 760 / **75.5** | 873 / **73.3** | 0 | no |
| split-pkg-pw2-pp0 | package 0.1.47, `STRATA_DECODE_TIMING=1` | 2 | 3 / 0.7, pw 2 | `presence_penalty 0` | 462 / **74.8** | 759 / **75.9** | 873 / **68.7** | 0 | no |

What the table says:

1. **The package runs on gfx1030** (one card and a two-card layer split, prompts to 128K, 0 stalls, every arm answered 256 tokens per request) and is as fast as a source build with setup's options on the same commit: prompt speed is the same number (within 1%) in every two-card row, decode within about 2%. The bundled ROCm 7.1 libraries make no difference against the system ROCm 10 libraries (rows 3 and 4). The one-card package arm (4/0.5) reads 479 / 505 / 490 tok/s and decodes 43-45 tok/s, against 477 / 506 / 489 and 45-47 for a 0.1.43.1 source build on 2026-10-13 (`2026-10-13-community-rx-6900xt-ud-q4_k_xl-iq3_s-0.1.43.1`).
2. **`--pipeline-windows 2` does nothing on 0.1.45+ for a request that inherits the card's presence penalty** (rows 3-6 against 7-8: 56-63 against 68-77 tok/s decode; prompt speed and draft acceptance unchanged). The `decode timing` lines locate it: 0.2.2 without a sampling block spends 31-34 ms per window = verify 27.6-30.4 + commit/emit 0.7-1.0 + **draft 2.3-2.8** (the serial loop's accounting), 0.1.44 and 0.2.2 with `presence_penalty 0` spend 25-28 ms with draft and commit at 0.00 (hidden behind the other window). The log line `decodes serially` is printed once per process. `STRATA_MTP_STAGE0=1` changes nothing on the serial path. Tracked in #2940 (reported for 0.1.45 by Cass67); the engine and the package are not the cause, the run-config default is, and `"sampling": {"presence_penalty": 0}` restores the 0.1.44 numbers on both the package and the source build.
3. Not a regression of prompt reading: 4K / 32K / 128K prompt speeds are 458-466 / 759-768 / 871-875 tok/s in every two-card row from 0.1.43.1 to 0.1.47.

Telemetry (`monitor_amd.py`, one sample per second, not included; peaks over the two-card package arms): 15.8-15.9 GiB VRAM in use per card once loaded, 250-253 W per card under load (the cap is 245 W; the sysfs reading overshoots briefly), `gpu_busy_percent` 99 on both cards, edge temperature 84 °C on the reference card and 70 °C on the Nitro+, 81.9 GiB of the 121.5 GiB RAM in use at the peak (the expert arena in RAM), swap untouched.

## Correctness and limitations

- Needle recall (`tools/needle_bench.py --lengths 32k,128k --depths 10,50,90`, added 2026-10-23 against the package on the split with spec 3/0.7, `--pipeline-windows 2` and `presence_penalty 0`, the `split-pkg-pw2-pp0` arm's configuration): **6 of 6 found** (31,643-31,644 and 125,551-125,553 prompt tokens; 42 s per 32K read, 143-144 s per 128K read, the last 128K case reused the prefix of the one before it: 77 s). Raw: `needle/needle.out`, `needle/needles.json`. No other answer-quality check; the speed prompts are synthetic. The 256-token answers were produced in every request (`finish_reason: length`), draft acceptance 80-85% with spec 3/0.7 and about 65% with spec 4/0.5, as on the source builds.
- Two runs per prompt length and the arms not interleaved; the differences this report rests on (20% on decode, the `draft` term of the timing line, the log line) are far outside the 2-5% decode spread between runs, the 1-2% ones (package against source) are not conclusions, only "no difference seen".
- One card was measured with the package only at spec 4/0.5; the Windows `rdna2` package was not tried; gfx1031 was not tried.
- The box's -81 mV offset and 245 W cap are its production setting and were not changed between arms; they do not affect decode on this hardware (measured separately) but a stock box will read slightly lower prompt speeds (2-3%).

Developed with an AI coding assistant; every number above was measured on the machine described here.
