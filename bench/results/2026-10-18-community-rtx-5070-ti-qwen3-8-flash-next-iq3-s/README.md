# Community benchmark on NVIDIA GeForce RTX 5070 Ti: IQ3_S speed, and its quality against Q8_0

Measured on 2026-10-18 by [Rui-Shi](https://github.com/Rui-Shi) with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. The speed suite has no correctness check; the same day, on the same PC and engine, IQ3_S was scored against Unsloth's Q8_0 with `tools/quant_quality.py`: [Quality against Q8_0](#quality-against-q8_0) below, all data and scripts in [quality/](quality/README.md).

Decode 94.5 tok/s (short chat), prompt 2,823 / 3,482 / 3,500 tok/s at 4K / 16K / 32K fresh tokens. Against Q8_0, this PC's settings pick the same top token at 87.1% of 4,995 positions (mean KL 0.143, perplexity 4.036 against 3.997).

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5070 Ti, 16 GiB VRAM
- CPU: Intel(R) Core(TM) Ultra 7 270K Plus (24 cores, 24 threads); installed RAM: 63.5 GiB; storage: not measured
- OS: Windows 11 (10.0.26300); GPU driver: 616.92; CUDA: not read
- Strata: project 0.1.45, commit 477ac13c; engine 0.1.45
- Background workloads and power limits: desktop programs open (browser, terminal), nothing heavy running; power limits not recorded
- Settings: calibrated on 0.1.41 with setup's `calibrate_config` (`--pcie-frac 0.00 --spec-min-p 0.70`); `--kv-grow`, `--vram-reserve-mib 1000`, the conversation-cache flags and the two env switches set by hand

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Files: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` IQ3_S (`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000{1,2}-of-00002.gguf`, 54,817,524,224 and 28,800,138,432 bytes, downloaded by setup on 2026-10-03); the n-gram table `ple-fp8.gguf` from the same day, 51,200,246,112 bytes (the current setup.py pins a 51,200,246,176-byte file; not checked further); the MTP draft layer from setup (`mtp/rt`)
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf ple-fp8.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 262144 --kv int8 --kv-grow --vram-reserve-mib 1000 --conversation-cache-mib 4096 --conversation-cache-slots 2 --conversation-cache-min-free-mib 256 --pcie-frac 0.00 --spec-min-p 0.70
```

Environment of the config: `STRATA_IQ3S_MT1=1`, `STRATA_ROUTE_TAIL_SKIP=0`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 94.5 (80.6-106.8) | 0.41 (0.38-0.47) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 2823.1 (2782.1-2840.9) | not measured | 1.50 (1.45-1.57) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 3481.5 (3481.0-3503.7) | not measured | 4.65 (4.61-4.66) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 3499.8 (3498.0-3509.1) | not measured | 9.21 (9.18-9.21) |
| agent session, follow-up turns | 14560 (14546-14583) at the last turn | see below | up to 64 | 3 | see below | 104.6 (76.6-152.3) | 1.26 (0.96-1.57) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.8); earlier-turn tokens read again per session: 72 (3-75) (earlier turns that the cache did not hold; the 13719 (13719-13719) new tokens of the files are read once); first turn (nothing cached) first token: 1.01 s (0.91-1.08).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 692.0 | 0.914 | 92.4 |
| 1 | 3757 | 634 | 3123 | 2356.6 | 1.37 | 104.5 |
| 2 | 5290 | 3820 | 1470 | 1529.7 | 0.98 | 88.3 |
| 3 | 8723 | 5283 | 3440 | 2239.0 | 1.569 | 115.6 |
| 4 | 11622 | 8760 | 2862 | 2276.5 | 1.288 | 152.3 |
| 5 | 14546 | 11647 | 2899 | 2343.6 | 1.27 | 106.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Quality against Q8_0

`tools/quant_quality.py` (teacher-forced, 4,995 positions of `data/quant_quality/v1`), Unsloth's Q8_0 run on this PC as
the reference (39 GiB of its experts in RAM, the rest read from the GGUF). Details, method, per-text tables and the
observations: [quality/README.md](quality/README.md).

| Run | Same top token as Q8_0 | Mean KL | KL p99 | Perplexity (Q8_0: 3.997) |
| --- | ---: | ---: | ---: | ---: |
| IQ3_S, experts on the CPU (the yardstick's default) | 87.51% | 0.1421 | 1.81 | 4.061 |
| IQ3_S, this PC's settings (GPU expert cache, tail skip off, int8 K/V, FP8 n-gram table) | 87.11% | 0.1435 | 1.81 | 4.036 |
| the same with `STRATA_ROUTE_TAIL_SKIP=7` and FP16 K/V | 86.67% | 0.1540 | 2.13 | 4.062 |
| experts on the CPU with the GGUF's IQ4_NL n-gram table | 87.43% | 0.1454 | 1.92 | 4.113 |

Per position against Q8_0 (95% ranges of 32-position blocks): the tail skip adds +0.0104 KL, at the edge of zero
overall (+0.0004 to +0.0196) and separated on four of the five texts; int8 K/V against FP16 adds nothing measurable
(-0.0001, -0.0083 to +0.0084); the IQ4_NL table against FP8 +0.0033 overall (not separated) and +0.030 on the agent
text. Per text, IQ3_S's perplexity is within 1% of Q8_0's on code, docs and the other languages
and 8-13% above it on the agent transcript.

## Running the tools on Windows

- `tools/strata_bench.py` 0.1.45 stopped twice before measuring when it started the server itself: `NameError: name
  'port' is not defined` (line 990, #3096, fix in #3098), then `UnicodeDecodeError` reading `CMakeLists.txt` on this
  Chinese-locale Windows (GBK default; #2792, fix in #3092). This run used the unchanged tool through a wrapper that
  sets the module's `port` to the config's 8080, with `PYTHONUTF8=1`.
- `tools/quant_quality.py` puts `lib_dirs` into `LD_LIBRARY_PATH` (not read on Windows) and its free-RAM check reads
  `/proc/meminfo`; see [quality/README.md](quality/README.md#how-the-runs-were-made).

## Correctness and limitations

The speed suite itself has no correctness check (the quality runs above are separate). The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 95 s.
