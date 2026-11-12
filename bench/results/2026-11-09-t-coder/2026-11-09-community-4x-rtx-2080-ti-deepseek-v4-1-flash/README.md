# Community benchmark on NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti

Measured on 2026-11-09 with `tools/strata_bench.py` (suite 1, 5 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti, 88 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz (14 cores, 28 threads); installed RAM: 121.4 GiB; storage: nvme0n1 Lexar SSD NM620 2TB nvme
- OS: Linux 7.0.0-42-generic; GPU driver: 595.91.07; CUDA (driver supports up to): 13.2
- Strata: project 0.1.55, commit f481fecd; engine ds4-0.1
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=82.30, pcie=gen3 x16, pstate=P2, temp_c=56
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=110.89, pcie=gen3 x16, pstate=P2, temp_c=73
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=138.03, pcie=gen3 x16, pstate=P2, temp_c=70
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=162.10, pcie=gen3 x16, pstate=P2, temp_c=64
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: deepseek-v4.1-flash
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-dsv41 --pack deepseek-v4.1-flash-ist-das-coder --max-context 131072 --eos 1 --vision on --dspark auto --dspark-conf 0.5 --tp 4 --experts 5 --threads on
```

## Method

Run by `python tools/strata_bench.py --long --runs 5` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 5 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 5 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 5 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 32 (29-38) | 0 | 100 (100-100) | 15 | not measured | 42.6 (38.7-59.6) | 0.35 (0.31-0.49) |
| 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 5 | 487.5 (427.5-489.0) | not measured | 8.48 (8.45-9.67) |
| 16K prompt, fresh | 16562 (16562-16562) | 0 (0-0) | 24 cap | 5 | 892.7 (869.5-894.1) | not measured | 18.60 (18.58-19.11) |
| 32K prompt, fresh | 33126 (33126-33126) | 0 (0-0) | 24 cap | 5 | 1054.9 (1054.2-1056.7) | not measured | 31.51 (31.45-31.52) |
| agent session, follow-up turns | 12636 (12630-12645) at the last turn | see below | up to 64 | 5 | see below | 39.1 (32.1-52.5) | 7.13 (4.43-9.87) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.5 % (69.5-69.6); earlier-turn tokens read again per session: 0 (0-2) (earlier turns that the cache did not hold; the 11796 (11796-11798) new tokens of the files are read once); first turn (nothing cached) first token: 2.18 s (2.17-3.40).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 602 | 0 | 602 | 178.0 | 3.401 | 28.1 |
| 1 | 3313 | 614 | 2699 | 314.7 | 8.611 | 38.8 |
| 2 | 4829 | 3376 | 1453 | 245.0 | 5.963 | 35.5 |
| 3 | 7926 | 4879 | 3047 | 310.1 | 9.869 | 40.5 |
| 4 | 10006 | 7986 | 2020 | 308.5 | 6.591 | 43.8 |
| 5 | 12634 | 10057 | 2577 | 311.1 | 8.331 | 46.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 574 s.

## Publication notes

**This is not a standard-protocol result.** It was produced by `tools/strata_bench.py` (the guide's quick-look path), not by
`strata bench run` from [BENCHMARK_PROTOCOL.md](../../../../docs/BENCHMARK_PROTOCOL.md); it is published as a community report,
which the guide allows for measurements made outside the protocol.

### Model identity

The pack is ISTA-DASLab's `DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder` at IQ2_XXS for the routed experts, built from the six
`...-IQ2_XXS-0000N-of-00006.gguf` shards of
[ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF).
Shard 6 of 6 in this copy is the set's Engram table (`DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`, the same file the set
publishes separately as its Engram), symlinked under the shard's name so the pack's six-shard name list resolves; the pack
itself was made with `tools/dsv41/pack_gguf.py` from those six files, the DSpark `Q8_0` drafter and the BF16 vision tower.

- pack descriptor `ds4.txt`, sha256 `b9dc59c07d93ffdcc32c5365d3f4fba72aeefb61d136556eb637bcfc9180d576`;
- pack tensor index `tensors.txt`, sha256 `6ade3221a1b79226bc873f6d8129bdf4e31ca2573c39e456c3c8f114f4d67472`;
- drafter `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf`, sha256
  `6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a` (the file the installer pins);
- vision tower `mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf`.

`--pack-info` on this pack reports 66.0 GB of routed experts at 1.94 bits a weight (272 G parameters), 8 files, and the
Engram 110.81 GB dense.

### Settings

Engine environment on top of the arguments in `config.json` / `env.txt`:

- `DS4_ASYNC_T=8` — keep the drafter's T=6 verify window asynchronous (also the engine's default).
- `DS4_DEFAULTS=075` — the measured 2.00 bpw speed profile of the expert path. **Note:** it was measured on the
  non-coder 2.00 bpw set, not on this coder pack, so it is carried here as the best known setting, not as one tuned for this
  pack. It changes numerics and stays an explicit opt-in token.
- `DS4_ATTN_SPLIT=0` — the default attention split, pinned.
- `DS4_LANE_FUSE=1` — the CPU lane's quantisation fused into its gate/up dispatch (engine default in this build).
- `DS4_GG_IQ1M=auto` — the IQ1_M dot chosen by a startup measurement on this CPU.

### Correctness and limitations

Not measured: answer correctness, and this is a speed suite. Not tested: the HIP/ROCm build (no toolchain here), the Windows
package (no Windows machine), and any card family other than four sm_75 RTX 2080 Ti. The 4K prompt rate (487.5 tok/s) is
lower than the same box reaches on the non-coder 2.00 bpw set (539.4) while 16K and 32K are slightly higher; that is reported
as measured, not explained. The `DS4_DEFAULTS=075` profile was tuned on the other pack. Shard 6 of this pack is the set's
Engram file, as described above; a copy with a differently packaged shard 6 would hash differently.
