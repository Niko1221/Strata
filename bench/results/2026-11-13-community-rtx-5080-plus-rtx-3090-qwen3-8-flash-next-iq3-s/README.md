# Community benchmark: RTX 5080 + RTX 3090 peer split, Flash-Next GSQ-RCO IQ3_S

Measured on 2026-11-13 on one machine, Flash-Next GSQ-RCO IQ3_S. Multiple arms (single-card baselines and peer-split variants) were benched back to back the same day with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software (all arms)

- GPUs: NVIDIA GeForce RTX 5080 16 GB + NVIDIA GeForce RTX 3090 24 GB (40 GiB VRAM total)
- CPU: AMD Ryzen 9 9950X3D 16-Core Processor (16 cores, 32 threads); installed RAM: 125.4 GiB
- Storage: NVMe (Samsung 990 EVO Plus 1 TB + CT4000T705SSD5)
- OS: Linux 7.2.15-arch1-1; GPU driver: 615.78.08; CUDA (driver supports up to): 13.4; engine built for CUDA 13.0
- Strata: project 0.1.58, commit 146df3c4; engine 0.1.59 (identical in every arm)

Per-GPU detail (read after each arm's run, near idle; rows are listed in the order the tool reported them):

- **A1:** power_limit_w=350.00, power_draw_w=340.68, pcie=gen4 x16, pstate=P2, temp_c=67
- **B1:** power_limit_w=350.00, power_draw_w=323.53, pcie=gen4 x16, pstate=P2, temp_c=66; power_limit_w=360.00, power_draw_w=59.23, pcie=gen4 x4, pstate=P1, temp_c=48
- **B1r:** power_limit_w=350.00, power_draw_w=324.32, pcie=gen4 x16, pstate=P2, temp_c=64; power_limit_w=360.00, power_draw_w=59.26, pcie=gen4 x4, pstate=P1, temp_c=49
- **C1:** power_limit_w=360.00, power_draw_w=77.14, pcie=gen4 x4, pstate=P1, temp_c=50; power_limit_w=350.00, power_draw_w=13.19, pcie=gen1 x16, pstate=P8, temp_c=40

Note: all four arms ran with the RTX 3090 in the x16 slot and the RTX 5080 in a gen4 x4 slot. The 350 W power limit identifies the RTX 3090, the 360 W limit the RTX 5080.

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 655360 tokens; parallel slots: 1
- Settings: the install's own configuration; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: three questions, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words and continue from the model's own earlier answers, 64 new tokens per turn, 3 sessions

Each arm used a freshly loaded engine; other GPU work was closed.

## Arms and results

| Arm | Host | Peer | Difference | Decode short (tok/s) | 4K prompt (tok/s) | 16K prompt (tok/s) | TTFT 4K / 16K (s) | Agent decode (tok/s) | Files |
| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| A1 | RTX 3090 | - | --vram-reserve-mib 2000 | 117.1 (99.0-127.7) | 2290.0 (2274.9-2299.6) | 2960.1 (2871.6-2990.6) | 1.77 (1.76-1.78) / 12.03 (5.39-49.17) | 134.0 (119.2-147.2) | `results-a1.json + engine-log-a1.txt` |
| B1 | RTX 3090 | RTX 5080 (device 1) | --peer-reserve-mib 2000 | 105.6 (89.1-133.4) | 1681.2 (1663.6-1688.6) | 2311.9 (2301.6-2316.9) | 2.36 (2.35-2.38) / 6.82 (6.80-6.85) | 120.9 (107.9-137.2) | `results-b1.json + engine-log-b1.txt` |
| B1r | RTX 3090 | RTX 5080 (device 1) | rerun of B1 with --vram-reserve-mib 3000 | 101.8 (88.8-120.7) | 1698.4 (1698.0-1703.8) | 2343.3 (2342.4-2345.3) | 2.38 (2.38-2.38) / 6.88 (6.87-6.88) | 130.9 (98.7-139.6) | `results-b1r.json + engine-log-b1r.txt` |
| C1 | RTX 5080 | RTX 3090 (peer-device 0) | baseline | 38.8 (37.0-45.5) | 467.3 (465.5-467.7) | 1406.1 (1404.4-1407.9) | 8.63 (8.62-8.69) / 11.46 (11.43-11.47) | 46.0 (36.9-52.2) | `results-c1.json + engine-log-c1.txt` |

Conversation-cache hit over the agent-session follow-up turns was 69-70 % in every arm.

## Full launch commands

### Arm A1

```text
strata --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 655360 --kv int8 --kv-resident 32768 --vram-reserve-mib 2000 --pcie-frac 0.20 --spec-min-p 0.5
```

### Arm B1

```text
strata --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 655360 --kv int8 --kv-resident 32768 --vram-reserve-mib 2000 --pcie-frac 0.20 --spec-min-p 0.5 --peer-device 1 --peer-reserve-mib 2000
```

### Arm B1r

```text
strata --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 655360 --kv int8 --kv-resident 32768 --vram-reserve-mib 3000 --pcie-frac 0.20 --spec-min-p 0.5 --peer-device 1 --peer-reserve-mib 2000
```

### Arm C1

```text
strata --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 655360 --kv int8 --kv-resident 32768 --vram-reserve-mib 2000 --pcie-frac 0.20 --spec-min-p 0.5 --peer-device 0
```

## Findings

- IQ3_S fits one 24 GB card, and the solo RTX 3090 (A1) is the fastest arm on every metric: decode 117.1 tok/s, prompt 2,290 / 2,960 at 4K / 16K.
- Adding the RTX 5080 as an expert peer (B1/B1r) does not help when the file already fits: decode 101.8-105.6 and prompt about 25-30 % below solo.
- Reversing the roles (C1: RTX 5080 host, `--peer-device 0`) is far worse: decode 38.8, 4K prompt 467.3. This arm is confounded — the RTX 5080 host sat on the gen4 x4 slot and no `--peer-reserve-mib` was passed — the cause is not isolated.
- A1's 16K TTFT was unstable (median 12.03 s, range 5.39-49.17 s) although its prompt rate was normal; the peer arms' 16K TTFT (6.8 s) is the steady value.
- Bottom line: for a quant that fits a single card, the peer split did not pay off on this machine in either direction.
- No solo RTX 5080 arm was run for this quant.

## Correctness and limitations

Not measured: this is a speed suite (no recall/needle checks, no 32K arm). The conversation uses the model's own replies, so token counts differ a little between runs. Background load is not recorded beyond the GPU power readings above. Per-request data for every arm is in the `results-*.json` files; engine start-up lines are in the `engine-log-*.txt` excerpts.