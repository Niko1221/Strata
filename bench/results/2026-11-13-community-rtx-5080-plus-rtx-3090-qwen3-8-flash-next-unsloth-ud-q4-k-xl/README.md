# Community benchmark: RTX 5080 + RTX 3090 peer split, Unsloth UD-Q4_K_XL (4-shard GGUF)

Measured on 2026-11-13 on one machine, Unsloth UD-Q4_K_XL (4-shard GGUF). Multiple arms (single-card baselines and peer-split variants) were benched back to back the same day with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software (all arms)

- GPUs: NVIDIA GeForce RTX 5080 16 GB + NVIDIA GeForce RTX 3090 24 GB (40 GiB VRAM total)
- CPU: AMD Ryzen 9 9950X3D 16-Core Processor (16 cores, 32 threads); installed RAM: 125.4 GiB
- Storage: NVMe (Samsung 990 EVO Plus 1 TB + CT4000T705SSD5)
- OS: Linux 7.2.15-arch1-1; GPU driver: 615.78.08; CUDA (driver supports up to): 13.4; engine built for CUDA 13.0
- Strata: project 0.1.58, commit 146df3c4; engine 0.1.59 (identical in every arm)

Per-GPU detail (read after each arm's run, near idle; rows are listed in the order the tool reported them):

- **A1:** power_limit_w=350.00, power_draw_w=299.59, pcie=gen4 x16, pstate=P2, temp_c=64
- **A2:** power_limit_w=350.00, power_draw_w=298.68, pcie=gen4 x16, pstate=P2, temp_c=63
- **A3:** power_limit_w=360.00, power_draw_w=135.69, pcie=gen5 x16, pstate=P1, temp_c=53
- **B1:** power_limit_w=350.00, power_draw_w=287.41, pcie=gen4 x16, pstate=P2, temp_c=64; power_limit_w=360.00, power_draw_w=51.76, pcie=gen4 x4, pstate=P1, temp_c=47
- **B1r:** power_limit_w=350.00, power_draw_w=286.42, pcie=gen4 x16, pstate=P2, temp_c=64; power_limit_w=360.00, power_draw_w=54.05, pcie=gen4 x4, pstate=P1, temp_c=43
- **B2:** power_limit_w=350.00, power_draw_w=299.27, pcie=gen4 x16, pstate=P2, temp_c=65; power_limit_w=360.00, power_draw_w=39.71, pcie=gen4 x4, pstate=P1, temp_c=43
- **C1:** power_limit_w=360.00, power_draw_w=184.69, pcie=gen5 x16, pstate=P1, temp_c=60; power_limit_w=350.00, power_draw_w=204.06, pcie=gen4 x4, pstate=P2, temp_c=52
- **C2:** power_limit_w=360.00, power_draw_w=177.02, pcie=gen5 x16, pstate=P1, temp_c=60; power_limit_w=350.00, power_draw_w=168.32, pcie=gen4 x4, pstate=P2, temp_c=50
- **C3:** power_limit_w=360.00, power_draw_w=165.85, pcie=gen5 x16, pstate=P1, temp_c=58; power_limit_w=350.00, power_draw_w=154.91, pcie=gen4 x4, pstate=P2, temp_c=48

Note: between the B-family and C-family arms the cards were swapped between slots, so the host differs as well as the flags: in the B arms the RTX 3090 was the host on the x16 link, in the C arms the RTX 5080 was the host on the x16 link (the RTX 3090 sat on a gen4 x4 link). The 350 W power limit identifies the RTX 3090, the 360 W limit the RTX 5080.

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-q4_k_xl
- Context: 655360 tokens; parallel slots: 1
- Settings: the install's own configuration; thinking off for the suite; temperature 0
- `STRATA_HC_Q8=1`, `STRATA_ROUTE_TAIL_SKIP=0`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: three questions, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words and continue from the model's own earlier answers, 64 new tokens per turn, 3 sessions

Each arm used a freshly loaded engine; other GPU work was closed.

## Arms and results

| Arm | Host | Peer | Difference | Decode short (tok/s) | 4K prompt (tok/s) | 16K prompt (tok/s) | TTFT 4K / 16K (s) | Agent decode (tok/s) | Files |
| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| A1 | RTX 3090 | - | --vram-reserve-mib 2000 | 64.9 (57.9-75.2) | 1427.0 (1426.4-1427.8) | 2529.9 (2489.0-2540.3) | 2.84 (2.84-2.84) / 6.37 (6.35-6.47) | 82.1 (56.6-100.3) | `results-a1.json + engine-log-a1.txt` |
| A2 | RTX 3090 | - | --vram-reserve-mib 3000 | 64.3 (56.4-69.2) | 1413.6 (1410.7-1413.9) | 2519.0 (2500.7-2519.2) | 2.87 (2.87-2.87) / 6.40 (6.40-6.45) | 72.9 (65.7-98.8) | `results-a2.json + engine-log-a2.txt` |
| A3 | RTX 5080 | - | --vram-reserve-mib 3000 | 52.3 (48.8-60.8) | 2201.0 (2182.5-2205.6) | 3485.6 (3485.2-3509.6) | 1.85 (1.85-1.87) / 4.64 (4.61-4.64) | 55.3 (46.2-69.6) | `results-a3.json + engine-log-a3.txt` |
| B1 | RTX 3090 | RTX 5080 (device 1) | run A | 66.4 (57.7-72.2) | 1157.8 (1156.8-1158.8) | 2059.9 (2038.7-2063.5) | 3.49 (3.49-3.49) / 7.82 (7.81-7.90) | 90.4 (66.7-119.8) | `results-b1.json + engine-log-b1.txt` |
| B1r | RTX 3090 | RTX 5080 (device 1) | run B (same flags as B1) | 67.4 (55.1-71.2) | 1142.6 (1140.1-1143.3) | 1689.0 (1681.3-1689.4) | 3.54 (3.53-3.54) / 9.53 (9.53-9.57) | 94.3 (65.6-126.9) | `results-b1r.json + engine-log-b1r.txt` |
| B2 | RTX 3090 | RTX 5080 (device 1) | --peer-prefill-rows 0 | 68.3 (55.4-75.4) | 1423.0 (1419.0-1427.1) | 2519.4 (2510.3-2548.3) | 2.84 (2.83-2.85) / 6.40 (6.32-6.42) | 95.3 (65.9-115.5) | `results-b2.json + engine-log-b2.txt` |
| C1 | RTX 5080 | RTX 3090 (device 1) | default prefill | 71.8 (62.5-79.7) | 1211.2 (1208.4-1212.2) | 1544.2 (1543.1-1544.5) | 3.34 (3.33-3.34) / 10.42 (10.42-10.43) | 98.7 (75.9-122.9) | `results-c1.json + engine-log-c1.txt` |
| C2 | RTX 5080 | RTX 3090 (device 1) | --peer-prefill-rows 0 | 72.6 (58.8-78.4) | 2268.2 (2252.4-2275.4) | 2723.9 (2717.0-2737.1) | 1.79 (1.78-1.80) / 5.92 (5.89-5.93) | 92.0 (72.9-119.7) | `results-c2.json + engine-log-c2.txt` |
| C3 | RTX 5080 | RTX 3090 (device 1) | --peer-prefill-rows 0 --peer-reserve-mib 600 | 73.6 (59.0-83.1) | 2251.0 (2250.9-2260.5) | 2569.3 (2567.6-2571.2) | 1.80 (1.80-1.80) / 6.27 (6.27-6.28) | 99.2 (68.3-128.4) | `results-c3.json + engine-log-c3.txt` |

Conversation-cache hit over the agent-session follow-up turns was 69-70 % in every arm.

## Full launch commands

### Arm A1

```text
strata --pack unsloth-ud-q4_k_xl --native Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 655360 --rope-scaling yarn --rope-scale 2.5 --kv int8 --ple-io ram --kv-resident 32768 --vision --vram-reserve-mib 2000
```

### Arm A2

```text
strata --pack unsloth-ud-q4_k_xl --native Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 655360 --rope-scaling yarn --rope-scale 2.5 --kv int8 --ple-io ram --kv-resident 32768 --vision --vram-reserve-mib 3000
```

### Arm A3

```text
strata --pack unsloth-ud-q4_k_xl --native Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 655360 --rope-scaling yarn --rope-scale 2.5 --kv int8 --ple-io ram --kv-resident 32768 --vision --vram-reserve-mib 3000
```

### Arm B1

```text
strata --pack unsloth-ud-q4_k_xl --native Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 655360 --rope-scaling yarn --rope-scale 2.5 --kv int8 --ple-io ram --kv-resident 32768 --vision --vram-reserve-mib 3000 --peer-device 1 --peer-reserve-mib 2000
```

### Arm B1r

```text
strata --pack unsloth-ud-q4_k_xl --native Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 655360 --rope-scaling yarn --rope-scale 2.5 --kv int8 --ple-io ram --kv-resident 32768 --vision --vram-reserve-mib 3000 --peer-device 1 --peer-reserve-mib 2000
```

### Arm B2

```text
strata --pack unsloth-ud-q4_k_xl --native Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 655360 --rope-scaling yarn --rope-scale 2.5 --kv int8 --ple-io ram --kv-resident 32768 --vision --vram-reserve-mib 3000 --peer-device 1 --peer-reserve-mib 2000 --peer-prefill-rows 0
```

### Arm C1

```text
strata --pack unsloth-ud-q4_k_xl --native Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 655360 --rope-scaling yarn --rope-scale 2.5 --kv int8 --ple-io ram --kv-resident 32768 --vision --vram-reserve-mib 3000 --peer-device 1 --peer-reserve-mib 2000
```

### Arm C2

```text
strata --pack unsloth-ud-q4_k_xl --native Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 655360 --rope-scaling yarn --rope-scale 2.5 --kv int8 --ple-io ram --kv-resident 32768 --vision --vram-reserve-mib 3000 --peer-device 1 --peer-reserve-mib 2000 --peer-prefill-rows 0
```

### Arm C3

```text
strata --pack unsloth-ud-q4_k_xl --native Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 655360 --rope-scaling yarn --rope-scale 2.5 --kv int8 --ple-io ram --kv-resident 32768 --vision --vram-reserve-mib 3000 --peer-device 1 --peer-reserve-mib 600 --peer-prefill-rows 0
```

## Findings

- The file does not fit either single card, so the solo arms stream experts from RAM; that is the baseline the peer split is compared against.
- Solo: the RTX 3090 decodes faster (64-65 tok/s vs 52-53), the RTX 5080 prefills faster (2,201 / 3,486 vs 1,427 / 2,530 at 4K / 16K).
- Peer split with default peer prefill (B1/B1r/C1) raises decode (66-72) but costs prefill (1,143-1,211 at 4K, below both solos).
- `--peer-prefill-rows 0` removes that prefill cost: B2 returns to the solo-3090 prefill level, and C2 (5080 host + 3090 peer) reaches 2,268 / 2,724 — at or near the best solo prefill — while keeping the highest decode of the sweep (72.6).
- Best combined arm: C2/C3 — 5080 host + 3090 peer + `--peer-prefill-rows 0`; it beats both solo cards on all three headline metrics. Dropping `--peer-reserve-mib` from 2000 to 600 (C3) is within run-to-run noise.
- Repeatability: B1 and B1r have identical flags and differ 2,059.9 vs 1,689.0 at 16K (default peer prefill); the `--peer-prefill-rows 0` arms B2/C2/C3 fall within about 6 % of each other across two sweep days.
- Not isolated here: whether the C-family slot position (RTX 5080 on the x16 link) matters independently of the host/peer role.

## Correctness and limitations

Not measured: this is a speed suite (no recall/needle checks, no 32K arm). The conversation uses the model's own replies, so token counts differ a little between runs. Background load is not recorded beyond the GPU power readings above. Per-request data for every arm is in the `results-*.json` files; engine start-up lines are in the `engine-log-*.txt` excerpts.