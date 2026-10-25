# 2x AMD Instinct MI50 16 GB (gfx906), Ryzen 7 5700X, 128 GB RAM: Strata 0.2.1, IQ3_S at 131K and IQ4_XS at 262K

Follow-up to [the 0.1.43.1 report](../2026-10-19-community-2x-mi50-5700x/README.md). Same box, moved to the 0.2.1
release (engine 0.1.46) and 128 GB of RAM, with two models measured on the same flags. Everything below is a
measurement on hardware we own; the machine is a headless server used for local inference, not a benchmark rig.

## Provenance

- Strata 0.2.1, engine `strata --version` = **0.1.46**, source build at tag `v0.2.1` (`9076725`).
- Build: `cmake -S . -B build-906 -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_HIP_GFX906=ON
  -DCMAKE_HIP_ARCHITECTURES=gfx906 -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++`, then `ninja -C build-906 strata strata-device`. No local patches.
  With clang 23 (ROCm 7.14) a 16-way build crashed the compiler itself (SIGSEGV in `globalopt`/`cgscc`) on two
  `mmq-instance-*.cu` units; `-j8` plus a retry pass built all targets.
- `strata-device --selftest`: OK.

## Hardware

| | |
| --- | --- |
| GPUs | 2x AMD Instinct MI50 16 GB, gfx906 (`1002:66af`), 200 W power cap each, PCIe 3.0 x8 |
| CPU | AMD Ryzen 7 5700X (8C/16T), stock BIOS (PBO Auto) |
| RAM | 128 GB DDR4 (4x32 GB Kingston KF3200C16D4/32GX), JEDEC **2400 MT/s** |
| OS | Ubuntu 24.04, kernel 6.8.0-147, ROCm 7.14 in `mixa3607/rocm-gfx906:7.14-complete` |
| Storage | 512 GB NVMe (455 GB root LV) |

Memory note: at DOCP 3200 and at manual 3600 the machine hung under sustained inference (an uncorrected machine
check and hard resets), so it runs at the JEDEC 2400 profile. On this engine RAM speed is worth ~2% at most
(measured 2400 vs 3600 in the earlier report), so the loss is not material.

## Models and configuration

- **IQ3_S**: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, `IQ3_S` (2 shards, 78 GB), 131,072-token context.
- **IQ4_XS**: `unsloth/Qwen3.8-Flash-Next-GGUF`, `UD-IQ4_XS` (3 shards, 87 GB), 262,144-token context.
  The pack needs `--compat-bf16`: 195 dense tensors are Q8_0 in this file and the engine wants BF16 for them.
- MTP draft runtime rebuilt from the same base model's draft layer (786 MB). The `MTP/mtp-*-Q8_0.gguf` file in the
  unsloth repository does not feed `mtp_rt.py` unchanged (`KeyError: mtp.layers.0.mlp.experts.gate_up_proj`).
- Vision: CPU encoder (`strata-vision`, mmproj BF16).
- Flags, identical for both models:

```text
--max-context <131072|262144> --kv int8 --kv-resident 32768 --layer-split 27
--spec 3 --spec-min-p 0.9 --prefill 2048 --vram-reserve-mib 1024
--pipeline-windows 2 --trim-stage-weights --vision
```

## Method

- Greedy (`temperature 0`), unique prompt text per run so the prompt cache never helps.
- Short chat: 5 runs, 128-token cap. 1.6K prompt: 5 runs, 128 tokens. 31K prompt: 2 runs, 64 tokens.
- Timing from the engine's own `timings` block (`prompt_per_second`, `predicted_per_second`).
- Quality: 17 tasks with programmatic answers (arithmetic, unit conversion, an executable Python function with
  edge cases, strict JSON, a needle with decoy numbers, Russian case, exact-length constraints).
- Long context: a 140,908-token prompt with a five-digit code in the middle, answer checked.

## Results

| Configuration | Prompt tokens | Decode tok/s | Prompt tok/s | Runs |
| --- | ---: | ---: | ---: | ---: |
| IQ3_S, short chat | 80 | 61.0 | 55 | 5 |
| IQ3_S, 1.6K | 726 | 71.3 | 180 | 5 |
| IQ3_S, 31K | 31,446 | **73.7** | **691** | 2 |
| IQ4_XS, short chat | 80 | 58.6 | 53 | 5 |
| IQ4_XS, 1.6K | 726 | 62.8 | 149 | 5 |
| IQ4_XS, 31K | 31,446 | **73.5** (57.5 before `--ple-io ram`) | 532 | 5 |
| IQ4_XS, 140K | 140,908 | 59.0 | 546 | 1 |

- Expert cache: IQ3_S 12,245 slots / 23.9 GiB, 98.0-99.7% hit; IQ4_XS 9,876 slots / 22.8 GiB, 90.4-94.0% hit
  (bigger experts, same VRAM). Experts in RAM: 46.8 GiB and 55.4 GiB.
- Load to ready: 40 s (IQ3_S) and 48 s (IQ4_XS); switching models between the two takes that long.
- Power under a 31K prompt: ~406 W peak for both cards, 49-54 C edge.

### Layer split: an even 24/24 beats 27/21 on long prompts

The 48 layers can be cut unevenly or in half. Measured with everything else fixed (`--ple-io ram`, 262K context),
five runs each, unique prompt text per run:

| Test | `--layer-split 27` (27/21) | `--layer-split 24` (24/24) | Delta |
| --- | ---: | ---: | ---: |
| 31K prompt, decode median | 70.4 tok/s | 70.3 tok/s | 0% |
| 31K prompt, prompt rate | 533 tok/s | **556 tok/s** | +4.2% |
| 118K prompt, decode median | **62.8 tok/s** | 59.6 tok/s | -5.1% |
| 118K prompt, prompt rate | 548 tok/s | **596 tok/s** | +8.7% |

The even split keeps both cards busy during the prompt pass (+9% at 118K), because the 27/21 cut leaves the second
card waiting for the first one in every window. At depth the uneven split keeps a small decode lead, because the
second card has the smaller expert cache (4,745 slots against 5,131) and more layers on it means more misses.

What that comes to for a request of 118K prompt and ~500 generated tokens: **223 s with 27/21 against 206 s with
24/24**, so the even split is the better default for long-prompt work. On short chats the two are within noise.

## Correctness and limitations

- Quality suite: 17/17 for both models; a second 13-task suite also 13/13 for both. On short, checkable tasks the
  3.06 vs 4.25 bit difference did not show.
- The 140K needle was found (answer `48213`), so the 262,144-token window is usable on 2x16 GB with KV streaming.
- Vision answered correctly on a two-shape image (10-11 s per image, 320 prompt tokens).
- Not measured: 262K prefill beyond 140K, two clients at once, power at 262K, `--kv fp16` (other gfx906 owners use
  it at 262K), and any GPU-encoder path (the encoder runs on the CPU here).

## The PLE table is the real cost here: `--ple-io ram`

This model carries a per-layer-embedding table of **320,001,536 rows at 90 bytes = 28.8 GB** inside the
`-00002-of-00003.gguf` shard. With the default `--ple-io direct` the engine reads those rows from the SSD with
unbuffered reads and a 90 MB row cache (`--ple-row-cache` default 1,048,576 rows), so every token puts a few SSD
reads on the decode path. On this box that showed up as 4-11 MB/s of disk reads during generation and a decode of
**57.5 tok/s** at a 31K prompt.

With 128 GB of RAM there is room for the whole table, so `--ple-io ram` locks it in memory:

```text
strata generate: PLE table locked in RAM (--ple-io ram) in 14.3 s
```

| | `--ple-io direct` (default) | `--ple-io ram` |
| --- | ---: | ---: |
| 31K prompt, decode | 57.5 tok/s (49.0-65.9) | **73.5 tok/s** (64.1-73.6, 5 runs) |
| 31K prompt, prompt rate | 533 tok/s | 532 tok/s |
| disk reads during generation | 4-11 MB/s | 0.02 MB/s |
| RAM | 55.4 GiB of experts | 55.4 + 28.8 GiB, 92 of 125 GB used |
| time to ready | 48 s | 63 s |

That is **+28% decode** and it puts IQ4_XS at the same speed as IQ3_S had on this machine (73.5 vs 73.7 tok/s)
while keeping the 262,144-token window and the 4.25-bit weights. The alternative for a smaller box is a larger
`--ple-row-cache` (90 B per row), which is a bounded compromise between the two.
