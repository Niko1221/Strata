# Community benchmark: RTX 4000 Ada 2-GPU vs 3-GPU layer split

Measured on 2026-10-02 on a Dell PowerEdge R7515 running Ubuntu 24.04.5 LTS. This is a community comparison of the same Qwen3.8-Flash-Next IQ3_S model under two Strata GPU layouts on one host.

The useful result is narrow: on this machine, adding a third, lower-power RTX 4000 SFF Ada did not materially improve decode throughput, while the 16.6K-token fresh-prompt case became substantially slower. This is consistent with a slow extra pipeline stage outweighing additional expert-cache capacity once the two faster cards are already sufficient for this workload.

## Hardware and software

- Dell PowerEdge R7515.
- CPU: AMD EPYC 7313P 16-Core Processor.
- 125.4 GiB usable system RAM.
- NVIDIA driver 580.178.04; CUDA 13.0 reported by `nvidia-smi`.
- GPU 0: NVIDIA RTX 4000 SFF Ada Generation, 20,475 MiB, 70 W power limit.
- GPU 1: NVIDIA RTX 4000 Ada Generation, 20,475 MiB, 130 W power limit.
- GPU 2: NVIDIA RTX 4000 Ada Generation, 20,475 MiB, 130 W power limit.
- Strata engine 0.1.35, locally built for Ada / SM89 with vision disabled.
- Strata source commit: `d9ab8435f654c368c586340d490915f6addf56a3`.

No user names, host names, API keys, LAN addresses, or personal filesystem paths are included in this directory.

## Model and configuration

Model reported by the server: `qwen3.8-flash-next-iq3_s`.

Common settings for both runs:

- IQ3_S model pack.
- 131,072-token context limit.
- INT8 KV.
- Vision disabled.
- Single-request serving.
- OpenAI-compatible `/v1/chat/completions` endpoint on loopback.
- Temperature 0, top-p 1, seed 42 in the local smoke suite.

GPU layouts:

1. **2 GPU:** GPU 1 + GPU 2, the two 130 W full-height cards.
2. **3 GPU:** GPU 1 + GPU 0 + GPU 2, inserting the 70 W SFF card between the two 130 W cards.

Strata selected the layer placement automatically. For the 2-GPU run, the saved startup log reports GPUs `[1, 2]`, auto layer split, and an expert cache of 8,092 experts using 14.56 GiB of VRAM. The exact per-GPU layer boundary was not retained. The corresponding detailed 3-GPU startup log was accidentally overwritten after the experiment, so no exact 3-GPU split boundary or cache-count claim is made here.

## Workload

A local deterministic 12-request smoke suite was replayed unchanged for both layouts. It contains small math, logic, Python, electronics, instruction-following, agent-planning, base-rate, and long-context prompts. The final case is a 16,591-token synthetic archive lookup with two relevant records and many distractors.

The raw request prompts, responses, usage, wall time, Strata timing fields, and MTP draft/acceptance counters are preserved in:

- `results-2gpu.jsonl`
- `results-3gpu.jsonl`

This suite is **not a standardized model-quality benchmark**. Two harness items are intentionally or accidentally unsuitable for naive automatic accuracy scoring:

- `math_02_integer_system` has an incorrect stored expected answer (`1,8,15`); the equations are satisfied by `3,5,16`, which both runs derived.
- `logic_02_implication` is not single-answer as written: both B (`P is false`) and C (`Q is true`) follow from the premises.

Several responses also reach the configured output-token cap before emitting the requested `FINAL:` line. Automatic pass counts are therefore retained only as harness diagnostics and should not be interpreted as model accuracy.

## Results

| Configuration | Suite wall time | Median decode tok/s | Mean decode tok/s | 16.6K prompt tok/s | 16.6K decode tok/s | 16.6K wall time |
|---|---:|---:|---:|---:|---:|---:|
| 2 x 130 W RTX 4000 Ada | 85.044 s | 72.95 | 71.83 | **2,110.3** | 72.9 | **9.843 s** |
| 2 x 130 W + 1 x 70 W RTX 4000 SFF Ada | 97.275 s | 73.70 | 72.08 | **778.6** | 75.0 | **23.146 s** |

For this workload, adding the third card changed median decode throughput by only about **+1.0%**, while the 16.6K fresh-prompt throughput fell by about **63.1%** and the long-request wall time increased by about **135.2%**. Total suite wall time increased by about **14.4%**.

The short-request portion of the suite was nearly unchanged (75.201 s on two GPUs vs 74.129 s on three GPUs). The aggregate slowdown is therefore dominated by the long-prompt case rather than by autoregressive decode.

## Interpretation

This single-host result supports a limited engineering conclusion: an additional GPU is not automatically beneficial for layer-split inference when it is substantially slower than the existing stages. On this R7515, the extra 20 GB of SFF Ada VRAM did not produce a measurable decode advantage in this suite, while long-prompt prefill was much slower with the third stage.

This should not be generalized to all three-GPU systems. A third card may still help when the two faster cards cannot hold enough of the routed expert working set, or when the cards have more closely matched per-layer performance.

## Limitations

- One host, one model, one quantization, one Strata engine build, and one small synthetic workload.
- DRAM bandwidth was not measured during the benchmark.
- Exact auto-selected per-GPU layer boundaries were not retained; the detailed 3-GPU startup log was accidentally overwritten after the experiment.
- The runs were single-shot rather than repeated medians at each prompt length.
- GPU clocks were not fixed.
- The smoke suite was designed for local regression checking, not standardized accuracy measurement.
- No claim is made that the observed slowdown is caused only by GPU power limit; layer allocation, clocks, PCIe behavior, cache residency, and pipeline scheduling can all contribute.

## Raw data

`summary.json` contains the aggregates used in this README. The JSONL files are the original result records with only the shell prompt line containing the local user/host name removed.
