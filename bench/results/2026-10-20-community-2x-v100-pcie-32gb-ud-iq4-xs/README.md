# Community benchmark on Tesla V100-PCIE-32GB + Tesla V100-PCIE-32GB (layer split)

Measured on 2026-10-20 by caolonghao with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

The re-measurement of this machine after the 0.2.0-beta tracker reset (the first two-V100 measurement stands; see the comparison at the end).

## Hardware and software

- GPU and VRAM: 2x Tesla V100 **PCIe** 32 GB, 64 GiB VRAM (the server's GPUs 0 and 1) - sm_70, PCI device ID
  `10DE:1DB5`, **no NVLink** (all links inactive; the driver's "V100-SXM2-32GB" name string is a VBIOS label, as
  in NVIDIA forum reports of PCIe V100s), PCIe Gen3 x16 (the engine's probe measured 13.1 GB/s host->device),
  power limit 250 W each (max 300 W, stock PCIe defaults)
- CPU: Intel(R) Xeon(R) Silver 4316 CPU @ 2.30GHz (40 cores, 80 threads); installed RAM: 125.5 GiB; storage: shared NVMe (other load present)
- OS: Linux 6.8.0-138-generic; GPU driver: 580.178.04; driver reports CUDA 13.0, engine built with the **CUDA 12.8 toolkit** (gcc 11.4)
- Strata: upstream `main` at `f57c0adc` (engine 0.1.45.3, 9 commits past `v0.2.0-beta`), **source build** with
  `-DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_EXPERIMENTAL_SM60=ON` (the NVIDIA_V100.md community build); this PR branch
  adds only this report folder
- Background workloads and power limits: GPUs otherwise idle, the suite ran alone on a freshly loaded server

## Model and configuration

- Model and quantization: `unsloth/Qwen3.8-Flash-Next-GGUF` at setup's pinned revision `38bb39ee`, **UD-IQ4_XS**
  (three shards, 93.7 GB); pack built by setup (`--compat-bf16`), setup's own MTP draft layer and expert profile;
  no custom draft vocabulary; vision off
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged - `--expert-cache auto`, `--prefill auto`, MTP
  `--spec 4 --spec-min-p 0.5`, `--kv int8 --kv-resident 32768`, `--remote-expert-opt`, `--layer-split auto`,
  CPU expert pool at its default worker count (18); no calibration, no experimental speed projection, no
  low-RAM mode; thinking off for the suite; temperature 0

```text
strata --pack unsloth-ud-iq4_xs --native Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8 --kv-resident 32768 --remote-expert-opt
```

## Method

Run by `python tools/strata_bench.py --url http://127.0.0.1:8080 --config strata-unsloth-ud-iq4_xs.json --long` (the recorded `--long` command in short) against the HTTP API of a Strata server started beforehand by its run script; model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 83.2 (76.2-106.2) | 0.36 (0.33-0.40) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1418.3 (1349.0-1420.8) | not measured | 2.86 (2.86-3.01) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2063.1 (2043.8-2064.4) | not measured | 7.83 (7.82-7.90) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 2432.3 (2426.8-2434.9) | not measured | 13.23 (13.22-13.26) |
| agent session, follow-up turns | 14631 (14606-14643) at the last turn | see below | up to 64 | 3 | see below | 78.5 (68.0-95.4) | 2.74 (1.73-3.20) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.4 % (69.3-69.5); earlier-turn tokens read again per session: 72 (3-73) (earlier turns that the cache did not hold; the 13748 (13744-13748) new tokens of the files are read once); first turn (nothing cached) first token: 1.17 s (1.10-1.19).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 532.8 | 1.172 | 82.4 |
| 1 | 4059 | 638 | 3421 | 1081.7 | 3.2 | 81.7 |
| 2 | 5592 | 4123 | 1469 | 781.1 | 1.914 | 95.4 |
| 3 | 9025 | 5656 | 3369 | 1083.1 | 3.152 | 78.5 |
| 4 | 11772 | 9018 | 2754 | 992.9 | 2.807 | 91.0 |
| 5 | 14643 | 11836 | 2807 | 1035.9 | 2.744 | 76.8 |

Per-request data: `results.json`. Engine log lines from the start (the server was started by its run script, so the excerpt is cut from the engine log by hand): `engine-log-excerpt.txt` - the auto split moved one layer down from 0.1.40.2 (**K=25 then, K=24 now**: CUDA0 layers 0-23, CUDA1 24-47 + the head; the caches hold 22,462 of 24,576 profiled pairs, ~99.8%, 10,757 of 12,288 on CUDA1).

## Against the same pair on 0.1.40.2 (2026-10-08)

The earlier report used the community `benchmark.py` (fresh nonce prompts, 256-token cap); this one uses `tools/strata_bench.py` (fresh tagged prompts, 24-token cap on the prompt cells). Both take prompt throughput from the engine's own timings with `reused: 0`, so the prompt cells compare engine to engine; the decode cells do not (different output caps and cases) and are left out.

| Prompt tokens | 0.1.40.2 prompt tok/s | 0.1.45.3 prompt tok/s |
| ---: | ---: | ---: |
| 4,096 | 1,277.8 (1,214.6-1,279.0) | 1,418.3 (1,349.0-1,420.8) |
| 32,768 | 2,355.3 (2,351.8-2,357.4) | 2,432.3 (2,426.8-2,434.9) |

Same model, pack, settings and machine; +11.0% and +3.3%. The earlier report's context ladder and concurrency cells were **not re-run on 0.1.45.3**; their 0.1.40.2 numbers, for reference only: ladder decode 81.2 / 77.5 / 73.9 / 74.8 / 70.6 tok/s at 131K / 262K / 393K / 512K / 1M context (the top two rungs YaRN x2 / x4, 450K-token prompts), and 117.7 tok/s total across 8 concurrent requests (14.9 each) with `--batch 8 --batch-groups 2 --trim-stage-weights` at split 25.

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; storage background load is unmonitored; power limits are read from `nvidia-smi`, not an external meter. `results.json` records commit `4f7612f1` - a report-only branch whose tree is `f57c0adc` plus `bench/results` files. One earlier suite attempt on the same hardware was discarded before any cell was kept: started in `--url` mode without `--config`, the tool had labelled the machine from the wrong GPU; no number in this report comes from it. The suite also ran in `--url` mode because the tool's own server-start path crashes (upstream issue #3188). Total time of the suite: 148 s.

Prepared with an AI assistant (Claude Code) from the raw measurements; the numbers are the engine's and the client's own output, unedited.
