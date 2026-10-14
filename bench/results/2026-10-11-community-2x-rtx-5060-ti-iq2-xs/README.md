# Community benchmark: 2x RTX 5060 Ti 16 GB (layer split), IQ2_XS, 262K

Measured on 2026-10-11 by Kurokosama. Two consumer Blackwell cards on an AM4
desktop run the original Flash-Next IQ2_XS at the model's native 262,144-token
window, split across both cards. The 5060 Ti pair is already in the index, but the
entries there are IQ3_XXS / IQ3_S on Xeon and i9 hosts on engines 0.1.34-0.1.39;
this one is IQ2_XS on 0.1.43 with the adaptive expert tier and `--spec-min-p 0.55`,
which is what this machine is actually served with. Measured with
`tools/strata_bench.py --long` against an already running server: three runs per
test, medians and ranges. Speed only: no correctness check was run.

## Hardware and software

- **GPU**: 2x NVIDIA GeForce RTX 5060 Ti 16 GB (16311 MiB each), driver 595.91.07
- **GPU 1 / card 0**: MSI PCI_E1, the CPU-attached x16 slot — negotiates **PCIe 3.0 x8**
- **GPU 2 / card 1**: MSI PCI_E3, the chipset slot — negotiates **PCIe 3.0 x4**
- **Link probes** (the engine's own, host→device): **6.7 GB/s** on card 0 and **3.1 GB/s** on card 1
- **CPU**: AMD Ryzen 5 4500 (Zen 2), 6 cores / 12 threads, **no AVX-512** (AVX-2 expert kernels); 8 pool workers + the host thread
- **Motherboard**: MSI B550-A PRO (MS-7C56), BIOS A.L1 (2025-09-19)
- **RAM**: 64 GB DDR4 in three modules (32 GB TEAMGROUP-UD4-2666 + 2x 16 GB Samsung M378A2K43BB1-CPB), linked at **2133 MT/s** — the mixed set runs below the modules' 2666 rating; 62.7 GiB usable
- **Storage**: WD Blue SN570 1 TB NVMe (model, pack, MTP layer; PLE reads from the same drive)
- **OS**: Ubuntu 24.04.5 LTS, kernel 7.0.0-38-generic
- **CUDA**: 13.0 (nvcc V13.0.88); source build, `CMAKE_CUDA_ARCHITECTURES=120`
- **Strata**: project 0.1.43, commit `7695859a`; engine 0.1.43, source build
- **Background**: 19 Chrome processes (desktop browser left open); no other GPU workload

Neither card runs at what the GPU can do, and the two are not equal: an RTX 5060
Ti is a PCIe 5.0 x8 part, but on this AM4 platform the CPU-attached slot negotiates
gen 3, and the second card sits behind the B550 chipset at x4. Expert traffic that
misses the VRAM cache crosses that link (the engine's probes read 6.7 and 3.1 GB/s
host→device), and the 6-core CPU runs the expert pool. That is the main reason this
report's prompt rates sit below the other 5060 Ti pair rows in the index
(Xeon E5-2690 v4, 911-2,087 tok/s at 4K-32K); decode is in the same band.

## Model and configuration

- Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, **IQ2_XS**; shards
  39,225,954,592 + 28,800,138,432 bytes. The repository revision was not recorded
  at download time.
- Native pack `packs/iq2_xs` (1.5 GB), MTP draft layer `mtp/rt` (787 MB), the
  10,752-pair expert profile from `data/`.
- Context 262,144 (the model's native window); `--kv int8 --kv-resident 32768`;
  `--prefill auto:32768`; vision on (GPU encoder); experimental speed projection
  on (`project`, per-layer, layers 4-44, 41 steered).
- Non-default engine settings kept from earlier tuning on this machine:
  `--spec-min-p 0.55` (the shipped value is 0.85), the adaptive expert tier at
  `--adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97` (engine default
  4 / 96 / 0.7), `--pipeline-windows 2`, `--vram-reserve-mib 700`,
  `--conversation-cache-mib 4096`. `STRATA_STAGE_TRIM=1` is set in the unit.

```text
serve/server.py --engine strata --config strata-iq2_xs.json --port 8081 --open

engine argv:
  --pack <data root>/packs/iq2_xs
  --native <data root>/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
  --ple-gguf <data root>/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf
  --expert-profile <strata checkout>/data/expert-profile.bin
  --prefill auto:32768 --spec 4 --spec-min-p 0.55 --mtp <data root>/mtp/rt
  --max-context 262144 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700
  --control-vector-scaled <strata checkout>/data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1
  --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer
  --pool-workers 8 --conversation-cache-mib 4096 --conversation-cache-slots 8
  --conversation-cache-min-free-mib 2560 --pipeline-windows 2
  --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97 --layer-split 27
```

Start-up shape from the engine log (`engine-log-excerpt.txt`): layer split 27 —
CUDA0 holds layers 0-26, CUDA1 layers 27-47; expert cache 7,941 slots (10.53 GiB)
on CUDA0 and 8,195 slots (11.18 GiB) on CUDA1, 16,136 slots in total;
`prompt chunk auto: 19968 tokens`; `STRATA_ROUTE_TAIL_SKIP=7 is ON (default)`;
R4 hit path on.

## Method

`python tools/strata_bench.py --url http://127.0.0.1:8081 --long --runs 3`, run
against the server that was already serving. Model loading is not timed and the
server was not restarted for the run. One warm-up chat and a 1K prompt are
discarded first; every prompt starts with a unique tag, so the conversation cache
cannot help the fresh-prompt tests (0 reused tokens in all nine prompt requests).

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K / 32K prompts: generated word lists of 4,019 / 16,058 / 32,049 actual
  tokens, 24 new tokens each, 3 fresh prompts per size
- agent session: system prompt and four tools, then 5 turns that each add about
  1,500 words of a project file and continue from the model's own earlier answers,
  64 new tokens per turn, 3 sessions

Speeds are the engine's own timings (`prompt_per_second`, `predicted_per_second`);
first-token times are measured by the client over HTTP. Suite time: 254 s.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 85.3 (58.6-104.7) | 0.36 (0.32-0.66) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 732.6 (729.1-735.6) | not measured | 5.51 (5.49-5.54) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1139.8 (1138.6-1142.7) | not measured | 14.13 (14.09-14.14) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1367.2 (1352.0-1370.0) | not measured | 23.50 (23.45-23.76) |
| agent session, follow-up turns | 14466 (14453-14506) at the last turn | see below | up to 64 | 3 | see below | 92.8 (65.0-114.6) | 5.37 (4.36-5.74) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that
returned a token. The first chat run of each session is visibly slower (58.6-85.9
tok/s against 82.2-104.7 later); the suite's single warm-up does not fully cover a
cold expert tier on this machine.

The prompt rates sit below the two existing 5060 Ti pair rows (911-2,087 tok/s at
4K-32K on a Xeon E5-2690 v4, 0.1.39). The candidates on this box are the CPU
(6 cores / 12 threads, no AVX-512, against 14-28 threads there) and the link
(gen 3 x8 / x4; the engine's own probes read 6.7 and 3.1 GB/s host→device). Decode
is in the same band as those rows.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.6-68.8). Earlier-turn tokens read
again per session: 2 (1-74). First turn (nothing cached) first token: 3.13 s
(2.74-3.20).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 224.3 | 2.743 | 34.5 |
| 1 | 3702 | 622 | 3080 | 558.0 | 5.551 | 65.0 |
| 2 | 5235 | 3765 | 1470 | 334.0 | 4.435 | 69.4 |
| 3 | 8679 | 5299 | 3380 | 592.1 | 5.744 | 77.3 |
| 4 | 11600 | 8742 | 2858 | 531.2 | 5.417 | 84.6 |
| 5 | 14506 | 11661 | 2845 | 534.2 | 5.367 | 80.1 |

Per-request data: `results.json`. Engine start-up lines: `engine-log-excerpt.txt`.
The launch configuration used, with local paths and the LAN origin replaced by
placeholders: `strata-iq2_xs.json`.

## Correctness and limitations

- Speed only: no needle recall test, no vision request and no answer-quality
  check was run.
- One engine session, one request at a time; `--clients` (concurrency) was not run.
- 19 Chrome processes were open during the run, which is the normal state of this
  desktop; the numbers carry whatever CPU that costs on a 6-core CPU.
- The PCIe link is the narrowest part of this build (gen 3 x8 / x4), so the prompt
  rates here should not be read as what the same cards give on a gen 4/5 board.
- The engine log excerpt is sanitized: local paths are `<data root>`,
  `<strata checkout>`.
- The suite's prompt test uses generated word lists. A separate, self-written
  harness on the same session, with repeated filler text and longer contexts, read
  32K prompts at 1,495 tok/s and 55K at 1,677 tok/s. The two prompt shapes are not
  interchangeable, so the tables above are the suite's numbers only.
- Untested: other quantizations, concurrent clients, thinking on, prompt lengths
  above 32,049 tokens.
