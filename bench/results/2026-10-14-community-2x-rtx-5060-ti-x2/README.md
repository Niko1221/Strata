# Community benchmark (rerun): 2x RTX 5060 Ti (16 GB), Xeon E5-2690 v4, engine 0.1.44.1

Measured on 2026-10-14 by R6DJO. Third run on this machine, after the
[2026-10-02 report](../2026-10-02-community-rtx-5060-ti-x2/README.md) (engine 0.1.35) and the
[2026-10-05 rerun](../2026-10-05-community-rtx-5060-ti-x2/README.md) (engine 0.1.39). Same
hardware, same model (original Flash-Next GSQ-RCO IQ3_XXS), two GPUs with a fixed layer split.
The launch configuration changed again since 2026-10-05, so the differences below are not
attributable to the engine version alone.

Full fresh-prompt sweep, three runs each at 4,096 / 32,768 / 128,000 prompt tokens with a
256-token output cap, plus six long-context recall checks. Median decode throughput was
**111.2 tok/s at 4K, 101.5 tok/s at 32K, 101.9 tok/s at 128K**; median prompt throughput was
**865 / 2,000 / 2,776 tok/s** (2026-10-05: 84.8 / 86.5 / 83.8 decode, 911 / 2,087 / 2,440 prompt,
with a different launch configuration). All nine sweep runs were fully fresh prompts
(`reused: 0`) and ended on the output cap. These are synthetic code-explanation requests with
greedy decoding; they do not establish general answer quality or performance on other workloads.

## Hardware and software

- 2x NVIDIA GeForce RTX 5060 Ti, 16,311 MiB reported VRAM each, 180 W power limit, compute
  capability 12.0. GPU clocks were not fixed for this test. PCIe link max Gen3 x16 per
  `nvidia-smi`; the idle link read Gen2 x8 on both cards at query time (the engine's own
  start-up PCIe probe is not in the log excerpt for this run).
- Intel Xeon E5-2690 v4 @ 2.60 GHz; 1 CPU (1 socket), 14 cores / 28 threads; no AVX-512 (the
  engine logged "the expert kernels run on AVX-2"). 13 expert-pool workers + the host thread.
  Motherboard HUANANZHI X99-BD4 (from the 2026-10-05 report; unchanged hardware).
- 128 GB installed RAM as 4x 32 GB DDR4-2133 ECC, quad-channel (from the 2026-10-05 report);
  71 GB MemAvailable at benchmark start. Swap: a 2 GB swapfile. Storage: repository and packs
  on a Netac NVMe SSD 1 TB; GGUF shards on a SATA SSD (MSI S270 960 GB) at /mnt/data.
- Linux Mint 22.3, kernel 7.0.0-31-generic, NVIDIA driver 610.57.04 (CUDA UMD 13.3).
- Engine **0.1.44.1** (reported by `/v1/status` and `/metrics`), locally compiled,
  `CMAKE_BUILD_TYPE=Release`, `CUDA_ARCHITECTURES=120`, CUDA toolkit 13.4, gcc 13.3.0;
  `engine/BUILD.json`: source local, `src 7dfd34378bd2d32a`, archs [120], vision gpu.
  Repository at `main` (`296b59d2`, in sync with origin at run time).
- Background workloads: none during the sweep; the server was freshly restarted for this run
  and the machine was otherwise idle. Regular desktop services on the same CPU remained
  running; this was not a fully isolated system.

## Model and configuration

Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, `IQ3_XXS` two-shard GGUF; the shard sizes
(47,039,860,096 and 28,800,138,432 bytes) match the 2026-10-05 report byte for byte; the files
were not re-downloaded and their SHA-256 hashes were not re-recorded this time. Names in
[model-files.txt](model-files.txt). Custom pack `Strata-data/packs/iq3_xxs`; the MTP draft
head is the released one (`Strata-data/mtp/rt`; its `draft_vocab.bin` was regenerated earlier
the same day, hash not recorded). Vision: none. Expert profile
`expert-profile-learned-iq3_xxs.bin` (24,576 ranked pairs, SHA-1 `d683338cca550ffb276e7348eb622ce52d3e1880`,
saved by `--expert-profile-save`), served model id `qwen3.8-flash-next-iq3_xxs`.

Context 204,800 (`--max-context`); KV int8 with a 32,768-token resident window per QSA layer
(the rest streams from 1.21 GiB of pinned RAM); expert cache auto, 14,086 resident experts
(8,232 slots / 11.92 GiB on CUDA0, 5,854 slots / 10.64 GiB on CUDA1 — the split is uneven
because CUDA1 also holds the session KV and the draft head); prefill `auto:16384`; MTP on
(`--spec 4 --spec-min-p 0.70`, engine reported `spec: 6, mtp_max: 4`); layer split fixed 24/24
(`--layer-split 24`; CUDA0 layers 0–23, CUDA1 24–47); `--pipeline-windows 2` (272 MiB kept out
of the CUDA0 cache); `--trim-stage-weights`; conversation cache 32,768 MiB / 6 slots;
`--vram-reserve-mib 900`; `--pcie-frac 0.20`; adaptive expert swaps on
(`--adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97`). The default
`STRATA_ROUTE_TAIL_SKIP=7` was on. Compared to 2026-10-05 this drops `--remote-expert-opt`
and changes the context (262,144 -> 204,800), split (25 -> 24), prefill (`auto` -> `auto:16384`),
conversation cache (8,192 MiB -> 32,768 MiB + 6 slots), reserve (700 -> 900 MiB),
`--pcie-frac` (0.19 -> 0.20), and the expert profile file.

- Sampling: temperature 0 (greedy), `reasoning_effort: "none"`.

```text
<strata-dir>/.venv/bin/python <strata-dir>/serve/server.py \
  --engine strata --config <strata-dir>/strata-iq3_xxs.json \
  --port 31000 --host 0.0.0.0
# config strata-iq3_xxs.json passes to the engine:
#   --pack .../packs/iq3_xxs --native <shard 1> --ple-gguf <shard 2>
#   --expert-profile expert-profile-learned-iq3_xxs.bin --expert-cache auto
#   --prefill auto:16384 --spec 4 --mtp .../mtp/rt --max-context 204800
#   --kv int8 --kv-resident 32768 --vram-reserve-mib 900 --pipeline-windows 2
#   --trim-stage-weights --conversation-cache-mib 32768 --conversation-cache-slots 6
#   --pcie-frac 0.20 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 80
#   --adapt-decay 0.97 --layer-split 24 --expert-profile-save expert-profile-learned-iq3_xxs.bin
```

## Method

The benchmark script is [the serial fresh-prompt script from the RTX 5090 community
report](../2026-09-30-community-rtx-5090/benchmark.py) (copy in this folder), run against this
server's tokenizer pack. For each of 4,096 / 32,768 / 128,000 prompt tokens it performs 3 runs;
each request carries a unique nonce so no prompt is ever reused (`reused: 0` on all nine runs,
confirmed by the engine's per-request metrics). One warm-up request (16 output tokens)
preceded the measured runs; model loading is not included in any timing. The server had been
restarted for this run; the expert cache was cold at start-up and warmed by the warm-up and the
measured runs themselves (adaptive swaps were on throughout, as in normal use).

Requests are greedy (`temperature 0`), `reasoning_effort: "none"`, with a 256-token output cap;
all nine runs ended on the cap (`finish_reason: "length"`, 256 generated tokens each). Prompt
and decode tok/s are computed from the engine's per-request timings (`prompt_ms`, `decode_ms`),
not from total request time. TTFT and elapsed time are client-side streaming measurements
(keep-alives and empty deltas excluded; the first token is answer text, reasoning is off).
RAM/VRAM use was not sampled during the runs (`not measured`). Median per-run decode expert hit
rate: 0.980 at 4K, 0.997 at 32K, 0.998 at 128K. Speculative drafts were accepted on 401 of 524
offered (4K), 405 of 534 (32K), 412 of 547 (128K) across each size's three runs. Canceled or
failed requests: none.

Per-run data: [results.json](results.json) (one row per run with engine timings, usage, and
response text), per-run `tokens-*-raw.json` / `tokens-*-request.json`, the run summary in
[summary.json](summary.json), the server status at start in
[initial-status.json](initial-status.json), the engine's start-up lines in
[engine-startup-lines.txt](engine-startup-lines.txt) (home paths replaced by `<home>`), and the
final `/metrics` snapshot in [engine-timing-lines.txt](engine-timing-lines.txt).

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median (min-max) | Decode tok/s median (min-max) | TTFT s median (min-max) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| IQ3_XXS, 2 GPUs, 4,096-token prompt | 4,096 | 0 | 256 | 3 | 865 (826–865) | 111.2 (92.1–116.0) | 4.76 (4.76–4.99) |
| IQ3_XXS, 2 GPUs, 32,768-token prompt | 32,768 | 0 | 256 | 3 | 2,000 (1,994–2,037) | 101.5 (99.8–106.1) | 16.44 (16.15–16.49) |
| IQ3_XXS, 2 GPUs, 128,000-token prompt | 128,000 | 0 | 256 | 3 | 2,776 (2,770–2,846) | 101.9 (100.8–102.1) | 46.27 (45.13–46.36) |

Total client-side request time (median): 7.0 s at 4K, 18.9 s at 32K, 48.8 s at 128K. The first
4K run decoded at 92.1 tok/s (cache still adapting); the other two were 111–116.

Compared with the 2026-10-05 run on this machine (engine 0.1.39, launch configuration listed
above): prompt 911 -> 865 (4K), 2,087 -> 2,000 (32K), 2,440 -> 2,776 (128K) tok/s; decode
84.8 -> 111.2, 86.5 -> 101.5, 83.8 -> 101.9 tok/s; TTFT 4.54 -> 4.76, 15.82 -> 16.44,
52.87 -> 46.27 s. Engine version and launch configuration changed together, so no part of these
deltas is attributable to a single change.

## Correctness and limitations

Long-context recall was checked with
[`tools/needle_bench.py`](../../../tools/needle_bench.py) at 32K and 128K prompt tokens, needle
at 10% / 50% / 90% depth: **all six checks found the code word**; see
[needles.json](needles.json). Actual prompt lengths: 32,778–32,779 tokens (17–23 s per request,
including generation) and 125,044–125,046 tokens (41–45 s). The needle runs reused 32,768
prefix tokens in total through the conversation cache (the three depths share each haystack);
the sweep runs above did not reuse anything.

Limitations: one quantization (IQ3_XXS), one workload (synthetic code explanation, greedy),
output capped at 256 tokens, GPU clocks not fixed, desktop services running on the same
machine, RAM/VRAM not sampled during the runs, and a single engine session. The engine version
(0.1.39 -> 0.1.44.1) and the launch configuration changed together since the previous report,
so this rerun does not isolate the cause of the differences. These numbers do not establish
answer quality beyond the needle checks or performance on other workloads.
