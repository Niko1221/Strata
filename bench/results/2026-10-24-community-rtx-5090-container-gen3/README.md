# Community benchmark: RTX 5090 in a container, PCIe Gen3 ×16, two Xeon Gold 5218

Measured by [XiaoBinGan](https://github.com/XiaoBinGan) on 2026-10-21 and 2026-10-24, on a shared
two-socket Linux server that another team also uses. This tests Strata 0.2.2 (engine 0.1.47) in the
official Docker image with the original Flash-Next **Q2_0**, no source build, one GPU, and a
32,768-token context.

Two standard runs are reported here. They use the same machine, image, model and settings; the only
difference is how much of the one graphics card other programs were holding at the time. That makes
them a measurement of what a shared card costs this engine, which is the reason to publish both:

| | 2026-10-21 (card shared) | 2026-10-24 (card exclusive) |
| --- | ---: | ---: |
| Other programs on the GPU | **18,106 MiB** in 3 containers | 0 |
| Expert cache the engine chose | **5,784 slots, 7.45 GiB** | **19,530 slots, 25.14 GiB** |
| Decode at 4,096 prompt tokens | 132.6 tok/s | **205.7 tok/s** |
| Prompt at 4,096 | 1,710.5 tok/s | **3,520.6 tok/s** |
| Decode at 16K | 134.8 tok/s | **218.4 tok/s** |
| TTFT at 16K | 4.99 s | **2.99 s** |
| Sanity checks | 11 of 12 | **12 of 12** |

Medians of three runs each, and the two runs are three days apart on the same installation, so this
is a comparison of the same engine under two memory conditions rather than a version comparison.

## Hardware and software

- NVIDIA GeForce RTX 5090; 32,607 MiB reported VRAM; 575 W power limit. **PCIe: Gen 3, x16** — the
  card reports a maximum of Gen 3 on this mainboard, and the engine's own startup probe measured
  **13.9 GB/s host-to-device**, which is why it selected `pcie_frac` 0.38 instead of its 0.55 default.
  GPU clocks were not fixed for these runs.
- 2 × Intel Xeon Gold 5218 at 2.30 GHz; 32 physical cores, 64 logical CPUs, 2 NUMA nodes. The CPUs
  have AVX-512F/DQ/CD/BW/VL/VNNI but not AVX512-BF16, and the engine says so at startup:
  `this CPU has no AVX-512: the expert kernels run on AVX-2`. NUMA pool split is on.
- 125.6 GiB installed RAM, no swap. The model's expert arena is 31.64 GiB of it.
- Host: Ubuntu 20.04.3 LTS, kernel 5.15.0-58-generic, glibc 2.31, NVIDIA driver 580.142.
- Docker 19.03.15 (client and server), default seccomp profile plus AppArmor.
- Image `ghcr.io/niko1221/strata:0.2.2-cuda13`, pulled from its mirror `ghcr.nju.edu.cn`
  (`0.2.6` and other newer tags are not published as images — see #4183). The image is
  2.2 GB on disk and runs Ubuntu 24.04.5. The engine package is
  `0.1.47-linux-x86_64-cuda-cuda13`; the Terminal reports commit `011c9e6`.
- The container is started with `--gpus device=1 --security-opt seccomp=unconfined`. The second
  option is needed on this daemon: its default seccomp profile denies `access(X_OK)`, so the shipped
  launcher prints "no usable Python found" although the bundled runtime is there (issue #4309,
  fix in #4312). The engine itself needs nothing else.
- The graphics card was dedicated to Strata while it was serving. The other card in the machine
  (RTX 4090, 48 GB) belongs to a different service and was not used here. Other services on the host
  (a vector database, a workflow server, rerankers) kept running; the machine is not isolated.

## Model and configuration

Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, size Q2_0. The two shards were downloaded
through the installer from ModelScope (`STRATA_SOURCE=modelscope`) and their **SHA-256 matches the
values pinned in this repository's `setup.py`** (`ISTA_FILES`):

| File | SHA-256 |
| --- | --- |
| `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` | `69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720` |
| `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |

The image encoder is `mmproj-Qwen3.8-Flash-Next-BF16.gguf`. The MTP draft layer was fetched by the
installer. The bench runner could not hash the model files itself (it runs inside the container and
the files were mounted from the host), so both runs carry status **nonconforming** for that one
reason; the hashes above were taken on the host with `sha256sum` and are the repository's own pinned
values.

Settings in both runs: `MODEL=Q2_0`, `CONTEXT=32k`, `KV=int8`, `PARALLEL=1`, `TUNE=skip`, `GPU=all`,
expert cache `auto`, prefill chunks 8,192 tokens, MTP `--spec 4 --spec-min-p 0.5`, and the protocol's
sampling (temperature 1.0, top_p 0.95, top_k 20, reasoning `xhigh`, thinking on, 512-token cap).

**Two settings differ between the runs, and both are stated here on purpose.** The 2026-10-21 run had
the picture reader off (`VISION=off`; its `/health` reported `images: false`), the 2026-10-24 run had
it on the CPU (`VISION=cpu`, `images: true`). No request in either run contains a picture, and the
engine reports the reader as `strata-vision: on the CPU, 16 threads`, so it works from host memory;
the `expert cache auto` line is the direct evidence of how much VRAM each run actually gave the
cache. The other difference is the one this report is about: the other programs holding the card.

The engine sized its expert cache from the VRAM it found free, which is the whole point of the two
runs:

```text
# 2026-10-21, three other containers holding 18,106 MiB of the card
strata generate: expert cache auto: 8.31 GiB free, 700 MiB reserved (+184 MiB for the draft head) -> 5784 slots
strata generate: expert cache 5784 slots, 7.45 GiB of VRAM

# 2026-10-24, card exclusive
strata generate: expert cache auto: 26.01 GiB free, 700 MiB reserved (+184 MiB for the draft head) -> 19530 slots
strata generate: expert cache 19530 slots, 25.14 GiB of VRAM
```

In both cases the engine ends with 410 MiB of VRAM free and warns that this is below the 512 MiB it
would like (`add --vram-reserve-mib 832`); that argument is not accepted by `strata launch` yet
(issue #4134), so both runs keep the default.

## Method and reproduction

`strata bench run` from inside the running container, against the server on loopback, following
`docs/BENCHMARK_PROTOCOL.md` (draft): the frozen `standard-v1` requests one at a time, streaming, one
warm-up excluded, three repetitions of the short, 4K and 16K prompts, then the twelve sanity checks.
Prompt and decode speed are the engine's own `timings`; TTFT is the client's monotonic clock to the
first non-empty reasoning, answer or tool-call delta. Nothing was retried.

```bash
docker exec -e STRATA_API_KEY="$KEY" strata \
  strata bench run --url http://127.0.0.1:8080 --skip-model-hash \
    --out /data/state/bench-exclusive --track default \
    --track-notes "RTX 5090 32GB, PCIe Gen3 x16, Q2_0, 32K, int8 KV, MTP spec4, container, card exclusive"
```

`--skip-model-hash` is why the runs are nonconforming; the hashes are verified above instead. The
2026-10-21 run used the same command with the card shared.

This folder is one complete run, and the shared run sits beside it in its own folder so that each
`results.json` finds the files it declares:

```text
2026-10-24-community-rtx-5090-container-gen3/          # the 2026-10-24 run, card exclusive
  README.md  results.json  config.json  outputs.jsonl
  provenance/{chat_template.jinja, corpus-build.json, corpus-manifest.sha256}
  shared-gpu-2026-10-21/                              # the 2026-10-21 run, card shared
    results.json  config.json  outputs.jsonl  provenance/...
```

`outputs.jsonl` holds every request, its usage and its timings, and `provenance/` is the corpus and
chat template the run declares; both runs used the same ones, so the hashes declared in the two
`results.json` files are the same.

## Results

2026-10-24, card exclusive (this folder's `results.json`; status nonconforming: model hash skipped):

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 235.4 (222.0-266.3) n=3 | 0.23 (0.23-0.28) n=3 | 0.55 (0.49-0.87) n=3 | 1.8 (1.5-2.1) n=3 | 0/3 |
| prompt-4k | 3520.6 (3493.5-3553.9) n=3 | 205.7 (202.2-269.5) n=3 | 1.18 (1.16-1.18) n=3 | 3.56 (1.83-3.67) n=3 | 3.7 (3.0-3.7) n=3 | 2/3 |
| prompt-16k | 5528.9 (5442.0-5532.0) n=3 | 218.4 (211.4-221.4) n=3 | 2.99 (2.99-3.04) n=3 | 4.74 (4.65-5.08) n=3 | 5.3 (5.3-5.4) n=3 | 3/3 |

Sanity checks **12 of 12**. Time after the model loaded: 0 min 47 s.

2026-10-21, card shared with 18,106 MiB held by three containers (`shared-gpu-2026-10-21/results.json`):

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 136.8 (135.2-144.3) n=3 | 0.39 (0.36-0.41) n=3 | 0.94 (0.81-1.00) n=3 | 2.8 (2.4-3.3) n=3 | 0/3 |
| prompt-4k | 1710.5 (1710.0-1710.5) n=3 | 132.6 (127.8-142.9) n=3 | 2.42 (2.41-2.42) n=3 | 4.66 (4.13-5.19) n=3 | 6.3 (6.0-6.4) n=3 | 3/3 |
| prompt-16k | 3314.4 (3308.9-3319.4) n=3 | 134.8 (124.5-142.7) n=3 | 4.99 (4.97-5.00) n=3 | 7.60 (6.66-8.32) n=3 | 8.8 (8.6-9.1) n=3 | 3/3 |

Sanity checks **11 of 12**: T4 asked for one structured tool call and got none. Time after the model
loaded: 1 min 21 s.

## What the two runs say (and what they do not)

- The card held five times the expert cache in the exclusive run (25.14 vs 7.45 GiB, 19,530 vs 5,784
  of the model's 24,576 expert slots), and the engine answers roughly 50-70% faster for it: decode
  +55% at 4K, +62% at 16K, prompt +106% / +67%, TTFT -51% / -40%.
- With 25.1 GiB of the 24,576 slots resident the engine reports no expert read over PCIe at all: a
  separate per-request sample of this run's own engine log (1,000 requests generating 150-260 tokens,
  medians below) reads **0.00% in every one of them**, against **0.50% median, 2.20% at the 90th
  percentile** in the shared run, whose expert hit rate was 86.5% against 99.7%. On a **Gen 3 ×16**
  link that difference is what the numbers above measure, so a reader on Gen 4/Gen 5 hardware should
  not carry this ratio over. That sample is not part of the standard benchmark, which reports no PCIe
  figure; it comes from the same two containers' engine logs and is disclosed here, not in the tables
  above.
- The one failed sanity check in the shared run is the tool-call case, and it passed in the exclusive
  run. **I did not establish the cause.** The two runs differ in free VRAM and in how many requests
  the server had answered before them; they are not a controlled test of answer quality, and 3
  repetitions per prompt cannot establish one either.
- These are single-GPU numbers with one request at a time (`total_slots: 1`). This machine runs
  `PARALLEL=1` on purpose: in separate runs on the same card, `parallel=4` was slower (single-stream
  decode 186 vs 204 tok/s, four-client decode 58 tok/s, aggregate 179 vs 220 tok/s), because the
  expert cache is divided among the slots.

## Failures, exclusions and limitations

- No request failed in either run. No request was retried.
- Both runs are **nonconforming** for one reason only: the model file hash was skipped, because the
  bench runner inside the container cannot read the host-mounted files it was started with. The
  SHA-256 of both shards is in this report and matches the repository's pinned values.
- Prompt-prefix reuse is the shared template prefix only; the expert cache stays warm and evolves
  between requests. These are not cold-model measurements.
- Output caps include reasoning, and a capped request's total time is the time to the cap, not to an
  answer: 2 of the 3 4K runs and all 3 16K runs reached the cap in the exclusive run.
- The runner cannot see other programs on the machine, the GPU clocks, or the model repository's
  revision; `results.json` lists every value it could not read under `missing`.
- **The main limit is the second run's condition:** stopping three containers on a shared server is
  not something a reader can reproduce on their own machine, the two runs are three days apart, and
  the picture reader was off in the first and on the CPU in the second (see *Model and
  configuration*). Treat the pair as a description of what a shared card costs this engine, not as a
  controlled A/B.
- PCIe Gen 3 ×16 is unusual for an RTX 5090 and is the most important thing to know before comparing
  these numbers with a Gen 4/Gen 5 report: this machine's link moves 13.9 GB/s host-to-device, and
  the engine says so itself.
- Storage speed was not measured. The model files live on a shared volume with 1.7 TB free.
- Windows, AMD, macOS, multi-GPU, vision and DeepSeek were not tested here.
