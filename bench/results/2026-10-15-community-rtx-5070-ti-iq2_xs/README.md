# Community benchmark on RTX 5070 Ti 16 GB (Linux), Flash-Next IQ2_XS, setup's own install

Measured on 2026-10-15 (06:01-06:06 CDT) by [qiuzan2001](https://github.com/qiuzan2001). Strata 0.1.44.1 source
build, the original Qwen3.8-Flash-Next in IQ2_XS (a row of the README's RTX 5070 table, and the size the 64 GB row of
[MODELS.md](../../../docs/MODELS.md) recommends), installed by `setup.sh` with its own choices (65,536-token context).
The official suite (`tools/strata_bench.py --long`, 3 runs per test), a 256-token decode sweep at about 4K, 32K and
58K prompt tokens (3 runs each) and needle recall at 32K and 64K. Main limitation: one engine session, about five
minutes of measurements, on a desktop in use.

## Hardware and software

- **GPU:** NVIDIA GeForce RTX 5070 Ti, 16,303 MiB, VBIOS 98.03.58.00.E6, power limit 300 W (default and maximum),
  clocks not locked; a PCIe gen 5 card (`lspci`: 32 GT/s capable) in the board's gen 4 x16 slot (`nvidia-smi` gen 4
  x16 in every busy sample); the engine's probe 27.8 GB/s host to device.
- **CPU:** AMD Ryzen 7 5800X3D, 8 cores / 16 threads, AVX2, no AVX-512; 7 expert-pool workers on CPUs 1-7 and the host
  thread on CPU 0. Board MSI MAG B550M MORTAR WIFI.
- **RAM:** 80 GiB DDR4, 3,200 MT/s (2x 32 GiB + 2x 8 GiB), 76.5 GiB usable (`MemTotal`). 8 GiB swap (8,191 MiB),
  4.1 GiB of it in use before and after the runs; which processes' pages were in swap was not checked.
- **Storage:** SK hynix Gold P31 1 TB NVMe, ext4.
- **OS:** Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic; driver 595.99.02 (`nvidia-smi` reports CUDA 13.2, the
  driver's version); CUDA toolkit 13.1 (nvcc 13.1.115, the build's compiler).
- **Strata:** 0.1.44.1 at `296b59d2`, engine 0.1.44.1 compiled by setup on this PC (`engine/BUILD.json`: source
  `local`, archs `[120]`, vision `none`, src `7dfd34378bd2d32a`); no local source change. The checkout was on a
  branch that adds only another report folder on top of `296b59d2` (`results.json` names that branch's commit
  `b1394919`); engine, server and tools are those of `296b59d2`.
- **Background:** a GNOME desktop session and several mostly idle terminal programs; no other inference or GPU work.
  Another Strata server on this PC was stopped for the measurements.

## Model and configuration

- [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
  at setup's pinned revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, IQ2_XS:
  - `IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf`, 39,225,954,592 bytes, SHA-256
    `92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7`
  - `IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf`, 28,800,138,432 bytes, SHA-256
    `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`
  - The files were fetched from that revision with `curl` and handed to setup (it reported them as already downloaded
    and did not fetch them again); local `sha256sum` matches the repository's.
- Setup's pack `packs/iq2_xs` (`tools/iq_pack.py`: 1,079 tensors, 302 served natively), the shipped
  `data/expert-profile.bin` (no profile save), MTP draft layer `mtp/rt` (835 MiB of VRAM). Served model id
  `qwen3.8-flash-next-iq2_xs`. No vision encoder.
- Setup ran as `./setup.sh --setup --yes --family qwen --model IQ2_XS --no-start --data-dir <data> --vision none
  --low-ram off --no-browser`. Its choices: **context 65,536** tokens, **KV** int8 with 32,768 of 65,536 cells per QSA
  layer in VRAM (0.77 GiB of pinned RAM), **expert cache** auto: **6,417 slots, 8.60 GiB of VRAM**; prefill auto
  (8,192-token chunks, borrowing 3,180 slots); `--spec 4 --spec-min-p 0.5` with MTP; expert arena 33.02 GiB in RAM
  (loaded at 2.87 GiB/s). `STRATA_ROUTE_TAIL_SKIP=7` on (the default). The engine logged 353 MiB of VRAM free with
  everything loaded, and its warning for below 512 MiB (#2065); no slowdown of that kind showed in the runs.
- Greedy, thinking off, no calibration, no experimental speed projection.

The settings file as setup wrote it, paths replaced: [strata-iq2_xs.json](strata-iq2_xs.json) (the measured run used
it with port 8091 and the log in another folder).

```text
.venv/bin/python serve/server.py --engine strata --config strata-iq2_xs.json --port 8091
engine: --pack <data>/packs/iq2_xs --native <data>/models/IQ2_XS/...-00001-of-00002.gguf
        --ple-gguf <data>/models/IQ2_XS/...-00002-of-00002.gguf --expert-profile <strata>/data/expert-profile.bin
        --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp <data>/mtp/rt
        --max-context 65536 --kv int8 --kv-resident 32768
```

## Method

- One server started for this report; the measurements began once it said it was loaded (loading not timed).
  Requests were serial, in this order, with no restart: `python tools/strata_bench.py --url http://127.0.0.1:8091
  --long` (three short chats x 3 runs with up to 100 new tokens; fresh 4K / 16K / 32K word-list prompts with 24 new
  tokens, 3 each; three 6-turn agent sessions with up to 64 new tokens per turn), then
  [decode-sweep.py](decode-sweep.py) `--sizes 4096,32768,60000` (fresh word-list prompts that end with a request
  for a long Python module, 256 new tokens, 3 runs each, one warm-up discarded), then
  `python tools/needle_bench.py --lengths 32k,64k --depths 10,50,90`.
- **Speeds are the engine's own timings** (`prompt_per_second`, `predicted_per_second`, the same numbers as the
  engine log's lines); decode counts every generated token. First-token time and total latency are client-side over
  HTTP on the same PC.
- GPU telemetry: `nvidia-smi` every 500 ms ([gpu-telemetry.csv](gpu-telemetry.csv)). Memory: `free -m` with the
  model loaded and after the runs ([hardware-and-setup.txt](hardware-and-setup.txt), with setup's own lines and
  the hashes).

## Results

### Official suite

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat (3 questions x 3) | 46 (43-51) | 0 | 100 each | 9 | not reported (43-51-token prompts) | 116.0 (95.4-125.2) | 0.33 (0.28-0.36) |
| 4K prompt, fresh | 4,019 | 0 | 24 cap | 3 | 2,817.2 (2,697.0-2,819.6) | not reported (24-token cap) | 1.44 (1.44-1.50) |
| 16K prompt, fresh | 16,058 | 0 | 24 cap | 3 | 3,380.2 (3,347.2-3,400.0) | not reported (24-token cap) | 4.77 (4.74-4.82) |
| 32K prompt, fresh | 32,049 | 0 | 24 cap | 3 | 3,393.4 (3,390.0-3,396.3) | not reported (24-token cap) | 9.47 (9.46-9.48) |
| agent session, follow-up turns | 3,741-14,596 | see below | 18-64 | 15 | see below | 126.7 (106.6-145.8) | 1.46 (1.27-1.64) |

Agent sessions: 68.7% (68.6-68.8) of the follow-up turns' prompt tokens came from the conversation cache, 4 (2-73)
tokens of earlier turns were read again per session, and the first turn (nothing cached) took 0.88 s (0.87-0.88) to
the first token. No request failed. Per question, short-chat decode was 117.3 / 124.3 / 116.0, 95.4 / 125.2 / 107.6
and 104.7 / 110.7 / 118.8 tok/s, with 69-70 of 83-88, 42-53 of 88-100 and 54-55 of 91-94 drafts accepted. Data:
[results.json](results.json).

### Decode at three context lengths

256 new tokens on every run, fresh prompts, 3 runs each.

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| ~4K prompt, 256 out | 4,027 | 0 | 256 | 3 | 2,806.1 (2,793.0-2,809.2) | 167.2 (142.9-182.7) | 1.45 (1.45-1.46) |
| ~32K prompt, 256 out | 31,963 | 0 | 256 | 3 | 3,385.0 (3,382.4-3,400.9) | 120.1 (119.1-201.7) | 9.47 (9.43-9.48) |
| ~58K prompt, 256 out | 58,480 | 0 | 256 | 3 | 3,222.3 (3,221.3-3,228.5) | 109.3 (104.6-155.6) | 18.19 (18.16-18.20) |

Total latency (client): 2.97 s (2.85-3.23), 11.55 s (10.73-11.62) and 20.53 s (19.83-20.59). The wide decode ranges
coincide with different draft acceptance and expert-cache hits: the fastest run at 32K (201.7 tok/s) accepted 184 of
205 drafts with a 97.5% decode hit rate, the two others 147 of 202 and 127 of 217 with 87.3% and 92.2%; at 58K, 148
of 181 against 101-106 of 223-226. At 4K the order differs (161 of 227 accepted in the run at 167.2 tok/s, 177 of
214 at 142.9). The three prompts of a size differ only in their leading tag. Data: [decode-sweep.json](decode-sweep.json).

### Needle recall

| Length label | Prompt tokens | Depth | Found | Prompt tok/s |
| --- | ---: | --- | --- | ---: |
| 32k | 32,778-32,779 | 10 / 50 / 90% | yes / yes / yes | 3,139 / 3,290 / 3,288 |
| 64k | 64,921-64,923 | 10 / 50 / 90% | yes / yes / yes | 3,320 / 3,324 / 3,270 (16,384 reused) |

**6 of 6 found**, one request each; the last one started with the same 16K tokens as the one before it, so the
conversation cache served those. Data: [needles-32k-64k.json](needles-32k-64k.json).

### GPU and memory

564 of 567 samples busy (utilisation 50% or more): SM clock 2,872 MHz median (2,850-2,917), memory clock 13,801 MHz,
power 260 W median (59-280) of the 300 W limit, temperature 66 C median (42-71), VRAM used 15,511-15,573 MiB. The
software power-cap flag (`0x4`) was set in 192 of the 564 busy samples; why at these draws was not looked into. No
thermal or hardware slowdown flag. Decode expert-cache hit rate on the short chats: 73.2-90.6% (engine log).
RAM (`free -m`): 44,506 MiB (43.5 GiB) in use with the model loaded before the runs and 45,466 MiB (44.4 GiB) after,
of 78,301 MiB; 4.1 GiB of
swap in use both times; no out-of-memory event. Peak use during the runs was not recorded.
Engine start-up and every request line: [engine-log.txt](engine-log.txt).

## Correctness and limitations

- **Recall:** 6 of 6 needles at 32K and 64K; an outcome on these inputs, not a quality test. No coding, tool-call or
  answer-quality check.
- **One session, one size of context.** About five minutes of measurements in one engine session; no repeat on
  another day. 128K and longer contexts were not run: setup chose 65,536 tokens for this PC, and that is what was
  measured.
- **Other RTX 5070 Ti results with IQ2_XS, not like-for-like:**
  - [#2173](https://github.com/Niko1221/Strata/issues/2173) (Threadripper 7960X with AVX-512, Windows 11, 0.1.43.1,
    65,536 tokens with `--vram-reserve-mib 1125` added by hand, 6,477 experts in VRAM, the same suite): chat decode 123.6 tok/s, prompt 2,795 / 3,287 / 3,320 tok/s
    at 4K / 16K / 32K. Here: 116.0 (-6%) and 2,817 / 3,380 / 3,393 (+1% / +3% / +2%), with 6,417 experts in VRAM.
  - [2026-10-06-community-rtx5070ti-5900x](../2026-10-06-community-rtx5070ti-5900x/README.md) (Ryzen 9 5900X,
    Windows 10, 0.1.39, 256-token code-explanation prompts): 1,992 / 2,817 tok/s prompt and 102.0 / 101.3 tok/s decode
    at 4K / 32K. Different prompts and engine.
  - The README's RTX 5070 (12 GB) row for IQ2_XS (Ryzen 5 7600, Windows, engine 0.1.26, one run per cell): 79 tok/s
    short-chat decode, 63 at 128K, 2,090 tok/s prompt at 32K, from one code-agent prompt per length. Different card,
    engine, OS and prompts.
- **Not measured:** 128K context, other sizes on this run (Swift IQ3_S on the same PC: PR #2580), vision, concurrency
  and `--batch`, thinking on, peak RAM and VRAM per request, power-limited or locked clocks.
- **Units:** VRAM and RAM in MiB and GiB as `nvidia-smi`, `free` and the engine print them; installed RAM in GiB
  (DIMM sizes); file sizes in bytes.
