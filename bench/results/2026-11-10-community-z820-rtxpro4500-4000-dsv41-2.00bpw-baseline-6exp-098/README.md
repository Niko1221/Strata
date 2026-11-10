# Community benchmark on RTX PRO 4500 32 GB + RTX PRO 4000 24 GB in a dual Xeon E5-2697 v2 (HP Z820)

Measured on 2026-11-10 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA RTX PRO 4500 Blackwell 32 GB + NVIDIA RTX PRO 4000 Blackwell 24 GB; both used (the tool names only the primary card)
- CPU: Intel(R) Xeon(R) CPU E5-2697 v2 @ 2.70GHz (24 cores, 48 threads); installed RAM: 251.8 GiB; storage: sda     SanDisk 3.2Gen1                 usb; sdb     CT1000BX500SSD1                 sas; sdc     CT2000BX500SSD1                 sas; sdd     Compact Flash                   usb; sde     SM/xD-Picture                   usb; sdf     SD/MMC                          usb; sdg     M.S./M.S.Pro/HG                 usb; sdh     SD/MMC/M.S.PRO                  usb; sr0     DVD A DH16ACSHR                 sas; zram0; nvme0n1 WD PC SN5000S SDEPNSJ-1T00-1006 nvme
- OS: Linux 7.0.14-101.fc43.x86_64; GPU driver: 580.178.04; CUDA (driver supports up to): 13.0
- Strata: app 0.2.13.1 (tag v0.2.13.1), commit 628f7495; engine strata-dsv41 0.1.59.1 (source build, AVX floor)
- GPU detail (after the run, near idle): power_limit_w=200.00, power_draw_w=82.73, pcie=gen3 x16, pstate=P1, temp_c=73
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: DeepSeek-V4.1-Flash ISTA-DAS 2.00 bpw (IQ2_XXS set; mixed IQ1_M/IQ2_XS/IQ2_XXS experts, 1.94 bpw effective), revision cb02422c
- Context: 65536 tokens; parallel slots: 1
- Settings: custom (see "Configuration and system-specific notes" below); thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 32 (29-38) | 0 | 100 (100-100) | 9 | not measured | 19.9 (15.7-22.0) | 1.01 (0.87-1.31) |
| 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 3 | 499.6 (482.8-522.4) | not measured | 8.28 (7.92-8.56) |
| 16K prompt, fresh | 16562 (16562-16562) | 0 (0-0) | 24 cap | 3 | 787.6 (782.5-809.5) | not measured | 21.08 (20.51-21.21) |
| 32K prompt, fresh | 33126 (33126-33126) | 0 (0-0) | 24 cap | 3 | 844.0 (839.2-844.5) | not measured | 39.33 (39.32-39.57) |
| agent session, follow-up turns | 13514 (13513-13520) at the last turn | see below | up to 64 | 3 | see below | 21.7 (18.7-24.9) | 9.36 (7.82-10.86) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.3 % (70.3-70.3); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 12631 (12631-12632) new tokens of the files are read once); first turn (nothing cached) first token: 6.05 s (5.89-6.75).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 602 | 0 | 602 | 89.5 | 6.748 | 15.3 |
| 1 | 3996 | 634 | 3362 | 310.9 | 10.858 | 18.7 |
| 2 | 5514 | 4059 | 1455 | 160.7 | 9.102 | 18.8 |
| 3 | 8626 | 5576 | 3050 | 289.7 | 10.574 | 19.5 |
| 4 | 10871 | 8689 | 2182 | 226.0 | 9.708 | 21.4 |
| 5 | 13513 | 10926 | 2587 | 264.9 | 9.821 | 21.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 498 s.

## Configuration and system-specific notes

**Hardware.** HP Z820, 2x Intel Xeon E5-2697 v2 @ 2.70 GHz (12 cores / 24 threads
each, 24C/48T; **AVX only, no AVX2/FMA/AVX-512**), 16 x 16 GB DDR3-1866 ECC,
251.8 GiB usable, two NUMA nodes. GPUs: **NVIDIA RTX PRO 4500 Blackwell 32 GB +
NVIDIA RTX PRO 4000 Blackwell 24 GB**, both PCIe 3.0 x16, no NVLink. The tool's
hardware line names only the primary card; **both cards were used** in every
benchmark arm.

**Storage and hot path.** The model files are on `/mnt/nvme` (WD PC SN5000S,
Gen2-crippled, about 2 GB/s). The SATA/USB devices in the tool's storage line are
not model storage. SGLang (:8082) and Apex (:8090) were stopped for the window;
no other GPU work ran. After load there is **no NVMe or SATA I/O in the hot path**:
the engine auto-copied all 10,519 CPU-lane experts (90.2 GB in the baseline, 88.7 GB
in the fill arms) into process RAM and pinned them. The 110.8 GB Engram was
page-cached in RAM at **102.7 of 103 GiB** after evicting the shard pages and
repeated passes (`fincore`). A 4,220-token request read 2.4 MB from NVMe; the
benchmark read only a few MB, not per-token streaming. The fit text nominally
describes the n-gram table as SSD-resident; **the measured run is RAM-resident for
all practical purposes**.

**Software.** Fedora 43, kernel 7.0.14; NVIDIA driver 580.178.04; CUDA 13.1
(the tool reports the driver's 13.0 support). Strata app **0.2.13.1** (tag
v0.2.13.1, commit `628f7495`); DeepSeek engine **`strata-dsv41 0.1.59.1`** build
`628f74955`, source-built with `STRATA_ISA_FLOOR=avx`, `STRATA_NATIVE_EXPERTS=ON`,
`STRATA_PORTABLE=OFF`, `CMAKE_CUDA_ARCHITECTURES=120`, against the vendored
`third_party/strata-ggml`.

**Model.** `ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF` @ `cb02422c`, the
**4-shard IQ2_XXS 2.00 bpw** set. The set is mixed: IQ1_M 60 / IQ2_XS 18 /
IQ2_XXS 42 expert tensors, effective **1.94 bpw**, 132.1 GB. Engram IQ4_NL
110.6 GB, DSpark Q8_0 8.0 GB (auto chose **off** and did not load it; the experts
are far larger than VRAM), shipped `profile-2.00.bin` (SHA-256
`8db93e483b34b15dbf63a0826601f1db2107a8cfea73774e7d21d8559e0a9568`), vision off
(mmproj packed but unused).

**Settings.** `--max-context 65536` (chosen so the 32K prompt plus the 24-token
output fits; 32K would have skipped the 32K case), `--experts 6` for the baseline
and fill-6 arms, `--experts 5` (`--speed faster`) for the fill-5 speed arm,
`--dspark auto` (off), `--vram-experts 0.98 --reserve-mib 900` for the baseline
and `--vram-experts 0.995 --reserve-mib 512` for the fill arms, `--cpu-threads 22`,
`--prefill-chunk 8192`, `--mmap-experts auto`; `--ssd-cache`, `DS4_STORE_GB`,
`DS4_COPY_GB` and `DS4_COPY_PIN` were **unset**; env `DS4_ASYNC_T=8
DS4_ATTN_SPLIT=0 DS4_LANE_FUSE=1 DS4_GG_IQ1M=auto DS4_ENGRAM_THREADS=16`. The
auto-copy path copied every CPU-lane expert into RAM, with no demand store.
`--tp 2` was requested; the expert blocks cannot split `n_ff 2304` in two, so the
engine fell back to TP 1 and a **layer split [0,19] / [20,39]+head**. The baseline
held 2,042 / 2,799 experts in VRAM (free 1,200 / 1,369 MiB by the engine's line;
1,792 / 1,606 MiB by `nvidia-smi`); the fill arms held 2,122 / 2,885 (engine free
584 / 627 MiB; `nvidia-smi` free 1,050 / 990 MiB).

**Configuration tracks.** These are **custom** configurations, not the standard
protocol. The standard `strata bench run` protocol is defined for
Qwen3.8-Flash-Next only; this report uses the community-format
`tools/strata_bench.py --long` (3 measured runs per case, medians and ranges,
thinking off, temperature 0, one request at a time, no concurrency). Fresh
prompts carry unique tags; the agent sessions are new; the expert and conversation
caches were warm and evolved. Conversation slots from earlier probes (33, 0.91 GB at the last arm)
were kept but were not reused by the tagged fresh prompts.

**Tuning result.** Baseline 6 experts (0.98/900): chat 19.9 tok/s, 4K 499.6,
16K 787.6, 32K 844.0, agent 21.7. Fill 6 experts (0.995/512): chat 19.9, 4K
561.5 (+12.4%), 16K 806.9 (+2.4%), 32K 856.9 (+1.5%), agent 22.1 (+1.8%). Fill 5
experts (0.995/512): chat **31.3** (+57% vs fill-6), 4K 546.8 (-2.6%), 16K
**863.3** (+7.0%), 32K **926.2** (+8.1%), agent **26.5** (+19.9%). The fill arms
used about 0.7-0.8 GiB more per card and kept the Engram fully resident. The
5-expert arm is the speed configuration; the 6-expert fill is the full-quality
configuration.

**Quality check (fill-5 only).** Tool-call probe **8/8 correct** with the scorer
self-test 15/15; exact-arithmetic chain **1/3 at 5, 10 and 20 steps** with thinking
off (weak exact arithmetic; not a tool-call failure). The 6-expert arms were not
quality-probed in this window.

**Failure observed.** The first arm set `DS4_COPY_PIN=1`; chat worked (28.7 tok/s)
but every 4K / 16K / agent / 32K prompt failed with
`cudaMemcpyAsync(dst + lay.trellis[2], it.src[2], mx_part_bytes(l, 2),
cudaMemcpyHostToDevice, c.cs): invalid argument`. Removing `DS4_COPY_PIN` completed
every stage. The failed folder is kept as evidence and is **not** a performance
result; it is a useful bug report.

**Power/background.** GPU power limit 200 W, idle about 83 W; no other GPU work;
SGLang and Apex were stopped. The tool's storage line lists all block devices;
only `/mnt/nvme` held the model.

