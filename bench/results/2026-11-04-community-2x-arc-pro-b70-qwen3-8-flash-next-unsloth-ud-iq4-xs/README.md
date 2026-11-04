# Community benchmark on Intel Arc Pro B70 + Intel Arc Pro B70

Measured on 2026-11-04 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Intel Arc Pro B70 + Intel Arc Pro B70, 32 GB per card (submitter note; the tool's `nvidia-smi`/`rocm-smi` probe reads neither on Intel cards; cards arrive through vfio, physical link invisible to the guest)
- CPU: AMD EPYC-Milan Processor (QEMU CPU model, real host CPU: AMD Ryzen 7 5700G) (5 cores, 10 threads); installed RAM: 94.3 GiB; storage: sda QEMU HARDDISK sata
- OS: Linux 7.0.0-42-generic; GPU driver: Intel `xe` (Battlemage) kernel driver; CUDA (driver supports up to): not applicable (SYCL/AOT engine build)
- Strata: project 0.1.56, commit ffa80d73; engine 0.1.56
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq4_xs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 57.8 (46.6-66.8) | 0.36 (0.34-0.45) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 995.5 (976.3-1000.7) | not measured | 4.07 (4.05-4.15) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1697.5 (1692.4-1700.4) | not measured | 9.50 (9.48-9.53) |
| agent session, follow-up turns | 13156 (13148-13159) at the last turn | see below | up to 64 | 3 | see below | 61.1 (54.8-71.6) | 3.67 (2.68-4.17) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.3-69.3); earlier-turn tokens read again per session: 5 (3-5) (earlier turns that the cache did not hold; the 12279 (12278-12279) new tokens of the files are read once); first turn (nothing cached) first token: 2.24 s (2.24-2.28).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 271.0 | 2.279 | 61.6 |
| 1 | 3394 | 636 | 2758 | 733.8 | 3.792 | 56.9 |
| 2 | 4927 | 3457 | 1470 | 526.5 | 2.825 | 62.7 |
| 3 | 8171 | 4990 | 3181 | 770.5 | 4.165 | 60.1 |
| 4 | 10399 | 8234 | 2165 | 666.7 | 3.282 | 54.8 |
| 5 | 13148 | 10438 | 2710 | 729.0 | 3.753 | 70.6 |

Per-request data: `results.json`. This suite did not start the server, so no engine-log excerpt was captured by the tool; a sanitized journal slice of the service for this suite's window is attached as `engine.log`.

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 142 s.

## Submitter addendum

- Background load: the suite ran directly after the standard-protocol run on the same freshly reloaded engine, with the resident Hermes agent idle during the whole window; the engine's own prompt/decode timings are robust against whatever little traffic remained, client TTFT is not if it had not been quiet.
- Tool gaps: VRAM 32 GB per card; driver is Intel `xe` (Battlemage kernel driver); CPU line is the QEMU model (real host CPU: Ryzen 7 5700G); the vfio-passed cards show no physical PCIe link to the guest, host links run at Gen3 rate (CPU ceiling), engine PcieAuto probe measured `pcie_frac 0.20` with this two-card `--layer-split auto` split.
- Drafting was active: request-level counters show 383 of 533 draft tokens accepted (71.9 %) across the nine chat requests (`--mtp` draft layers; the separately downloadable DFlash2 drafter is not installed, which is what the engine's startup line `Drafting: off` refers to).
- The companion standard-protocol run (same engine, same reload, thinking on) is submitted in the same pull request at `bench/results/2026-11-04-standard-arc-pro-b70-plus-arc-pro-b70-b80ac808/`.
