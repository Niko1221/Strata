# Community benchmark on RX 6700 XT

Measured on 2026-11-10 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU: one RX 6700 XT (gfx1031), nominal 12 GiB dedicated VRAM; selected device 0. The harness saw 12272 MiB usable VRAM.
- CPU: Ryzen 5 3600, 6 physical cores / 12 host logical threads; physical RAM 64 GiB DDR4-3200. Benchmark LXC: 10 visible CPU threads and 56 GiB RAM limit. Storage: Intel 660p 1 TB NVMe (INTEL SSDPEKNW010T8).
- OS: Debian 13 (trixie), kernel 7.0.14-26-pve; driver: kernel amdgpu, module source version B1D135B579C294611EB792D. Signed portable HIP runtime/driver API 7.13.26193; see parent runtime provenance. GPU cap 211 W; load-time PCIe 16 GT/s x16.
- Strata setup release v0.2.13; installed Terminal/Control 0.1.59.1, commit 628f74955a56fcba092495b800dd02af75373a0f; signed RX6000 engine 0.1.59. The unmodified quick harness leaves release source fields null because it probes Git; the pre-run manifest records the installed COMMIT and exact harness hash.
- Background: dedicated loopback benchmark guest, other GPU guests stopped; one sequential client and a lightweight host resource sampler.

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq2_xs
- Context: 131072 tokens; parallel slots: 1
- Settings: deliberate custom Flash profile with upstream-compatible tool template, DFlash2 Q4-selector, launch spec 6 / confidence 0.8 / five workers, int8 KV with 32768 resident tokens; normal conversation cache retained; thinking off and temperature 0 for this official quick suite. Vision off; no engine calibration or experimental projection stage requested.

```text
<engine-component>/strata --serve --pack iq2_xs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --max-context 131072 --kv int8 --kv-resident 32768 --draft dflash2 --draft-model qwen3.8-flash-next-dflash2.q4_0-selq4.pack --draft-fallback none --mtp-draft-vocab draft_vocab.bin --spec 6 --draft-conf 0.8 --pool-workers 5
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. Two warm-up requests (chat and 1K) and two word/token-ratio calibration probes are excluded from stored measurements. The 36 stored requests match 36 of 40 full native timing lines. Fresh-prompt requests have zero measured reused tokens; agent turns deliberately reuse prefixes. Expert placement evolves, so this is not a cold-expert-cache measurement.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 28.2 (23.6-30.5) | 1.43 (1.26-1.69) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 329.2 (324.8-330.7) | not measured | 12.25 (12.19-12.42) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 351.7 (351.2-352.1) | not measured | 45.70 (45.65-45.77) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 350.5 (350.3-350.7) | not measured | 91.50 (91.44-91.54) |
| agent session, follow-up turns | 9025 (9020-9076) at the last turn | see below | up to 64 | 3 | see below | 32.2 (21.2-40.0) | 5.93 (5.74-6.33) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 71.1 % (71.1-71.2); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 8207 (8207-8207) new tokens of the files are read once); first turn (nothing cached) first token: 3.76 s (3.65-3.78).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 645 | 0 | 645 | 178.6 | 3.655 | 26.8 |
| 1 | 2305 | 663 | 1642 | 287.9 | 5.739 | 28.6 |
| 2 | 4011 | 2368 | 1643 | 280.6 | 5.896 | 34.2 |
| 3 | 5717 | 4074 | 1643 | 277.4 | 5.964 | 28.1 |
| 4 | 7376 | 5734 | 1642 | 261.0 | 6.329 | 21.2 |
| 5 | 9020 | 7378 | 1642 | 275.4 | 5.999 | 28.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background isolation and resource-sampler scope are recorded above. Total time of the suite: 632 s.

Complete native timing evidence: [native-engine.log](../../native-engine.log). Exact replay argument array: [command.json](../../command.json); source serving config: [config.json](../../config.json); pre-run file/runner identities: [predeclared-provenance.json](../../predeclared-provenance.json). Replace the documented path placeholders for a local installation. The release server adds `--serve` to the resolved native invocation; the attached complete log records actual startup.
