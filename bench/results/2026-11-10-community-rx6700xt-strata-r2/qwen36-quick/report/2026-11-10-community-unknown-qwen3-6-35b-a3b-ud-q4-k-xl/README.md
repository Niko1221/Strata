# Community benchmark on RX 6700 XT

Measured on 2026-11-10 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU: one RX 6700 XT (gfx1031), nominal 12 GiB dedicated VRAM; selected device 0. The harness saw 12272 MiB usable VRAM.
- CPU: Ryzen 5 3600, 6 physical cores / 12 host logical threads; physical RAM 64 GiB DDR4-3200. Benchmark LXC: 10 visible CPU threads and 56 GiB RAM limit. Storage: Intel 660p 1 TB NVMe (INTEL SSDPEKNW010T8).
- OS: Debian 13 (trixie), kernel 7.0.14-26-pve; driver: kernel amdgpu, module source version B1D135B579C294611EB792D. Signed portable HIP runtime/driver API 7.13.26193; see parent runtime provenance. GPU cap 211 W; load-time PCIe 16 GT/s x16.
- Strata setup release v0.2.13; installed Terminal/Control 0.1.59.1, commit 628f74955a56fcba092495b800dd02af75373a0f; signed RX6000 engine 0.1.59. The unmodified quick harness leaves release source fields null because it probes Git; the pre-run manifest records the installed COMMIT and exact harness hash.
- Background: dedicated loopback benchmark guest, other GPU guests stopped; one sequential client and a lightweight host resource sampler.

## Model and configuration

- Model and quantization: qwen3.6-35b-a3b-ud-q4_k_xl
- Context: 65536 tokens; parallel slots: 1
- Settings: setup-generated Qwen3.6 profile copied to a dedicated log path, 65536 context, automatic prefill and launch --mtp 2; actual hybrid/expert/KV settings remain in the engine info and complete startup log; thinking off and temperature 0 for this official quick suite. Vision off; no engine calibration or experimental projection stage requested.

```text
<engine-component>/strata --serve --native Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --max-context 65536 --prefill auto --mtp 2
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 32.6 (32.1-33.0) | 0.52 (0.51-0.59) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 820.2 (817.1-820.4) | not measured | 4.91 (4.91-4.92) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 683.2 (683.1-683.8) | not measured | 23.52 (23.50-23.53) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 548.5 (548.5-548.5) | not measured | 58.46 (58.46-58.46) |
| agent session, follow-up turns | 9021 (9004-9036) at the last turn | see below | up to 64 | 3 | see below | 28.7 (26.1-31.9) | 7.24 (3.20-12.25) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 0.0 % (0.0-0.0); earlier-turn tokens read again per session: 20061 (20018-20148) (earlier turns that the cache did not hold; the 8185 (8185-8186) new tokens of the files are read once); first turn (nothing cached) first token: 0.85 s (0.85-0.85).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 716.0 | 0.855 | 32.5 |
| 1 | 2282 | 0 | 2282 | 714.3 | 3.199 | 31.7 |
| 2 | 3978 | 0 | 3978 | 803.4 | 4.957 | 30.3 |
| 3 | 5680 | 0 | 5680 | 781.7 | 7.272 | 28.7 |
| 4 | 7358 | 0 | 7358 | 759.8 | 9.69 | 27.2 |
| 5 | 9036 | 0 | 9036 | 738.2 | 12.247 | 26.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background isolation and resource-sampler scope are recorded above. Total time of the suite: 449 s.

Complete native timing evidence: [native-engine.log](../../native-engine.log). Exact replay argument array: [command.json](../../command.json); source serving config: [config.json](../../config.json); pre-run file/runner identities: [predeclared-provenance.json](../../predeclared-provenance.json). Replace the documented path placeholders for a local installation. The release server adds `--serve` to the resolved native invocation; the attached complete log records actual startup.
