# Community benchmark on NVIDIA GeForce RTX 4060 Ti + NVIDIA GeForce RTX 3060

Measured on 2026-10-15 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4060 Ti + NVIDIA GeForce RTX 3060, 28 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2696 v3 @ 2.30GHz (18 cores, 36 threads); installed RAM: 125.6 GiB; storage: not measured
- OS: Linux 5.15.0-190-generic; GPU driver: 595.91.07; CUDA: 13.2
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1
- Background workloads and power limits: no background GPU workload during the suite; both cards at their **default** power limits (RTX 3060 170 W, RTX 4060 Ti 165 W) — not power-capped

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 262144 tokens; parallel slots: 1
- Settings: **`"layer_split": "28"` set explicitly** in `strata-iq3_s.json`; thinking off for the suite; temperature 0
- **Note:** `layer_split` is a top-level config key, not an `args` entry, so it does **not** appear in the command line below. The split in force here is 3060 = layers 0-27, 4060 Ti = layers 28-47.
- **This is not the setup default.** `layer-split auto` chose **3 / 45** on this asymmetric pair and measured 452.6 tok/s prefill / 41.8 tok/s decode. See [Layer split](#layer-split-the-largest-tuning-effect-on-this-pair) below.

```text
strata --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto:32768 --spec 4 --spec-min-p 0.5 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 1024 --control-vector-scaled Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer --conversation-cache-mib 8192
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 50.4 (45.4-60.0) | 0.74 (0.66-0.81) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 488.2 (486.5-488.2) | not measured | 8.27 (8.27-8.30) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 835.0 (730.7-866.2) | not measured | 19.28 (18.59-22.03) |
| agent session, follow-up turns | 14554 (14506-14568) at the last turn | see below | up to 64 | 3 | see below | 36.2 (27.1-55.8) | 8.45 (7.64-9.18) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.4-68.7); earlier-turn tokens read again per session: 74 (3-144) (earlier turns that the cache did not hold; the 13729 (13728-13730) new tokens of the files are read once); first turn (nothing cached) first token: 6.00 s (5.82-6.03).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 105.6 | 5.817 | 19.0 |
| 1 | 3748 | 626 | 3122 | 354.4 | 8.863 | 27.1 |
| 2 | 5281 | 3741 | 1540 | 193.1 | 8.027 | 27.4 |
| 3 | 8725 | 5274 | 3451 | 378.1 | 9.179 | 38.1 |
| 4 | 11633 | 8771 | 2862 | 331.7 | 8.679 | 42.6 |
| 5 | 14554 | 11655 | 2899 | 344.3 | 8.474 | 27.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Layer split: the largest tuning effect on this pair

The same machine, same model, same benchmark, with only `layer_split` changed. Each point
below is a cold prefill of an identical 14,564-token prompt against a freshly started
server (`--layer-split` values; `auto` is the setup default):

| `layer_split` | RTX 3060 layers | RTX 4060 Ti layers | prefill tok/s |
| ---: | ---: | ---: | ---: |
| `auto` | 3 | 45 | 452.6 |
| 12 | 12 | 36 | 581.8 |
| 18 | 18 | 30 | 693.0 |
| 24 | 24 | 24 | 761.7 |
| **28** | **28** | **20** | **802.1** |
| 32 | 32 | 16 | 803.2 |

`auto` left the RTX 3060 holding **3 of 48 layers**, with 12 GB of its VRAM idle and the
card drawing 17 W through a whole prefill while the 4060 Ti sat at 100 % utilisation. The
engine chose that deliberately (its log cites the tiny-expert-cache clamping case #1616),
but on a pair this asymmetric the caution costs **44 % of prefill throughput**.

The intuitive rule — give the *faster* card proportionally more layers — predicts the wrong
direction here. Memory bandwidth tracked the result rather than SM count or FP32 TFLOPS: the
RTX 3060 has 360 GB/s against the 4060 Ti's 288 GB/s, making it the better host for
expert-heavy layers despite fewer and slower SMs.

Two further observations from the same sweep:

- **A bad split shows up as a flat prefill curve.** Under `auto`, prefill barely moved with
  prompt length (491.7 tok/s at 4K, 512.1 at 16K — no batching benefit). With `--layer-split 28`
  it becomes a slope (488.2 at 4K, 835.0 at 16K). The 4K figure is unchanged; only the long
  prompts benefit, so a 4K-only benchmark would not have detected this at all.
- **VRAM headroom depends on the split, and the tight card is not the smaller one.** The
  engine warned *"only 201-434 MiB of VRAM free at the start of this request (below 512 MiB)"*
  (#2065) at the **low** split values, where the 4060 Ti carried 30-36 layers, and the
  warning disappeared at 24 and above. `--vram-reserve-mib` was raised 700 → 1024 as well.

Decode also improved (41.8 → 50.4 tok/s chat median; 25.1 → 36.2 tok/s agent median) — with
fewer layers per card the expert cache covers each layer better.

The `auto` baseline and the intermediate sweep were measured with the project's own
`tools/strata_bench.py` on the same machine; the sweep points are single cold-prompt runs and
are labelled as such. Absolute figures drift roughly 10-15 % with server age (expert profiles
adapt after start), so the sweep is comparable within itself, not against unrelated runs.

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded (none was running) and the power limits are stated above at their defaults. Total time of the suite: 289 s.
