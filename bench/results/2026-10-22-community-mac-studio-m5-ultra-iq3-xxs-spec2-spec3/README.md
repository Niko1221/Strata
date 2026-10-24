# Community benchmark: Mac Studio M5 Ultra 96 GB, IQ3_XXS, spec 2 vs spec 3

Measured on 2026-10-22 (UTC). Two standard-benchmark runs on the same machine,
same model and pack, differing only in the engine's speculative-decoding depth
(`--spec 2` on port 8080, `--spec 3` on a separate port). Both runs are
`complete`, 21 of 21 required requests, 12 of 12 sanity checks. Main
limitation: single run per configuration (no repeat runs, so run-to-run
variance is not measured); prompt-prefix reuse is the shared template prefix
only; the expert cache stays warm and evolves (not cold-model measurements).

## Hardware and software

- GPU: Apple M5 Ultra (96.0 GiB shared); CPU: Apple M5 Ultra; RAM: 96.0 GiB
- Storage of the model files: not measured (external USB volume; exact link
  speed not measured)
- OS: macOS 27.2; runtime: metal 27.2; drivers: not measured
- Strata commit: `011c9e600a93bbc9fd451c563501b5831c32bbca`; engine
  0.1.47-metal; source build (built on this machine)
- Background work: not measured (other programs closed during the runs; no
  power-limit value available from local tools)

## Model and configuration

- Model: qwen3.8-flash-next-iq3_xxs; quantization IQ3_XXS; revision not
  measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf`
  `219ea929900dfa9e...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf`
  `316b46f3a2dbd68c...` (full hashes in each run's `results.json`)
- Custom pack `/Volumes`-independent path `packs/iq3_xxs` with
  `expert-profile.bin`, `conversions.json`, `dense.bin`, `index.txt`,
  `native_experts.txt`; MTP realtime pack; tokenizer from the same pack
- Context 262144 tokens; server capacity 1 slot; KV type / streaming window /
  expert-cache size not exposed by the server (engine log has
  `--expert-cache 64 --mmap-experts --prefill auto --max-context 262144`)
- MTP on; reasoning effort xhigh, thinking on and preserved; sampling:
  temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence 0.0,
  repetition 1.0; calibration / experimental speed projection: not enabled
- Launch (per run, ports differ; credentials removed — none configured):

```text
build-metal/strata --serve --expert-cache 64 \
  --expert-profile data/expert-profile.bin --max-context 262144 \
  --mmap-experts --mtp <mtp-rt-dir> --native <00001-of-00002.gguf> \
  --pack <packs/iq3_xxs> --ple-gguf <00002-of-00002.gguf> \
  --prefill auto --spec 2        # spec2 run (port 8080)
  --prefill auto --spec 3        # spec3 run (port 8082)
strata bench run --url http://127.0.0.1:<port> --track default \
  --track-notes 'release 0.2.2 engine 0.1.47 Mac Studio M5 Ultra 96GB IQ3_XXS <specN>'
```

Only `--spec` differs between the two runs. The two servers were never run at
the same time (sequential runs to avoid memory pressure).

## Method

Frozen standard-benchmark requests (`bench/prompts/standard-v1`, protocol
`strata-qwen38-thinking-v1-draft.1`, corpus `e75eff8b9feea04f...`), sent over
loopback HTTP, streaming, one at a time: 1 warm-up (excluded), 3 repetitions
each of short (~80-token), 4K and 16K prompts with a 512-token cap
(reasoning + answer counted together), then the twelve sanity checks.
Prompt and decode speeds are the engine's own `timings`; time to first token
is the client's monotonic clock from dispatch to the first nonempty delta.
No request was retried; no optional-concurrency stage. Time after the model
loaded: spec2 1 min 27 s, spec3 1 min 38 s.
Separately, `tools/needle_bench.py` recall checks at 32k/128k × depths
10/50/90 on each server (`needles-spec2.json`, `needles-spec3.json`).

Per-run evidence: `spec2/` and `spec3/` each hold the runner's `README.md`
(source), `results.json`, `outputs.jsonl`, `config.json`, and `provenance/`.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| spec2 short | ~77 | template only | 443–512 | 3 | n/a (fixed request cost dominates) | 133.8 (130.7–139.1) | 0.18 (0.15–0.19) |
| spec2 prompt-4k | 4093 | template only | 512 (cap) | 3 | 2614.3 (2600.7–2646.1) | 122.5 (121.6–127.0) | 1.58 (1.56–1.59) |
| spec2 prompt-16k | 16387 | template only | 512 (cap) | 3 | 2656.1 (2646.4–2658.3) | 121.8 (115.7–126.5) | 6.19 (6.18–6.21) |
| spec3 short | ~77 | template only | 443–512 | 3 | n/a (fixed request cost dominates) | 142.3 (140.1–145.2) | 0.19 (0.18–0.19) |
| spec3 prompt-4k | 4093 | template only | 512 (cap) | 3 | 2579.7 (2088.9–2606.0) | 120.0 (113.3–122.8) | 1.60 (1.59–1.98) |
| spec3 prompt-16k | 16387 | template only | 512 (cap) | 3 | 2571.7 (2556.7–2622.6) | 110.7 (109.1–112.9) | 6.39 (6.27–6.43) |

spec2 vs spec3 (same machine, same workload, medians): short decode
**133.8 → 142.3 tok/s (+6.4 %)**; prompt-4k **2614.3 → 2579.7 tok/s
(−1.3 %)**; prompt-16k **2656.1 → 2571.7 tok/s (−3.2 %)**; 16k decode
**121.8 → 110.7 tok/s (−9.1 %)**. On this box spec 3 gains a little short
decode but loses prompt throughput and long-prompt decode against spec 2;
single run each, so treat small deltas as indicative, not definitive.
Total latency and per-case detail (including the capped-request caveat and
the two missing first-answer times) are in each run's `README.md` /
`results.json`. Artifact hashes (sha256):

```text
9a2f05c893f04de0e60e907ec8b9cd2814d5403c0cb86ea312ccd247fb103177  spec2/results.json
4f7410bb116acff2d677cbbdbe2ac39d98c8bf1ffe4198f93e0d325e0487422c  spec2/outputs.jsonl
c9bb0b6a8fc01913bac5748c6f9a27c53934bd5ef73bebfa630ee0165f368c16  spec3/results.json
85c3f95d27ae91028620493727ef339e0546cc9d2d82d86090bcd8c848682c5f  spec3/outputs.jsonl
e1ff8b882ae4e396bbfb8a45b832eb7dfb6dd743c88bae27b2d6fae9973a7224  needles-spec2.json
2773c56cf5064453b74fe8d8942e0046023b80deb1fd71bbcf77da00190c1889  needles-spec3.json
```

Sanity checks: **12 of 12 pass on both runs** (instruction 4/4, retrieval
4/4, tool calls 4/4). Needle recall: **6 of 6 found on both** (32k and 128k
at depths 10/50/90; actual prompt tokens 31,643 and ~125,349). No request
failed or was skipped on either run. The 4K/16K performance requests reach
the 512-token cap while still reasoning (7 of 9 truncated per run plus the
warm-up), so their total time is the time to the cap and first-answer time
is partly missing — a property of the frozen protocol cap, not a failure.

## Correctness and limitations

- Correctness checks: the protocol's 12 sanity checks (exact text / JSON /
  tool-call matching) all pass on both runs; needle recall 6/6 on both. No
  broader quality evaluation (coding tasks, HumanEval, long-horizon agents).
- Not tested: optional-concurrency stage; 32k/128k prompt sweeps; TTFT at
  larger contexts; power draw; cold-model timings; repeat runs of the same
  configuration (run-to-run variance unknown); any other quantization or GPU.
- The spec3 server's warm-up prompt rate (7.4 tok/s on the first request vs
  34.3 on spec2) reflects a cold speculative path on the first request only;
  measured repetitions agree within their ranges except the noted medians.
- Redaction: home folders, user/host names and local addresses replaced or
  absent before publishing; inspected with `grep` over the folder (clean).
  The `run.command` paths were rewritten to the in-repo result-folder form.
