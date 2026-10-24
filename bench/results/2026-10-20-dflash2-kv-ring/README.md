# DFlash2 local K/V ring: long-context regression check

Run on 2026-10-20 with the current PR candidate on an RTX 4070 Ti SUPER 16 GiB,
Ryzen 9 5900X, Linux, CUDA toolkit 13.4.92 / driver 610.57.04 / SM89. The
standalone exact-prompt request used binary
`24142cb8fc6051396cfa2a4d0bd3eb758ae958341e98b196384f52d144033ac1`. After
enabling the CUDA snapshot test target, the final context-stress request and
current build hash are
`cd45d9b4456113eea46ea622c114b308c3ca3f1bd8f6f53fa7d94ab28829c35b`; the final
CUDA test group and final PR-style benchmark used this build. The production
sources were unchanged between those builds. Target: IQ3_S Qwen3.8-Flash-Next
with the official Q8_0 DFlash2 artifact.

## The reported failure

The attached engine log reached 135,296 prompt tokens and failed after 12,097
generated tokens with `the K/V cannot grow: no VRAM is left`. The attached prompt
file itself is 25,907 UTF-8 bytes (25,903 characters) and renders to 5,238 tokens
with this model's chat template. Its SHA-256 is
`b65afcc369dab659c2660a394a8848816bbd517d7bd5f0fbf28e6c0e1f77f5bd`.
The remaining 130,058 tokens from the logged context are not present in the
provided prompt file, so the exact original 135,296-token input cannot be rebuilt
from the attachments alone.

The exact prompt was run with the settings from the log and a 12,300-token output
limit. It returned all 12,300 tokens with no K/V error: 70.96 decode tok/s, 4,500
accepted proposals from 5,464 offered. This exact-prompt run has a 5,238-token
input and does not by itself reach the logged context depth.

A final-binary memory-pressure run preserved that exact prompt as the final user
turn and prepended 130,058 synthetic prior-context tokens to reach 135,296 input
tokens. It returned all 12,300 requested output tokens without the K/V error:
67.11 decode tok/s, 4,412 accepted proposals from 5,247 offered, and 239.47 seconds
for the request. It ran past the failure point in the attached log (12,097 generated
tokens). The test used `--max-context 524288`, YaRN scale 2, INT8 target KV,
`--kv-resident 32768`, auto expert cache, a 1486 MiB VRAM reserve, spec 8,
DFlash2 block 7 / floor 0.85, and `--dflash-window 0`. The engine logged a 2060-cell
DFlash2 device ring per draft layer, held 7422 MiB (3793 slots) for experts, and
left 1181 MiB free. The synthetic prefix matches the logged prompt length, not its
unavailable text. The initial-build long-context run remains in
`context-stress-initial.json`; `context-stress.json` and
`context-stress-final.log` contain the final-binary run. The reports retain the
prompt hash, token counts, resolved engine settings, acceptance, timing and error
state; they do not contain the prompt text or its generated long-form answer. The
PR-style perf JSONs contain only the 96-token benchmark outputs.

Reproduction command (use local paths for the model files and expert profile):

```sh
.venv/bin/python bench/results/2026-10-20-dflash2-kv-ring/stress.py \
  --prompt <attached-prompt.txt> --pack <target-pack> --native <target.gguf> \
  --ple-gguf <ple.gguf> --dflash <dflash2-Q8_0.gguf> \
  --expert-profile <expert-profile.bin> --engine <build-dir>/strata \
  --output <results-dir>/exact-prompt.json --max-new 12300
```

For the long-context run, repeat the command with `--pad-to 135296` and replace
the output path with `<results-dir>/context-stress.json`. The script fixes the
settings to the values listed above and uses temperature 0, seed 1234. The prompt
file is intentionally supplied by the caller and is not checked into this repository.

## Build and tests

The current CUDA build command was `cmake --build build-dflash2 --target strata
conversation_snapshot_test -j2` (Ninja reported no work on the final check). The
isolated snapshot test passed. The related CTest command was:

```sh
ctest --test-dir build-dflash2 --output-on-failure -j1 -R '^(conversation_snapshot_test|dflash2_parity|dflash_head_bf16_test|dflash_rope_parity|dflash_owned_kv_test|kv_stream_parity|kv_window_parity|dflash_policy_test|dflash_load_test|draft_kv_plan_test)$'
```

It passed sequentially: `conversation_snapshot_test`, `dflash2_parity`,
`dflash_head_bf16_test`, `dflash_rope_parity`, `dflash_owned_kv_test`,
`kv_stream_parity`, `kv_window_parity`, `dflash_policy_test`, `dflash_load_test`,
and `draft_kv_plan_test` (10/10). The snapshot test covers a device-only ring with
no full host K/V duplicate, wraparound capture/restore, direct ring copy, prefix
restore boundaries and rejection of full-context disk sessions. The detailed
sanitized output is in `cuda-ctest-final.log`.

One parallel run of this same group failed `conversation_snapshot_test` at the
`A/B/A byte-exact K/V and indexer state` assertion. The isolated snapshot run and
the sequential 10-test run passed; the cause of the parallel failure is unknown.

HIP compilation was not run because `hipcc` is not installed in this environment.
SYCL compilation was not run because `icpx`/`dpcpp` are not installed; DFlash2 is
currently unavailable on the SYCL runtime. The PR remains a draft pending the
backend validation already requested in its description.

## PR-style long-prompt performance

The controlled comparison follows the PR's 2425-token long fixture, one warmup,
three measured 96-token requests, same executable, 2900 actual expert slots,
context/resident 8192, prefill 512, `--short-read 64`, reserve 480 MiB, and greedy
seed 1234. DFlash2 used Q8_0, block 6 and probability floor 0.85; MTP used floor
0.5. All DFlash2 measured outputs matched its target-only control exactly (96/96
IDs each).

| Drafter | Decode median [range], tok/s | Request median [range], s |
| --- | ---: | ---: |
| DFlash2 Q8_0 | 52.40 [52.24–52.40] | 7.732 [7.731–7.738] |
| MTP | 50.64 [50.62–50.69] | 7.740 [7.739–7.740] |

DFlash2 decoded 3.49% faster than same-build MTP on this fixture. Client request
time was 0.09% lower (7.732 vs 7.740 s); the difference is under 8 ms and does not
establish a request-latency improvement. Compared with the selected pre-change
DFlash2 profile in the PR (51.95 tok/s median, 51.82–51.98 range), the candidate
median had 0.87% higher decode throughput. The fix did not show a material
performance regression. This result covers the long fixture only, not the PR's
other short prompts or other hardware.

Exact commands used for the performance arms:

```sh
.venv/bin/python bench/results/2026-10-20-dflash2-fast-prompt/run.py dflash2-il \
  --draft-quant Q8_0 --block 6 --floor .85 --expert-cache 2900 --reserve 480 \
  --prefill 512 --short-read 64 --tasks long --only-selected --repeats 3 \
  --output /tmp/dflash2-kv-ring-perf/dflash

.venv/bin/python bench/results/2026-10-20-dflash2-fast-prompt/run.py mtp \
  --floor .5 --expert-cache 2900 --reserve 480 --prefill 512 --short-read 64 \
  --tasks long --only-selected --repeats 3 \
  --output /tmp/dflash2-kv-ring-perf/mtp
```

The final build's raw measured rows, resolved commands and sanitized engine logs
are in `dflash2-perf-final.json`, `dflash2-command-final.json`,
`dflash2-perf-final.log`, and the matching MTP files. The first same-source run,
before the test-enabled rebuild, is retained in `dflash2-perf.json`,
`dflash2-command.json`, and the matching MTP files. Functional test summaries are
in `exact-prompt.json` and `context-stress.json`.
