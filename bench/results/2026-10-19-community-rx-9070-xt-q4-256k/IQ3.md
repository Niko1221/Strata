# IQ3_XXS: speed, context and coding results

Same RX 9070 XT / Ryzen 7 9800X3D / 64 GB workstation and Windows HIP engine package 0.1.45.3 as the [main report](README.md), measured October 19 local time. IQ3 generated substantially faster than Q4. Independent tests found incomplete Express and WordPress repairs despite passing visible tests; the full-stack repair passed.

## Configuration and method

Full Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, using the two existing GGUF shards named in [iq3_xxs-baseline.example.json](iq3_xxs-baseline.example.json). Existing pack, expert profile and MTP draft reused. Artifact hashes/revision were not captured for this reused installation.

Baseline: 32768 context, int8 KV, mmap experts, requested MTP depth 4, auto prefill/expert cache, 2048 MiB VRAM reserve, vision off. [iq3xxs-128k.example.json](iq3xxs-128k.example.json) raises context to 131072; [iq3xxs-tail0.example.json](iq3xxs-tail0.example.json) disables expert skipping with `STRATA_ROUTE_TAIL_SKIP=0` at 32768 context.

The same [benchmark script](scripts/benchmark.py) used for Q4 produced these requests: temperature 0, seed 42, reasoning off, 192-token output cap. Twenty-four individual records are in [iq3-runs.json](iq3-runs.json), with server timings, actual token counts, client latency, UTC timestamps and system CPU/RAM samples. No prompt KV reuse. Expert cache state persists within each group; run 0 can include warming. Physical disk counters were not captured during this phase.

To reproduce a group, load the appropriate configuration and run the benchmark script with its original label. For example:

```powershell
python scripts/benchmark.py --label iq3xxs-current-short --runs 3
python scripts/benchmark.py --label iq3xxs-current-32k --context 32768 --runs 3
# Load the 131072-context profile before these two groups.
python scripts/benchmark.py --label iq3xxs-varied-64k --context 65536 --probe varied --runs 3
python scripts/benchmark.py --label iq3xxs-varied-128k --context 131072 --probe varied --runs 3
# Load the 32768-context, expert-skipping-disabled profile.
python scripts/benchmark.py --label iq3xxs-tail0-short --runs 3
```

## Short-prompt generation

Every request below had 42 input tokens and 192 output tokens. Means include run 0; the warm-run range is identified separately rather than silently discarding slower runs.

| Group / label suffix | Individual generation tokens/s | Mean | Desktop/cache conditions |
| --- | --- | --- | --- |
| loaded-desktop-short | 17.3, 36.4, 38.3 | 30.67 | Ollama also active; engine reported no free VRAM |
| current-short | 57.3, 83.5, 88.2 | 76.33 | Ollama worker no longer present |
| tail0-short | 25.9, 59.0, 65.9 | 50.27 | Expert skipping disabled |
| tail0-postagent | 46.4, 70.7, 69.5 | 62.20 | Same profile after agent task |

The often-quoted 59–71 tokens/s refers to warm repeats with expert skipping disabled. It is not the mean across every request. The background process exited independently; the evaluator did not stop it. These groups do not isolate the effect of a single tuning variable. GPU expert slots were 2681 at the later baseline restart, 2788 with skipping disabled and 1839 at 128K context.

## Prompt processing and recall

Three runs per row; all individual recall checks passed. Short outputs (six tokens for repeated-code probes, 27 for varied-source probes) are not sustained generation benchmarks.

| Probe / actual input tokens | Individual prompt tokens/s | Individual TTFT seconds | Probe type |
| --- | --- | --- | --- |
| 8K / 7634 | 192.1, 608.9, 649.1 | 39.831, 12.603, 11.819 | Repeated code, one marker; loaded desktop |
| 32K / 32204 | 852.1, 857.5, 860.3 | 37.839, 37.596, 37.474 | Repeated code, one marker |
| 64K / 64982–65006 | 702.9, 938.1, 908.9 | 92.524, 69.356, 71.577 | Varied code, three markers |
| 128K / 130508–130536 | 959.4, 953.8, 967.1 | 136.120, 136.941, 135.080 | Varied code, three markers |

At 128K, mean prompt processing was 960.1 tokens/s and median TTFT 136.120 seconds. Initial loaded-desktop short requests used 58.76–59.62 GiB system RAM; loaded near-8K used 59.85–61.63 GiB. Other applications contributed to these totals. Start-to-generation-ready was 15.81 seconds initially and 23.47 seconds on a later restart, including readiness generation.

## Native Claude Code coding tasks

Claude Code 2.1.233 used the local Anthropic API, safe mode, medium effort and a requested 2048-token thinking budget. Tasks used fresh synthetic fixtures with failing baseline tests. The 32K server window is distinct from the cumulative input counters across turns. No evaluator implementation fixes were made.

| Task | Wall seconds | Visible tests | Independent outcome |
| --- | ---: | --- | --- |
| Express | 234.36 | 9/9 pass | FAIL: concurrent conflicting idempotency payload accepted |
| Express, expert skipping off | 255.09 | 13/13 pass | FAIL: same requirement still missed |
| WordPress | 201.17 | 14/14 pass | FAIL: database read errors mishandled |
| Full-stack async debugging | 190.63 | 16/16 pass | PASS, including five extra edge checks |
| Q4 Express, for comparison | 1501.52 | 11/11 pass | PASS: conflicting payload rejected |

IQ3 produced one complete fixture pass and two partial task categories. Across the four IQ3 attempts, all 69 tool calls received results; five tool-error responses were handled. Original tests were preserved. The first Express repair needed a correction after visible HTTP failures; the full-stack attempt corrected a newly added test expectation. A separate interrupted pilot is excluded from these timings.

The Express service accepted two concurrent same-key/different-payload requests instead of rejecting the conflict. With skipping disabled, the added test named for conflicting payloads actually sent identical payloads. The independent HTTP checker reported statuses [200,201] for the first IQ3 attempt and [201,201] for the second, versus [201,409] for Q4; each created one order.

The WordPress fixture uses deterministic API/database doubles. On an empty result with a database error it returned HTTP 200; on a null result it raised TypeError instead of the required WP_Error 500. Insert failures were handled. This was not a live WordPress/MySQL installation.

The full-stack repair handled tenant isolation, stock reservation/rollback and stale asynchronous responses. The five extra checks covered shared-database concurrency, invalid quantities and three stale-completion cases. Its database is in-memory and single-process.

[coding-evidence](coding-evidence) contains audit counters, model-produced patches and independent-check results. Baselines are in [fixtures](fixtures); apply a task's patch with `git apply -p1` from a disposable copy of its corresponding fixture directory. The added scripts check Express concurrency and WordPress database failures; their path handling now accepts a fixture directory. Run the Express checker with `node --import tsx scripts/check_express_http_race.mjs <fixture-directory>` after installing the fixture dependencies; run the PHP checker with `php scripts/check_wordpress_db.php <fixture-directory>`. The WordPress `test.mjs` wrapper reflects the original Windows/Ubuntu-24.04 setup; direct `php tests.php` works where PHP is installed. Transcript cost estimates/session IDs were omitted; these were local model calls, not cloud charges.

The Q4 Express pass is a useful observed difference, but one attempt cannot establish a general quantization-quality ranking. Background memory use, model artifacts, cache allocation and generation variability also differed. The full-stack Q4 run used a much larger context and a different agent loop, so its elapsed time is not a direct IQ3 speed comparison.
