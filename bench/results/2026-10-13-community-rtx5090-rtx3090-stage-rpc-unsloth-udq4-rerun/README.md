# Unsloth UD-Q4_K_XL on RTX 5090 + RTX 3090: two-host stage-RPC rerun

Measured 2026-10-13 by @petria. This is a new run of the model already reported in
[PR #2166](https://github.com/Niko1221/Strata/pull/2166), not a different model or an update to that PR.
It uses our custom Strata stage-RPC fork, **not released upstream RPC support**.

## Hardware, model and software

- Two Linux hosts: Ryzen 9 9950X3D / RTX 5090 coordinator; Ryzen 9 5900X / RTX 3090 prefix worker.
- GPUs: approximately 32 GiB / 24 GiB physical VRAM. Coordinator VM RAM approximately 42.6 GiB; worker approximately 98.2 GiB. Exact observations, drivers, CPU topology and mount details: [hardware-and-integrity.json](hardware-and-integrity.json).
- Strata 0.1.43 base `7695859aa13afb248bdb4d36f9b2e95707db079d`, branch `feature/tcp-stage-rpc`, matching custom source/dependency fingerprint `85ea98c3eb3c1d6e14760c83506d3265d6f5eaabdd44fd049df704936fe3080f`. CUDA 13.3, K-quant MMQ enabled; retained binaries, no rebuild or upgrade for this test.
- [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), revision `38bb39ee97821de2c9009abb7e93950eec396e66`, UD-Q4_K_XL: four shards totaling 111,334,654,784 bytes. Native experts/compat-BF16 dense pack, base-Qwen MTP pack and retained vision projector. Manifest hashes and workload identities: [provenance.json](provenance.json).
- Each node reads local model files; the private Ethernet link transports activations/control/state, not expert weights. Desktop applications and an unrelated idle llama.cpp RPC process remain on the worker; they were not killed for the benchmark. No disk/network capacity or power-limit tuning test was performed.

## Launch configuration

192K context (196,608), INT8 GPU-resident KV, 24/24 stage split, fixed 2048 prefill;
resident experts, automatic cache; MTP maximum 4 / minimum probability .5, suffix lookup retained;
VRAM reserves 4096 MiB worker / 1000 MiB coordinator; CPU pools 11 / 31; one active sequence/FIFO.
Medium reasoning default; speed/recall explicitly turn it off. Vision remains enabled but no images are tested.
Sanitized actual configs: [launch.json](launch.json). Native port 8090, Responses for Codex; no custom llama.cpp proxy.
The current worker cache is 3,522 experts / 10,526 MiB versus 4,110 / 12,292 MiB in #2166;
the coordinator retains 7,244 / 21,650 MiB. Automatic allocation responds to available desktop VRAM:
this is a measured configuration difference, not a controlled claim about the cause of a speed change.

## Method and speed results

Unchanged upstream `tools/strata_bench.py --runs 3 --long`, executed on the coordinator against its existing server.
Frozen tools **and** public document/source corpus at the base commit; both-node telemetry sampled before the first request.
36 measured requests: 9 chats, 9 fresh prompt tests and 18 agent-cache turns. Four warm-up/calibration requests are excluded.
Reasoning off, temperature zero; chat cap 100 tokens, long-prompt cap 24, agent cap 100. Report actual generated lengths, not assumed caps.
Fresh prompts have unique tags; expert cache warms/adapts and is retained between tests. Loading and packing are excluded.
The agent-cache fixture asks for document summaries and explicitly forbids tool calls: **it is not the coding test**.

[Upstream speed report](upstream-speed-report.md), [all speed requests](results.json), [console](benchmark.log),
[native accounting](accounting.json), [both-node sampling](telemetry.jsonl).

| Metric | Previous Unsloth (#2166) | New Unsloth | GSQ IQ3_S |
| --- | --- | --- | --- |
| Chat decode t/s | 71.30 (60.00–78.80; n=9) | 66.40 (51.70–80.40; n=9) | 135.60 (107.90–173.10; n=9) |
| 4K fresh prefill t/s | 1226.40 (1224.20–1226.40; n=3) | 1177.20 (1062.70–1182.20; n=3) | 2581.60 (2568.10–2606.20; n=3) |
| 16K fresh prefill t/s | 1420.40 (1419.30–1424.00; n=3) | 1362.80 (1319.70–1366.20; n=3) | 3273.00 (3258.10–3290.30; n=3) |
| 32K fresh prefill t/s | 1446.10 (1445.10–1454.40; n=3) | 1399.10 (1395.00–1399.10; n=3) | not measured |
| Agent follow-up decode t/s | 88.70 (61.50–124.90; n=15) | 81.10 (72.70–114.10; n=15) | 140.80 (81.80–191.10; n=15) |
| Agent follow-up TTFT s | 2.29 (1.86–3.20; n=15) | 2.96 (1.92–4.49; n=15) | 1.29 (0.90–1.52; n=15) |

Values are medians (range; sample count). The old Unsloth run and this rerun use the same target model/context/split,
but worker cache availability, desktop load and generated answers differ. GSQ uses 128K, 16/32 layers and text-only mode,
while this run uses 192K, 24/24 layers and vision enabled. GSQ lacks the extra 32K speed arm.
These are practical-stack comparisons, **not isolated model-quality, RPC-overhead or hardware A/B conclusions**.
GSQ comparison source: [retained raw speed results](comparison-baselines/gsq-iq3s-results.json);
baseline identities/hashes: [source.json](comparison-baselines/source.json).

## Repository-only recall and coding

- Recall: 6/6 cases passed, targets 32K/128K and depths 10/50/90, fixed seed 7, greedy, reasoning off, cap 40. Actual counts and answers: [needles.json](quality/needles.json).
- Coding: 9/9 attempts passed; independently verified 243/243 assertions. This is **81 distinct acceptance cases repeated three times**, not 243 distinct tasks.
- Fresh ephemeral Codex 0.162.1 sessions, workspace-write, approvals never, medium reasoning, temperature 1 / top-p .95 / top-k 20 / min-p 0; Node v22.23.2. No user config or personal conversation history is used.
- Only the three existing RX 9060 XT TypeScript fixtures, each copied byte-for-byte from the frozen upstream commit. Visible tests, source-only scope, no installation/network/delegation, at most two repair rounds. Independent acceptance uses read-only, network-disabled Bubblewrap copies and 60-second timeout; no evaluator repairs. Attempt timeout 15 minutes, quality collection limit four hours.
- Completed tool calls: 65. Per-attempt timing, scope/integrity checks, errors and independent logs are retained below; failures are not retried away.
- One command exited 128 because `git status` ran in a non-Git fixture directory after a test command. It was not a backend failure or failed acceptance check; all independent test runs passed.

| Fixture | Attempt | Passed | Acceptance | Wall seconds | Tool calls |
| --- | ---: | --- | --- | ---: | ---: |
| merge-windows | 1 | True | 25/25 | 89.3 | 6 |
| merge-windows | 2 | True | 25/25 | 81.9 | 5 |
| merge-windows | 3 | True | 25/25 | 82.3 | 7 |
| refactor-orders | 1 | True | 24/24 | 104.1 | 11 |
| refactor-orders | 2 | True | 24/24 | 103.6 | 9 |
| refactor-orders | 3 | True | 24/24 | 91.3 | 8 |
| reservation-workflow | 1 | True | 32/32 | 102.6 | 7 |
| reservation-workflow | 2 | True | 32/32 | 107.1 | 6 |
| reservation-workflow | 3 | True | 32/32 | 89.0 | 6 |

Quality collection weighted native rates: fresh prefill 1313.8 tok/s, decode 76.4 tok/s,
MTP acceptance 73.03%. Engine-generated tokens include reasoning/tool serialization and are kept separate from Codex usage.
These rates include recall and medium-reasoning coding, and are **not mixed with the speed suite**.
Detailed [quality summary](quality/summary.json), [native requests](quality/native-requests.json),
[per-attempt rates](quality/per-attempt-native-rates.json), [telemetry](quality/telemetry.jsonl).

## Memory, validation and limitations

- Worker native-process sampled peak 16290 MiB (15.91 GiB), below the 20-GiB inference guard. Whole-card allocations include the desktop; the 4-GiB reserve is a startup budget, not a driver quota. Peaks are sampled lower bounds.
- Speed and quality native request/token/timing/draft records reconcile with boundary counters. Both engine configurations, binary identities, base/source-diff identities and service invocations remain unchanged during testing.
- Optional request-body monitoring stays disabled. Collectors support its HTTP 404 without dropping native measurements. Telemetry covers the start of each workload; no gap from the old collector incident is carried into this run.
- A staging preflight initially found the remote frozen-workload directory missing; it issued **zero inference requests**. The corrected directory and 517 consumed source files were verified before the published run. Failed preflight artifacts are retained privately, not counted as measured tests.
- Recall measures these six synthetic positions, not broad reasoning or full-192K reliability. Coding uses visible tests and only three small community fixtures. No gistfile1/fable, broad leaderboard, vision, concurrency, cold-cache or same-seed equivalence test.
- NIC byte deltas include telemetry and other interface traffic; do not sum sender and receiver as distinct payload. RX/TX peaks are sampled, not link-capacity measurements. CPU/storage/desktop load and output/MTP variability limit comparisons.
- The custom engine patches/binaries are not distributed in this results-only PR: upstream base alone cannot reproduce network RPC. Reproduction assumes the compatible installed deployment. No model files, generated packs, credentials or private network addresses are published.

See [REPRODUCE.md](REPRODUCE.md) for replay commands, units and data boundaries.
