# RX 9070 XT / 9800X3D / 64 GB: native Windows HIP, full Q4 and 256K context

Measured October 19, 2026 (America/Los_Angeles; JSON timestamps are October 20 UTC), contributed by visualizedigital with AI assistance. Full Qwen3.8-Flash-Next UD-Q4_K_XL ran beyond installed RAM through SSD-backed experts. Short generation averaged 9.1 tokens/s at a 32K configured window. Near-256K fresh prompts took about eight minutes to the first token. A controlled 225K–230K-input coding loop passed its tests in 13.93 minutes.

These are custom historical measurements, **not the current standard benchmark protocol**, a benchmark of current main, or proof of production coding-agent reliability. No engine or default changes are proposed.

## Hardware and software

- AMD Radeon RX 9070 XT, 16 GB VRAM, gfx1201; Ryzen 7 9800X3D, 8 cores / 16 threads; 64 GB DDR5. Actual memory clock, PCIe link width/speed, GPU power limit and temperatures were not captured.
- Windows 11 Pro 10.0.26200; display driver 32.0.31041.1004. Native Windows HIP, official RX 9000 prebuilt engine; no Linux/WSL GPU inference. Exact underlying TheRock/HIP SDK version was not retained in this report.
- Strata v0.2.0-beta, engine/control package 0.1.45.3; source/documentation checkout `e88ee39d1d6be6e3364183af3b39b62a7ed6f211`. The server metrics identify the engine as `0.1.45`; both identifiers are retained rather than treated as identical fields.
- Model storage: Crucial T705 NVMe. Runtime and logs: WD Black SN850X. Other applications remained running, including WSL2 with a 32 GiB limit. Strata was native and was not subject to that limit. This is not an idle-desktop benchmark.
- A local updater detection workaround used existing HIP architecture detection to select the RX 9000 package instead of RDNA3. No inference kernels were modified. GPU execution was verified by engine identification and independent Windows WDDM counters.

## Model and configuration

Full Unsloth Qwen3.8-Flash-Next UD-Q4_K_XL, not the reduced-expert Coder. The four GGUF filenames, byte sizes and verified SHA-256 hashes are in [model-manifest.json](model-manifest.json). Model revision was not separately recorded; hashes identify the tested files. Download size is 111.3 decimal GB; expert weights approximately 71.7 GiB. Standard prepared pack, existing expert profile and MTP draft were reused; profile/draft hashes are not supplied here.

[config.example.json](config.example.json) preserves the tested 256K flags with filesystem paths replaced. Adjust paths and GPU selection to your installation; physical GPU 1 was the discrete GPU on this machine. Do not copy that index blindly.

- Total context 262144; int8 KV; GPU-resident KV positions 32768, with the remainder offloaded (about 3.09 GiB pinned host KV). **32768 is not the total context limit.** Vision off.
- Requested host resident budget 38 GiB, VRAM reserve 2048 MiB, expert cache and prefill auto, requested MTP depth 4, minimum draft probability 0.5. `STRATA_ROUTE_TAIL_SKIP=0`.
- During 256K recall/coding: actual host expert cache 18.61 GiB, GPU expert cache 1097 slots / 3269 MiB. Runtime metrics, including actual prefill cap and worker count, are in [coding.json](coding.json). Requested MTP depth differs from some internal runtime `spec` fields; raw fields are preserved.
- Earlier 32K-profile short/8K probes: host expert cache 22.08 GiB, GPU cache 1234 slots / approximately 3.59 GiB. Available Windows memory/commit clamped the requested host budget.
- Temperature 0, seed 42, reasoning effort `none`, streaming; inference output cap 192, coding per-turn output cap 4096. No intentional experimental projection setting; automatic cache planning was enabled.

## Method and reproduction

[scripts/benchmark.py](scripts/benchmark.py) contains the exact prompt construction and timing method. [scripts/long_context_coding.py](scripts/long_context_coding.py) contains the coding tool loop. Copies differ from the executed scripts only in workspace/output paths and resolving Node through PATH. These portable copies were syntax checked. Publication validation replayed the recorded model writes under local Node: the original fixture failed, the repaired fixture passed all 15 tests and five independent checks, and original tests stayed byte-identical. This replay is not a new inference measurement; the historical GPU benchmarks were not rerun for publication.

With Windows Python, Node on PATH, and the configured model already running on loopback port 8081, run from this report directory:

```powershell
python -m pip install psutil
New-Item -ItemType Directory -Force results
# Use the 32K profile for the first two historical groups.
python scripts/benchmark.py --label q4-quality-short --runs 3
python scripts/benchmark.py --label q4-quality-8k --context 8192 --runs 3
# Restart with config.example.json's 262144-context configuration.
python scripts/benchmark.py --label q4-varied-256k --context 262144 --probe varied --runs 3
python scripts/benchmark.py --label q4-context256-spec4-short --runs 3
python scripts/long_context_coding.py --label q4-coding-220k --target-input 225280
node scripts/check_fullstack_edges.mjs coding/q4-coding-220k
```

These commands assume a prepared Strata installation; the JSON is a runtime configuration example, not a standalone installer. The historical 32K profile used the same principal flags without `--kv-resident`, with `--max-context 32768`. Test only one group at a time. The coding script refuses to overwrite an existing task directory. Its tools can read the synthetic fixture, edit three source files/a new regression test, and execute Node tests; no production project is involved.

Three requests per inference group, with varied run identifiers and zero reused prompt tokens in the published groups. Model/expert cache was retained within each group; there was no cold restart between requests. Readiness generation preceded the groups, but cache residency can adapt. Long recall prompts are seeded synthetic TypeScript bundles with markers near the beginning, middle and end. Counts come from the server, not declared context limits.

Prompt and generation throughput use server timing fields separately. Client TTFT starts immediately before the streaming request and ends at the first content delta (or first tool delta in the coding loop); empty events do not count. Client wall time excludes loading. Start-to-generation-ready was 30.73 seconds for the earlier Q4 profile and 31.95 seconds for 256K, including readiness generation.

[runs.json](runs.json) retains all 15 selected per-request timing/usage records, UTC timestamps, prompt hashes, request settings, system CPU/RAM samples, physical disk read deltas and recall outcomes. Large generated prompts, duplicated streaming events and device/process identifiers are omitted. Prompts are reproducible with the script and original labels. Disk deltas cover all processes and a slightly longer interval than HTTP wall time. RAM is whole-system usage; CPU samples are whole-system percentages.

## Results

Numbers below are medians (minimum–maximum), except explicitly labelled means. Individual values and total request latencies are in `runs.json`.

| Profile / probe | Actual input / output tokens | Prompt tokens/s | TTFT seconds |
| --- | --- | --- | --- |
| 32K / short, n=3 | 42 / 192 | 6.9 (6.6–7.5) | 6.216 (5.715–6.530) |
| 32K / near-8K recall, n=3 | 7631 / 6 | 440.6 (440.1–448.8) | 17.484 (17.139–17.538) |
| 256K / near-full recall, n=3 | 261593–261605 / 27 | 526.1 (524.2–550.0) | 497.455 (475.929–499.276) |

The very short prompt throughput includes fixed overhead and should not be extrapolated to large prompts. Six- and 27-token recall outputs are too short to establish sustained generation speed.

| Short 42-input / 192-output group | Individual generation tokens/s | Mean | Median |
| --- | --- | --- | --- |
| 32K configured, MTP 4 | 8.6, 9.8, 8.9 | 9.10 | 8.9 |
| 256K configured, MTP 4 | 7.7, 8.5, 8.3 | 8.17 | 8.3 |
| 256K configured, MTP 3 | 7.6, 8.3, 8.0 | 7.97 | 8.0 |

The MTP-3 restart also received a larger host expert cache (23.10 GiB) and 1031 GPU slots. This is **not a clean single-variable comparison** and does not establish a causal MTP effect. Depth 4 was retained. Likewise, the 32K/256K comparison also changes cache allocation.

All nine marker checks across the three full-window requests passed. First full-window request: physical T705 reads increased 2842.62 GiB in 481.99 seconds (5.90 GiB/s, all processes); system RAM 54.58–57.91 GiB, mean CPU 99.22%, dedicated GPU usage 14.79–14.83 GiB. This demonstrates heavy physical storage activity, not a measurement of time stalled exclusively on SSD I/O.

## Coding and API findings

[coding.json](coding.json) retains the six model turns, tool arguments/results, engine configuration and Node test output. Local absolute workspace paths were replaced; timings and code are unchanged. [fixtures/fullstack](fixtures/fullstack) is the deliberately broken starting project; the model's writes in the transcript reconstruct the repair. The independent checker and [independent-checks.json](independent-checks.json) record additional verification.

- Actual input grew from 225601 to 230602 tokens, with no history truncation or compaction. Total elapsed 835.61 seconds; output 2667 tokens across six model calls and nine tool calls.
- Initial TTFT 428.85 seconds, fresh prefill 526.3 tokens/s. Subsequent first-token/tool latency 8.40–11.91 seconds with 225719–229814 cached tokens. The two substantial source/test writes generated at 7.1 and 7.4 tokens/s.
- Baseline tests failed as intended. Final 15/15 tests passed (five original, ten added), plus five independent checks. Original tests were byte-identical; no manual implementation changes. No malformed tool calls; the initial failing test command was expected feedback.
- System RAM averaged 53.63 GiB (52.52–54.36); mean CPU 82.01%; dedicated GPU usage about 14.85 GiB, shared GPU memory about 24.92 GiB. Shared usage includes intentional pinned host memory and is not alone evidence of accidental paging.
- Separately, native Claude Code 2.1.233 completed an Express multi-file Q4 task in 1501.52 seconds with 21 tool calls, three recovered tool errors, 11 visible tests and an independent HTTP concurrency check passing. This was a 32K-configured baseline, **not** a full-window Claude Code test. Its full transcript is not included here; the included reproducible coding evidence is the controlled full-stack loop.
- OpenAI and Anthropic structured tool round trips, streaming, malformed-request HTTP 400 and subsequent successful recovery were checked locally. Codex/OpenCode interoperability was not exercised.

## Limitations and follow-ups

- The coding project is small and embedded in a large synthetic distractor bundle; tools reread its files. Success does not establish comprehension of a 225K-token production repository. Marker recall is not a broad reasoning evaluation.
- This submission selects Q4 measurements from a larger local evaluation. Earlier IQ3 task outcomes and Vulkan results are not a controlled quality/backend comparison and are not included as headline numbers.
- No 128 GB RAM system or Mac was measured here. No upgrade speedup or keep-versus-sell recommendation follows from this report.
- Browser/IDE/Docker responsiveness and thermal behavior were not measured. Other applications were preserved, not standardized. No pressure-free baseline is available.
- Some Windows formatted GPU engine counters exceeded 100% and were excluded from local utilization summaries; no aggregate GPU utilization claim is made. `/metrics.hardware_static` labelled the integrated GPU, whereas the engine log and WDDM adapter memory identified the discrete RX 9070 XT. These are observations on the tested version, not confirmed current-main bugs.
- One later restart allocated 25.33 GiB host cache; that is not the allocation used for the published long-context measurements.
- Higher precision, 512K context, larger real repositories, full-window Claude Code tasks and the newer standard benchmark protocol remain untested. No GPU tests were rerun against current main for this historical submission.
