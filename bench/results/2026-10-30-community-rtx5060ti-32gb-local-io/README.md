# RTX 5060 Ti, 32 GB RAM: custom 128K checks with local I/O builds

Measured on 2026-10-30 by `dhsgud`, with AI-assisted preparation and analysis.
This report records a local upgrade decision: keep a modified 0.1.52 engine and
use the 0.2.8 Python server. It contributes observations from a Linux machine
with 32 GB RAM; it does not propose an engine change or a new default.

**These are custom, nonstandard checks, with one request per condition.** All
engines contain local changes. There is no stock-release control, isolated I/O
ablation, statistical speedup claim, or demonstrated upstream regression.

## Hardware and software

- One RTX 5060 Ti, 16,311 MiB reported VRAM; GPU 0 only.
- Intel Core i7-13700, 24 logical CPUs; 32 GB installed RAM, Linux MemTotal
  32,047,516 KiB (about 30.56 GiB). Memory speed was not recorded.
- NVMe model storage, NTFS3 mounted with `rw,noatime,prealloc`.
  Active PCIe link, storage throughput and inference peak memory were not measured.
- Ubuntu 26.04, Linux 7.0.0-38-generic; NVIDIA driver 595.91.07;
  CUDA 13.1.115, GCC 15.2.0; source builds for `sm_120` and native CPU instructions.
- Background inspection and command execution continued during the session;
  the machine was not isolated. A binary checksum read also occurred around the
  candidate's long-prefill interval. There was no OS-cache reset, randomized run
  order, or temperature, clock, power, paging or disk-load trace.

## Source and configuration

The archived sources are evidence for reproducing these measurements, **not
proposed merge branches**. Their full experimental bundle includes disabled paths
and has not been validated for other backends or platforms.

| Label, in measured order | Engine | Python server | Process state before checks |
| --- | --- | --- | --- |
| baseline | local 0.1.52 | local 0.2.7 | Existing process, previously used |
| candidate | local 0.1.54 | local 0.2.8 | Restarted |
| baseline-fresh | local 0.1.52 | local 0.2.7 | Restarted |
| integrated | local 0.1.52 | local 0.2.8 | Restarted |

- [0.1.52 source snapshot](https://github.com/dhsgud/Strata/tree/1138fb7f3e4367c112752a0c3c1fb79ceee2f325), based on upstream
  `5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d`.
- [0.1.54 source snapshot](https://github.com/dhsgud/Strata/tree/3a47337a3cf6d3d8211c473c3205f2f208e37b41), based on upstream
  `870aedd9e1ebe1c91f94da93b59af17729f6c5c0`.
- Both use the llama.cpp pin `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
  [provenance.json](provenance.json) records binary hashes and model/runtime asset hashes.
- Exact sanitized configurations: [baseline](configs/baseline.json) (also used for
  baseline-fresh), [candidate](configs/candidate.json), [integrated](configs/integrated.json).
  Replace bracketed paths and the API-key placeholder in a private copy.

All four arms enable the same local cold-read bundle: persistent worker,
per-expert completion publication, logical CPU 1 affinity, and bounded completion
polling of up to 2,000 microseconds. Direct read-ahead, direct destinations and
registered io_uring buffers are disabled. The source also contains diagnostic
and optional tuning changes; inspect the evidence commit rather than treating it
as only a worker-reuse patch. No default-path equivalence claim is made here.

The local server adds a context threshold for `sampling.strata_tune`: startup
`spec_min_p=0.5`, and `0.8` for prompts of at least 120,000 tokens. All arms use
this customization. The 0.2.8 arms also set `tool_call_recovery=true`; their
upstream server and engine changes are not isolated from one another.

## Model and runtime

- [loscrossos/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF](https://huggingface.co/loscrossos/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF/tree/2828d17d8bc5128e9fa8c8eb329f7a041d37787d),
  revision `2828d17d8bc5128e9fa8c8eb329f7a041d37787d`, IQ3_XXS. Its two filenames
  begin `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000`; see the exact names and
  previously verified checksums in provenance.json. The original fine-tune
  checkpoint revision was not recorded. This is not the original base model.
- A contiguous expert pack made with `tools/iq_pack.py --compat-bf16 --experts-bin`.
  The saved conversion report recorded 96 exact conversions and zero rounded
  conversions. A separate byte comparison checked all 24,576 experts
  (42,912,972,800 bytes) and the dense data against the original pack.
- The MTP head comes from `Qwen/Qwen3.8-Flash-Next` revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`, with Q2_0 draft experts. It was
  prepared with upstream `mtp_fetch.py`, `mtp_pack.py`, and `mtp_rt.py`.
- [expert-profile-long-trained.bin](expert-profile-long-trained.bin) is the numeric
  routing-count profile used in all arms. The public code-context prompt was
  prepared separately from profile training. The profile hash is in provenance.json.
- 131,072-token context; int8 KV, 20,480-token resident KV window; automatic
  prefill and expert cache; resident expert budget 21 GiB; VRAM reserve 1,024 MiB;
  MTP `--spec 3`; P-core pool affinity; async adaptation and RAM adaptation on.
  Three prompt checkpoints; conversation cache disabled. No vision or
  experimental speed projection enabled. All environment overrides are in the configs.

## Method and results

Each arm ran the upstream `standard-v1` **12-case sanity subset first**, then one
short coding request and two byte-identical long coding requests. Sanity uses
the upstream sampling and thinking settings (`temperature=1`, `xhigh`); the three
timed coding requests use temperature 0, thinking off and a 256-token output cap.
The model and its expert cache have therefore seen sanity work before timing.
Model loading is excluded. Restarted does not mean a cold OS page cache.

The long request contains [code-context.txt](code-context.txt), public Strata
source with a synthetic `ZEBRA-4729` marker, followed by the question in
[run_checks.py](run_checks.py). [prompt-provenance.json](prompt-provenance.json)
records source files and the document hash. The full rendered long prompt has
130,433 tokens. First use reads all of them; the repeated request reuses 130,426
and reads seven. A repeated prompt is a separate cache condition, not a second
independent measurement of the first condition.

| Arm | Short decode tok/s | First long prompt tok/s | First long decode tok/s | First long TTFT s | Reused long decode tok/s | Reused long TTFT s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline (existing process) | 42.3 | 722.1 | 48.3 | 181.680 | 56.0 | 0.280 |
| candidate | 39.9 | 651.8 | 23.0 | 201.606 | 55.9 | 0.647 |
| baseline-fresh | 41.1 | 702.6 | 44.5 | 185.848 | 53.3 | 0.281 |
| integrated | 40.8 | 741.0 | 46.6 | 176.235 | 56.6 | 0.175 |

All three coding requests in every arm produced 256 tokens. The short prompt has
29 tokens and no reused tokens. Speeds are engine `prompt_per_second` and
`predicted_per_second`, not tokens divided by total latency. TTFT and total
latency are client wall times; TTFT ignores keep-alives and empty deltas. Loading
is excluded from both. Full precision client times, engine milliseconds,
acceptance counts, response text and usage are preserved in the raw data.

| Arm | Raw responses (12 sanity + 3 coding) | Selected original engine timing lines |
| --- | --- | --- |
| baseline | [JSONL](baseline-outputs.jsonl) | [log](baseline-timing.log) |
| candidate | [JSONL](candidate-outputs.jsonl) | [log](candidate-timing.log) |
| baseline-fresh | [JSONL](baseline-fresh-outputs.jsonl) | [log](baseline-fresh-timing.log) |
| integrated | [JSONL](integrated-outputs.jsonl) | [log](integrated-timing.log) |

[results.json](results.json) preserves all per-request summaries and sanity
verdicts. JSONL records use `case`, then either `raw`/`verdict` for sanity or
`result`/`summary` for coding. Timing logs contain only the matching measured
requests and adjacent diagnostic lines; unrelated requests are excluded.

## Correctness and limits

- Sanity: 12/12 in each arm, with corpus SHA-256
  `e75eff8b9feea04f745520b392c4c17d97aefc1ba185e222aa6740efa8f68407`.
  This is a small sanity subset, not a complete standard benchmark or model-quality score.
- The marker appears in all eight long responses. This checks substring recall
  only, not strict compliance with the requested first-comment placement.
- Coding outputs hit the 256-token cap and are incomplete. Their code was not
  executed or scored for correctness. Different arms produced different text;
  these runs do not establish bitwise parity.
- No request failed in this saved set. The candidate's first long request was
  much slower, and that result is retained. It needs controlled repeated tests
  before attributing the difference to an upstream change or to any local patch.
- Fresh old-engine and new-engine configurations also differed in automatically
  sized GPU expert caches: startup logs reported 5,425 slots / 8.79 GiB for the
  old engine and 5,272 / 8.54 GiB for the new one. These are startup allocations,
  not memory peaks; see [selected startup lines](startup-cache.log).
  Exact adaptive placement during requests was not frozen.
- There is only one physical machine and one request per condition. There are
  no confidence intervals, cache-isolated comparisons, or cross-platform claims.

## Repeating the workload

See [REPRODUCE.md](REPRODUCE.md) for build options, model preparation and commands.
The portable runner adapts paths and argument parsing from the original local
script; the request construction, ordering and evaluators are unchanged. It has
not been rerun for new performance numbers after packaging.

The archived 0.1.54 source was built and exercised on CUDA; its saved validation
logs are under [validation](validation). HIP, SYCL, Metal and Windows builds and
hardware execution were not run. The results-only PR does not request review or
merge of those engine changes. Context on Linux file-tier behavior is in
[#773](https://github.com/Niko1221/Strata/pull/773) and
[#1194](https://github.com/Niko1221/Strata/issues/1194); this report does not claim
that forcing O_DIRECT is beneficial when experts fit in the page cache.
