# NCM feasibility study

Status: closed by user decision, stop/document, 2026-10-19. Partial results retained. No engine integration or default changes.

## In short

NCM is a new kind of small language model that writes many words at once instead of one at a time. We
checked whether it could help Strata guess its next words faster. The answer for now is no, and nothing in
Strata changed.

- NCM's released example model ran on an RTX 5070: 128 tokens in about 40 ms. But it writes children's
  stories with its own word list. It cannot make guesses Strata's model can check.
- No released NCM model makes guesses for Strata's model. To help, at least 83 to 96 of every 100 of its
  guesses would have to be right (worked out from same-day measurements, not measured). This study did
  not test whether any NCM model gets close.
- On the same PC, Swift 1.5 IQ2_XS ran at 18.8 to 20.2 tokens per second with its normal guessing. With
  perfect guesses of 8 tokens it ran at 30.6 to 38.2. That is the most an 8-token guesser could add in this
  test setup (single runs, short prompts, not the saved settings).

The details, every number and how to repeat it follow. The raw files are in
[`docs/benchmarks/ncm-2026-10-19-rtx5070/`](benchmarks/ncm-2026-10-19-rtx5070/README.md).

NCM's released TinyStories model ran on this PC. This establishes standalone generation cost. It does not
establish acceptance by Strata or a speedup. Target pilot measurements are below. The user selected stop/document;
no training, integration or skeleton phase will follow.

## What the source says

Reviewed [HazyResearch/ncm](https://github.com/HazyResearch/ncm/tree/c70274273f489abc73832b2b98584a8483b6b203)
at commit `c70274273f489abc73832b2b98584a8483b6b203`. These are source observations, not integration tests:

- `models/next_concept.py` generates residual code levels in order. The VQ-VAE decoder converts the result
  to text token logits. Code predictions within a level are parallel; they are not final text tokens.
- `config/eval/tinystories.yaml` uses a custom BPE vocabulary of 8,192 text tokens and 12 codebooks of 4,096
  codes. Its first, one-code level is sampled from the checkpoint's unigram distribution. Eleven transformer
  passes predict levels of 8, 16, 24, 32, 40, 52, 64, 76, 92, 108 and 128 codes, followed by text decoding.
- The plan's statement that no release uses Qwen tokenization was too broad.
  [`config/eval/wmt14.yaml`](https://github.com/HazyResearch/ncm/blob/c70274273f489abc73832b2b98584a8483b6b203/config/eval/wmt14.yaml)
  uses `Qwen/Qwen3-0.6B` with 151,680 decoder outputs. It is a translation model with a 96-token source prefix
  and a 64-token target. This does not establish usefulness for arbitrary continuation.
- The installed Swift pack's `tokenizer/tokenizer.json` reports vocabulary size 248,320, EOS 248046 and
  padding 248044. The WMT14 config uses EOS `<|im_end|>` and padding ID 151643. Their IDs cannot simply be
  treated as the installed model's IDs. Text decode followed by target tokenization is possible in principle;
  its latency, boundary handling and draft acceptance have not been tested. Training a new model is not the
  only conceivable route, contrary to the plan's original decision checkpoint.
- `evaluate.py` starts CUDA generation shards and a scoring process. No serving API was found in that checkout.

The VQ tokenizer has a text-token embedding and a decoder over its input vocabulary
(`models/vqvae.py`, `models/vae.py`). Replacing its text tokenizer requires adapting those components and
training the representation and generator; changing codebook IDs alone does not make it Qwen-compatible.

The [paper, Appendix C](https://arxiv.org/html/2610.14162v1#A3) reports TinyStories tokenizer training with
30,000 updates and global batch 4,096, then generator training with 50,000 updates and batch 2,048, on
128-token targets. That implies about 28.84 billion target-token positions across both stages, including
repeat exposure. No GPU-hour budget for a Qwen continuation drafter was found. It cannot be inferred from
these updates. The paper's inference comparison uses an H100 SXM 80 GB and diffusion baselines; it is not
a measurement against Strata on this PC.

The target's [upstream config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json), fetched
on 2026-10-19, lists 48 layers, 512 routed experts per layer and 10 selected per token per layer. Thus 24,576
is the number of routed experts across layers, not a per-layer count; 10 is also per layer. These are model
metadata, not a measured routing union. This study does not use the plan's unverified “125B” figure.

## Standalone measurement

Hardware: NVIDIA GeForce RTX 5070, `nvidia-smi` reports 12,227 MiB; AMD Ryzen 9 7900X, 24 logical CPUs,
61 GiB system RAM as displayed by `free -h`. Linux kernel 7.0.0-38, Python 3.12.3, PyTorch 2.12.0+cu130,
CUDA runtime 13.0, driver 610.43.02. Desktop display applications remained active. GPU checking and inference
required execution outside the filesystem/network sandbox.

Command, from the repository root:

```sh
/tmp/strata-ncm-venv/bin/python -I tools/ncm_probe.py \
  --repo /tmp/strata-ncm-evaluation --ckpt /tmp/strata-ncm-checkpoint \
  --output /tmp/strata-ncm-probe.json
```

The probe uses the released TinyStories config, FP32, no autocast, batch 1, seed 42, four CPU threads,
three warm-up generations and 20 measured generations. Each iteration separately times latent generation
and full generation. Full generation includes CPU first-level sampling, device transfers, all levels,
VQ decoding, token argmax, transfer to CPU and BPE text decoding. Model loading is excluded. CUDA is
synchronized at timing boundaries. A separate generation profiles each complete transformer level;
those instrumented times are not included in throughput. Peak memory includes that profile.

| Quantity | Observed on this RTX 5070 |
| --- | ---: |
| Generator parameters, counted from loaded tensors | 88,605,707 |
| Tokenizer parameters, counted from loaded tensors | 36,434,816 |
| Combined unique parameters | 125,040,523 |
| Final text positions per generation | 128 |
| Transformer level passes per generation, separately profiled | 11 |
| Median full generation time | 40.504 ms |
| Throughput, total text positions / total elapsed time | 3,140.869 BPE tokens/s |
| Visible tokens after EOS/pad filtering | 2,560 of 2,560 positions |
| Peak PyTorch allocated memory | 777,734,144 bytes |
| Peak PyTorch reserved memory | 799,014,912 bytes |

These are TinyStories tokens, not target tokens. Memory excludes the CUDA context, desktop applications
and a loaded Strata model. Simultaneous residency, text quality and acceptance were not measured.
Per-level code counts and all individual timings are preserved below.

Checkpoint files were fetched from `hazyresearch/ncm-checkpoints/tinystories`. Their SHA-256 values,
produced with `sha256sum /tmp/strata-ncm-checkpoint/*`, are:

```text
bpe_merges_8192.pickle  0d4cab13f2e7b684b75eaa290a52e596ba87744ef13d943cf3a50a5a8e080ec1
bpe_vocab_8192.pickle   fb176b4c68108e4b5ef61b9ff2af3731600124e3e160970a6bd78cd0a5d3ff6c
config.yaml            4009baccad5a159bfc12cd48f3da17c22c4a281b86f71f2fc70b7c1bbb62004b
generator.pt           0acaf4f795390f9f7ccc8f84047867619a1d00e291681e299b855d7e563be6ff
tokenizer.pt           76e26003719db32dc3d542518951c286be07d6a5881f29ad5152d299764b0f81
```

To repeat, clone the pinned checkout outside Strata, create a virtual environment there, install its
dependencies, and download those five files. The local scratch venv reused installed CUDA dependencies
through a `.pth` file; no system or existing environment packages were changed. The probe intentionally
loads upstream Python and checkpoint pickle files. `python -I` prevents implicit current-directory imports;
it is not a security boundary for that code or those files.

## Break-even calculation

Measure the target baseline as committed tokens per second, `B`. For a candidate verify window of `T`
tokens (one target token plus `T-1` drafts), let `V(T)` be total verify/commit overhead in milliseconds,
and `D` the complete drafter cost including conversion and transport. Then:

```text
candidate tokens/s = 1000 * E[committed tokens] / (V(T) + D)
required E[committed tokens] > B * (V(T) + D) / 1000
E[committed tokens] = 1 + q + q^2 + ... + q^(T-1)
```

The last equation assumes constant conditional acceptance `q`. An observed aggregate acceptance ratio
alone is insufficient: draft prefix lengths and their dependence matter. Compare against the measured
MTP-plus-lookup baseline as well as MTP alone. MTP's published tokens/pass are not a substitute for either.
If the required expectation is at least `T`, even perfect drafts cannot beat that baseline.

The probe's `--break-even` mode solves the equality threshold without importing PyTorch. For example:

```sh
python3 -I tools/ncm_probe.py --break-even --window 4 \
  --baseline-tps 80 --verify-ms 30 --draft-ms 5
```

This is an **arbitrary arithmetic example**, not a hardware measurement: it requires 2.8 committed
tokens and gives an equality threshold `q = 0.7654846613`. Beating the baseline needs a higher value.
The full 128-token NCM generation cost must not be divided by 128 and used as a one-block draft latency
without demonstrating reuse. The TinyStories run is also not a Qwen drafter latency measurement.

## Target measurements

The user's local model was located under `../Strata-data`: Swift 1.5 IQ2_XS,
with a prepared pack and MTP assets. This is a derivative model, not the original Qwen checkpoint.
`python3 strata.py launch --status` reported no launch-managed server. The installed model can be
benchmarked directly without opening a network listener.

Static limit: [`kVerifyMaxT`](../include/strata/kernels/verify_kernels.hpp) is 8. A 16-token oracle window
cannot be measured under the plan's prohibition on engine changes. `STRATA_SPEC_DEPTH` logs MTP acceptance;
it does not raise this limit. The existing `--dump-routing` trace can measure complete routing unions.
The CLI's “CPU experts distinct” counter covers CPU misses only and must not be labelled all unique experts.

### Target pilot configuration

Same hardware as the standalone run. The existing `engine/strata` identifies itself as engine 0.1.45;
its SHA-256 is `245b6e55b3bf1659dfc0bd9105cedfe08a3229714a1a69c9af7229db17d5936c`.
The checkout is `e88ee39d1d6be6e3364183af3b39b62a7ed6f211`. The binary was already built; its precise build
revision was not established. These are existing-binary measurements, not validation of a new build.

Ten sequential runs, one per mode and prompt, each generated 64 tokens with greedy sampling. The full
Terminal test run and standalone NCM run had finished before these final measurements. They use the
saved `strata-swift-iq2_xs.json` engine arguments, with explicit diagnostic overrides: context 4,096,
KV residency 0 and `--mmap-experts`. Saved production settings were not edited. Expert cache remained
`auto`, prefill `auto`, KV `int8`, and MTP confidence floor 0.5. Thinking was disabled in the rendered
prompt. No early EOS stopping was requested. This is a short custom pilot, not the draft standard
benchmark protocol, and not a measurement of the saved 64K production configuration.

Prompt text, rendered using the installed pack's chat template and tokenizer:

```text
prose: Explain how a bicycle stays balanced in three short paragraphs.

code: Change only the function name add to sum_pair and return the complete file.
```

The code prompt then contains this fenced Python file, including the opening and closing fences:

```python
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

assert add(2, 3) == 5
```

The rendered prompts contain 24 tokens for prose and 70 for code. Token files, exact commands,
stdout, stderr and generated IDs are in [`docs/benchmarks/ncm-2026-10-19-rtx5070/target/`](benchmarks/ncm-2026-10-19-rtx5070/target/).
The routing traces are not kept; their SHA-256 values are in `target/routing-bin.sha256`. The driver is
[`tools/ncm_target_bench.py`](../tools/ncm_target_bench.py). It ran from a scratch file with fixed paths;
the repository copy takes them as arguments and builds the same ten argument lists. Run it with:

```sh
python tools/ncm_target_bench.py --pack "$DATA/packs/swift-iq2_xs" \
  --config strata-swift-iq2_xs.json --out /tmp/strata-ncm-target-v2
```

To reproduce the engine runs, set `PACK`, `SHARD1` and `MTP` to the installed pack, first Swift shard,
and MTP runtime directory. Each `*.command.json` holds the exact argument array. Common arguments were:

```sh
engine/strata --pack "$PACK" --native "$SHARD1" --ple-gguf "$SHARD1" \
  --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto \
  --spec 4 --spec-min-p 0.5 --mtp "$MTP" --max-context 4096 \
  --kv int8 --kv-resident 0 --mmap-experts --max-new 64 --greedy \
  --tokens-file /tmp/strata-ncm-target-v2/prose.prompt.ids
```

Use the code token file for the other prompt. Mode changes to those common arguments:

- MTP only: add `--suffix-draft 0`.
- Default prompt lookup: leave suffix options unset. The engine automatically raises the available
  verify capacity from 4 to 6; the policy chooses actual windows. This is not `--lookup-chain`.
- Greedy reference: remove `--mtp DIR`, set `--spec 2`, add `--suffix-draft 0 --spec-corrupt 1`.
  The native IQ pack rejected `--spec 0`. This reference verifies dummy drafts; it is not plain decode.
- Oracle: remove `--mtp DIR`, set `--spec 4` or `8`, set `--spec-min-p 0`, add `--suffix-draft 0`,
  `--spec-oracle PROMPT.oracle.ids` and `--dump-routing TRACE.bin`. Oracle IDs are the reference output.

| Workload | MTP only, tok/s | Default prompt lookup, tok/s | MTP tokens/round | Default tokens/round |
| --- | ---: | ---: | ---: | ---: |
| Prose, RTX 5070 / Ryzen 9 7900X | 19.58 | 20.15 | 1.88 | 1.83 |
| Code, RTX 5070 / Ryzen 9 7900X | 19.01 | 18.79 | 3.14 | 3.30 |

All modes' 64 output IDs were exactly equal to their prompt's greedy reference, including both oracle
sizes. This tests these short greedy fixtures only. It does not establish NCM output equivalence, sampled
decoding equivalence or release-to-release equivalence. The plan's quoted 53–94 tok/s and 2.4–3.2 tokens/pass
are published context, not this machine's same-day baseline.

Single runs, fixed ordering, automatic cache sizes and mapped file/page-cache state can affect these
results. Removing MTP frees VRAM for additional experts; an actual NCM drafter would consume VRAM again.
The oracle rows are therefore optimistic proxy costs for an integrated drafter. No percentage speedup or
change to defaults is justified by this pilot.

### Window cost and conditional acceptance

Routing unions were computed from `--dump-routing`: read each `(layer, count, IDs, weights)` record,
combine consecutive records for a layer within a window, and count distinct IDs. Include only windows
with all 48 layers and exactly T records per layer. Final short windows and prompt-only groups are
excluded. This counts all routed experts, rather than only CPU misses.

| Hardware and workload | Window T | Complete trace windows | Mean unique experts/layer | Cycle proxy, ms | Conditional q threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX 5070 / Ryzen 9 7900X, prose | 4 | 16 | 25.439 | 122.029 | 0.867931 |
| RTX 5070 / Ryzen 9 7900X, prose | 8 | 8 | 40.818 | 186.256 | 0.830834 |
| RTX 5070 / Ryzen 9 7900X, code | 4 | 16 | 27.710 | 156.247 | 0.955395 |
| RTX 5070 / Ryzen 9 7900X, code | 8 | 8 | 45.005 | 232.489 | 0.871326 |
| RTX 5070 / Ryzen 9 7900X, either | 16 | not run | not measured | unsupported | not established |

**Cycle proxy** is reported total decode milliseconds divided by reported speculative rounds, including
initial/final bookkeeping and graph costs. It is not an isolated CUDA forward time. The raw component
timings are preserved in the target results below. The fixed-window runs used 17 rounds for T=4 and 9
for T=8; the trace includes respectively 16 and 8 complete full-sized windows.

The q thresholds are **conditional calculations**, not measured NCM acceptance. They use the faster
observed baseline for each workload (20.15 tok/s prose, 19.01 tok/s code), the cycle proxy as V(T), and the
standalone TinyStories full-generation median 40.5040355 ms as D. They omit simultaneous residency,
retokenization, context adaptation and transport. Under those assumptions both tested sizes can beat
the baseline at sufficiently high q; four is the smallest size tested, not a proven minimum viable size.
Run `tools/ncm_probe.py --break-even` with those values to reproduce the thresholds.

The 8-token windows increased both expert unions and cycle cost relative to four. Neither measured
window establishes near-single-token pass latency. A native plain-decode baseline was unavailable;
the dummy-draft reference is explicitly a two-token verifier. No largest near-decode window is established,
and extrapolating these unions to a 16-token or full-sequence pass would be unconfirmed static analysis.

## Validation and decision

Passed:

- Standalone released-checkpoint probe, exact tensor counts, synchronized timings and checkpoint hashes.
- Ten target pilot runs exited successfully; all mode outputs matched their greedy reference.
- Probe syntax, arithmetic equality and feasibility boundary checks, `--help`, and rejection of NaN input.
- `git diff --exit-code -- src include serve strata_cli strata_control apps requirements.txt requirements-monitor.txt`
  returned 0. Engine source, application source and dependency manifests remain unchanged from the checkout.
  The research adds only this document and `tools/ncm_probe.py`, plus `.paul` bookkeeping.

Failed:

- `/tmp/strata-terminal-venv/bin/python -m pytest tests/terminal -q`: **834 passed, 18 failed, 17 skipped**,
  440.34 seconds. Full log: [`docs/benchmarks/ncm-2026-10-19-rtx5070/terminal-tests.log`](benchmarks/ncm-2026-10-19-rtx5070/terminal-tests.log). Failures occur in
  discovery (3), shell syntax (1), access-screen text (1), lifecycle (7), private-file handling (1), process
  discovery (1), runtime (1) and light-theme UI (3). No code in those components was changed. Causes were
  not fully diagnosed, and the suite is not reported as passing.
- Initial test collection lacked runtime packages. A clean scratch test venv with pytest 8.3.5, Textual
  8.2.8, Jinja2 3.1.6, PyYAML 6.0.3, psutil 7.2.2, NumPy 2.5.3 and regex 2026.9.10 resolved collection.
  Sandbox execution prevented some local-process tests, so the complete run above was outside it.
- Initial native-pack attempt with `--spec 0` was rejected as unsupported. An initial target sweep also
  allowed default suffix lookup to enlarge the windows; it was interrupted and replaced by the explicit
  final sweep. Those partial initial logs are not kept.

Not run / unresolved:

- 16-token windows: unsupported without an out-of-scope engine change.
- Target-conditioned NCM acceptance, retokenization, shared-GPU memory pressure, quality and an end-to-end speedup.
- Repeated/interleaved target runs and the full draft standard benchmark protocol.
- CUDA/HIP/SYCL rebuilds: no backend code was changed; no new-build claim is made.

**Final decision: stop/document, selected by the user. Do not integrate a released NCM checkpoint now.** There is no tested
target-conditioned drafter, and the standalone benchmark cannot establish one. The evidence does not rule
out future NCM work. A retokenization experiment is conceivable before committing to training; a skeleton
workflow would change the user's output and needs separate quality evaluation.

The phase is closed with partial evidence: AC-1 is satisfied for TinyStories; AC-2 has a conditional model and same-day pilot
but no measured NCM acceptance; AC-3 has 4/8-token unions but no supported 16-token or near-decode bound;
AC-4 is satisfied for runtime source, with the specified `.paul` bookkeeping exception. Terminal tests
also have failures. Closure records the user's stop/document choice; it does not establish that the unmet
criteria passed. The roadmap's sole phase ends here. No training, integration or skeleton work is planned.

## Raw standalone result

Same as [`docs/benchmarks/ncm-2026-10-19-rtx5070/standalone-probe.json`](benchmarks/ncm-2026-10-19-rtx5070/standalone-probe.json).

```json
{
  "utc": "2026-10-19T19:14:12.919455+00:00",
  "platform": "Linux-7.0.0-38-generic-x86_64-with-glibc2.39",
  "python": "3.12.3",
  "torch": "2.12.0+cu130",
  "cuda": "13.0",
  "gpu": "NVIDIA GeForce RTX 5070",
  "gpu_bytes": 12351569920,
  "ncm_revision": "c70274273f489abc73832b2b98584a8483b6b203",
  "task": "tinystories",
  "precision": "float32; no autocast",
  "seed": 42,
  "cpu_threads": 4,
  "batch_size": 1,
  "warmup": 3,
  "iterations": 20,
  "generator_parameters": 88605707,
  "tokenizer_parameters": 36434816,
  "combined_unique_parameters": 125040523,
  "text_vocab_size": 8192,
  "latent_vocab_sizes": [
    4096,
    4096,
    4096,
    4096,
    4096,
    4096,
    4096,
    4096,
    4096,
    4096,
    4096,
    4096
  ],
  "levels": [
    1,
    8,
    16,
    24,
    32,
    40,
    52,
    64,
    76,
    92,
    108,
    128
  ],
  "profiled_level_forwards": [
    {
      "code_positions_per_sample": 8,
      "ms": 2.831814001183375
    },
    {
      "code_positions_per_sample": 16,
      "ms": 3.1110180007090094
    },
    {
      "code_positions_per_sample": 24,
      "ms": 3.0543069988198113
    },
    {
      "code_positions_per_sample": 32,
      "ms": 3.0696579997311346
    },
    {
      "code_positions_per_sample": 40,
      "ms": 2.9543059990828624
    },
    {
      "code_positions_per_sample": 52,
      "ms": 3.0188959990482545
    },
    {
      "code_positions_per_sample": 64,
      "ms": 3.5049139987677336
    },
    {
      "code_positions_per_sample": 76,
      "ms": 3.482993999568862
    },
    {
      "code_positions_per_sample": 92,
      "ms": 3.4329919999436243
    },
    {
      "code_positions_per_sample": 108,
      "ms": 3.684766999867861
    },
    {
      "code_positions_per_sample": 128,
      "ms": 4.2333849996794015
    }
  ],
  "raw_text_positions_per_sample": 128,
  "latent_generation_ms": [
    42.0768699987093,
    39.1205150008318,
    40.294313001140836,
    38.59390599973267,
    40.47771499972441,
    39.078754000001936,
    39.04281299946888,
    38.94544200011296,
    38.894291001270176,
    41.5004209990002,
    38.661108001178945,
    38.87690100054897,
    41.83202600142977,
    39.29651700127579,
    39.45005900095566,
    40.56587699960801,
    38.53724600048736,
    38.84473000107391,
    45.26655900008336,
    39.576451999892015
  ],
  "full_generation_ms": [
    40.0040380009159,
    40.496944999176776,
    40.42606499933754,
    43.31312899921613,
    39.81946499879996,
    42.26320299858344,
    42.106259999854956,
    40.00360899954103,
    40.76892000011867,
    39.99350800040702,
    40.54692600038834,
    40.5036360007216,
    40.69874899869319,
    40.33918400091352,
    40.50443500091205,
    39.92783700050495,
    40.60782700071286,
    40.1690310009144,
    41.309977999844705,
    41.258267001467175
  ],
  "full_generation_median_ms": 40.504035500816826,
  "raw_text_positions": 2560,
  "visible_text_tokens": 2560,
  "raw_text_positions_per_second": 3140.869164769696,
  "visible_text_tokens_per_second": 3140.869164769696,
  "peak_allocated_bytes": 777734144,
  "peak_reserved_bytes": 799014912,
  "limits": "TinyStories BPE tokens, not Qwen tokens; no acceptance or quality measurement"
}
```

## Raw target pilot results

Layer means follow layer order 0 through 47. Full per-window unions remain in the trace result JSON.

```json
[
  {
    "prompt": "prose",
    "mode": "plain",
    "exit_code": 0,
    "generated": 64,
    "decode_ms": 3762.4,
    "tps": 17.01,
    "equals_plain": true,
    "speculation": [
      "speculation              64 rounds of 2, drafts accepted 0 of 63 (0.000), 1.00 tokens per round"
    ],
    "verify_window": [
      "verify window            wait for rings 8.906  pool 46.128  host 0.023  commit 0.016 ms/round; CPU experts 5.58 distinct / 5.92 routed per layer"
    ],
    "output_ids_sha256": "9214d775b01561e7cef079bd764f3d05ed94e04a4db5dd0d278b2e3a5ade0f6f"
  },
  {
    "prompt": "prose",
    "mode": "mtp",
    "exit_code": 0,
    "generated": 64,
    "decode_ms": 3269.1,
    "tps": 19.58,
    "equals_plain": true,
    "speculation": [
      "speculation              34 rounds of 4, drafts accepted 30 of 64 (0.469), 1.88 tokens per round"
    ],
    "verify_window": [
      "verify window            wait for rings 9.222  pool 80.599  host 0.007  commit 0.015 ms/round; CPU experts 9.70 distinct / 13.23 routed per layer"
    ],
    "output_ids_sha256": "9214d775b01561e7cef079bd764f3d05ed94e04a4db5dd0d278b2e3a5ade0f6f"
  },
  {
    "prompt": "prose",
    "mode": "default",
    "exit_code": 0,
    "generated": 64,
    "decode_ms": 3176.9,
    "tps": 20.15,
    "equals_plain": true,
    "speculation": [
      "speculation              35 rounds of 6, drafts accepted 29 of 62 (0.468), 1.83 tokens per round"
    ],
    "verify_window": [
      "verify window            wait for rings 9.228  pool 75.488  host 0.007  commit 0.012 ms/round; CPU experts 9.30 distinct / 12.52 routed per layer"
    ],
    "output_ids_sha256": "9214d775b01561e7cef079bd764f3d05ed94e04a4db5dd0d278b2e3a5ade0f6f"
  },
  {
    "prompt": "prose",
    "mode": "oracle4",
    "exit_code": 0,
    "generated": 64,
    "decode_ms": 2074.5,
    "tps": 30.85,
    "equals_plain": true,
    "speculation": [
      "speculation              17 rounds of 4, drafts accepted 48 of 48 (1.000), 3.82 tokens per round"
    ],
    "verify_window": [
      "verify window            wait for rings 10.042  pool 107.346  host 0.047  commit 0.015 ms/round; CPU experts 13.24 distinct / 19.85 routed per layer"
    ],
    "output_ids_sha256": "9214d775b01561e7cef079bd764f3d05ed94e04a4db5dd0d278b2e3a5ade0f6f",
    "complete_trace_windows": 16,
    "mean_unique_experts_per_layer": 25.438802083333332,
    "layer_means_0_to_47": [
      37.375,
      34.75,
      33.25,
      23.9375,
      27.375,
      30.375,
      29.0625,
      31.875,
      23.1875,
      22.125,
      24.0625,
      23.1875,
      30.6875,
      26,
      25.25,
      16.4375,
      18.375,
      24.1875,
      31.1875,
      22.625,
      25.5625,
      28.3125,
      35.125,
      20.6875,
      23.8125,
      21.375,
      24.4375,
      23.4375,
      30.5,
      25.8125,
      24.9375,
      15.75,
      15.3125,
      22.875,
      29.75,
      21.75,
      22.6875,
      27.5,
      35.0625,
      16.1875,
      17.4375,
      21.625,
      30.5,
      22.375,
      23.8125,
      23.375,
      24.4375,
      31.3125
    ]
  },
  {
    "prompt": "prose",
    "mode": "oracle8",
    "exit_code": 0,
    "generated": 64,
    "decode_ms": 1676.3,
    "tps": 38.18,
    "equals_plain": true,
    "speculation": [
      "speculation              9 rounds of 8, drafts accepted 56 of 56 (1.000), 7.22 tokens per round"
    ],
    "verify_window": [
      "verify window            wait for rings 17.069  pool 162.941  host 0.073  commit 0.015 ms/round; CPU experts 22.21 distinct / 44.15 routed per layer"
    ],
    "output_ids_sha256": "9214d775b01561e7cef079bd764f3d05ed94e04a4db5dd0d278b2e3a5ade0f6f",
    "complete_trace_windows": 8,
    "mean_unique_experts_per_layer": 40.817708333333336,
    "layer_means_0_to_47": [
      69.375,
      61.625,
      58.75,
      36.75,
      45.625,
      51.125,
      51.75,
      55.375,
      35.375,
      32.25,
      38.125,
      34.875,
      53.875,
      43.25,
      39.875,
      22.625,
      25.125,
      38.125,
      55.25,
      34.75,
      40.25,
      47.25,
      61.625,
      29.875,
      36.625,
      30.875,
      38.625,
      33.75,
      53,
      41.75,
      38.375,
      19.75,
      18.375,
      36,
      51.25,
      33.125,
      35.25,
      45,
      60,
      21.875,
      23.875,
      32.75,
      50.25,
      32.25,
      37.25,
      36.25,
      38.375,
      51.75
    ]
  },
  {
    "prompt": "code",
    "mode": "plain",
    "exit_code": 0,
    "generated": 64,
    "decode_ms": 6476.0,
    "tps": 9.88,
    "equals_plain": true,
    "speculation": [
      "speculation              64 rounds of 2, drafts accepted 0 of 63 (0.000), 1.00 tokens per round"
    ],
    "verify_window": [
      "verify window            wait for rings 8.848  pool 88.349  host 0.018  commit 0.034 ms/round; CPU experts 7.25 distinct / 7.54 routed per layer"
    ],
    "output_ids_sha256": "dd7aefb01280c8042a7ef8a7716451767abfc9536bfa47e10427fd3a183d8b57"
  },
  {
    "prompt": "code",
    "mode": "mtp",
    "exit_code": 0,
    "generated": 64,
    "decode_ms": 3366.9,
    "tps": 19.01,
    "equals_plain": true,
    "speculation": [
      "speculation              21 rounds of 4, drafts accepted 45 of 60 (0.750), 3.14 tokens per round"
    ],
    "verify_window": [
      "verify window            wait for rings 10.154  pool 143.676  host 0.009  commit 0.016 ms/round; CPU experts 19.05 distinct / 25.74 routed per layer"
    ],
    "output_ids_sha256": "dd7aefb01280c8042a7ef8a7716451767abfc9536bfa47e10427fd3a183d8b57"
  },
  {
    "prompt": "code",
    "mode": "default",
    "exit_code": 0,
    "generated": 64,
    "decode_ms": 3405.7,
    "tps": 18.79,
    "equals_plain": true,
    "speculation": [
      "speculation              20 rounds of 6, drafts accepted 46 of 63 (0.730), 3.30 tokens per round"
    ],
    "verify_window": [
      "verify window            wait for rings 10.761  pool 152.120  host 0.010  commit 0.017 ms/round; CPU experts 20.44 distinct / 28.18 routed per layer"
    ],
    "output_ids_sha256": "dd7aefb01280c8042a7ef8a7716451767abfc9536bfa47e10427fd3a183d8b57"
  },
  {
    "prompt": "code",
    "mode": "oracle4",
    "exit_code": 0,
    "generated": 64,
    "decode_ms": 2656.2,
    "tps": 24.09,
    "equals_plain": true,
    "speculation": [
      "speculation              17 rounds of 4, drafts accepted 48 of 48 (1.000), 3.82 tokens per round"
    ],
    "verify_window": [
      "verify window            wait for rings 10.200  pool 141.405  host 0.021  commit 0.016 ms/round; CPU experts 19.42 distinct / 26.19 routed per layer"
    ],
    "output_ids_sha256": "dd7aefb01280c8042a7ef8a7716451767abfc9536bfa47e10427fd3a183d8b57",
    "complete_trace_windows": 16,
    "mean_unique_experts_per_layer": 27.709635416666668,
    "layer_means_0_to_47": [
      34.25,
      30.9375,
      30.5625,
      27.4375,
      28.5625,
      28.375,
      28.9375,
      30.5,
      25.625,
      29.3125,
      28.0625,
      27.8125,
      29.5625,
      29.625,
      30.6875,
      29.0625,
      30.125,
      29.4375,
      29.75,
      30.125,
      28.125,
      28.25,
      29.8125,
      30.3125,
      27.5,
      29,
      29.9375,
      29.0625,
      29.125,
      29.1875,
      29.6875,
      29,
      26.9375,
      27.5,
      25.5625,
      24.125,
      23.0625,
      23.75,
      28.5625,
      25,
      24.3125,
      24.6875,
      22.6875,
      24.9375,
      22.1875,
      21.1875,
      22.75,
      25.0625
    ]
  },
  {
    "prompt": "code",
    "mode": "oracle8",
    "exit_code": 0,
    "generated": 64,
    "decode_ms": 2092.4,
    "tps": 30.59,
    "equals_plain": true,
    "speculation": [
      "speculation              9 rounds of 8, drafts accepted 56 of 56 (1.000), 7.22 tokens per round"
    ],
    "verify_window": [
      "verify window            wait for rings 16.933  pool 209.351  host 0.043  commit 0.016 ms/round; CPU experts 31.22 distinct / 50.92 routed per layer"
    ],
    "output_ids_sha256": "dd7aefb01280c8042a7ef8a7716451767abfc9536bfa47e10427fd3a183d8b57",
    "complete_trace_windows": 8,
    "mean_unique_experts_per_layer": 45.005208333333336,
    "layer_means_0_to_47": [
      58.75,
      52.625,
      49.625,
      42.5,
      46.25,
      46.5,
      47.25,
      51.375,
      40.625,
      47.375,
      45.5,
      46.125,
      49.25,
      49.125,
      50,
      47.25,
      49,
      49.125,
      51.375,
      50.25,
      45.125,
      46.75,
      50,
      49.75,
      46,
      47.125,
      51.5,
      48,
      46.75,
      48.875,
      49.25,
      45.875,
      42.25,
      45.5,
      40.625,
      35.875,
      34.5,
      36.75,
      46.625,
      40,
      38.25,
      38.75,
      36.25,
      40.5,
      33.75,
      31.375,
      35.375,
      39
    ]
  }
]
```
