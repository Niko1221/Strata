# Domain-informed expert profiles: a real-routing pilot

**Author: Francisco Angulo de Lafuente** · [ORCID](https://orcid.org/0009-0001-1634-7063)

The new [profile tool](../../../tools/evaluate_domain_profiles.py) measures a useful
domain signal in Strata's expert routing and writes profiles that the existing
`--expert-profile` option can load. In this 45-request pilot, a profile learned from
other requests in the same coarse domain covers more of the held-out routing than
a profile learned from the whole training mixture. Every evaluated fold is excluded
from both training sets.

| Cache slots | Global profile | Domain profile | Gain | Domain with the global profile's slots per layer | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,582 | 17.41% | 22.61% | +5.20 percentage points | 22.42% | +5.01 points |
| 3,215 | 30.00% | 37.77% | +7.77 points | 37.57% | +7.57 points |
| 7,041 | 51.88% | 61.00% | +9.11 points | 60.81% | +8.93 points |

These are **static coverage measurements on fixed real routing**, with each request
weighted equally. The three slot counts use the same traces. They are cache-budget
analyses, not three hardware runs or measurements of tokens per second.

## What this contributes

The tool validates request boundaries and separates complete decode windows from
the preceding prompt windows. It compares domain information with a learned global
control, supplies a control with unchanged slots per layer, and exports a reusable
profile bank. It leaves the engine and its router unchanged.
Two generated profiles were also loaded by the release engine in a small
[live pilot](live-pilot/README.md); all nine coding answers passed the fixed
correctness checks. Its full records and frozen comparison gates are retained.

The motivation comes from the author's earlier modular system with a resident
director and specialist models loaded on demand. The public
[Director model](https://huggingface.co/Agnuxo/Qwen2-1.5B-Instruct_MOE_Director_16bit)
and [biology specialist](https://huggingface.co/Agnuxo/Qwen2-1.5B-Instruct_MOE_BIOLOGY_assistant_16bit)
date to August 2024. Here, the placement unit is an expert within Strata's model.
The model router still selects the experts and their weights.

Attribution covers this Strata tool, the collected measurements and their evaluation.
The broader field includes
[Fiddler](https://arxiv.org/abs/2402.07033),
[MoE-Infinity](https://arxiv.org/abs/2401.14361),
[ProMoE](https://arxiv.org/abs/2410.22134) and
[MoE-CORE](https://arxiv.org/abs/2610.01950), whose domain-informed initialization
is particularly relevant. Their performance numbers are not used as ours.

## Measurements and controls

The release engine was 0.1.59, app 0.2.13, at source
`37d5c9e84514150ee74d69f4f0f0466ba4b91b51`. The analysis starts from
`f33c8c30588ff00e24b6b63bad0bf6a14da19081`; the intervening changes are sponsor
README entries. One RTX 3090 on Debian 13 ran Qwen3.8-Flash-Next GSQ RCO IQ3_XXS
with 1,582 expert-cache slots, 10 GiB resident RAM, speculation 4, temperature 0,
thinking disabled and at most 250 generated tokens per request. The host exposed
31.3 GiB RAM and the measured PCIe transfer rate was 0.3 GB/s.
The recorded model revision is `ed59f92082b1e93c0e96d60a8b11aab089b52f09` from
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09).
The original download-metadata check is frozen in
[model-source.json](model-source.json). A subsequent full SHA-256 read of both
source GGUF files matched the pinned LFS digests: 75,839,998,528 bytes verified.
The [payload receipt and exact verifier](payload-verification/README.md) preserve
that check. The existing release binary's SHA-256 is recorded in the provenance.

The three bounded runs produced 10,170 generated tokens and **6,363,360 decode expert
entries**. The routing dumps also contain 759,840 prompt entries. All 45 decode
suffixes match whole-window boundaries and the integer decode counts in the
corresponding [engine logs](data/). The manifest records those exact counts.
The initial 37-request pilot is preserved in commit `7e316925`; eight remaining
requests completed the original corpus, using the same evaluator, split rule and
budgets. No completed request was dropped.
The cutoff uses `lookups + offloaded`, rather than generated-token counts: speculation
also routes proposed positions that are not emitted.

The engine prints `hits / lookups`, where offloaded entries are excluded from lookups.
Trace coverage uses `hits / all routed entries`. These are different denominators.
Prompt routing is excluded from the static decode-profile score. The tool fails on
truncated records, overlapping request ranges, incomplete windows and invalid IDs.

The split uses five folds, seed 20261109, with stable request IDs shuffled within
each coarse group. Training normalizes each request to one distribution, as
`make_profile_mix.py` does. Both the global and the domain profile exclude every
request in the evaluated fold. The constrained domain control keeps the global
profile's number of cache slots in every layer, matching the layer-local scope of
the existing adaptive swap rule.

## Reproduce the analysis

From the repository root, using Python with NumPy:

```sh
python -m unittest tools.test_evaluate_domain_profiles tools.test_make_profile tools.test_make_profile_mix
python tools/evaluate_domain_profiles.py bench/results/2026-11-09-domain-expert-profiles/manifest.json --slots 1582 --out /tmp/domain-results.json
python tools/evaluate_domain_profiles.py bench/results/2026-11-09-domain-expert-profiles/manifest.json --slots 3215 --out /tmp/domain-results-3215.json
python tools/evaluate_domain_profiles.py bench/results/2026-11-09-domain-expert-profiles/manifest.json --slots 7041 --out /tmp/domain-results-7041.json
```

Replace `/tmp/` with a writable directory on Windows. Compare with the checked-in
`results-*.json`. To write profiles for **future requests**, add
`--profiles-dir NEW_DIRECTORY`. Its `index.json` identifies the global profile and
each domain's file, training IDs, author and SHA-256. Use the selected `.bin` with
the existing `--expert-profile` option. These final profiles use all input requests;
the reported scores instead use the held-out fold profiles.

[manifest.json](manifest.json) contains corpus IDs and exact uncompressed byte
ranges. [provenance.json](provenance.json) records the source revision, configuration,
collection status and compressed/uncompressed SHA-256 values. Raw routing records
are gzip-compressed in `data/`; no model weights are included. The underlying public
requests are indexed in `bench/profile_corpus.json`. The configuration uses `<LAB>`
in place of local paths. The upstream MIT license applies to the tool, traces and
corpus; model weights are obtained separately under their own terms.

## Validation

Passed: 22 unit tests on Python 3.13.7 and NumPy 2.5.4, including the existing
profile writer/mix checks; the three analysis commands above; profile write/read
round trips. Independent read-only review reproduced the decode boundaries and
checked the domain signal and layer quota control.

The same 22 focused checks also passed inside the official extended tools suite,
which passed 1,675 checks locally on Windows. The [per-case record and base
comparison](validation/README.md) retain its failures and skips with exact scope.

## Observations

- This is an exploratory profiling corpus: all 45 planned requests, with code 12,
  STEM 11, languages 6 and other 16. It remains a small corpus rather than a broad
  workload sample.
- At 1,582 slots, gains vary by group: code +12.40 points, other +4.70, STEM +2.29
  and languages −2.56. A mixed six-language group is not a uniformly useful profile.
  Per-request values, training IDs and group labels remain in the result files.
- Static coverage establishes a placement signal. It does not establish a gain
  over the running adaptive cache, lower copy traffic, faster inference, a measured
  classifier, or equal model outputs. GPU and CPU arithmetic can round differently
  even when expert IDs and weights stay unchanged.
- The 0.3 GB/s PCIe link and small GPU cache do not represent an ordinary 12 GB card.
  The subsequent source GGUF hash check does not re-derive or attest the prepared
  native pack. This pilot is not a standard or approved benchmark under the draft
  protocol.
- No engine source is changed, so CUDA/HIP/SYCL builds are outside this tooling
  change. A runtime feature needs a separate, agreed scope and
  actual same-hardware tests, including copy cost and output/quality controls.
- The extended local tools step failed six cases and skipped 78. All six failures
  reproduced on the unmodified upstream base in this Windows environment, with
  matched messages after replacing checkout paths. The focused 22 checks passed.
  [Full records and limits](validation/README.md) are retained; these local results
  do not establish that GitHub's full CI matrix passes.
- A separate three-second collection startup failed before model loading because
  a relative output path resolved against the server's directory. The collector
  was changed to resolve an absolute path, then all eight remaining requests
  completed. The [failure log](data/startup-failure.txt) is retained.
- The live profile pilot passed all nine fixed correctness checks, but the code
  profile changed the clamp answer from 21 to 26 tokens. Its frozen output-parity gate
  fails; no live speedup is established. [All runs and outputs](live-pilot/README.md)
  are included. Static coverage and this runtime outcome are separate results.
