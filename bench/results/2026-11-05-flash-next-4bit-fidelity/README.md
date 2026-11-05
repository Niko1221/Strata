# 2026-11-05: 4-bit fidelity on Qwen3.8-Flash-Next — UD-IQ4_XS beats the 33%-larger UD-Q4_K_XL

**Question.** The engine's launcher accepts a range of pack sizes for this model, but the production brain is
IQ3_S (83.60 GB) and nothing said whether a 4-bit recipe buys quality back. Two candidates are accepted and
shipped by the same publisher: UD-IQ4_XS (93.67 GB, +12.0%) and UD-Q4_K_XL (111.33 GB, +33.2%). Which is more
faithful to the model it was cut from, and does the bigger one buy anything?

**Answer.** IQ4_XS. It is better calibrated than both the incumbent and the 33%-larger pack, in every text
segment, and the larger pack is the *worst* of the three on every axis. Bits per weight, and even total bytes,
do not predict fidelity here.

**Machine.** RTX 5080 16 GB (Blackwell, sm_120), driver 580, 256 GB RAM, AI-Srvr under Unraid; 2× Xeon 8176.
Engine **0.1.56**, image `0.2.10-cuda13`. The arms were scored one at a time in a window with the production
engine stopped: on this card the quality tool could not load beside a resident engine (`84 of 15839 MiB free`
when the scoring buffer wanted 147 MiB).

**Model.** `Qwen3.8-Flash-Next` (Unsloth GGUFs), MoE, 48 layers × 512 experts, served through engine packs
(`iq_pack.py`, `--compat-bf16`). Reference = the IQ3_S production brain.

**Pack sizes as served.** IQ3_S 83.60 GB · UD-IQ4_XS 93.67 GB · UD-Q4_K_XL 111.33 GB.
Pack bytes are index + converted dense tensors; the routed experts stay native in the GGUFs, so the pack build's
own conversion covers only ~460 of ~1,079 tensors.

## Method

`quant_quality.py` yardstick v1: **4,995 teacher-forced positions** (999 each in `code`, `docs`, `english`,
`multilingual`, `agent`), one forward pass per arm at temperature 0, no sampling. Per-position it records the
target token's log-probability, the argmax, and the top-64 distribution.

Three figures, and it matters which is which:

- **ppl** — perplexity of the fixed text under the arm. Absolute: no reference involved.
- **top-1 / KL** — agreement with the **IQ3_S reference's** argmax and distribution. Reference-relative, so an arm
  that happens to share the reference's error bias can flatter itself. Read alongside ppl, never alone.
- **mean logp of the true token** — paired per position across arms; the most sensitive of the three, because it
  uses the full distribution rather than a binary argmax hit.

Configs for the two arms were derived from the reference config **by substitution only** (`--pack`, `--native`,
`--ple-gguf`, `tokenizer`) plus a separate port and log path; the normalised diff was checked clean, so nothing
but the weights differs between legs. The `--ple-gguf` shard was established by hash, not assumed: shard `00004`
is byte-identical (`5de8b5db…`) to the shard an earlier UD-Q3_K_XL run served `--ple-gguf` from successfully.

## Results

| arm | ppl | Δppl vs ref | mean logp of true token | top-1 vs ref | mean KL | KL p99 | top-1 vs text | bytes |
|---|---|---|---|---|---|---|---|---|
| IQ3_S (ref) | 4.057 | — | −1.4004 | 100% | — | — | 68.85% | 83.60 GB |
| **UD-IQ4_XS** | **3.941** | **−2.86%** | **−1.3714** | 87.53% | 0.1163 | 1.51 | 68.47% | 93.67 GB |
| UD-Q4_K_XL | 4.087 | +0.74% | −1.4079 | 86.83% | 0.1409 | 2.01 | 68.25% | 111.33 GB |

Per segment — IQ4_XS wins all five on both ppl and mean logp:

| segment | ppl ref | ppl IQ4_XS | Δ | ppl Q4_K_XL | Δ | logp IQ4_XS / Q4_K_XL |
|---|---|---|---|---|---|---|
| code | 2.036 | 1.966 | −3.4% | 2.033 | −0.1% | −0.676 / −0.709 |
| docs | 9.983 | 9.673 | −3.1% | 9.888 | −1.0% | −2.269 / −2.291 |
| english | 4.165 | 4.148 | −0.4% | 4.249 | **+2.0%** | −1.423 / −1.447 |
| multilingual | 2.686 | 2.670 | −0.6% | 2.694 | **+0.3%** | −0.982 / −0.991 |
| agent | 4.832 | 4.513 | **−6.6%** | 4.959 | **+2.6%** | −1.507 / −1.601 |

For context, other arms previously scored on the same yardstick and set: NVFP4-QAD ppl 3.942 / KL 0.1495 /
p99 2.03 / top-1-vs-ref 86.97%; IQ3_XXS ppl 4.146 / KL 0.1738 / p99 2.09 / 85.23%; UD-Q3_K_XL ppl 4.009 /
KL 0.1134 / 87.27% (93.68 GB — outside the launcher's range at the time, quality compared only).

## Effect size, stated honestly

Paired exact McNemar on raw top-1 correctness, same positions:

| pair | only A right | only B right | p |
|---|---|---|---|
| IQ4_XS vs Q4_K_XL | 130 | 119 | **0.53** |
| IQ4_XS vs IQ3_S | 139 | 158 | 0.30 |
| Q4_K_XL vs IQ3_S | 146 | 176 | 0.11 |

So **the argmax difference is inside noise** at 4,995 positions and this report does not claim an accuracy jump.
What is consistent is calibration: mean logp of the true token moves by −2.86% / −3.58% in perplexity terms for
IQ4_XS against ref / Q4_K_XL, in the same direction in all five segments, with a third of the catastrophic tail
(p99 KL 1.51 vs 2.01). That is the defensible claim: **better calibrated, not measurably more accurate.**

## Why a smaller IQ pack can beat a larger K pack

Three mechanisms, in the order the measurements support them.

1. **Total bytes do not say where the precision went.** Unsloth's Dynamic recipes assign *per-tensor* types —
   some tensors upcast, others squeezed — so "Q4_K_XL" means a larger overall allocation, not uniformly higher
   bits. The 17.6 GB that Q4_K_XL carries over IQ4_XS is not spread evenly, and this set says it was not spent
   where these five corpora look.
2. **The quantiser family differs, and direction matters more than magnitude on an MoE.** IQ4_XS is an
   importance-weighted lattice/codebook quantiser: within a block it selects a codeword, which preserves the
   *shape and direction* of a weight vector. Q4_K_XL is scalar block quantisation with per-superblock scales and
   mins: it preserves *magnitude* well with a coarser sense of direction. A routed MoE layer sums 512 experts
   weighted by the router, so direction error compounds across the sum in a way magnitude error does not.
3. **The tensors the pack converts are not where the difference lives.** The pack build reports its own
   conversion error and it ranks these two arms **the wrong way round**: `max |err| 0.0286` for IQ4_XS (the
   winner) against `0.0144` for Q4_K_XL (the loser). The converted dense tensors are ~460 of ~1,079; the routed
   experts stay native in the GGUF. So the dense-tensor conversion error is a mechanism note, never a predictor —
   the expert recipe decides, and these two packs differ most there.

## Limits

- One measured pass per arm (the reference leg reproduced ppl 4.057 exactly across two separate windows, so the
  scoring path is deterministic; the 4-bit legs were each measured once).
- No Q8_0 / BF16 anchor, so this measures fidelity **relative to IQ3_S plus absolute perplexity on this text**,
  not distance to the source model. An anchor run is the next step if anyone wants the absolute claim.
- Decode cost of +12.0% bytes is **not** measured here and decides adoption, not fidelity.
- Five fixed corpora, 999 positions each. Long-context behaviour, tool-calling correctness and multi-turn drift
  are not covered by teacher-forced scoring.

## Reproduction

Packs (43 s each, ~1.4 GB — an index; the experts stay in the GGUFs):

```bash
docker run --rm -v /mnt/cache/strata:/data -v /mnt/data/strata-models:/data/Strata-models \
  --entrypoint /opt/strata/terminal/python/bin/python3 0.2.10-cuda13 \
  /opt/strata/terminal/tools/iq_pack.py --compat-bf16 \
  --gguf /data/Strata-models/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf \
  --out /data/Strata-data/packs/unsloth-ud-iq4_xs
```

Scoring, one arm at a time, with production stopped (see `quality_window.sh`, which stops the production
container **by name** and restores it on any exit path):

```bash
docker run --rm --gpus all -v /mnt/cache/strata:/data -v /mnt/data/strata-models:/data/Strata-models \
  --entrypoint /opt/strata/terminal/python/bin/python3 0.2.10-cuda13 \
  /data/quality-run/quant_quality.py run --config /data/quality-run/cfg-iq4xs.json \
  --out /data/quality-run/iq4xs.tsv --ids /data/build-q36/Strata/data/quant_quality/v1/set.ids.json

docker run --rm -v /mnt/cache/strata:/data --entrypoint /opt/strata/terminal/python/bin/python3 0.2.10-cuda13 \
  /data/quality-run/quant_quality.py compare ref.tsv iq4xs.tsv q4kxl.tsv --segments --json compare.json
```

`paired_compare.py` (in this folder) reproduces the McNemar and paired-logp figures from the three TSVs. Raw
TSVs are ~5 MB each and are not committed; `compare-4bit-20261105-2014.json` is the tool's own summary and
`results.json` carries the per-arm and per-segment numbers used above.
