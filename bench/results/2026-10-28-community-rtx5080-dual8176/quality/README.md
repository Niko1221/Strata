# Quant quality: NVFP4-QAD and IQ3_XXS against the IQ3_S reference

A second measurement in this folder, unrelated to the speed runs: how much each quant's next-token distribution moves
away from the IQ3_S reference on a frozen text set. Tool: `tools/quant_quality.py` (set v1). Engine `0.1.48`, image
`ghcr.io/niko1221/strata:0.2.3-cuda13`, one RTX 5080.

## Method

Five segments of 999 positions each (`code`, `docs`, `english`, `multilingual`, `agent`), 4,995 positions in total. Every
position is scored **teacher-forced**: the model sees the true prefix and only its next-token distribution is recorded,
via `--short-read` (route all positions through the decode windows) and `STRATA_LOGPOS` (per-position logprobs on
stderr). Top-1 agreement is the share of positions where the arm's argmax equals the reference's; KL is the mean
KL(reference ‖ arm) in nats; `ppl` is the perplexity of the scored text under that arm.

Reference: IQ3_S, the same 4,995 positions, run on the same engine minutes apart. Sampling is the engine's own
(temperature 1.0), so these perplexities are internally consistent but **not** comparable with a GPU-side
`llama-perplexity` number: the comparison that matters here is between arms.

Files: `cfg-iq3s.json`, `cfg-nvfp4.json`, `cfg-xxs.json` (the harness config for each run), `compare.json` and
`compare-xxs.json` (the tool's own comparison output), `*.tsv.seg.json` (per-segment aggregates). The per-position TSVs
(~5.3 MB each) are not committed.

## Results

| Arm | top-1 agreement | mean KL | ppl | ppl (reference) |
| --- | --- | --- | --- | --- |
| NVFP4-QAD | 86.97% | 0.1495 | 3.942 | 4.103 |
| IQ3_XXS | 85.23% | 0.1737 | 4.146 | 4.103 |

Per segment (top-1 agreement / KL):

| Segment | NVFP4-QAD | IQ3_XXS |
| --- | --- | --- |
| code | 93.29% / 0.0846 | 91.39% / 0.1146 |
| docs | 78.98% / 0.2108 | 75.48% / 0.2442 |
| english | 86.19% / 0.0970 | 83.78% / 0.1292 |
| multilingual | 91.19% / 0.0611 | 91.99% / 0.0723 |
| agent | 85.19% / 0.2943 | 83.48% / 0.3085 |

Both are directionally sane and both cost something: the harder the text, the larger the drift. `docs` is the worst
segment for both (technical prose with many near-tied continuations), and `agent` has the highest KL despite a decent
agreement rate, which is the signature of a small set of confidently wrong positions rather than uniform noise.

## The Qwen3.6-35B arm could not be scored, and that is an engine limitation

The `qwen36-35b` speed arm has no quality number here. The Qwen3.6 pack is served by its own engine lineage
(`strata-q36`), which does not implement the two features this measurement needs:

- `--short-read` and `STRATA_LOGPOS` are absent. The engine does not fail; it prints
  `ignoring --short-read 1064 (not an option of this engine)` and takes the batched prompt path, so the harness reports
  `code: 0 positions … (the prompt took the batched path?)` and writes no TSV. The same applies to `--expert-cache`,
  `--adapt-every`, `--pcie-frac`, `--suffix-draft` and `--prompt-cache`.
- `serve/server.py` in 0.2.3 has no `logprobs` support, so there is no OpenAI-compatible prompt-logprob route to fall
  back on.
- `--dump-full-logits` works and is clean — float32 rows of 248,320 logits, 993,280 bytes each (verified:
  `63,569,920 = 64 x 993,280`) — but it emits one row per **generated** token, i.e. the model's own continuation rather
  than the text to be scored. `--force` is the intended teacher-forcing hook and is undocumented; a probe with
  `--tokens 9707 --force 11,1879,314` generated free tokens (`13 321 25 279`) instead of the forced ids.

Config kept as evidence: `cfg-qwen36.json`, and the failing probe's command is in this folder's history.

**When a scoring path exists, the comparison is free:** the Qwen3.6-35B pack and the Qwen3.8 IQ3_S pack have
byte-identical tokenizers (`vocab.json` `4ba64f033…`, `merges.txt` `4ba698529…`, 248,320 tokens each), so the 35B would
score the same positions on the same tokens as `iq3s.tsv` and its perplexity would be directly comparable with the
reference's 4.103.

## Limits

- Fidelity to IQ3_S, not to the unquantized model: no full-precision reference was run.
- 4,995 positions is enough to rank these two arms and not enough to resolve small differences; a repeat run would move
  the third decimal, not the ranking.
- One engine, one GPU, one machine, one afternoon.
- No task-based evaluation: KL says how far the token distribution moved, not whether an agent or a code task succeeds.
