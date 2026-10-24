# Which model for a dual 12 GB rig without AVX2? A measured comparison (2026-10-22)

A same-rig, same-configuration comparison of the model options Strata offers that fit this
box (2x RTX 3060 12 GB, layer split, Xeon E5-2697 v2 no AVX2, 128 GB RAM). All runs with
`tools/strata_bench.py --url` against a localhost-bound server, three runs per size, medians;
quality from a 20-question tricky set (counting traps, negation, format compliance, code bug,
reverse word, logic) plus a censoring probe. Engine 0.1.47 source build.

## Speed and quality

| Metric | Swift IQ2_XS (SC117 abl.) | Swift IQ3_XXS (SC117 abl.) | Swift IQ3_XXS (standard) | Qwen3.6-35B (UD-Q4_K_XL) |
| --- | ---: | ---: | ---: | ---: |
| short chat decode tok/s | 55.7 | 44.9 | 45.5 | 28.6 |
| short chat TTFT (s) | 0.65 | 0.71 | 0.71 | 0.45 |
| 4K prompt tok/s | 537.6 | 522.9 | 520.2 | 944.9 |
| 16K prompt tok/s | 809.6 | 781.9 | 809.2 | 907.1 |
| agent decode tok/s | 56.6 | 47.0 | 49.2 | 23.5 |
| conversation cache hit % | 69.6 | 69.5 | 69.5 | 0.0 |
| tricky quality (18 objective) | 17/18 | **18/18** | **18/18** | 16/18 |

## Takeaways

- **More bits = slower on this hardware.** The reported "Q3 faster than Q2" did not reproduce
  here; decode is ~15-19 % lower for IQ3_XXS over IQ2_XS. It only *feels* faster on easy
  questions because it thinks shorter.
- **IQ3_XXS is the quality tier:** 18/18 on the tricky set (IQ2_XS slips one counting error).
  The **abliterated** (SC117) and standard variants behave identically in speed and quality;
  the abliterated one is the flagged-uncensored file, which is why we run it in production at a
  ~15 % decode cost in exchange for zero counting errors.
- **Qwen3.6-35B-A3B is not an agent model here:** it reads prompts much faster (945 tok/s at 4K)
  and starts fast (TTFT 0.45 s), but shows **0 % conversation-cache reuse** (every resend
  re-reads ~31k tokens) and half the decode - fine for a quick single answer or a big-document
  read, not for resending conversations. Setup ran it on GPU 0 with experts mostly on the CPU.
- **Counting is unstable across all of them** (same prompt, temp 0: 5 then 6 on an e-count), and
  with thinking on a low `max_tokens` can leave `content` empty (only `reasoning_content`) -
  worth knowing when setting output caps, not a server bug.

## Uncensored route

Swift mainline is a "thinks shorter" fine-tune, not itself an uncensored build. Uncensored =
the SC117 `-GSQ-RCO-abliterated-` files (IQ2_XS, IQ3_XXS, IQ3_S available), or the OrcaRouter
uncensored pack (manual `iq_pack --compat-bf16` workflow). We run SC117-abliterated-IQ3_XXS in
production; the censoring probe (harmlessly-framed topics) answered openly on every model here.