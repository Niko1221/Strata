# Swift 1.5 as the ./run.sh default (2026-10-05)

Executes `plans/run-default-swift-15-iq3xxs-2026-10.md` on the gfx1101 host (RX 7700 XT,
12 272 MiB; Strata's share capped at 10 240 MiB - unchanged, the ~2 GiB remainder stays with the
desktop). Card: this machine only; every number below is attributed to it.

## What ships

`./run.sh` with no arguments starts **swift-1.5-iq3_xxs** (Swift 1.5, UkisAI's fine-tune,
Swift Open License 1.0 - named in the startup log). `--release qwen|swift|coder` names a
release; default quants per release: swift IQ3_XXS, qwen IQ3_S, coder IQ1_M. Packs are
release-tagged (`packs/swift-iq3_xxs` beside `packs/iq3_xxs`) and the bootstrap refuses a pack
whose `experts.bin.src.json` names another release's shards.

## Live gates (runner scripts here; server on 127.0.0.1:19931)

| gate | result | threshold |
| --- | --- | --- |
| `--release swift --model IQ3_XXS`, zero `-e` (`swift-flag-*`) | fresh prefill 255.0 tok/s, decode 32.9, follow decode 30.2 | >= 235 / >= 29 |
| TTFT @1k / @4k fresh | 4.51 / 15.67 s | <= 17 s @4k |
| VRAM share, arm (`swift-flag-vram-share.json`) | peak 8,696.2 MiB, PASS | every sample <= 10,240 |
| VRAM share, whole gate window (`swift-flag-gate-vram-share.json`) | peak 8,697.5 MiB, 19,871 samples, PASS | same |
| 32 768 / 65 536 / **130 944** fresh | 259.4 / 241.8 / 229.7 tok/s | >= 200 @130 944 |
| follow-ups at those sizes | reuse 32 761 / 65 529 / 130 937, TTFT 0.68-1.07 s | no truncation |
| cancel mid-stream -> recovery (`*-gate-cancel.json`) | next request `finish_reason=stop` | stop |
| coding smoke, arm and gate | PASS 205/205 both | 205/205 |
| Qwen regression (`--release qwen --model IQ3_S`, `qwen-regression-*`) | boots the tuned shapes (695-slot cache, 534 borrowed), smoke 205/205 | byte-identical -e line (contract) |
| no-flag default after the flip (`final-default-*`) | READY `swift-1.5-iq3_xxs`, smoke 205/205; docker line byte-identical to the gated one | - |
| manual `docker run`, zero `-e` | READY `swift-1.5-iq3_xxs` from the aligned image ENV | - |
| `./run.sh --check-only` | model ok, **pack provenance ok**, pack ok, MTP ok | throwaway inspector, no port |
| swift launch pointed at the Qwen pack (`-e STRATA_PACK_DIR=/work/packs/iq3_s`) | refused before the engine: "was built from another release's shards", naming `Qwen3.8-...-IQ3_S-00001-of-00002.gguf` | the corruption door stays shut live, not just in tests |
| `./run.sh --offline --model IQ2_XS` | dies with `hf download ukisai/Swift-1.5-... --include '*IQ2_XS*.gguf'` | no server, no bogus download |

`default-swift-explicit.txt` is the pinned docker line; `swift-flag-gate-engine.log` carries the
engine's own accounting ("expert cache 1006 slots", "the prompt path borrows 619 CUDA0 cache
slots" - chunk stayed 2,048 at every size).

## Preserved and broken behavior

Preserved byte-for-byte: both Qwen pins (contract fixtures) and the old manual `-e` launch style
(equivalence test compares the last-wins -e dicts). `STRATA_HF_REPO` on the host now actually
reaches the container (it never did - the old path only worked by re-passing it as `-e`).

Broken by design, disclosed in the launcher header: `$STRATA_MODEL`/`--model` alone now name a
quant of the **default** (swift) release - Qwen quants pin as `--release qwen --model IQ3_S` -
and the container coder's advertised id gained its release tag
(`qwen3.8-flash-next-coder-<quant>`, matching the host installer).

## Defect found while gating

`docker/hfmodel.py` ended its known-release file search with a name-blind catch-all glob, so a
quant the release does not ship resolved a **sibling** file: `--model IQ2_XS` against Swift
returned the Swift IQ3_XXS shard and reported `CACHED=1` - the server would have advertised
`swift-1.5-iq2_xs` while loading IQ3_XXS. Fixed (a table-known release now matches only file names
containing the requested quant; the catch-all is reserved for un-taught repos), covered by
`test_hfmodel_release.py::test_a_quant_absent_from_a_release_is_never_a_sibling_file` and
mutation-checked. This is why the `--offline` row above is part of the gate evidence: before the
fix it started a container instead of failing.

## Reproduce / regress

```bash
python -m unittest discover -s docker -p 'test_*.py'     # 33 tests, no GPU or downloads
./run.sh --dry-run                                       # the pinned swift line
bash arm-run.sh <label>                                  # prefill+stream+smoke+guards
bash gate-run.sh <label>                                 # 128K ladder + cancel + guards
```
