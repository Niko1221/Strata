# Model and engine lineage — run 20261021T130039Z-36b71c5e

Provenance record for the files the benchmark used. These are provenance records only: they do not imply that the
draft standard-v1 protocol has an approved Swift profile; every run in this PR stays on the custom track.

## Quantized model (VERIFIED)

- Repository: https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF
- Revision used: `0142554c963a8653912c0c8283f05f13f63653f3` (recorded as `model.revision` in results.json; matches
  the local HF cache `refs/main`).
- Pinned card: https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF/raw/0142554c963a8653912c0c8283f05f13f63653f3/README.md
- The five Q6_K shards are byte-identical to the LFS objects at that revision; the five SHA-256 values are
  published in the machine-readable record (`model.files[].sha256` in [../results.json](../results.json)) and
  match the Hugging Face LFS OIDs:
  1. `da4cedfee7e5298c14a20e1d6bc7ee697d9af5a77ef8eed629cc7c2af6f22776` (locally re-hashed, matches)
  2. `8aeb4c8e478809a91b2094f61ba276265d10425d036a3a9c5c9959ea650858d4`
  3. `c93bf8b3d818fa22ac1e50b524817bd85cf6b161f1bdf4714308fb7ffe328646`
  4. `8dc4f1ed14d0187e3e0e69458e8698da5ebd847ba07fe3697a16d1066195ea66`
  5. `44d0706c626f8bb04adb717c6cf66871c60de708f99e1053568ce5b320e93fad`
- Q6_K content entered the repo at commit `8968074ca5e8fd8335b2637d4e7a81f0947cd415` (2026-09-24T16:21:20Z);
  the used revision `0142554c` (2026-09-24T18:20:03Z) adds only the vision projector (Q6_K LFS OIDs unchanged).
- Conversion recipe, from the pinned card (VERIFIED as the publisher's statement): llama.cpp commit `d2e5458`,
  BF16 conversion of the published Swift 1.5 safetensors, bartowski's per-tensor type map, imatrix from his
  calibration-v6 corpus. Minor discrepancy: the card says 583 chunks; the GGUF records
  `quantize.imatrix.chunks_count = 582`.

## Source Swift checkpoint (INFERRED — not populated in results.json)

- Repository: https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next (the GGUF card cites the old name
  `ukisai/Swift-Qwen3.8-Flash-Next`, an HTTP 307 redirect to this repo).
- Commit `0bd4fe22431372cdad1979267d3ab45aa7e6150a` is the repo's only published revision (repo created
  2026-09-22; content commit stamped 2026-09-24T16:22:00Z).
- Pinned card / NOTICE:
  https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next/raw/0bd4fe22431372cdad1979267d3ab45aa7e6150a/README.md
  https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next/raw/0bd4fe22431372cdad1979267d3ab45aa7e6150a/NOTICE
- Status: INFERRED, not VERIFIED. The card says "derived directly from" the Swift checkpoint but pins no commit;
  no commit message, PR, or LFS history names a conversion input; the weights path (safetensors -> BF16 -> GGUF)
  has no published hash chain; the GGUF's `general.name` ("Swift 1.5 Flash Miroslav2.0") matches nothing
  published. Timeline caveat: the Q6_K initial release (16:21:20Z) is stamped 40 s BEFORE the checkpoint repo's
  single content commit (16:22:00Z). `model.source_revision` therefore stays null; the single published revision
  `0bd4fe22…` is the strong inference, not a verified conversion input.

## Tokenizer (VERIFIED — amended into results.json)

- Effective tokenizer revision: `0bd4fe22431372cdad1979267d3ab45aa7e6150a`
  (ukisai/Swift1.5-Qwen3.8-Flash-Next).
- Basis — byte equality, established 2026-10-21, at that pinned revision:
  - GGUF `tokenizer.ggml.tokens[0:248044]` == `vocab.json`@0bd4fe22 by rank (exact, 0 mismatches; +276 llama.cpp
    added tokens).
  - GGUF `tokenizer.ggml.merges` (247,587) == `merges.txt`@0bd4fe22 (exact; local pack copy = same content,
    trailing newline removed).
  - GGUF `tokenizer.chat_template` == `chat_template.jinja`@0bd4fe22 == local pack file, sha256
    `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` (8,952 B).
- Basis caveat (stated plainly): the checkpoint NOTICE says vocab, merges and template are "unmodified from
  Qwen3.8-Flash-Next" and the equality also holds on the Qwen base, so the byte match discriminates lineage
  strongly and revision exactly only because the cited Swift repo has a single published revision.
- Amendment (post-run metadata, 2026-10-21): `model.tokenizer_revision` set from null to
  `0bd4fe22431372cdad1979267d3ab45aa7e6150a`, and the `/model/tokenizer_revision` key removed from `missing`
  (the validator requires `missing` to list exactly the null values). The reasons recorded in `missing` for
  `/model/source_revision` and `/software/build_options` were extended to point at this document. No metric,
  timing, request result, score or hardware reading was changed; `strata bench report` revalidated the amended
  file (complete, 21/21, 0 failed, 12/12, identical speed medians).

## Qwen base model (repository VERIFIED, revision NOT-RECOVERABLE-VERIFIED)

- Repository: https://huggingface.co/Qwen/Qwen3.8-Flash-Next (cited by the Swift card's YAML
  `base_model` and L37).
- No pinned base commit appears anywhere in the published chain (Swift card/NOTICE, GGUF card, commit messages).
- Context: the Qwen repo has 3 commits; the weights landed at `34567a4712bc…` (2026-08-26T08:17:26Z) and the tree
  at that revision differs from the tip `de4b8e4d43b917e7706784d8bb445c9af86a3540` by README.md only — base
  model/tokenizer content has been constant, but no published record attributes a specific conversion revision.

## Model-card revisions (documented here; protocol field left unpinned)

- Quant card at `0142554c963a8653912c0c8283f05f13f63653f3`; Swift card/NOTICE at
  `0bd4fe22431372cdad1979267d3ab45aa7e6150a` (links above, immutable).
- `protocol.model_card_revision` stays null: the draft protocol's field belongs to the Qwen profile and the
  protocol has no approved Swift profile.

## Engine build provenance (VERIFIED chain; compile flags NOT published)

- Installed engine: `engine/strata`, sha256 `e54498e95ee5974c5a2b92b430aa867861baf884d217e7761dd68e2d905a1e2e`.
- `engine/BUILD.json` verbatim:
  `{"version": "0.1.46", "source": "f-021-rel a7bdbfdabc9a", "archs": [75, 86, 89, 120], "ptx": true, "cuda": "13.0", "portable": true, "vision": "cuda13.0"}`
- Source commit resolves to `a7bdbfdabc9a3e7c83e3dbc2441d0933569d773e` (2026-10-20T04:27:42Z) on
  https://github.com/Niko1221/Strata (branch `f-021-rel` no longer published; commit reachable via API).
- Integrity chain (verified 2026-10-21): local `engine/engine-component.json` sha256
  `7ff1d2840136bc64ba1a46b35114826624881bfce05911c1e15a5d2e945393f5` == `manifest_sha256` of
  `strata-feed.json` (sequence 106) of release https://github.com/Niko1221/Strata/releases/tag/v0.2.1 ;
  GitHub asset digest of `strata-engine-0.1.46-linux-x86_64-cuda-cuda13.tar.gz` `d9e564cf9c60…`; feed row fields:
  `build_id "a7bdbfdabc9a+cuda13.0"`, `archs ["sm_75","sm_86","sm_89","sm_120","compute_120"]`,
  `min_driver {"nvidia":"580.65"}`, `variant "cuda13"`; manifest Ed25519-signed (key 5d1bc23abbeffd0d).
- `software.build_options` stays null: the published records for the linux-x86_64 cuda13 engine carry build
  identity and targets, not compile flags (packaging/ENGINE-COMPONENT.md documents a cmake line only for the
  Windows cuda13 row). No flags were inferred from unrelated builds.
