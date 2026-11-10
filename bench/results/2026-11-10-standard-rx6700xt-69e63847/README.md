# RX 6700 XT / Flash-Next IQ2_XS / draft protocol pilot

This is **strata-qwen38-thinking-v1-draft.1**, a custom-track pilot produced by the official unmodified `strata bench run`. It is not an activated or maintainer-approved standard result.

Required duration: **507.111 seconds** after loading. All 21 measured requests plus warm-up are retained: zero failed or skipped; sanity **12/12** (instruction 4/4, retrieval 4/4, tools 4/4). Optional concurrency was not run. Six performance requests reached their fixed 512-token cap; the warm-up reached its separate 128-token cap. These caps include reasoning. Small sanity checks do not establish general model quality.

## Hardware and software

One RX 6700 XT, nominal 12 GiB, gfx1031; Ryzen 5 3600, 6 cores / 12 host logical threads; physical 64 GiB DDR4-3200. The Debian 13 benchmark LXC had a 56 GiB memory limit and 10 visible CPU threads. Storage: Intel 660p 1 TB NVMe. Kernel 7.0.14-26-pve; amdgpu module source version B1D135B579C294611EB792D; 211 W GPU cap, PCIe 16 GT/s x16. Other GPU guests were stopped; one loopback client and a host resource sampler ran. Process RAM peak, energy per token and continuous VRAM peak were not measured. Original harness observations remain in runner-results.sanitized.json. Canonical hardware fields now record installed physical capacity, with the 56 GiB/10-thread guest allocation retained separately in extensions.

Release setup source: Strata v0.2.13, d2f096475817d4666c364c820b0ca830f751cfc5. Installed Terminal/Control: 0.1.59.1, commit 628f74955a56fcba092495b800dd02af75373a0f. Signed RX6000 engine: 0.1.59, BUILD source literal `f-0213-rc 712fec6b4297`. Portable HIP runtime and driver API version: 71326193, decoded as HIP 7.13.26193; exact signed library identities and direct pre-run API observation are in [runtime evidence](provenance/portable-runtime-identity.json). This is separate from the PolyStrata build SDK. Unknown release compiler options and an independent driver package version are not inferred.

## Model, configuration and template

The quantized Flash-Next files are from `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, pinned to revision ed59f92082b1e93c0e96d60a8b11aab089b52f09. All model filenames/hashes are in results.json. Source-content revision 34567a4712bc9766c4449e2e98e4468bfa24d915 is the earliest published immutable snapshot: all 131 source-weight LFS identities and computational/tokenizer blobs match across every published source revision. See [complete history evidence](provenance/source-content-history.json). This identifies source content; it does not recover the quantizer private checkout or independently reproduce quantization.

The model-card and tokenizer reference was pinned before loading to Qwen/Qwen3.8-Flash-Next revision 812d96e10a0dd82b69d749e68693698cf6c36daa. The separate custom template restores function-schema wrappers after Strata OpenAI preprocessing. All 34 frozen required/optional renderings matched the pinned upstream rendering offline. The original theoretical shared-prefix warning is retained in the offline audit; actual cached tokens were zero for all four tool cases and passed the unchanged official cache limits. Every required request used the frozen messages, seeds, caps, order and scoring.

Deliberate custom settings: DFlash2 Q4-selector pack, spec 6, confidence 0.8, five expert workers, no MTP; int8 KV, resident window 32768, context 131072, one server slot, automatic expert cache/prefill and normal conversation cache. No calibration or experimental projection stage was requested. This is not an untouched-default claim or an isolated speedup comparison. Startup/resolved configuration is recorded in [config.json](config.json), [observed native arguments](provenance/observed-engine-argv.json) and [engine.log](engine.log).

## Measurements

| Case | Prompt tok/s median (min–max) | Decode tok/s median (min–max) | TTFT seconds median (min–max) | Total seconds median (min–max) | n | Capped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short | not shown: fixed cost dominates | 28.787 (28.757–30.217) | 1.637 (1.447–1.659) | 17.162 (15.062–18.551) | 3 | 1/3 |
| prompt-4k | 325.342 (325.221–328.051) | 27.220 (25.629–29.431) | 12.638 (12.536–12.643) | 31.291 (28.929–32.568) | 3 | 2/3 |
| prompt-16k | 349.784 (349.651–349.864) | 28.256 (26.616–29.176) | 46.909 (46.905–46.941) | 64.982 (64.402–66.116) | 3 | 3/3 |

Prompt speed uses actual freshly read tokens and engine prompt time; decode includes reasoning and uses engine decode time. Per-request actual input/read/reused/output counts, early EOS/caps, client TTFT, first answer and total latency are retained in [results.json](results.json) and [outputs.jsonl](outputs.jsonl). Expert placement evolves; this is not a cold-cache benchmark. Loading and initial file hashing are outside the reported required duration. No quality or speed ranking against different models/quants is claimed.

## Evidence and reproduction

The exact [CLI argument array](provenance/command.json) and [source serving config](provenance/run-config.json) are attached; their original hashes are in [predeclared provenance](provenance/predeclared.json). Replace `<campaign>`, `<terminal>` and `<engine-component>` with local installation/model paths; keep the same configuration, attached template, model file hashes and immutable runner revision. Run the recorded official command against a freshly loaded dedicated loopback profile. All original fixture settings and the 1800-second request timeout remain unchanged. Validate with `strata bench report <report-directory>`.

The [independent execution audit](provenance/execution-audit.json) checks all 22 request records, corpus identity, messages/tools/settings/order, actual cache limits and input lengths, counts, native timings, strict scores, latency traces, summaries and original artifact hashes. It verifies execution; final publication conformance review is separate.

Factual identity, hardware and inventory corrections are explicit in [metadata-corrections.json](provenance/metadata-corrections.json): the prospectively pinned model-card reference replaces the runner hardcoded null, and the actual loaded Q4-selector pack replaces the CLI-preview fallback pack identity. The complete native pack inventory, including the 1.538 GB dense-weight file that the runner file discovery omitted, is added from post-run hashing; the observation time is explicit. [runner-results.sanitized.json](runner-results.sanitized.json) preserves the original runner fields after string redaction and artifact-hash updates. Every request record, performance summary, count, score and failed/skipped outcome is unchanged. Original owner evidence is retained with hashes.

This new custom-track run replaces an earlier withdrawn candidate whose template/provenance was insufficient for submission. That earlier candidate remains in owner evidence and prior branch history; its measurements are not merged into this run. All outcomes here are retained. No model weights, generated packs, private comparison tests or engine changes are part of this report.

Complete path-substitution and isolated-profile instructions: [REPRODUCE.md](REPRODUCE.md). Main-weight repository/LFS verification: [quantization identity](provenance/quantization-identity.json).

Native-pack preparation used the released `strata launch --family qwen --model IQ2_XS --context 128k --vision off` path into the campaign data directory. `tools/iq_pack.py` produced the native pack; its conversion metadata records the two original shard names/sizes and `compat_bf16=false`. [Native pack identities](provenance/native-pack-identities.json) records every packed file and source tokenizer hash after testing. The separate custom tokenizer copy differs only in its attached chat template. Generated tensors are not included in this submission. The native startup also records its Q5_K output head and verification windows up to eight tokens; launch requested `--spec 6`. These are retained as distinct observations.

[Hardware observations](provenance/hardware-observations.json) and direct runtime evidence fill canonical identity fields; their actual observation times are retained. The original runner observations remain alongside the corrected canonical report. The [pinned source card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/812d96e10a0dd82b69d749e68693698cf6c36daa/README.md) is linked at its immutable revision. Its original pre-run hash is recorded in the attached manifest; the owner retains the original source copy. Filenames in that pre-run manifest identify original collection files, while canonical artifacts enumerate the public attachments.
