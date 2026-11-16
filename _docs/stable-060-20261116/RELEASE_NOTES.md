# Hakua / Strata 0.6.0 — Windows x64

The Windows distribution contains the per-user desktop installer, the identical installer inside the tar.7z archive, and separate signed CUDA engine and Control packages. The app, Terminal, Control and native engine use product version 0.6.0. SOURCE.json binds the exact shipped source, component, installer and upstream cutoff. QUALITY.json records the observed checks and their limits.

This fork incorporates official Strata v0.2.15 at the fixed source cutoff c77572806996349b49995e4f98d41b5294d36a21. Fork versions remain independent of official engine versions. Existing user settings, skills and model files are preserved by the protected overwrite process. The separate engine package must also be imported and selected; replacing the desktop app alone does not replace an existing engine.

RustXNG 0.31.1 Full and its exact corresponding source, LICENSE and NOTICE are supplied. The Windows Terminal installer uses this fork's signed component feed. No Linux or Mac Terminal archive is included in this Windows distribution.

The separate native CUDA 13 SM120 runner loads original Qwen3.8-27B-family GGUF files through its own metadata and tensor admission checks. It supports the admitted IQ formats and PrismML 0.4.1 PQ2_0 files. Conversation-memory choices include explicit Spin8 K8V4 and F16 K with Turbo4 V, plus automatic selection within the configured memory budget. A Soya GGUF containing MTP tensors can use its main model; the draft tensors remain unused.

Japanese output and visible reasoning use an instruction profile with finite byte-BPE suppression of a fixed set of 24 characters before sampling. This changes neither training nor model weights. The profile preserves literal names and code from the input, and the model-default choice disables it. These mechanisms do not establish universal language or formatting compliance. Native MTP, a full execution graph and native vision are not implemented or qualified. Full-window 98k or 131k completion within five minutes remains unqualified; examples do not establish 100 tokens per second.

The native evidence distinguishes the retained two generic IQ2 spill-default failures from the explicitly bound Windows compiler profile. The ten GPU numerical/lifecycle checks qualify only their recorded source and binary equality scope. They do not establish model quality, full-window inference, speed improvements or concurrent GPU operation. Hosted compile checks do not establish hardware inference on those platforms.

Installed WebView2 and Harness fixtures use synthetic model responses; genuine local-model observations use the selected Qwen3.5 9B reference through PrismML 0.4.1 and are reported separately. If that model declines reusable learning, QUALITY.json retains NOT_REUSABLE for learning and automatic application. A limited publication in that case requires a separate pinned owner adjudication; it does not make those checks pass. Current factual accuracy and recall review statuses remain visible in QUALITY.json.

General model quality, full-window generation and GPU concurrency remain unqualified; releaseQualified=false. The Windows app installer has no Authenticode signature. Engine component and feed signatures are separate and are verified against bundled production public keys.

Before overwriting an installation, finish requests, unload models and close the app. Preserve a private backup. Use INSTALL.md for the desktop and separate-engine steps. Verify SHA256SUMS.txt for the public assets and BUNDLE-SHA256SUMS.txt for the archive's flat members. Older immutable releases and their assets remain available.
