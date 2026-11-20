# Hakua Windows third-party notices

Strata, the GUI and the native engine retain the repository MIT licence. The exact ggml dependency licence is included. PrismML and TurboQuant retain their own licences in their individual runtime directories.

The owned Q38 CUDA runner includes attributed PrismML 0.4.1 PQ2_0 and ggml packed-format mathematics under Q38-formats-MIT.txt. Turbo scalar tables/layout adaptations retain their immutable source pins and MIT notice in Turbo-dialects-MIT.txt. Spin8's SHA-256 and rotate-bits source notices accompany the component. q38-build.json records the source-relative compiled closure and fixed embedded profile. These numerical format ports do not introduce a llama runtime dependency into Q38; the older native engine and standalone runtimes retain their separate dependencies.

The upstream source attribution for libstrix's workgroup barrier is included as libstrix-NOTICE.md. Its original author and licence link are retained.

Node's official licence contains its bundled dependency notices. Python's licence and the locked Python distributions' available licence files accompany this package. Harness keeps its own licence and dependency notices inside its runtime directory. The GUI includes React, React DOM, Scheduler, KaTeX, PrismJS, PDF.js and html-to-image notices. Public icons use the pinned Lucide ISC source and the repository's owned artwork; supplied GT Walsheim Pro fonts and unqualified supplied artwork are excluded.

The unmodified NVIDIA CUDA/cuBLAS DLLs carry the supplied toolkit EULA and licence documents. The installed NVIDIA driver is supplied by Windows/the user and is not redistributed. Microsoft VC release CRT and OpenMP DLLs retain Microsoft terms. The original Community 2022, Diagnostic Build Tools 2022 and C Runtime 2015–2022 licence documents are copied from the vendor URLs pinned in microsoft-2022-notice-sources.json. These components are distributed as part of this application, not relicensed under MIT.

The upstream optional 480 KB Qwen3.8-Flash-Next control vector is kept unchanged and off by default. It is derived from model activations and carries Qwen Community License 1.0, included both in its data folder and here. Its source revision, byte size and SHA-256 are pinned in data/experimental-speed-projection/PROVENANCE.json. This component is not relicensed under MIT.

NOTICES.json identifies every notice file and SHA-256. Complete inference model weight packs, personal memory databases, user keys and model-provider artwork are not included.
