# Hakua / Strata

[English](README.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [Polski](README.pl.md) · [Português](README.pt-BR.md) · [简体中文](README.zh-CN.md)

Local AI for Windows, with chat, agent work, research and a choice of compatible models. Hakua is the zapabob fork of Niko1221/Strata; it is independently maintained and is not an upstream-endorsed distribution.

## What changed in 0.7.2

This release absorbs all of official upstream **Niko1221/Strata v0.2.18** while keeping every fork-only feature. The native CUDA engine is now the **0.6.4 family** with two opt-in components, and the product follows semantic versioning at 0.7.2. The update includes the Hy4-Flash 220B MoE support (`strata-hy4` 0.1.64, Tree-8 CUDA Graphs, VNNI 1.5x), the official API/settings at control and engine 0.1.64, and upstream correctness fixes. A rebuilt Q38 engine covers both 27B (64 layers) and 9B (32 layers) GGUFs with control vectors; the official Hy4-Flash engine ships as a separate opt-in component. Per maintainer decision, ed25519 component signature checks were removed (no signing key exists); SHA-256 digest verification remains. All 16 fork anchors (Q38, DS4, RAM32, Hakua, 8 languages) are verified intact, and the 32 GB RAM squeeze remains. Full-window inference and general model quality stay unqualified (`releaseQualified=false`).

## Version and downloads

<!-- release-062-stable -->
Current app release: [0.7.2](https://github.com/zapabob/Strata/releases/tag/v0.7.2) (App `0.7.2`; Terminal, Control `0.6.3`; native engine family `0.6.4`) — this is GitHub Latest. The Windows app ships the **0.6.4 engine family** as separate opt-in components on the release page: a rebuilt Q38 engine (27B + 9B) and the official Hy4-Flash `strata-hy4` 0.1.64 engine. Component signature verification was removed (no signing key); SHA-256 digests remain the integrity check. `releaseQualified=false` remains explicit; full-window inference and general model quality remain unqualified.

[0.7.2](https://github.com/zapabob/Strata/releases/tag/v0.7.2) · [setup EXE](https://github.com/zapabob/Strata/releases/download/v0.7.2/Strata-0.7.2-windows-x64-setup.exe) · [tar.7z](https://github.com/zapabob/Strata/releases/download/v0.7.2/Strata-0.7.2-windows-x64.tar.7z) · [SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.7.2/SHA256SUMS.txt) · [BUNDLE-SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.7.2/BUNDLE-SHA256SUMS.txt)

![Hakua / Strata real-model Japanese chat on Qwen3.8-27B](docs/media/native-qwen38-japanese-chat.png)
*Hakua / Strata desktop running `qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf` via `strata-q38.exe` (CUDA 13 / RTX 5060 Ti 16 GB).*

## Recommended GGUF for the Qwen3.8 prototype

For local text chat on an RTX 5060 Ti with 16 GB VRAM, start with the tested [3.69 bpw mixed-quantized GGUF by soyaakinohara](https://huggingface.co/soyaakinohara/qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf). Thanks to **soyaakinohara** for quantizing and sharing it and to **AEON-7** for the source weights. Download the model separately and start with a 4,096-token context. The filename does not guarantee that it fits on a 12 GB card.

The [signed native CUDA 13.5 preview](https://github.com/zapabob/Strata/releases/tag/v0.7.2) passed two Japanese turns, mid-response Stop, and unload in installed app 0.6.3. In two first-request tests, short F16 prefill took 35–43% less time than the native control; Prism (llama.cpp) still decoded faster in the matched comparison. [Setup, public key and limits](docs/Q38_NATIVE_PREFILL_PREVIEW.md).

## Install or update

Use the Windows x64 NSIS installer for this fork. It installs per user and carries private Node, Python and Control; system development tools are unnecessary. The native CUDA engine is a separate component. The `tar.7z` bundle contains the installer, CUDA engine package, Control package, instructions and checksums, excluding model weights and personal data.

Finish requests, unload models, back up app data, then quit the app including its tray icon. Run the installer in the existing installation folder to overwrite-update while preserving model folders, chats, memory, tools, keys and approval choices. Import the compressed native component with *I already have the file*, choose a compatible model or local GGUF, then Load. A model name alone does not establish engine compatibility. See [installation and rollback](docs/HAKUA_INSTALL.md).

## Features kept in this fork

0.6.3 upgrades the [local Qwen3.8-27B-family dense GGUF runner (`strata-q38.exe`)](docs/QWEN38_DENSE_NATIVE.md) on Windows CUDA 13.5 / Blackwell (SM120):
- **Per-tensor mixed-quantization polymorphism (`StrataQuantTensor`)**: supports 11 GGML/Strata types (`F32`, `F16`, `BF16`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ3_S`, `IQ2_S`, `IQ4_XS`, and PrismML `PQ2_0=142`) on a per-tensor basis.
- **Tensor Core batched prefill + `spin8_triality.cuh` / fused Spin8 Attention + Multi-token GDN + CUDA Graph Decode (CUDA 13.5 SM120)**:
  - `140`-token prefill (`--kv spin8_k8v4 --ctx-size 4096`): improved from `36,760.7 ms` (`3.81 tok/s`, `--conservative-sync`) to **`1,558.3 ms` (`89.8 tok/s` warm, 23.6x speedup)**.
  - Steady-state decode (`GEN 32`, 31 steps, packed GEMV + 8-layer-fence CUDA Graph): improved from `206.4 ms/step` to **`85.13 ms/step` (`11.75 step/s`, 2.42x speedup)**.
  - **Long-context real-device measurements (`--kv spin8_k8v4`, RTX 5060 Ti 16 GB)**: **65K (`65,536`)** passed (`1,024`-token prefill `10,813.5–11,220.7 ms`, `91.3–94.7 tok/s`; decode `670.4 ms` / 7 steps); **96K (`98,304`)** passed (`required_with_reserve=15.09 GB`, `140`-token prefill `1,968.7 ms`); **131K (`131,072`)** is guarded by the 768 MiB GPU reserve (`required=15.97 GB > available=15.91 GB`).
- **Real MTP `K=3` draft/verify (`blk.64.nextn.*`)**: supports `--spec-draft-n-max 0..3` with `KvCache::rewind` and GDN state snapshot/restore.
- **KV cache & Japanese profile**: supports `F16`, `Spin8 K8V4`, and `F16-K / Turbo2/3/4-V` cache modes (`auto` order: `f16` → `f16_turbo4` → `spin8_k8v4`), plus optional Japanese output profile (`model-default` disables constraints).

`/btw` starts a real tool-free side-question subagent with its own stop control. `/research` and `/deepresearch` collect sources; configured web search can use bundled RustXNG Full. Search queries go to enabled external providers, so search is not entirely offline. Source counts do not establish answer accuracy.

Workflow skill learning and automatic application are separate options; new profiles keep both off. Existing choices, editable private skills, revisions and undo are preserved. Workspace and sidebar file operations, scoped chat history and memory, compatible PrismML/TurboQuant paths, bounded expert memory, pack integrity and diagnostic measurements remain part of the fork. See [skills](docs/WORKFLOW_SKILLS.md) and [search](docs/BROWSER_SEARCH.md).

## Official integration and compatibility

The pinned official source is [6e38f588](https://github.com/Niko1221/Strata/commit/6e38f588a646c0ee6948a74afbe670eb75896106), associated with official 0.2.18 / engine 0.1.64. It brings networking, drafting selection, prompt kernels, memory handling and platform fixes. Fork versions are independent. DeepSeek compatibility is retained as an extension where official paths were removed; a missing dedicated runtime prevents loading. This Windows package does not imply native image input, a distributed native DeepSeek engine, or qualified Linux/Mac inference.

## Protection and measured limits

Retain the 3 GiB available-RAM floor, 4 GiB commit headroom, at least 768 MiB GPU reserve and disk floors. The Windows RAM guard observes loading and generation. Capacity is checked for the actual model and selected backend; 32 GB RAM and 16 GB VRAM do not guarantee every model or context fits. Full-window inference, general model quality and simultaneous GPU use remain unqualified: `releaseQualified=false`. The installer has no Windows Authenticode signature. Historical measurements keep their original versions in [release evidence](docs/HAKUA_RELEASE.md).

## Documentation and development

[Install](docs/HAKUA_INSTALL.md) · [Release evidence](docs/HAKUA_RELEASE.md) · [API and hardware](docs/AI_REFERENCE.md) · [Terminal](docs/TERMINAL.md) · [Source build](docs/SOURCE_INSTALL.md) · [Packaging](packaging/README.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [License](LICENSE)

```sh
python -m pytest tests/terminal
```

```sh
cd apps/host
pnpm test
pnpm run build
```

### Niko1221/Strata — upstream Terminal reference

```sh
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

[Niko1221/Strata](https://github.com/Niko1221/Strata) · [Hermes Agent](https://github.com/NousResearch/hermes-agent) · [Third-party sources and notices](third_party/)

## Acknowledgements & Gratitude

We extend our most sincere and heartfelt gratitude to all upstream authors, open-source maintainers, researchers, and technology providers whose licence grants, foundational engineering, and generous stewardship make Hakua / Strata possible:

- **Niko1221/Strata** — We are deeply indebted to the upstream Strata project and its contributors for establishing the core architecture, engine, and terminal ecosystem upon which this independent fork is built.
- **ggml & llama.cpp** — Our warmest thanks for the enduring MIT-licensed tensor primitives, quantisation formats, and numerical foundations that underpin local inference.
- **NousResearch (Hermes Agent)** — With genuine appreciation for the agentic design patterns and memory-stack inspirations shared with the community.
- **Qwen Team (Alibaba Cloud)** — We gratefully acknowledge the Qwen model family and the Qwen Community License 1.0, enabling high-capability local multilingual reasoning.
- **NVIDIA Corporation** — Thank you for CUDA, cuBLAS, cuBLASLt, and the Blackwell (SM120) Tensor Core ecosystem, without which hardware-accelerated local inference would not be possible.
- **RustXNG, libstrix, Node.js, Python & Tauri** — Our profound thanks to every maintainer and contributor of these essential open-source runtimes, search tools, and cross-platform frameworks.
- **Microsoft Corporation** — With appreciation for the Windows Build Tools, Visual C++ Redistributable runtime, and WebView2 platform support.
- **Quantisation & Research Pioneers (ISTA-DASLab, PrismML, TurboQuant, Spin(8), and community GGUF authors)** — We express our deepest respect and thanks for advancing extreme low-bit quantisation, orthogonal rotation, and memory-efficient inference for consumer GPUs.

## What changed in 0.6.3

- Native Q38 prefill uses cuBLAS Tensor Core GEMM with token-scaled FP16 activations and a 20 MiB L2-persisting weight tile; decode uses format-specific aligned Q8_1 dp4a GEMV and a LayerFence(8) CUDA Graph.
- Long-context admission selects the prefill chunk from a measured VRAM ladder; the 768 MiB GPU reserve, 3 GiB RAM floor and 4 GiB commit floor are unchanged.
- Numerical fixtures normalise synthetic packed weights to real-model amplitude without weakening the strict `_exact_q8` tolerances; unaligned packed GEMV falls back to a byte-load scalar kernel.
- Verification: 17/17 native tests pass, and compute-sanitizer memcheck/synccheck/racecheck report 0 errors and 0 hazards. General model quality, full-window inference and GPU concurrency remain unqualified (`releaseQualified=false`).
- Research basis: NVIDIA CUDA 13.5 release notes (cuBLASLt algorithm-66 Tensor Memory correctness fix) and the Qwen3.8-Next architecture report (3:1 GDN/global-attention hybrid, QSA with r=4 compression and K=2048 token budget, gated residual, host-prefetched n-gram embedding).