# Hakua / Strata

[English](README.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [Polski](README.pl.md) · [Português](README.pt-BR.md) · [简体中文](README.zh-CN.md)

Local AI for Windows, with chat, agent work, research and a choice of compatible models. Hakua is the zapabob fork of Niko1221/Strata; it is independently maintained and is not an upstream-endorsed distribution.

## Version and downloads

<!-- release-061-stable -->
Current release: [0.6.1](https://github.com/zapabob/Strata/releases/tag/v0.6.1) (App `0.6.1`, Native Q38 `0.6.1` SM120, Control `0.6.0`). `releaseQualified=false` remains explicit; full-window inference and general model quality remain unqualified.

[0.6.1](https://github.com/zapabob/Strata/releases/tag/v0.6.1) · [Strata-0.6.1-windows-x64-setup.exe](https://github.com/zapabob/Strata/releases/download/v0.6.1/Strata-0.6.1-windows-x64-setup.exe) · [tar.7z](https://github.com/zapabob/Strata/releases/download/v0.6.1/Strata-0.6.1-windows-x64.tar.7z) · [SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.6.1/SHA256SUMS.txt)

![Hakua / Strata real-model Japanese chat on Qwen3.8-27B](docs/media/native-qwen38-japanese-chat.png)
*Hakua / Strata desktop running `qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf` via `strata-q38.exe` (CUDA 13 / RTX 5060 Ti 16 GB).*

## Install or update

Use the Windows x64 NSIS installer for this fork. It installs per user and carries private Node, Python and Control; system development tools are unnecessary. The native CUDA engine is a separate component. The `tar.7z` bundle contains the installer, CUDA engine package, Control package, instructions and checksums, excluding model weights and personal data.

Finish requests, unload models, back up app data, then quit the app including its tray icon. Run the installer in the existing installation folder to overwrite-update while preserving model folders, chats, memory, tools, keys and approval choices. Import the compressed native component with *I already have the file*, choose a compatible model or local GGUF, then Load. A model name alone does not establish engine compatibility. See [installation and rollback](docs/HAKUA_INSTALL.md).

## Features kept in this fork

0.6.1 upgrades the [local Qwen3.8-27B-family dense GGUF runner (`strata-q38.exe`)](docs/QWEN38_DENSE_NATIVE.md) on Windows CUDA 13 / Blackwell (SM120):
- **Per-tensor mixed-quantization polymorphism (`StrataQuantTensor`)**: supports 11 GGML/Strata types (`F32`, `F16`, `BF16`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ3_S`, `IQ2_S`, `IQ4_XS`, and PrismML `PQ2_0=142`) on a per-tensor basis.
- **Tensor Core batched prefill + `spin8_triality.cuh` batch KV (`KvCache::enqueue_batch`)**: replaces per-token prefill GEMV with ping-pong double-buffered `cublasGemmEx` (`prefill_gemm=cublas_tensor_op`, plus opt-in `STRATA_FP8_PREFILL=1`). In a 140-token prefill + 8-token decode benchmark (`--kv spin8_k8v4 --ctx-size 4096`) on RTX 5060 Ti SM120, prefill improved from `36,760.7 ms` (`3.81 tok/s`, `--conservative-sync`) to `11,387.4 ms` (`12.29 tok/s`, **3.23x** speedup).
- **Real MTP `K=3` draft/verify (`blk.64.nextn.*`)**: supports `--spec-draft-n-max 0..3` with `KvCache::rewind` and GDN state snapshot/restore.
- **KV cache & Japanese profile**: supports `F16`, `Spin8 K8V4`, and `F16-K / Turbo2/3/4-V` cache modes (`auto` order: `f16` → `f16_turbo4` → `spin8_k8v4`), plus optional Japanese output profile (`model-default` disables constraints).

`/btw` starts a real tool-free side-question subagent with its own stop control. `/research` and `/deepresearch` collect sources; configured web search can use bundled RustXNG Full. Search queries go to enabled external providers, so search is not entirely offline. Source counts do not establish answer accuracy.

Workflow skill learning and automatic application are separate options; new profiles keep both off. Existing choices, editable private skills, revisions and undo are preserved. Workspace and sidebar file operations, scoped chat history and memory, compatible PrismML/TurboQuant paths, bounded expert memory, pack integrity and diagnostic measurements remain part of the fork. See [skills](docs/WORKFLOW_SKILLS.md) and [search](docs/BROWSER_SEARCH.md).

## Official integration and compatibility

The pinned official source is [c775728069](https://github.com/Niko1221/Strata/commit/c77572806996349b49995e4f98d41b5294d36a21), associated with official 0.2.15 / engine 0.1.60. It brings networking, drafting selection, prompt kernels, memory handling and platform fixes. Fork versions are independent. DeepSeek compatibility is retained as an extension where official paths were removed; a missing dedicated runtime prevents loading. This Windows package does not imply native image input, a distributed native DeepSeek engine, or qualified Linux/Mac inference.

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
