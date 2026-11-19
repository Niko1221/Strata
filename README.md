<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="app/host/public/brand/strata-lockup-dark.svg">
    <img src="app/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>
# Hakua / Strata

[English](README.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [Polski](README.pl.md) · [Português](README.pt-BR.md) · [简体中文](README.zh-CN.md)

Local AI for Windows, with chat, agent work, research and a choice of compatible models. Hakua is the zapabob fork of Niko1221/Strata; it is independently maintained and is not an upstream-endorsed distribution.

## Version and downloads

<p align="center"><sub><b>English</b> · <a href="docs/lang/README.zh-CN.md">简体中文</a> · <a href="docs/lang/README.ja.md">日本語</a> · <a href="docs/lang/README.de.md">Deutsch</a> · <a href="docs/lang/README.fr.md">Français</a> · <a href="docs/lang/README.es.md">Español</a> · <a href="docs/lang/README.pt-BR.md">Português</a> · <a href="docs/lang/README.pl.md">Polski</a></sub></p>
<!-- release-062-stable -->
Current release: [0.6.3](https://github.com/zapabob/Strata/releases/tag/v0.6.3) (App `0.6.3`, Native Q38 `0.6.3` CUDA 13.5 SM120, Control `0.6.3`). `releaseQualified=false` remains explicit; full-window inference and general model quality remain unqualified.

[0.6.3](https://github.com/zapabob/Strata/releases/tag/v0.6.3) · [Strata-0.6.3-windows-x64-setup.exe](https://github.com/zapabob/Strata/releases/download/v0.6.3/Strata-0.6.3-windows-x64-setup.exe) · [tar.7z](https://github.com/zapabob/Strata/releases/download/v0.6.3/Strata-0.6.3-windows-x64.tar.7z) · [SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.6.3/SHA256SUMS.txt)

![Hakua / Strata real-model Japanese chat on Qwen3.8-27B](docs/media/native-qwen38-japanese-chat.png)
*Hakua / Strata desktop running `qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf` via `strata-q38.exe` (CUDA 13 / RTX 5060 Ti 16 GB).*

## Recommended GGUF for the Qwen3.8 prototype

For local text chat on an RTX 5060 Ti with 16 GB VRAM, start with the tested [3.69 bpw mixed-quantized GGUF by soyaakinohara](https://huggingface.co/soyaakinohara/qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf). Thanks to **soyaakinohara** for quantizing and sharing it and to **AEON-7** for the source weights. Download the model separately and start with a 4,096-token context. The filename does not guarantee that it fits on a 12 GB card.

The [signed native CUDA 13.5 preview](https://github.com/zapabob/Strata/releases/tag/v0.6.4-preview.1) passed two Japanese turns, mid-response Stop, and unload in installed app 0.6.3. In two first-request tests, short F16 prefill took 35–43% less time than the native control; Prism (llama.cpp) still decoded faster in the matched comparison. [Setup, public key and limits](docs/Q38_NATIVE_PREFILL_PREVIEW.md).

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

The pinned official source is [c775728069](https://github.com/Niko1221/Strata/commit/c77572806996349b49995e4f98d41b5294d36a21), associated with official 0.2.15 / engine 0.1.60. It brings networking, drafting selection, prompt kernels, memory handling and platform fixes. Fork versions are independent. DeepSeek compatibility is retained as an extension where official paths were removed; a missing dedicated runtime prevents loading. This Windows package does not imply native image input, a distributed native DeepSeek engine, or qualified Linux/Mac inference.

## Protection and measured limits

Retain the 3 GiB available-RAM floor, 4 GiB commit headroom, at least 768 MiB GPU reserve and disk floors. The Windows RAM guard observes loading and generation. Capacity is checked for the actual model and selected backend; 32 GB RAM and 16 GB VRAM do not guarantee every model or context fits. Full-window inference, general model quality and simultaneous GPU use remain unqualified: `releaseQualified=false`. The installer has no Windows Authenticode signature. Historical measurements keep their original versions in [release evidence](docs/HAKUA_RELEASE.md).

## Documentation and development

[Install](docs/HAKUA_INSTALL.md) · [Release evidence](docs/HAKUA_RELEASE.md) · [API and hardware](docs/AI_REFERENCE.md) · [Terminal](docs/TERMINAL.md) · [Source build](docs/SOURCE_INSTALL.md) · [Packaging](packaging/README.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [License](LICENSE)

<sub>Files: `Strata-0.2.16-windows-x64-setup.exe` · `Strata-0.2.16-linux-x86_64.AppImage` · `Strata-0.2.16-macos-arm64.dmg`</sub>

<p align="center">
  <img src="docs/media/app-chat.png" width="900" alt="Strata desktop app with a chat and live code preview side by side">
  <br><sub>Chat, files and a live preview in one window.</sub>
</p>

<a id="option-b-the-terminal-one-line"></a>

### Terminal app

Prefer the terminal? Install Strata, then run **`strata launch`**.
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

On Windows, open a new terminal afterwards. Then:

```sh
strata launch
```

Follow the prompts to download and start a model. [Terminal guide](docs/TERMINAL_LAUNCH.md).

</details>

<p align="center">
  <img src="docs/media/terminal-monitor.png" width="900" alt="Strata terminal app showing model activity, speed, memory and recent requests">
  <br><sub>Watch your model and hardware while it runs.</sub>
</p>

<a id="build-from-source"></a>

[Build from source](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md) · [Full install guide](docs/INSTALL.md)

<a id="what-you-need"></a>

## Hardware

| Supported hardware | Guide |
| --- | --- |
| **NVIDIA GeForce RTX 20 / 30 / 40 / 50 series** | [NVIDIA setup](docs/INSTALL.md#what-you-need) |
| **AMD Radeon RX 7700 / 7800 / 7900, RX 9060 / 9070, AI PRO R9700** | [AMD setup](docs/AMD_HIP.md) |
| **AMD Radeon RX 6800 / 6900 · experimental** | [RX 6000 setup](docs/AMD_HIP.md#rdna2-gfx1030) |
| **Intel Arc · Linux** | [Intel setup](docs/INTEL_ARC.md) |
| **Apple Silicon · M1 or newer** | [Mac setup](docs/MAC.md) |
| **AMD Ryzen AI Max · Strix Halo** | [Strix Halo setup](docs/STRIX_HALO.md) |
| **NVIDIA DGX Spark · Linux** | [DGX Spark setup](docs/DGX_SPARK_LINUX.md) |

<a id="using-it"></a>

## Tools

**Claude Code · Codex CLI · OpenCode · oh-my-pi · pi.dev · Hermes Agent · OpenClaw**, and more through the OpenAI- and Anthropic-compatible APIs.

Run **`strata connect`** to see setup instructions for your tool. [Connection guide](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect).

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## Help

[All guides](docs) · [Troubleshooting](docs/TROUBLESHOOTING.md) · [Report a bug](https://github.com/Niko1221/Strata/issues/new/choose) ·
[Benchmarks](docs/COMMUNITY_BENCHMARKS.md) · [How it works](docs/HOW_IT_WORKS.md) · [Contribute](.github/CONTRIBUTING.md)

## Support Strata

**Help keep local AI open to everyone.** A GitHub sponsorship or a coffee helps fund development and testing on more hardware.

<a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor_on_GitHub-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="GitHub Sponsors: sponsor Strata" height="48"></a>
<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee: support Strata" height="48"></a>

### Loyal supporters

Thank you to everyone who supports Strata.

<!-- supporters:start (generated from the public GitHub sponsors: one row per level, the biggest supporters first and largest; private sponsors are not listed) -->
<a href="https://github.com/codeconduit-official"><img src="https://images.weserv.nl/?url=github.com/codeconduit-official.png&w=240&h=240&fit=cover&mask=circle&output=png" width="120" height="120" alt="codeconduit-official" title="codeconduit-official"></a>

<a href="https://github.com/kod88vn"><img src="https://images.weserv.nl/?url=github.com/kod88vn.png&w=160&h=160&fit=cover&mask=circle&output=png" width="80" height="80" alt="kod88vn" title="kod88vn"></a>
<a href="https://github.com/router0mail"><img src="https://images.weserv.nl/?url=github.com/router0mail.png&w=160&h=160&fit=cover&mask=circle&output=png" width="80" height="80" alt="router0mail" title="router0mail"></a>
<a href="https://github.com/kingdonb"><img src="https://images.weserv.nl/?url=github.com/kingdonb.png&w=160&h=160&fit=cover&mask=circle&output=png" width="80" height="80" alt="kingdonb" title="kingdonb"></a>
<a href="https://github.com/Daisame"><img src="https://images.weserv.nl/?url=github.com/Daisame.png&w=160&h=160&fit=cover&mask=circle&output=png" width="80" height="80" alt="Daisame" title="Daisame"></a>

<a href="https://github.com/ryan4yin"><img src="https://images.weserv.nl/?url=github.com/ryan4yin.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="ryan4yin" title="ryan4yin"></a>
<a href="https://github.com/themactep"><img src="https://images.weserv.nl/?url=github.com/themactep.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="themactep" title="themactep"></a>
<a href="https://github.com/macintog"><img src="https://images.weserv.nl/?url=github.com/macintog.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="macintog" title="macintog"></a>
<a href="https://github.com/bisonbet"><img src="https://images.weserv.nl/?url=github.com/bisonbet.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="bisonbet" title="bisonbet"></a>
<a href="https://github.com/vizv"><img src="https://images.weserv.nl/?url=github.com/vizv.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="vizv" title="vizv"></a>
<a href="https://github.com/seoulworker"><img src="https://images.weserv.nl/?url=github.com/seoulworker.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="seoulworker" title="seoulworker"></a>
<a href="https://github.com/jaceknoga"><img src="https://images.weserv.nl/?url=github.com/jaceknoga.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="jaceknoga" title="jaceknoga"></a>
<a href="https://github.com/jackyini"><img src="https://images.weserv.nl/?url=github.com/jackyini.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="jackyini" title="jackyini"></a>
<a href="https://github.com/moruku36"><img src="https://images.weserv.nl/?url=github.com/moruku36.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="moruku36" title="moruku36"></a>
<a href="https://github.com/hrkgk"><img src="https://images.weserv.nl/?url=github.com/hrkgk.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="hrkgk" title="hrkgk"></a>
<!-- supporters:end -->
[Niko1221/Strata](https://github.com/Niko1221/Strata) · [Hermes Agent](https://github.com/NousResearch/hermes-agent) · [Third-party sources and notices](third_party/)

All supporters and how to support Strata: [SUPPORTERS.md](SUPPORTERS.md)

[MIT license](LICENSE) · [Credits and model licenses](docs/HOW_IT_WORKS.md#credits) · [Privacy](docs/SAFETY.md)
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
