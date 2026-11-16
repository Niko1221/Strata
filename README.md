# Hakua / Strata

[English](README.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [Polski](README.pl.md) · [Português](README.pt-BR.md) · [简体中文](README.zh-CN.md)

Local AI for Windows, with chat, agent work, research and a choice of compatible models. Hakua is the zapabob fork of Niko1221/Strata; it is independently maintained and is not an upstream-endorsed distribution.

## Version and downloads

<!-- release-060-candidate -->
This source is the 0.6.0 integration candidate. App, Terminal, Control and native source versions are coordinated at 0.6.0. Candidate build, installation and publication checks are still in progress; a version number is not proof that an asset is available. The retained verified release is [0.5.9](https://github.com/zapabob/Strata/releases/tag/v0.5.9). Use the delivery record to check the status of 0.6.0.

[0.6.0](https://github.com/zapabob/Strata/releases/tag/v0.6.0) · [Strata-0.6.0-windows-x64-setup.exe](https://github.com/zapabob/Strata/releases/download/v0.6.0/Strata-0.6.0-windows-x64-setup.exe) · [tar.7z](https://github.com/zapabob/Strata/releases/download/v0.6.0/Strata-0.6.0-windows-x64.tar.7z) · [SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.6.0/SHA256SUMS.txt)

## Install or update

Use the Windows x64 NSIS installer for this fork. It installs per user and carries private Node, Python and Control; system development tools are unnecessary. The signed native CUDA engine is a separate component. The tar.7z bundle must contain the identical installer, component packages, instructions and checksums. It excludes model weights and personal data. Extract 7z first, then tar, and check BUNDLE-SHA256SUMS.txt.

Finish requests, unload models, back up app data, then quit the app including its tray icon. Run the installer in the existing installation folder. Preserve model folders, chats, memory, tools, keys and approval choices. Import the compressed native component with I already have the file; choose a compatible model or a local GGUF, then Load. A model name alone does not establish engine compatibility. See [installation and rollback](docs/HAKUA_INSTALL.md).

## Features kept in this fork

The 0.6.0 source candidate adds a [local Qwen3.8-27B-family dense GGUF trial](docs/QWEN38_DENSE_NATIVE.md) on Windows CUDA 13 / Blackwell. Strata owns its loader, forward pass and CUDA operations, with attributed numerical format ports and no llama runtime dependency in this runner. The signed component must cover the additional `strata-q38.exe`; original model weights stay separate. Metadata and full tensor validation determine compatibility, including renamed checkpoints. Existing native engines and standalone PrismML 0.4.1 / TurboQuant 0.4.0 choices remain available.

The new path supports original PrismML PQ2_0 weights and explicit F16, Spin8 K8V4 or F16-K/Turbo2/3/4-V cache modes. Auto tries F16, Turbo4, then Spin8 under the unchanged resource reserves. Japanese output uses a no-training instruction profile and finite generation-time suppression, preserving literal input; model-default disables both. This cannot guarantee universally Japanese answers. Short-context mechanics and sampled answers have been measured; practical 98,304/131,072-token input and a five-minute first response remain pending. Prefill is still sequential. MTP, full CUDA graphs and batched prefill are deferred until after packaging.

`/btw` starts a real tool-free side-question subagent with its own stop control. `/research` and `/deepresearch` collect sources; configured web search can use bundled RustXNG Full. Search queries go to the enabled external providers, so search is not entirely offline. Source counts do not establish answer accuracy.

Workflow skill learning and automatic application are separate options; new profiles keep both off. Existing choices, editable private skills, revisions and undo are preserved. Workspace and sidebar file operations, scoped chat history and memory, compatible PrismML/TurboQuant paths, bounded expert memory, pack integrity and diagnostic measurements remain part of the fork. See [skills](docs/WORKFLOW_SKILLS.md) and [search](docs/BROWSER_SEARCH.md).

## Official integration and compatibility

The pinned official source is [c775728069](https://github.com/Niko1221/Strata/commit/c77572806996349b49995e4f98d41b5294d36a21), associated with official 0.2.15 / engine 0.1.60. It brings networking, drafting selection, prompt kernels, memory handling and platform fixes. Fork versions are independent. DeepSeek compatibility is retained as an extension where official paths were removed; a missing dedicated runtime must prevent loading. This Windows package does not imply native image input, a distributed native DeepSeek engine, or qualified Linux/Mac inference.

## Protection and measured limits

Retain the 3 GiB available-RAM floor, 4 GiB commit headroom, at least 768 MiB GPU reserve and disk floors. The Windows RAM guard also observes loading and generation. Capacity is checked for the actual model and selected backend; 32 GB RAM and 16 GB VRAM do not guarantee every model or context fits. Full-window inference, general model quality and simultaneous GPU use remain unqualified: releaseQualified=false. The installer has no Windows Authenticode signature; component/feed signatures are a separate mechanism. Historical measurements keep their original versions in [release evidence](docs/HAKUA_RELEASE.md).

## Documentation and development

[Install](docs/HAKUA_INSTALL.md) · [Release evidence](docs/HAKUA_RELEASE.md) · [API and hardware](docs/AI_REFERENCE.md) · [Terminal](docs/TERMINAL.md) · [Source build](docs/SOURCE_INSTALL.md) · [Packaging](packaging/README.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [License](LICENSE)

```sh
python -m pytest tests/terminal
```

```sh
cd apps/host
npm test
npm run build
```

### Niko1221/Strata — upstream Terminal reference

```sh
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

[Niko1221/Strata](https://github.com/Niko1221/Strata) · [Hermes Agent](https://github.com/NousResearch/hermes-agent) · [Third-party sources and notices](third_party/)
