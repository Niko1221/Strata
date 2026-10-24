<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="380">
  </picture>
</p>

<p align="center"><a href="README.md">English</a> · <b>简体中文</b> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a></p>

<h3 align="center">在你自己的游戏电脑上运行最先进的模型</h3>

<p align="center">支持 NVIDIA、AMD、Intel · Windows、Linux、Mac · 免费开源</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?include_prereleases&label=release" alt="最新版本"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT 许可证"></a>
  <a href="https://github.com/Niko1221/Strata/stargazers"><img src="https://img.shields.io/github/stars/Niko1221/Strata?style=flat" alt="GitHub 星标"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?logo=buymeacoffee&logoColor=black" alt="请我喝杯咖啡"></a>
</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="Strata 的模型写出的体素宝塔花园，在浏览器里运行"></a><br>
<sub>一座体素宝塔花园，由一条提示词一次生成，在装有 RTX 5070 的电脑上用 Strata 运行（IQ3_S，128K 上下文） ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">完整视频（49 秒）</a></sub></p>

<p align="center">
  <a href="#安装"><b>安装</b></a> ·
  <a href="#模型">模型</a> ·
  <a href="#速度如何">速度</a> ·
  <a href="#使用方法">使用方法</a> ·
  <a href="#工作原理">工作原理</a> ·
  <a href="#出问题了">帮助</a> ·
  <a href="#支持-strata">支持</a>
</p>

Strata 运行 **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**。这是一个 1250 亿参数的模型，
通常需要服务器才能跑，现在用普通电脑就行。从 0.2.4 起，它在终端版里也能运行 **DeepSeek V4.1-Flash**。它能聊天、写代码、
看图片，也能配合你的应用和编程智能体使用。你的聊天记录和文件都留在你自己的电脑上。

- **游戏电脑就能跑。** 内存和显存加起来有 32 GB 就可以开始。
- **能接入你的工具。** Claude Code、Codex CLI、OpenCode 等智能体通过 OpenAI 和 Anthropic 的 API 连接。
- **能看图片**，默认开启。
- **免费且隐私。** MIT 许可证，不用注册账号，不走云端，没有遥测（[它下载什么、什么时候下载](docs/SAFETY.md)）。

## 硬件要求

**内存和显存（VRAM）加起来至少要有 32 GB。** Mac 和 Ryzen AI Max 上指的是统一内存。32 GB 内存加一张 12-16 GB 显存的显卡，
可以跑较小的尺寸；64 GB 可以跑所有尺寸。

| 适用于 | 指南 |
| --- | --- |
| **NVIDIA** GeForce RTX 20、30、40 和 50 系列，12 GB 及以上 | [安装](#安装) |
| **AMD** Radeon RX 6800 及更新型号、Radeon AI PRO R9700 | [AMD 显卡](docs/AMD_HIP.md) |
| **Intel** Arc | [Intel Arc](docs/INTEL_ARC.md) |
| **Apple 芯片 Mac**，M1 及更新，macOS 15 及以上 | [Mac](docs/MAC.md) |
| **AMD Ryzen AI Max（Strix Halo）** 及其他统一内存的电脑 | [Strix Halo](docs/STRIX_HALO.md) |
| 较旧的显卡和处理器 | [旧显卡](docs/OLDER_GPUS.md) |

你还需要约 80 GB 的可用磁盘空间（固态硬盘）、Windows 10 / 11、Linux 或 macOS，以及较新的显卡驱动。同一品牌的两三张显卡可以共同运行一个模型（[多显卡](docs/MULTI_GPU.md)）。

## 安装

### 1. Strata 应用（最简单）

下载适合你电脑的文件，打开它，然后对推荐的模型点击安装。

**Windows 10 / 11**

[![Windows，NVIDIA，驱动 580 或更新](https://img.shields.io/badge/Windows-NVIDIA%20%C2%B7%20driver%20580%2B-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-cuda13.exe)
[![Windows，NVIDIA，较旧的驱动](https://img.shields.io/badge/Windows-NVIDIA%20%C2%B7%20older%20driver-4C7A00?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-cuda12.exe)
[![Windows，AMD RX 9000 或 R9700](https://img.shields.io/badge/Windows-AMD%20RX%209000%20%C2%B7%20R9700-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-hip.exe)
[![Windows，AMD RX 7000 或 Ryzen AI Max](https://img.shields.io/badge/Windows-AMD%20RX%207000%20%C2%B7%20Ryzen%20AI%20Max-B3131B?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-hip-rdna3.exe)

**Linux**

[![Linux，NVIDIA，驱动 580 或更新](https://img.shields.io/badge/Linux-NVIDIA%20%C2%B7%20driver%20580%2B-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-nvidia-cuda13.AppImage)
[![Linux，NVIDIA，较旧的驱动](https://img.shields.io/badge/Linux-NVIDIA%20%C2%B7%20older%20driver-4C7A00?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-nvidia-cuda12.AppImage)
[![Linux，AMD RX 9000 或 R9700](https://img.shields.io/badge/Linux-AMD%20RX%209000%20%C2%B7%20R9700-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-rx9070.AppImage)
[![Linux，AMD RX 7000](https://img.shields.io/badge/Linux-AMD%20RX%207000-B3131B?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-rx7900.AppImage)
[![Linux，AMD Ryzen AI Max](https://img.shields.io/badge/Linux-Ryzen%20AI%20Max-8A0E15?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-strixhalo.AppImage)

**Mac**（Apple 芯片，macOS 15 及以上）

[![Mac，Apple 芯片](https://img.shields.io/badge/Mac-Apple%20Silicon-333333?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-mac.dmg)

NVIDIA 驱动低于 580（`nvidia-smi` 能看到版本）？请选较旧驱动的文件。AMD RX 6000 显卡用 RX 7000 的文件，它会自己下载合适的引擎。
所有文件都在[发布页](https://github.com/Niko1221/Strata/releases/latest)。

| 首次启动 | 怎么做 |
| --- | --- |
| **Windows** | 运行 `.exe`。不需要管理员权限。如果 Windows 提示“Windows 已保护你的电脑”，点击**更多信息**，再点**仍要运行**。 |
| **Linux** | 给文件加上可执行权限（`chmod +x`），然后启动。需要 FUSE 2：`sudo apt install libfuse2t64`（Ubuntu 24.04 及更新）或 `libfuse2`（22.04）。 |
| **Mac** | 打开 `.dmg`，把 Strata 拖到“应用程序”。 |

模型下载很大（40-70 GB）。如果中断，会从断点继续。

### 2. 终端版（一行命令）

`strata` 命令，适合脚本、服务器和 AI 智能体。运行 DeepSeek 也要用它。

Linux 和 Mac：

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

Windows（PowerShell）：

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

然后启动它。它会一次问一个问题（模型、上下文、图片、允许谁连接），下载需要的东西，并开始提供服务：

```sh
strata launch
```

不需要管理员权限。安装程序会校验它下载的所有内容的 SHA-256。在 Windows 上，装完后请打开一个新的终端窗口。
所有界面和参数见 [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md)。

### 3. 让你的 AI 来安装

把下面这段话粘贴到 Claude Code、Cursor、Codex、GitHub Copilot 或其他编程助手里：

```text
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

它会检查你的显卡、内存和磁盘，选出合适的模型，完成安装和启动，并告诉你怎么连接你的应用。AI 工具也可以通过 Strata 的
[MCP 服务器](docs/MCP_SERVER.md)安装、启动和停止 Strata。

想自己动手编译？请看[从源码构建](#从源码构建)。

## 模型

应用和 `strata launch` 会按你的内存推荐尺寸。尺寸小的更快，尺寸大的稍微更聪明一点。

### 内存不够？试试 Gyro

[Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF)（作者 agentionai）是运行 Qwen3.8-Flash-Next 占用最小的方式。
它可在 NVIDIA、AMD、Intel Arc 和 Apple 芯片上运行。

| 尺寸 | 所需内存（RAM + VRAM） | 下载大小 | 适合 |
| --- | ---: | ---: | --- |
| **Gyro-S** | 27.6 GB | 58.5 GB | 其他尺寸都放不下的电脑。32 GB 的 Mac mini 上每秒 9 个 token。 |
| **Gyro-M** | 35 GB | 92.0 GB | 质量约相当于 IQ3_XXS，显存占用少 20%。 |

用 `strata launch --model gyro-s` 启动。详情：[docs/GYRO.md](docs/GYRO.md)。

### Qwen3.8-Flash-Next 的尺寸

| 你的内存（RAM + VRAM） | 选择 | 原因 |
| --- | --- | --- |
| **32 GB** | **Gyro-S**，写代码可选 **[Coder](docs/MODELS.md#coder)** | Coder 保留代码用到的专家 |
| **40 GB** | **[Swift 1.5](docs/MODELS.md#swift-15)** IQ2_XS（39.2 GB）或 **Gyro-M** | Swift 回答前思考的时间更短 |
| **48 GB** | **IQ3_XXS**（47.0 GB） | 更慢，更好 |
| **64 GB** | **IQ3_S**（54.8 GB） | 在公开测试中与完整模型持平 |
| **96 GB 及以上** | Unsloth 的 **UD-IQ4_XS** 到 **Q8_0** | 最接近完整模型 |

内存比这一行写的少，也可能能用，只是更慢：Strata 会自动切换到[低内存模式](docs/MODELS.md#a-big-graphics-card-and-little-ram)。
另外还有：给 NVIDIA 显卡用的 [NVFP4](docs/NVFP4.md)、[Qwen3.6-35B-A3B](docs/QWEN36.md)（更小，能同时处理多个请求），
以及 [OrcaRouter 的 Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs)。所有尺寸，以及各种配置能跑什么：
[docs/MODELS.md](docs/MODELS.md)。

### DeepSeek V4.1-Flash（终端版，0.2.4 起）

一个大得多的模型，用一块 12 GB 显卡加 46 到 64 GB 内存就能运行。Strata 把最常用的部分留在内存里，其余的从你的固态硬盘读取。
这些版本由 [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) 制作。它可在
NVIDIA RTX 30 到 50 系列显卡和 Radeon AI PRO R9700 上运行。请在 `strata launch` 里选择它。桌面应用暂时不提供。

| 版本 | 下载大小 | 质量 | 写代码速度 |
| --- | ---: | --- | --- |
| **IQ1_M**（1.75 位） | 239 GB | 最小，普通文字表现较弱 | 还没在电脑上测量 |
| **IQ2_XXS**（2.00 位） | 257 GB | 平衡得不错 | 每秒 10 个 token（RTX 3060），5（RTX 5070），48（4 块 R9700） |
| **IQ2_XS**（2.25 位） | 274 GB | 最接近原模型 | 每秒 15 个 token（一块 R9700），80（4 块 R9700） |
| **IQ4_NL**（Engram 表） | 已包含在上面每个下载中 | 三个版本共用的查找表 | |

用 `strata launch --model ist-das-2.00bpw`（或 `ist-das-1.75bpw`、`ist-das-2.25bpw`）启动。质量数据、各台电脑的速度和手动安装：
[docs/DEEPSEEK41.md](docs/DEEPSEEK41.md)。

## 速度如何？

以下数据基于 12 GB 显卡（RTX 5070、64 GB 内存）。其他显卡的结果见：
[各尺寸的速度](docs/MODELS.md#how-fast-is-each-size)和[社区测试结果](docs/COMMUNITY_BENCHMARKS.md)。

| 模型 | 回答输出速度 | 读取提示词速度 |
| --- | --- | --- |
| **Q2_0** | `████████████` 94 tokens/s | `████████████` 2,650 tokens/s |
| **IQ2_XS** | `██████████░░` 79 tokens/s | `█████████░░░` 2,090 tokens/s |
| **IQ3_XXS** | `████████░░░░` 62 tokens/s | `████████░░░░` 1,750 tokens/s |
| **IQ3_S** | `███████░░░░░` 53 tokens/s | `███████░░░░░` 1,620 tokens/s |
| **Coder** | `███████░░░░░` 55 tokens/s | `██████████░░` 2,180 tokens/s |

在 16 GB 的 AMD 显卡上（RX 9070 XT、47 GB 内存）：

| 模型 | 回答输出速度 | 读取提示词速度 |
| --- | --- | --- |
| **Q2_0** | `████████░░░░` 60 tokens/s | `█████░░░░░░░` 1,160 tokens/s |
| **IQ2_XS** | `███████░░░░░` 52 tokens/s | `█████░░░░░░░` 1,110 tokens/s |
| **Coder** | `██████░░░░░░` 44 tokens/s | `██████░░░░░░` 1,420 tokens/s |

一个 token 约等于 ¾ 个英文单词，所以每秒 60 个 token 比你阅读的速度还快。“读取提示词”指 Strata 吸收长文档、代码或聊天历史的速度。

## 使用方法

### 在应用里

聊天、显示模型以及 GPU、CPU 和内存状态的**监控**、**模型**页面和**设置**。聊天可以在你电脑上的某个文件夹里工作。
在**手动**模式下，修改文件或运行命令之前它会先询问你。它可以搜索网页（DuckDuckGo 或 Bing，不需要账号）。
发出电脑的只有搜索词。

<p align="center"><img src="docs/media/app-chat.png" width="900" alt="Strata 应用：一次对话，智能体做出了一个小型体素宝塔花园，页面在右侧的预览中打开"><br>
<sub>智能体用一个 HTML 文件写出了体素宝塔花园，页面在聊天旁的预览中打开</sub></p>

### 在终端里

<p align="center"><img src="docs/media/terminal-monitor.png" width="900" alt="模型写回复时，Strata 终端监控的总览界面"><br>
<sub>模型回复时的 <code>strata monitor</code>：速度、GPU、内存和最近的请求</sub></p>

```sh
strata launch                              # guided: model, context, pictures, access, then serve
strata launch --model q2_0 --yes --no-tui  # no questions: serve and print the address
strata launch --status                     # what is being served now
strata launch --stop                       # stop it
strata monitor                             # watch speed, GPU, memory and requests
strata update                              # get the newest engine
strata doctor                              # check the installed files
```

要从手机或另一台电脑访问 Strata，请使用 `strata launch --access lan`。它会为你生成一个 API 密钥。
`strata --help` 会列出所有命令，[docs/TERMINAL.md](docs/TERMINAL.md) 有详细说明。

### 配合 AI 智能体和其他应用

智能体用上面的一行命令安装 Strata，然后运行 `strata launch --model NAME --yes --no-tui`。步骤、参数和退出码见
[docs/AI_SETUP.md](docs/AI_SETUP.md)。给智能体看的仓库规则：[AGENTS.md](AGENTS.md)。

| 你的应用使用 | 基础 URL |
| --- | --- |
| OpenAI API（大多数应用和智能体） | `http://127.0.0.1:8080/v1`（API 密钥和模型名称随便填） |
| Anthropic API（Claude Code） | `http://127.0.0.1:8080`，不带 `/v1`。Claude Code：`ANTHROPIC_BASE_URL=http://127.0.0.1:8080` |
| OpenAI Responses API（Codex CLI） | `http://127.0.0.1:8080/v1/responses`（[设置方法](docs/DETAILS.md#the-responses-api-and-codex-cli)） |

`strata connect` 会打印 OpenCode、Codex CLI、Claude Code、oh-my-pi、pi、Hermes Agent 和 OpenClaw 的具体设置，
`strata connect opencode --write` 会帮你写入（并保留备份）。

### 使用须知

- **图片**对每个能看图片的模型都是默认开启的。在聊天里附加或粘贴图片，或者从你的应用发送图片。详情：[docs/IMAGES.md](docs/IMAGES.md)。
- **思考级别**可以是 off、low、medium 或 high，在聊天菜单或你的应用的“reasoning effort”里设置。off 最快。
- **聊天的第一条消息**会被完整读取，每 30,000 个 token 约需 1 分钟。之后的消息几秒钟就开始。
- **默认一次处理一个请求。** 要处理更多，请使用 `--parallel N`（[BATCHING.md](docs/BATCHING.md)）。在 12 GB 显卡上，这会让每个回答变慢。

## 从源码构建

<details>
<summary>从仓库构建 Windows、Linux 和 Mac 版本</summary>

[下载 Strata](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip)并解压，或者用 `git clone` 获取。
**Windows：** 双击 `START-HERE.bat`。**Linux：** 在 Strata 文件夹里运行 `./setup.sh`。

**Mac（Apple 芯片，macOS 15 及以上）：** 先用 `xcode-select --install` 安装一次 Apple 的命令行工具（不需要完整的 Xcode）。
在 Strata 文件夹里打开终端，运行 `./setup.sh`。它会在你的 Mac 上构建 Mac 引擎（约 7 分钟），问同样的问题，下载模型并启动。
下次运行 `./setup.sh` 会直接启动。你的 Mac 适合哪个尺寸：[docs/MAC.md](docs/MAC.md)。

安装程序会找到你的显卡，并设置合适的引擎。它会问你用哪个模型和尺寸、上下文多长（模型能记住多少文字），以及是否要读取图片。
直接按回车就是推荐的选择。然后它会下载模型（约 70 GB）并启动。如果下载中断，再运行一次就会继续。浏览器会打开
`http://127.0.0.1:8080` 上的 Strata 页面。

> 模型启动期间，你的电脑可能会变慢，或者在 1-3 分钟内没有响应（第一次最久）。Strata 会把 35-55 GB 加载到内存里，并锁定其中一部分
> 给显卡使用。请等待，不要关闭窗口。

下次再运行 `START-HERE.bat`（或 `./setup.sh`）会立即启动，不会重复下载。关闭它的窗口就能停止模型。`UPDATE.bat`
（`./update.sh`）只更新 Strata，不启动。更新、Docker、多显卡、文件存放位置和所有选项：[docs/INSTALL.md](docs/INSTALL.md)。

要测试你的电脑，运行 `BENCH.bat`（`./bench.sh`）。需要 5-10 分钟，并生成一份可以分享的报告
（[方法](docs/COMMUNITY_BENCHMARKS.md#run-the-benchmark)）。

**可选：整个模型都放得进内存时，让单卡更快。** 这是 RubenTorney 提供的两个需要手动开启的设置（#1904、#1905）。
当你只用一张显卡，并且内存装得下所有专家时，它们会有帮助，例如在 64 GB 的电脑上跑 Q2_0 或 IQ3_XXS。打开 Strata 数据文件夹里的
`strata-<model>.json`（[位置](docs/INSTALL.md#where-things-are-stored)），然后添加：

- 在 `"args"` 列表中加入 `"--adapt-async", "1"`。这样专家缓存会在辅助线程上更新，不会让回答暂停。在 RTX 5070
  （Windows，Q2_0）上测得：解码快 4-5%，代码和长文档快 7-10%。回答可能与默认设置略有不同，但每次运行的结果保持一致。
- 在 `"env"` 块中加入 `"STRATA_PIN_BY_NEED": "1"`（Windows）。Windows 只能为显卡锁定大约一半的内存，
  这个设置会优先锁定显卡最先需要的专家。在 RTX 5070 上测得：读取提示词快 4.4%。

```json
"args": [ ..., "--adapt-async", "1" ],
"env": { "STRATA_PIN_BY_NEED": "1" }
```

修改后请重启 Strata。要撤销，删除这两项即可。

</details>

## 工作原理

这类模型通常运行在有几百 GB 显存的服务器上。你的显卡只有 12-24 GB。Strata 把工作分给整台电脑，让模型装得下。
可以把它想成一座图书馆。

- 模型是一个由 24,576 位小专家（“experts”）组成的团队，好比 24,576 本参考书。每个词只需要其中 10 本。
- **你的显卡是书桌。** 它小而快。最常用的大约 4,000 本书摊开放在上面。
- **你的内存是阅览室的书架。** 24,576 本书都在这里。某个词需要一本不在书桌上的书时，处理器会立刻从书架上取来读，显卡则继续工作。
- **你的固态硬盘是档案室。** 里面存着一张 29 GB 的查找表。模型每个词只读其中几行，所以不用常驻内存。

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="模型的 24,576 位专家：最忙的在显卡上，全部在内存里，查找表在固态硬盘上"></p>

另外有一个小助手会猜接下来的几个词。大模型一次检查所有猜测，保留正确的部分。得到的答案不变，但快了 1.6-1.8 倍。
长文本按大块读取，一次最多 8,192 个 token，速度超过每秒 1,000 个 token。

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="小助手猜接下来的几个词；大模型一次检查全部猜测，保留正确的部分"></p>

更详细的解释：[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)。每个部分和它的数据：
[详细说明](docs/DETAILS.md#how-it-works)和[论文](docs/paper/Strata-Paper.pdf)。

## 出问题了？

- **第一次启动 Strata 时我的电脑卡住了。** 它在把模型加载到内存。请等待，不要关闭窗口。10 分钟后还是卡住？重启电脑，
  关闭其他程序后再试，或者选一个更小的尺寸。
- **下载或安装时中断了。** 重新启动应用或 `strata launch`。它会从中断的地方继续。
- **非常慢，或者提示“the engine stopped unexpectedly”。** 你的电脑可用内存不够。关闭其他程序（浏览器很占内存），
  或者选一个更小的尺寸，比如 Q2_0 或 IQ2_XS。
- **提示这台电脑没有可用的引擎。** 在应用里打开**设置，引擎**，点击**更新引擎**。在终端里运行 `strata update`。
- **提示 8080 端口已被占用。** Strata 已经在运行了。找一下它的窗口。

更多解决办法：[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)。还是解决不了？请提交
[issue](https://github.com/Niko1221/Strata/issues) 并附上日志：在应用里是**设置，引擎，诊断，保存日志**，在终端里是
`strata doctor` 的输出。如果是安全问题，请私下报告：[SECURITY.md](SECURITY.md)。

## 致谢与许可证

模型是 Qwen 团队的 [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)。压缩工作由
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)、UkisAI（Swift 1.5）、agentionai（Gyro）
和 Unsloth 完成。DeepSeek V4.1-Flash 的各个版本也由 ISTA-DASLab 制作。Strata 使用了
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) 的部分代码。全部致谢：
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits)。Strata 以 [MIT 许可证](LICENSE)开源。少数部分和所有模型有各自的许可证
（[具体是哪些](docs/HOW_IT_WORKS.md#license)）。

## 参与贡献

发现了 bug，或者想提交修改或基准测试结果？请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。其中说明了需要附上哪些内容，
方便其他人核实你的工作。

## 支持 Strata

Strata 免费，采用 MIT 许可证。我在能拿到的各种硬件上开发和测试它。如果它在你的电脑上跑得不错，请我喝杯咖啡，能帮我继续测试新显卡、
修复你们反馈的问题。不用有压力。使用它并反馈你的发现，同样很有帮助。

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>

有兴趣赞助 Strata？请通过 [GitHub](https://github.com/Niko1221) 联系我。
