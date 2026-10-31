<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">在你自己的电脑上运行强大的 AI。</h3>
<p align="center">本地聊天、写代码、处理图片。</p>

<p align="center">
  <a href="#模型">模型</a> · <a href="#安装"><b>安装</b></a> ·
  <a href="#硬件">硬件</a> · <a href="#工具">工具</a> ·
  <a href="#帮助">帮助</a> · <a href="#支持-strata">支持</a>
</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=download&color=0ab887" alt="下载 Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT 许可证"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="在 Buy Me a Coffee 上支持 Strata"></a>
</p>

<p align="center"><sub><a href="README.md">English</a> · <b>简体中文</b> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.pl.md">Polski</a></sub></p>

**免费、开源，由你自己运行。** 在桌面上使用 Strata，或搭配你喜欢的编程工具。无需账号。

> **个人分支 (c-OOL-io/Strata-WebSearch)。** 此分支为内置聊天与 serve 前端加入了**网页搜索与完整网页读取**: 轻量的终端 Web 应用, 上游在这里没有搜索功能 (上游的搜索在 App 中)。它按 Strata 聊天本身的方式构建: **默认完全关闭** (开启之前, 任何搜索相关内容都不会离开电脑), 未开启搜索的聊天**不增加任何 token**, 搜索文本被限制在模型上下文窗口的一小部分 (默认值随 `max_context` 缩放, 小上下文实例依然安全)。
>
> - **搜索引擎:** 默认 DuckDuckGo (无需密钥), SearXNG (自建实例), Brave, Tavily, Serper (Google), Exa 和 Perplexity 使用你自己的密钥。密钥只留在浏览器; 服务器上不再存储任何搜索相关内容。
> - **查询优化器:** 口语化的表述由本地模型的一次快速调用改写成干净的搜索查询, 并保持你写作时使用的语言。失败时改搜原始消息。若实际搜索的内容与你输入的不同, Sources 行会显示出来。
> - **阅读结果页面:** 抓取排名靠前的结果页面并以提取出的正文注入 (去除导航/页脚/脚本, 优先 `main`/`article`), 提供 Direct / Jina 回退 / Jina 全部三档阅读模式, 第三方可看到你的 URL 时会有显式警告。
> - **可核查的回答:** 每条回答带有 `Sources` 行 (引擎、实际查询、可点击链接、`read` 徽标), 并随 Markdown 导出一起保存; 搜索失败会留下具名原因警告 (缺少密钥、引擎不可达、bot 检查), 而不是静默失败。
> - **安全加固:** `/search` 位于应用自有页面防护之后 (拒绝外部 Origin), 仅 http(s) 的 URL 检查确保网页读取永远不会变成本地文件读取器, 注入的网页文本被标记为不可信, 后端 (`serve/websearch.py`) 仅用标准库, 无新依赖, 支持 Windows/Linux/macOS。
> - **本次更新的新功能:** 输入框的地球图标现在循环切换 **Off → On → Auto** (Auto 让模型逐条判断, 失败时回退到关键词启发式); 可选的**仅浏览器・消息框草稿** (默认关闭) 在刷新和切换聊天后仍会保留; 以及可在聊天设置中选择的**14 种语言界面** (de, es, fr, ja, ko, pt, ru, zh, hi, ar 为 RTL, tr, it, pl)。语言列表将英语排在首位, 然后是浏览器语言(如支持), 其余按字母顺序排列, 每项均显示其两位字母代码。
>

> 下载前需知两点:
>
> 1. **下方的下载链接是 Niko1221/Strata 的原版应用。** 预编译二进制**不包含**此分支的搜索功能。带搜索的版本只存在于源码中:`git clone`,分支 `web-search`,然后 START-HERE.bat / setup.sh。
> 2. **此分支也适合精简的 Terminal 路线**:只需要 headless 服务器上的轻量 serve/CLI (无应用、无聊天界面) 的人,用同一份源码即可,搜索内置于 serve 前端。
>
> **通过 CLI 安装**（此分支，`web-search` 分支）：克隆并按 Strata 源码方式安装，搜索已内置。本 Fork
> 的完整安装指南（终端 / 源码，含更新）见 [docs/INSTALL.FORK.md](docs/INSTALL.FORK.md)：
>
> ```sh
> git clone -b web-search https://github.com/c-OOL-io/Strata-WebSearch.git && cd Strata-WebSearch
> ./setup.sh          # first run: .venv, the engine build, the model; then it serves
> ```
>
> RTX 4080 SUPER / Ryzen 9 7950X3D 的基准测试结果在 `bench/`。

<p align="center">
  <img src="docs/media/web-search-chat.png" width="360" alt="Strata 聊天中的网页搜索">
  <img src="docs/media/web-search-settings.png" width="360" alt="设置中的网页搜索:引擎与密钥">
</p>
<p align="center"><sub>左：内置聊天的回答与回复下方的 Sources 行（所用引擎与实际检索词），可在输入区用地球图标开关。<br>右：设置中的 Web search 区：开关、查询优化、引擎选择（DuckDuckGo / SearXNG / Brave / Tavily / Serper / Exa / Perplexity）、密钥输入、网页读取。</sub></p>

## 模型

我们**首先推荐 ISTA-DASLab**，其次是 **Unsloth** 和其他提供方，前提是有内存放得下的兼容模型。

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

| 提供方 | 可选项 |
| --- | --- |
| **ISTA-DASLab · 推荐** | Q2\_0, IQ2\_XS, IQ3\_XXS, IQ3\_S。[Coder：IQ1\_M](docs/MODELS.md#coder)，适合写代码。 |
| **Unsloth** | UD-IQ4\_XS, UD-Q4\_K\_XL, UD-Q5\_K\_XL, UD-Q6\_K\_XL, Q8\_0 |
| **Gyro · agentionai** | [Gyro-S, Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF)。占用内存更少，但速度更慢。 |
| **其他选项** | [Swift 1.5](docs/MODELS.md#swift-15), [NVFP4](docs/NVFP4.md), [OrcaRouter](docs/ORCA.md) |

[比较大小、内存需求和兼容性](docs/MODELS.md)。

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.6-35B-A3B</b></summary>

**Unsloth:** UD-Q4_K_XL, UD-Q6_K_XL, Q8_0。这些是目前文档中收录的该模型的版本。

[模型指南和兼容性](docs/QWEN36.md)。

</details>

<details>
<summary><img src="docs/media/models/deepseek.svg" width="22" alt="" align="middle"> &nbsp;<b>DeepSeekV4.1-Flash-1017</b></summary>

可在**桌面应用**和**终端**中使用。内存放不下的部分会从 SSD 读取，所以需要大量空闲磁盘空间。

| 版本（IST-DAS） | 下载大小 | 所需空闲磁盘 |
| --- | ---: | ---: |
| 1.75 bpw | 239 GB | 约245 GB |
| 2.00 bpw | 257 GB | 约265 GB |
| 2.25 bpw | 274 GB | 约280 GB |

支持 NVIDIA RTX 30、40 和 50 系列（Windows 和 Linux），以及 AMD Radeon AI PRO R9700（Linux）。安装程序会下载模型文件和 DeepSeek 引擎，然后启动模型。

```sh
strata launch --family deepseek41 --model ist-das-2.00bpw --plan
```

以下为 Strata 0.2.5 上 2.00 bpw 版本的实测结果。“输出”是回答的速度，“提示词”是 Strata 读取你输入内容的速度。

| 硬件 | 输出 | 提示词 |
| --- | ---: | ---: |
| RTX 5070 12 GB, 64 GB 内存 | 4 到 6 tokens/s | 约 110 tokens/s（4K） |
| RTX 3060 12 GB, 46 GB 内存 | 9 到 11 tokens/s | 约 155 tokens/s（4K） |
| 1x Radeon AI PRO R9700, 251 GB 内存 | 21 到 25 tokens/s | 320 到 360 tokens/s (8K) |
| 4x Radeon AI PRO R9700 | 74 到 77 tokens/s | 400 到 450 tokens/s (8K) |

[内存、磁盘和详细信息](docs/DEEPSEEK41.md) · [发行说明](https://github.com/Niko1221/Strata/releases)

</details>

<a id="the-other-sizes"></a>

不确定哪个放得下？[为你的电脑选择模型](docs/MODELS.md#pick-by-memory)。

## 安装

<a id="option-a-the-strata-app"></a>

### 桌面应用

**[下载适用于 Windows、Linux 或 Mac 的 Strata →](https://github.com/Niko1221/Strata/releases/latest)**

安装应用，选择模型，就可以开始聊天。每个系统只有一个安装程序。首次启动时，应用会下载适合你显卡的引擎。[安装程序帮助](docs/INSTALL.md#the-app)。

<sub>文件: `Strata-0.2.9-windows-x64-setup.exe` · `Strata-0.2.9-linux-x86_64.AppImage` · `Strata-0.2.9-macos-arm64.dmg`</sub>

<p align="center">
  <img src="docs/media/app-chat.png" width="900" alt="Strata 桌面应用，聊天和实时代码预览并排显示">
  <br><sub>聊天、文件和实时预览都在一个窗口里。</sub>
</p>

<a id="option-b-the-terminal-one-line"></a>

### 终端应用

更喜欢终端？安装 Strata，然后运行 **`strata launch`**。

<details open>
<summary><b>安装命令</b></summary>

**Linux / Mac**

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

在 Windows 上，安装后请打开一个新的终端。然后运行：

```sh
strata launch
```

按提示下载并启动模型。[终端指南](docs/TERMINAL_LAUNCH.md)。

</details>

<p align="center">
  <img src="docs/media/terminal-monitor.png" width="900" alt="Strata 终端应用，显示模型活动、速度、内存和最近的请求">
  <br><sub>模型运行时，随时查看模型和硬件状态。</sub>
</p>

<a id="build-from-source"></a>

[从源码构建](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md) · [完整安装指南](docs/INSTALL.md)

<a id="what-you-need"></a>

## 硬件

| 支持的硬件 | 指南 |
| --- | --- |
| **NVIDIA GeForce RTX 20 / 30 / 40 / 50 系列** | [NVIDIA 安装](docs/INSTALL.md#what-you-need) |
| **AMD Radeon RX 7700 / 7800 / 7900、RX 9060 / 9070、AI PRO R9700** | [AMD 安装](docs/AMD_HIP.md) |
| **AMD Radeon RX 6800 / 6900 · 实验性** | [RX 6000 安装](docs/AMD_HIP.md#rdna2-gfx1030) |
| **Intel Arc · Linux** | [Intel 安装](docs/INTEL_ARC.md) |
| **Apple Silicon · M1 或更新** | [Mac 安装](docs/MAC.md) |
| **AMD Ryzen AI Max · Strix Halo** | [Strix Halo 安装](docs/STRIX_HALO.md) |

<a id="using-it"></a>

## 工具

**Claude Code · Codex CLI · OpenCode · oh-my-pi · pi.dev · Hermes Agent · OpenClaw**，也可以通过兼容 OpenAI 和 Anthropic 的 API 使用更多工具。

运行 **`strata connect`**，查看你所用工具的设置方法。[连接指南](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect)。

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## 帮助

[全部指南](docs) · [故障排查](docs/TROUBLESHOOTING.md) · [报告问题](https://github.com/Niko1221/Strata/issues/new/choose) ·
[基准测试](docs/COMMUNITY_BENCHMARKS.md) · [工作原理](docs/HOW_IT_WORKS.md) · [参与贡献](CONTRIBUTING.md)

## 支持 Strata

**帮助本地 AI 保持开放。** 一杯咖啡能帮助我们在更多硬件上开发和测试。

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee：支持 Strata" height="48"></a>

[MIT 许可证](LICENSE) · [致谢和模型许可证](docs/HOW_IT_WORKS.md#credits) · [隐私](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>面向 AI 工具和自动阅读器： <a href="AGENTS.md">技术参考</a> · <a href="docs/AI_SETUP.md">安装说明</a></sub>

**赞助、硬件合作或其他合作：** [niko.veit@outlook.com](mailto:niko.veit@outlook.com).
