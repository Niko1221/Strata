# Hakua / Strata

[English](README.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [Polski](README.pl.md) · [Português](README.pt-BR.md) · [简体中文](README.zh-CN.md)

在 Windows 上运行本地 AI，支持聊天、智能体任务与带来源的研究。Hakua 是 zapabob 独立维护的 Niko1221/Strata 分支，不代表原项目的推荐或认证。

## 版本与下载

<!-- release-060-candidate -->
当前源码是 0.6.0 集成候选版。应用、Terminal、Control 与原生引擎源码均为 0.6.0。构建、覆盖安装与发布验证仍在进行，版本号并不证明文件已经公开。已验证的旧版 [0.5.9](https://github.com/zapabob/Strata/releases/tag/v0.5.9) 保留。请通过交付记录确认 0.6.0 的实际状态。

[0.6.0](https://github.com/zapabob/Strata/releases/tag/v0.6.0) · [Strata-0.6.0-windows-x64-setup.exe](https://github.com/zapabob/Strata/releases/download/v0.6.0/Strata-0.6.0-windows-x64-setup.exe) · [tar.7z](https://github.com/zapabob/Strata/releases/download/v0.6.0/Strata-0.6.0-windows-x64.tar.7z) · [SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.6.0/SHA256SUMS.txt)

## 安装与更新

请使用本分支的 Windows x64 NSIS 安装程序。按用户安装，内含专用 Node、Python 和 Control，无需系统开发工具。签名的原生 CUDA 引擎是独立组件。tar.7z 应包含完全相同的安装程序、组件、说明与校验文件，不包含模型权重或个人数据。先解压 7z，再解开 tar，核对 BUNDLE-SHA256SUMS.txt。

先完成请求、卸载已加载模型、备份应用数据，再退出应用及托盘图标。安装到原目录，保留模型、会话、记忆、工具、密钥与审批设置。通过使用已有文件的选项导入压缩引擎，选择兼容模型或本地 GGUF 后加载。模型名称本身不能证明引擎兼容性。[安装与回退](docs/HAKUA_INSTALL.md)。

## 保留的分支功能

`/btw` 会启动真实的无工具旁支子智能体，并提供独立停止操作。`/research` 与 `/deepresearch` 收集来源。网页搜索可使用内置 RustXNG Full。查询会发送给已启用的外部提供方，因此搜索并非完全离线。来源数量不能保证回答正确。

工作流技能学习与自动应用是独立开关，新建配置默认都关闭。保留已有选择、可编辑的私有技能、修订历史与撤销。也保留文件操作、限定范围的会话历史与记忆、兼容的 PrismML/TurboQuant 路径、有界专家内存、pack 完整性检查及诊断功能。[技能](docs/WORKFLOW_SKILLS.md) · [搜索](docs/BROWSER_SEARCH.md)。

## 官方集成与兼容性

固定的官方基准为 [09b6038869](https://github.com/Niko1221/Strata/commit/09b603886925994d0d4461857b01321ef3aa431c)，属于官方 0.2.14 / 引擎 0.1.60 代际，包含网络、草稿模型选择、提示词内核、内存与平台修复。分支版本独立编号。官方删除的 DeepSeek 路径作为扩展保留；缺少实际专用运行时必须阻止加载。本 Windows 包不证明原生图像输入、原生 DeepSeek 引擎分发或 Linux/Mac 推理已通过验证。

## 保护条件与测量限制

保留可用 RAM 3 GiB、提交余量 4 GiB、GPU 预留至少 768 MiB 及磁盘空闲下限。Windows RAM 保护同时监测加载和生成。32 GB RAM 与 16 GB VRAM 并不保证所有模型或上下文都能运行。完整上下文推理、一般模型质量及 GPU 并发仍未获得资格确认：releaseQualified=false。安装程序没有 Windows Authenticode 签名，组件与更新源签名是独立机制。[历史测量](docs/HAKUA_RELEASE.md) 保留原版本信息。

## 文档与开发

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

[Niko1221/Strata](https://github.com/Niko1221/Strata) · [Hermes Agent](https://github.com/NousResearch/hermes-agent) · [第三方源码与声明](third_party/)
