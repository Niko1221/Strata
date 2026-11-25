# 在单张 NVIDIA GPU 上运行 hy4-flash

[English](README.md) · [Apple Silicon(Metal)](README_MAC-Metal.zh-CN.md)

本文说明如何使用 Strata 在单张 NVIDIA GPU 上部署与运行 `HY4-Flash`。

## 1. 适用模型（GGUF）

| 模型版本                                | 文件                                                                                                              | 说明                         |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------- |
| `Hy4-flash-Q4_K_M_MTP_Q8_0`         | `Hy4-flash-Q4_K_M_MTP_Q8_0-00001-of-00002.gguf` `Hy4-flash-Q4_K_M_MTP_Q8_0-00002-of-00002.gguf`                 | 完整专家模型，含内置 MTP            |
| `Hy4-flash-Q4_K_M_MTP_Q8_0_Prune25` | `Hy4-flash-Q4_K_M_MTP_Q8_0_Prune25-00001-of-00002.gguf` `Hy4-flash-Q4_K_M_MTP_Q8_0_Prune25-00002-of-00002.gguf` | 专家剪枝 25%，适用于系统内存（RAM）不足的机器 |

## 2. 测试环境

- **软件环境**
  - 系统：Linux x86_64
  - CUDA Toolkit 12.8（本指南使用的版本）。
  - 构建与运行依赖：CMake >= 3.24、C++20 编译器、Ninja、Python >= 3.10
- **硬件与内存要求**
  - 编译示例面向 NVIDIA GeForce RTX 4090D（SM 89）。
  - **建议使用至少 16 GiB 显存的显卡。** 以完整版、64K 上下文上限和 Q8_0 KV 为例，启动器为常驻权重、KV 缓存和工作区预留约 12 GiB，之后才分配显存专家池。16 GiB 显卡建议从 `CTX=32768` 起步，生成时可启用 DFly（下文实测，相对默认关闭投机解码，正文解码快约 5%，代码快约 27%）。
  - 调整上下文上限或启用 MTP、DFly 后，在目标显卡执行 `--plan` 查看当前可用显存下的专家池预算。第 2 分片的 Engram 表按需从文件读取，访问过的页可能留在系统页缓存中。
  - 默认模式在主存保留全量主干专家副本（完整版约 83.4 GiB），显存缓存活跃专家。主存还要容纳引擎和系统开销，建议预留至少 128 GiB 可用主存。
  - 内存较紧时可加 `--resident-experts`，用显存和主存拆分存放主干专家。对于完整版，用 83.4 GiB 减去 `--plan` 显示的显存专家池大小，即可估算剩余专家权重所需的主存。换槽、中转缓冲及其他进程开销另计。

## 3. 编译说明

在仓库根目录执行以下命令。

```bash
# 1. 编译引擎（RTX 4090D 对应 CMAKE_CUDA_ARCHITECTURES=89，可替换为实际设备架构号或 native）
cmake -S engine -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -G Ninja
cmake --build build --target strata --parallel 16

# 2. 安装 Python 运行依赖
python3 -m venv .venv
.venv/bin/python -m pip install -r terminal/requirements.txt -r terminal/requirements-monitor.txt
```

- CUDA 构建使用 `engine/third_party` 随仓库提供的 GGML 源码。运行时将 `libstrata_kq.so` 和 `strata` 放在同一目录，供 K 量化预填充使用。离线安装 Python 依赖需要本地 wheel。
- **终端面板**：`requirements-monitor.txt` 安装 Textual。只使用后台 HTTP 服务时可以只装 `requirements.txt`，启动时追加 `--no-tui`。
- **图片输入编译支持（可选）**：hy4-flash CUDA 构建默认包含读取 mmproj GGUF 的视觉编码器。HTTP 图片输入所需的 Pillow 已列在 `requirements.txt` 中。运行时保留 `build/strata` 旁的 `libstrata_kq.so`，供量化预填充按需加载。

## 4. 使用说明

启动器根据主模型文件选择引擎。hy4-flash GGUF 使用 `strata`，原生 SQ 的 `.strata` 模型使用 `strata-hy4`。以下以 GGUF 为例。

按上节编译后，启动器会自动找到 `build/strata`。在同一个终端设置一次 `MODEL`，指向模型目录或第一个 GGUF 分片，即可预览配置和启动服务。

```bash
export MODEL=/path/to/Hy4-flash-Q4_K_M_MTP_Q8_0
./tools/hy4_flash/launch-hy4-flash.sh --plan
./tools/hy4_flash/launch-hy4-flash.sh
```

首次启动会在模型目录生成 `pack/`。如需使用其他可写目录，可设置 `PACK`。

服务就绪后，可在另一个终端测试 OpenAI 兼容接口：

```bash
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"hy4-flash","messages":[{"role":"user","content":"What is two plus two?"}],"temperature":0,"reasoning_effort":"none","max_tokens":20}'
```

后台运行可在启动命令后追加 `--no-tui`。停止默认服务使用 `.venv/bin/python terminal/strata.py launch --stop --preset hy4-flash`。若设置过 `PRESET` 或 `PORT`，相应替换停止命令中的预设名或请求 URL 中的端口。

- **参数自适应与覆盖**：启动时脚本会根据当前系统与可用显存规划运行参数（专家池预算、MTP 草稿步数、Prefill 分块及 CPU 专家分流等），也可按需手动覆盖。规划结果是启动估算；各项显存实际占用随请求变化。

### 投机解码：内置 MTP 或外挂 DFly

默认由启动器根据主模型和可用显存决定是否启用内置 MTP。要启用 DFly，先从 Hugging Face 下载[已量化的 hy4-flash DFly Q8_0 草稿模型](https://huggingface.co/AngelSlim/Hy4-flash-GGUF/blob/main/hy4-flash_dfly_Q8_0.gguf)，放在主模型目录之外。`MODEL` 继续指向前面设置的主模型。

```bash
./tools/hy4_flash/launch-hy4-flash.sh --dfly-model /path/to/drafts/hy4-flash_dfly_Q8_0.gguf
```

在 16 GiB 显存预算、33,024 token 上下文上限下，启动器为 Q8_0 DFly 草稿预留约 1 GiB，给 GPU 专家池分配了 4.90 GiB。显式开启 MTP 时，草稿专家放在主存，GPU 主干专家池为 4.22 GiB。使用完整版，正文和代码各测一次，均输入 4,096 token、生成 512 token。相对默认关闭投机解码，DFly 的正文解码从 36.8 提高到 38.7 tok/s，代码从 37.5 提高到 47.6 tok/s。

传入 `--dfly-model` 即可选用 DFly。默认每轮最多猜 3 个 token，可加 `--draft-tokens 4` 尝试猜 4 个，或加 `--plan` 预览配置。

### 常用启动参数（`VAR=val ./tools/hy4_flash/launch-hy4-flash.sh [flags]`）

| 参数 / 变量 | 默认值 | 说明 |
|---|---|---|
| `MODEL` | 必填 | 模型目录路径，或第 1 个 `.gguf` 分片路径 |
| `STRATA_BIN` | 自动查找 | 可选。引擎编译在其他目录或需要指定版本时，填写 GGUF 引擎的可执行文件路径。 |
| `PACK` | `<模型目录>/pack` | 预处理目录路径（不存在时首次启动自动生成） |
| `GPU` / `PORT` / `PRESET` | `0` / `8080` / `hy4-flash` | 单卡 GPU 编号、监听端口、服务预设名 |
| `CTX` | `196608` | 最大上下文长度 |
| `EXPERT_VRAM` | 自动推算 | GPU 主干专家池预算（GiB，不含稠密层/KV/MTP），以当前上下文和草稿配置的 `--plan` 输出为准 |
| `MTP=on/off` 或 `--mtp on/off` | 自动 | 控制主模型内置的 MTP。省略时按模型支持情况和可用显存决定是否开启。 |
| `DRAFTING` 或 `--drafting` | `auto` | 可选 `auto`、`mtp`、`dfly`、`off`。用 `off` 关闭投机解码；MTP 与 DFly 二选一。 |
| `DFLY_MODEL` 或 `--dfly-model PATH` | 未指定 | 指定与主模型匹配的外挂 DFly Q8_0 GGUF。单独传此参数即可选中 DFly，`MODEL` 仍指向主模型。 |
| `DRAFT_TOKENS=N` 或 `--draft-tokens N` | MTP 自动；DFly 为 `3` | 每轮最多猜的 token 数。MTP 可设 1～7，自动开启时完整版通常猜 3 个，Prune25 猜 1 个。DFly 可设 1～5，例如 `--dfly-model /path/to/dfly-Q8_0.gguf --draft-tokens 4`。 |
| `KV` | `int8` | KV Cache 精度：`int8`（`Q8_0`）或 `fp16` |
| `CPU_MISS_MAX` | `16` | 每层未命中专家交由 CPU 计算的上限（`0` 关闭 CPU 专家分流） |
| `--resident-experts` | 关闭 | 开启显存与系统内存互斥拆分驻留（内存受限时使用） |
| `--vision on/off` | `off` | 开启后，`/v1/chat/completions` 可接收 `image_url`。含图请求由主模型直接解码，同一服务的纯文本请求按所选 MTP 或 DFly 策略运行。 |
| `STRATA_HY4_FLASH_VISION_SHARD` | 开启视觉时自动查找 | 模型目录只有一个 `mmproj*.gguf` 时自动选用。文件在其他目录或有多个候选时，用此变量指定完整文件路径，例如 `STRATA_HY4_FLASH_VISION_SHARD=/path/to/mmproj-Hy4-flash-bf16.gguf`，并传 `--vision on`。 |
| `--set KEY=VAL` | — | 手动覆盖底层环境变量或高级参数，优先级最高 |

### 常用底层开关（默认已自动配置，可通过 `--set KEY=VAL` 覆盖）

| 环境变量 | 默认行为 | 说明 |
|---|---|---|
| `STRATA_HY4_FLASH_CPU_WORKERS` | `32` | CPU 专家计算线程数 |
| `STRATA_HY4_FLASH_PREFILL_CHUNK` | 按显存自动适配 | Prefill 计算分块大小（如 `512` / `1024` / `2048` / `4096` 或 `auto`） |
| `STRATA_HY4_FLASH_PREFILL_BORROW` | 按配置自动选择 | `1` 允许预填充临时借用主干专家池的显存，`0` 关闭。拆分驻留模式（`--resident-experts`）不执行借还。 |
| `STRATA_HY4_FLASH_PREFILL_LAYER_MAJOR` | `1` | 长提示按层预填充（12 GiB 等极限显存跑长文本时可设为 `0`） |
| `STRATA_HY4_FLASH_PREFILL_GROUP_ROWS` | `1024` | 单专家预填充行容量上限（超长提示若报 `exceeds row capacity` 可调至 `2048`） |
| `STRATA_HY4_FLASH_MTP_HOST_EXPERTS` | 自动（小显存置 `1`） | 将 MTP 草稿层专家映射至系统内存，可节省约 3.6 GiB 显存 |
| `STRATA_HY4_FLASH_ENGRAM_IO` | `mmap` | Engram 表读取方式：`mmap`（页缓存读取）或 `direct`（`O_DIRECT`） |

在支持借池的单卡普通模式下，启动器会综合可用显存、分块大小和专家池预算决定是否开启。开启时，规划器可把原本为大分块单独留出的预算计入解码专家池。预填充需要工作区时，引擎临时减少 GPU 专家槽位；切回解码时恢复池容量并回填专家。KV cache、稠密权重和草稿模型仍有各自的预算。

在平时的启动命令后加 `--plan --output json`，查看 `plan.hy4_flash_prefill.borrow` 和 `chunk`。要对比关闭后的预算，再加 `--set STRATA_HY4_FLASH_PREFILL_BORROW=0`。实际运行时，引擎日志的 `HY4_FLASH_PREFILL_LOAN phase=prefill` 和 `phase=decode` 分别记录借用和归还；其中 `scratch` 以字节计。
