# 在 Apple Silicon(Metal)上运行 hy4-flash

[English](README_MAC-Metal.md) · [NVIDIA](README.zh-CN.md)

这是 Metal 后端:一个为 Apple Silicon 写的原生 hy4-flash 引擎,不是 CUDA 版的移植。它包含架构支持、一套 Metal kernel 与融合、MTP 投机解码、图像输入,以及可由 `strata launch` 直接驱动的服务入口。解码输出与参考实现**逐位相同**;prefill 使用 simdgroup MMA 与 half 操作数,由独立的数值闸门验收。

## 模型

与 CUDA 引擎用同一批 GGUF 文件和 pack。Apple Silicon 上额外识别包含 IQ2_XXS 的专家量化。

| | |
|---|---|
| 模型 | GGUF 第一个分片,`--model-path` |
| pack | tokenizer、chat template、stop id、按引擎布局排好的 dense 权重,`--pack-path` |
| 视觉分片 | `hy_v4_flash` projector 的 mmproj GGUF,放在模型旁边或用 `STRATA_HY4_FLASH_VISION_SHARD` 指定 |

pack **不是**构建产物。要么随模型一起获得,要么从分片生成:

```sh
python terminal/tools/strata_pack.py build  --gguf <第一个分片> --out <pack 目录>
python terminal/tools/strata_pack.py verify --gguf <第一个分片> --out <pack 目录>
```

`verify` 会把写好的 pack 重新读一遍,要求每个张量都满足 `decode(pack) == dequant_ggml(source)` 逐位相同。没通过它的 pack 不应该拿来用。

## 硬件与内存

Apple Silicon,统一内存;实测环境为 64 GB 的 M4 Pro。请用交流电运行 —— 电池模式下 GPU 会降频。模型放本地 SSD:专家权重按需从磁盘读取,外置盘会成为瓶颈。

IQ2 模型的内存占用,单位 GB:

| 组成 | 大小 | 说明 |
|---|---|---|
| dense 权重 | ~10.8 | 常驻 |
| embedding | ~0.7 | 常驻 |
| 专家权重(全量) | ~53.4 | 在磁盘上,按路由命中进 arena |
| 专家 arena | 计算得出 | LRU + 跨层预读 |

arena 不是一个固定的默认值,而是四项取最小:GPU 工作集、必须留给 macOS 的 RAM、专家权重总字节,以及一个保守上限 —— 引擎会打印是哪一项卡住的。不需要手动抬高任何 GPU 内存上限。

**不要手工固定 arena 大小。** 设定 `STRATA_METAL_ARENA_GB` 会取代整个计划并关掉运行中的收缩/增长守卫,在内存更小的机器上得到的是 OOM 被杀而不是一个更小的 arena。内存不足时先让引擎自行收缩,再考虑降低上下文长度。

## 构建

唯一的前置依赖是 metal-cpp 头文件:从 <https://developer.apple.com/metal/cpp/> 下载对应 macOS 版本的压缩包并解压,目录里必须有 `Metal/Metal.hpp`。

```sh
cmake -S engine/metal -B build-metal -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_METAL_ENGINE=ON \
  -DMETAL_CPP_DIR=$PWD/metal-cpp
cmake --build build-metal -j
```

其中两个是入口:`strata`(服务,供 `strata launch` 使用)与 `strata_hy4_flash_metal`(离线单次推理与测速驱动器)。首次构建需要已接受的 Xcode 许可。

## 启动与请求

```sh
strata launch --no-tui --yes \
  --model-path <第一个分片> \
  --pack-path <pack 目录> \
  --engine-path build-metal/strata \
  --port 8099
```

| 参数 | 说明 |
|---|---|
| `--model-path` | GGUF 第一个分片 |
| `--pack-path` | pack 目录 |
| `--engine-path` | 上面构建出的 Metal `strata` |
| `--port` | HTTP 监听端口 |

启动后 `/v1/models` 会把模型列为 `loaded`,`/v1/chat/completions` 接受标准请求。

注意 hy4-flash **默认开启思考**:未指定 `reasoning_effort` 时,正文先进入 `reasoning_content`,`content` 在推理块闭合前为 `null`。只读 `content` 的客户端在那之前会什么都看不到。

### 图像

加 `--vision on`。视觉分片默认在模型旁边寻找,也可以直接指定:

```sh
strata launch --no-tui --yes --vision on \
  --model-path <第一个分片> --pack-path <pack 目录> \
  --engine-path build-metal/strata --context 32k --port 8099
```

图片按内容块里的 `image_url` 配 `data:` URL 发送。一张 `h × w` 的图(两边都取到 32 的倍数)占 `(h/32) · (w/32 + 1)` 个位置 —— 那个 `+1` 是每个 merge 行一条 image-newline 行 —— 所以大图会吃掉上千个上下文位置,`--context` 要给够。

几何不是推测出来的,而是钉在 HuggingFace 的图像处理器上:目标尺寸、patch 网格、以及主干的三轴 rotary 位置,各自都按整数与它比对。畸形的图片包会被**点名拒绝**,而不是被解码成噪声。

无论是否构建视觉路径,纯文本 decode **逐位相同**,所以打开图像不会让纯文本请求付出任何代价。

### 投机解码

MTP 默认开启,每轮起草 2 个 token,由主干一次 verify。它不改变输出:被接受的 token 与普通解码会产出的完全一致。

## 启动选项

`strata launch` 在 Apple Silicon 上会给出合适的值,通常一个都不需要手动指定。

| 开关 | 默认 | 说明 |
|---|---|---|
| `--vision` | off | `on` 打开图像 |
| `--prefill-chunk` | Mac 上 4096 | 分块 prefill 的块大小 |
| `--draft-tokens` | 2 | MTP 每轮起草的 token 数 |
| `--context` | 模型默认 | 发大图之前要调高 |

### 引擎设置

| 变量 | 默认 | 说明 |
|---|---|---|
| `STRATA_METAL_ARENA_GB` | 不设 | 固定 arena 并关掉守卫,见上文 |
| `STRATA_HY4_FLASH_VISION_SHARD` | 不设 | mmproj GGUF 的路径 |
| `HY4_FLASH_MTP_EXPERT_GB` | 引擎默认 | 起草阶段的专家预算 |

非 Darwin 平台的行为不受本后端影响。

## 实测结果

Mac mini,M4 Pro(16 核 GPU),64 GB 统一内存;IQ2 模型,MTP k=2,8K 上下文。

| 指标 | 起点 | 当前 |
|---|---|---|
| prefill | 35.8 tok/s | **125–127 tok/s** |
| decode(生成 256) | 7.31 tok/s | **17.300 tok/s** |
| decode(生成 1024) | — | 18.979 tok/s |

decode 取四条 prompt 各自中位数的几何平均。生成越长 MTP 接受率越高,所以 1024 的数字更好看;256 是协议规定的口径。

decode 的上限是结构性的:每个 token 约读 11.7 GB 权重,其中 86% 是 dense,而 arena 只能缓存专家那 14%。在实测 252 GB/s 下,不开 MTP 的上限约 21.6 tok/s —— MTP 把 dense 的读取摊到多行 verify 上,才把天花板抬上去。
