# Running hy4-flash on Apple Silicon (Metal)

This is Strata's Metal backend: a native hy4-flash engine rewritten for Apple Silicon, not a patch on llama.cpp. It brings architecture support, a set of Metal kernels and fusions, MTP speculative decoding, pictures, and a serve entry point that `strata launch` drives directly. Decode output is **bit-for-bit identical** to the reference; prefill uses simdgroup MMA with half operands and is accepted by a separate numerical gate.

## 0. Decoding modes

| Mode | What it does | Speed | Pictures |
|---|---|---|---|
| Autoregressive | No MTP, one token per round | Slowest | Yes |
| **MTP (default)** | The MTP head drafts k tokens, the trunk verifies them in one pass | Fastest | Yes |
| DFly | Upstream's other drafting scheme | Not wired up | — |

Pictures work. Turn them on with `--vision on`; the vision shard is looked for beside the model, or pointed at with `STRATA_HY4_FLASH_VISION_SHARD`. Text-only decode is bit-for-bit unchanged whether or not the vision path is built, so a picture costs a text request nothing.

## 1. Build

One dependency has to be on disk first: **metal-cpp**, Apple's C++ headers for Metal. Download the archive for your macOS version from <https://developer.apple.com/metal/cpp/> and unzip it; the directory you point at must contain `Metal/Metal.hpp`. (The headers also ship inside MLX, at `<site-packages>/mlx/include/metal_cpp`, if you already have that installed.)

```sh
cmake -S engine/metal -B build-metal -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_METAL_ENGINE=ON \
  -DMETAL_CPP_DIR=$PWD/metal-cpp
cmake --build build-metal -j
```

ggml comes from the vendored `engine/third_party/strata-ggml` at the commit this repository pins, so nothing has to be fetched — the engine links `ggml-cpu` and `ggml-base` for the native expert formats' type traits and the CPU-side dot products. If you would rather build against your own llama.cpp checkout at that commit, point `-DSTRATA_GGML_DIR=` at it.

Two of the resulting binaries are entry points: `strata` (the server, used by `strata launch`) and `strata_hy4_flash_metal` (an offline single-shot and benchmarking driver). The first build needs an accepted Xcode licence.

## 1a. The pack

The build does **not** produce a pack. It is a separate artifact beside the GGUF, holding the tokenizer and chat template, the stop ids, the dense weights in the engine's own layout, and an expert profile. Either `setup.py` fetches it with the model, or you build it from the shards:

```sh
python terminal/tools/strata_pack.py build  --gguf Hy4-flash-IQ2_AH-00001-of-00002.gguf --out ./pack
python terminal/tools/strata_pack.py verify --gguf Hy4-flash-IQ2_AH-00001-of-00002.gguf --out ./pack
```

`verify` is not optional: it re-reads the written pack from disk and requires `decode(pack) == dequant_ggml(source)` bit for bit, for every tensor. A pack that has not passed it should not be used.

## 2. System setup

- Run on AC power. On battery the GPU clocks down and timings are not comparable.
- Keep the model on local SSD. Expert weights are read from disk on demand, so an external drive becomes the bottleneck.
- Close other GPU-heavy processes before measuring. macOS compresses memory during the arena's fill phase, which can get an otherwise-workable configuration killed.
- You do not need to raise the GPU memory limit by hand. The engine sizes its expert arena from available memory and shrinks or grows it while running.

## 3. Memory budget

For IQ2_AH, in GB:

| Component | Size | Notes |
|---|---|---|
| Dense weights | ~10.8 | Resident |
| Embeddings | ~0.7 | Resident |
| Expert weights (all) | ~53.4 | On disk; enter the arena when routing hits them |
| Expert arena | computed | LRU plus cross-layer read-ahead |

The arena is **not a fixed default**. It is the smallest of: the GPU working set, the RAM macOS must be left with, the total expert bytes, and a conservative 28 GB cap — and the engine prints which of the four bound it.

## 4. Runtime switches

| Switch | Default | What it does |
|---|---|---|
| `--prefill-chunk` | 4096 on Mac via launch | Chunk size for chunked prefill |
| `--draft-tokens` | 2 | Tokens the MTP head drafts per round |
| `STRATA_METAL_ARENA_GB` | unset | Pins the arena and disables its guards; normally leave alone |
| `HY4_FLASH_MTP_EXPERT_GB` | engine default | Expert budget during drafting |

`strata launch` picks sensible values on Apple Silicon, so you usually need none of these. Behaviour on non-Darwin platforms is unaffected by this backend.

## 5. Start the server

```sh
strata launch --no-tui --yes \
  --model-path Hy4-flash-IQ2_AH-00001-of-00002.gguf \
  --pack-path ./pack \
  --engine-path build-metal/strata \
  --port 8099
```

| Flag | Meaning |
|---|---|
| `--model-path` | First GGUF shard |
| `--pack-path` | Pack directory holding the tokenizer and chat template |
| `--engine-path` | The Metal `strata` built above |
| `--port` | HTTP listen port |

Once up, `/v1/models` lists the model as `loaded` and `/v1/chat/completions` takes ordinary requests. Note that hy4-flash **thinks by default**: with no `reasoning_effort` set, the text goes to `reasoning_content` and `content` stays `null` until the reasoning block closes.

## 5a. Pictures

```sh
strata launch --no-tui --yes --vision on \
  --model-path Hy4-flash-IQ2_AH-00001-of-00002.gguf \
  --pack-path ./pack \
  --engine-path build-metal/strata \
  --context 32k --port 8099
```

The vision shard is looked for beside the model; `STRATA_HY4_FLASH_VISION_SHARD=<path>` points at it directly. Send pictures the usual way, as an `image_url` content part with a `data:` URL.

A picture of `h × w` (both rounded to multiples of 32) occupies `(h/32) · (w/32 + 1)` positions — the `+1` is one image-newline row per merged row — so a large picture is worth thousands of context positions. Give the server enough `--context`.

Geometry is pinned to the HuggingFace image processor rather than inferred: the target size, the patch grid and the trunk's three-axis rotary positions are each checked against it as integers. Malformed picture bundles are refused by name rather than decoded into noise.

Text-only decode is bit-for-bit identical to a build without the vision path — 24 of 24 token ids and 24 of 24 logit rows — so enabling pictures does not perturb text generation.

## 6. Measured results

Mac mini, M4 Pro (16-core GPU), 64 GB unified memory; IQ2_AH, MTP k=2, 8K context.

| Metric | Start | Now |
|---|---|---|
| Prefill | 35.8 tok/s | **125–127 tok/s** |
| Decode (256 generated) | 7.31 tok/s | **17.300 tok/s** |
| Decode (1024 generated) | — | 18.979 tok/s |

The decode ceiling is structural: every token reads about 11.7 GB of weights, 86% of it dense, while the arena can only cache the 14% that is experts. At a measured 252 GB/s that caps plain decoding near 21.6 tok/s — MTP raises the ceiling by amortising the dense read across several verified rows.

---

# 在 Apple Silicon(Metal)上运行 hy4-flash

这是 Strata 的 Metal 后端:一个为 Apple Silicon 重写的原生 hy4-flash 引擎,不是 llama.cpp 的补丁。它包含架构支持、一套 Metal kernel 与融合、MTP 投机解码、图像输入,以及可由 `strata launch` 直接驱动的服务入口。解码输出与参考实现**逐位相同**;prefill 使用 simdgroup MMA 与 half 操作数,由独立的数值闸门验收。

## 0. 解码模式

| 模式 | 说明 | 速度 | 图像 |
|---|---|---|---|
| 自回归 | 不开 MTP,每轮一个 token | 最慢 | 支持 |
| **MTP(默认)** | MTP 头起草 k 个 token,主干一次 verify | 最快 | 支持 |
| DFly | 上游的另一种起草方案 | 未接入 | — |

图像输入可用。用 `--vision on` 打开;视觉分片默认在模型旁边寻找,也可以用 `STRATA_HY4_FLASH_VISION_SHARD` 指定。无论是否构建视觉路径,**纯文本 decode 逐位不变**,所以支持图片不会让纯文本请求付出任何代价。

## 1. 构建

只需要先准备一个依赖:**metal-cpp**,Apple 的 Metal C++ 头文件。从 <https://developer.apple.com/metal/cpp/> 下载对应 macOS 版本的压缩包并解压;指向的目录里必须有 `Metal/Metal.hpp`。(如果已装了 MLX,它也自带一份,在 `<site-packages>/mlx/include/metal_cpp`。)

```sh
cmake -S engine/metal -B build-metal -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_METAL_ENGINE=ON \
  -DMETAL_CPP_DIR=$PWD/metal-cpp
cmake --build build-metal -j
```

ggml 来自仓库内置的 `engine/third_party/strata-ggml`,commit 已钉住,所以不需要额外拉取 —— 引擎链接 `ggml-cpu` 与 `ggml-base`,用的是原生专家格式的 type traits 与 CPU 侧点积。如果你更想用自己那份同 commit 的 llama.cpp checkout,用 `-DSTRATA_GGML_DIR=` 指过去即可。

其中两个是入口:`strata`(服务,供 `strata launch` 使用)与 `strata_hy4_flash_metal`(离线单次推理与测速驱动器)。首次构建需要已接受的 Xcode 许可。

## 1a. pack 是怎么来的

**构建不产生 pack。** 它是 GGUF 旁边一个独立的产物,里面装的是 tokenizer 与 chat template、stop id、按引擎自己布局排好的 dense 权重,以及一份专家 profile。要么由 `setup.py` 随模型一起取下来,要么从分片自己生成:

```sh
python terminal/tools/strata_pack.py build  --gguf Hy4-flash-IQ2_AH-00001-of-00002.gguf --out ./pack
python terminal/tools/strata_pack.py verify --gguf Hy4-flash-IQ2_AH-00001-of-00002.gguf --out ./pack
```

`verify` 不是可选步骤:它把写好的 pack 从磁盘重新读一遍,要求每个张量都满足 `decode(pack) == dequant_ggml(source)` **逐位相同**。没通过 `verify` 的 pack 不应该拿来用。

## 2. 系统设置

- 用交流电运行。电池模式下 GPU 会降频,测速结果不可比。
- 模型放本地 SSD。专家权重按需从磁盘读取,外置盘会成为瓶颈。
- 测速前关掉其他占 GPU 的进程。macOS 的内存压缩会在 arena 填充阶段介入,导致本来能跑的配置被杀。
- 不需要手动抬高 GPU 内存上限。引擎按可用内存决定专家 arena 的大小,并在运行中收缩或增长。

## 3. 内存预算

以 IQ2_AH 为例,单位 GB:

| 组成 | 大小 | 说明 |
|---|---|---|
| dense 权重 | ~10.8 | 常驻 |
| embedding | ~0.7 | 常驻 |
| 专家权重(全量) | ~53.4 | 在磁盘上,按路由命中进 arena |
| 专家 arena | 计算得出 | LRU + 跨层预读 |

arena **不是一个固定的默认值**,而是四项取最小:GPU 工作集、必须留给 macOS 的 RAM、专家权重总字节,以及一个保守的 28 GB 上限 —— 引擎会打印是哪一项卡住的。

## 4. 运行时开关

| 开关 | 默认 | 说明 |
|---|---|---|
| `--prefill-chunk` | Mac 上由 launch 给 4096 | 分块 prefill 的块大小 |
| `--draft-tokens` | 2 | MTP 每轮起草的 token 数 |
| `STRATA_METAL_ARENA_GB` | 不设 | 固定 arena 并关掉守卫,一般不要用 |
| `HY4_FLASH_MTP_EXPERT_GB` | 引擎默认 | 起草阶段的专家预算 |

`strata launch` 在 Apple Silicon 上会给出合适的值,通常一个都不需要手动指定。非 Darwin 平台的行为不受本后端影响。

## 5. 启动服务

```sh
strata launch --no-tui --yes \
  --model-path Hy4-flash-IQ2_AH-00001-of-00002.gguf \
  --pack-path ./pack \
  --engine-path build-metal/strata \
  --port 8099
```

| 参数 | 说明 |
|---|---|
| `--model-path` | GGUF 第一个分片 |
| `--pack-path` | tokenizer 与 chat template 所在的 pack 目录 |
| `--engine-path` | 上面构建出的 Metal `strata` |
| `--port` | HTTP 监听端口 |

启动后 `/v1/models` 会把模型列为 `loaded`,`/v1/chat/completions` 接受标准请求。注意 hy4-flash **默认开启思考**:未指定 `reasoning_effort` 时,正文先进入 `reasoning_content`,`content` 在推理块闭合前为 `null`。

## 5a. 图像

```sh
strata launch --no-tui --yes --vision on \
  --model-path Hy4-flash-IQ2_AH-00001-of-00002.gguf \
  --pack-path ./pack \
  --engine-path build-metal/strata \
  --context 32k --port 8099
```

视觉分片默认在模型旁边寻找,也可以用 `STRATA_HY4_FLASH_VISION_SHARD=<路径>` 直接指定。图片按常规方式发送,即内容块里的 `image_url` 配 `data:` URL。

一张 `h × w` 的图(两边都取到 32 的倍数)占 `(h/32) · (w/32 + 1)` 个位置 —— 那个 `+1` 是每个 merge 行一条 image-newline 行 —— 所以大图会吃掉上千个上下文位置,`--context` 要给够。

几何不是推测出来的,而是钉在 HuggingFace 的图像处理器上:目标尺寸、patch 网格、以及主干的三轴 rotary 位置,各自都按整数与它比对。畸形的图片包会被**点名拒绝**,而不是被解码成噪声。

纯文本 decode 与不含视觉路径的构建**逐位相同** —— 24/24 个 token id、24/24 个 logit 行 —— 所以打开图像不会扰动文本生成。

## 6. 实测结果

Mac mini,M4 Pro(16 核 GPU),64 GB 统一内存;IQ2_AH,MTP k=2,8K 上下文。

| 指标 | 起点 | 当前 |
|---|---|---|
| prefill | 35.8 tok/s | **125–127 tok/s** |
| decode(生成 256) | 7.31 tok/s | **17.300 tok/s** |
| decode(生成 1024) | — | 18.979 tok/s |

decode 的上限是结构性的:每个 token 约读 11.7 GB 权重,其中 86% 是 dense,而 arena 只能缓存专家那 14%。在实测 252 GB/s 下,不开 MTP 的上限约 21.6 tok/s —— MTP 把 dense 的读取摊到多行 verify 上,才把天花板抬上去。
