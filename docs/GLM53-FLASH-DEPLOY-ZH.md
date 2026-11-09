# GLM-5.3-Flash 部署指南(2× V100,中文版)

本指南在双卡工作站上部署 GLM-5.3-Flash 引擎路径(`strata-glm`)与多模态扩展
(`strata-vision` + GENI 协议),并复现文中全部数字。英文版在同目录
(`GLM53-FLASH-DEPLOY-EN.md`)。

以下所有数字都在这台机器上实测:

- 2× Tesla V100-SXM2-32GB(sm70),同一 PCIe 主机
- 64 GB 主机内存,Ubuntu 24.04,CUDA 12.x(`/usr/local/cuda`)
- 模型:`GLM-5.3-Flash-UNCENSORED`(dealignai 的 FP8 发布版,由我们重量化为
  UD-Q2_K_XL 级混合精度 GGUF,见 §4)

## 1. 本功能带来什么

- **`strata-glm`**,第二个引擎二进制,由 `STRATA_BUILD_GLM=ON` 显式开启(默认关,
  默认路径逐字节不变)。运行 glm5-next 架构:45 个 block,288 专家 MoE + 共享专家,
  GDN(gated delta net)注意力,12 层 DSA 索引器,以及 NextN/MTP 草稿块(blk.45)。
  stdin/stdout 服务(`GEN`、`GENI`、`LOGP`、`STOP`、`QUIT`),沿用引擎的专家缓存
  方案:热专家驻留显存(每卡一个池),工作集在钉住的内存层,其余从模型文件读。
- **量化管线**:mainline-llama.cpp 转换 → Q8_0 中间体 → imatrix → 逐张量类型表 →
  最终混合精度 GGUF → `tools/iq_pack.py --compat-bf16` 打包(引擎读的 index +
  tokenizer + dense 边车文件)。
- **终端集成**:家族以 `glm53` 出现在 `strata launch --plan/--family glm53`,
  另有 `tools/glm/start.sh` 一条命令起 OpenAI 兼容的局域网服务。
- **多模态**:`tools/vision/strata_vision.cpp`(图像编码进程)与引擎的 GENI 线上命令。
  图像行替换模型的 `<|image_pad|>` 串;GLM 图像 token 不占 rope 位置。
- **chat 与测试工具**:`tools/glm/chat.py`(引擎+分词器+GLM chat 模板一条命令)、
  分词器测试、打包测试。

## 2. 构建

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 \
      -DSTRATA_BUILD_GLM=ON
cmake --build build --target strata-glm -j8
```

说明:

- `GGML_CUDA` 保持 OFF:GLM 引擎自带 CUDA 核(`src/glm_port`,注意力/专家/mtp
  核),不走 ggml-cuda。
- `CMAKE_CUDA_ARCHITECTURES=70` 对应 V100;其他卡改成自己的架构号。
- 本功能只构建并测试了 CUDA。HIP/SYCL 未触及、未构建。

## 3. 模型文件与 pack

引擎读一个 *pack* 目录(`index.txt` + tokenizer 文件 + dense 边车文件),专家从
旁边的 GGUF 读。两种获得方式:

**a) 直接用产物**(本指南数字所用的那份):
`GLM-5.3-Flash-UNCENSORED-UD-Q2_K_XL.gguf`(108.8 GB,1412 张量)与其
`glm-pack/`(1.6 GB),放同一目录。

**b) 自己量化**:从 safetensors 源(UNCENSORED FP8 发布版;作者的 NVFP4 导出已确认
损坏,见 runbook)。管线简写:

```sh
# 1. HF→GGUF 转换(mainline llama.cpp 转换器),FP8 张量解码为 F16/BF16
python convert_hf_to_gguf.py <fp8源目录> --outfile GGUF_A_q8_0.gguf   # q8_0 中间体
# 2. imatrix,约 34 个校准块(校准文本见 tools/glm/build_calib_corpus.py)
./llama-quantize --imatrix imatrix.dat GGUF_A_q8_0.gguf mid.f16.gguf ...      # 详见 runbook
# 3. 逐张量类型表(参考 UD-Q2_K_XL 的表,tools/extract_ref_typemap.py)
./llama-quantize --tensor-type-file tensor_types_iq.txt mid.gguf final.gguf
# 4. 引擎要的 pack
python tools/iq_pack.py --gguf final.gguf --out glm-pack --compat-bf16
```

下载源:优先 ModelScope,其次 hf-mirror。磁盘:预留 ≥ 250 GB(源 + 中间体 + 成品)。

`--hc-npz`:UNCENSORED 的 FP8 发布版里,hc_* 递归张量被损坏的 safetensors 头压毁。
`tools/glm/nvfp4_dequant_gguf.py --hc-npz <官方值.npz>` 在反量化阶段嫁接官方值
(270 张量)。缺这步,这些层会按零填充 scale 解码。

## 4. 启动引擎

```sh
./build/strata-glm \
  --glm-pack /models/GLM-5.3-Flash-UNCENSORED-UD-Q2/glm-pack \
  --serve --max-context 8192 --kv int8 --parallel 1
```

- 预热约 4–5 分钟(dense 权重、MTP 块、专家池、内存层),stdout 出
  `READY 8192 stop pause`。
- 层切分自动:layers [0,24) 在 CUDA0,[24,45) 在 CUDA1。
- 实测稳态(8192 ctx,kv int8):显存 ~31.8 GB/卡(dense ~3.5 G + 专家池 ~27.5 G +
  KV + 草稿头),主机 RSS ~40 GB(钉住的内存层),热后磁盘读取 0.00/token。
- **主机内存红线**:引擎 + 操作系统 ≤ 总内存 90%(本机:64 GB 的 57.7 GB)。
  8192 ctx 下引擎约 40 GB 匿名内存,安全。**不要**给引擎套 `MemoryMax` cgroup 限制:
  108 GB GGUF 的页缓存计入该 unit,压力上来后 systemd-oomd 会杀引擎(实测两次)。
  改看 `MemAvailable`。
- 请求按行写 stdin(即 `tools/glm/chat.py` 说的协议):
  - `GEN <max_new> [采样键] <id,id,...>`
  - `GENI <max_new> <embeddings文件> <id,id,...>` —— 图像(§6)
  - `LOGP`、`STOP`、`QUIT`。

## 5. 服务给应用

```sh
tools/glm/start.sh            # 一条命令:引擎 + OpenAI 兼容服务(:8000)
strata connect <应用> --check # 按该应用的方式发一条 prompt 和一次工具调用
```

`start.sh` 启动引擎,前置 `serve/server.py`(OpenAI/Anthropic 兼容,
/v1/chat/completions),并改写 GLM 工具调用。实测:本机同时跑两个会话、各 512K
上下文(验证过 1M token)。

手写 serve 配置(`--config`,自定义模型)要带两个键:`"arch": "glm5-next"`(缺了
family 判成 qwen,图片请求走错视觉分支)和 `"reasoning_effort": "high"`(缺了默认
low;官方基座 low 档不思考,本模型 low 档仍输出无 marker 的思考,思考文本会混进
content,解析器无从切分)。上下文越大解码越慢,由专家池覆盖决定(KV 挤占显存池,
池外专家走 PCIe):512K×2 池 81% → 22.6 tok/s;1M×2 池 64% → 10.5 tok/s(同题同
150 token)。本机双 V100 的推荐点是 524288×2。

## 6. 多模态(图片)

文本模型没有视觉权重;视觉来自独立的 **mmproj** 文件(视觉编码器 + 投影器,
glm5v 投影器,448 px / patch 14,348 张量)。我们用的是 unsloth 基座的
`mmproj-F16.gguf`(1.13 GB);UNCENSORED 源不带视觉权重,所以投影器是基座那份。

构建编码器(链接 llama.cpp 的 mtmd;checkout 必须认识 glm5v):

```sh
cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_DIR=/path/to/llama.cpp      # 须含 src/models/glm5-next.cpp
cmake --build build-vision --target strata-vision -j8
```

编码图片(常驻进程,每图一行 `ENC`):

```sh
./build-vision/bin/strata-vision --mmproj <mmproj-F16.gguf> --model <那个.gguf>
# stdin:  ENC /path/picture.png /tmp/pic.bin
# stdout: OK 256 256 1 2259        (n_tokens, nx, ny, 毫秒) → SVE1 文件
```

向引擎提问(prompt 里出现一次 `caff`;驱动把它扩成 256 token 的串;引擎把
SVE1 行放到那里):

```python
# multimodal/e2e_test.py,简写
ids = tok.encode(渲染后的模板, parse_special=True)
i = ids.index(154854)                      # 词汇表解析:154854 = caff
ids = ids[:i] + [154854]*256 + ids[i+1:]   # 每行 embedding 一个 pad
stdin.write(f"GENI 300 /tmp/pic.bin {','.join(map(str, ids))}\n")
```

本机端到端实测:一张合成图(红底、蓝色大圆中上部偏左、黄色方块右下)经引擎
GENI 路径被正确描述;prefill 280 tok 1.4 s,decode 29.9 tok/s,投机接受 85.9%。
同一对文件也能过 llama.cpp 的 `llama-mtmd-cli`:

```sh
llama-mtmd-cli -m final.gguf --mmproj mmproj-F16.gguf --image pic.png \
  -p "describe" -ngl 20 -t 6 -c 2048 -ub 256 -b 256
```

(小的 `-c/-ub/-b` 是要点:默认值要 4.7 GB 计算缓冲,与专家池并存时放不下。)

## 7. 实测精度与性能

全部在本机(2× V100-SXM2-32GB,64 GB 主机):

| 项目 | 数值 | 方法 |
|---|---|---|
| 解码速度 | 26.9–35.9 tok/s | 日常对话,kv int8,投机开 |
| 投机接受率 | 81.9%(740/903) | 10 个多类提示 + 计数 |
| prefill | ~96 tok/s | 128 token 块 |
| 显存命中率 | 83–86% | 专家池命中计数器 |
| MMLU(228 题零样本,首字母判定) | 79.8% | 见注 1 |
| HarmBench-50 拒绝 | 1/50(2%) | 正则判别,无分类器 |
| 权重保真度 vs unsloth 基座 UD-Q2_K_XL | attn cos 0.9993,experts cos 0.900 | 21 张量探针 vs FP8 源 |
| PPL 比值(我们 / unsloth 基座量化),wikitext-2,20 块 | 1.115 | llama-perplexity -ngl 20 |
| top-1 一致率 | 77.7% | 同一轮 |

注 1:答案字母在回复开头("B Explanation: …"),228 条里 221 条如此——判**第一个**
独立 A–D,永远不要判最后一个(解释段里的杂散选项字母在早前一轮里错判了 54 题)。
参考库 UD-Q2 是**基座**模型的量化,所以 PPL/top-1 数字混入了微调漂移与管线差异,
是同级别健全性检查,不是纯量化损失。

## 8. 限制

- 仅 CUDA(sm70 实测)。HIP/SYCL 未构建。
- mmproj 是基座的投影器;UNCENSORED 源不带视觉权重,图像质量的说法止步于
  "正确描述简单场景"。
- `serve/server.py` 的图片路径与 BGENI 并行路径未做端到端实测(实测的是终端 chat
  路径与 e2e_test.py)。
- 量化管线(§3b)是文档化而非打包化:imatrix 与类型表步骤按 runbook 手工执行。
