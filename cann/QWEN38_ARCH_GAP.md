# Qwen3.8-Flash-Next 与现有移植的差距

对照 `config.json`（`Qwen4ExpForConditionalGeneration`）和 `strata/cann` 现在支持的 `qwen35moe`。

## 结论先说

**量化格式不是障碍，架构才是。**

GGUF 的 IQ3_XXS 版本用了 14 种存储类型：

```
BF16=484  F16=1  F32=292  IQ2_S=20  IQ2_XS=20  IQ2_XXS=18  IQ3_S=82
IQ3_XXS=12  IQ4_NL=58  IQ4_XS=67  Q2_0=38  Q4_K=45  Q5_K=15  Q6_K=71
```

我们的 `cann/tests/quant_parity.cpp` 已经逐位验证过其中 **11 种量化类型**
（IQ2_XXS / IQ2_XS / IQ3_XXS / IQ4_NL / IQ3_S / IQ2_S / IQ4_XS / Q2_0 / Q4_K / Q5_K / Q6_K），
BF16/F16/F32 由 `tensor_load.cpp` 直接读。**这一层不用动。**

## 架构对照

| 项 | Qwen3.8-Flash-Next | 现有引擎（qwen35moe） | 工作量 |
|---|---|---|---|
| 层数 | 48（36 线性 + 12 全注意力，每 4 层一个注意力） | 40（30 + 10） | 参数化 |
| hidden | 2560 | 2048 | 参数化 |
| 线性注意力 | 16 k 头、**48 v 头**、d=128、conv k=4、状态 fp32 | 16 k 头、**32 v 头**、d=128、conv k=4、状态 fp32 | 小改（头数） |
| 全注意力 | 24 q 头 / 2 kv 头 / **head_dim 256**，部分旋转（0.25） | 16 q / 2 kv / 512，部分旋转 | 中改（head_dim、GQA 比例） |
| 稀疏注意力 | **QSA**：indexer 4 头 × 128、kv 1 头、budget 2048、compress 4 | 无 | **新增** |
| MoE | **512 专家选 10**，inter 640 + 共享专家 640 | 256 选 8，inter 512 | 参数化 |
| 残差 | **HyperConnection**（hc_count 4、低秩 320） | 普通残差 | **新增** |
| 词嵌入增强 | **PLE**：51.2B 参数的 n-gram 查表，第 2 层，ngram_size 3、conv k=4 | 无 | **新增** |
| 投机解码 | **MTP**：1 层 draft head | 无 | 可选 |
| 词表 | 248320 | 248320 | 相同 |

## 三块真正的新工作

**1. PLE（per-layer token embedding）** —— 这个模型最特别的地方。

51.2B 参数的 n-gram 表（28.8 GB，就是这个 GGUF 的第二个分片），在**第 2 层**注入。
按 3-gram 查表 + 短卷积（k=4），每 token 只读一行。
llama.cpp 的 `-lzm on`（lazy mode）就是为它设计的：表留在 SSD，按行读。

移植要点：查表逻辑 + 卷积 + 与残差流的合并方式。**不需要懂矩阵乘**，但需要把表放在 host/SSD 上而不是显存里——28.8 GB 进了 HBM 就装不下模型了。

**2. QSA（Qwen Sparse Attention）**

12 个全注意力层里，用一个轻量 indexer（4 头 × 128，kv 1 头）挑出 2048 个 key，再做注意力。
compress_ratio=4 意味着 key 先按 4 个一组压缩。

移植要点：这是"选哪些 key"的额外网络，不是标准注意力。选错就是精度崩，选对才是稀疏加速。

**3. HyperConnection**

hc_count=4、hc_lowrank=320：残差流不是简单相加，而是 4 路低秩混合后再合并。
论文里叫 hybrid mamba-attention 的配套设计。

移植要点：每层多两个小矩阵乘（320 秩），计算量不大，但结构改动影响所有层。

## 建议的顺序

1. **先让 llama.cpp 跑通**（进行中）：拿到输出样本和 tok/s，作为对照物。
2. **再扩现有引擎的参数**：48 层 / 2560 hidden / 48 v 头 / 512→256 head_dim / 512 专家选 10。
   这些是"改常数 + 改循环"，能用 Qwen3.6 的同一套验证方法逐层对齐。
3. **加 PLE**：相对独立，可以单独验证（查表 + 卷积的输出对不对）。
4. **加 QSA**：最难，且没有 llama.cpp 之外的参考实现。
5. **HyperConnection**：最后做，因为它是全局结构改动。
6. **MTP**：纯性能项，可跳过。

## 一个已经踩过的坑

第 3.18 节的教训：ggml-cann 的 `GGML_OP_GATED_DELTA_NET` 在旧版里不存在，导致 30 个循环层
静默回退 CPU，只有 3.8 t/s。新版 llama.cpp 有 `GGML_OP_GATED_LINEAR_ATTN` 和
`aclnn_recurrent_gated_delta_rule` 的完整实现——所以这次先确认了整条链路在 NPU 上跑（0.6B
模型实测预填 1332 t/s），再往下走。

移植时同样要**先确认算子真的在设备上跑**，而不是看它"没报错"。

## CANN 后端的算子覆盖（关键）

把 llama.cpp 的 `qwen4exp.cpp` 逐函数扫了一遍，看每个阶段用了哪些 ggml 算子；
再对照 `ggml-cann.cpp` 的 `supports_op` 列表：

| 阶段的算子 | CANN 支持 |
|---|---|
| `GATED_LINEAR_ATTN`（36 个线性层） | ✅ |
| `FLASH_ATTN_EXT`（12 个注意力层） | ✅ |
| `MUL_MAT` / `MUL_MAT_ID`（MoE） | ✅ |
| `GET_ROWS`（词嵌入、PLE 查表） | ✅ |
| `ROPE`（含 mrope） | ✅ |
| `CUMSUM` / `SET_ROWS` / `FILL` / `CLAMP` | ✅ |
| **`DSV4_HC_PRE_GATED` / `DSV4_HC_POST`**（HyperConnection） | ❌ **不支持** |
| **`LIGHTNING_INDEXER`**（QSA 索引器） | ❌ **不支持** |
| **`TOP_K` / `TOP_K_MASK`**（QSA 选 key） | ❌ **不支持** |

**后果**：HyperConnection 在全部 48 层都有，QSA 索引器在 12 层有。它们会**静默回退到 CPU**，
每层都要在 NPU 与 CPU 之间搬数据。模型能跑，但速度会明显低于论文数字，而且慢在哪要实测才知道。

这正是 3.18 节那个教训的重演：算子缺失不报错，只是慢。

好消息是 `GATED_LINEAR_ATTN` 有实现（旧版没有），所以 36 个线性层——占计算量大头——在 NPU 上。

## 对这五个算子的补法

`aclnn` 层面能对上的：

- `TOP_K` → `aclnnTopk`（我们已经在 `strata/cann` 里封装并验证过，误差 4e-7）
- `LIGHTNING_INDEXER` → 没有现成 aclnn 算子，需要拆成 matmul + softmax + topk 组合
- `DSV4_HC_PRE_GATED` / `DSV4_HC_POST` → 低秩矩阵乘 + sigmoid + 加权的组合，可拆
- `TOP_K_MASK` → 比较 + 掩码填充

也就是说这五个都能用现有 aclnn 算子拼出来。真要做性能，给 llama.cpp 的 CANN 后端补
`supports_op` + 实现是更直接的路径（比在 Strata 里重写一遍划算）。