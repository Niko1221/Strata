# PLE、QSA、HyperConnection：三块新东西到底怎么算

读 `qwen4exp.cpp` 的 `build_inp_ple` / `build_ple` / `build_qsa_sel` 得到的结论。
这三块是移植里"没有现成代码可抄"的部分，写清楚以后就不用再猜。

## PLE（per-layer token embedding）

分两步，**大步在 host，矩阵乘很小**。

### 第一步：查表（`build_inp_ple`，1427-1445）

```c
rows = i32[n_heads * n_tokens]          // host 上算好的行号
emb  = get_rows(per_layer_tok_embd, rows)   // 51.2B 参数的表，只取这些行
emb  = reshape(emb, ple_head_dim * n_heads, n_tokens)
```

- `n_heads = 8`（`heads_per_ngram`），`ple_head_dim = 160`（GGUF 的 `embedding_length_per_layer_input`），
  所以每 token 取 8 行 × 160 维 = **1280**，不是 2560。`config.json` 的 `ple_embed_dim: 2560` 会误导。
- 行号由**最近 3 个 token**（`ngram_size=3`）哈希得到，在 host 上算。
- 表有 51.2B 参数、28.8 GB，但**只按行读**。这就是 `-lzm on`（lazy mode）存在的理由：
  表留在 SSD，每 token 读 8 行（约 10 KB）。

**移植要点**：查表本身不是矩阵乘，不需要 NPU。真正要做的是把 28.8 GB 的表放在 host/SSD 上，
通过 PCIe 每 token 搬 10 KB。如果为了省事把表拷进 HBM，64 GB 卡上模型就装不下了。

### 第二步：门控注入（`build_ple`，1447-1535）

```c
key   = ple_key   @ emb       // [n_embd, hc]
value = ple_value @ emb       // [n_embd, 1]
key   = rms_norm(key 按 hc 流分组) * ple_norm_key
query = rms_norm(hidden 按 hc 流分组) * ple_norm_query
s     = sum_rows(key * query) / sqrt(n_embd)          // 每流一个标量
mag   = sqrt(clamp(|s|, 1e-6, inf))
gate  = sigmoid(sign(s) * mag)                        // 带符号的软门
gated = repeat(value, hc) * gate                      // 单值广播到 hc 流
norm  = rms_norm(gated 按 hc 分组) * ple_norm_conv
conv  = 深度可分离因果卷积(hc_dim 通道, 核 4, 膨胀 3)  // 4 个移位副本相加
conv  = silu(conv)
out   = hidden + gated + conv                          // 两路残差
```

三个容易看漏的点：

1. **`gate` 用带符号的平方根**（`sigmoid(sign(s)*sqrt(|s|))`），不是普通 sigmoid。抄错这一步不会报错，
   只会让数值慢慢偏。
2. **卷积是膨胀的**：核 4、膨胀 = `ngram_size` = 3，所以抽头间隔 3 个 token，感受野是 9 个 token。
   不是普通的 4-tap 卷积。
3. **历史要跨 ubatch**：`build_conv_state_at` 把前一块的历史拼在前面，这样分块预填和一次性预填
   结果一致。移植时如果只处理单次前向，分块时会不一致。

计算量：每层两次 `[2560 × hc*2560]` 的投影（hc=4，所以是 2560×10240）—— 这个不小，
但只在**第 2 层**做一次（`ple_layer_ids=[2]`）。

## QSA（Qwen Sparse Attention）

只在 12 个全注意力层里。核心是"先挑 key，再做注意力"。

### 索引器（`build_qsa_sel`，816-890）

```c
k_raw = index_k_proj @ cur                    // 原始 key
pooled = mean(k_raw, 每 compress_ratio=4 个 token)   // 池化
pooled = rms_norm(pooled) 然后 rope
q     = index_q_proj @ cur, rms_norm, rope    // 4 头 × 128 维
score = lightning_indexer(q, pooled, weights=1/sqrt(128), mask)   // [n_pool, n_tokens]
n_top = min(n_pool, indexer_top_k / kpool) = 2048/4 = 512
top_k = top_k(score, n_top)
```

- 索引器很小：4 头 × 128 维，`kv 1 头`。
- **池化在 norm 和 rope 之前**：cache 里同时存"原始 key | 池化 key"，原始 key 不参与 norm/rope。
- `lightning_indexer` 是把各头分数**相加**（权重固定为 `1/sqrt(dim)`），不是 softmax。
- 预算 2048 是 token 数，池化比 4，所以实际选 **512 个池化单元**。

### 注意力（`build_attn_qsa`）

用 top-k 结果构造一个 f16 掩码，再做普通 flash attention——只算被选中的那些 key。

### 移植要点

全都能用现成 aclnn 算子拼：matmul、mean(池化)、rms_norm、rope、topk、掩码、flash attention。
难点不在算子，而在**两块 KV cache**（原始 key 和池化 key 各有自己的索引）和分块预填时
"哪些池化单元是这一块完成的"这套簿记。

## 对移植顺序的建议

1. **PLE 先做**：结构独立，只需在层 2 之后插一段，验证方法明确（用固定 token 序列算出门控值对表）。
2. **QSA 后做**：它和 KV cache 的簿记纠缠，而且选错 key 会直接掉精度，需要 llama.cpp 的输出做对照。
3. 两者都不需要碰线性层，可以和线性层的移植并行。

## 一个提醒

这两块的 ggml 实现里出现了 `lightning_indexer`、`top_k`、`top_k_mask`、`d sv4_hc_*` 这些
**CANN 后端没有的算子**（第 11.5 节）。llama.cpp 会把它们放到 CPU 上算，所以第一次跑
Qwen3.8 时，PLE 和 QSA 的耗时大概率全在 CPU——这也是为什么解码速度要以实测为准，不能按
理论 FLOPs 估。


## HyperConnection

残差流不是一条，是 **4 条并行**（`hc_count=4`），形状 `[n_embd, hc, T]`，每层进出都经过它。

### 进块前：混合（`build_hc_mix`，348-415）

```
xn    = grouped_rms_norm(x) * w_norm        // 每条流内部归一，gamma 形状 [n_embd, hc]
xn    = reshape(xn, hc*n_embd, T)
lo    = silu( (w_down @ xn) / hc )          // 低秩 320
gate  = w_up @ lo                           // [hc*n_embd, T]，再 reshape 成 [n_embd, hc, T]
gate  = sigmoid(gate)
mixed = mean_over_streams(xn * gate)        // 塌成 [n_embd, T]，这就是块的输入
inject= (w_inject @ xn) / hc                // [hc, T]，供出块时用
```

### 出块后：散布（`build_hc_combine`，417-445）

```
w   = 2 * sigmoid(inject)                   // 以 1 为中心，所以 inject=0 时就是普通残差
cur = residual + repeat(block_out, hc) * w  // 每条流加同一个块输出，各自乘权重
```

三个要点：

1. **两条矩阵乘把 2560 维压到 320 维再展开**（`w_down`/`w_up`），计算量不大，但每层都有。
2. **门是逐流的**：`gate` 和 `w` 都是 `[hc, T]` 或 `[n_embd, hc, T]`，不是每层一个标量。
   抄成标量不会报错，只会掉精度。
3. **`2*sigmoid` 的中心化**：不是 `sigmoid`。用错会让残差的量级整体偏一倍，靠调学习率是救不回来的
   （这里没有学习率，就是错）。

### 算子层面

`DSV4_HC_PRE_GATED` / `DSV4_HC_POST` 是把上面两步各融合成一个算子（省一次读写）。
**它们不是新算法**——拆开就是 matmul + rms_norm + sigmoid + mean + broadcast-mul + add，
全部有对应的 aclnn 算子。

这对移植是好消息：CANN 后端不支持这两个融合算子（第 11.5 节），llama.cpp 会自动走未融合的
等价路径，只是多几次内存读写。**正确性不受影响，只有速度受影响。**

## 三块合起来的判断

| 块 | 新算法？ | 计算量 | 难在 |
|---|---|---|---|
| PLE | 查表 + 门控 + 膨胀卷积 | 每层一次 2560×10240，但只在第 2 层 | 表放哪（28.8 GB 不能进 HBM）、历史跨块 |
| QSA | 小索引器 + 池化 + top-k + 掩码注意力 | 12 层，索引器很小 | 两套 KV cache 的簿记、选错 key 掉精度 |
| HyperConnection | 4 路残差 + 低秩门 | 每层两次小矩阵乘 | 逐流的门不能写成标量 |

三块都能用现有 aclnn 算子拼出来，**没有需要写 AscendC 自定义 kernel 的地方**。
