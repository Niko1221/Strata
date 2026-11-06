# 线性注意力层：llama.cpp 的实现 vs 我们的 gdn_step

对照 `dl/llama.cpp-master/src/models/qwen4exp.cpp:1106-1243`（`build_layer_attn_linear`）
和 `strata/cann/src/engine.cpp` 的 `gdn_step`。

这两段代码算的是同一件事：门控 DeltaNet。但有五处结构差异，移植时逐条改。

## 数据流对照

| 步 | llama.cpp（qwen4exp） | 我们的 gdn_step | 差异 |
|---|---|---|---|
| 投影 | `build_qkvz`：**一次矩阵乘**出 q\|k\|v\|z | `attn_qkv.weight` 和 `attn_gate.weight` **两次** | **要合并** |
| beta/alpha | `ssm_beta_alpha_cat`：一次矩阵乘，再 view 切两段 | `ssm_beta.weight`、`ssm_alpha.weight` 两次 | **要合并** |
| beta 激活 | `sigmoid` | `1/(1+exp(-x))` | 相同 |
| gate | `softplus(alpha + ssm_dt) * ssm_a` | `log1p(exp(sp)) * negA` | 相同（都是 log 尺度） |
| 短卷积 | `ggml_ssm_conv`，通道数 `d_k*n_k*2 + d_v*n_v` | 4 次 mul+add 累加，通道 `kQkvDim` | 通道数不同 |
| Q/K 归一 | `build_gdn_l2_norm`（L2，带 eps） | `rms_norm(ones, eps/dim)` | 数学等价，可保留 |
| q/k 重复 | 有融合算子时**不重复**（算子内部做块映射） | 重复到 32 头（旧 fp32 路径需要） | **要删** |
| 递推 | `build_recurrent_attn` → `GGML_OP_GATED_LINEAR_ATTN` | `gdn_batched_f32`：8 个 aclnn 调用 | 换成融合算子 |
| 门控归一 | `build_norm_gated(output, ssm_norm, z)` = rms_norm × silu(z) | 同 | 相同 |
| 输出投影 | `ssm_out` | `ssm_out.weight` | 相同 |

## 五处要改的

### 1. qkvz 合并（省一次 2560×16384 的矩阵乘）

Qwen3.8 的线性层要输出 q(16×128)、k(16×128)、v(48×128)、z(48×128)，合计 16384 维。
我们现在的两颗权重是分开的，Qwen3.8 的 GGUF 里是**一颗** `attn_qkv.weight`（16384 行）。

移植时按一个 gemv 出全部，再按 2048 / 2048 / 6144 / 6144 切开。

### 2. beta/alpha 合并

Qwen3.8 用 `ssm_beta_alpha_cat`（96 行 = 48 beta + 48 alpha），我们用了两颗。
llama.cpp 的 view 边界是 `ggml_row_size(type, num_v_heads)`，也就是按行切。

### 3. 头数从 32 变 48

`kVHeads` 是编译期常量，改用配置读；`kValueDim = 48*128 = 6144`；
`kQkvDim = 2*2048 + 6144 = 10240`（现在是 8192）。

### 4. 删掉 q/k 的重复

我们的 `gdn_step` 里有一段把 16 个 k 头复制成 32 个头（旧 fp32 批量路径需要）。
换成 `aclnnRecurrentGatedDeltaRule` 后**必须删掉**——算子自己按 `h/(Nv/Nk)` 做块映射，
再复制一遍会让 descriptor 的尺寸和填充内容对不上（我在第 36 轮踩过，表现为输出全 NaN）。

### 5. 短卷积的通道数

现在是 `kQkvDim`（8192），Qwen3.8 是 10240。卷积抽头、历史缓冲、置换表都按它走。

## 不用改的

- **状态布局**：llama.cpp 的 `ggml_reshape_4d(state, d_v, d_k, n_v_heads, n_seqs)` 中
  ne0=d_v（v 最快）→ 内存是 `[head][k][v]`，即 `S^T`。我们的转置存放 `M[j][i] = S[i][j]`
  和 `aclnnRecurrentGatedDeltaRule` 的 `[Dk][Dv]` **三者一致**（第 36 轮实测 rel L2 0.0035）。
- **门控归一**：都是 rms_norm × silu(z)。
- **输出投影**：一样。

## 已有的两个算子事实（第 36、37 轮实测）

1. `aclnnRecurrentGatedDeltaRule` 只吃 **bf16**（我们现在的 fp32 路径与浮点参考逐位一致，
   换过去会退到 bf16）。算子层面 attn 误差 4e-3、state 误差 3.5e-3。
2. 它的头映射是 **blocked**（v 头 h 用 k 头 `h/(Nv/Nk)`），llama.cpp 的参考实现是 mod。
   上游 llama.cpp 的 CANN 后端已经处理了这个差异；我们自己接的时候要注意。

## 一句话

线性层是这次移植里**最接近已有代码**的部分：五处结构性改动，都是"合并两次矩阵乘、改三个常数、
删一段重复"，没有新算法。真正的新东西是 PLE、QSA、HyperConnection 三块。
