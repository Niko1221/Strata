# Qwen3.8-Flash-Next 的实际张量表（从 GGUF 读出来的）

用我们自己的 `gguf_dump` 直接读**下载中的文件**（元数据和张量索引在文件开头，不需要权重）。
这比 `config.json` 权威：`config.json` 是原始模型的结构，GGUF 是**这个量化版本**的结构，
两者有几处不一样。

## 超参数（`qwen4exp.*`）

```
block_count                          48
embedding_length                     2560
attention.head_count                 24
attention.head_count_kv              2
attention.key_length                 256
attention.value_length               256
attention.indexer.head_count         4
attention.indexer.key_length         128
attention.indexer.top_k              2048
attention.layer_norm_rms_epsilon     (见下面的"读取器的坑")
expert_count                         512
expert_used_count                    10
expert_feed_forward_length           640
expert_shared_feed_forward_length    640
hyper_connection.count               4
hyper_connection.low_rank            320
ple.conv_kernel                      4
ple.heads_per_ngram                  8
ple.ngram_size                       3
embedding_length_per_layer_input     160
rope.dimension_count                 64
rope.freq_base                       10000000
ssm.conv_kernel                      4
ssm.group_count                      16
ssm.inner_size                       6144
ssm.state_size                       128
ssm.time_step_rank                   48
context_length                       262144
split.count 2   split.no 0   split.tensors.count 1224
```

## 每层都有（48 × 8 = 384 个张量）

| 张量 | 类型 | 形状 |
|---|---|---|
| `hc_attn_norm.weight` | F32 | 10240 |
| `hc_attn_down.weight` | BF16 | 10240 × 320 |
| `hc_attn_up.weight` | BF16 | 320 × 10240 |
| `hc_attn_inject.weight` | BF16 | 10240 × 4 |
| `hc_ffn_norm/down/up/inject.weight` | 同上 | 同上（FFN 那一半） |
| `ffn_gate_inp.weight` | BF16 | 2560 × 512（路由） |
| `ffn_gate_inp_shexp.weight` | BF16 | 2560（共享专家门） |
| `ffn_gate_exps.weight` | IQ2_XS | 2560 × 640 × 512 |
| `ffn_up_exps.weight` | IQ2_XS | 2560 × 640 × 512 |
| `ffn_down_exps.weight` | IQ4_NL / Q2_0 | 640 × 2560 × 512 |
| `ffn_{gate,up,down}_shexp.weight` | IQ4_XS / IQ3_S / Q4_K / Q5_K | 2560 × 640 |

注意 `hc_*_norm` 是 **10240 = 4 × 2560**，也就是每条残差流一份 gamma，不是 2560——这和
HyperConnection 的 4 路残差对应。

## 线性注意力层（36 个）

| 张量 | 类型 | 形状 | 说明 |
|---|---|---|---|
| `attn_qkv.weight` | IQ4_XS | 2560 × **10240** | q\|k\|v 合并：16×128 + 16×128 + 48×128 |
| `attn_gate.weight` | IQ3_S | 2560 × 6144 | z 门（48×128） |
| `ssm_out.weight` | IQ4_XS | 6144 × 2560 | |
| `ssm_norm.weight` | F32 | 128 | 每头一份 |
| `ssm_a` | F32 | 48 | 每头一个 |
| `ssm_dt.bias` | F32 | 48 | |
| `ssm_alpha.weight` | BF16 | 2560 × 48 | |
| `ssm_beta.weight` | BF16 | 2560 × 48 | |
| `ssm_conv1d.weight` | F32 | 4 × 10240 | 核 4，通道 10240 |

**和我们现有引擎的差异**：`kQkvDim` 8192 → **10240**，`kValueDim` 4096 → **6144**。

## 全注意力层（12 个）

| 张量 | 类型 | 形状 | 说明 |
|---|---|---|---|
| `attn_q.weight` | IQ4_XS | 2560 × **12288** | 24 头 × 512 = q(256) + gate(256) **合并** |
| `attn_k.weight` | IQ4_XS | 2560 × 512 | 2 个 kv 头 × 256 |
| `attn_v.weight` | Q6_K | 2560 × 512 | |
| `attn_output.weight` | Q6_K | 6144 × 2560 | 24 × 256 |
| `attn_q_norm.weight` | F32 | 256 | 每头一份 q-norm |
| `attn_k_norm.weight` | F32 | 256 | |

`attn_q` 是 12288 而不是 6144：`output_gate_type=sigmoid`，q 和门一起算出来。

## QSA 索引器（12 个，只在注意力层）

| 张量 | 类型 | 形状 |
|---|---|---|
| `indexer.q_proj.weight` | BF16 | 2560 × 512（4 头 × 128） |
| `indexer.k_proj.weight` | BF16 | 2560 × 128（1 头 × 128） |
| `indexer.q_norm.weight` | F32 | 128 |
| `indexer.k_norm.weight` | F32 | 128 |

## PLE（全局各一份，用在第 2 层）

```
ple_key.weight, ple_value.weight, ple_norm_key.weight,
ple_norm_query.weight, ple_norm_conv.weight, ple_conv1d.weight
```

配合 `per_layer_token_embd.weight`（51.2B 参数的 n-gram 表，在**第二个分片**里）。

## 三处"和预期不一样"的地方

### 1. `ssm_beta` / `ssm_alpha` 是分开的两颗，不是 `ssm_beta_alpha_cat`

llama.cpp 的代码两条路径都支持，但这个 GGUF 用的是分开的。我在 `QWEN38_GDN_DELTA.md`
里写的"要合并 beta/alpha"**对本文件不成立**——照现在这样两颗分别做就行。

### 2. `embedding_length_per_layer_input = 160`

`config.json` 里 `ple_embed_dim` 是 2560，但 GGUF 里 PLE 每头是 **160**：8 头 × 160 = **1280**
（不是 2560）。移植时按 1280 算，不是按 hidden 算。

### 3. 我们的 GGUF 读取器读不到浮点元数据

dump 里 `attention.layer_norm_rms_epsilon` 显示 **0**，真实值是 1e-6——因为 `Gguf` 只存了
整数和字符串，浮点元数据被丢掉了。

**移植前必须先补这个**：eps、rope 的 `freq_base`（这个恰好是整数所以出来了）、
`attention.key_length` 之类只要有浮点就会读成 0，而读成 0 不报错，只会让数值慢慢偏。

## 怎么复现

```bash
# 在机器上：解包源码、编 gguf_dump、读下载中的文件
tar -xzf /data/cann-src.tar.gz -C /data/strata-src
cmake -S /data/strata-src/cann -B /data/build-strata -DCMAKE_BUILD_TYPE=Release
cmake --build /data/build-strata --target gguf_dump -j 32
/data/build-strata/gguf_dump /data/models/IQ3_XXS/.../....incomplete
```
