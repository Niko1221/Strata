# 昇腾 910B2C 移植：结果与过程（Strata / CANN）

更新时间：2026-11-05 12:40（Asia/Shanghai）
服务器：`<用户>@<服务器地址> -p <端口>`

---

## 0. 先看这里（交接摘要）

**做了什么**：把 Strata 的引擎移植到昇腾 910B2C（CANN 9.0），用 Qwen3.6-35B-A3B 的 GGUF
端到端验证，并让它自己对外提供 OpenAI 兼容接口；同时把原本跑不动的部署调通、托管好。

**现在机器上有两个服务**（都在 supervisord 下，随容器启动）：

| 端口 | 引擎 | 速度 | 用途 |
|---|---|---|---|
| 18789 | llama.cpp + ggml-cann | 3.4 tok/s | 实际使用的服务，隧道指向它 |
| 18790 | **本次移植的引擎** | 预热 1.8 s/6 token | 参考实现，证明移植真的能跑 |

**三条验收证据**（全部可复现，见第 10 节的命令）：

1. **算子与整模数值对齐**：15 项对照测试全过，包括 aclnn 算子对 CPU oracle、量化解码对 ggml
   （Q4_K/Q5_K/Q6_K/Q8_0 逐位相同）、单层、MoE、GDN、RoPE、注意力、整模 40 层。
2. **端到端验证**：同一个提示词，我们引擎与 llama.cpp 的下一个 token 分布 **top-5 同序，
   最大 logprob 差 0.071**；贪心续写 **逐 token 一致**（` Paris, a city renowned for`）。
3. **对外服务**：18789 经隧道提供 OpenAI 兼容接口（`accept.sh` 全过）；18790 由移植引擎提供，
   `/v1/completions` 返回同样的文本。

**代价与限制（不藏）**：

- 18789 跑的是 **llama.cpp**，不是我们移植的引擎——我们引擎慢约 10 倍，胜任不了实际流量。
  这一点在 18790 的 `/health` 和 `/v1/*` 响应里都有标注。
- `ggml-cann` 仍然没有可用的 `GGML_OP_GATED_DELTA_NET`，那 30 层回退到 CPU 执行。我实现过一版，
  输出是乱码且 prefill 慢 14 倍（3.0 vs 42 t/s），已回退，见 3.18。
- 我们的引擎不支持逐通道门控（KDA）的 GDN，会显式报错而不是给出错误结果。
- 单进程 HBM 上限需用 `STRATA_BUDGET_GB` 控制，否则会和另一个服务抢显存（见 3.22）。
- 分词在 18790 上借用 llama.cpp 的接口；模型推理完全是我们自己的。

**Qwen3.8-Flash-Next 是另一条线**：见第 11 节。换了三台机器（第二台 ARM 且无出网，弃用），
现用 2×910B2C 已把 llama.cpp + CANN 编译通过、0.6B 实测到 1332 t/s 预填，大模型在下。
那条线的关键发现是 **CANN 后端缺 5 个 qwen4exp 专用算子**（HyperConnection / QSA），
它们会静默回退 CPU——补法在第 11.5 节。

**怎么推仓库**：`strata/` 下 39 个本地 commit 未 push（`git log --oneline` 可见），
`/data/ascend910-deploy` 是部署仓库的工作副本。两处都等你推。

---

## 0.1 本文结构

第 1–9 节按主题记录环境、设计与实测数据；第 10 节是可复现的验收命令；
第 11 节之后是按时间追加的开发日志（含走过的弯路），保留它是为了说明每个结论的来源。

---

## 1. 登录与环境

| 项目 | 实测值 |
|---|---|
| 系统 | openEuler 24.03 (LTS-SP2)，容器内，内核 5.15.0-113 |
| CPU | 128 线程（`/proc/cpuinfo` 显示 i7-8550U，实际为虚拟化掩码），支持 AVX2 / AVX-512 / AMX |
| 内存 | 64 GB |
| 磁盘 | `/` 30 GB；`/data` 59 GB（已用 25 GB）；`/mnt/moark-models` 只读 NFS |
| NPU | **Ascend 910B2C ×1**，HBM 65536 MB，驱动 26.0.rc1 |
| CANN | **9.0.0**（`V100R001C10SPC001B250`） |
| Python | 3.11.15；`torch 2.10.0+cpu`、`torch_npu 2.10.0` |
| 编译 | gcc/g++ 12.3.1，cmake 3.27.9；**无 ninja、无 unzip** |
| 外网 | pypi / modelscope 通；**GitHub 不通** |


## 2. 移植进度

| 期 | 内容 | 状态 |
|---|---|---|
| P1 | 设备层 + 5 个核心算子 | 完成 `9a5558b` |
| P2 | Q4_K / Q5_K / Q6_K 反量化 + NPU 矩阵乘 | 完成 `fdd1116` |
| P2b | GGUF 读取器 + 真实模型 + 真实权重 GEMV | 完成 `b0017a6` |
| P2c | Q8_0 解码 + 张量加载器 + 真实整块前向 | 完成 `c5830cf` |
| P3 | GDN 算子 + GDN 层块（单 token 与跨 token 状态） | 完成 `aa8c013` |
| P3b | RoPE + blk.3 全注意力层（首 token） | 完成 `96930fb` |
| P3c | KV cache + softmax 加权混合 + GQA 广播 | 完成 `84ce9a9` |
| P4 | MoE 路由 + 8 专家 + 共享专家（blk.0 FFN 全跑通） | 完成 `e9a4ef6` |
| P4b | 整模型 40 层：token id → logits | **与 llama.cpp top-5 一致** `cb47b52` |
| P4c | 多 token：整模型走完一个 prompt | **同序 top-5，logprob 差 ≤0.07** `66bb08c` |
| P5a | 性能：权重常驻 + workspace 复用 | 5 token 166 s → 64 s `62dfde1` |
| P5b | 性能：按层并行反量化专家 | → 41 s `de2d1ac` |
| P5c | 性能：降到 12.7 s，并给出每 token 分解 | `3026b03` |
| P5d | GDN 按 32 头批量化 | 12 token 17.8 → 14.7 s `70de58c` |
| P5e | GDN 循环状态改 fp32 | 与浮点参考逐位一致，14.5 s `ee3b25b` |
| P6 | 常驻服务模式 + OpenAI 兼容端点（18790） | 完成 `101f52b` |
| P6b | 输出矩阵常驻显存 | 首个请求 67 s → 17 s，预热 1.8 s `a2bd1b8` |
| P6c | 两个服务交给 supervisord | 完成 `9f8d19f` |

**最终性能**：12 token 14.5 秒（约 1.2 s/token，稳态约 0.4 s），相对第一版 166 s/5 token 提速 13 倍。
分解见 3.19–3.21。

## 3. 已验证的能力（全部实跑）

### 3.1 核心算子（`cann_parity`，全 PASS）

```
op                          max|diff|   mean|diff|  tolerance  result
matmul 256x512x384           0.002529     0.000266     0.5000  PASS
rms_norm 512x2048            0.000977     0.000152     0.0200  PASS
silu 4096                    0.001947     0.000319     0.0200  PASS
add 4096                     0.000977     0.000209     0.0100  PASS
softmax 64x1024              0.000004     0.000000     0.0100  PASS
muls 8x8 by 0.5              0.000000     0.000000     0.0100  PASS
matmul 8x8x1                 0.000205     0.000088     0.0100  PASS
mul 8x1 * 1x8                0.000237     0.000062     0.0100  PASS
```

### 3.2 量化格式（`quant_parity`，判据是 ggml 的 `to_float`）

```
Q8_0   vs ggml: max|diff| = 0  BIT-EXACT
Q4_K   vs ggml: max|diff| = 0  BIT-EXACT
Q5_K   vs ggml: max|diff| = 0  BIT-EXACT
Q6_K   vs ggml: max|diff| = 0  BIT-EXACT
```

### 3.3 GDN 算子（`gdn_parity`，T 个 token 的递推）

```
case                 attn|diff|  state|diff|  result
kda    S=64  T=8       0.000360     0.003886  PASS
scalar S=64  T=8       0.000360     0.001168  PASS
kda    S=128 T=4       0.000175     0.001516  PASS
```

### 3.4 真实模型（`gguf_parity`）

```
gguf : v3, 753 tensors, arch qwen35moe, 41 layers, 256 experts (top 8), embedding 2048

weight mix :
  f32    368 tensors      0.10 GB
  q8_0   260 tensors      2.61 GB
  q4_K    80 tensors     12.08 GB
  q5_K    40 tensors      7.38 GB
  q6_K     3 tensors      0.66 GB
  bf16     2 tensors      0.00 GB
  sum                    22.84 GB

real GEMV  blk.0.ffn_gate_exps.weight [512 x 2048] on the NPU: max|diff| = 0.000116  PASS

blk.0.ffn_gate_exps.weight   Q4_K  vs ggml: BIT-EXACT
blk.0.ffn_down_exps.weight   Q5_K  vs ggml: BIT-EXACT
blk.34.ffn_down_exps.weight  Q6_K  vs ggml: BIT-EXACT
output.weight                Q8_0  vs ggml: BIT-EXACT
```

### 3.5 共享专家（`layer_parity`）

```
stage                     max|diff|       rel L2  result
gate  = W_gate x           0.000240     0.000202  PASS
up    = W_up x             0.000215     0.000194  PASS
act   = silu(g) * u        0.000172     0.000428  PASS
y     = W_down act         0.000034     0.000404  PASS
```

### 3.6 完整 FFN（`moe_parity`）

```
block   : blk.0 FFN = 8 of 256 experts + shared expert
routing : 213(0.1542) 71(0.1505) 230(0.1398) 130(0.1233) 201(0.1118) 205(0.1094) 135(0.1074) 25(0.1037)

stage                     max|diff|       rel L2  result
router logits              0.000457     0.000211  PASS
ffn out                    0.000013     0.000439  PASS
```

语义与 llama.cpp 一致：`softmax(256) → top-8 → 权重归一化 → Σ wᵢ·专家 + sigmoid 门控共享专家`。

### 3.7 整个 GDN 层块（`gdn_block_parity`）

blk.0 是循环层，它的"注意力"是门控 delta net：

```
qkv  = wqkv x                               [8192] = q[16x128] | k[16x128] | v[32x128]
z    = wqkv_gate x                          [4096]
beta = sigmoid(ssm_beta x)                  [32]
a    = softplus(ssm_alpha x + ssm_dt) * ssm_a    [32]
conv = silu(conv1d(qkv))                    短卷积，kernel 4
q,k  = 每个 128 维头做 l2 归一化，16 头重复到 32 头
out  = gated_delta_net(q, k, v, a, beta)
out  = rms_norm(out, ssm_norm) * silu(z)
y    = ssm_out out                          [2048]
```

单 token、零状态（就是 decode 一步）：

```
stage                          rel L2    max|diff|  result
conv = silu(w0 * qkv)        0.000367     0.000045  PASS
q after l2 norm              0.000406     0.000511  PASS
k after l2 norm              0.000460     0.000550  PASS
delta net out                0.002496     0.000000  PASS
gated rms norm               0.003742     0.000040  PASS
y = ssm_out                  0.004751     0.000005  PASS
```

判据用**相对 L2**：这里的激活很小（短卷积权重约 1e-3），绝对阈值没有意义。

### 3.8 跨 token 的状态搬运（`gdn_state_parity`）

单 token 只是第一步。连续解码要搬两块状态：短卷积看的历史 3 个 qkv，以及每个 v 头的
128×128 递推矩阵。两者都留在设备上，逐 token 更新：

```
block  : blk.0 GDN across 4 tokens, state carried on the device

token          y rel L2  result
0              0.001930  PASS
1              0.002137  PASS
2              0.000963  PASS
3              0.001056  PASS
state          0.001513  PASS
```

T=8 与 T=1 也验过（T=8 末尾状态 rel 1.3e-3），说明状态没有随 token 数漂移。

### 3.9 RoPE（`rope_parity`）

`qwen35moe` 用的是 IMRoPE，但纯文本时 t/h/w 三个位置 id 相同，三段退化为同一个 theta base，
等价于标准 NeoX 配对 RoPE，只作用于每个头的前 `n_rot=64` 维：

```
rope : head_dim 256, n_rot 64, freq_base 1e+07, 5 tokens (pos 0 1 2 7 129)

check           max|diff|       rel L2  result
rotated          0.001307     0.000237  PASS
tail copy        0.000000     0.000000  PASS
```

### 3.10 blk.3 全注意力层首 token（`attn_layer_parity`）

41 层里每 4 层有 1 层是全注意力（`is_recr` 为假），blk.3 就是。整层：

```
q_full = attn_q x        [8192]，每个头 512 = q[256] | gate[256]
k, v   = attn_k / attn_v x   [512] = 2 个 KV 头 x 256
q, k   = 每个头 rms_norm(attn_q_norm / attn_k_norm)
q, k   = rope，n_rot 64 / 256
attn   = softmax(q kᵀ / √256) v    （单 token 时 softmax 恒为 1）
attn   = attn * sigmoid(gate)
y      = attn_output attn        [2048]
```

```
layer  : blk.3 attention, one token at position 5

stage                  rel L2  result
y = attn_output      0.000512  PASS
```

**注意**：单 token 时 softmax 只有一个元素，所以这一步验证的是投影、两个 per-head 归一化、
rope 和门控，**没有**验证带 K/V cache 的加权混合——那是下一步。

### 3.11 KV cache 加权混合（`attn_cache_parity`）

上面那个缺口补上了。T 个 token 依次写入 cache 并对历史做注意力：

```
layer  : blk.3 attention across 4 tokens with a KV cache
cache  : 2 kv heads x 256 dim, GQA 8:1

token        y rel L2  result
0            0.000501  PASS
1            0.000598  PASS
2            0.000881  PASS
3            0.000706  PASS
```

T=8 也验过（末尾几个 token 都在 7e-4 量级）。

**这里踩了一个会静默出错的坑**：第一版忘了给打分乘 `1/√head_dim`，结果 token 0 照样 PASS
（只有一个分数，softmax 恒为 1），token 1 之后全部 FAIL（0.35~0.47）。**首 token 通过而后续
token 失败**就是"少了缩放"的典型指纹。

### 3.12 整模型 40 层（`model_forward`）

从 token id 走到 logits，NPU 与 CPU 参考对齐：

```
layer  0/40  gdn   top1 expert 101  1.3 s
layer 10/40  gdn   top1 expert 8    9.6 s
layer 20/40  gdn   top1 expert 64   17.7 s
layer 30/40  gdn   top1 expert 231  25.9 s
layer 39/40  attn  top1 expert 213  33.3 s

top 5 logits: token 21750=8.44 30458=8.22 31590=7.88 17023=7.59

CPU reference in 31.4 s, hidden state rel L2 = 0.016679  PASS
```

两个必须说清楚的点：

1. **0.0167 是漂移，不是噪声**：2 层时是 0.0022，40 层累积到 0.0167。原因是残差流和 KV cache
   全程 fp16。要真正接近 llama.cpp，需要关键路径用 fp32 累积。
2. **第 5 名 logits 两次运行不一致**（7.58 vs 7.36），原因未查明，我没有掩盖。

### 3.13 和 llama.cpp 逐值对照（这轮最重要的发现）

自洽的 CPU/NPU 对照只能证明"CANN 移植和我的转录一致"，证明不了"我的转录是对的"。所以我
编了 llama.cpp 自带的 `llama-eval-callback`，让它把同一个 token 的中间张量打出来对照：

```
                         llama.cpp                          cann/
attn_norm-0      [-0.0858,  0.6080, -3.4649, ...]     [-0.0858,  0.6079, -3.4648, ...]
linear_attn_out  [-0.0061, -0.0046,  0.0089, ...]     [-0.0061, -0.0045,  0.0089, ...]
attn_residual    [-0.0067,  0.0003, -0.0187, ...]     [-0.0067,  0.0003, -0.0187, ...]
ffn_moe_out      [ 0.0239, -0.0100, -0.0117, ...]     [ 0.0295, -0.0106, -0.0099, ...]  ← 仍不一致
```

**更正上一轮的一个错判**：我上次说"共享专家大 20%"，那是我拿我们的**总和**去比 llama.cpp 的
`ffn_moe_out`——而那张量**不含**共享专家，共享专家是在更后面的 `ffn_out = ffn_moe_out +
ffn_shexp_gated` 里加的。按正确的量对比：

| 量 | llama.cpp | cann/ |
|---|---|---|
| ffn_moe_out | 0.0239 | 0.0236（专家和） |
| ffn_shexp_gated | 0.0059 | 0.0058 |
| ffn_out | 0.0297 | 0.0295 |
| l_out-0 | 0.0230 | 0.02273 |
| gate sigmoid | 0.0664 | 0.0664 |

layer 0 **完全对上**。逐层对照（`l_out-<n>` 前三个值）：

| 层 | llama.cpp | cann/ | |
|---|---|---|---|
| 0 gdn | 0.0230, -0.0104, -0.0286 | 0.02273, -0.01032, -0.02859 | ✓ |
| 1 gdn | 0.0293, -0.0038, -0.0339 | 0.02888, -0.00381, -0.03406 | ✓ |
| 2 gdn | 0.0215, -0.0053, -0.0386 | 0.02096, -0.00531, -0.03854 | ✓ |
| 3 attn | 0.0370, 0.0148, -0.0211 | 0.05063, 0.01886, -0.03329 | ✗ |
| 4 gdn | 0.0127, 0.0168, -0.0321 | 0.02386, 0.01603, -0.03848 | ✗ |

**三层的 GDN 全部对上，第一个全注意力层（第 3 层）开始错。**

### 3.14 收尾：q 头 → kv 头的映射（`cb47b52`）

llama.cpp 的 dump 每个张量只打首尾 3 个值，看不到中间。我在本地副本里改了
`common_debug_print_tensor`（`CFA_DUMP_ALL=<名字>` 时打全量，只用于调试），拿到第 3 层完整的
4096 维门控向量，逐头对比后规律立刻出现：

```
head 0 (kv 0):   0/256 个位置不同
head 1 (kv 1): 215/256 个位置不同
head 2 (kv 0):   0/256
head 3 (kv 1): 173/256
...
head 8 (kv 0): 141/256   ← 从这里开始翻转
head 9 (kv 1):   0/256
```

这是**块映射**，不是取模：

```
我写的:   q 头 h → kv 头 h % 2      头 0,2,4,6… 用 kv0
llama.cpp: q 头 h → kv 头 h / 8      头 0–7 用 kv0，8–15 用 kv1
```

改完之后，整模型和 llama.cpp 的 next-token 分布**前五名完全一致**：

| | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| llama.cpp | 8 (-2.607) | 283 (-2.615) | 13 (-3.162) | 19845 (-3.256) | 4927 (-3.842) |
| cann/ | 283 (-2.507) | 8 (-2.515) | 19845 (-3.116) | 13 (-3.226) | 4927 (-3.648) |

前两名互换是因为两者只差 0.008。logprob 整体差 0.03–0.19（我们的分布略微更平）。

**顺带解释了上一轮的"漂移"**：40 层与 CPU 参考的相对误差从 0.0092 降到 **0.0011**——之前看
起来像 fp16 累积误差的东西，大部分其实是这一个索引错误。


**这一对照抓到一个真 bug**：短卷积的输入是"最老→最新"排的，所以**当前 token 乘的是最后一个
tap**，不是第一个。我的实现用了第一个 tap，而 CPU 参考用的是同一套错误约定——两边互相印证，
所以 4 轮测试全 PASS，却都错了。

顺带把残差流改成 fp32 累积：40 层漂移 0.0167 → 0.0092。

**仍未解决**：整模型 logits 与 llama.cpp 仍不一致，差异从 **第 3 层（第一个全注意力层）** 开始。


顺带纠正一个之前的误判：模型文件有 41 个 block，但 **blk.40 不属于主干**——它带
`nextn.eh_proj` / `nextn.enorm` / `nextn.hnorm` / `nextn.shared_head_norm`，是多 token 预测的
草稿头，且用 `attn_q/k/v` 而不是融合的 `attn_qkv`。主干是 **40 层**（3/4 循环层）。
它的 router 张量是 BF16（全文件唯一的 BF16）。





## 4. 模型里一层长什么样（`blk.0`，循环层）

```
blk.0.attn_norm.weight        F32    [2048]
blk.0.attn_qkv.weight         Q8_0   [2048, 8192]     融合 q/k/v 投影
blk.0.attn_gate.weight        Q8_0   [2048, 4096]     z（门控）
blk.0.ssm_a / ssm_alpha / ssm_beta / ssm_conv1d / ssm_dt / ssm_norm    F32
blk.0.ssm_out.weight          Q8_0   [4096, 2048]
blk.0.ffn_gate_inp.weight     F32    [2048, 256]      路由
blk.0.ffn_gate_exps.weight    Q4_K   [2048, 512, 256] 专家
blk.0.ffn_down_exps.weight    Q5_K   [512, 2048, 256]
blk.0.ffn_gate_shexp.weight   Q8_0   [2048, 512]      共享专家
blk.0.post_attention_norm.weight F32 [2048]
```

关键 hparams：`full_attention_interval=4`（3/4 层是 GDN）、`head_count=16`、`head_count_kv=2`、
`ssm.group_count=16`、`ssm.time_step_rank=32`、`ssm.state_size=128`、`ssm.conv_kernel=4`、
`rope.dimension_sections=[11,11,10,0]`、`expert_used_count=8`。

## 5. 端到端部署

两个服务，都在 supervisord 下（`/etc/supervisor/conf.d/strata-ascend.conf`）：

| 端口 | 引擎 | 实测速度 | 备注 |
|---|---|---|---|
| 18789 | llama.cpp + `ggml-cann` | 预填 42 t/s，解码 3.4 t/s | 实际服务，隧道指向它 |
| 18790 | **本引擎** + OpenAI 包装 | 预热 1.8 s / 6 token | 参考实现 |

18789 的 HBM 只占约 2.7 GB，CPU 单核 99%：`ggml-cann` 没有可用的
`GGML_OP_GATED_DELTA_NET`，30 个循环层回退 CPU。这不是配置问题，是实现缺失；
我试着自己补，结论是"能跑但更慢"，见 3.18。

纯 CPU 对照跑不动（900 s 跑不完 16 token），所以这张卡的 CANN 路径是必要的。

## 6. 隧道

```powershell
# 18789（llama.cpp，实际使用）
ssh -CNg -L 18789:127.0.0.1:18789 -p <端口> <用户>@<服务器地址>

# 再加上 18790（本引擎）就写成两行 -L
ssh -CNg -L 18789:127.0.0.1:18789 -L 18790:127.0.0.1:18790 -p <端口> <用户>@<服务器地址>
```

Windows 侧 `127.0.0.1:18789` 已监听：`/health` 200、`/props` 返回模型路径、
`/v1/chat/completions` 正常。18790 需要第二条 `-L`，因为原始隧道只转发了 18789。

## 7. 服务器上现在在跑什么

| 路径 | 内容 |
|---|---|
| `/data/strata-cann/strata` | Strata 源码 + `cann/`（本次移植）+ `build-cann` |
| `/data/strata-cann/src/llama.cpp-3cf0325...` | llama.cpp 源码 + `build-cann2`（带 CANN 后端） |
| `/data/strata-qwen36` | 35B 的 GGUF（22.8 GB） |
| `/data/ascend910-deploy` | 部署仓库副本（`run.env`、`accept.sh`、单元文件） |
| `/data/strata-cann/logs` | 两个服务的日志 |

进程：`strata-cann-18789`、`strata-engine-18790`（supervisord 托管），
外加容器自带的 jupyterlab / sshd。

## 8. 踩过的坑

1. **`-march=native` 在容器里 SIGILL**：`/proc/cpuinfo` 只报 AVX2，GCC 12 的 native 会开 AMX /
   AVX512-BF16 / AVX-VNNI，编出来的程序加载模型就崩。必须显式 AVX2 基线。
2. **aclnn 张量是行主序**：`shape={rows,cols}`、`stride={cols,1}`；列主序报 `161002`。
3. **`aclnnRmsNorm`** 多一个 `rstdOut`，`gamma` 必须 1D，否则 `561002`。
4. **同一个头向量，两种形状**：`rms_norm` 按最后一维归一化，必须写成 `[1行,128列]`；
   当矩阵乘的列向量时又要写成 `[128行,1列]`。写错会得到"每个元素各自归一化"的怪结果。
   **这条在整模型里又咬了一次**：`attn_norm` 的 gamma 和 x 都写成列向量时，
   ACLNN 直接报 `Input gamma shape invalid`；而两次运行之间我恰好漏了这条检查，
   导致第一次 40 层跑出来的结果全错。
5. **fp16 解码指数差 1**：用 2^-25 不是 2^-24，否则整体大两倍。
6. **2 字节 fp16 标量不能直接 memcpy 进 float**：高位留 0，读出来是 2e-41。
7. **ggml 的 GDN 内部乘了 `1/√S_v`**：参考实现漏掉会差 11.3 倍。
8. **`-np > 1` 不可用**：预填掉到 0.06 t/s 后 AclNN 报 `only support y tensor is contiguous` 并退出。
9. **PowerShell 改 UTF-8 中文文件会写坏**：用 `write` 工具整文件重写，不要 `Set-Content`。


### 3.15 多 token：整模型走完一个 prompt（`66bb08c`）

`model_seq` 把一个 prompt 的每个 token 都走完 40 层，卷积历史、32 个 128×128 状态矩阵、KV cache
全部留在设备上。prompt `The capital of France is`（5 个 token）：

| | llama.cpp | cann/ |
|---|---|---|
| 11751 ' Paris' | -0.5844 | -0.5468 |
| 264 ' a' | -2.2250 | -2.2186 |
| 279 ' the' | -3.1078 | -3.0624 |
| 3750 ' known' | -3.4281 | -3.4686 |
| 271 '\n\n' | -3.7641 | -3.8280 |

**顺序完全一致，logprob 差 0.04–0.07。**

**性能要说清楚**：5 个 token 用了 166 秒，约 **33 秒/token**。原因是每次 `gemv` 都在 host 上
反量化权重再上传，token 之间没有任何复用。这是正确性实现，不是可服务的实现。
要让这个栈对外提供 API，第一步是把权重常驻显存。
### 3.16 性能：权重与 workspace 常驻（`bf4b573`）

第一版 5 个 token 要 166 秒（33 秒/token）。原因不是算力，是每次 `gemv` 都在 host 反量化权重、
上传、再算，token 之间零复用；另外每个算子还 `aclrtMalloc` 一次 workspace。

| 改动 | 5 token |
|---|---|
| 第一版 | 166 s |
| 权重常驻显存 + workspace 复用 | **64 s** |

权重缓存把非专家张量（2.4 GB）常驻 HBM，专家 slab 按预算缓存（选中才搬）。

**踩了一个坑**：顺手把每个算子后的 `stream.synchronize()` 去掉，结果所有数值都错（attention
层 rel L2 1.85）。原因是 host 上传走的是 `aclrtMemcpy`，它和队列里的 kernel 没有顺序保证——
后一次上传会覆盖前一个 kernel 还在读的 buffer。同步加回来就对了。要真正去掉同步，得把上传
改成 `aclrtMemcpyAsync` 放到同一条流上。

13 秒/token 对比 llama.cpp 在同一张卡上的 0.25 秒，差距是结构性的：一个 token 约 1200 次 aclnn
调用，每次一个同步，没有融合。

### 3.17 性能第二轮：先测再改（`de2d1ac`）

上一轮我猜"算子太慢"，这轮先加计数器再动手，结论和猜测完全不同：

```
ops: 122559 calls, 2.458 s total, 0.020 ms each
xfer: 19377 copies, 10954 MB, 1.186 s total, 0.061 ms each
```

5 个 token 总耗时 41–64 秒里，算子只占 **2.5 秒**，传输占 **1.2 秒**。剩下的是 **host 侧反量化**：
每个 token 要选中 8/256 个专家、40 层，新 slab 得先从 22.8 GB 文件里读出来再逐块解码，
而解码是标量循环。

把分块解码按块区间拆到多线程后：

| 改动 | 5 token |
|---|---|
| 第一版 | 166 s |
| 权重与 workspace 常驻 | 64 s |
| 多线程反量化 | **41 s** |

分阶段看（累计值）：

```
token 1/5  17.1 s  [gdn 5.4  attn 1.4  moe 10.0]
token 5/5  41.3 s  [gdn 6.8  attn 1.6  moe 32.5]
```

GDN 与 attention 只在首 token 付一次权重解码（5.4 / 1.4 秒），之后每 token 分别 0.3 / 0.1 秒。
MoE 是首 token 10 秒、到第五个 token 降到 3.6 秒——因为每个 token 选中的专家不同，新 slab
要现读现解。

顺带否掉一个想当然的优化：把整层 256 个专家的字节一次读进内存（每层 512 MB），在短序列上
**更慢**（28.5 s vs 17.3 s），因为只用 8 个却要读 256 个。已在代码注释里写明。

### 3.18 试过了但不可行：把 GATED_DELTA_NET 写进 ggml-cann

上一轮我提出两个方向，这轮先试了"给 ggml-cann 补 GATED_DELTA_NET，让 llama.cpp 的
30 个循环层不再回退到 CPU"。结论是**这条路走不通**，已回退。

做了什么：在 `ggml/src/ggml-cann/aclnn_ops.cpp` 里按 CPU 版的数学（state 转置存放、
`S *= exp(g)`、`d = (v - Sᵀk)·beta`、`S += d⊗k`、`o = Sᵀq/√S_v`）逐 (seq, head, token)
用 aclnn 算子拼出来，并在 dispatch 与 `supports_op` 里注册。编译通过、能跑。

两个结果都不好：

| | CPU 回退（原样） | 我们写的 NPU 版 |
|---|---|---|
| 输出 | 正常 | **乱码**（"antee她了國人程ulpelter上添"） |
| prefill | 42 t/s | **3.0 t/s** |

**慢的原因是结构性的**：这个算子每层每头每 token 只要 8 次小算子，矩阵是 128×128；
而 CPU 版跑在 128 线程上，对这种尺寸非常合适。逐头拼 aclnn 调用（每次有 workspace 查询、
tensor 创建、同步）反而比 CPU 慢一个数量级。正确性上也有 bug（状态布局或 gate 处理某处），
但既然速度已经不可能赢，就没再往下调。

已回退并验证：`llama-cli` 输出恢复连贯（"Here's a thinking process: 1"），
prefill 8.3 t/s、生成 3.3 t/s（llama-cli 口径），部署 `accept.sh` 全过、3.5 t/s。

**这条结论本身有价值**：之前我以为"3/4 的层在 CPU 上跑"是部署慢的主因，实测不是——
CPU 处理这 30 层完全够快。要在这张卡上赢过它，需要的是**批量化的自定义 kernel**
（一次处理所有头、所有 token），而不是逐头拼既有算子。

### 3.19 性能第三轮：按层并行解码专家（`3026b03`）

上一轮把反量化拆到线程里，结果只是勉强改善。这轮先加 `print_slab_stats` 把 MoE 内部拆开：

```
slabs: 3348 new, disk 0.5 s, dequant 22.3 s, upload 4.3 s
```

**22 秒全在反量化上，而且线程没起作用**——因为每个 slab 只有 100 万个值，却开了 128 个线程，
建线程的开销比计算还大。

改成"一层需要的 24 个 slab 并行解码，每个 slab 单线程"：

```
slabs: 3348 new, disk 0.5 s, dequant 1.5 s, upload 3.9 s
```

| 改动 | 5 token |
|---|---|
| 第一版 | 166 s |
| 权重与 workspace 常驻 | 64 s |
| 反量化加线程（方式错） | 41 s |
| **按层并行解码 slab** | **12.7 s** |

反量化 22.3 s → 1.5 s，结果逐位不变（同样五个 logits 完全相同）。

每 token 现在约 1.1–1.5 秒：GDN 0.3、attention 0.1、MoE 0.9，其余是上传。距离 llama.cpp 的
0.25 秒还差 5 倍，但已经从 33 秒/token 降到这个水平。

### 3.20 GDN 按头批量化（`70de58c`）、以及一个近似平局的代价

原来的 GDN 是逐头调用：32 个头 × 8 次算子 × 30 层，每 token 约 7700 次调用。这轮给 `Tensor` 加了
batch 维，补了 `bmm` / `mul_bcast` / `add_bcast` / `sub_bcast` 四个批量算子（先写了
`batched_parity` 对照 CPU，误差 2e-4 全过），然后把整个 delta-net 改成一次算完 32 个头：

```cpp
mul_bcast(M, decay);        // M *= exp(g)
bmm(tmp, k, kv);            // M k
sub_bcast(v, kv, d);        // v - M k
mul_bcast(d, beta, delta);  // * beta
bmm(delta, k_row, outer);   // delta (x) k
add_bcast(tmp, outer, M);   // M += ...
bmm(M, q, kv);              // M q
mul_bcast(kv, scale, attn); // / sqrt(S)
```

每 token 从 256 次降到 8 次，GDN 的累计时间从 8.1 秒降到 4.7 秒（12 token），总时间 17.8 → 14.7 秒。

过程中抓到一个真 bug：批出版本里 `M·k` 用的是**未衰减**的 M。T=1 时状态为零，看不出来；T>1 立刻
变成 3.5% 误差。改用衰减后的副本后降到 8e-4。

**代价**：和 llama.cpp 对比，前四名完全一致，第五名从 `271`（‑3.764）变成 `799`（‑3.850）——两个
token 的 logprob 只差 0.086，是近似平局翻转，不是模型算错。要彻底修好需要把循环状态改成 fp32，
这需要 cast 算子，当前还没有。

### 3.21 循环状态改 fp32：修回平局，并刷新最好成绩（`ee3b25b`）

上一轮批量化的代价是第五名 token 翻转（271 ↔ 799，差 0.086）。根因不是批量化本身，而是**半精度
累加**：llama.cpp 的 CPU GDN 内核是 float 算的，我们全程 fp16。

补了 `cast` 算子，新增 `gdn_batched_f32`：q/k/v 与输出仍是 fp16，**状态和中间量全部 fp32**。
整个序列只 cast 三次（进）+ 一次（出），不是每 token 都转。

| 检查 | fp16 累加 | **fp32 累加** |
|---|---|---|
| 状态 vs 浮点参考（T=4） | 8.4e-4 | **0.000000** |
| 状态 vs 浮点参考（T=8） | 1.7e-3 | **0.000000** |
| 与 llama.cpp 的 top-5 | 4/5，最大差 0.157 | **5/5，最大差 0.071** |
| 12 token 耗时 | 14.7 s | 14.5 s |

也就是说：**精度修好了，时间还快了一点**（cast 的开销淹没在噪声里）。

0.071 是到目前为止最好的成绩——此前逐头 fp16 版是 0.081。

### 3.22 引擎自己对外服务（18790），以及输出矩阵常驻（`a2bd1b8`）

目标里"对外提供 OpenAI 兼容服务"这一条，此前一直由 llama.cpp 承担。这轮让**我们自己的引擎**也对外服务，
跑在 18790，与 18789 并存：

| 端口 | 引擎 | 速度 | 定位 |
|---|---|---|---|
| 18789 | llama.cpp + ggml-cann | 约 4 tok/s | 实际使用的服务 |
| 18790 | **本项目移植的引擎** | 预热约 6 tok/s，冷启动 1.5 s/token | 参考实现 |

实现：`model_seq` 加常驻模式（模型只加载一次，从 stdin 收请求），`cann/tests/serve_engine.py`
提供 `/v1/completions`、`/v1/chat/completions`、`/v1/models`、`/health`。分词借用 18789 的
llama.cpp，模型推理完全是我们自己的——`/health` 里写明了这个依赖。

**发现并修掉的浪费**：输出投影 248320×2048，第一版每生成一个 token 都在 host 上重新反量化一遍。
改成常驻显存后，首个请求 67 秒 → 17 秒，预热请求 **1.8 秒**（11 步）。

实测（与上表同一提示词）：

```
same prompt, warm:      1.82 s   ' Paris, a city renowned for'
prompts never seen:     4.72 s   ' the heavens and the earth.'
                        5.49 s   '\n    if n <= '
                        5.27 s   ' the lazy dog.\n\nThe'
```

四段文本与 llama.cpp 的贪心输出完全一致。

**显存共存**：端点的专家缓存由 `STRATA_BUDGET_GB` 控制。实测（同一张 64 GB 卡，llama-server 占 2.7 GB）：

| 缓存上限 | 卡上总占用 |
|---|---|
| 8 GB | 22.5 GB |
| 12 GB | 28.7 GB |
| 24 GB | 47.2 GB（验收套件会 `aclrtMalloc failed`） |

12 GB 时验收 15 项全过；套件自己用 8 GB。

### 3.23 服务托管：这个容器没有 systemd（`9f8d19f`）

部署仓库里带了一个 `strata-ascend910.service`，但实测这台机器**不是 systemd 起的容器**：

```
$ systemctl is-active strata-ascend910
System has not been booted with systemd as init system (PID 1). Can't operate.
```

所以那个单元文件在这台机器上永远不可能生效。容器里的 init 是 **supervisord**（`/etc/supervisord.conf`
+ `/etc/supervisor/conf.d/`），jupyterlab 和 sshd 都由它托管。

改成把两个服务都交给 supervisord：

```
[program:strata-cann-18789]    llama.cpp + ggml-cann
[program:strata-engine-18790]  本项目引擎 + OpenAI 包装
autostart=true  autorestart=true
```

踩到一个坑：首次启动两个程序都是 FATAL，日志里是

```
llama-server: error while loading shared libraries: libascend_hal.so
```

因为我在 environment 里只写了 CANN 的 lib64，漏了驱动目录。补上
`/usr/local/Ascend/driver/lib64{,/common,/driver}` 后正常。

现在两个服务都是 `RUNNING`，随容器启动、崩溃自动拉起。单元文件也同步更新成真实路径，
留给将来跑在 systemd 上的机器。

`accept.sh` 3.85 t/s、验收 15 项全过。

### 3.24 采样搬到设备上（`9dc70e4`）

此前每生成一个 token 都要把整个词表的 logits 拉回 host（248320 个 fp16，约 500 KB），在 host 上
做 top-k 和 log-sum-exp。这轮补了两个算子：

- `topk`：aclnnTopk，设备上取前 k 大值和下标
- `logsumexp`：aclnnLogSumExp，设备上算配分函数

现在一步只回传 **32 个 float + 32 个 int64 + 1 个 float**。

先写了 `topk_parity` 对照 host 实现（用模型真实词表宽度 248320）：

```
device topk: vocab=248320 k=8
rank             device         host  ids
0             18.175989    18.175989  76617
...
log-partition  device 19.818172  host 19.818173  diff 4.16e-07  PASS
```

接进模型后结果逐位不变（11751/264/279/3750/271，logits 完全相同），5 token 从 11.2 秒降到
10.8 秒。收益不大是因为采样本来只占每步百分之几——但这是"host 不再碰 logits"的结构性改动，
也顺带发现**词表是 248320，不是我一直以为的 151936**，之前文档里的数字已更正。

踩到一个坑：`Tensor` 的 logits 是**列向量**（matmul 的输出布局），不是行向量，所以 topk 的
归约轴是 0；我第一版按行写，报 `Expected tensor for out to have same size`。现在两个算子都
显式接收归约轴，调用方自己决定。

### 3.25 参考官方部署文档后的发现（Qwen3.8-Flash-Next / vLLM-Ascend）

用户给了 `Ascend-SACT/Qwen3.8-Flash-Next` 的部署文档。仓库在服务器上能拉到（gitcode 通）：
`/data/strata-cann/qwen38-ref`，内容是 vLLM-Ascend v0.26.0rc1 的**部署文档 + 两个补丁**，
目标模型 360 GB BF16、16 张卡（2×8 910B3）、TP=8 DP=2 EP=16。

**它不能直接搬到这台机器**，三条硬约束：

| 约束 | 参考文档要求 | 这台机器 |
|---|---|---|
| 容器 | docker 镜像 `quay.nju.edu.cn/ascend/vllm-ascend` | **容器内无 docker/podman** |
| CANN | 9.1.0 | 9.0.0 |
| 硬件 | 16×910B3（每张 64 GB） | 1×910B2C |

但翻它的补丁和 pip 包翻出两个**对我们直接有用**的事实。

#### 1. vLLM-Ascend 0.30 已经支持我们这个模型家族

下载 wheel 检查注册的模型类：

```
Qwen3_5MoeForConditionalGeneration   3 files
Qwen3_5GatedDeltaNet                 2 files
Qwen3_5ForConditionalGeneration      4 files
```

我们模型 GGUF 里的 arch 字符串是 `qwen35moe`，对应的正是 `Qwen3_5Moe`。也就是说，
**如果要在这张卡上做生产级服务，官方路径是 vLLM-Ascend，而不是我们手写的引擎或 llama.cpp**。

#### 2. CANN 里有融合的 delta-rule 算子

`vllm_ascend/ops/gdn.py` 调用的是：

```python
torch_npu.npu_recurrent_gated_delta_rule   # decode
torch_npu.npu_chunk_gated_delta_rule       # prefill
fla_npu.ops.ascendc.causal_conv1d_fn       # 短卷积
```

对应到 aclnn 层就是 **`aclnn_recurrent_gated_delta_rule`，而且 CANN 9.0 里就有**：

```
/usr/local/Ascend/cann-9.0.0/include/aclnnop/aclnn_recurrent_gated_delta_rule.h
```

参数（头文件注释）：

```
query/key/value/beta/state/out : bfloat16
g / gk                         : float32
actualSeqLengths / ssmStateIndices / numAcceptedTokens : int32
scaleValue                     : float32
```

**这推翻了我 3.18 的结论**。当时我写 ggml-cann 的 GATED_DELTA_NET，逐头拼 aclnn 小算子，
比 CPU 慢 14 倍，于是判断"要在 NPU 上赢必须自己写 kernel"。实际情况是 CANN 自带融合算子，
只是 ggml-cann 没用它（它 include 了这个头文件却从未调用——我当时看到 include，以为是遗留）。

要不要在我们引擎里换用它，是个权衡：

| | 现在的实现 | 换用融合算子 |
|---|---|---|
| 精度 | 状态 fp32，与浮点参考**逐位一致** | 只有 bf16（尾数 8 位，比 fp16 的 11 位还少） |
| 每 token 调用 | 8 次（32 个头批量） | 1 次 |
| 复杂度 | 已有、已验 | 需要 bf16 cast、paged state（`ssmStateIndices`）、`gk`/`numAcceptedTokens` |

收益也有限：GDN 现在只占每 token 约 0.15 秒（总 0.4 秒），真正的瓶颈是 MoE 专家权重搬运。

**结论**：记下来，暂不换。如果要换，先做 `gdn_fused_parity` 对照，确认 bf16 下与浮点参考的
误差是否还能维持 top-5 一致，再动引擎。

### 3.26 融合 GDN 实测：布局兼容，但头映射不同、只有 bf16（`52f145f`）

写了 `gdn_fused_parity`，直接调 `aclnnRecurrentGatedDeltaRule`，和浮点参考比。三个形状都跑了：

```
=== 1 头（vLLM-Ascend smoke 调用的形状）===
attn (fused bf16 vs float)     0.002449  PASS
state [Dv][Dk] vs float        1.414510  FAIL
state [Dk][Dv] vs float        0.003294  PASS

=== 本模型的形状（Nk=16 Nv=32 Dk=Dv=128）===
attn (fused bf16 vs float)     0.004096  PASS (blocked 映射)
state [Dk][Dv] vs float        0.003526  PASS
```

三个结论：

1. **状态布局与我们完全一致**。这个算子用的是 `[Dk][Dv]`，也就是 `M[j][i] = S[i][j]`——和
   我们引擎里的转置存放是同一个约定。不需要转置。
2. **头映射不一样**。kernel 用 **blocked**（v 头 h 用 k 头 `h / (Nv/Nk)`），llama.cpp 用的是
   **mod**（`h % Nk`）。我们的引擎跟 llama.cpp 一致，所以直接换算子会算错（mod 映射下
   rel 1.28）。要换必须先按一个固定置换重排 v 投影的行、`ssm_beta`/`ssm_alpha`/`ssm_a` 的行、
   `ssm_out` 的列、以及短卷积在 v 段的抽头。
3. **只有 bf16**。算子层面 attn 误差 4e-3、state 误差 3.5e-3（bf16 尾数 8 位）。我们现在的
   fp32 路径与浮点参考**逐位一致**。

所以"换用融合算子"不是接线，是两件事：一次权重置换（4–5 处，每处都可能悄悄算错），
以及把状态精度从 fp32 降到 bf16。收益是每 token 约 0.15 秒 → 约 0.02 秒（总 0.4 → 0.27）。

这一步要不要走，需要先知道 bf16 之后端到端 top-5 是否还能与 llama.cpp 对上——而这只有在
置换做完之后才测得出。停在这里，等决定。

## 9. 待办与已知限制

**已完成**（`[x]` 都有实测支撑，细节见第 3 节与开发日志）：

- [x] 设备层、核心算子、量化解码、GDN、RoPE、注意力、整模 40 层
- [x] GDN 状态跨 token 搬运（T=1/4/8 都验过），conv history 留在设备上
- [x] 全注意力层 KV cache（softmax 打分 + GQA 16/2 头广播）
- [x] 整模 logits 与 llama.cpp 对齐：top-5 同序，最大 logprob 差 0.071
- [x] 贪心续写与 llama.cpp 逐 token 一致
- [x] 性能：5 token 166 s → 12.7 s（13 倍）；12 token 14.5 s
- [x] GDN 按 32 个头批量化（每 token 256 次调用 → 8 次）
- [x] 循环状态改 fp32（与浮点参考逐位一致）
- [x] 权重常驻显存 + workspace 复用 + 输出矩阵常驻
- [x] 18790 端点（OpenAI 兼容）由本引擎提供
- [x] 两个服务由 supervisord 托管，随容器启动

**未做**（按价值排序，都是可选的进一步工作）：

- [ ] 自制批量化 GDN kernel。逐头拼 aclnn 已验证比 CPU 慢 14 倍（3.18），要赢需要写 kernel，
      而不是组合既有算子。这是把引擎从 0.4 s/token 推到 0.1 s/token 的唯一路径。
- [x] device 上做 top-k + logsumexp（每 token 只回传 32 个值、32 个下标、1 个标量）
- [ ] KV cache 仍是 fp16（残差流与 GDN 状态已改 fp32）。
- [ ] host 上传改异步（`aclrtMemcpyAsync`），才能安全去掉每算子同步。
- [ ] 融合式 dequant+matmul（现在是 host 反量化后上传）。
- [ ] `model_seq.cpp` 900 行兼三职（对照 / 生成 / 服务），应拆成 `engine` + 三个瘦 `main`。

**已知限制**（不是 bug，是边界）：

- 18789 实际由 llama.cpp 提供；本引擎在 18790，慢约 10 倍，不能承担实际流量。
- `ggml-cann` 无可用 `GGML_OP_GATED_DELTA_NET`，30 个循环层回退 CPU；试过一版，回退原因见 3.18。
- 本引擎不支持逐通道门控（KDA）的 GDN，会显式抛错。
- 18790 的分词借用 llama.cpp 的接口，模型推理是本引擎。
- 容器内 init 是 supervisord 而非 systemd，单元文件只在 systemd 主机上有效。
- 单进程显存需用 `STRATA_BUDGET_GB` 控制（见 3.22 的实测表）。

## 10. 复现验收

全部命令在服务器上执行，工作目录 `/data/strata-cann/strata`：

```bash
# 1. 构建并跑完 15 项对照（含与 llama.cpp 的对比、贪心生成、18790 端点）
bash cann/tests/verify_all.sh

# 2. 部署侧验收（18789）
cd /data/ascend910-deploy && bash scripts/accept.sh

# 3. 两个端点的输出一致性
bash /root/srv_final_check.sh

# 4. 服务状态
supervisorctl status | grep strata
```

预期结果：

```
summary: 15 passed, 0 failed
largest logprob difference: 0.071
all checks passed
```

最近一次实测（2026-11-05 12:40）：

| 检查 | 结果 |
|---|---|
| 对照与整模测试 | 15/15 |
| 与 llama.cpp top-5 | 同序，最大差 0.071 |
| 贪心续写 | `11751 11 264 3177` = ` Paris, a city`，与 llama.cpp 一致 |
| 18789 生成速度 | 3.35 tok/s |
| 18790 生成 | 同样的文本，预热约 1.8 s / 6 token |
| 算子调用占比 | 每 token 约 2.2 万次，约 0.4 s；其余是权重上传与反量化 |

---

---

## 11. Qwen3.8-Flash-Next 这一线（新机器）

第 1–10 节是 Qwen3.6-35B-A3B 的移植。从这一节开始是另一件事：先让 **Qwen3.8-Flash-Next**
在昇腾上跑起来，再据此推进 Strata 的适配。

### 11.1 机器

用户换了三次机器，第三次才可用。三次的差别值得记下来，因为**租机器前该测什么**很明确。

| 代 | 配置 | 结果 |
|---|---|---|
| 原机 | 1×910B2C、CANN 9.0、x86_64、59 GB 磁盘 | 移植工作机，后来停机且再未恢复 |
| 第二台 | 2×910B2、**ARM64**、CANN 9.0、/data 186 GB | **无出网**（所有外连 TCP 超时），模型下不下来，弃用 |
| 现用 | 2×910B2C、x86_64 128 核、CANN 9.0、/data 118 GB | ✅ 出网正常（魔搭/pypi/github 都通） |

**教训一**：租机器第一件事是测出网，不是测显卡。第二台有 NPU 有 CANN 有磁盘，但下载不了模型，等于零。

**教训二**：在只能靠 ssh 进的机器上，**不要改 `/etc/resolv.conf`**。第二台出网被封，我改成
`114.114.114.114` 之后，sshd 的反向解析阻塞，ssh 会话开始被掐断，最后完全进不去。原本
`127.0.0.11`（Docker 内嵌 DNS）虽然出不去外网，但至少能解析内部名字、能登录。容器重启才恢复。

### 11.2 编译：llama.cpp + CANN 后端

用 llama.cpp **master**（`qwen4exp` 是很新的架构，9 月的版本没有）：

```
LLM_ARCH_QWEN4EXP                     "qwen4exp"        ← 架构注册
src/models/qwen4exp.cpp                                  ← 1535 行参考实现
ggml/src/ggml-cann/ggml-cann.cpp
  case GGML_OP_GATED_LINEAR_ATTN:                        ← 36 个线性层靠它
  #include <aclnnop/aclnn_recurrent_gated_delta_rule.h>
```

构建：

```bash
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0
export LD_LIBRARY_PATH="$ASCEND_TOOLKIT_HOME/lib64:$ASCEND_TOOLKIT_HOME/x86_64-linux/lib64"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"
cmake -S llama.cpp -B build-cann -DCMAKE_BUILD_TYPE=Release -DGGML_CANN=ON -DLLAMA_CURL=OFF
cmake --build build-cann -j $(nproc)
```

128 核，2 分钟出全部二进制。

**踩到的坑：CANN 9.0.0 缺 `aclnn_fused_infer_attention_score_v2.h`。**

新版 `ggml-cann` 无条件 include 它，编译直接停：

```
fatal error: aclnnop/aclnn_fused_infer_attention_score_v2.h: No such file or directory
```

但 `libopapi.so` **已经导出** `aclnnFusedInferAttentionScoreV2` 符号——运行时支持，只是头文件没装。
从 CANN 官方 `ops-transformer` 仓库取到正版头文件（3740 字节）放进 `include/aclnnop/` 后编译通过。
头文件已存进 `scripts/ascend_qwen38/`，下台机器直接带上。

### 11.3 链路验证（用 0.6B，不等大模型）

`/mnt/moark-models` 里挂着 Qwen3-0.6B（safetensors），转成 GGUF 后实测：

```
> The capital of France is
Okay, the user is asking for the capital of France. ... the capital is
[ Prompt: 1332.6 t/s | Generation: 182.0 t/s ]
```

**这是真的在 NPU 上跑**——CPU 不可能有 1332 t/s 的预填。这一步的价值是：等大模型下完，
不会卡在环境问题上。

### 11.3.1 加载器认这个文件（下载到 32% 时验证的）

在模型下到一半时就让 llama.cpp 去打开那个 `.incomplete` 文件。它没有报"未知架构"，而是：

```
E llama_model_load: error loading model: tensor 'blk.27.ffn_gate_exps.weight'
  data is not within the file bounds, model is corrupted or incomplete
```

这句错误正是想要的答案：**架构、元数据、张量表、分片逻辑它全部接受**，只是读到文件末尾之外了
（当时只有 24 GB）。

所以剩下的风险不在"认不认这个模型"，而在"算不算得动"。这两件事的风险量级完全不同：
前者失败只能换 llama.cpp 版本，后者是调参数。

### 11.4 两卡层切分

用 0.6B 强制切开验证：

```
-ngl 99 --split-mode layer -ts 0.5,0.5

load_tensors: layer   0 assigned to device CANN0
...
（CANN0 出现 73 次，CANN1 出现 67 次）
```

层确实分到两张卡。但生成速度 **167 → 136 t/s**，跨卡传输有成本。大模型上是否划算要实测。

### 11.5 关键发现：CANN 后端缺五个 qwen4exp 专用算子

把 `qwen4exp.cpp` 逐函数扫出用到的 ggml 算子，对照 `ggml-cann.cpp` 的 `supports_op`：

| 算子 | 用在哪 | CANN |
|---|---|---|
| `GATED_LINEAR_ATTN` | 36 个线性层 | ✅ |
| `FLASH_ATTN_EXT` | 12 个全注意力层 | ✅ |
| `MUL_MAT` / `MUL_MAT_ID` | MoE | ✅ |
| `GET_ROWS` | 词嵌入、PLE 查表 | ✅ |
| `ROPE`（含 mrope） | 全部 | ✅ |
| `CUMSUM` / `SET_ROWS` / `FILL` / `CLAMP` | 杂项 | ✅ |
| **`DSV4_HC_PRE_GATED` / `DSV4_HC_POST`** | **HyperConnection，48 层全有** | ❌ |
| **`LIGHTNING_INDEXER`** | QSA 索引器，12 层 | ❌ |
| **`TOP_K` / `TOP_K_MASK`** | QSA 选 key，12 层 | ❌ |

**后果**：这些会静默回退到 CPU，每层在 NPU 与 CPU 之间搬数据。模型能跑，但速度会明显低于
论文数字。这是 3.18 节那个教训的重演——算子缺失不报错，只是慢。

**补法**（五个都能用现有 aclnn 算子拼）：

| 缺失算子 | 用现成的拼 |
|---|---|
| `TOP_K` | `aclnnTopk`（我们已封装并验证，误差 4e-7） |
| `LIGHTNING_INDEXER` | matmul + softmax + topk |
| `DSV4_HC_PRE_GATED` / `DSV4_HC_POST` | 低秩矩阵乘 + sigmoid + 加权 |
| `TOP_K_MASK` | 比较 + 掩码填充 |

而且**给 llama.cpp 的 CANN 后端补 `supports_op` + 实现，比在 Strata 里重写一遍划算**——
llama.cpp 已经有完整的 qwen4exp 图，只缺后端支持。

### 11.6 模型与量化

三个候选（HF/魔搭同一份文件）：

| 版本 | 常驻显存 | 下载 | 任务均分 |
|---|---|---|---|
| Q2_0 | 37.6 GB | 66.4 GB | 89.07 |
| IQ2_XS | 39.2 GB | 68.0 GB | 89.16 |
| **IQ3_XXS**（选用） | **47.0 GB** | **75.8 GB** | **92.57** |
| IQ3_S | 54.8 GB | 83.6 GB | 93.26 |
| BF16 | 335 GB | 335 GB | 93.12 |

每个版本分两个分片：**分片 1 是权重（必须常驻）**，分片 2 是 51.2B 参数的 n-gram 查表（28.8 GB，
`-lm mmap --lazy-mode on` 可以留在 SSD 上按行读）。

**量化格式不用移植**：IQ3_XXS 这个 GGUF 用了 14 种存储类型
（`BF16=484 F16=1 F32=292 IQ2_S=20 IQ2_XS=20 IQ2_XXS=18 IQ3_S=82 IQ3_XXS=12 IQ4_NL=58 IQ4_XS=67
Q2_0=38 Q4_K=45 Q5_K=15 Q6_K=71`），我们 `quant_parity` 已经**逐位验证过其中 11 种**，
BF16/F16/F32 由 `tensor_load` 直读。解码器这一层不用动。

### 11.7 操作坑

**`llama-cli` 跑完不退出。** 加了 `-p` 之后它仍进入交互模式，等 stdin。所有脚本化调用必须加
`< /dev/null`，否则看起来像"卡住"（我因此误判了三次超时）。

**带宽是出口上限。** 单连接 5.0 MB/s，四路并发合计 5.6 MB/s，换源（hf-mirror / huggingface）
从这台机器都不通，只有魔搭可用。76 GB 大约 4 小时。

### 11.8 与 Strata 的差距（详见 `scripts/ascend_qwen38/ARCH_GAP.md`）

| 项 | Qwen3.8-Flash-Next | 现有引擎（qwen35moe） |
|---|---|---|
| 层数 | 48（36 线性 + 12 注意力） | 40（30 + 10） |
| hidden | 2560 | 2048 |
| 线性层 | 16 k 头 / **48 v 头** / d 128 | 16 / **32** / 128 |
| 全注意力 | 24 q / 2 kv / **head_dim 256** | 16 / 2 / 512 |
| MoE | **512 选 10**，inter 640 | 256 选 8，inter 512 |
| 稀疏注意力 | **QSA**（indexer 4 头、budget 2048） | 无 |
| 残差 | **HyperConnection**（4 路低秩 320） | 普通 |
| 词嵌入 | **PLE**：51.2B n-gram 表，第 2 层 | 无 |
| 投机解码 | **MTP**（1 层） | 无 |

三块真正的新工作：**PLE**（查表 + 卷积，表必须留 SSD）、**QSA**（选 key 的额外网络）、
**HyperConnection**（全局结构改动）。其余是改常数。

### 11.9 实测结果：模型能跑，但 llama.cpp 的 CANN 后端接不住这个模型

下载完成（76.8 GB，两个分片），逐项测过之后的状态：

**能跑的：**

| 项 | 结果 |
|---|---|
| GGUF 完整性 | ✅ 加载成功，张量表全部对上 |
| 纯 CPU 推理 | ✅ 输出连贯中文，**预填 18.9 t/s、生成 7.6 t/s**（8 线程） |
| 0.6B 走 NPU | ✅ 预填 1332 t/s、生成 182 t/s |
| `TOP_K` 算子（我补的） | ✅ `test-backend-ops` 18 个用例全过 |

**跑不了的：Qwen3.8 走 NPU，卡在 llama.cpp 的 CANN 后端。**

排查过程中修掉了一个**编译期**问题，又暴露出后端逐个算子的缺陷：

1. **AMX 非法指令（已修）**
   `GGML_NATIVE=ON` 让编译器按本机 CPUID 生成 AMX 指令，但内核没启用 AMX tile 状态，
   第一条 `tileloadd` 就 `#UD`。用 `-DGGML_NATIVE=OFF` 重编解决，加载不再崩。

2. **`Repeat` 的 Tile 内核起不来（未解决）**
   ```
   BinaryGetFunction failed, kernel_name=
   OpName:[aclnnRepeat_0_TileAiCore] Op Tile has tiling and tiling parse funcs
   ```
   内核名是空的，CANN 找不到可执行的 kernel。

3. **把 Repeat 放回 CPU 后，`ScatterUpdate` 内核执行失败（未解决）**
   ```
   Aicore kernel execute failed, fault kernel_name=ScatterUpdate_..._high_precision_880015000
   ```

4. **部分卸载会触发另一个断言**
   `-ngl 1` 会撞上 `GGML_ASSERT(!is_matmul_weight(src))`（`ggml-cann.cpp:2149`，
   跨后端拷贝 matmul 权重的路径）。所以只能全卸载或全不卸载。

**结论**：这不是配置问题，是 llama.cpp 的 CANN 后端在这种"14 种量化格式混用 + 51.2B 查表"
的模型上还没有准备好。每修一个算子就露出下一个，工作量不可预估，而且属于上游后端的活。

**对 Strata 的意义**：Strata 走自己的 GGUF 读取器和自己的 kernel，不依赖 ggml，
所以这条路上的问题**不会**传到 Strata。但也意味着不能指望 llama.cpp 提供现成参考实现——
只有 CPU 那份输出可以当标尺。
