# 给 llama.cpp 的 CANN 后端补那五个算子

如果实测发现解码被 CPU 回退拖死，这就是下一步。先把**事实**钉住：缺的是 llama.cpp 的接线，
不是 CANN 的能力。

> **进度**：`TOP_K` 已经补完并在硬件上验证通过（见文末"已验证"一节）。
> 补丁在 `cann/patches/llama-cpp-cann-topk.patch`。

## 结论先说

CANN 9.0.0 里**已经有**这些算子，而且 `libopapi.so` 都导出了：

```
exported: aclnnLightningIndexer
exported: aclnnTopk
exported: aclnnMoeGatingTopK
```

头文件也在（818 个 aclnn 头，命名是下划线风格）：

```
/usr/local/Ascend/cann-9.0.0/include/aclnnop/aclnn_topk.h
/usr/local/Ascend/cann-9.0.0/include/aclnnop/aclnn_lightning_indexer.h
/usr/local/Ascend/cann-9.0.0/include/aclnnop/aclnn_moe_gating_top_k.h
```

**注意**：第一次查的时候我用 `aclnnTopk.h`（驼峰）去 `ls`，全都没找到，差点误判成"CANN 也没有"。
头文件名是 `aclnn_topk.h`，符号名是 `aclnnTopk`，两者风格不一样。

## 签名（从这台机器的头文件里抄的）

```c
aclnnStatus aclnnTopkGetWorkspaceSize(const aclTensor* self, int64_t k, int64_t dim,
                                      bool largest, bool sorted,
                                      aclTensor* valuesOut, aclTensor* indicesOut,
                                      uint64_t* workspaceSize, aclOpExecutor** executor);
```

```c
aclnnStatus aclnnLightningIndexerGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *weights,
    const aclTensor *actualSeqLengthsQueryOptional, const aclTensor *actualSeqLengthsKeyOptional,
    const aclTensor *blockTableOptional, char *layoutQueryOptional, char *layoutKeyOptional,
    int64_t sparseCount, int64_t sparseMode, int64_t preTokens, int64_t nextTokens,
    bool returnValues, const aclTensor *sparseIndicesOut, const aclTensor *sparseValuesOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);
```

## 接线位置（三个地方，每个算子都要改）

| 位置 | 文件 | 行 | 做什么 |
|---|---|---|---|
| 实现 | `ggml/src/ggml-cann/aclnn_ops.cpp` | — | 加 `ggml_cann_<op>()` |
| 声明 | `ggml/src/ggml-cann/aclnn_ops.h` | — | 加原型 |
| 分发 | `ggml/src/ggml-cann/ggml-cann.cpp` | 1780 `ggml_cann_compute_forward` | 加 `case GGML_OP_X:` |
| 放行 | `ggml/src/ggml-cann/ggml-cann.cpp` | 2410 `ggml_backend_cann_supports_op` | 加 `case GGML_OP_X: return true;` |

调用宏（`aclnn_ops.h:922`）自动处理 workspace 和两段式：

```c
GGML_CANN_CALL_ACLNN_OP(ctx, Topk, acl_src.get(), (int64_t) k, (int64_t) 3,
                        true, false, acl_values.get(), acl_indices.get());
```

**维度方向**：`ggml_cann_create_tensor` 会把维度反转成 `[ne3,ne2,ne1,ne0]`，
所以 ggml 的 dim 0 对应 CANN 的**最后一维**（3）。`ggml_cann_cumsum` 里就是这么写的，
`Topk` 的 `dim` 参数同理。

## 先更正一个悲观结论：融合算子会自动降级

`llama-context.cpp:661` 的 `resolve_fused_ops` 会在建图后**检查融合节点落在哪个设备上**：

```cpp
if (device_fused != device_layer) {
    LLAMA_LOG_WARN("layer %d is assigned to device %s but %s is assigned to device %s (usually due to missing support)");
    device_mismatch = true;
}
```

不匹配就把 `cparams.fused_*` 关掉，**下一块图改用未融合的等价写法**。所以：

| 融合算子 | CANN 不支持时 | 结论 |
|---|---|---|
| `DSV4_HC_PRE/POST` | 走未融合路径（matmul + rms_norm + sigmoid + mean） | **全在 NPU 上**，只是多几次读写 |
| `TOP_K_MASK` | 走 `llama-graph.cpp:3466` 的未融合分支（fill + set_rows + add + view） | **全在 NPU 上** |
| `GATED_LINEAR_ATTN` | 支持 | 在 NPU 上 |

我原来写"这五个算子都会回退 CPU"是**错的**。真正会掉到 CPU 的只有两类：

1. **`TOP_K`** —— 图里有这个节点，`supports_op` 返回 false 就整块跑 CPU。**已经补上并验证**。
2. **`LIGHTNING_INDEXER`** —— `build_qsa_sel` 里**没有未融合的替代写法**，这个节点一旦不支持就必然在 CPU 上。

所以剩下的 CPU 开销比我原来估的小得多：每个注意力层只有一个小算子（4 头 × 128 维 × 512 个池化单元），
而不是"每层 HC + QSA + topk 全在 CPU"。

**这件事要实测确认**：跑起来看日志里有没有上面那条 WARN，以及算子被分到哪个设备。
`GGML_SCHED_DEBUG=2` 能打出每个节点的后端。

## 五个算子逐个看

### `GGML_OP_TOP_K` — 值得先做

- 语义：`ggml_top_k(ctx, a, k)` 返回 `[k, ne1, ...]` 的 **i32 索引**，无序。
- 接法：`aclnnTopk(self, k, dim=3, largest=true, sorted=false, valuesOut=scratch, indicesOut=dst)`。
- 难点：`aclnnTopk` 要求同时给 values 和 indices 两个输出；values 用一个临时 buffer 丢掉即可。
  `dst` 是 i32，`indicesOut` 的类型要按 i32 建。

### `GGML_OP_TOP_K_MASK` — 顺着 top_k 做

- 语义：把不在 top-k 里的位置填成 `-inf`（供 softmax 用）。
- 接法：`aclnnTopk` 拿索引，再 `aclnnScatter`/`aclnnInplaceFillScalar` 组掩码。
- 也可以直接用 `aclnn_apply_top_k_top_p.h`（fused），但要确认它的语义和 ggml 的完全一致。

### `GGML_OP_LIGHTNING_INDEXER` — 有现成算子，但语义要对齐

- ggml 的版本返回**分数** `[n_pool, n_tokens]`，之后单独调 `ggml_top_k`。
- aclnn 的版本是 **fused**（`sparseCount` / `sparseMode` / 返回 indices+values），
  一步就把"打分 + 选 top-k"做完。
- 两种走法：
  1. 用 aclnn fused 版本，把 ggml 的 `lightning_indexer` + `top_k` 两节点合并成一个（要改图）。
  2. 只用它的打分能力（`returnValues=true`，`sparseCount` 给全量）拿分数，再走自己的 topk。
- **风险**：`sparseMode` 的语义（causal / 分块）和 QSA 的 pooled-key 掩码不一定一一对应。
  这是五项里唯一需要真读文档、做数值对照的一项。

### `GGML_OP_DSV4_HC_PRE_GATED` / `DSV4_HC_POST` — 建议先不做

- 它们只是**融合**（sigmoid + 逐流平均；broadcast mul + add），不是新算法。
- 拆开的等价路径 llama.cpp 已经会走（第 `PLE_QSA.md` 的 HC 一节），正确性没问题。
- 单层的算子很小（`[10240, 320]` 的乘），CPU 上做得动；真正贵的是**每层两次跨设备搬数据**。
- 所以优先级：**先把 CPU 回退的传输成本测出来**，再决定要不要为它写融合算子。

## 验证方法

1. **单元级**：拷一份小 fixture，同一个输入分别跑 CPU 后端和 CANN 后端，比输出。
   `ggml-backend` 有 `ggml_backend_compare_graph_backend()` 可以直接用。
2. **图级**：`GGML_SCHED_DEBUG=2` 跑一次，数日志里 `CANN0` 与 `CPU` 的算子条数——
   补之前和补之后各跑一遍，就能量化"少搬了多少次"。
3. **端到端**：`llama-bench -m <model> -p 512 -n 128`，对比 `-ngl 99` 前后的 tok/s。

## 顺序建议

```
1. TOP_K          （最简单，QSA 12 层都要，先拿它练手并验证工具链）
2. TOP_K_MASK     （紧挨着，共用 topk 结果）
3. LIGHTNING_INDEXER （要读文档对齐语义，工作量最大）
4. DSV4_HC_*      （等前面测出传输成本再决定）
```


## 已验证：TOP_K 在 NPU 上跑通了

补丁：`cann/patches/llama-cpp-cann-topk.patch`（73 行，改三个文件）。

`test-backend-ops -o TOP_K -b CANN0` 全部通过，18 个用例，是逐元素与 CPU 结果比对的：

```
TOP_K(type=f32,ne=[4,2,1,3],k=4,ties=1):          OK
TOP_K(type=f32,ne=[4096,2,1,1],k=10,ties=0,inf=1): OK
TOP_K(type=f32,ne=[248320,1,1,1],k=40,...):        OK
TOP_K(type=f32,ne=[12,1,2,1],k=1,ties=0):          OK
```

最后两个用例的形状正是 QSA 要用的（`[n_pool, n_tokens]` 里选 k 个）。

### 补的过程踩了两下

1. **`indices expected dtype is DT_INT64 but found DT_INT32`**
   `aclnnTopk` 只吃 int64 索引，而 `ggml_top_k` 输出 int32。要经一个 int64 的临时 buffer，
   再用 `aclnnCast` 降下来。`ggml_cann_argsort` 里有一模一样的做法，照抄即可。

2. **`ggml_cann_create_tensor(void*, aclDataType, size_t, int64_t[4], int64_t[4], int)` 不匹配**
   这个模板要求 `type_size` 和 `nb` 同类型（都是 `size_t`），`ne` 是 `int64_t`。
   我一开始把 `nb` 写成 `int64_t` 就编不过。

### 补完之后剩下的

`TOP_K_MASK`、`LIGHTNING_INDEXER`、`DSV4_HC_*` 还没做。有了 TOP_K 这一步，工具链、
验证方法、调用约定都已经跑通，后面三个是同样的套路。


## `LIGHTNING_INDEXER` 的设计（还没实现）

这是唯一一个**没有未融合替代写法**的算子，所以它一定在 CPU 上。补它的收益不在于算得慢
（算量很小），而在于**消除每 token 每层一次的图切分**：NPU 等 → 拷到 CPU → 算 → 拷回，12 层就是 12 次同步。

### 参考语义（`ggml-cpu/ops.cpp:12450`）

```c
for each token t, stream s:
  for each kv cell ik:
      score = 0
      for h in n_head:
          qk = dot(q[:, h, t, s], k[:, ik, s])     // n_embd=128 维点积
          score += max(qk, 0) * w[h, t, s]         // ReLU 后按头加权
      dst[ik, t, s] = score + mask[ik, t]          // f16 掩码转 f32 再加
```

### 形状（`ggml.c:6538` 的断言）

| 张量 | 形状 | 说明 |
|---|---|---|
| `q` | `[n_embd, n_head, n_tokens, n_stream]` f32 | 128 × 4 × T |
| `k` | `[n_embd, 1, n_kv, n_stream]` f32 | 128 × 1 × n_kv |
| `w` | `[n_head, n_tokens, 1, n_stream]` f32 | 4 × T |
| `mask` | `[n_kv, n_tokens, 1, n_stream']` **f16** | |
| `dst` | `[n_kv, n_tokens, 1, n_stream]` f32 | |

### 用 aclnn 拼（5 步）

n_stream 一般为 1，按它展开：

1. `q` 的布局是 `[t][h][d]`（`nb1` 是头步长、`nb2` 是 token 步长），所以固定 t 的 4 个头是
   **连续**的：`q[:, h, t]` 在偏移 `(t*n_head + h) * n_embd`。
   重排成 `[n_embd, n_head*n_tokens]`，列序是 `t*n_head + h`。
2. `k` 重排成 `[n_embd, n_kv]`。
3. `scores = k^T @ q` → `[n_kv, n_head*n_tokens]`（一次 `aclnnMatmul`）。
4. `scores` 重排成 `[n_kv, n_tokens, n_head]`，`relu`，乘 `w` 重排成 `[1, n_tokens, n_head]`，
   再对最后一维 `sum` → `[n_kv, n_tokens]`（`aclnnRelu` + `aclnnMul` + `aclnnSum`）。
5. 加 `mask`（f16 先 `aclnnCast` 成 f32）→ `dst`（`aclnnCast` + `aclnnAdd`）。

### 查到一个会卡住实现的约束

`aclnnMatmul` **没有转置参数**。而我们需要的乘法是

```
scores[c, j] = sum_d q[d, c] * k[d, j]        // c = t*n_head + h, j = 池化单元
```

也就是 `qᵀ @ k`。`q` 在内存里是 `d` 最快、`h` 次之、`t` 最慢，所以 `q` 天然是 `[d, c]` 的形状，
要当 `[c, d]` 的 A 用就得先转置。

**更正**：我原以为"用 nb 表达转置视图不能指望"。读了 `ggml_cann_mat_mul_fp`（`aclnn_ops.cpp:2227`）
发现**CANN 后端每次矩阵乘都是这么干的**：

```cpp
int64_t transpose_ne[] = { bcast_weight_ne[1], bcast_weight_ne[0], ... };
size_t  transpose_nb[] = { bcast_weight_nb[1], bcast_weight_nb[0], ... };
acl_weight_tensor = ggml_cann_create_tensor(weight, transpose_ne, transpose_nb, n_dims, ACL_FORMAT_ND);
GGML_CANN_CALL_ACLNN_OP(ctx, Matmul, acl_input_tensor.get(), acl_weight_tensor.get(), acl_dst.get(), 1);
```

也就是 `aclnnMatmul` **接受**用交换过的 `ne`/`nb` 表达的转置视图，不需要真的 `aclnnTranspose`。

所以第一版这么写：

1. `q` 直接按 `[n_embd, n_head*T]` 建（`nb0`, `nb1` 用原值，因为 `nb2 == nb1*n_head`）；
2. `k` 按交换 `ne`/`nb` 的转置视图建（`[n_kv, n_embd]`）；
3. `aclnnMm` / `aclnnMatmul` 直接出 `[n_kv, n_head*T]`。

不用拷贝，也就没有额外开销。

### 为什么要小心

- **维度方向**：`ggml_cann_create_tensor` 会把 ggml 维度反转成 `[ne3,ne2,ne1,ne0]`，
  所以 ggml 的 ne0 是 CANN 的最后一维。上面第 1、2 步的"重排"其实是**建 acl tensor 时给对 ne/nb**，
  不是真的拷数据。
- **`n_stream`**：Qwen3.8 单序列时是 1，但 `-np 2` 或多序列并行时不是。第一版可以只支持 1，
  在 `supports_op` 里用 `op->src[0]->ne[3] == 1` 卡住，其余走 CPU。
- **验证**：`test-backend-ops` **没有**这个算子的用例（只有 TOP_K），所以要自己搭对比：
  同一个随机输入分别跑 CPU 和 CANN，比 `dst`。可以照 `test-backend-ops.cpp` 里 `test_top_k`
  的结构加一个 `test_lightning_indexer`，几十行。

### 什么时候做

**先测基线**。如果解码已经两位数 tok/s，这 12 次同步就不是主要矛盾，不做。
如果只有个位数，再按上面的方案做，并用 `test_lightning_indexer` 逐元素验证。
