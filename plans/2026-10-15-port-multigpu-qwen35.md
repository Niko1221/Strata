# 把本地多卡实现移植到上游 qwen35 引擎

## 已核实的事实（2026-10-15）

### 本地（67720bd8）有两套多卡机制

**A. `--expert-cache-device1..3`（RemoteExperts，src/core/remote_experts.cpp）**
- 在 CUDA1/2/3 上各放一个独立专家缓存
- 通过 `ExpertSource::remote[3]` 挂进现有的逐层专家调度
- **形状无关**：`open()` 收 `n_embd`、`n_ff`、`k` 参数，跟着模型几何走
- generate.cpp:2268 的报错文案明确说它"follows this model's own row and routing width"，是给 qwen35moe 留的路
- 上游 origin/main **完整保留**了这套（remote_experts.cpp 存在，generate.cpp 有 41 处 expert_cache_remote 接线）

**B. `--peer-device N`（PeerExperts，src/core/peer_experts.cpp）**
- 自适应第二级专家缓存（按热度换入换出）
- **硬编码 Flash-Next 形状**：`peer_experts.cpp:24-25` 的 `H = 2560`、`CAP = MAXT*10`
- qwen35moe（n_embd 2048）被 generate.cpp:2262 的 `!has_hc()` 守卫明确拒绝

### 上游（origin/main 296b59d2）

- `src/qwen35/q36_engine.cu`：独立引擎，单 GPU，无任何多卡 API
- `--expert-cache-device1..3` 存在但挂在 Flash-Next 的 ExpertSource 上；qwen35 引擎在 `generate.cpp:1875` 之前就 return 了，走不到那
- 上游没有删多卡代码——只是 q36 引擎没接线

### 结论

"移植多卡"不是把文件拷过来（文件两边都有），而是**把 q36 引擎接进 RemoteExperts 机制**。这是 q36 引擎的架构级改动：
- 专家数据目前按 `[gate|up|down] per expert` 连续 blob 布局（q36_engine.cu:975-980）
- RemoteExperts 的 `begin()/finish()` 接口走 `ExpertSource` 的行协议
- 两边要统一成一种专家数据布局，才能让 CUDA1 上的缓存算 q36 的 MoE

## 已核实的关键事实（2026-10-15，主代理亲自读代码确认）

### 本地 a0459b4e 做的正是"多卡同步优化"

作者 biaoge，标题「Pascal sm_60/sm_61 adaptation: dual-GPU expert helper geometry」。

**核心改动：把 `remote_experts` 的 H=2560 硬编码参数化**

| 位置 | 上游 origin/main | 本地 a0459b4e |
|---|---|---|
| `remote_experts.hpp` open() 签名 | `open(..., bool auto_size = false)` | `open(..., bool auto_size = false, int64_t n_embd = 2560, int64_t n_ff = 640, int64_t k = 10)` |
| 新增成员字段 | 无 | `n_embd_`、`n_ff_`、`k_`、`cap_` |
| `remote_expert_opt.cu` reduce kernel | `constexpr H = cpu::H`（L12），`h_mask_ = h_sum_ + MAXT*H`（L61） | 从 `peers_[0].remote->n_embd_ / k_ / cap_` 读（L73-75），加 3 个 static_assert 保证 Flash-Next 数字不变 |
| `generate.cpp` 调用处 | `open(..., auto_size)`（L5749-5751） | `open(..., auto_size, g.n_embd, g.n_ff, g.npert_used)`（L5006-5008） |

**这是上游没有的、用户的自定义优化。移植它不会重复工作。**

### 但它不解决 q36 的多卡问题

- 本地参数化服务于 **Flash-Next 引擎路径**（走 `ExpertSource` 协议）
- 上游 qwen35moe 走**独立引擎** `src/qwen35/q36_engine.cu`，在 `generate.cpp:1875` 就 `return strata::qwen35::run(...)`
- `q36_engine.cu` 全文**没有 ExpertSource / RemoteExperts / remote** 任何引用（已 grep 确认为空）
- q36 的专家是连续 blob 布局（`[gate|up|down]` per expert，`q36_engine.cu:975-980`），不经过 ExpertSource 的行协议

所以要分两步：第一步搬参数化（地基），第二步把 q36 接进 RemoteExperts（真正的目标）。

### 上游的其他状态

- `peer_experts.cpp:24-25` 上游仍硬编码 `H = cpu::H`、`CAP = MAXT * 10`；本地也硬编码（两边一致，peer 路线本地也没参数化）
- 上游 `generate.cpp` 里**没有**本地那个 `!has_hc()` 拒绝 `--peer-device` 的守卫文案（上游删了或没这分支）
- 上游 remote_experts.cpp 比本地多 2 个 commit（`bf4b79ab` Metal 兼容层、`268647d2` prefill 从 helper VRAM 读 blob）

## Workflow 方案设计结论（wf_2ea391874799，2026-10-15）

### 关键推翻：第一步不是第二步的地基

我原先以为「搬参数化 → q36 接 RemoteExperts」是先后关系。实测证明：**两件事必须并排做**。

- 上游 `generate.cpp:1875` 在 Options 构造前就 `return` 到 q36 引擎，走不到 `:5750` 那次 `open()` 调用
- q36 引擎零引用 ExpertSource
- 本地 a0459b4e 的树里**根本没有 `src/qwen35` 目录**（`git ls-tree -r a0459b4e src/qwen35` 返回空）——本地当时走 Flash-Next 老引擎，所以参数化和闸门都是必需的；上游改用独立引擎后，它们成了死代码

**所以第一步只搬机制（7 个文件的签名/尺寸参数化），不搬 generate.cpp 的调用处改动。**

### 「让 qwen35moe 走回 Flash-Next 引擎」不可行

- `family_layout.hpp` 的 `compiled_geometry_mismatch()` 第一条就比 family
- `model_family_runs()` 只认 `Qwen4Exp`
- 上游 `generate.cpp` 里 `has_hc()` 零匹配（本地叫 `has_hc()`，上游这个名字不存在）

### 路线 C（24GB 单卡）不成立

P100-12G 和 P104-8G 都装不下 35B 专家（需要 16.36 GiB）。P104 是 8 GB 卡，连一个专家层都装不下。

### 推荐路线 2A

在 q36 的 MoE 循环里直调 `RemoteExperts::begin/finish`。**两边行协议完全一致**：

- 同一份 `NativeExpertLayout` 的 `[gate|up|down]` 连续 blob
- 同一份 q8_1 激活
- 同一个 `native_expert_grouped`
- 每 entry 一行 `n_embd` 浮点

差别只在 blob 基址怎么来、输出回哪——这正是 `begin/finish` 现成处理的事。不用改 q36 任何 kernel、不用动图捕获、不用做跨卡 event 同步。

### 合并可行性（已实测）

`git merge-tree --write-tree origin/main a0459b4e`：

- 7 个目标文件全部 **Auto-merging**（不是冲突）
- `remote_expert_opt.hpp` 完全干净
- 冲突清单：CMakeLists.txt / gguf_reader.hpp / layout.hpp / prefill/kernels.hpp / setup.py / layer.cpp / mtp.cpp / verify.cpp / pool.cpp / iq_kernels.cu / qsa_decode_attn.cu / prefill/kernels.cu / generate.cpp / sycl prefill

### 步骤与工作量

| 步骤 | 内容 | 可独立验证 |
|---|---|---|
| 1 | 搬参数化到 7 个文件（不含 generate.cpp） | 是（git merge-tree 已证） |
| 2 | **不搬** generate.cpp 的两处改动 | 一条 grep |
| 3 | 补 `begin()` 取每层布局的入口（open() 接 per-layer NativeExpertLayout） | 是 |
| 4 | q36 引擎接助手卡（路线 2A） | 依赖 1、3 |
| 5 | 备选 2D：place_experts 阶段搬权重（仅当 4 太慢） | 依赖 4 实测 |
| 6 | 实机验证 | 需目标机 |

总计约 **55–70 小时**（不含实机验证）。

### 必须知道的三个坑

1. **`begin()` 依赖全局 `expert_layout().fmt[layer]`**，而 q36 从不调 `expert_layout_load`，这张表是空的。不做步骤 3 会拿到 `gu_type=-1`，静默算出垃圾结果——不崩溃，很难靠崩溃发现。
2. **q36 每层的 `n_ff/n_embd/gu_type/d_type` 来自 `load()`**（`q36_engine.cu:976`），不一定全层相同（MTP 块可能不同）。必须逐层取 `L.L`，不能按模型级参数化。
3. **环境**：上游明确 CUDA 13 砍掉 sm_60/sm_70（`CMakeLists.txt:172-183`），必须 CUDA 12.x。本机 v13.2 目录只有 bin/lib，无 nvcc.exe/cmake.exe，本机构建不了。
