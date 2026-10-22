# q36 prefill 用上 helper 显存：方案与否定结论

- 状态：设计定稿，未实施
- 日期：2026-10-20
- 模型：Qwen3.6-35B-A3B-UD-Q4_K_XL
- 机器：192.168.1.98，CUDA0 = P100-12G（主卡），CUDA1/2/3 = P104-8G（helper）

## 0. 先说结论

**问题不是「专家留在内存」，是 prefill 阶段 helper 完全不参与计算。**

用户观察到的「prefill 时显卡功率没用满」是真实的。查证后，最可能的原因是主卡独自承担了 40 层的全部专家计算，三张 P104 在 prefill 期间完全空闲。

**但「把全部专家放进显存」做不到，也不是瓶颈。** 18.75 GiB 专家 + 权重 + KV 超出 36 GB 总显存。真正的可做项是让 prefill 也用上 helper 的算力，或者减少主卡的等待。

## 1. 已核实的事实

### 1.1 prefill 的专家分流（MEASURED）

`src/qwen35/q36_engine.cu:2695-2715`：

| 条件 | 走哪条路 | helper 参与 |
|---|---|---|
| `T <= MAXT`（=8）且有 helper | `moe_helpers` | 是 |
| `T > kGemmT`（=8）且 MMQ | 主卡本地 MMQ int8 | **否，完全闲置** |
| 其余 | 其它路径 | 视情况 |

`kGemmT = 8` 定义在 `q36_engine.cu:2169`，`MAXT = 8` 是 `strata::kernels::cpu::MAXT`。

**`--bench-pp 512` 的 T=512、以及默认 `prompt chunk 2048`，都落第二档。** 所以 prefill 全程 helper 不参与。

启动行原文（MEASURED）：
```
strata-q36: a prompt MMQ chunk keeps a helper-owned layer local; its experts stay idle on the
helper GPU (MMQ int8 vs the helper q8_1 cache; T=512 > MAXT=8)
```

### 1.2 「q8_1 缓存」的准确含义（MEASURED，纠正一处误读）

`src/core/remote_experts.cpp:504`：
```cpp
strata::kernels::quantize_q8_1_rows(xe, n_tok, n_embd_, w.d_q8_, s);
```

**`q8_1` 是激活值（`xe`）的量化格式，不是专家权重的格式。**

所以「helper 存 q8_1 权重、MMQ 要 int8 权重，格式冲突」这个说法**不准确**。真实约束是：
- helper 的 `RemoteExperts` 按 **token 行**（`MAXT=8` 行）为单位工作，激活值量化成 q8_1
- prefill 的 MMQ 按 **大 chunk**（T=512/2048）工作，走 int8 MMQ 内核
- **两者的差异是工作粒度（8 行 vs 512 行）和激活量化路径，不是权重格式**

这条纠正很重要：它意味着「让 helper 也存一份 int8 权重」不是必需的，**真正的障碍是 helper 的执行模型只吃 ≤8 行**。

### 1.3 q36 不使用 Flash-Next 的 Prefill 类（MEASURED）

`grep -c "Prefill" src/qwen35/q36_engine.cu` → **0**

q36 只 include 了两个头文件：
```
src/qwen35/q36_engine.cu:28  #include "strata/prefill/gemm.hpp"
src/qwen35/q36_engine.cu:30  #include "strata/prefill/moe_mmq.hpp"
```

**所以 Flash-Next 的任何 prefill 优化（多级 pipeline、stream_all 阈值、Stager 环形缓冲）对 q36 都不生效。**

### 1.4 Flash-Next 的 prefill 确实从内存流式读权重（MEASURED）

`src/prefill/prefill.cpp:522` `Stager`：
```cpp
int kRing = 16;                                  // STRATA_STAGER_RING 默认 16
std::vector<char> pinned;
std::vector<std::vector<uint8_t>> pageable;      // pin 不够时的 fallback
std::vector<cudaEvent_t> dma_done;
```

`prefill.cpp:2898-2910` 逐层逐专家决定谁上传：
```cpp
for (int64_t l = LB; l < LE; ++l)
    for (int32_t e = 0; e < m.g->n_expert; ++e) {
        if (m.host_res && m.host_res[l * ne + e] >= 0) continue;   // 常驻显存的跳过
        if (m.pp && m.pp->peer && m.pp->peer->has(l, e)) continue; // peer 持有的跳过
        // 其余入队，走 Stager 环形缓冲上传
    }
```

**所以 Flash-Next 的做法是：常驻 + peer 持有的跳过，其余按需从 mmap 的 GGUF / page cache 流式读，16 深环形缓冲 + 多线程预取。**

### 1.5 Flash-Next 的 prefill 也用 peer 卡（MEASURED）

`prefill.cpp:3822-3836`：每层每步**动态**决定哪些 expert 发给 peer，受 `cap_rows` 限制：
```cpp
if (c > 0 && ps_flag[l * ne + e]) { on_peer[e] = 2; rows_peer += c; }
if (rows_peer + c > m.pp->cap_rows) { ++m.pp->over_cap; continue; }  // 超限就退回本地
```

`prefill.cpp:757` 的注释点明了代价：发 expert 的 row 要付 K 倍（top_k=8）链路开销。

### 1.6 q36 的专家放置有降级路径（MEASURED）

`place_experts`（`q36_engine.cu:3928`）不是简单「预先展开 18.75 GiB」，它有多档：
- 全部装得下 → 全 VRAM
- 装不下但能放弃 packed Q8_0 decode 副本 → 放弃它们，全 VRAM（`STRATA_Q36_KEEP_PACKED=1` 可阻止）
- 仍不行 → hybrid：部分 VRAM + 其余 pinned host + CPU pool

当前配置（43/256 每层在 VRAM，18.75 GiB pinned）是第三档。

## 2. 否定结论（写清楚，避免重走）

| 假设 | 判定 | 依据 |
|---|---|---|
| prefill 慢因为层间同步太多 | **否** | q36 prefill 只有 2 个同步点（`q36_engine.cu:3065`、`3069`），比 Flash-Next 少 |
| helper 闲置是 bug | **否** | 既有设计，代码注释明说 MMQ chunk 与 helper 窗口不兼容 |
| 格式冲突导致 helper 不能用 | **表述不准** | 真实约束是 helper 只吃 ≤8 行（`MAXT`），不是权重格式 |
| 把 18.75 GiB 专家全放进显存能解决问题 | **否** | 36 GB 总显存装不下 18.75 GiB 专家 + 权重 + KV |
| 照搬 Flash-Next 的 Prefill 类到 q36 | **不划算** | q36 有自己的 prefill 主循环，替换等于重写引擎 |

## 3. 可行方案（按推荐顺序）

### 方案 A（推荐）：prefill 让 helper 分担，按行分批

**思路**：prefill 的 T=512 不一次性全在主卡算，而是**按 `MAXT=8` 行一批切分**，每批走现有的 `moe_helpers` 路径分给三张 helper。

**为什么可行**：`moe_helpers` 已经能工作（decode 时 100% GPU 命中就是它在跑）。缺的只是「prefill 也调用它」这个分支。

**改动位置**：`q36_engine.cu:2699` 那个 `else if (T > kGemmT && use_mmq)` 分支，在 helper 可用且用户显式开启时，改走分批 `moe_helpers`。

**改动量**：中等。需要处理
1. 分批循环（T=512 → 64 批 × 8 行）
2. 每批的路由结果独立
3. 结果合并回 T=512 的输出
4. opt-in 旗标（如 `STRATA_Q36_PREFILL_HELPERS=1`），默认关闭走原路径

**预期收益**：这个必须实测，不能估。理由是分批后主卡和 helper 都能并行算，但每批的固定开销（同步、上传路由表）会乘以 64 倍。

**可证伪预测**：如果方案成立，pp512 应显著高于 156.9。如果每批固定开销过大，pp512 会**低于** 156.9——这本身就是判定依据。

**风险**：分批可能比单次大 chunk 更慢（更多同步）。这是最可能的结果，要有心理准备。

### 方案 B：降低主卡等待（不改分流逻辑）

**思路**：prefill 期间主卡要等 expert 权重从 pinned host 传到显存。改成 Flash-Next 那样的深环形缓冲预取（`Stager` 思路），把传输藏到计算后面。

**改动位置**：`q36_engine.cu` 的 hybrid 分支里权重上传的地方。

**问题**：q36 当前是「专家在 pinned host，按 batch 需要传」。如果传输已经和计算重叠，这条无收益。**需要先测一下 prefill 期间 PCIe 是否打满**，再决定要不要做。

**前置测量**（不改代码）：跑 prefill 时看 `nvidia-smi dmon` 的 PCIe 收发速率，或 `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current`。

### 方案 C：不预先展开 18.75 GiB pinned（省内存，不提速）

**思路**：仿照 Flash-Next，不预先展开 pinned 副本，改成按需从 mmap 的 GGUF 读。

**收益**：省 18.75 GiB 主机内存（31 GiB 机器上这是实实在在的好处，能避免 OOM 和 SSH 卡死）。
**代价**：可能更慢，因为失去了 pinned 内存的 DMA 优势。

**判定**：这是**资源优化不是速度优化**。如果目标是解决主机内存紧张，做这个；如果目标是 pp tok/s，不要做。

## 4. 推荐执行顺序

1. **先测方案 A 的前置**：不改代码，确认 prefill 期间三张 helper 的 GPU 利用率确实是 0（用 `nvidia-smi dmon` 采样）。这能证实「helper 完全闲置」这个前提。
2. **如果前提成立**，实施方案 A，带 opt-in 旗标。
3. **方案 B 和 C 暂缓**，等 A 有了实测结果再说。

## 5. 验证标准

改完之后，必须同时满足：

| 检查 | 期望 |
|---|---|
| 默认路径（不带新旗标） | `gen: 220` 不变，字节级一致 |
| `--mtp 0/1` parity | 与新旗标无关，保持当前结果 |
| 新旗标开启后 pp512 | 记录实测值，与 156.9 对比 |
| 新旗标开启后 tg128 | **不能低于 47.8**（decode 不能退化） |
| 三 helper auto 配置 | GPU 命中保持 100%，CPU pool 保持 0 条 |

## 6. 本轮没做的事

- 没有实施任何代码改动
- 没有测 prefill 期间 helper 的实际 GPU 利用率（方案 A 的前置条件）
- 没有测 prefill 期间 PCIe 带宽（方案 B 的前置条件）
- mapped MTP 在这台机器上恒为 INERT（MTP 块 hybrid），本次不涉及

## 附：mapped MTP 的当前状态（与本方案无关，一并记录）

`STRATA_Q36_MTP_MAPPED=1` 在本机**永远不生效**，stderr 打印：
```
STRATA_Q36_MTP_MAPPED=1: INERT - the MTP block's layer is hybrid
```

根因：MTP 块只有 43/256 slot 在 VRAM → `lhy(40)` 恒真 → `mapped_chain_live()`（`q36_engine.cu:1428-1430`）恒 false。helper 按设计只托管 40 个主层（注释 "41 != 40"）。

要让它 live，需要让 helper 也托管 MTP 块并放宽 guard。属独立任务。

## 2026-10-20 后轮：「搬 Flash-Next prefill 跨卡」已实施并远端实测（STRATA_Q36_PREFILL_PEERS）

> 状态更新：本节把 §3「方案 A」从「设计定稿，未实施」改为**已实施、远端实测**。
> 机器 192.168.1.98（P100 主 + 3×P104，12 对 GPU 全无 P2P），模型 Qwen3.6-35B-A3B-UD-Q4_K_XL。
> 远端二进制 `489b635a`（`build/` 与 `bin/` md5 一致，5 源码文件两端 md5 一致，引擎启动确认
> `libstrata_kq.so` 已从 `bin/` 加载）；数字全部出自远端 `/root/zz9q36/` 日志，
> 可 `grep '###' /root/zz9q36/all*.log` 逐条复核。

### 结论（先给）

1. **默认路径 parity 保住**：`gen: 220` 门过（`--tokens 32 --n 1`，gate.out + 2 次复跑）；off 时
   stderr 有 **0** 行 PREFILL_PEERS，字节级不变要求满足。
2. **本机无 P2P（12 对 GPU 全 0），新旗标开启时干净降级，性能 = off**（pp512 157.0-157.4、
   tg128 47.7-48.2，全在 off 自身范围内，属噪声）。**`PeerPrefill36::prepare/launch/wait` 的算术
   正确性在本环境完全没有被覆盖**——只验证了编译、链接、以及「不满足前置条件时干净降级」；
   真实跨卡收益要在有 P2P 的机器上测。
3. **唯一真实收益在 host-staging 路线**（`STRATA_Q36_PREFILL_PEERS_HOST=1`，纯 prompt）：
   pp512 **158.5 → 229.9（+45%）**。但带 decode 即崩（窗口 ≥256 `illegal memory access`，
   源码注释记 160+tg8 亦崩），唯一干净的小窗配置 decode 低于基线——**HOST 路线保持 opt-in，
   不进默认路径**（当前 `=1` 语义正确）。

### 从 origin/main 搬了什么（对应位置）

搬的是**调用形态**，不是类定义副本：`PeerPrefill` 是 `prefill.cpp` 的私有 struct
（`prefill.hpp` 里 `grep -c PeerPrefill` = 0，MEASURED），只服务 `Prefill` 类、
`set_peer()` 是唯一入口。q36 不搬类，按同一份行切分契约新写 executor（recon 的路线 A）。

| 搬来的做法 | origin/main 出处 | q36 落点 |
|---|---|---|
| 按 routed entry（行，不是专家）切分 + 事件回流 | `prefill.cpp:695-816`（`PeerPrefill`：三条流、`ev_in/ev_done`、`compact`+`G` 分组、双缓冲 `Dm_b[2]`）与 `:3819-3838`（`rows_local/rows_peer/cap_rows/over_cap`，判据 = peer cache 的 `has()`） | 新 `src/qwen35/q36_peer_prefill.cu/.cuh` 的 `PeerPrefill36`（748+112 行） |
| 「主卡条目在前、helper 依次在后」的行契约 | `prefill.cpp:3819-3838` | `q36_engine.cu` `peer_plan()`：主表 `peer_owner` + `base[3]` + 重复行检查 + `rows()` 第二见证 |
| `device()` 访问器（调用方问 tier，不从指针反推） | `peer_experts.hpp:49` | `remote_experts.hpp` 新增 `int device() const`（逐字对应） |
| P2P「链路是否真带数据」探测 | `peer_experts.cpp`（#2666 同一探测与同一注释） | `q36_engine.cu` `peer_copies_data()` |
| executor 上下文绑到 helper 设备 | `prefill.cpp:2119-2121`（`cudaSetDevice` 后建 `mmq::Context`） | `PeerPrefill36::open()` |

**明确没跟的**：`prefill.cpp:130` 的文件级常量 `N=2560, K=10, NE=512`——q36 几何是
2048/512/top-8，`PeerPrefill36::open` 显式接收 `n_embd/n_ff/k/cap_entries/layouts`，行宽全运行时。

### 跳过了什么（及原因）

| 跳过项 | 原因 |
|---|---|
| cap_rows 回退（超限 over_cap 退回本地） | `cap_entries = tmax×top_k` 且 chunk 全部 ≤tmax，该分支今日不可达（`q36_engine.cu` 注释已记），第二期（B4）再搬 |
| 组级事件重叠（B5） | 第二期 |
| `on_peer` 三态 / streamed 模式 | q36 不用 streamed |
| 复用 `PeerExperts::launch/finish`（路线 B） | 其 `H=2560` 硬编码对 q36 几何是静默 host 堆越界写（`memcpy` :284 每 token 多读 512 float、`finish` :330 越界写）；上游没有 q36 参数化先例（`eaf852fa` 不在 origin/main，`kMaxCap=MAXT*10` 仍硬编码、`Meta` 定长数组）——全要新写。**且 `PeerPrefill` 本身从不调 `launch/finish`**（只用 `valid/p2p/device/has/slot_ptr`），路线 A 彻底绕开 H/CAP 问题 |
| 改 `expert.hpp` 的 `H` static_assert | 绑 `kCompiledGeometry.n_embd`，12+ 个 Flash-Next kernel 各自 `static_assert`，改 = Flash-Next 全编译失败。两种路线都不碰它 |
| HOST 路线进默认 | 有崩，见下「实测数字」 |

### 本地改动（均未提交）

| 文件 | 改动 |
|---|---|
| `CMakeLists.txt:733-740` | `strata_qwen35` 注册 `q36_peer_prefill.cu`。**这份 748 行 executor 此前从未注册、从未参与编译**；不注册则 15 处 `PeerPrefill36` undefined reference，链接必失败 |
| `include/strata/core/remote_experts.hpp:68-72` | 新增 `int device() const { return device_; }`，逐字对应 origin/main 的 PeerExperts 访问器；纯 inline 追加，Flash-Next 路径语句级不变 |
| `src/qwen35/q36_peer_prefill.cu`（新） | `PeerPrefill36` executor：`open()` 设备来源从「扫 slot_ptr 反推」改为 `helper.device()`，保留一次 `cudaPointerGetAttributes` 一致性校验（不一致 = 答错而非报错，值得启动期拦一次）与空缓存拒绝（错误文案按成因分开，行为集合不变） |
| `src/qwen35/q36_peer_prefill.cuh`（新） | executor 的冻结契约 |
| `src/qwen35/q36_engine.cu` | 引擎侧行计划：`peer_open/peer_plan/peer_copies_data/peer_host_route`、主表 `peer_owner`、两个旗标 `STRATA_Q36_PREFILL_PEERS` / `STRATA_Q36_PREFILL_PEERS_HOST` |

注：`q36_peer_prefill.cu/.cuh` 仍是未跟踪文件，本次只是把它们接进构建；`git add` 与否留给用户。

### 实测数字（远端 `489b635a`，3-helper auto，bench 2 交错 off×3 / on×3，`/root/zz9q36/`）

| 配置 | pp512 | tg128 |
|---|---|---|
| off ×3 | 157.4 / 157.3 / 157.3 | 47.8 / 47.8 / 47.5 |
| on ×3（P2P 降级） | 157.3 / 157.3 / 157.0 | 48.2 / 47.7 / 47.7 |
| off，纯 prompt（tg0） | 158.5 | — |
| on+HOST，纯 prompt（tg0） | **229.9** | — |
| on+HOST，pp8+tg128 | 113.8 | 48.8 |
| on+HOST，pp128+tg128 | 94.9 | 38.8 |
| on+HOST，pp256+tg128 | **illegal memory access** | — |

- 极差 ≤0.4（pp）/ 0.7（tg），<2% 属噪声；on 的 tg128（47.7-48.2）落在 off 自身范围（47.5-47.8）内，无回归。
- P2P 探测 MEASURED：`nvidia-smi topo -m` 全 `PHB`，`cudaDeviceCanAccessPeer` 12 对 GPU 全 0 →
  on 时 4 行「out of the prompt path」降级日志，helper 全走本地回退；on=off 是 DERIVED（本地回退 ≡ off）。
- `PREFILL_PEERS_HOST=1` 启动行：3 executor 打开（10240 experts, 40 layers）；pp8+tg128 与
  pp32/160/512（tg0）全干净，`gen` 逐字节相同。
- 崩因（DERIVED）：host-staged buffer 跨 prompt→decode 边界未等待，崩在 prompt 结束后首次 stream sync。
- Flash-Next 影响 MEASURED：`strata` 目标构建通过、`--help` 正常、`file_expert_source_test` PASS；
  本地 `pytest tools/test_*.py` 989 过 / 19 挂，同 19 个在 HEAD 基线同样挂（GBK 解码、缺文件），与本改动无关。
  远端无 Flash-Next 模型（仅 Qwen3.6 gguf），运行时验不了。

### Parity（明确回答：保住）

- 默认路径 `--tokens 32 --n 1` → `gen: 220`（gate.out + 2 次复跑），MEASURED。
- 旗标 off/on（`--tokens 1 --n 8`，3 helper）`gen` 逐字节相同：
  `5328 1301 279 2450 2144 321 4087 264`；off 时 stderr 0 行 PREFILL_PEERS。

### 已知风险与下一步

1. **R1 静默错行（未修）**：`peer_owner` 在 `peer_open` 时建一次，而 `PeerPrefill36::prepare`
   每层重读 `slot_ptr`；若 decode begin/adapt 期间 RemoteExperts 换 slot，k_peer_rank 的 rank 与
   host 的 `base_j` 不一致 → 静默错行。`peer_plan` 现已有两道响亮防线（逐层表 vs 活缓存对比、
   `rows()` 第二见证，不一致即 `exit(1)`），结构性修法（plan 时用 `owns()` 现算 rows，或禁用旗标
   期间的 rebalance）未实施。
2. HOST 路线修复（跨边界的等待）之前，不能进默认路径。
3. 没测的：P2P 机器上的真实跨卡收益；本机 executor 的算术正确性（全降级，未覆盖）。
4. B4（cap_rows 回退）/ B5（组级事件重叠）第二期。

## 2026-10-20 第二轮：移植 Flash-Next prefill 跨卡——origin/main 逐行规格 + B4/B6 落地

> 状态：五条并行逐行阅读线（PeerPrefill struct / set_peer 初始化 / 3810-3870 分流 / 4040-4330 发射回流 / PeerExperts 真相）全部 ok:true；
Synthesize 产出移植规格——**4/5 核心做法已在 `q36_peer_prefill.cu` 落地，缺 cap_rows 回退（B4）与组级事件重叠（B5，第二期）；`on_peer` 三态不搬（q36 不用 streamed 模式）**。
本轮派 6 块实施（B2 / B4 / B6a / B6b / D，全部构建通过并经远端测量）。本节取代「后轮」节末尾「B4 / B5 第二期」的判定：
**B4 本轮已做，B5 仍是第二期**；R1 静默错行本轮只加了两道响亮防线，结构性修法未做。
机器 192.168.1.98（P100 主 + 3×P104，`cudaDeviceCanAccessPeer` 12 个有序对全 0，无 P2P），
模型 Qwen3.6-35B-A3B-UD-Q4_K_XL；远端二进制 `489b635adefaee360dc7fdc02b996d81`（`build/` 与 `bin/` md5 一致）。

### 结论（先给）

1. **搬的是调用形态，不是类副本**：`PeerPrefill` 是 `prefill.cpp` 的私有 struct，q36 按同一份行切分契约自建 executor `PeerPrefill36`（路线 A，见「后轮」节）。
2. **规格逐行核对后：4/5 核心做法已落地**（按 routed entry 行切分、has() 判据、P2P 回流、主卡等 peer 再 combine）；缺的两项——B4 cap_rows 回退本轮已实现，B5 组级事件重叠第二期。
3. **本机根本没有 P2P**（实测 `MemcpyPeer` 报成功却不搬数据，驱动偷偷走主机——假阳性）→ 追加 B6a/B6b 主机暂存路由；因实测跨 prompt→decode 边界崩，主机路由改为 `STRATA_Q36_PREFILL_PEERS_HOST=1` 显式开启，不进默认路径。
4. **默认路径 parity 保住**：`gen: 220`，off / P2P 回退 / 主机路由三态生成逐位相同（数字见下「实测数字」，只记录不判收益）。

### 做法清单（origin/main 逐行读出，每条带 path:line）

#### A. PeerPrefill 结构与几何常量
- 三条流：peer 计算 `s`（primary 侧 `cudaStreamWaitEvent(ps, ev_in)` 后使用）/ 回流 `s_out`（组回传 ∥ 下一组计算）/ primary 辅助 `x_stream`（host D2H 与 primary 计算并行）——`origin/main:src/prefill/prefill.cpp:699, 703, 760-761`
- 两个跨设备事件：`ev_in` 记在 primary（「激活 + 行表就绪」，`origin/main:src/prefill/prefill.cpp:700`，使用点 `:4063, 4067`）；`ev_done` 记在 peer（「行已落进 primary 的 Dm」，`:701`，下一层入口 `:4068` 消费）
- 组级事件环 `ev_grp[kGrpEv]`，`kGrpEv = NE/16 + 2 = 34`（`NE=512` 在 `:130`，`MMQ_GROUP=16` 在 `:1115`；一层 ≤32 组，+2 为防越界余量，compact 路取模）——`origin/main:src/prefill/prefill.cpp:704-705`
- compact 逐组缓冲（`G=T`，8192-token chunk 下 ~0.9 GB → ~0.26 GB）+ 双缓冲 `Dm_b[2]`/`ev_dm[2]`/`dm_live[2]`（奇偶组 `b = g2&1` 交替，写前 `cudaStreamWaitEvent(ps, ev_dm[b])` 门控）——`origin/main:src/prefill/prefill.cpp:708-717`
- `out_pipe`（开关：组间回传/计算重叠；默认开，`STRATA_PF_PEER_PIPE=0` 关但 compact 恒开）/ `out_pending`（跨层防竞态：下一层入口 `:4068` 一次性消费）——`origin/main:src/prefill/prefill.cpp:706-707, 2034`
- peer-streamed 专家环：`RP` 槽（`min(helper.ring, kSplitHelpRing)`，默认 48，`STRATA_PF_PEER_RING`）、`pstage/pcopied/pused` 事件、`plive` 占用位；`ps_frac` 默认 0.35（`STRATA_PF_PEER_STREAM`，0=关）——`origin/main:src/prefill/prefill.cpp:719-727, 2068-2079, 2162`
- 无 P2P 用 mapped pinned host 四件套 `host_x/host_rows/host_src/host_bounds` + `kHostBounds` 容量判定（超限走 `up_ring` pinned 上传，#1996）——`origin/main:src/prefill/prefill.cpp:745, 748-753, 776`
- `back_at/back_rows` 标回传行区间——`origin/main:src/prefill/prefill.cpp:754, 4069-4070, 4709`
- SUMS 模式：peer 端把行聚成每 token 一个 T×N 加权和再回传（代替逐行 rows_peer×N，最坏省 K 倍字节）；`f16` 传输、`gather` 两种每组合并方式；无 P2P compact 时默认开——`origin/main:src/prefill/prefill.cpp:755-774, 2091-2095`
- peer 侧自建 `mmq::Context`（`ctx`），layer-split helper 则借用下一 stage 的 ctx（`run_ctx`）——`origin/main:src/prefill/prefill.cpp:739-740, 1957`
- 析构 `cudaSetDevice(dev)` 统一清理 `owned`——`origin/main:src/prefill/prefill.cpp:780-796`
- 几何常量是文件级 `N=2560, K=10, NE=512`（非运行时）——`origin/main:src/prefill/prefill.cpp:130`
- 资源 flag：所有 stream `cudaStreamNonBlocking`、所有 event `cudaEventDisableTiming`、host 缓冲 `cudaHostAllocPortable|cudaHostAllocMapped`——`origin/main:src/prefill/prefill.cpp:2015-2152`

#### B. `set_peer` 初始化序列（`origin/main:src/prefill/prefill.cpp:1991-2170`）
- 早期 guard：peer 无效 = 不启用（不分配任何资源）；无 P2P 需 `STRATA_PF_PEER_HOST=1` 否则报错（`:1994`）；仅限 MMQ 路径（`:2004`）；NVFP4 专家拒绝；APR 折叠包不参与（`:2008`）
- `cap_rows = max(1, min(cap_rows, T_max*K))`（`:2013`）；调用方默认 `prefill_chunk*K/2`、CLI `--peer-prefill-rows`——`origin/main:src/program/generate.cpp:8566, 2621`
- **device 切换时序是核心**：先 primary 上建 `ev_in`（`:2015-2016`）→ `cudaSetDevice(peer)`（`:2027`）→ 建 `s/ev_done`（`:2028-2029`）、`s_out/ev_grp`（`:2030-2037`）、peer 设备缓冲（`:2039-2044`）、compact 分配（`:2048-2079`，其中 `cap_rows` 被覆盖为 `T_max*K` `:2053`、`G=T_max` `:2054`）→ peer 设备建 `mmq::Context`（`:2117-2121`）→ host mapped 缓冲（`:2123-2152`）→ `cudaMemGetInfo` + `cudaSetDevice(prev)` 还原（`:2154-2155`）→ `m.pp = move(pp)`（`:2170`）
- 无 P2P 时恒 compact（`:2050`）；sums/f16/gather 仅在无 P2P compact 默认开（`:2091-2095`）
- 中途失败全部 `return false`，由 `unique_ptr` 析构按 `dev` 统一清 peer 侧资源（`:2156`，析构 `:760-796`）；`ev_in` 在 primary 上建、析构按 `dev` 清——上游自身潜在泄漏点（DERIVED）

#### C. 逐层专家分流（`origin/main:src/prefill/prefill.cpp:3810-3870`）
- 起算 `rows_local = T*K`（`:3819`，先假设全本地）；peer 走走的行逐专家累加 `rows_peer`，收尾 `rows_local = T*K - rows_peer`（`:3836`），再减 CPU 接管的 `rows_cpu`（`:3852`）；peer 行块落在尾部连续区间 `[rows_local, T*K)`
- `on_peer` 三态：初值 0=本地（`:3821`）；1=peer-held（`has()` 判据，`:3833`）；2=peer-streamed（`ps_flag` 判据，`:3824-3826`）
- `host_res[layer*NE+e] >= 0` = 专家已常驻 primary 显存，短路不搬（`:3829`；同一 residency 语义贯穿 `:3787, 2899-2900`）
- **cap_rows 超限：`++m.pp->over_cap; continue` → 该专家落回本地（不丢行），`over_cap` 只是统计计数（`:3832`，计数打印在 `:4992`）**
- `order_peer` 两遍分配：held（1）在前、streamed（2）追加在尾；原因是组需要行连续、streamed 副本排最后准备时间最长（`:3845-3848, 3863-3868`）

#### D. 逐层发射与回流（`origin/main:src/prefill/prefill.cpp:4040-4330`，同一 host 线程内联 enqueue，无独立线程、层内无 `cudaStreamSynchronize`）
- 激活三条路：P2P → `cudaMemcpyPeerAsync`（`:4070`，不经 host）；无 P2P → D2H 到 mapped（`:4055-4056`）+ peer 侧 kernel 读映射内存（`:4071-4072`）
- 路由表 `src` 恒经 host：P2P 时 mapped H2D（`:4076`）或 `up_ring` pinned（`:4079`）；无 P2P 时 `std::memcpy` + kernel 读（`:4086-4087`）
- `ev_in` 记在 `xs`（默认 `m.cs`；`x_stream` 时先 `ev_x` 追平，`:4048-4053`），记录点 `:4062`；peer 流等它（`:4067`）；`out_pending` 下一层入口一次性消费（`:4068`）
- peer 专家权重取 `P.peer->slot_ptr(l,e)`（held，`:4185, 4196`）或 streamed 环 `P.pstage[...]`（`:4182-4193`）
- compact 组循环：`quantize`（`:4207`）→ `run_ctx->run(gu)`（`:4209`）→ swiglu → `run_ctx->run(dn)` 写 `Dm_b[b]`（`:4215-4221`）
- **out_pipe 组回传 4 步：`record(ev_grp[g2%kGrpEv]) → wait(s_out) → cudaMemcpyPeerAsync → record(ev_dm[b])`，本组回传 ∥ 下一组计算（`:4234-4243`；非 compact 同形 `:4300-4304`；无 P2P 改 D2H `host_rows` `:4240-4241`）**
- SUMS 收尾：`sums_to_f16` + D2H `host_sum16`（`:4309-4317`），置 `ev_done` + `out_pending`（`:4316-4317`）；无 pipe 兜底走一次整块 `cudaMemcpyPeerAsync`（`:4318-4323`）
- 发完 `cudaSetDevice(prev)` 回主卡（`:4326`）

#### E. 主卡等待与 combine（`origin/main:src/prefill/prefill.cpp:4706-4717`）
- combine 前 `cs` 等 `ev_done`（`:4707`）；无 P2P 非 SUMS 补一次 `copy_f32_wide` 把 `host_rows` 读进 `m.Dm`（`:4709`）；SUMS 的 combine kernel 直接读映射的 `host_sum(16)`（`:4711-4714`）

#### F. PeerExperts = 注册表（prefill 路的真相）
- prefill 路只用 `valid/p2p/has/slot_ptr`；**`peer->launch/finish` 在 `prefill.cpp` 全文零匹配**（使用点 `origin/main:src/prefill/prefill.cpp:1891, 2011, 4185, 4269, 4669`）
- `has`/`slot_ptr` 是纯索引查表（`res_` 表 + `ExpertCache::device_slot`），几何无关——`origin/main:include/strata/core/peer_experts.hpp:45-51`
- `H=2560`/`CAP=80` 是编译期常量，只用于 `launch/finish` 的 staging 缓冲与 memcpy 宽度（`origin/main:src/core/peer_experts.cpp:28-29`，使用点 `:171-330`）；`launch/finish` 属 verify-window/decode 路（`origin/main:src/core/expert_source.cpp:4541, 4759`）
- 完整公开 API（`open/close/launch/finish/adapt/apply_pending/resident/gib/...`）——`origin/main:include/strata/core/peer_experts.hpp:27-75`
- `set_stage_helper` 与 `set_peer` 互斥：`bind_stage_helper` 见 `m.pp` 存在直接 `return false`（`origin/main:src/prefill/prefill.cpp:1930`）；两独立成员 `m.pp`/`m.help_pp`，层切分场景 `:2601` 交换

#### G. env 开关组（q36 对应命名参照）
`STRATA_PF_PEER_HOST`（`origin/main:src/prefill/prefill.cpp:1994`）/ `_XSTREAM`（`:2018-2024`）/ `_OUT_PIPE`（`:2033-2034`）/ `_COMPACT`（`:2049-2050`）/ `_RING`（`:2069-2079`）/ `_STREAM`（`:2068`）/ `_SUMS`（`:2091`）/ `_F16`（`:2093`）/ `_GATHER`（`:2095`）

### 移植落点（origin/main 做法 → q36 侧）

| origin/main 做法 | q36 落点 | 状态 |
|---|---|---|
| 按 routed entry 行切分 + peer 行尾部连续块（`prefill.cpp:3819-3836`） | `k_route_count/k_route_scatter` 的 `owned` 跳过（`q36_engine.cu:823, 842`）+ `Engine::peer_plan`（`:3487`）；行契约 `base_j = n_local + Σrows_i` + entry 升序（`q36_peer_prefill.cuh:52-62`） | 已落地 |
| `has()` 判据（`prefill.cpp:3829`） | `PeerPrefill36::owns`（`q36_peer_prefill.cu:352`）/ `RemoteExperts::holds`（`remote_experts.hpp:59`）→ `peer_owner` 表（`q36_engine.cu:3622-3634`） | 已落地 |
| P2P 回流不过 host（`prefill.cpp:4070, 4237`） | `cudaMemcpyPeerAsync`（`q36_peer_prefill.cu:413/415/530`） | 已落地 |
| 主卡等 peer 再 combine（`prefill.cpp:4706-4709`） | `peer_prefill_wait`（`q36_engine.cu:3660`）→ `moe_end`（`:2974`） | 已落地 |
| executor 上下文建在 helper 设备（`prefill.cpp:2117`） | `PeerPrefill36::open()` | 已落地，**本轮 B2 强化**：设备来源从「扫 slot_ptr 反推」改为 `helper.device()`，`Impl::primary` 绑真实 primary 序号，修掉三处硬编码 `0`；拒绝 helper==primary / primary<0 / blob 实际所在设备与 tier 不符 |
| cap_rows 超限回退本卡（`prefill.cpp:3832`） | `peer_cap_rows`/`peer_over_cap` + 超限**整层**回退本地，判定放在共用谓词 `peer_layer()`（`q36_engine.cu`，`peer_plan` `:3556` 一带） | **本轮 B4 实现** |
| R1 归属对账（上游无此风险：`prefill.cpp` 每层运行时重读 `has/slot_ptr`、`cap_rows` host 侧同源，q36 的 `peer_owner` 是启动时建一次而 `prepare` 每层重读 `slot_ptr`） | `peer_plan` 双重归属对账：逐层表 vs 活缓存对比 + `rows()` 第二见证，不一致 `exit(1)` | 本轮加响亮防线（结构性修法未做，见下） |
| 无 P2P host 路（`prefill.cpp:4055-4056, 4240-4241, 4709` 的 mapped 四件套） | `STRATA_Q36_PREFILL_PEERS_HOST=1` 主机暂存路由（pinned staging + 三段拷贝：`prepare` D2H→H2D、`launch` D2H、`wait` H2D；`Impl::h_x/h_ids/h_out`、`pending_*`）+ 无 P2P helper 启动门禁 `peer_host_route()`（`peer_copies_data()` 探针一字未动） | **本轮 B6a/B6b 实现**；偏离：上游用 mapped pinned（GPU 直读），q36 用普通 pinned + 显式三段拷贝 |
| `device()` 访问器（`peer_experts.hpp:49`）/ P2P 真搬数据探测（`peer_experts.cpp` #2666 同探测） | `remote_experts.hpp:68-72` `int device() const`；`q36_engine.cu` `peer_copies_data()` | 已落地（「后轮」） |

**几何适配（q36 侧全部取运行时值，不跟上游常量）**：
- `N 2560→2048`、`K 10→8`、`NE 512→256`、`gu w_rows 1280→2*n_ff=1024`、`down w_cols 640→512`：全走 `g.n_embd/g.n_ff/m.top_k/g.n_expert`（MEASURED：`q36_peer_prefill.cu:199-233` executor 已参数化）
- 上游 `MMQ_GROUP=16`/`kGrpEv=34`（`prefill.cpp:1115, 704-705`，16 专家一组的 MMQ 布局）**不适用**：q36 按 blob 地址连续性切 `Impl::Group`（`q36_peer_prefill.cu:167-188`）
- `cap_entries = tmax*top_k`（`q36_engine.cu:3601`，与 `prefill.cpp:2142` 同语义）；每层 `gu_type/d_type/up_off/down_off/bytes` 必须走运行时 `layouts[]`
- `PeerExperts` 的 `H=2560/CAP=80` 硬编码（`peer_experts.cpp:28-29`）与 q36 无关：q36 用 `RemoteExperts`，不走 `launch/finish`（MEASURED：`prefill.cpp` 全文无 `peer->launch`）

### 实测数字（只记录，不判收益；判读见「后轮」节）

机器 192.168.1.98（P100 主 + 3×P104，无 P2P），模型 Qwen3.6-35B-A3B-UD-Q4_K_XL，二进制 `489b635a`，3-helper auto 行切分，取 `bench 2`，off/on 交错各 3 次。测量在独占窗口内进行（先停掉同机另一会话的 sweep——它 `pkill` 引擎并占 VRAM，前几次测量被污染，RC=137/OOM 均属竞争，独占后复现干净）。

**Parity 与三态生成**（`--model $M --tokens 32 --n 1`，`/tmp/sw/V2_parity_default.out`）：
- off → `gen: 220`（RC=0）；完整序列（前 8 token）`220 16 15 15 15 20332 1740 369`
- `PREFILL_PEERS=1`（P2P 回退）/ `+_HOST=1` → 与 off **逐位相同**，rc 全 0
- off 时 stderr：0 行 PREFILL_PEERS

**bench（`--bench-pp 512 --bench-tg 128`，交错 ×3）**：

| 配置 | pp512 | tg128 | rc |
|---|---|---|---|
| off ×3 | 157.1 / 157.3 / 157.4 | 47.8 / 47.9 / 47.8 | 0 |
| on（`PREFILL_PEERS=1`，P2P 回退）×3 | 157.3 / 156.6 / 156.6 | 47.7 / 47.7 / 47.6 | 0 |
| on + `_HOST=1`，纯 prompt（tg0） | 227.9 / 227.9 / 226.4 | tg128 崩 IMA | IMA |

**过程实测（实施 agent，`--bench-pp 160 --bench-tg 0`）**：主机路由干净 135.5 / 144.1 / 141.6 tok/s（rc=0）；同配置关掉主机路由 49.3 tok/s。`--bench-tg 2`：prompt 后第一次同步炸 `illegal memory access`（与 helper 数、层切分/行切分、launch-blocking 均无关；同配置关该路由则干净）。

**P2P 实测（本机 12 个有序对）**：`nvidia-smi topo -m` 全 `PHB`；`cudaDeviceCanAccessPeer` 全 0；`cudaDeviceEnablePeerAccess` 报 not supported；**`cudaMemcpyPeer` 报成功却什么都没搬**（驱动偷偷走主机）——探针（`peer_copies_data()`）防的正是这个假阳性。

**揭穿的坑**：`hybrid_t=128` 使 T≤128 根本不进 MMQ/peer 分支——早先所有「通过」的验证（T≤128 配置）**无效**，未覆盖 peer 算术；真实 peer 路径只在 T>128 走到，且最初每次都崩（B2 修复后才干净）。

**构建**：远端 `make strata-q36` RC=0；`cp -f build/strata-q36 bin/` 后 `build/` 与 `bin/` md5 均 `489b635adefaee360dc7fdc02b996d81`；两个 TU 单文件类型检查均 RC=0。

### 默认路径 parity：保住

- `--tokens 32 --n 1` → `gen: 220`（RC=0），off / P2P 回退 / 主机路由三态生成序列逐位相同，off 时 stderr 零新增行——「字节级不变」满足。
- Flash-Next 侧未触碰（本轮改动文件不含 Flash-Next 共享文件；`remote_experts.hpp` 的 `device()` 追加在上一轮已验证语句级不变）。

### 没做的部分（及原因）

| 项 | 原因 |
|---|---|
| B5 组级事件重叠（`prefill.cpp:4234-4243` 的 `ev_grp`/`s_out`/`Dm_b[2]` 4 步） | 第二期：q36 现每 helper 一次整 run 回传（`q36_peer_prefill.cu:530`）+ 单事件 `wait()`（`:536`），收益小；先在 P2P 机器实测真实收益再定 |
| `on_peer` 三态 / streamed 模式（`prefill.cpp:3824-3826`，`ps_frac` `:2068`） | q36 不用 streamed 模式（无 peer 侧专家环） |
| SUMS 省 K 倍回传（`prefill.cpp:755-759, 4309-4317`） | 最坏回传 `rows_peer×2048×4` ≈ 32 MB/层，40 层 ≈ 40 ms/chunk，相对 3.26 s/chunk 可接受（DERIVED），不值得引入 |
| `PeerExperts::launch/finish` 路线（路线 B） | `H=2560/CAP=80` 硬编码（`peer_experts.cpp:28-29`）对 q36 几何 2048 是静默 host 堆越界（launch `:284` 每 token 多读 512 float、finish `:330` 越界写）；且 `PeerPrefill` 本身从不调 `launch/finish`（全文零匹配），路线 A 彻底绕开 |
| R1 结构性修法（plan 时用 `owns()` 现算 rows / 禁用旗标期间的 rebalance） | 本轮只加了两道响亮防线（表 vs 活缓存、`rows()` 第二见证，不一致 `exit(1)`）；结构性修法留下一轮——R1 仍是唯一会静默污染 logits 的真实风险 |
| 主机路由跨 prompt→decode 边界 IMA 根因 | 怀疑在 `wait()` 发出的 H2D 与下一层 `launch()` 复用 pinned `h_out` 的次序（`pending_n` 只清了计数、无真正流序保证）；`compute-sanitizer` 在该机不可用，需另找定位手段；修复前主机路由保持 opt-in（`=1`），不进默认路径 |
| P2P 机器的真实跨卡收益 + executor 算术覆盖 | 本机无 P2P（12 对全 0），on 时全本地回退，`PeerPrefill36` 的算术正确性在本环境完全没有被覆盖，要在有 P2P 的机器上测 |

### 本轮改动文件（未提交）

| 文件 | 改动 |
|---|---|
| `src/qwen35/q36_peer_prefill.cu` | primary 序号绑定（`Impl::primary` 取 `primary_device()`，修三处硬编码 `0`，含 `cudaDeviceEnablePeerAccess(0,0)`）；无 P2P 主机暂存路由（pinned staging + 三段拷贝，`Impl::h_x/h_ids/h_out`、`pending_*`）；`STRATA_Q36_PREFILL_PEERS_HOST=1` 才启用 |
| `src/qwen35/q36_engine.cu` | `peer_cap_rows`/`peer_over_cap` + 超限整层回退本地（对齐 `prefill.cpp:3832`）；`peer_owner` 与活缓存的双重归属对账（R1 防线）；`peer_host_route()` 启动门禁；失败/回退 stderr |
| `docs/QWEN36.md` | 行契约、失败策略、归属对账、设备约定、无 P2P 实测证据与已知缺陷 |

关键符号：`PeerPrefill36::open/prepare/launch/wait`（新增 `Impl::primary`、`Impl::p2p`、`h_x/h_ids/h_out`、`pending_*`）；`Engine::peer_owned_layer/peer_window_fits/peer_plan/peer_cap_rows/peer_over_cap/peer_host_route`。

**偏离计划（均已在文档与代码注释写明）**：
1. 计划假定 P2P 可用 → 实测不成立，追加 B6a/B6b 主机暂存块。
2. 主机路由由「默认开」改为「需 `=1` 显式开」——实测跨 prompt→decode 边界必崩，默认开会把可用路径变崩溃路径。
3. B4 超限判定放在共用谓词 `peer_layer()` 而非 MMQ 分支的局部表达式——否则 `stream_kick` 的跳过与路由核的跳过会不一致，导致暂存区空洞或残留行被合并（coder 上报，reviewer 认可）。

**未解决（下一轮的活）**：主机路由跨 prompt/decode 边界的 `illegal memory access` 根因（见上表）；R1 结构性修法；B5 组级事件重叠（第二期，需 P2P 机器先测真实收益）。

