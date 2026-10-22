# Q36：降同步税 + 汇合离开每层 moe_end

状态：设计定案（2026-10-19）
机台：192.168.1.98 · P100 + 2×P104 · Ornith IQ4_XS
起点：相位修好后 RING=2 ≡ ASYNC **41.9 tok/s**（止损）；残差 ~6.5 ms 仍钉在每层汇合

## 结论（先说）

**要超过 41.9，必须让「助手结果汇合」离开每层 `moe_end` 之前。**
只拧同步原语 / ZC / Spin / 加 K，都动不了这根钉子。

同步税拆成两轴一起做：

| 轴 | 含义 | 本机现状 |
|---|---|---|
| **次数** | 每 token 验票多少次 | ~40 次 pre-begin sync + ~40 次 helper 汇合 |
| **单次硬件延迟** | 每次验票固定开销 | Spin 后仍有驱动/队列往返；旧测有效 PCIe ~59 MB/s → 延迟主导 |

主推：**Option A（真跨层 / 延后 combine）**。
Option B（设备侧路由，砍 pre-begin）作 A 落地后的减税插件。
Option C（同层双助手）作 A 失败时的降级，不与 A 第一波混做。

## 钉死的约束（为何现在追平不超出）

1. **单块 `eparts`**：引擎级一块，本地 MoE / CPU / helper merge 都写它；`moe_end`→`k_moe_out*` 读它（`q36_engine.cu` ~1712, ~2247–2263）。
2. **层循环硬顺序**：`moe_helpers_wait(li)` 紧挨 `moe_end`（~2521–2527）——helper 行必须先进入当层 `eparts`。
3. **至多一个 pending slot**（~1589）：单线程层循环 + wait-before-`moe_end`；K=2 第二槽几乎不形成同 helper 双 inflight。
4. **下层 `StreamSynchronize(st)`**（~2716–2718）：吃掉上一层已入队的 `WaitEvent`，主机临界路径仍是「每层一次汇合」。
5. **pre-begin 主机读路由**：`begin_slot` 扫 `ids`+`cache_.slot_of`（`remote_experts.cpp` ~394–411）——这道 sync 去不掉，除非路由离主机。
6. Pascal **无 P2P**；`generate.cpp` 不动；RE 与 Flash-Next 共用；关旗默认路径语句级一致。

相位修复只恢复了与 ASYNC **同级的同层隐藏**；跨层从未兑现。

## 硬件延迟预算（本机类，有出处）

| 同步类 | 量级 | 出处 |
|---|---|---|
| 驱动 sleep 唤醒（Spin 已规避） | ~0.3 ms/往返 | `remote_experts.cpp` ~92–97 |
| pre-begin D2H + `StreamSynchronize` | ≤~1.5 ms/tok | runbook：删了只赚 1.5 |
| `ms_begin`（staging+launch） | ~1.5 ms/tok | 旧拆分 |
| 旧主机 `finish` 阻塞 | ~21 ms/tok | runbook §真正的瓶颈 |
| ASYNC/相位 RING 后残差汇合 | **~6.5 ms/tok** | Phase-3 两 helper 合计 |
| 有效 PCIe 回传 | ~59 MB/s | 非带宽瓶颈 |
| 非 hide 微残差上限 | **≤~2 ms/tok** | pre-begin ≤1.5 + compact scatter 未测尾巴 |

**死路（不再做）：** 再拧 ZC / Spin；CUDA 上再抄 `StreamQuery`；指望换 `EventSynchronize` 吃掉 6.5；靠 `use_graph` 吃 helper sync。

## 两轴怎么一起动

```text
                    降「单次硬件延迟」              降「次数」
                    (每次验票更便宜)                (少验票)
                           │                            │
         ┌─────────────────┴─────────────────┐          │
         │ 已饱和: ZC=0, ScheduleSpin        │          │
         │ 微: compact-finish (≤0.3–1 未测)   │          │
         │ 结构性: 设备侧路由 (Option B)     │◄─────────┤ 砍掉 ~40 次
         │           去掉 pre-begin 整流 sync│   pre-begin
         └─────────────────┬─────────────────┘          │
                           │                            │
                           ▼                            ▼
              残差 ~6.5 不在「单次更便宜」里      汇合离开 moe_end
              → 主战场是「少做/挪走汇合」         (Option A) 次数↓
```

目标墙钟带（相对 41.9）：

| 只做 | 期望 |
|---|---|
| 仅 B（砍 pre-begin） | ~43–46 |
| 仅 A（汇合离开 moe_end） | **50–57**（乐观顶 ~58） |
| A+B | 抠 50–57 上沿 |
| 仅 C（同层双助手） | ~45–50，汇合仍在 moe_end 前 |

## Option A — 主推：汇合离开每层 moe_end

### 机制（二选一，实现前再钉）

**A1 — 双 `eparts` ping-pong**
- `eparts[0|1]`：层 L 的 merge 可与层 L+1 的本地写并行。
- `moe_end(L)` 读 `eparts[L%2]`；要求该面 helper 行已就绪。
- 仍可能每层一次 wait，但 wait 可与**下一层 attention/路由**重叠 → 真跨层。

**A2 — deferred combine（更彻底）**
- `moe_end` / `k_moe_out*` **直接**读：本地行（仍在某缓冲）+ 未 drain 的 `helper_compact[slot]`（经 `src_row`/mask）。
- **取消**「必须先 `k_merge_helper_entry` 进共享 `eparts`」这道闸。
- `WaitEvent` 挪到：同 helper 下一 `begin_win` 需要该槽时，或旁路 combine stream；**不再**每层 `moe_end` 前强制 rendezvous。

推荐实现顺序：**先 A2 可行性探针（combine 核能否直接吃 compact）→ 再 A1 若 A2 改核成本过高。**

### 次数 / 延迟

- 汇合次数：从「每层 1」→「每 helper 每 K 窗 1」或「与下一层工作重叠后的隐性 1」。
- 单次硬件延迟：不变；总税下降。
- 必须配套：`helper_wait_*` 从单 slot 改为**每 helper 深度 ≤K 的队列**；打破 ~1589 单 pending 不变式。

### 不变量

- 关旗 / `rings==1`：字节意图不变。
- RING∥ASYNC 启动仍拒绝。
- cache mutate / `adapt` 前 `drain_all`（`d_meta_` 指针）。
- 缺 `WaitEvent` = 静默错数 → 只靠四配置 gen md5 抓。

### 验收

- ZC=0，`RING=2`，unset ASYNC；`--bench-tg 128` ≥3 轮。
- **及格：中位 >41.9 且 ≥48**；目标带 **≥50**。
- parity 四配置；与 flag-off gen 一致。
- **止损（kill）：** 中位 ≤41.9×1.02 或 &lt;48 → 停加 RING 深度；残差改打主卡/CPU/greedy。

### 工作量 / 爆炸半径

- 中高：主要 `q36_engine.cu`；可能小幅 RE API（多 pending 查询）。
- Flash-Next：低（slot-0 包装保留）。

## Option B — 插件：设备侧路由（降次数 + 一类硬件 RT）

### 机制

- 主卡上用 device expert-bitmap / 预计算 meta 建 owned、group、`src_row`。
- **删除**主机为读 `ids` 而做的 D2H + `StreamSynchronize(st)`（~2716–2718）。
- 无 P2P ⇒ 不做跨卡直接读 `d_ids`；ZC=0 保持。

### 次数 / 延迟

- **次数 −40/tok**（每层一道主卡整流 sync）。
- 单次：去掉一类「排空主卡流到 router」的硬件往返（预算 ≤1.5 ms/tok）。
- **不**单独吃掉 6.5 ms 汇合残差。

### 验收

- 先在 ASYNC=1 下测：墙钟相对 41.9 至少 **+1 ms/tok 等价**（约 ≥43.5）。
- flag-off 比特一致。
- 建议：**A 的 Verify 进 48+ 后再开 B**，避免两变量缠在一起。

### 风险

- 高：`begin_slot` 合同迁移；动态 launch 几何；与 Flash-Next 共享面。

## Option C — 降级：同层 Dual-Helper（Design B）

- 每层两卡不相交专家子集同时 `begin`；wait ≈ max 而非 sum。
- **汇合仍在 `moe_end` 前**，只把 6.5 → ~max(3.3)。
- 期望 ~45–50；作 A 失败后的退路。
- **拒绝跨 token pipeline**（依赖 `generate.cpp`）。

## 明确不做

| 不做 | 原因 |
|---|---|
| 加 K=3 当第一刀 | 单 pending + wait-before-moe_end 下第二槽已闲；加深度骗不过依赖 |
| 只做环形、合并且仍每层 `moe_end` 前 wait | 已证 ≡ ASYNC |
| 微同步税当主路径 | ≤~2 ms，撼不动 6.5 |
| 改 ZC 默认 / 动 `generate.cpp` / 假设 P2P | 项目硬约束 |

## 建议执行顺序（短 workflow 串）

```text
D0  探针（只读+最小核草稿）：k_moe_out 能否直接读 helper_compact+src_row？
      → 决定 A2 vs A1
D1  Implement A（多 pending 或双 eparts）→ Review → 远端 Verify
      及格 ≥48，目标 ≥50；kill &lt;48
D2  仅 D1 绿：开 B（设备路由）抠上沿
D3  仅 D1 红：评估 C 或承认单会话舒适区 ~42，转 batch/换主卡
```

文档落点：本文件；验收数追加到 `plans/2026-10-19-q36-23ms-breakthrough.md`。

## 证据索引

- 相位修复结果：`.run/R-ring-phase-local.txt`（41.9 ≡ ASYNC）
- 追平不超出定因：单 `eparts` + wait-before-`moe_end` + 单 pending（本文件 §约束）
- 同步税清单：`viking://…/q36_sync_tax.md`；runbook §真正的瓶颈
- RING 设计经验：设备合并必须搬走主机 scatter；ids/x sync 保留直到 Option B

### D0 探针结论 (2026-10-19)

- **primary: A1**（双 `eparts` ping-pong）
- **why（Decide）：** `scout.a2_feasible` 只证明 `k_moe_out*` 能按 mask/src_row 直读 compact（工程可做），但 `a2_blockers[0]` 已钉死：单靠融核不能把 WaitEvent/汇合挪出 `moe_end` 之前——`moe_end` 写出的 `h` 仍是下层 attn/rms 的硬依赖，残差 ~6.5ms 仍钉在同层 wait→`moe_end`。单块 `eparts` + 单 `helper_compact`/`d_helper_src_row` + 单 `helper_wait_slot` 属于 data layout/lifetime 阻塞（不是小改核能绕开的）；多 pending 若仍共用一块 `eparts` 面会与 L+1 local 写竞态。按决策规则应走 A1。hybrid（A1+A2）并不更便宜，且 A2 融核只是合并消除细节；D1 先只做 A1 便于单一变量验收。单 A2 预计量不变，过不了 ≥48。

- **scout：** `a2_feasible=true`（核级可融读 compact，但不能单独挪走汇合）

- **key blockers（短）：**
  - 融核仍不能把 WaitEvent 挪出同层 `moe_end` 前：下层消费 `h`
  - 单共享 `helper_compact` + `d_helper_src_row`（`q36_engine.cu:1580-1581`, `:4621-4626`）→ 延后 combine 前会被下一 wait 覆盖
  - 单 pending `helper_wait_slot`（`:1589-1593`, `:2821-2831`）→ 多 inflight 需每槽队列
  - `busy_`/`h_out_`：`begin_win` drain 复用（`remote_experts.cpp:384-386`, `:490-496`）→ 过早 defer = 静默错数
  - `rings==1`/关旗无 `ev_done`；Prompt MMQ 不走 RING → A2 不得扩到这些路径

- **key_sites：**
  - `src/qwen35/q36_engine.cu:156-181` — `k_moe_out` 从 `parts`/`eparts` 按 entry 行读
  - `src/qwen35/q36_kernels.cuh:226-264` — `k_moe_out_small` 同索引
  - `src/qwen35/q36_engine.cu:598-615` — `k_merge_helper` / `k_merge_helper_entry`
  - `src/qwen35/q36_engine.cu:1712` — 单块 `eparts` 分配
  - `src/qwen35/q36_engine.cu:2238-2263` — `moe_end` 启 combine
  - `src/qwen35/q36_engine.cu:2521-2527` — `moe_helpers_wait` 紧挨 `moe_end`（硬汇合）
  - `src/qwen35/q36_engine.cu:2821-2845` — RING wait：WaitEvent + compact H2D + merge
  - `src/core/remote_experts.cpp:384-496` — `begin_slot` drain / `ev_done` / `busy_`

- **d1_scope：**
  - `q36_engine.cu`：`eparts[2]` 分配/释放；所有写者绑 `face=li%2`
  - `moe_helpers_wait` / `moe_end` 读匹配 face；层循环 face handoff，使 face L 的 WaitEvent+merge 可与 face L⊕1 的 local MoE 写重叠
  - 打破单 pending：每 helper 深度 ≤RING K 的队列（含 slot/face/meta）；merge 目标 `eparts[face]`
  - 多 pending 下为每个 inflight 配备独立 compact（及匹配 mask/src_row），或证明同 face 生命周期内不复用
  - 文档化 `busy_`/`h_out_`：同槽下一 `begin_win` drain 不得早于该槽已被消费；保留 forward 末 / adapt 前 `drain_all`
  - 优先 engine-only；禁止 `generate.cpp`；D1 **保留** `k_merge_helper_entry→eparts[face]`（不切 A2 融核）

- **invariants：**
  - 关旗 / `rings==1` / 非 RING：语句级意图不变
  - RING>1 与 HELPER_ASYNC 启动仍拒绝
  - Flash-Next 共享 RE：slot-0 / 无 cudaEvent 路径保持
  - `moe_end(face)` 入队前：该 face 上 local 行已写完，且对应 pending 的 WaitEvent+merge 已入同一消费流
  - 缺 WaitEvent = 静默错数 → 四配置 gen + flag-off 一致性抓
  - cache mutate / adapt 前 `drain_all`；Prompt MMQ 路径语义不动；默认 unset byte-identical

- **verify_gate：** ZC=0 + `STRATA_Q36_HELPER_RING=2` + unset ASYNC；`--bench-tg 128`，≥3 轮取中位。及格：中位 >41.9 且 ≥48；目标带 ≥50。四配置 parity；RING gen 与 flag-off 一致。

- **kill：** 中位 ≤41.9×1.02 或 <48 → 停加 RING 深度 / 停扩 A；不进 D2(B)；转 D3 评估 C 或承认单会话舒适区 ~42。

- **falsify_off_moe_end：** 时间线/CUDA 序必须显示：层 L 的 `cudaStreamWaitEvent(ev_done(slot_L))+merge` 与层 L+1 写入 `eparts[(L+1)%2]` 的 local MoE 在主 GPU 上时间重叠；或调用序上 wait/merge(L) 不再紧挨同层 `moe_end(L)` 且与 L+1 local 无重叠。若仅把 `k_merge` 改名/并入 `k_moe_out`，而 WaitEvent 仍在同层 `moe_end` 正前方、且与 L+1 无重叠 → 判定未挪走汇合（假性 A）。

- **out_of_scope：**
  - D1 不做 Option B / Option C
  - D1 不加 K=3 作第一刀
  - 不改 `generate.cpp`；不改 ZC 默认；不假设 P2P
  - D1 不做 A2 融核（直读 compact / 删 `k_merge_helper_entry`）——留待 A1 验绿后的可选细节

- **state：** D0 done；**next = D1 Implement** short workflow **only after this section exists**。
