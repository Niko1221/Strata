# Q36 从 23.866 ms/tok 突破方案

状态：执行细案（2026-10-19）
机台：192.168.1.98 · P100 + 2×P104 · Ornith-1.5 IQ4_XS
起点：Phase-3 中位 **41.9 tok/s = 23.866 ms/tok**（ZC=0 + HELPER_ASYNC=1）

## 结论（先说）

**单会话能挖的软件头是把临界路径上剩的 ~6.5 ms 主机 wait 移走，目标带 50–57 tok/s（乐观顶 ~58）。**
再往上要动主卡前向或换卡；100 tok/s 在这台机器上结构上到不了。

| 阶段 | 目标 ms/tok | 目标 tok/s | 靠什么 |
|---|---|---|---|
| 现在 | 23.866 | 41.9 | ZC=0 + ASYNC |
| **B1 收口 RING** | ~17–20 | **50–57** | 藏掉剩余 ms_wait |
| B2 再抬命中 | ~15–18 | ~55–65 | 第三卡 / 更少 CPU 池（边际小） |
| B3 主卡变便宜 | ≤10 | 逼近 100 | 换主卡或核级加速 ≥20% |

## 时间预算（必须按这个拆活）

测于 `.run/R-phase3-mswait-local.txt`，中位 41.9。

| 分量 | ms/tok | RING 能不能动 | 说明 |
|---|---|---|---|
| helper `ms_wait` 残差 | **6.527** | **能（主目标）** | ASYNC 同层重叠之后仍堵在主机路径 |
| helper `ms_begin` | 1.754 | 几乎不能 | staging + 启动；RING 不消 |
| CPU 池残差 | ~3.0 | 间接 | 命中 94.5%；第三卡最多再啃一点 |
| greedy() | ~2.0 | 不能 | 与 helper 正交 |
| 主卡设备段 + 其它同步 | ~10–11 | 不能靠 RING | attn / local MoE / shared / 每层 D2H+sync |
| **墙钟** | **23.866** | — | — |

天真上界：`23.866 − 6.527 = 17.339 ms → 57.67 tok/s`。
现实带 50–57：begin、每层 ids/x D2H+`cudaStreamSynchronize`、槽满 `drain` 仍在。

ASYNC 已藏过约 2.2 ms（矩阵：37.4 → 40.8）。**不要再拿旧的 21 ms wait 当 RING 收益。**

## 突破路径（按顺序，上一阶绿了再开下一阶）

### B0 — 先修默认路径（阻塞验收）

workflow Review **FAIL check#5**。不修就不能谈远端稳态数。

| 卡点 | 文件 | 改法 |
|---|---|---|
| `rings==1` 仍 `cudaMalloc` + 每层 H2D `d_src_row_` | `src/core/remote_experts.cpp` `alloc_win` / `begin_slot` ≈120、455 | **仅 `rings_>1` 时**分配并拷 `d_src_row_`；`rings_==1` 跳过（设备合并只用 ring） |
| `open()` 里多一次 ZC probe HostAlloc/Free | 同文件 ≈288–303 | 去掉独立 probe；沿用 `alloc_win` 后按槽位 `z_x_/z_out_` 成败决定 `zero_copy_`（与改 RING 前顺序对齐） |

验收（本机 + 远端均可先做逻辑门）：
1. `unset STRATA_Q36_HELPER_RING`：四配置 gen md5 与改前 tip 一致（T=1，n=32）。
2. `STRATA_Q36_HELPER_RING=2` + `env -u STRATA_Q36_HELPER_ASYNC`：启动不拒；与 ASYNC 同时开则启动拒绝。
3. Flash-Next 全量能编过（slot-0 包装未破）。

**执行方式**：workflow（coder 修 → reviewer 复检 check#5 → explorer/coder 远端 Verify）。禁止手改散落。

### B1 — RING 稳态突破（本方案主战场）

配置：
```
STRATA_REMOTE_ZEROCOPY=0
env -u STRATA_Q36_HELPER_ASYNC
STRATA_Q36_HELPER_RING=2
--expert-cache-device1 auto --expert-cache-device2 auto
--bench-pp 512 --bench-tg 128 --no-warmup
```

及格线：
| 项 | 要求 |
|---|---|
| tg128 中位 | **≥50 tok/s**（相对 41.9 至少 +20%） |
| 乐观区 | 54–57 |
| 止损 | 中位 <45 或相对 41.9 增量 <5% → 停，查槽满 drain / WaitEvent 缺失 / 与 ASYNC 误开 |
| 正确性 | 四配置 gen md5 与 flag-off 一致；RING×ZC on/off 各至少 2 轮 |

量法注意：RING 走 `cudaStreamWaitEvent`，**不一定再记进 `ms_wait_`**。以墙钟 tok/s 为准，不要只看 PHASE3 行是否归零。

K=3：仅当 K=2 已进 50+ 且 timing 显示槽满 `drain` 仍占可见比例时再试；默认不先开。

### B2 — 边际（RING 贴带后再做）

1. **第三张 P104**（若有第四 CUDA 设备）：`--expert-cache-device3 auto`，啃残差 ~5% CPU。期望 ≤+3 ms/tok，破不了 56–58 结构顶。
2. **`--serve --batch 2..4`**：量整机聚合 tok/s。可能高于单会话 56，但不是「一条对话变快」。单独短测即可。

### B3 — 只有 RING 已贴顶仍不够时

主卡侧 ~10–11 ms 是下一堵墙。Pascal 无 CUPTI，用 `cudaEvent` 拆：
- attn（含 QSA/GDN）
- local grouped MoE
- shared expert
- 每层 ids/x D2H+sync

再决定是否值得动核。换更强主卡是唯一不靠核改就可能逼近 100 的路。

## 明确不做

| 不做 | 原因 |
|---|---|
| 用 MTP 冲单会话 100 | D=1 完美接受封顶 ~50–53；渐近 ~81 < 100 |
| 默认开 MTP / 加深 N | 好提示词打平；故事更慢 |
| 只做环形不做设备合并 | 已否决；散射留在主机等于把同一堵墙挪位置 |
| 走 `--layer-split` / `--peer-device` | qwen35 family switch 进不去 |
| 改 `remote_experts` ZC 默认 | 与 Flash-Next 共享 |
| 假设有 P2P | topo 全 NS/CNS |

## 建议 workflow 切分

```
B0-fix-default  → coder（两处卡点）→ reviewer（check#5）
B1-verify-ring   → coder/explorer 远端：parity + tg128×≥3 + 墙钟对比 41.9
B1-record        → 把稳态数与是否进 50–57 写进 runbook / 本文件状态行
B2-optional      → 仅 B1 绿且仍要系统吞吐时开 batch 扫
```

每阶结束停在可检查状态：有 md5 或有 tg128 中位数字，再开下一阶。

## 证据索引

- Phase-3 数：`.run/R-phase3-mswait-local.txt`；gate PROCEED_RING
- 56–57 语义收紧：`plans/2026-10-19-q36-beyond-100tok-roadmap.md` §56–57 复核
- Review FAIL：`plans/2026-10-17-q36-three-card-runbook.md` §Review (FAIL on check #5 only)
- ASYNC 同层 defer：`src/qwen35/q36_engine.cu` `moe_helpers_wait`
- RING 设计约束：设备合并必须搬走主机 scatter；`ids/x` D2H+sync 保留

### B0/B1 result (2026-10-19)

- **B0 fix**: `remote_experts.cpp` only — gate `d_src_row_` `cudaMalloc`/`H2D` behind `rings_>1`; remove `open()` ZC `HostAlloc`/`Free` probe before `alloc_win` (env-boolean `zero_copy_` → `alloc_win` map → post-clear). `rings1_no_dsrc` / `rings_gt1_keeps_dsrc` / `zc_probe_removed` = true. `generate.cpp` untouched; no commit.
- **B0 review**: **PASS** (check#5 statement-identity fixed; checks 1–4, 6 still pass).
- **B1 verify**: `median_tok_s` **38.9** (reps 38.8 / 38.9 / 39.1) vs Phase-3 41.9 → **−7.16%**, `pass_throughput=false`, `stop_low=true`. `parity_ok=true` (gen md5 `e6844a66…`); `ring_async_reject_ok=true`; binary md5 `5dda616f…`.
- **Log**: `.run/R-b1-ring-local.txt` (remote copies under `/home/Strata/.run/R-b1-ring-20261019-101438-*`). Wall-clock authoritative; do not trust PHASE3 `ms_wait` under RING.

### RING WaitEvent phase fix (2026-10-19)

- **What moved**: Defer HELPER_RING `cudaStreamWaitEvent` + `src_row`/compact H2D + `k_merge_helper_entry` from `moe_helpers` → `moe_helpers_wait` (same phase as ASYNC finish). `moe_helpers` RING keeps `begin_win` + owns→mask + `d_helper_mask` H2D + stash `helper_wait_{hi,ent,N,slot}`; `moe_helpers_wait` RING runs WaitEvent+merge when `helper_wait_slot>=0` (removed early `helper_ring_on()` return). Call order: `moe_helpers` → local MoE → `moe_helpers_wait` → `moe_end`. File: `src/qwen35/q36_engine.cu` only (`+163/−25` this phase); `generate.cpp` untouched; RING+ASYNC reject at `open_helpers` kept; no commit.
- **Review**: **PASS** (checks 1–6; blockers empty). Nits: remote wall-clock was required for ≥41.9 (done below); `h_out` async H2D vs later slot drain acceptable under RING=2 same-layer wait.
- **Verify**: `median_tok_s` **41.9** (reps 41.9 / 41.4 / 41.9) vs prior bad RING **38.9** → recovered; vs ZC=0+ASYNC baseline **41.9** → **0%** (`pass_vs_async=true`, `pass_throughput=false`, `stop_low=true`). `parity_ok=true`; `ring_async_reject_ok=true`; `build_ok`/`idle_ok`; binary md5 `6eabdd1608d0ec40296f4ec70940b74b`. Stamp `20261019-110240`. Wall-clock authoritative; recovers ASYNC baseline but does not beat it → stop_low.
- **Log**: `.run/R-ring-phase-local.txt`.

相关设计定案（汇合离开 moe_end + 两轴同步税）：`plans/2026-10-19-q36-rendezvous-off-moe-end.md`（D0 已定 primary=A1）。
