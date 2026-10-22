# Q36 后续吞吐方案（相对 ~100 tok/s）

状态：研究定案（2026-10-19）
机台：192.168.1.98 · P100-12G 主卡 + 2×P104-8G helper · Ornith-1.5-35B-A3B-IQ4_XS
当前稳态：`STRATA_REMOTE_ZEROCOPY=0` + `STRATA_Q36_HELPER_ASYNC=1` → **~41.9 tok/s**（Phase-3 median）

## 结论（先说）

**100 tok/s 在这台机器、这个模型上结构上不可达。** 不是「再调几个开关」的事。

| 路径 | 已测 / 推算上限 | 依据 |
|---|---|---|
| 无投机，把**剩余**助手 wait 藏掉（RING 理想） | **~50–57 tok/s**（乐观顶 ~58） | 见下「56–57 复核」 |
| MTP 完美接受（渐近） | **~81 tok/s** | 行模型 `18.01 + 6.16·R`；a=1 时每多 1 token 付 2 行 ≈12.3 ms > 10.0 |
| MTP D=1、接受率 100%（实测） | **~50–53 tok/s** | 验证步 40.4 ms / 2 tok → 20.2 ms/tok；最快实测 **53.3** |
| bubble 全消的乐观拆分 | **~84–93 tok/s 仍 < 100** | runbook 2026-10-19 MTP 标定定稿 |
| **目标 100** | **10.0 ms/tok** | 比主卡剩余段还小 |

100 tok/s 要求每 token ≤10 ms。现状 ~24 ms。RING 最多再挖临界路径上剩的 ~6.5 ms；剩下是主卡前向、残差 CPU 池、greedy —— 都不是「藏 wait」能消掉的。

### 56–57 复核（2026-10-19）

**算术对，语义要收紧。**

| 量 | 数 |
|---|---|
| Phase-3 墙钟（中位 41.9 tok/s） | **23.866 ms/tok** |
| 两 helper `ms_wait` 合计（ASYNC 下测） | **6.527 ms/tok** |
| 天真减法 `墙钟 − wait` | **17.339 ms → 57.67 tok/s** |
| 再减 `ms_begin` 合计 1.754 | **15.585 ms → 64.16**（夸大：begin 藏不掉） |

关键点：Phase-3 的 `ms_wait` **已经是在 `HELPER_ASYNC=1` 下测的**。ASYNC 把 `finish()` 挪到同层本地核 / shared / CPU 池之后（`q36_engine.cu` `moe_helpers_wait`），所以这 6.5 ms 是**同层重叠之后仍堵在临界路径上的残差**，不是「从冷启动起的全部助手等待」。

对照（同一机、ZC=0 四格矩阵中位）：async 关 37.4 → 开 40.8，ASYNC 大约已藏 **~2.2 ms**。Phase-3 再量到的 6.5 ms，才是 RING（跨层 + 设备合并、主机不再 `finish` 阻塞）理论上还能挖的部分。

因此：
- **~57–58 tok/s** = 把这 6.5 ms 残差**全部**从墙钟拿掉的乐观上界（`23.9 − 6.5`）。
- **现实带 ~50–57**：`ms_begin`、每层 ids/x 的 D2H+`cudaStreamSynchronize`、槽满时 `drain` 仍在主机路径上；RING 实跑往往到不了天真上界。
- 旧文「17.8 ms 主卡地板 → 56」来自 ZC 默认、wait≈21 时代的差减；runbook 后文已写明该地板在本机配置下没出现过。数字碰巧接近，**不要混用那套前提**。
- 旧「藏掉整条助手路径 → 56」若按「藏 wait+begin」会虚高到 ~64；begin 不能靠 RING 消掉。

## 当前时间预算（Phase-3，ZC=0+ASYNC，tg128）

测于 2026-10-19，日志 `.run/R-phase3-mswait-local.txt`。

| 分量 | ms/tok | 说明 |
|---|---|---|
| 墙钟 decode | **~23.9** | 中位 41.9 tok/s |
| helper ms_wait 合计 | **~6.53** | CUDA1 3.269 + CUDA2 3.258 |
| helper ms_begin 合计 | **~1.75** | 0.890 + 0.864 |
| CPU 池残差 | **~3.0** | 命中 94.5%，约 5.5% 仍 CPU |
| greedy() | **~2.0** | 验证步分解 |
| 主卡设备段 + 其它同步 | **~10–11** | 未做核级拆分（CUPTI 不可用） |

到 100 还差 **≈14 ms/tok**。RING 理想只贡献其中 **临界路径残差 ~6.5**（不是「从未重叠的 21 ms」）。

## 后续方案（按优先级，全部相对「逼近上限」而不是 100）

### P0 — 做完并验收 `HELPER_RING`（Implement 已回；Review 未过默认路径）

- **做什么**：`STRATA_Q36_HELPER_RING=2` + 设备侧合并；与 `HELPER_ASYNC` 互斥。
- **期望**：把 ASYNC 后仍剩的 ~6.5 ms 主机 wait 移出临界路径 → **目标带 50–57 tok/s**（乐观顶 ~58；不是 100）。
- **验收**：ZC=0、unset ASYNC、RING=2；`--bench-tg 128` 交错 ≥3 轮；parity 四配置。注意：RING 路径上 GPU 等待走 `WaitEvent`，**不一定再记进 `ms_wait_`**（Review nit）；要以墙钟 tok/s 为准，不要只看 PHASE3 行是否归零。
- **已知卡点（workflow Review FAIL check#5）**：`rings==1` 仍每层分配/H2D `d_src_row_`；`open()` ZC probe 改了默认分配顺序——默认路径尚未做到语句级一致。先修这两处再谈远端稳态数。
- **止损**：若稳态 <45 或相对 41.9 几乎无增益 → 查槽满 drain / 与 ASYNC 误开，不继续加深度。

### P1 — 第三张 P104（抬命中，不破 56）

- **做什么**：`--expert-cache-device3 auto`（需第四个 CUDA 设备或换绑卡；本机目前只有 CUDA0/1/2）。
- **期望**：残差 ~5.4% CPU → 更低，约再挖 **≤3 ms/tok** → 顶多 +几 tok/s。
- **不要指望**：命中率从 46%→94% 只换 +19%；第三卡同样不消主卡前向。

### P2 — serve 多会话聚合吞吐（未测，可能是「系统吞吐」出路）

- **做什么**：`--serve --batch 2..N`，量的是**整机 tokens/s**，不是单会话时延。
- **为什么值得测**：单会话被主卡前向钉在 ~56；多会话若能吃满两张 P104 空闲算力，**聚合**可能明显高于 56，但仍不是「一条对话 100 tok/s」。
- **验收**：同硬件、同模型，batch=1 vs 2 vs 4 的聚合 tok/s + 单会话 P50 时延。

### P3 — 主卡前向变便宜（唯一可能逼近 100 的方向，成本高）

100 要求主卡侧从 ~17–24 ms 压到 ~10。可能手段（均为大工程，且 **Pascal 无 P2P**）：

1. **注意力 / MoE 核再快 ~20%+**（每行 6.16→≤5.0 ms）—— 无 CUPTI，先用事件打点拆 attn vs local MoE vs shared。
2. **换更强主卡**（非软件）：P100 换成带宽/算力更高的卡，helper 拓扑仍 NS/CNS 时 ZC=0 逻辑仍适用。
3. **不要走的**：`--layer-split` / `--peer-device` / `STRATA_SPLIT_OVERLAP` / `--pipeline-windows` —— qwen35 family switch 后进不了 Flash-Next 路径（`docs/QWEN36_PASCAL.md`）。

### 明确不做 / 低收益

| 杠杆 | 原因 |
|---|---|
| 默认开 `--mtp` | 好提示词打平；故事续写更慢；D≥2 接受饱和 |
| 加深 MTP N | 接受率随 N 崩；边际 +12 ms/草稿 |
| `--kv int8` 当吞吐手段 | +0.1 tok/s 量级 |
| 只做环形、不做设备合并 | 已否决；散射必须进设备 |
| 改 `remote_experts` ZC 默认 | 与 Flash-Next 共享，禁改 |
| 假设有 P2P | topo 全 NS/CNS |

## 建议执行顺序

从 **23.866 ms/tok** 往下挖的细案见：`plans/2026-10-19-q36-23ms-breakthrough.md`。

1. **B0**：修 Review FAIL check#5（`rings==1` 的死 `d_src_row_`；`open()` ZC probe 顺序）→ 默认路径语句级一致。
2. **B1**：远端验收 `HELPER_RING=2` 墙钟是否进 **50–57** 带（乐观顶 ~58）。
3. **写清产品预期**：单会话舒适区 ~40–55；100 需换主卡或引擎级把每行压到 ≤5.0 ms。
4. **B2 可选**：第三卡抬命中；或 `--batch` 聚合吞吐。
5. **B3 仅当 RING 已贴顶仍不够**：主卡事件级拆分，再决定是否动核。

## 证据索引

- Phase-3：`.run/R-phase3-mswait-local.txt`；gate **PROCEED_RING**（ms_wait≈3.26≥3）
- 从 23.866 突破细案：`plans/2026-10-19-q36-23ms-breakthrough.md`
- 瓶颈与 ZC/async：`plans/2026-10-17-q36-three-card-runbook.md` §2026-10-18
- 100 不可达钉死：同文件 §2026-10-19 MTP 标定定稿
- Pascal 拒 layer-split：`docs/QWEN36_PASCAL.md`
