# Strata NVLink Prefill Relay · NVLink 预填充中转

[English](#english) · [简体中文](#简体中文) · [Try the relay](#try-the-nvlink-prefill-relay) · [构建与试用](#构建与试用) · [Feature branch / 功能分支](https://github.com/tubsGreatVictory/Strata/tree/nvlink-full-prefill-relay) · [Upstream PR #6131 / 上游 PR](https://github.com/Niko1221/Strata/pull/6131)

## English

### 128K benchmark

On Linux with two RTX 2080 Ti GPUs connected by NVLink, the relay increased median prefill throughput by **71.15%** and reduced median time to first token by **41.53%**.

| Median metric | Direct upload | NVLink relay | Change |
| --- | ---: | ---: | ---: |
| Prefill throughput | 1,103.5 tok/s | 1,888.6 tok/s | +71.15% |
| Time to first token | 118.90 s | 69.51 s | −41.53% |

The paired A/B benchmark used UD-Q4_K_XL + Q8 Engram, 131,086 uncached prompt tokens, 128 generated tokens, and three runs per configuration. CUDA0 used PCIe Gen3 x4 and CUDA1 used PCIe Gen3 x16. The Release build used CUDA 13.5, SM75, MMQ, and MMQ K-quants. All eight expert-bearing prefill chunks used the relay in every enabled run, with matching planned and submitted byte counts. These results apply to the tested layout and workload.

### How the relay works

- **Direct path:** host RAM → PCIe Gen3 x4 → CUDA0.
- **Relay path:** host RAM → PCIe Gen3 x16 → CUDA1 staging buffers → NVLink → CUDA0.
- CUDA1 alternates between two 4 MiB staging slots for eligible streamed expert weights. CUDA events coordinate slot reuse and receiver-stream ordering, and `cudaMemcpyPeerAsync` copies the staged data to CUDA0. The 8 MiB total is reusable staging space. Large transfers are split across slot-sized copies.
- `expert_relay_test` verified 20 payloads totaling 41,944,896 bytes, including slot reuse and receiver-stream ordering.

### Try the NVLink prefill relay

Use Linux with the CUDA Toolkit, exactly two visible RTX 2080 Ti GPUs (SM75), CUDA peer access, and active bidirectional NVLink. Identify the x4 receiver and x16 staging GPU by PCI bus ID and negotiated link width at load, then record their UUIDs. Check that the two cards are connected via NVLink.

```bash
nvidia-smi --query-gpu=index,pci.bus_id,uuid,pcie.link.width.current,pcie.link.width.max --format=csv
nvidia-smi topo -m
```

```bash
git clone --branch nvlink-full-prefill-relay --single-branch \
  https://github.com/tubsGreatVictory/Strata.git Strata-NVLink
cd Strata-NVLink
./setup.sh --build --no-start --gpus 0,1
```

Setup may download model files. Use the newly compiled engine and keep both GPUs selected. Replace `0,1` if the two RTX 2080 Ti cards have different `nvidia-smi` indices.

Strata builds the engine's `CUDA_VISIBLE_DEVICES` from the run configuration's GPU selection, so a shell export of that variable can be overridden. To pin the physical order, merge this entry into the `env` object of the generated model run config. Replace the placeholders with the full UUIDs from the query above. CUDA enumerates the first UUID as CUDA0 and the second as CUDA1.

```json
{"env": {"CUDA_VISIBLE_DEVICES": "GPU-<x4-receiver-uuid>,GPU-<x16-staging-uuid>"}}
```

The relay is off by default. Export the switch and launch the Strata CLI or the generated `run-<model>.sh` from that same shell. A desktop app opened from the system launcher does not inherit this shell export. If the run config already defines the switch in `env`, set it to `"1"` there as well.

```bash
export STRATA_LOCAL_PREFILL_RELAY=1
```

Confirm activation in engine stderr or the engine log with `strata local: STRATA_LOCAL_PREFILL_RELAY active: RAM -> CUDA1 ... -> CUDA0 ...`. Check the logged PCI bus IDs against the two physical cards. Without that line, activation is unconfirmed. Check the effective environment, engine build, and initialization errors. During a long prefill, `route=enabled (full relay, solo threaded issuer)` confirms that a chunk actually used the relay. For an A/B check, restart with the variable unset, then set, and use the same model and uncached prompt in both runs. [Linux source build guide](https://github.com/tubsGreatVictory/Strata/blob/nvlink-full-prefill-relay/docs/SOURCE_INSTALL.md#build-from-git-on-linux-to-try-a-pull-request-or-change-the-code)

## 简体中文

### 128K 跑分

在 Linux 双 RTX 2080 Ti + NVLink 环境中，启用中转后，预填充吞吐中位数提升 **71.15%**，首 token 时间中位数缩短 **41.53%**。

| 中位数指标 | PCIe 直传 | NVLink 中转 | 变化 |
| --- | ---: | ---: | ---: |
| 预填充吞吐 | 1,103.5 tok/s | 1,888.6 tok/s | +71.15% |
| 首 token 时间 | 118.90 秒 | 69.51 秒 | −41.53% |

A/B 使用 UD-Q4_K_XL + Q8 Engram，输入 131,086 个未命中缓存的 token，生成 128 个 token，每组跑 3 次。CUDA0 为 PCIe Gen3 x4，CUDA1 为 PCIe Gen3 x16。Release 构建使用 CUDA 13.5、SM75，并启用 MMQ 和 MMQ K-quants。开启中转的每轮测试中，8 个包含专家权重传输的预填充块均走中转，计划与实际提交的字节数一致。以上结果对应这套拓扑和测试负载。

### 技术路线

- **直传路径**：系统内存 → PCIe Gen3 x4 → CUDA0。
- **中转路径**：系统内存 → PCIe Gen3 x16 → CUDA1 暂存缓冲区 → NVLink → CUDA0。
- CUDA1 用两个 4 MiB 暂存槽交替处理符合条件的专家权重传输。CUDA 事件协调槽位复用与接收流顺序，`cudaMemcpyPeerAsync` 将暂存权重复制到 CUDA0。合计 8 MiB 的暂存空间可反复复用，较大的传输按槽位容量拆分。
- `expert_relay_test` 验证了 20 份数据，共 41,944,896 字节，覆盖槽位复用和接收流顺序。

### 构建与试用

需要 Linux、CUDA Toolkit、恰好两张 CUDA 可见的 RTX 2080 Ti（SM75）、CUDA peer access，以及已激活的双向 NVLink。用 PCI Bus ID 和负载下实际协商的链路宽度识别 x4 接收卡和 x16 中转卡，记下各自的 UUID，并确认两卡之间存在 NVLink 连接。

```bash
nvidia-smi --query-gpu=index,pci.bus_id,uuid,pcie.link.width.current,pcie.link.width.max --format=csv
nvidia-smi topo -m
```

```bash
git clone --branch nvlink-full-prefill-relay --single-branch \
  https://github.com/tubsGreatVictory/Strata.git Strata-NVLink
cd Strata-NVLink
./setup.sh --build --no-start --gpus 0,1
```

安装过程可能下载模型文件。将模型配置指向本次编译的引擎，并保留双卡选择。若两张 RTX 2080 Ti 在 `nvidia-smi` 中的编号不同，请替换 `0,1`。

Strata 会根据运行配置里的 GPU 选择设置引擎的 `CUDA_VISIBLE_DEVICES`，因此 shell 中同名的变量可能被覆盖。要固定物理顺序，请把以下条目合并到生成的模型运行配置的 `env` 对象。把占位符替换为上面查到的完整 UUID。CUDA 会把排在前面的 x4 卡枚举为 CUDA0，把排在后面的 x16 卡枚举为 CUDA1。

```json
{"env": {"CUDA_VISIBLE_DEVICES": "GPU-<x4-receiver-uuid>,GPU-<x16-staging-uuid>"}}
```

功能默认关闭。执行下面的命令后，从同一个 shell 启动 Strata CLI 或生成的 `run-<model>.sh`。从系统菜单或桌面图标启动的 App 不会继承这个 shell 的变量。如果运行配置的 `env` 已定义此开关，也要在那里设为 `"1"`。

```bash
export STRATA_LOCAL_PREFILL_RELAY=1
```

以引擎 stderr 或日志里的 `strata local: STRATA_LOCAL_PREFILL_RELAY active: RAM -> CUDA1 ... -> CUDA0 ...` 为激活判据，并用日志中的 PCI Bus ID 核对两张物理卡。没有这行就需要检查引擎收到的环境变量、构建版本和初始化报错。长提示词预填充时，`route=enabled (full relay, solo threaded issuer)` 表示对应的块实际走了中转。做 A/B 对照时，在关闭和开启两种设置下分别重启 Strata，使用相同模型和相同的未缓存提示词。 [Linux 源码构建说明](https://github.com/tubsGreatVictory/Strata/blob/nvlink-full-prefill-relay/docs/SOURCE_INSTALL.md#build-from-git-on-linux-to-try-a-pull-request-or-change-the-code)
