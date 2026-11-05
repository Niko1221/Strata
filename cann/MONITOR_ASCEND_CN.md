# 昇腾 NPU 接入 Strata 自带监控

监控链路：浏览器 → 本机 Strata Host → SSH 本地转发 → 昇腾模型服务。
浏览器在 `http://127.0.0.1:18793/#/monitor` 查看远端指标，模型 API 地址为
`http://127.0.0.1:18791/v1`。这两个本地端口用途不同。

## 启动并访问

1. 在昇腾服务器按 [中文部署文档](QWEN38_IQ3_XXS_ASCEND_CN.md) 启动
   `cann/python/serve.py`，监听 `127.0.0.1:18791`，指定 `--api-key-file`。
   用 `--device npu:1` 选择第二张卡；单卡机器使用 `npu:0`。
2. 在浏览器所在电脑建立 SSH 转发，并保持这个进程运行：

   ```powershell
   ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 127.0.0.1:18791:127.0.0.1:18791 -p <ssh-port> <user>@<host>
   ```

3. 使用 Node.js 24.12 或更新版本，在 `apps/host` 目录安装、构建和启动界面：

   ```powershell
   npm ci
   npm run build
   $env:STRATA_HOME = Join-Path $env:LOCALAPPDATA 'Strata-Ascend-Monitor'
   $env:STRATA_REAL_ENGINE = '1'
   $env:STRATA_BIND = '127.0.0.1'
   $env:STRATA_PORT = '18793'
   node server/index.mjs --dist dist
   ```

4. 在 Settings → Models → Custom model API 保存连接：

   | 字段 | 内容 |
   |---|---|
   | 名称 | Ascend 910B2C · Qwen3.8 |
   | Base URL | `http://127.0.0.1:18791/v1` |
   | 协议 | OpenAI Chat |
   | 模型 ID | `qwen3.8-flash-next-iq3_xxs` |
   | API key | 远端 key 文件中的值 |
   | 上下文 | 8192，需与服务参数一致 |

5. 打开 `http://127.0.0.1:18793/#/monitor`。自定义连接会自动出现在监控页，
   不依赖本机硬件探测或本机引擎安装；无需先在聊天中选择模型。

Strata Host 将连接 key 保存在 `STRATA_HOME/data/model-providers.json` 的受限文件中；
列表接口只返回 `apiKeySet`，浏览器通过 Host 代理请求指标，不会收到远端 key。
页面使用本机回环地址访问。其他电脑需要在那台电脑运行 Host 和 SSH 转发。

## 指标口径

| 指标 | 来源与含义 |
|---|---|
| NPU 利用率 | `npu-smi info` 的 AICore 百分比；空闲及未采到计算峰值时可能为 0 |
| HBM 已用 / 总量 | 优先采用 `npu-smi` 的整卡值，包含同卡其他进程；缺失时回退 Torch |
| 本进程 Torch 分配 | `torch.npu.memory_allocated(device)`，不同于整卡 HBM 占用 |
| 温度 / 功耗 | `npu-smi` 的摄氏温度和瓦数 |
| 提示词处理速度 | 已完成的输入 token / 已同步 NPU 的处理时间 |
| 生成速度 | 首个 token 之后的 decode token 数 / decode 时间；首个 token 不计入 decode 吞吐 |
| 权重缓存 / 预算 | 当前模型缓存已用量和启动参数预算，不是整卡 HBM 上限 |
| 完成 / 错误请求数 | 当前模型服务进程生命周期内累计，重启清零 |
| 最近请求 | 时间、状态、输入/输出 token、生成速度和总耗时 |

有请求运行时显示当前速度；空闲时保留最近完成请求的速度。指标每次读取结束后
等待 3 秒再刷新，后台页签暂停读取。上游异常时清空旧读数并显示错误；缺失字段
显示 `—`。最近请求最多保留 120 条，累计 token 数不会随着历史行淘汰而减小。
UI 字节量按 1024 进制换算，卡片沿用 GB 标签。

## 2026-11-06 实机验收

此次在新服务器第二张 910B2C 上重新部署，使用本地完整 IQ3_XXS 两分片、
Torch/CANN FP32 路径、12 GiB 权重缓存预算。逻辑设备为 `npu:1`，
`npu-smi` 的物理板号为 10；解析器按设备顺序匹配，不能把逻辑编号直接当物理板号。

| 检查 | 结果 |
|---|---|
| 模型健康检查 | `/health` HTTP 200 |
| 指标鉴权 | 无 key HTTP 401，正确 Bearer key HTTP 200 |
| Strata Host 指标代理 | HTTP 200，返回真实 `npu:1` 指标 |
| 推理请求 | 输入 token `[785]`、输出上限 3、temperature 0 |
| 推理结果 | HTTP 200，输出 `ither of them`，3 个 token |
| 提示词 / decode 时间 | 8.917 s / 15.908 s |
| 生成速度 / 请求总耗时 | 0.126 tok/s / 24.826 s |
| 运行状态 | 捕获 reading、generating；完成后累计请求 1、错误 0 |
| 浏览器页面 | 显示 HBM 19.1 / 64.0、29 °C、86.3 W 和上述请求记录；读数随时间变化 |
| 既有服务 | 原 8080 服务健康检查仍为 HTTP 200 |

这些是短请求的功能与监控验收，不代表吞吐优化已完成。服务配置上下文为 8192，
本次没有执行 8192 长提示词或 256K 上下文验收。

## 当前部署的文件位置

本次远端实例的启动脚本为 `/data/strata-monitor-20261106/start.sh`，日志为
`/data/strata-monitor-20261106/service.log`。key 实体文件位于
`/data/strata-monitor-state/api-key`，权限为 `0600`；
`/run/strata-qwen38-api-key` 是指向该文件的链接。启动脚本会在链接缺失时重建，
不会重新生成已有 key。key 不在仓库中。

监控服务当前由 `nohup` 启动，未配置服务器重启后自动启动；重启服务器后需要
重新运行启动脚本。SSH 隧道断开后，页面会提示读取失败；恢复转发即可继续查看。
