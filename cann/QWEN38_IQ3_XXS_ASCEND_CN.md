# Qwen3.8-Flash-Next IQ3_XXS 在昇腾 910B2C 上运行

这是当前 `cann/python` 中可选启用的 Qwen3.8 后端。它使用 Torch 和
`torch_npu` 运行原始 GSQ-RCO IQ3_XXS GGUF，不依赖 vLLM、CUDA、llama.cpp
或 Qwen3.6 C++ 参考服务。

[Ascend-SACT Qwen3.8 部署文档](https://ai.gitcode.com/Ascend-SACT/Qwen3.8-Flash-Next)
是 PLE、QSA、GDN、HyperConnection 和 MoE 语义的参考实现。该项目面向搭载
CANN 9.1 的多卡 vLLM-Ascend 部署，使用 BF16/W8A8 权重。本移植保留这些模型
语义，但使用 75.84 GB 的 IQ3_XXS GGUF 和精确 Range 读取，使单张 64 GB
显存的 910B2C 能够运行受约束的单请求路径。

## 已测试环境

实测主机环境如下：

| 项目 | 值 |
|---|---|
| 加速卡 | 1 × Ascend 910B2C，64 GiB HBM |
| 操作系统 | openEuler 24.03 LTS SP2 |
| CANN / 驱动 | CANN 9.0.0 / driver 26.0.rc1 |
| Python | 3.11.15 |
| Torch | 2.10.0+cpu，配合 `torch_npu` 2.10.0 |
| 模型 | Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS，两个 GGUF 分片 |
| 已验证上下文 | 8,192 个位置 |

模型实测使用 FP32 权重。FP16 目前可以作为实验选项使用，但文档中的数值
验收结果均来自 FP32。

## 准备模型

仓库在 [`python/model_spec.py`](python/model_spec.py) 中固定了两个完整分片的
大小以及两个小型 GGUF 头部文件的哈希值。只准备头部文件即可启动；缺失的
张量区间会从固定的 ModelScope 源下载，并保存到磁盘缓存。

```bash
cd /path/to/Strata
python3 -m pip install -r cann/python/requirements.txt

python3 -B cann/python/prepare_model.py \
  --model-dir /data/strata-qwen38/headers
```

在无网络环境中，将经过 SHA-256 校验的完整分片放入同一目录，然后执行：

```bash
python3 -B cann/python/prepare_model.py \
  --model-dir /data/strata-qwen38/full \
  --verify-full
```

不要使用任意截断文件替换头部文件。首次前向计算之前，`WeightStore` 会检查
固定的头部摘要、分片大小、张量数量和 GGUF 架构。

## 执行一次前向计算

解码动态库由 CANN 构建目录 `build-cann` 生成：

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export ASCEND_RT_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/x86_64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"

python3 -u cann/python/qwen38.py \
  --model-dir /data/strata-qwen38/headers \
  --cache-dir /data/strata-qwen38/cache \
  --decoder build-cann/libstrata_cann_decode.so \
  --budget-gib 12 \
  --weight-dtype fp32 \
  --context 8192 \
  --tokens 785 \
  --max-tokens 1
```

可复现的单 token 检查会返回 token `2405`、文本 `ither`，并且与独立 CPU
oracle 的 top-five 顺序一致。

## 启动 OpenAI/Anthropic 兼容服务

`serve.py` 默认使用回环地址和 18791 端口。请使用已安装 `torch_npu` 和
Jinja2 的同一个 Python 解释器。服务只有一个推理槽位；并发请求会返回 HTTP
429，不会共享循环状态或 KV 状态。

```bash
python3 -u cann/python/serve.py \
  --model-dir /data/strata-qwen38/headers \
  --cache-dir /data/strata-qwen38/cache \
  --decoder build-cann/libstrata_cann_decode.so \
  --budget-gib 12 \
  --disk-cache-gib 16 \
  --weight-dtype fp32 \
  --context 8192 \
  --host 127.0.0.1 \
  --port 18791 \
  --api-key-file /run/strata-qwen38-api-key
```

仅限回环地址的实验可以不提供 key 文件。key 文件应放在仓库之外，并设置为
只有服务账号可读。健康检查接口不需要认证；如果提供了 key 文件，模型列表
和推理接口都需要 `Authorization: Bearer <key>`。

最小检查命令：

```bash
curl -s http://127.0.0.1:18791/health
curl -s http://127.0.0.1:18791/v1/models \
  -H "Authorization: Bearer $STRATA_QWEN38_API_KEY"
curl -s http://127.0.0.1:18791/v1/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $STRATA_QWEN38_API_KEY" \
  -d '{"model":"qwen3.8-flash-next-iq3_xxs","prompt":[785],"max_tokens":1,"temperature":0}'
```

支持 `/v1/completions`、`/v1/chat/completions` 和 `/v1/messages`。文本聊天内容
会通过 GGUF chat template 渲染；暂不支持工具、多模态内容和 tool-choice。
completion 和 chat 请求支持 SSE 流式输出。

## 已完成的验证

以下检查均针对 910B2C 主机以及指定的 GGUF 头部文件和缓存执行：

| 检查项 | 实测结果 |
|---|---|
| IQ3_XXS/IQ2/IQ4 标量解码和 NPU GEMV | 解码 bit-exact 检查及真实 NPU 矩阵乘通过 |
| 单 token FP32 层级 trace | 48 层全部通过；相对 L2 误差低于 `7.4e-6` |
| 单 token logits | top-five 顺序一致；最大 top-five logprob 误差 `4.64e-5` |
| 自然 8-token 贪心运行 | 8/8 的 top-five 顺序一致；最大误差 `0.00274` |
| `/health`、`/v1/models` | HTTP 200；模型 ID 为 `qwen3.8-flash-next-iq3_xxs` |
| OpenAI completion | HTTP 200；token `2405` / 文本 `ither` |
| OpenAI completion SSE | HTTP 200，`text/event-stream`，观察到 token 事件和 `[DONE]` |
| OpenAI chat | HTTP 200；观察到单 token 响应 |
| Anthropic messages | HTTP 200；`type=message`，`stop_reason=max_tokens` |
| API key 分支 | 无 key 时 HTTP 401；正确 Bearer key 时 HTTP 200 |
| 既有服务 | 18789 和 18790 在全部 18791 检查期间持续监听 |

这些是硬件实测结果，不能推断所有 910 型号或所有 CANN 版本都具有相同结果。
Chat 和 Anthropic 检查使用最短合法的空 user 消息，并只生成一个 token；由于
Python 路径逐 token 处理，较长文本的 prefill 目前很慢。

## 当前边界

- 实际运行使用 `--context 8192`。GGUF 虽然声明了更大的训练上下文，但没有在
  910B2C 上完成 256K 验收，不能根据元数据推断支持 256K。
- 后端暂不支持 batching、投机解码、视觉、工具调用或生产级调度器；这是一个
  正确性和可移植性路径。
- 量化张量在主机侧解码后上传到 NPU。完整的 75.84 GB 模型放不进这张卡，因而
  头部文件模式依赖精确的 HTTP Range 支持和足够的磁盘缓存；也可以预先准备完整
  分片。
- 为与既有服务共存，实测使用了 12 GiB 专家缓存预算。只有在目标主机测量出足够
  的 HBM 余量后，才应提高该预算。
- 该服务不会自动加入 Strata 通用 `serve/` 启动器或现有 supervisor 配置；请
  作为独立且单独预算的可选进程显式启动。
