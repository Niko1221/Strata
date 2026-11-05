# Qwen3.8-Flash-Next IQ3_XXS on Ascend 910B2C

This is the current, opt-in Qwen3.8 backend in `cann/python`. It runs the
original GSQ-RCO IQ3_XXS GGUF with Torch and `torch_npu`; it does not use
vLLM, CUDA, llama.cpp, or the Qwen3.6 C++ reference endpoint.

The [Ascend-SACT Qwen3.8 deployment](https://ai.gitcode.com/Ascend-SACT/Qwen3.8-Flash-Next)
is the reference for the model's PLE, QSA, GDN, HyperConnection and MoE
semantics. That package targets a multi-card vLLM-Ascend deployment with CANN
9.1 and BF16/W8A8 weights. This port keeps those model semantics but uses the
75.84 GB IQ3_XXS GGUF and exact range reads so that one 64 GB 910B2C can run a
bounded single-request path.

## Tested environment

The measured host was:

| item | value |
|---|---|
| accelerator | 1 × Ascend 910B2C, 64 GiB HBM |
| OS | openEuler 24.03 LTS SP2 |
| CANN / driver | CANN 9.0.0 / driver 26.0.rc1 |
| Python | 3.11.15 |
| Torch | 2.10.0+cpu with `torch_npu` 2.10.0 |
| model | Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, two GGUF shards |
| verified context | 8,192 positions |

The model was measured with FP32 weights. FP16 is available as an experiment,
but the reported numerical acceptance results are FP32 results.

## Prepare the model

The repository pins both full-shard sizes and both small GGUF header hashes in
[`python/model_spec.py`](python/model_spec.py). A header-only directory is
enough to start; missing tensor ranges are fetched from the pinned ModelScope
source and stored in the disk cache.

```bash
cd /path/to/Strata
python3 -m pip install -r cann/python/requirements.txt

python3 -B cann/python/prepare_model.py \
  --model-dir /data/strata-qwen38/headers
```

For a disconnected host, place complete, SHA-256-verified shards in the same
directory and run:

```bash
python3 -B cann/python/prepare_model.py \
  --model-dir /data/strata-qwen38/full \
  --verify-full
```

Do not replace a header with an arbitrary truncated file. `WeightStore` checks
the pinned header digest, shard size, tensor count and GGUF architecture before
the first forward pass.

## Run one forward pass

The decoder library is built by the CANN build in `build-cann`:

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

The reproducible one-token check returns token `2405`, text `ither`, and the
same top-five order as the independent CPU oracle.

## Start the OpenAI/Anthropic-compatible service

`serve.py` defaults to loopback and port 18791. Use the same Python executable
that has `torch_npu` and Jinja2 installed. The service has one inference slot;
concurrent requests receive HTTP 429 rather than sharing recurrent/KV state.

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

The key file is optional for loopback-only experiments. Keep it outside the
repository and readable only by the service account. The health endpoint is
unauthenticated; model and inference endpoints require `Authorization: Bearer
<key>` when a key file is supplied.

Minimal checks:

```bash
curl -s http://127.0.0.1:18791/health
curl -s http://127.0.0.1:18791/v1/models \
  -H "Authorization: Bearer $STRATA_QWEN38_API_KEY"
curl -s http://127.0.0.1:18791/v1/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $STRATA_QWEN38_API_KEY" \
  -d '{"model":"qwen3.8-flash-next-iq3_xxs","prompt":[785],"max_tokens":1,"temperature":0}'
```

Supported routes are `/v1/completions`, `/v1/chat/completions`, and
`/v1/messages`. Text chat content is rendered through the GGUF chat template;
tools, multimodal content, and tool-choice are intentionally rejected. SSE
streaming is supported for completion and chat requests.

## Validation performed

The following checks were run against the 910B2C host and the exact GGUF
headers/cache:

| check | observed result |
|---|---|
| IQ3_XXS/IQ2/IQ4 scalar decoder and NPU GEMV | bit-exact decoder checks and real NPU matmul passed |
| one-token FP32 layer trace | all 48 layers passed; relative L2 stayed below `7.4e-6` |
| one-token logits | top-five same order; maximum top-five logprob error `4.64e-5` |
| natural 8-token greedy run | 8/8 top-five orders matched; maximum error `0.00274` |
| `/health`, `/v1/models` | HTTP 200; model id `qwen3.8-flash-next-iq3_xxs` |
| OpenAI completion | HTTP 200; token `2405` / text `ither` |
| OpenAI completion SSE | HTTP 200, `text/event-stream`, token event and `[DONE]` observed |
| OpenAI chat | HTTP 200; one-token response observed |
| Anthropic messages | HTTP 200; `type=message`, `stop_reason=max_tokens` observed |
| API key branch | no key HTTP 401; correct Bearer key HTTP 200 |
| existing services | 18789 and 18790 remained listening throughout the 18791 checks |

These are hardware measurements, not claims about every 910 or every CANN
release. The chat and Anthropic checks used the shortest legal empty user
message and one output token; long text prefill is currently very slow because
the Python path processes one token at a time.

## Boundaries

- The live run used `--context 8192`. The GGUF advertises a larger trained
  context, but 256K acceptance on 910B2C was not performed and must not be
  inferred from the metadata.
- The backend has no batching, speculative decoding, vision, tool calling or
  production-grade scheduler. It is a correctness/portability path.
- Quantized tensors are decoded on the host and uploaded to the NPU. A full
  75.84 GB model does not fit in this card, so header-only operation depends on
  exact HTTP Range support and sufficient disk cache unless complete shards are
  available.
- A 12 GiB expert-cache budget was used to coexist with the existing services.
  Increase it only after measuring HBM headroom on the target host.
- The service is not automatically added to Strata's generic `serve/` launcher
  or to the existing supervisor configuration. Start it as an explicit,
  separately budgeted opt-in process.
