# Experimental DeepSeek backend through deepMoE

Strata can use a separately built [deepMoE](https://github.com/acupof-ai/cachedMoE)
process for text chat with **DeepSeek-V4.1-Flash**. Strata supplies its existing
web app and OpenAI, Anthropic, and Responses endpoints; deepMoE supplies the
Vulkan/Slang inference engine, native FP4/FP8 arithmetic, an NVMe expert cache,
and optional identical-checkpoint mirror reads. Qwen's engine and installer
remain the default.

This is an explicit, source-built backend for Linux/RADV on a **128 GB AMD Strix
Halo** PC. It does not make a 510 GB checkpoint fit the hardware in Strata's
normal Qwen requirements, and it does not add DeepSeek kernels to Strata's
CUDA/HIP engine. No weights are downloaded or converted by this integration.
Both projects are MIT licensed; the checkpoint retains its own license.

## Start

1. Build deepMoE and prepare the native checkpoint as described in its
   [README](https://github.com/acupof-ai/cachedMoE#quick-start).
2. Install the optional tokenizer package in Strata's Python environment:
   `python -m pip install tokenizers`. Strata's standard server dependencies
   are also required.
3. Copy [deepmoe.json](examples/deepmoe.json) and set absolute paths for `exe`,
   `cwd`, and `model`. Select a cache budget that fits your machine.
4. Stop other GPU inference processes, then start:

```bash
python -m serve.server --engine deepmoe --config deepmoe.json --port 8095
```

Wait for the ready message and open <http://127.0.0.1:8095/>. The server keeps
one deepMoE process resident and serialises requests. The native tokenizer runs
on the CPU independently of generation. The checkpoint's own `encoding/encoding.py`
renders prompts; its directory is trusted local code, as in deepMoE's chat client.

```bash
curl http://127.0.0.1:8095/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"What is 2 + 3?"}],"reasoning_effort":"none","max_tokens":32,"temperature":0}'
```

Streaming uses the existing API's separate reasoning and content fields.
The native EOS ID (1) ends a reply. Cancellation drains the old turn before
another request starts, including when the client disconnects or a reasoning
budget closes the iterator. Use the existing `--api-key` option before listening
beyond loopback.

## Supported controls and limits

- Temperature, top-p, and seed reach deepMoE without translation. Request token
  budgets and context limits are enforced by Strata.
- Strata reasoning levels map to native effort: low = 50, medium = 75,
  high = 100; disabled thinking uses the checkpoint's chat mode. The renderer's
  unrequested thinking budget is 75.
- Top-k sampling is unavailable in deepMoE's current protocol. The web app sets
  it to zero for this backend; API clients must omit it or use zero. Non-neutral
  min-p and repetition/frequency/presence penalties are also rejected.
- **Text chat only.** Image input and tool calls are rejected before inference.
  DeepSeek DSML tool syntax has not been adapted to Qwen's parser. MCP, parallel
  engine slots, lazy startup, and trailing-effort mode are not supported.
- Mirror directories and other engine options can be appended to `args`, e.g.
  `"--mirror", "/mnt/second/models/DeepSeek-V4.1-Flash"`.
- The example disables persisted KV to avoid reusing unrelated saved state during
  initial validation. Remove `--no-kv-disk` to use deepMoE's normal KV persistence.

Exact routing is the example's default. Mask and top-K speculative acceptance
are approximate modes with documented quality limits; enabling them does not
establish equal quality or a speedup. See deepMoE's [measured results](https://github.com/acupof-ai/cachedMoE#measured-performance).
The backend does not download a draft model or turn speculation on implicitly.

## Validate

```bash
python -m unittest serve.test_deepmoe serve.test_server serve.test_detok
# Optional CPU-only checks with the local checkpoint and tokenizers package:
DEEPMOE_MODEL_DIR=/path/to/DeepSeek-V4.1-Flash python -m unittest serve.test_deepmoe_native
```

The new tests use a fake JSON-lines process and require no GPU, model download,
or tokenizer package. They cover startup, consecutive requests, EOS, cancellation,
iterator close/drain, restart, unsupported sampling, template effort, and both
OpenAI and Anthropic response paths. Native checkpoint CPU checks additionally
compare tokenization, incremental UTF-8 decoding, and prompt effort against the
checkpoint's renderer. Hardware smoke-test results are reported in the PR;
they are separate from throughput and quality benchmarks.
