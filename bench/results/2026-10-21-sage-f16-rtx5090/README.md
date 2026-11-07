# Sage-A1.5 F16 at a near-full 256K window

The opt-in Strata path generates **36.58 tokens/s versus llama.cpp's 29.95: 22.2% faster** on this machine and workload. Both final runs use the same publisher F16 weights, F16 keys and values, 262,144 context capacity, **261,888 actual input tokens and 256 greedy outputs**, 512-token prompt chunks, MTP off and no input reuse.

This is a custom engine comparison, not the draft standard benchmark. The primary evidence is one fresh near-full run per final configuration. Two earlier controlled decodes from a saved real prefix reached 37.38 and 37.08 tokens/s; those are diagnostics, not additional fresh-prompt runs.

| Run | KV format | Expert placement | Decode tokens/s |
| --- | --- | --- | ---: |
| Initial llama.cpp reference | F16 | First 28 layers' experts on CPU, 8 threads | 29.636 |
| Initial Strata F16 prototype | FP32 | 31 resident experts per layer, 15 pool workers + caller | 7.459 |
| Final llama.cpp reference | F16 | First 27 layers' experts on CPU, 8 threads | 29.946 |
| Final Strata candidate | F16 | 60 resident experts per layer, 15 pool workers + caller | 36.581 |

The initial F16 prototype is captured in commit `6c1f1e2c`; all seven files were verified against the source hashes recorded when it was measured. It was not a released F16 path. The original loss is retained rather than replaced by the optimized results. The prototype-to-candidate difference combines attention, KV storage and cache-policy changes; it is not an isolated kernel speedup.

Decode throughput counts **255 timed intervals**, because prefill supplies the first output token. Final generation times are 8,515.299 ms for llama.cpp and 6,970.900 ms for Strata. Both evaluated all input tokens and returned all 256 outputs. llama.cpp's `truncated=true` is the final output reaching the context boundary; `cache_n=0`, `tokens_evaluated=261888` and the server log confirm no input truncation or reuse. Its ending `tokens_cached=262143` describes the populated slot after generation.

Generation was measured with other inference and builds excluded. The desktop, file indexer and other user services remained running. Preparatory fixture generation briefly overlapped baseline prefill and overlapped candidate prefill; it was paused well before candidate decode. The final reference's earlier 4K/32K rows also overlapped preparation and are excluded from the primary comparison. Raw prompt timings and first-turn wall times remain in the data, but this report makes no clean prefill-speedup claim.

## Machine, artifacts and settings

- NVIDIA RTX 5090 32 GB, Ryzen 9 9950X3D, 128 GB installed RAM; Linux, NVIDIA 615.78.08, CUDA 12.9, GCC/G++ 14.3.1. The desktop used roughly 1.5-1.9 GiB of VRAM.
- [Publisher F16 GGUF](https://huggingface.co/InternScience/Sage-A1.5-F16-GGUF), revision `f759478543247989c112b08d098d55b88fedf5ec`, `Sage-A1.5-F16.gguf`, 71,066,995,232 bytes, SHA256 `6eda214a950f3a1a359b58f9d027012dfefe3b56d13d613014bbf22c8e49654e`. The vision projector is excluded.
- llama.cpp `9febe63d5a379f8568bae7116a5b2ba844a4a4a6`, built on the measurement date with CUDA sm_120. Explicit `-ngl 99 -ncmoe 27 -t 8 -tb 8 -c 262144 -np 1 -b 512 -ub 512 -fa on --fit off`. This fits one more GPU expert layer than the initial reference. Placement and thread-count probes are retained.
- Strata base `011c9e600a93bbc9fd451c563501b5831c32bbca` (v0.2.2), plus the opt-in implementation. `--f16-weights --kv f16 --max-context 262144 --prefill 512 --pool-workers 15 --mtp 0`. The candidate source and binary hashes are in [provenance.json](provenance.json).
- Strata holds all 61.50 GiB of experts in pinned host memory and uses 14.41 GiB of resident experts plus 3 GiB of prompt staging. Its F16 cache still includes the MTP block even with MTP off (5.5 GiB versus llama.cpp's 5 GiB). Resident counts depend on other VRAM use. These are different placement strategies with the same weight and KV precision.

The input is a deterministic sequence of valve-inspection records with a request for a Python validator. Both continuations are readable reasoning about that request; the output texts are included. This is a speed and arithmetic check, not an assessment of reasoning, record counting or retrieval quality.

## Reproduce the timing

Build both CUDA engines with CUDA 12.9 and GCC 14; repository build guidance is in [DETAILS.md](../../../docs/DETAILS.md). Install the runner dependency with `python -m pip install psutil`. From the repository root, use one output directory so Strata reads the exact token ids made by llama.cpp's tokenizer:

```sh
python bench/results/2026-10-21-sage-f16-rtx5090/run.py llama \
  --name normal --out /tmp/sage-comparison --model /path/to/Sage-A1.5-F16.gguf \
  --llama-server /path/to/llama-server --threads 8 --ncmoe 27
python bench/results/2026-10-21-sage-f16-rtx5090/run.py strata \
  --name strata --out /tmp/sage-comparison --model /path/to/Sage-A1.5-F16.gguf \
  --strata-bin /path/to/strata --threads 15 --kv f16
```

The defaults are 262,144 context cells, 261,888 input tokens, 256 outputs and 512-token chunks. The measured final processes also ran `--cells 4096,32768,261888`; input reuse was off for every request. [prompt-hashes.json](prompt-hashes.json) identifies the exact inputs as packed little-endian int32 arrays.

## Correctness and default-path checks

CUDA CTest passed 4/4: model-family layout, independent F16 matrix/expert arithmetic, independent F32/F16 attention arithmetic, and existing Q8 expert parity. The attention oracle covers a populated 262,144-cell cache, grouped heads, causal tails, offset query rows and session pointer tables. CUDA, HIP and SYCL builds passed; HIP/SYCL logs show compilation rather than hardware execution. SYCL still does not run this model family.

For the actual F16 model, 128 teacher-forced outputs after a 16,384-token prompt gave mean KL divergence 0.001957 and 126/128 matching top-token choices. Repeating with the published helper gave 0.002022 and 126/128. The final cache budget is exercised across multiple 32-token updates. This checks the new arithmetic and changing CPU/GPU placement; it does not establish long-context task accuracy.

The default regression uses a synthetic Q8 file with four trunk layers plus MTP, 2,048 hidden width, FF 512, 256 experts/top-8, 16 query/2 KV heads and 16/32 GDN heads. It is **not a downloaded quantized Sage checkpoint**. At GPU and forced hybrid placement, release v0.2.2 and candidate tokens and complete logit bytes match across short prompts, streamed prompts, MTP-block preparation and both fixed and changing expert slots. Separate protocol runs verify MTP 0/2 outputs against the release and against plain decode. Small F16 fixtures also passed MTP 0/2/7 and session/batch-copy checks with fixed slots.

To reproduce the default check, build the release separately, then:

```sh
python -m pip install numpy
python bench/results/2026-10-21-sage-f16-rtx5090/make-fixture.py \
  --out /tmp/sage-fixture.gguf --cxx g++-14
python bench/results/2026-10-21-sage-f16-rtx5090/check-default.py \
  --stock /path/to/release/strata --candidate /path/to/candidate/strata \
  --fixture /tmp/sage-fixture.gguf --out /tmp/sage-default-check
ctest --test-dir /path/to/candidate/build \
  -R '^(model_family_test|q36_f16_parity|q36_attn_parity|native_expert_parity_q8_0_q8_0)$' \
  --output-on-failure
```

The fixture is 4,482,441,344 bytes; SHA256 `364f742c565ba7e96664e57685773d91ef029d4c3272ff6eb0260d860e2e836d`. The published generator reproduced that exact hash.

The actual-model logit check has a separate reference helper (its 28 CPU-expert layers are for correctness, not the timing reference):

```sh
g++-14 -O3 -std=c++17 bench/results/2026-10-21-sage-f16-rtx5090/llama-dump.cpp \
  -I/path/to/llama/include -I/path/to/llama/ggml/include \
  -L/path/to/llama/build/bin -Wl,-rpath,/path/to/llama/build/bin \
  -l:libllama.so.0 -lggml -lggml-base -lggml-cpu -o /tmp/sage-llama-dump
python bench/results/2026-10-21-sage-f16-rtx5090/check-logits.py \
  --model /path/to/Sage-A1.5-F16.gguf --prompts /tmp/sage-comparison/prompts-262144.json \
  --llama-dump /tmp/sage-llama-dump --strata-bin /path/to/candidate/strata \
  --out /tmp/sage-logits-check
```

## Failures, diagnostics and limits

The first 16K reference helper exceeded its 512-token batch and aborted; the corrected helper reads 512-token chunks. An initial older SYCL toolchain failed unchanged dependency helpers; the documented oneAPI 2026.1 toolchain built successfully. Review caught a stale SSM width in the first synthetic fixture; it was corrected and all 96 tensor shapes were checked before GPU regression. These were preparation failures, not successful measurements.

Nsight Systems is an uninstalled launcher on this host. Separate attention timings use CUDA events: the old F32 attention+merge took 9.612 ms/layer at 256K, versus about 0.848 ms for coalesced F32 attention. F16 tile selection and expert-cache sweeps are diagnostics. [cache-tuning.json](cache-tuning.json) labels saved-prefix decodes separately; the checkpoint restore reproduced all 256 original output tokens before it was used for tuning.

Not run: AMD/Intel hardware execution, Sage vision, multi-GPU, application-level model setup, real-model MTP/concurrent serving, long-context task accuracy, other GPUs or model quantizations. No global default is changed or general speedup claimed.

[results.json](results.json) retains the four near-full runs, settings, generated tokens and resource samples. [all-probes.csv](all-probes.csv) retains individual preliminary and final measurements; its non-primary rows are tuning data, including shared-host activity, not additional controlled acceptance pairs. [validation.json](validation.json) records checks and limits. Logs and source/binary hashes are adjacent.
