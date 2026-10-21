# Community benchmark: L40S-48Q (vGPU, 48 GB), Strata 0.2.0-beta, six Unsloth quantizations of Qwen3.8-Flash-Next

**Status: complete.** Everything below was measured. The follow-up, with the same six GGUFs on Strata 0.2.1, interleaved 0.2.0/0.2.1 starts and a 26-task 3D suite in Blender, is in [2026-10-21-community-l40s-48q-0.2.1-unsloth-quants-3d](../2026-10-21-community-l40s-48q-0.2.1-unsloth-quants-3d/).

Measured 2026-10-19 and 2026-10-20 on Linux. These are six Unsloth GGUFs of the same model, UD-Q2_K_XL through Q8_0. They run on the same Strata 0.2.0-beta engine with the same run config. One question drives the report: on one 48 GB card with every expert in RAM, which quantization should you pick?

**Short answer.**
- **UD-Q4_K_XL is the knee.** Its first-token KL against Q8_0 is 0.037 (60 prompts) and 0.065 (160 prompts). It decodes about 30% faster than Q8_0.
- **UD-Q3_K_XL and UD-Q2_K_XL drift 3 to 6 times further from Q8_0.** They still score about the same as Q8_0 on the task suites we ran. Those suites are too easy for this model to separate the quantizations: most results are 95% or more.
- **Q8_0 on 0.2.0 matches Q8_0 on 0.1.45** with the same config. Decode is equal or 2-5% faster, prompt speed is equal, and the KL between the two is 0.012.

## Hardware and software

- **GPU:** NVIDIA L40S-48Q, a VMware vGPU profile of an L40S (Ada, sm_89). 49,152 MiB reported, about 42.2 GiB usable by one process. Driver 550.127.05, CUDA 12.4 driver API.
- **CPU and memory:** Intel Xeon Platinum 8580, 32 vCPU (AVX-512, AMX), one NUMA node. 251 GiB RAM for the speed and quality runs. The VM was moved to 503 GiB afterwards. On 503 GiB we ran only the 0.1.45 reference's GSM8K, IFEval, MATH and BBH and the tail-skip A/B; see Limits.
- **Storage:** an LVM volume on the host's storage. It was not characterised.
- **OS:** Ubuntu 22.04, kernel 6.8.0-138-generic.
- **Strata 0.2.0-beta, Linux packages:**
  - `strata-control-0.1.45.3-linux-x86_64.tar.gz`, run with its own Python 3.12;
  - `strata-engine-0.1.45.3-linux-x86_64-cuda-cuda12.tar.gz`, run as shipped.
  - Both were checked against the release's `SHA256SUMS.txt`, unpacked side by side, and launched with `serve/server.py`. The desktop app was not used.
- **Reference:** Strata 0.1.45, a source build of the engine with the 0.1.45 server, using the same config.
- **Tools:** `strata_bench.py` from the 0.1.45 tree, lm-evaluation-harness 0.4.13 and EvalPlus 0.3.1, all over the HTTP API.

## Models and configuration

- **Model files:** [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/766911a6b7369840a91dbcd95f9f997acaab6cd6), revision `766911a6b7369840a91dbcd95f9f997acaab6cd6`. Every shard's size and SHA-256 is in [model-manifest.json](model-manifest.json).
- **Packs** (tokenizer, dense weights, expert index; the experts are read from the GGUF): one per model, built with the 0.2.0 control package's `tools/iq_pack.py --compat-bf16`. Each took about 25 s.
- **MTP draft layer:** the one the 0.1.45 setup prepared, the same for all runs.
- UD-Q2_K_XL and UD-Q3_K_XL are not in the setup menu. They loaded and served with no special handling.

**One config for every model,** copied from our production config with only the paths changed. See [configs/](configs/), with local paths replaced by `<data>`:

```text
--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp <draft layer>
--max-context 262144 --kv fp16 --kv-resident 32768 --vision (GPU) --vram-reserve-mib 700
--conversation-cache-mib 32768 --conversation-cache-slots 4 --conversation-cache-min-tokens 4000
```

The `env` block is the one the 0.2.0 setup writes for Unsloth files when every expert is in RAM (`setup.py unsloth_default_env`):
- `STRATA_HC_Q8=1` for the UD files;
- `STRATA_MMQ_KQ=1` for UD-Q5/Q6;
- `STRATA_ROUTE_TAIL_SKIP=0` for all six.

The 0.1.45 reference uses the same block. For the last setting, see "Route tail skip: no speed effect here".

## Method

- **Order:** one model at a time, lightest first, one request at a time. Every model was started on its own fresh server. Between runs only the OS file cache stayed warm.
- **Speed:** one fresh request at 8K, 32K and 128K prompt tokens (98% of target). Each starts with a unique tag, so nothing comes from the conversation cache. Thinking off, 512 generated tokens, temperature 0. Decode and prompt rates are the engine's own timings. In the same session, `tools/strata_bench.py --runs 3` ran its short-chat, 4K/16K prompt and agent-session suite.
- **Quality against Q8_0:** the server ran with `STRATA_DUMP_FIRST_LOGITS`, and the full first-token distribution was saved for:
  - 60 general prompts;
  - 160 prompts by domain: Python, C++, English and Portuguese, 40 each.

  KL(Q8_0 ‖ model) is computed over the whole vocabulary. "Same top-1" counts the prompts whose argmax matches. This is the prompt (prefill) path only.
- **Needles:** `tools/needle_bench.py` at 128K, with needles at 10, 50 and 90% depth.
- **Tasks:** lm-evaluation-harness 0.4.13 `local-chat-completions` with `--apply_chat_template --seed 1234`. `--limit N` takes the first N documents. Separately, EvalPlus 0.3.1 HumanEval+ with all 164 tasks, greedy.
  - **Thinking was on**, the server's default. The server caps it to fit the request:
    - 3,072 of 4,096 tokens for GSM8K, MATH and BBH;
    - 768 of 1,280 for IFEval;
    - 384 of 768 for HumanEval+.
  - **GSM8K** ran with `max_gen_toks=4096`. With lm-eval's default of 256, correct answers were cut off before the final number. On UD-Q2_K_XL the strict score was 74% at 256 tokens and 98% at 4,096. Reports that use the default with a thinking model are measuring answer length, not accuracy.
  - **BBH and MATH geometry** also ran with `max_gen_toks=4096`.
- **Images:** 24 synthetic images: small text at 9-24 px, counting, a table, colour and position. Run on UD-Q4_K_XL and Q8_0 only.

## Results

### Speed (Strata 0.2.0, 262,144-token context)

| Model | GGUF (GiB) | First start (s) | Experts in GPU cache | Decode tok/s 8K / 32K / 128K | Prompt tok/s 8K / 32K / 128K | strata_bench chat / agent decode | strata_bench 4K / 16K prompt |
|---|---:|---:|---|---|---|---|---|
| UD-Q2_K_XL | 73.5 | 39 | 19,414 (33.9 GiB) | 126 / 122 / 111 | 1,340 / 3,194 / 3,425 | 138.9 / 136.5 | 2,731 / 3,472 |
| UD-Q3_K_XL | 83.8 | 43 | 15,568 (33.0 GiB) | 107 / 108 / 105 | 1,435 / 3,484 / 3,417 | 116.5 / 128.1 | 2,692 / 3,464 |
| UD-Q4_K_XL | 103.7 | 118 | 11,203 (32.7 GiB) | 98 / 99 / 96 | 1,345 / 2,828 / 2,739 | 104.4 / 113.0 | 1,723 / 2,908 |
| UD-Q5_K_XL | 147.4 | 159 | 8,758 (32.7 GiB) | 89 / 90 / 83 | 1,166 / 2,288 / 2,167 | 86.9 / 98.4 | 1,249 / 2,289 |
| UD-Q6_K_XL | 157.5 | 148 | 7,878 (32.6 GiB) | 82 / 82 / 78 | 1,091 / 2,037 / 1,928 | 77.9 / 90.7 | 1,086 / 2,020 |
| Q8_0 | 175.3 | 198 | 6,627 (32.2 GiB) | 75 / 73 / 72 | 977 / 1,703 / 1,613 | 69.9 / 76.3 | 898 / 1,677 |
| Q8_0, Strata 0.1.45 | 175.3 | 78* | 6,629 (32.2 GiB) | 74 / 70 / 70 | 972 / 1,712 / 1,616 | 66.5 / 75.5 | 902 / 1,682 |
| UD-Q4_K_XL, tail skip at the default (7) | 103.7 | 53* | 11,203 (32.7 GiB) | 97 / 103 / 96 | 1,275 / 2,778 / 2,726 | 104.2 / 111.8 | 1,729 / 3,053 |

\* The shards were already in the file cache.

- Every step from Q2 to Q8 costs about 10% decode.
- Decode barely depends on context length: 3-10% lower at 128K than at 8K.
- In prompt speed, Q2 and Q3 tie at about 3,400 tok/s. From Q4 on, prompt speed falls with size.
- VRAM with everything loaded: about 43.4 GB in every run. The engine warned "only ~350 MiB of VRAM free", as 0.1.45 does on this card. We saw no slowdown.
- **1M context:** Q8_0 with `--max-context 1048576` and YaRN, the options from our production config, started and answered: 59 tok/s decode at 8K. We did not run long-context tests at 1M. The first start took 824 s, against 112-187 s for the same model at 1M on 0.1.45 in production on this box. We did not look into why.

### Quality against Q8_0 (same engine, first-token logits)

| Model | KL, 60 prompts | median / p90 | same top-1 | KL, 160 prompts | same top-1 | KL by domain: Python / C++ / English / Portuguese | needles 128K |
|---|---:|---|---|---:|---|---|---|
| UD-Q2_K_XL | 0.202 | 0.146 / 0.429 | 53/60 | 0.405 | 115/160 | 0.341 / 0.529 / 0.324 / 0.425 | 3/3 |
| UD-Q3_K_XL | 0.108 | 0.063 / 0.311 | 54/60 | 0.158 | 133/160 | 0.163 / 0.193 / 0.132 / 0.143 | 3/3 |
| UD-Q4_K_XL | 0.037 | 0.012 / 0.096 | 57/60 | 0.065 | 143/160 | 0.091 / 0.083 / 0.036 / 0.049 | 3/3 |
| UD-Q5_K_XL | 0.023 | 0.008 / 0.070 | 55/60 | 0.034 | 152/160 | 0.043 / 0.032 / 0.029 / 0.033 | 3/3 |
| UD-Q6_K_XL | 0.015 | 0.006 / 0.033 | 56/60 | 0.037 | 148/160 | 0.079 / 0.029 / 0.015 / 0.025 | 3/3 |
| Q8_0 | - | - | - | - | - | - | 3/3 |

By domain, 0.2.0 against 0.1.45 on Q8_0: Python 0.037, C++ 0.024, English 0.012, Portuguese 0.022.

**The same Q8_0 on two engine versions,** with the same config and profile and the same expert-cache size (6,627 against 6,629 slots):

| Comparison | KL, 60 prompts | same top-1 | KL, 160 prompts | same top-1 |
|---|---:|---|---:|---|
| 0.2.0 against 0.1.45 | 0.012 | 57/60 | 0.024 | 155/160 |
| 0.1.45 against our 0.1.42 run of 2026-10-10 | 0.030 | 56/60 | - | - |
| 0.2.0 against our 0.1.42 run of 2026-10-10 | 0.029 | 57/60 | - | - |

We have no repeat on the same engine, so we cannot give the run-to-run floor. The 0.1.45 → 0.2.0 difference is the size of the UD-Q6_K_XL → Q8_0 difference. The KL values of UD-Q5/Q6 should be read with that in mind.

### Task accuracy

| Model | HumanEval+ base / plus (164) | GSM8K strict (100) | IFEval prompt / instruction strict (100) | MATH geometry, math_verify (100) | BBH geometric_shapes (250) | BBH navigate (250) |
|---|---|---|---|---|---|---|
| UD-Q2_K_XL | 93.3 / 89.0 | 98 | 88 / 91.4 | 95 | 80.0 | 99.6 |
| UD-Q3_K_XL | 89.6 / 87.2 | 99 | 83 / 89.0 | 96 | 80.0 | 100.0 |
| UD-Q4_K_XL | 93.9 / 91.5 | 99 | 84 / 89.0 | 95 | 80.0 | 100.0 |
| UD-Q5_K_XL | 92.7 / 89.6 | 98 | 82 / 87.7 | 96 | 80.0 | 100.0 |
| UD-Q6_K_XL | 92.1 / 89.0 | 99 | 86 / 90.2 | 96 | 80.0 | 100.0 |
| Q8_0 | 92.7 / 89.0 | 99 | 86 / 90.2 | 97 | 80.0 | 100.0 |
| Q8_0, Strata 0.1.45 | 93.3 / 90.2 | 99 | 85 / 89.6 | 95 | 80.0 | 99.2 |
| UD-Q4_K_XL, tail skip at the default (7) | 93.9 / 90.2 | 99 | - | - | - | - |

- With 100 documents, a difference of 5 points is within noise.
- **BBH geometric_shapes is capped at 80% for every model.** The same 50 documents fail everywhere: the gold label is "(K) trapezoid", and every model answers rectangle. So this task does not separate quantizations. BBH navigate is saturated.

### Images (synthetic, vision encoder on the GPU)

| Model | text 9 px | 12 px | 16 px | 24 px | counting | table | colour + position |
|---|---|---|---|---|---|---|---|
| UD-Q4_K_XL | 0/2 | 1/2 | 2/2 | 2/2 | 2/6 | 6/6 | 4/4 |
| Q8_0 | 0/2 | 1/2 | 2/2 | 2/2 | 2/6 | 6/6 | 4/4 |
| Q8_0, Strata 0.1.45 | 0/2 | 1/2 | 2/2 | 2/2 | 3/6 | 6/6 | 4/4 |

## Route tail skip: no speed effect here

The setup writes `STRATA_ROUTE_TAIL_SKIP=0` for Unsloth files when every expert is in RAM. We ran UD-Q4_K_XL once more on 0.2.0 without it, so the engine default applied. The engine log confirms: "STRATA_ROUTE_TAIL_SKIP=7 is ON (default)". During the run, about 2.8 million missed experts were skipped.

| UD-Q4_K_XL on 0.2.0 | Decode 8K / 32K / 128K | strata_bench chat / agent | HumanEval+ base / plus | GSM8K |
|---|---|---|---|---|
| `STRATA_ROUTE_TAIL_SKIP=0` (setup) | 98 / 99 / 96 | 104.4 / 113.0 | 93.9 / 91.5 | 99 |
| default (7) | 97 / 103 / 96 | 104.2 / 111.8 | 93.9 / 90.2 | 99 |

- On this card the skip gives no measurable speed. 96-98% of decode expert lookups hit the GPU cache, and almost nothing is read over PCIe. The engine's own message says the gain is for 12-16 GB cards that miss experts.
- The task scores are the same within noise.
- So the setup's `=0` costs nothing here, and we would keep it.

In an earlier draft we guessed that this setting explained why UD-Q4_K_XL decodes at 98 / 99 / 96 tok/s here, against 120 / 130 / 116 in our older production runs. **The A/B rules that out.** The remaining known difference is the workload: the old runs generated 256 tokens of a different prompt, and these generate 512 tokens of a summary, which changes how many MTP drafts are accepted. We did not test that. The 0.1.45 reference with the same config, Q8_0 at 74 / 70 / 70 tok/s, shows that the engine version is not the cause either.

## Issues found

- [#3269](https://github.com/Niko1221/Strata/issues/3269): the 0.2.0 Linux control package's `tools/iq_pack.py` could not find `gguf-py`. We worked around it with `STRATA_GGUF_PY`. In 0.2.1 the tool looks in more places, and its message says to run `setup.py` once or set the variable.
- No engine errors and no failed requests in any run.

## Limits

- **A host problem after the RAM upgrade, now fixed.** The VM's RAM was raised from 251 to 503 GiB after the speed and quality runs. After that, registering a large amount of pinned host memory crashed the vGPU guest driver (a kernel Oops in `rm_ioctl` during `cudaHostRegister`) and then hung the VM. A bare `cudaHostRegister` test reproduced it without Strata.
  - **Cause:** the new RAM was placed on the host's other CPU socket (NUMA node 0), while the GPU sits on CPU 1. The hosting team moved the VM's memory, and the problem is gone.
  - **After the fix:** pinned registrations of 1 to 192 GiB pass, and Q8_0 at 1M context loads and finds 3/3 needles.
  - If you run Strata in a VM with a passthrough or vGPU card, check that the VM's memory sits on the same NUMA node as the GPU.
- **One machine, a vGPU, one request at a time.** No concurrency, no thermal or clock capture, and the background load of the host is unknown.
- **Speed runs are single requests per length,** inside one server session per model. Each table row is one session. The `strata_bench` columns are medians of 3 runs.
- **The first-token KL covers the prompt path only.** It does not cover the decode or verify path where the route tail skip acts.
- **No 3D or Blender task suite on 0.2.0.** The geometry and spatial tasks above are the nearest proxy. The 3D suite ran on 0.2.1 only; see the 0.2.1 folder.

## Files

| File | Contents |
|---|---|
| `tables.md`, `summary.json` | the tables above and every number behind them, generated by `scripts/gerar-relatorio.py` |
| `results/<run>/` | `velocidade.json` (speed), `agulhas.json` (needles), `kl-*.json`, `info.json` (start-up lines), `strata_bench/`, `lm-eval*/` (results and samples, gzipped), `evalplus/`, `visao-detalhe.json` (images), VRAM and RAM at ready |
| `configs/` | the run configs, test key removed |
| `engine-logs/` | engine and server logs (gzipped), the run's event log, the pack and download logs (`.log.txt`) |
| `scripts/` | the scripts that ran the battery: `pipeline.sh`, `medir.py`, `alternar.sh`, `gerar-configs.py`, `empacotar.sh`, and the rest |
| `model-manifest.json` | GGUF paths, sizes and SHA-256 at the pinned revision |

Left out:
- the first-token logit dumps, about 270 MB per run;
- the EvalPlus raw outputs, only the sanitised ones are kept.

Local paths are `<data>` and `<home>`. The comments in the scripts are in Portuguese.
