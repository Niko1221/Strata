# Synthetic repetition reproduction for Strata issue #606

This bundle contains two entirely synthetic requests and a Python standard-library HTTP client. It reproduces a **shared, intermittent** failure seen with both CUDA 12 native engine 0.1.58 and 0.1.59 on two V100s. It is **not evidence of a new 0.1.59 regression**, and it does not diagnose the underlying kernel or model cause.

Use a **fresh server process for each complete two-request sequence**, with no intervening requests. Do not add warmups. Keep the model, GPU power/clock settings, native options and 65,536-token context identical across comparisons. The client does not start, stop, reconfigure or monitor the server; manage its lifecycle separately.

## Requests and observed behavior

1. `01-long-64000.json`: 64,000 actual rendered input tokens, non-thinking, 512 output tokens. It asks for three synthetic audit codes plus parser code. The client requires the full 512-token output before proceeding.
2. `02-coding-61000-budget2999.json`: 61,000 actual rendered input tokens, thinking enabled, explicit `reasoning_budget_tokens: 2999`, maximum 4,000 output tokens. It asks for an escaped ledger parser using only Python built-ins.

Both use greedy settings (`temperature=0`, `top_k=1`, `top_p=1`, seed 42). Token counts were verified using the installed serving normalization/template/tokenizer and the exact input-ID sequences matched between frontends. They may differ on other models/templates; the client rejects mismatched counts or cached inputs rather than treating them as this reproduction.

In three failed attempts (new native/new frontend, new native/old frontend, and a repeat old native/old frontend), the second request returned **2,198 API output tokens**, with **2,199 generated tokens in the native log**, no visible answer, and the frontend's existing guard for 256 consecutive `0` tokens. The decoded reasoning and API choice/delta sequence matched exactly across those attempts. An earlier old-native/old-frontend attempt instead returned valid code passing 25 internal assertions. This is therefore intermittent even with the old package. No thermal-counter increase, ECC increase or Xid was observed in those bounded attempts. No native token-ID trace was enabled.

The repeat guard is left enabled. Failed shorter output is not a performance result. The code checker used internally is intentionally not included: **this client never executes generated code** and reports only text properties, counts and hashes.

## Run

Python 3.10+ is sufficient; no third-party packages are required. Supply the API key through the `STRATA_API_KEY` environment variable (or select another variable with `--key-env`). No key is accepted as a command-line value or printed.

```text
python reproduce.py --url http://localhost:8080 --confirm-fresh-server
```

The request files preserve the exact tested model alias. If your serving alias differs, add `--model YOUR_ALIAS`; only that JSON field changes, and the emitted wire-body hash records the difference. `manifest.json` contains the original file and wire hashes. The client sends the two files in order, without retries, and prints one JSON summary per response. It emits no prompts, model answers, endpoint, authentication value or server identifiers.

`longest_zero_character_run_in_reasoning` counts decoded characters; it is not a native-token repetition metric. `contains_parse_ledger_definition` is a text check, not code validation. Inspect the server's local log for the native token ledger and repeat-stop message. The client's timings are diagnostic observations, not a controlled performance qualification.

## Exact tested profile

- Qwen3.8-Flash-Next, Unsloth Dynamic v3.1 **UD-IQ2_XS**, separate **Q8_0 engram**, ordinary MTP with English draft vocabulary. Model/quantization/engram files were unchanged between arms.
- Two V100 SXM2 16 GiB GPUs with working NVLink/P2P; two E5-2695 v4 CPUs, 256 GiB RAM; Linux, NVIDIA 580.196.07. These describe the reproduction hardware, not required universal settings.
- Diagnostic limits: **150 W per GPU**, application clocks **1,312 MHz graphics / 877 MHz memory**, cooled to at most 65°C core and memory before each request. No hardware/power setting is changed by this client.
- Native engine 0.1.58 source [0fe4ae5](https://github.com/Niko1221/Strata/tree/0fe4ae5090bd7474995127eb5af95e0f26b9264b) versus 0.1.59 source [712fec6](https://github.com/Niko1221/Strata/tree/712fec6b4297c9cfd32ab43daaa2979a9aa01edf), official Linux CUDA 12 packages. Both reproduced the failure.
- The old serving frontend is based on [9780a8c](https://github.com/Niko1221/Strata/tree/9780a8c27d84f408240ed4b07062ab7275f02f09) plus the bounded thinking-carry patch [fe38f9f](https://github.com/midhatn/Strata/commit/fe38f9f12138640a32a572a55a64983cada335dd). The failure occurs before the thinking budget, so the carry continuation is not triggered. New Control 0.1.59.1 with the corresponding port also reproduced it.

Native options (replace only the asset placeholders with your installed assets):

```text
--serve --pack <matching-pack> --native <primary-GGUF>
--expert-profile <matching-expert-profile> --expert-cache auto
--prefill auto:16384 --spec 4 --spec-min-p 0.5 --mtp <matching-MTP>
--max-context 65536 --kv int8 --ple-io mmap --kv-resident 32768
--pool-workers 32 --aux-cpus auto --batch-groups auto
--vram-reserve-later-mib 1011 --adapt-every 4 --vision --layer-split auto
```

The serving config has one active request, `fit_max_tokens=false`, both GPUs selected, English draft vocabulary and CPU vision. No image is sent in this reproduction. Automatic prefill resolves to a 13,056-token chunk and 384-slot ring on this hardware; other hardware may resolve differently.

Explicit environment:

```text
STRATA_HC_Q8=1
STRATA_ROUTE_TAIL_SKIP=0
STRATA_SM70_TABLE=1
STRATA_FUSE_DECODE=1
STRATA_ARGMAX_ROWS_SM70=1
STRATA_PIPED_CONCURRENT=1
STRATA_NVML=0
STRATA_B6_MMVQ_ROWS=0
STRATA_PF_WMMA=1
STRATA_GDN_REC_WARP=2
```

The old frontend also supplies `STRATA_GDN_TC=1` and `STRATA_GDN_CONVL2=2`; those effective values were checked in the isolated new-native/old-frontend process. No experimental MTP batching or selector mode 2 is enabled. No PCIe share refinement was logged during these runs; rounded startup probes alone do not prove identical internal routing numerators.

All message content was checked against fixed synthetic task specifications and generated numeric/hash archive records. There are no user conversations, credentials, private paths, private network addresses, raw telemetry or device identifiers in the bundle. The request fixtures and client are provided under the MIT license in `LICENSE`.
