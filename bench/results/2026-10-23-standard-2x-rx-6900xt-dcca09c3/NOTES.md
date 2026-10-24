# Notes from the submitter (not part of the runner's output)

Five more runs of the same protocol on this box (default track on one and two cards, custom track on one card and with the release package) are in the sibling folders `2026-10-23-standard-{1x,2x}-rx-6900xt-*`; `MATRIX.md` here puts the six side by side.

Draft-protocol pilot data, `strata-qwen38-thinking-v1-draft.1`, track **custom**. `README.md`, `results.json`, `outputs.jsonl`, `config.json` and `provenance/` are exactly what `strata bench run` wrote; nothing in them was edited (the artifact hashes in `results.json` match). This file adds what the runner could not know and one measurement that explains the decode column.

## How the run was made

- Box: 2x AMD Radeon RX 6900 XT 16 GB (gfx1030, one reference card and one Sapphire Nitro+, PCIe 4.0 x8 each), Ryzen 5 5600X, 128 GB DDR4-3200, Ubuntu 26.04, kernel 7.0.0-38, ROCm 10.0 (HIP 7.15). Both cards at a 245 W power cap and a -81 mV core offset (this box's production setting; it moves prompt speed by 2-3% and decode not at all, measured separately). No other GPU work during the run.
- Engine: a source build of Strata `34cd8534` = v0.2.2 merged into a local branch that carries five opt-in patches (a port of #2985 `STRATA_PIPED_MTP`, #3381 `STRATA_PROMPT_ATTN_RDNA2`, #3382 `STRATA_PREFILL_CPU_SHARE_BATCH`, #3073 `STRATA_SELECT_SIMT=2`, #3095 `STRATA_PW_WITH_BATCH`), built with setup's HIP options (`STRATA_NATIVE_EXPERTS`, `STRATA_PREFILL_MMQ`, `STRATA_Q6K_EXPERTS`, `STRATA_MMQ_KQUANTS=OFF`, `GGML_HIP_GRAPHS`, `GGML_HIP_NO_VMM`) and ggml from llama.cpp `fb5d45967`. Engine version string 0.1.47. This is why the track is `custom`, not `default`.
- Run config: this box's production config (`config.json`): Qwen3.8-Flash-Next GSQ-RCO IQ3_S native pack, 262144 context, `--kv int8 --kv-resident 32768`, layer split 0-23 / 24-47, `--spec 3 --spec-min-p 0.7 --pipeline-windows 2 --batch-mtp`, `"parallel": 2`, vision on, `"sampling": {"temperature": 0, "presence_penalty": 0}` as the server defaults (the protocol's requests send their own sampling, which overrides them). The server was started by hand on 127.0.0.1:8089 with `serve/server.py --engine strata --config config.json` from the frozen engine folder; the runner used `--url` (`--profile` is not set up on this box). The model had answered no request before the run.
- The run: `python strata.py bench run --url http://127.0.0.1:8089 --out <folder> --track custom --track-notes "..." --build-options "..." --optional-concurrency` from the git checkout `34cd8534` (so the commit is known). Required suite 2 min 48 s after loading, plus the model-file hashing before the clock started; the optional concurrency stage ran too (three rounds, 0 failed).

## What the runner could not read on this AMD box

`results.json` lists them under `missing`; the ones a reviewer will notice:

- GPU names come out as "AMD Radeon; AMD Radeon", `memory_bytes` and `power_limit_w` are null for both cards, drivers and runtime are "not measured": the hardware facts come from `nvidia-smi` and there is no ROCm equivalent in `bench_facts.py` yet. The cards are two RX 6900 XT with 16 GiB each, the amdgpu driver of kernel 7.0.0-38, ROCm 10.0 user space (the engine's `/props` does not say which HIP it was linked against).
- `model_revision` / `source_revision` are null: the quantization repository's revision is not known to the runner and I did not pass one (the files are the ones Strata's setup downloaded from `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`; their sha256 are in `results.json`).
- Background workloads: "not measured" by the runner; none ran.

## Why decode reads 62-66 tok/s here while greedy decoding on the same engine reads 76-84

The performance cases decode at 62.9 (4K) / 62.0 (16K) / 66.4 (short) tok/s with 6 of 9 requests reaching the 512-token cap while still reasoning. The same engine decodes 256-token greedy answers at 74-82 tok/s in our own A/B runs. To see which part of the protocol's shape makes the difference, I sent the protocol's `performance-prompt-4k-r1` request to the same engine (started the same way, `STRATA_DECODE_TIMING=1` added) in five shapes, 512 tokens each, fresh nonce per send, one after the other (`notes/e529-requests.py`, raw lines in `notes/e529-*.txt`):

| request shape | decode tok/s (engine) | ms per window | tokens per window | drafts accepted / offered |
|---|---|---|---|---|
| greedy (`temperature 0`), thinking off, non-streaming | 81.6, 83.9, 80.1 | 25.7-27.3 | 2.15-2.19 | 274/318, 276/334, 280/340 |
| greedy, thinking off, streaming (`stream_options.include_usage`) | 84.0 | 25.4 | 2.13 | 272/313 |
| greedy, thinking on | 88.4, 90.0 | 24.9-25.2 | 2.20-2.27 | 279/318, 286/339 |
| the protocol's sampling (temperature 1.0, top_p 0.95, top_k 20, reasoning_effort xhigh), thinking on, non-streaming | 73.4, 73.1, 64.3 | 25.2-26.3 | 1.63-1.92 | 237/287, 246/298, 199/251 |
| the same, streaming (the runner's shape) | 62.9, 61.1 | 23.6-24.0 | 1.47-1.48 | 169/210, 163/211 |

The time per verify window is the same in every row (24-27 ms; GPU-reach wait 16-17 ms, no draft or commit time outside the overlap). What changes is the number of tokens a window yields: under the card's sampling the draft layer's greedy guesses match the sampled token less often and the `--spec-min-p 0.7` gate offers fewer drafts per window, so a window yields 1.5-1.9 tokens instead of 2.1-2.3. The spread between repeats of the sampled request (64 vs 73 tok/s for the same prompt and seed) is the text it sampled, not the engine. Streaming itself costs nothing (greedy streaming 84.0). So the decode column of this report is the speculative decoder's yield under the model card's sampling, which is what the protocol means to measure; it is not comparable with greedy numbers from `benchmark.py` or `tools/strata_bench.py`, and that may be worth a sentence in `BENCHMARK_PROTOCOL.md` for readers who compare the two. Not tried here: `STRATA_SPEC_COUPLED=1` / `STRATA_SPEC_PROB=1` (docs/DETAILS.md reports no gain on an RTX 3060 at temperature 0.3-1.0).

Developed with an AI coding assistant; every number above was measured on the machine described here.
