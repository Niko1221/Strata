# Fleet calibration: faster candidate settings and a rejected RTX 4090 combination

Measured 2026-10-09 on six Linux hosts. This is a results-only report of `tools/calibrate.py`, including a PRO 6000 rebuild so all three adaptive-cache policies could run. It changes no engine code or defaults.

The tuner found faster individual stages on the P4s, RTX 3070, RX 5500 XT and RX 7900 XTX. A separate full-configuration check rejected the RTX 4090 candidate despite its apparent calibration wins. Treat the other selections as workload-specific candidates, not fleet-wide defaults. The original launch configurations were preserved.

## PCIe share and draft confidence

Three interleaved confirmation rounds, median decode tok/s across three prompts per round, answers capped at 128 generated tokens. These comparisons start from product defaults. In particular, the earlier manually tuned three-P4 setup already used PCIe share zero, so its percentage below is not an additional gain over that earlier report.

| Hardware/build | Default tok/s | Candidate tok/s | Change | Decision |
| --- | ---: | ---: | ---: | --- |
| 3x Tesla P4 | 26.819 | 29.356 | +9.46% | selected |
| RTX 3070 | 10.947 | 11.159 | +1.94% | below 3%; keep defaults |
| RX 5500 XT | 16.487 | 17.369 | +5.35% | selected |
| RX 7900 XTX | 59.101 | 62.009 | +4.92% | selected |
| RTX 4090 | 54.645 | 75.816 | +38.74% | rejected later |
| RTX PRO 6000 (old build) | 174.506 | 181.303 | +3.90% | selected |
| RTX PRO 6000 (0.1.41) | 231.884 | 240.647 | +3.78% | selected |

## Adaptive decode expert cache

Each cell is median [minimum, maximum] of **two** measured prompt-set medians; each prompt set contains three answers capped at 512 tokens. The PCIe/draft and worker choices from earlier stages are held for this stage. The candidates are `(adapt-every, adapt-swaps, adapt-decay)`. These gains must not be added to the preceding table, and neither table measures the new prefill layer cache.

| Hardware/build | Default | 1 / 80 / 0.97 | 1 / 160 / 0.97 | Tuner selection and stage gain |
| --- | ---: | ---: | ---: | --- |
| 3x Tesla P4 | 29.055 [28.870, 29.240] | 28.838 [28.626, 29.050] | 28.656 [28.637, 28.676] | default; +0.00% |
| RTX 3070 | 11.727 [11.656, 11.798] | 11.909 [11.739, 12.079] | 12.176 [12.176, 12.176] | 1/160/0.97; +3.83% |
| RX 5500 XT | 8.335 [8.054, 8.615] | 10.217 [9.482, 10.953] | 10.120 [9.474, 10.765] | 1/80/0.97; +22.59% |
| RX 7900 XTX | 59.724 [57.959, 61.489] | 69.053 [67.863, 70.244] | 69.223 [68.373, 70.073] | 1/160/0.97; +15.91% |
| RTX 4090 | 34.989 [34.506, 35.472] | 40.339 [35.930, 44.747] | 39.624 [35.878, 43.371] | 1/80/0.97; +15.29% (rejected later) |
| RTX PRO 6000 (old build) | 177.400 [175.404, 179.395] | did not start | did not start | default; +0.00% |
| RTX PRO 6000 (0.1.41) | 236.144 [235.891, 236.398] | 236.212 [236.039, 236.386] | 235.984 [235.817, 236.150] | default; +0.00% |

## Candidate settings

`auto` preserves the engine's PCIe probe. Worker counts shown include retained defaults; the host thread is additional. `default` preserves the engine's adaptive policy. The RTX 4090 row records the rejected candidate for reproducibility, not a recommendation.

| Hardware/build | --pcie-frac | --spec-min-p | --pool-workers | Adaptive policy |
| --- | ---: | ---: | ---: | --- |
| 3x Tesla P4 | 0.00 | 0.70 | 27 | default |
| RTX 3070 | auto | 0.5 | 7 | 1/160/0.97 |
| RX 5500 XT | 1.00 | 0.70 | 5 | 1/80/0.97 |
| RX 7900 XTX | 0.35 | 0.70 | 7 | 1/160/0.97 |
| RTX 4090 | 0.90 | 0.70 | 3 | 1/80/0.97 |
| RTX PRO 6000 (old build) | 0.90 | 0.70 | 15 | default |
| RTX PRO 6000 (0.1.41) | 0.75 | 0.70 | 15 | default |

The flags for a selected adaptive policy are, for example, `--adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97`. Use only the candidate for the matching model, context and hardware; JSON files contain all other flags and environment settings.

The upgraded PRO 6000 held all 24,576 routed experts (46.84 GiB) in VRAM. Its cache-policy medians were 236.144, 236.212 and 235.984 tok/s, so neither additional policy passed the 3% threshold. Its PCIe-share sweep was also effectively flat; the useful part of its selected pair is the draft-confidence change. A smoke request through the existing installed Python frontend succeeded after the engine replacement. The original executable and BUILD.json are backed up for rollback.

## Why the RTX 4090 candidate was rejected

The tuner selected PCIe share 0.90, draft confidence 0.70, three workers, and the 80-swap policy. A separate default / tuned / tuned / default check used a fresh engine and a three-prompt warmup per arm, followed by the same three prompts with a 512-token cap. The arm medians were **35.203, 23.862, 42.271, 37.131 tok/s**. Aggregated default versus tuned medians were **36.167 versus 33.067 tok/s (-8.57%)**. The candidate did not establish a reliable gain and was not applied.

Its mapped-expert timings vary substantially. Identical prose tokens also showed timing variation; the list response differed between configurations. The full token arrays and engine metrics are in `rtx4090.json`. This is a speed check, not a general answer-quality gate. The cause of the variability was not isolated. An earlier rough 30 GiB RAM description was not substantiated: an older inventory lists 60 GiB, and this run did not capture a contemporaneous total. We do not attribute the result to the model exceeding physical RAM.

## Hardware, models and builds

| Hardware/build | Quant | Configured context | CPU | RAM GiB |
| --- | --- | ---: | --- | ---: |
| 3x Tesla P4 | IQ3_XXS | 40960 | Intel(R) Xeon(R) CPU E5-2697 v3 @ 2.60GHz | 251.8 |
| RTX 3070 | IQ3_XXS | 16384 | AMD Ryzen 7 5800X 8-Core Processor | 62.7 |
| RX 5500 XT | IQ3_XXS | 16384 | AMD Ryzen 5 3600 6-Core Processor | 31.3 |
| RX 7900 XTX | IQ3_S | 262144 | AMD Ryzen 7 2700 Eight-Core Processor | 60.7 |
| RTX 4090 | IQ3_XXS | 16384 | Ryzen 5 7600X3D (earlier inventory) | 60 GiB in earlier inventory; not reverified for this run |
| RTX PRO 6000 (old build) | IQ3_S | 16384 | AMD Ryzen 9 7950X 16-Core Processor | 124.9 |
| RTX PRO 6000 (0.1.41) | IQ3_S | 16384 | AMD Ryzen 9 7950X 16-Core Processor | 124.9 |

- P4s: three 7,680 MiB cards, CUDA 12 / sm_61 corrected pipeline build from [#1674](https://github.com/Niko1221/Strata/pull/1674), source `06abdfa50aef6e5068869bc7552fafa5e99b7b9d`. GPU order 0,1,2, split 19,37, two pipeline windows; NUMA interleave; 64 GiB process memlock allowance. Two links x16 and one x8; 75 W limits.
- RTX 3070: 8 GiB, CUDA 13.2 / sm_86. RX 5500 XT: 8 GiB, HIP 5.7 / gfx1012. Both used source `76ce987ae5af9f08c4e6ce59edabb8099714a06d` with its experimental prefill layer cache **disabled**. The HIP build used a local header overlay adding missing `inline` qualifiers to six BF16 helpers; system headers were unchanged, as documented in [the prior cross-GPU report](https://github.com/CC-David-CC/Strata-a5500/blob/ec8eecdab4b6306d0cb8430fd1c99dbb49d3da33/docs/measurements/p4-layer-cache/cross-gpu/README.md).
- RTX PRO 6000: 96 GB class, CUDA 13.2 / sm_120. The earlier installed build was stamped 0.1.26 and rejected `--adapt-decay`. It was backed up and replaced with public main `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` (0.1.41), built with `STRATA_ENABLE_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=120`, Release, and pinned ggml `3cf03257f219afbe7334045ff7c6a06ac68c627d`. The full calibration was rerun on the installed replacement; the old run is retained separately.
- RX 7900 XTX: 24 GB, existing HIP build. RTX 4090: 24 GB, existing native-GGUF build. Exact source commits for these installed snapshots were not established; **binary SHA-256 values** are attached. Stale BUILD.json labels are not used to claim source identity.
- Quantizations are the existing Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS or IQ3_S packs and split GGUFs. Filenames follow `Qwen3.8-Flash-Next-GSQ-RCO-<QUANT>-00001-of-00002.gguf` / `00002-of-00002.gguf`. Model repository revisions were not recorded. Expert-profile hashes are included where collected. No image inputs or private conversation content were used.
- KV is int8. P4, 3070, 5500 XT and 4090 runs use spec 2; PRO 6000 and 7900 XTX use spec 4. The 7900 XTX has a 32,768-token resident KV window within its 262,144-token configured context. Other exact settings, including mmap versus resident experts, are in each JSON.
- Hosts were checked for competing GPU inference before launch. The upgraded PRO run waited until a separate remote-stage test completed. A single-P4 control was intentionally stopped after 28 requests to prioritize the actual three-card setup; it supplies no completed calibration claim. The 4090's first older IQ3_S engine attempt failed before measurement because its mapped mode required `experts.bin`; the reported successful run uses IQ3_XXS.

## Method, timing and limitations

The calibration source is identical to public main's `tools/calibrate.py` at `fb58e0d` (the initial harness was packaged from the report-independent branch at `ec8eecda`). The three prompts request a sorted-list merge function, two paragraphs about refrigeration, and twelve European capitals. Temperature is zero, with the script's no-thinking chat template. The existing model, draft data and expert profile are reused. Loading and prompt processing are excluded from decode tok/s; actual prompt/generated token counts, decode milliseconds, prefix reuse and cache-hit lines are attached in `*-timing-lines.txt`.

The first engine receives two warmup prompt sets. PCIe shares and draft floors are visited three times, reversing sweep order on alternate rounds; the chosen pair is confirmed against defaults with three interleaved rounds. Worker and adaptive candidates each restart the engine, warm up once, and measure twice. The threshold is more than 3%. This preserves the official script's two-repeat worker/cache policy rather than presenting it as a three-repeat benchmark. Expert caches adapt during the workload and the OS page cache is not flushed. The short prompts and repeated warmups are not a cold-start or long-context production workload.

The script optimizes stages sequentially; stage wins do not prove that the complete selected configuration is faster. Only the RTX 4090 received the separate full-configuration check in this report. Output equality and general answer quality are not calibration gates. No failed or cancelled request is included in successful throughput summaries. No universal default change is proposed.

## Reproduce and evidence

1. Use the matching engine build and recreate the local paths represented by `$ENGINE`, `$SOURCE`, `$PACK`, `$TOKENIZER`, `$MODEL_SHARD_1`, `$MODEL_SHARD_2`, `$EXPERT_PROFILE`, `$MTP` and, for the P4 split, `$SHARED_ARENA`.
2. Extract `configuration` from the hardware JSON into a local config file and substitute the local paths. Keep the recorded model, context, KV, draft, GPU layout and environment.
3. From a matching source checkout run `python tools/calibrate.py config.json` using the environment that can import `serve.server` and the tokenizer. This prints results without changing an installed config. Setup's `./setup.sh --calibrate` additionally saves settings and may update/start the engine.
4. To repeat the full-settings check, place the original and candidate configs as described by `confirm-calibration.py`, then run it. It writes token arrays and timing metrics for every measured request.

Each hardware JSON contains the complete calibration arrays, selected flags, timestamps, exact binary hash, sanitized configuration, and available inventory. `rtxpro6000-before.json` preserves the unsupported-cache failure; `rtxpro6000.json` is the replacement-build run. `confirm-calibration.py` is the actual independent-check harness, with local path setup generalized. Timing logs contain only the engine's measurement lines. No credentials, personal prompts, or private implementation sources are included.

Related history: [adaptive tier proposal #906](https://github.com/Niko1221/Strata/issues/906), [full PCIe sweep and repeated measurements #1332](https://github.com/Niko1221/Strata/issues/1332), [preserving earlier results after a later startup failure #1337](https://github.com/Niko1221/Strata/issues/1337), and [Linux HIP calibration #566](https://github.com/Niko1221/Strata/issues/566).
