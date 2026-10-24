# DFlash2 vs MTP on a 256K code prompt

This paired run measures speculative decoding after processing a long code prompt. It uses one sample per method; decode speed is indicative and has no statistical interval.

## Result

| Metric | MTP | DFlash2 Q8_0, block 7 | DFlash2 vs MTP |
| --- | ---: | ---: | ---: |
| Decode speed | 34.900 tok/s | 40.080 tok/s | **+14.84%** |
| Decode time for 96 tokens | 2.751 s | 2.395 s | 0.356 s saved |
| Full request time, prompt + decode | 1012.19 s | 1019.47 s | **0.72% slower** |
| Accepted / offered draft tokens | 50 / 88 (56.82%) | 53 / 62 (85.48%) | — |

The requested +32.23% decode-speed advantage would require 46.149 tok/s at the measured MTP rate. All 96 greedy output token IDs match between methods. The DFlash2 arm's prompt-plus-decode wall time was 7.280 seconds longer: its 0.356-second decode saving did not offset prompt processing time. The request wall measurement wraps the full request; it should not be interpreted as a prompt-only timer.

## Workload and machine

- Prompt: **261,747 input tokens**, generated with the Qwen tokenizer from 90 non-repeating source files. DFlash2/MTP runtime sources and speculative-decoding headers were prioritized. The exact token IDs are stored in `prompts.json`; SHA256 `90c337eefd41eda2767f8c8f72d76f47161495420659d187106de579bd7c98ce`.
- Context capacity: 262,144 tokens. Target KV: INT8 with 32,768 resident tokens. Completion: 96 greedy tokens, seed 1234.
- Hardware: NVIDIA GeForce RTX 4070 Ti SUPER 16 GiB, Ryzen 9 5900X, 125.7 GiB RAM, Linux, CUDA 13.4, SM89.
- Software/model: Strata engine 0.1.47; same target model, Q8_0 DFlash2 drafter where applicable, expert profile and executable for both arms. Engine SHA256 `42a56c9a217ddb47c58fa6f42ad331c77ae2c296a532d84c53b3610fedb4a3f2`.
- Shared settings: 2,700 actual expert slots, 11 pool workers, 480 MiB VRAM reserve, verifier capacity 8, prefill 512, short-read 64, no prompt cache, adaptive swaps, or PCIe sharing.
- Method settings: DFlash2 block 7 and `spec_min_p=0.7`; MTP max T=4 and `spec_min_p=0.5`. These established settings were held fixed; no launch-parameter sweep was performed.

Each method was run once because each full prompt took about 17 minutes to process. The DFlash2 run ended with 948 MiB free VRAM; MTP ended with 632 MiB. The two arms were run separately, so small environmental changes between runs are possible.

## Reproduction

Run `run.py` from the repository root with the same model artifacts, expert profile, and benchmark harness. It is the copied benchmark driver with a `--dflash-window` option added to allow the drafter's local window to cover the full input. The raw command records are `dflash2-command.json` and `mtp-command.json`; they include the environment variables and complete arguments, with the local executable path sanitized to `./build-dflash2/strata`. `dflash2.json` and `mtp.json` contain per-token timing and token-ID results. Logs are retained alongside them.

Common settings in both recorded commands include `--max-context 262144 --kv int8 --kv-resident 32768 --prefill 512 --short-read 64 --expert-cache 2700 --vram-reserve-mib 480 --spec 8`.

## Failed setup attempts

The first DFlash2 startup used 2,900 expert slots and failed the benchmark harness's 512 MiB headroom precheck with 415 MiB free; it did not issue a request. After reducing both arms to 2,700 slots, DFlash2's default drafter window (32,768) rejected the 261,747-token prompt before decoding. The measured run used `--dflash-window 262144`; the drafter's local ring remains bounded (the engine log reports 2,060 cells per draft layer). These setup failures are retained under `failed-attempts/` and are not included in the timing results.
