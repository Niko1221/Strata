# Metal: DFlash2 drafter (--dflash) against the MTP layer, Apple M4 Max

Measured on 2026-10-20 by hotocoo. One Mac, one target model (an abliterated IQ3_S fine-tune of Qwen3.8-Flash-Next),
greedy decoding. Main limitation: most cells are one run each; the leaders were repeated once (`-r` rows).

## Hardware and software

- Apple M4 Max, 64 GB unified memory, internal SSD. macOS 26.5.
- Strata source build: main e88ee39d plus this change, Metal engine (`cmake -S metal -G Ninja -DCMAKE_BUILD_TYPE=Release
  -DMETAL_CPP_DIR=… -DSTRATA_GGML_DIR=<llama.cpp 3cf03257>`), with `-DSTRATA_DFLASH_LLAMA=<llama.cpp 3cf03257 Metal
  shared build>` for the DFlash2 build.
- MTP arms: `build-metal/strata` without this change (the binary holds no `dflash` symbol or string).
  DFlash2 arms: `build-metal-dflash/strata`, built from the submitted source.

## Model and configuration

- Target: `DavidAU/Qwen3.8-Flash-Next-Abliterated-Uncensored-Turbo-Brilliance-50x-Reasoning-Instruct-GSQ-RCO-GGUF`,
  revision b7f9593a, `Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-0000{1,2}-of-00002.gguf`, mmproj BF16 (vision on).
- MTP layer (the baseline): not part of the target. DavidAU's GGUF has no MTP tensors (0 of 1224). It is the `mtp.*`
  block (31 tensors) of the official `Qwen/Qwen3.8-Flash-Next` BF16 checkpoint, revision de4b8e4d, extracted by Strata's
  setup and packed by `tools/mtp_pack.py` as `mtp-q2_0.gguf` (experts Q2_0, the rest BF16/F32;
  sha256 a22206a4d74eafbf31063a84610292535fb3b49af9a339963817fd04aed81e4d). It was trained on the official model, not on
  this abliterated fine-tune, and its experts are Q2_0 while the DFlash2 drafter is Q8_0: both favor DFlash2. The
  comparison is what Strata runs for this GGUF today against this drafter, not two drafters made the same way.
- Draft vocabulary (both arms): Strata's `data/draft_vocab_en.bin` (40525 token ids, English/code subset, sha256
  369151522226a5edaa5f12cfd1e2ae7db8f4fbdbd222f3dcf327dced9597fb25). A list of token ids only; tokens outside it are never
  drafted.
- Drafters:
  - stock: `incoai/Qwen3.8-Flash-Next-DFlash2` converted with llama.cpp's `convert_hf_to_gguf.py`, then
    `tools/dflash_head.py --draft-vocab` (Q8_0, the target's own output-head rows for the draft-vocabulary token ids).
    `dflash2-q8-dv.gguf`, sha256 32cd18db68465301d3281dd9a9e5a9e50f97ee070e5fd40683b1e5e822797312.
  - tuned: the stock drafter fine-tuned for 3000 steps (batch 32) on this target's own captured features (layers
    5/15/25/35/45, HC mean) and greedy continuations. `t3000-q8-dv.gguf`,
    sha256 e5412a73583755eea8daa4c4c5793f5b827af1fbc0e76bcdd5cc181cdbf46eb1. Published as
    `hotocooo/Qwen3.8-Flash-Next-Abliterated-Uncensored-Turbo-Brilliance-50x-Reasoning-Instruct-GSQ-RCO-DFlash`.
    Held-out drafts accepted per 7-token block: 3.568 (stock) to 3.796 (tuned).
- Both arms: context 262144, `--mmap-experts`, `STRATA_METAL_STREAM=1`, `STRATA_METAL_LEAVE_SHARE=0.3`, prompt
  cache 12, prefill auto, `--suffix-draft 0` unless the name ends in `-sfx`, no reasoning limit, temperature 0.
- `STRATA_METAL_ARENA_GB` (the expert arena) is the `aNN` in each name; `sN` is `--spec`.

```text
MTP:     strata --pack <pack> --native <IQ3_S 1 of 2> --ple-gguf <IQ3_S 2 of 2> --expert-profile data/expert-profile.bin
         --expert-cache 64 --prefill auto --spec 3 --suffix-draft 0 --max-context 262144 --mmap-experts --vision
         --prompt-cache 12 --prompt-cache-spread --mtp <mtp rt> --spec-min-p 0.7
DFlash2: the same, with --spec 4 --dflash t3000-q8-dv.gguf --spec-min-p 0.75 in place of --mtp, and STRATA_DFLASH_NATIVE=1
```

## Method

Scripts in `scripts/` (as run; local paths replaced by `<strata>` and `<data>`). `swap.sh` stops the server and starts
the next config on a fresh process, so each config starts with a cold expert cache.

- `matrix*.py`: for each config and each `spec_min_p` (sent per request as `strata_tune`): three short prompts
  (600 output tokens each), then a ~29K-token code prompt (first 90,000 characters of
  `src/core/mtp.cpp`) and one follow-up on the same prefix (prompt cache). Decode tok/s is generated tokens over the
  engine's generation time from its `generated in` line (prompt reading excluded). `tok/window` is tokens per verify
  window on the short prompts. The `swap` column is macOS swap used before and after the config.
- `long120.py`: a 101,872-token prompt (`verify.cpp` + `mtp.cpp`), then a follow-up reusing 101,867 tokens. One run
  per arm; engine lines in `long-102k-engine-lines.txt`.
- `run.py`: six fixed prompts, greedy, 700 output tokens: the outputs of MTP and DFlash2 are compared as text.

## Results

The full matrix is `matrix.tsv` (every row, including the paging runs). Best settings at the 42 GB arena:

| Configuration | Runs | Short decode tok/s | Tokens/window | 29K cold tok/s | 29K cached tok/s |
| --- | ---: | --- | ---: | --- | --- |
| MTP spec 3, min-p 0.7 | 2 | 48.1, 46.9 | 1.71 | 50.4, 50.0 | 50.4, 50.1 |
| DFlash2 tuned spec 4, min-p 0.75 | 2 | 49.7, 49.7 | 2.11 | 51.0, 51.7 | 51.7, 51.8 |
| DFlash2 tuned spec 4, min-p 0.6 | 2 | 50.4, 49.6 | 2.35 | 51.9, 50.8 | 51.4, 50.3 |

101,872-token prompt, one run each:

| Configuration | Prompt read, s (tok/s) | Decode cold tok/s | Cached follow-up decode tok/s |
| --- | --- | --- | --- |
| MTP spec 3, min-p 0.7 | 268.8 (379.1) | 46.8 | 44.4 |
| DFlash2 tuned spec 4, min-p 0.75 | 276.9 (367.9) | 50.5 | 45.1 |

Other observations in `matrix.tsv`:

- The **stock** incoai drafter (`dfbase-s4-a33`) is slower than MTP on short prompts (41.1-42.5 vs 42.1-43.7 tok/s at
  arena 33). The gain comes from the fine-tune on this target, not from DFlash2 alone.
- At arena 33 the two are within noise (MTP 42.1-43.7, tuned DFlash2 41.9-45.4 short).
- Arena 45 and 47 page on this 64 GB Mac: short decode falls to 8.8-41.1 tok/s for both drafters.
- Draft cost: the native Metal drafter takes 3.55 ms (1 token) to 9.73 ms (7 tokens) per draft
  (`dflash_drafter_test`, measured while another model server ran, so an upper bound).

## Correctness and limitations

- `run.py`, greedy: MTP (`greedy-mtp.json`) and DFlash2 tuned (`greedy-tuned.json`) are identical, reasoning and
  answer, character for character, on 6 of 6 prompts.
- `dflash_native_parity_test`: the native Metal drafter against an f64 host reference: passed.
- Not measured: sampling (temperature > 0) acceptance and quality, time to first token, peak memory beyond swap,
  other Macs, the official Qwen weights or other quantizations, concurrent requests.
- Few repetitions: differences under about 2 tok/s on short prompts are within the spread seen between repeats.
