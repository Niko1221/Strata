# Metal: DFlash2 on the native-only build against MTP, Apple M4 Max (2026-10-22)

Measured on 2026-10-22 by hotocoo. The build of this PR at ee5e11b9 (main 9de04cd4 merged; `-DSTRATA_DFLASH=ON`, no
llama.cpp): the native Metal drafter with the gemv tiles of ee5e11b9. Both arms run the same binary. Greedy.

## Hardware and software

- Apple M4 Max (40-core GPU), 64 GB unified memory, internal SSD. macOS 26.5.
- `cmake -S metal -G Ninja -DCMAKE_BUILD_TYPE=Release -DMETAL_CPP_DIR=… -DSTRATA_GGML_DIR=<llama.cpp 3cf03257>
  -DSTRATA_DFLASH=ON`; `otool -L strata` lists no `libllama`.

## Model and configuration

- Target, MTP layer and draft vocabulary: as in [2026-10-20-metal-dflash2](../2026-10-20-metal-dflash2/README.md)
  (DavidAU's abliterated IQ3_S GSQ-RCO GGUF; the MTP block of the official checkpoint packed as `mtp-q2_0.gguf`;
  `data/draft_vocab_en.bin`).
- Drafter: `DFlash2-v2-Q8_0.gguf` (sha256 5afa38b9598e42d091491a96f443f276fb3fa03b0b14925deeb72ec5137a6a2b), the
  incoai drafter fine-tuned on this target's own features (checkpoint v2-9000), Q8_0, made with
  `tools/dflash_head.py --draft-vocab`. Published in the same Hugging Face repository as the 2026-10-20 drafter.
- Both arms: context 262144, `--mmap-experts`, `STRATA_METAL_STREAM=1`, `STRATA_METAL_LEAVE_SHARE=0.3`,
  `STRATA_METAL_ARENA_GB=42`, prompt cache 12, prefill auto, `--suffix-draft 0`, vision on, temperature 0.
  MTP: `--spec 3 --mtp <mtp rt>`, `spec_min_p` 0.7. DFlash2: `--spec 4 --dflash DFlash2-v2-Q8_0.gguf`, or `--spec 3`.

## Method

`scripts/run4.sh` waits until no other engine, benchmark or training job has run on the Mac for a minute, then
runs `scripts/matrix4.py` and the greedy check. `matrix4.py`: as 2026-10-20's `matrix.py` (three short prompts of
600 output tokens; a ~29K-token code prompt, cold, and a follow-up on the same prefix), each config on a fresh
server (`scripts/swap4.sh`, port 8092), configs alternated MTP / DFlash2 and the leaders repeated (`-r`). A row is
marked `CONTAMINATED` if another Strata engine ran at any request of it; none was. `scripts/run.py`: six fixed
prompts, 700 output tokens, temperature 0, compared field by field. Local paths in the scripts are replaced by
`<strata>` and `<data>`.

## Results

`matrix.tsv`, all rows. Decode tok/s is generated tokens over generation time (prompt reading excluded).

| Configuration | Runs | Short decode tok/s | Tokens/window | 29K cold tok/s | 29K cached tok/s |
| --- | ---: | --- | ---: | --- | --- |
| MTP spec 3, min-p 0.7 | 2 | 46.9, 46.8 | 1.71 | 50.2, 49.5 | 50.8, 46.8 |
| DFlash2 v2 spec 4, min-p 0.75 | 2 | 50.8, 50.4 | 2.12 | 52.9, 53.3 | 51.7, 51.4 |
| DFlash2 v2 spec 4, min-p 0.6 | 2 | 50.3, 50.0 | 2.37 | 52.5, 43.0 | 50.5, 50.6 |
| DFlash2 v2 spec 3, min-p 0.6 | 1 | 48.8 | 2.09 | 50.5 | 50.8 |

- Short prompts: DFlash2 v2 at spec 4, min-p 0.75 decodes 50.4-50.8 tok/s against MTP's 46.8-46.9, **+7.5% to +8.5%**
  (two runs each, same binary, same day). On 2026-10-20 (the earlier drafter, before ee5e11b9) the gap was 1.6-2.8 tok/s.
- 29K cold: 52.9-53.3 against 49.5-50.2 (+5% to +8%).
- The 43.0 cell (`-r`, min-p 0.6, 29K cold) is one run 9.5 tok/s below its twin; macOS swap grew 1 GB during that row,
  so it is kept and not explained further.

## Correctness

- `run.py`, greedy, 700 output tokens, six prompts: MTP and DFlash2 v2 give identical `reasoning_content` and
  `content`, character for character, on **6 of 6** (`greedy-mtp.json`, `greedy-dflash2-v2.json`).

## Limits

- One Mac, one target, two runs per leading cell. Differences under about 2 tok/s on short prompts are within the
  spread between repeats.
- Swap grew 0.5-2.3 GB during each config (64 GB with a 42 GB expert arena); the MTP and DFlash2 arms ran under the
  same conditions.
- Not measured: sampling (temperature > 0), other Macs or quantizations, concurrent requests.
