# RX 6800 XT 16 GB on Windows: Strata 32K baseline vs DFlash2 + prefill cap

I tested my usual 32K Work preset against the same preset with DFlash2 and a smaller prefill cap. On the 16K prompt, the tuned run reached the first token 2.56 seconds sooner and finished 3.81 seconds sooner. Both configurations hit the same 512-token cap on every 4K and 16K run.

## Hardware and software

- AMD Radeon RX 6800 XT (RDNA2, 16 GB), Ryzen 7 5800X3D, 64 GiB installed RAM, Samsung 980 PRO NVMe SSD.
- Windows build 26300; GPU driver `32.0.21045.29003`; Strata Desktop 0.2.20, HIP release engine 0.1.66. Engine and benchmark runner commit: `2e87c08db18aa4577c22fd623d6ba53252719d63`.
- Tested November 23, 2026 US Central (November 24 UTC). PCIe link, GPU power limit, background load and peak memory were not recorded for this comparison.

## Model and settings

Qwen3.8-Flash-Next GSQ-RCO IQ2_XS split GGUF. See `model-files.csv` for the exact filenames and SHA-256 values. Both runs used Work mode, 32,768-token context, one slot, int8 KV, Auto expert cache with the same expert profile, 7 pool workers, 1,536 MiB VRAM reserve, and an 8 GiB system RAM reserve.

| Setting | Baseline | Tuned |
|---|---|---|
| Drafting | Off | DFlash2 Q4_0, no fallback |
| Draft settings | Inactive | `--spec 6 --draft-conf 0.35` |
| Prefill | `auto` | `auto:2048` (maximum chunk size) |

“Baseline” is my original daily preset, not a claim about factory defaults. The prefill setting is a cap; actual chunk sizes were not measured.

## Method

Native `standard-v1`, protocol `strata-qwen38-thinking-v1-draft.1`, custom track. Load each preset in Desktop and run `strata bench run --profile <active-native-profile.json> --no-start --suite standard-v1 --track custom --quantization IQ2_XS --out <output-dir>`. The suite uses fixed prompts and thinking enabled (xhigh, temperature 1.0, top-p 0.95, top-k 20; other captured sampling fields were min-p 0, presence penalty 0 and repetition penalty 1.0). Corpus SHA-256: `e75eff8b9feea04f745520b392c4c17d97aefc1ba185e222aa6740efa8f68407`.

One suite per preset, baseline first. Same three seeds and 512-token cap per prompt length. The model was reloaded before each suite, each suite ran its own warm-up, and the measured prompts had zero cached prefix tokens. Load time is excluded. Prompt and decode rates are the engine's reported metrics; latency is client-observed. First-token timing can include reasoning, not just visible answer text.

## Results

Medians over three requests per prompt length. Each cell shows baseline → tuned.

| Input tokens | Prompt tok/s | Decode tok/s | First token (s) | Total request (s) |
|---:|---:|---:|---:|---:|
| 77 | 57.42 → 52.99 | 46.07 → 57.49 (+25%) | 1.375 → 1.484 | 7.250 → 5.906 |
| 4,093 | 681.20 → 667.80 | 45.45 → 57.63 (+27%) | 6.031 → 6.156 | 17.250 → 15.016 |
| 16,387 | 847.35 → 977.32 | 45.66 → 51.35 (+12%) | 19.375 → 16.813 | 30.563 → 26.750 |

Required suite wall time: **266.703 → 236.531 s** (30.172 s less, 11.3%). Both passed **12/12** built-in sanity cases. The 77-token runs generated different amounts of output, so their end-to-end times aren't an equal-work comparison. Both tuning changes were applied together here, and run order wasn't reversed. These results describe one system and are not a general quality or stability claim.

### Files

- `requests.csv`: native performance and warm-up metrics. Counts are tokens; columns ending in `_s` are seconds and `_tps` are tokens/s. Output count is the API completion count; decode count and rate are the runner's engine metrics. A blank means the runner didn't report the metric.
- `sanity.csv`: case IDs and outcomes from the native checks, without response text.
- `model-files.csv`: model and helper artifact basenames and SHA-256 values, not the files themselves.

Exports include measurements only: no raw outputs, prompts, credentials, profiles, or machine-local paths.
