# Measurements of 2026-10-21 (evening): Strata 0.2.2 on this gfx1100

Prebuilt engine 0.1.47 `hip-rx7900` with control 0.1.47 (SHA-256 checked against the release's `SHA256SUMS.txt`), compared with the 0.2.1 packages (engine 0.1.46, control 0.1.46.2) that this machine served with since the morning. Both use the same run config, so only engine and control are swapped: IQ3_XXS native pack, MTP `rt`, `--max-context 262144 --kv int8 --spec 4 --vram-reserve-mib 832`, default pool workers, the gfx1100 hipBLASLt table via `STRATA_HIPBLASLT_TUNING`. The Docker image `ghcr.io/niko1221/strata:0.2.2-rocm-gfx110x` is included as well. Script: `messen/t022.py` in the linked hardware repo.

| File | What |
|---|---|
| `version-pairs.jsonl` | 3 interleaved pairs, 0.2.2 (`b022`) against 0.2.1 (`p021`), a fresh engine per run, warm-up, then 2 measured answers (63-token prompt, 900 tokens out, temperature 0) |
| `fresh-engines.jsonl` | 8 fresh 0.2.2 engines with MTP, two 4K prompts each |
| `strata-bench/`, `needles-32k-128k.json` | `tools/strata_bench.py --long` and `tools/needle_bench.py` on 0.2.2 |
| `standard-v1/package-0.2.2/` | `strata bench run --suite standard-v1 --track custom` against that server: **complete**, the control archive has a `COMMIT` file now |
| `standard-v1/docker-image-0.2.2-default/` | the unchanged Docker image (`MODEL=IQ3_XXS`, `API_KEY` only), `--track default`: complete |
| `startup-logs/package-0.2.2.txt` | engine log lines of the package runs |
| `startup-logs/docker-image-0.2.2.txt`, `docker-run-config-0.2.2.json` | engine lines of the container start, and the run config setup wrote |

## Results

| median tok/s | 0.2.2 | 0.2.1 |
|---|---:|---:|
| pairs: decode (range) | 62.1 (61.1-63.5) | 60.5 (59.8-62.3) |
| pairs: 63-token prompt | 93.6 | 93.9 |
| `strata_bench.py --long`: chat / agent decode | 66.0 / 72.5 | 65.9 / 74.6 |
| 4K / 16K / 32K prompt | 736 / 1,211 / 1,222 | 742 / 1,204 / 1,210 |
| agent cache hit | 68.8 % | 68.6 % |
| `standard-v1`, package: short decode / 4K / 16K prompt | 70.6 / 746.1 / 1,220.7 (complete) | 69.0 / 743.9 / 1,236.7 (nonconforming) |
| `standard-v1`, Docker image defaults: short decode / 4K / 16K prompt | 62.0 / 667.1 / 780.8 | 58.9 / 605.4 / 772.0 |

- Same plan as 0.2.1: 10 workers on CPUs 1-10, expert cache 8,832 slots, 728 MiB free with everything loaded.
- Decode is ahead in 3 of 3 pairs, but by 0.2 to 2.1 tok/s, so I count it as equal. Needles 6/6 at 32K and 128K. 8 of 8 fresh engines read the 4K prompt at normal speed (659-701 tok/s).
- The Docker image setup chose the same as with 0.2.1 (`--max-context 131072 --vram-reserve-mib 700 --pool-workers 10`, no hipBLASLt table), and the engine still warns `only 459 MiB of VRAM free at the start of this request (below 512 MiB)`. It names the card `AMD Radeon RX 7900 XTX (gfx1100)`, the bench report still says `AMD Radeon`.
