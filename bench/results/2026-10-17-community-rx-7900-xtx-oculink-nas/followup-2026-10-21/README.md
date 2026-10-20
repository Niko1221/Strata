# Measurements of 2026-10-21: Strata 0.2.1 on this gfx1100

Prebuilt engine 0.1.46 `hip-rx7900` with control 0.1.46.2 (SHA-256 checked against the release's `SHA256SUMS.txt`), and the Docker image `ghcr.io/niko1221/strata:0.2.1-rocm-gfx110x`. Same machine and model files as the main report. Script: `messen/t021.py` in the linked hardware repo.

| File | What |
|---|---|
| `startup-logs/package-default-workers.txt` | engine log lines without `--pool-workers`: 10 workers on CPUs 1-10, the LP-E cores 14/15 left free (#3262) |
| `startup-logs/package-pool-workers-9.txt` | the same with `--pool-workers 9`: CPUs 1-9 (0.2.0 used `1,2,3,4,5,14,15,6,7`) |
| `worker-pairs.jsonl` | 3 interleaved pairs, default (10) against 9 workers, a fresh engine per run, warm-up, then 2 measured answers (63-token prompt, 900 tokens out) |
| `fresh-engines.jsonl` | 8 fresh engines with MTP, two 4K prompts each |
| `strata-bench/`, `needles-32k-128k.json` | `tools/strata_bench.py --long` and `tools/needle_bench.py` on the engine package with the run config of the main report |
| `standard-v1/package-custom-config/` | `strata bench run --suite standard-v1` against that server; nonconforming only because the release control archive has no `COMMIT` file |
| `standard-v1/docker-parallel1/` | the Docker image with `PARALLEL=1` added |
| `standard-v1/docker-parallel1-hipblaslt-table/` | the Docker image with `PARALLEL=1` and `STRATA_HIPBLASLT_TABLE=on` |
| `startup-logs/docker-image.txt`, `docker-run-config-after-third-start.json` | engine lines of the three container starts, and the run config setup wrote |

The complete default-track run of the unchanged Docker image is its own result folder: [`../../2026-10-21-standard-rx-7900-xtx-2c7f637a/`](../../2026-10-21-standard-rx-7900-xtx-2c7f637a/README.md).

## Results

| `strata bench run --suite standard-v1`, medians of 3, tok/s | short prompt / decode | 4K prompt / decode | 16K prompt / decode | sanity |
|---|---:|---:|---:|---:|
| Docker image, defaults (`--track default`, complete) | 43.3 / 58.9 | 605.4 / 55.9 | 772.0 / 55.0 | 12/12 |
| + `PARALLEL=1` | 57.8 / 63.7 | 623.9 / 59.3 | 773.8 / 58.7 | 12/12 |
| + `PARALLEL=1 STRATA_HIPBLASLT_TABLE=on` | 60.4 / 65.8 | 768.3 / 62.4 | 1,238.5 / 69.3 | 12/12 |
| engine package, run config of the main report | 51.7 / 69.0 | 743.9 / 63.5 | 1,236.7 / 62.5 | 12/12 |

Docker setup's own choices: `--max-context 131072 --kv int8 --vision` (GPU reader) `--vram-reserve-mib 700 --pool-workers 10`, 3 slots, expert cache 5,808 slots (9,294 with `PARALLEL=1`), no hipBLASLt table. The shipped gfx1100 table is worth +23 % at 4K and +60 % at 16K here; the identity gate was not run.

| Worker pairs (`worker-pairs.jsonl`) | decode tok/s, median (range) | 63-token prompt tok/s |
|---|---:|---:|
| 0.2.1 default, 10 workers | 61.1 (58.6-62.3) | 97.6 (91.8-99.3) |
| 0.2.1 `--pool-workers 9` | 61.5 (59.9-64.3) | 88.0 (86.2-90.3) |
| 0.1.44.1 `--pool-workers 9` (2026-10-19) | 63.3 | 66.1 |

`strata_bench.py --long`: chat decode 65.9 tok/s, prompt 742 / 1,204 / 1,210 tok/s at 4K / 16K / 32K, agent cache hit 68.6 %. Needles 6/6 at 32K and 128K. Fresh engines: 8 of 8 at 665-699 tok/s. Since 2026-10-21 the NAS serves with 0.2.1 (default worker count, hipBLASLt table on).

Fields as in `../followup-2026-10-19/README.md`. The standard results' `outputs.jsonl` hold the model's full answers to the frozen requests, as written by `strata bench run`.
