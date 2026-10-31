# Operator notes for run 20261031T192326Z-93cf5301

`README.md`, `results.json`, `outputs.jsonl`, `config.json` and `provenance/` are written by `strata bench run` and are
unedited. This file adds what the runner cannot see.

- **What ran:** the unchanged image `ghcr.io/niko1221/strata:0.2.8-rocm-gfx110x`, started with `MODEL=IQ3_XXS` and
  `API_KEY` only, `--track default`. Setup chose `--max-context 131072 --kv int8 --vision --vram-reserve-mib 700 --pool-workers 10`
  (start-up lines in `engine-log.txt`).
- **GPU and host:** the same machine as run `669d34e7`: RX 7900 XTX 24 GB (gfx1100) over OCuLink PCIe 4.0 x4 in an eGPU
  dock, UGREEN iDX6011 Pro, Core Ultra 7 255H, 64 GB LPDDR5X, UGOS Pro (Debian 12 base), kernel 6.18.15. The Ubuntu
  version in the README is the container's.
- **Fresh engine:** the container was started for this run; `strata bench run` was the first client.
- **Background:** the same as run `669d34e7`: the unneeded NAS services were stopped; a reverse tunnel, three small MCP
  servers and a container agent were running idle. No other model was loaded.
- **VRAM:** `854 MiB of VRAM free with everything loaded`, and no "only N MiB of VRAM free" warning during the run. With
  the 0.2.3 image on this card the engine had 594 MiB free after loading and warned once (my note in #3263).
