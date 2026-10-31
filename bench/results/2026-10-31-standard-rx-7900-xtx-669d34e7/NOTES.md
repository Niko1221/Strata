# Operator notes for run 20261031T182724Z-669d34e7

`README.md`, `results.json`, `outputs.jsonl`, `config.json` and `provenance/` are written by `strata bench run` and are
unedited. This file adds what the runner cannot see.

- **GPU:** AMD Radeon RX 7900 XTX 24 GB (gfx1100) in a UGREEN eGPU dock with its own PSU, connected over
  **OCuLink PCIe 4.0 x4**. The runner reports it as "AMD Radeon" with no memory size.
- **Host:** UGREEN iDX6011 Pro NAS, Core Ultra 7 255H (6 P + 8 E + 2 LP-E), 64 GB LPDDR5X, model files on an NVMe RAID 1
  (btrfs). Host OS UGOS Pro (Debian 12 base), kernel 6.18.15. "Ubuntu 24.04.5 LTS" in the README is the OS of the
  container the engine runs in (`ROCm` from the release package).
- **Strata:** release 0.2.8 packages, engine `strata-engine-0.1.54-linux-x86_64-hip-rx7900` and control 0.1.54, unpacked
  from the GitHub release and checked against `SHA256SUMS.txt`. No source build.
- **Model name `b028`** is only the label of this run config in my proxy. The model is the original Flash-Next
  IQ3_XXS (`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000{1,2}-of-00002.gguf`, hashes in `results.json`) with the native
  pack `iq3_xxs`, the control package's `expert-profile.bin` and the MTP head `rt` (paths in `engine-log.txt`). No
  steering vector.
- **Run config** (track `custom`, the one I use every day): `--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5
  --mtp rt --max-context 262144 --kv int8 --vram-reserve-mib 832 --pcie-frac 0 --kv-resident 32768`, a conversation cache
  of 4 slots / 9,216 MiB, `reasoning_effort` from the protocol, default pool workers (0.2.x plans 10 on this CPU),
  `STRATA_HIPBLASLT_TUNING` set to the shipped `gfx1100-hipblaslt-100200.txt`. Engine start-up lines: `engine-log.txt`.
- **Fresh engine:** the engine was started for this run and had answered one 4-token request (the proxy's load check)
  before `strata bench run` began, as the README says.
- **Background:** for this run I stopped every service on the NAS that was not needed (photo library, media server, an
  agent, a search engine, a mail bridge, file sync). Still running, all idle: a reverse tunnel, three small MCP servers
  and a container agent. The proxy in front of the engine forwards only.
- **Not measured:** card power during this run, PCIe link state, GPU clocks.
