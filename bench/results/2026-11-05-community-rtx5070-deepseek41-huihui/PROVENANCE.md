# Post-run model-file and configuration provenance

Collected by the operator at **2026-11-05 17:23:42 +01:00**, **after** the performance/recall benchmark. This is a read-only file inspection, **not** a live attestation of which GGUFs the inference process loaded during the earlier tests. Full original evidence: [`provenance.json`](provenance.json) (copied without modifications).

## Observations

- Strata checkout commit: `ffa80d73133ea5c548aefe57889a16e8085a938f`, matching the benchmark checkout commit. The post-run Git tree was **clean** with no changed source paths, **whereas the benchmark's earlier `ENVIRONMENT-original.md` recorded a modified tree**. Both observations are retained; this snapshot cannot reconstruct the earlier diff or prove the checked-out files were unchanged throughout.
- The expected Huihui model directory and expected Huihui configuration file were present. **Presence does not demonstrate that the measured running process used them**.
- The allowlisted expected configuration recorded local port `8070`, `--max-context 131072`, `--vision` and `--reserve-mib 900`. The server independently reported a 131,072-token context, but active `--reserve-mib 900` and other effective flags cannot be verified from these captured metrics alone.
- Four GGUF shards present: **138,238,893,056 bytes (128.75 GiB)** combined. Their sizes agree with the four IQ2_XXS shards specified by the operator's earlier launcher. SHA-256 strings below were reported by the local collector; this report validates syntax and cross-checks against the launcher-declared *original* ISTA base hashes, but has not re-hashed the 138 GB of GGUF bytes itself.

## SHA-256 comparison with launcher-declared original ISTA base

The operator's earlier launcher defines four base ISTA shard hashes, and is designed to apply the Huihui patch to the original model. A hash that differs from the base establishes that the files are not byte-identical; **a checksum alone does not identify the changed tensors, prove a particular patch revision, or identify the active server's loaded weights**.

| Shard | Size (GiB) | Reported SHA-256 | Matches original ISTA base hash? |
| --- | ---: | --- | --- |
| 1/4 | 36.39 | `d46efb67ccffd8fb5d7f18b423896381b9e06d8491e161c370ddf3b05b07e050` | Yes |
| 2/4 | 36.68 | `cec1b2c3da429dc768b8bb9bc4dc4403d81f74d34db660a6a34ee3e555bf2c70` | Yes |
| 3/4 | 36.55 | `9d9c2140fcb134a0b9a74dedd0e5167e68077af6d08ce7f856bce112f0a35dd3` | Yes |
| 4/4 | 19.12 | `9bb56083b354363bfbc006918681e0efca6e438fd5955d53990eee5f81e33312` | No |

The **first three hashes** match those in the operator's earlier *original ISTA* launcher inventory. The **fourth hash differs** from that base (`9b790a2297ed3cdede379fc984f4a52bb95f3b23724c797af11d1fa6a69db059`), which is consistent with a patched shard but **does not independently verify the Huihui 40-tensor patch or its exact origin**. In particular, it would be wrong to publish the base fourth-shard hash as the final modified GGUF hash.

## Launcher-declared upstream sources (not independently proven by the server capture)

- Base ISTA-DASLab repo: `ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF`, launcher-pinned revision `cb02422c900df489cea5833e8a92171224059b3c`.
- Huihui repo: `huihui-ai/Huihui-DeepSeek-V4.1-Flash-1017-abliterated-GGUF`, launcher-pinned revision `c59352db18acc847d9db6763df992d41b1d7d419`.
- Launcher-described model: original IQ2_XXS 2.00 bpw GGUF shards with a Huihui patch replacing 40 `attn_output_b` tensors; this description has **not** been verified against the running process.

## Remaining uncertainty

To claim exact reproducibility of the original benchmark, evidence would still be needed of the **active model/pack/executable/configuration and command line during the measured run**, the **Git changes present at that time**, and relevant engine flags or environment overrides not surfaced by the collector. There was no engine restart between the two sweeps, and no continuous background-load or power/thermal trace. The post-run clean working tree does **not** retroactively clear the benchmark-time warning.
