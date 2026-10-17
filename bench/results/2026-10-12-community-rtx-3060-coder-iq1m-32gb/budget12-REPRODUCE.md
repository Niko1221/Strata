# Reproducing the Budget12 64K trial profile (read first)

This report shares a **path-sanitized copy of an actual tested configuration**, not a newly invented configuration and not a ready-to-launch command. The original live config was verified by SHA-256 before sanitization, then left untouched.

## Tested environment

- Upstream Strata **v0.1.43**, commit `7695859aa13afb248bdb4d36f9b2e95707db079d`; unmodified source and engine.
- Windows 11 build 26200, RTX 3060 12,288 MiB (NVIDIA 610.88), Intel i5-11400F, 31.8 GiB RAM.
- Model name `qwen3.8-flash-next-coder-iq1_m` (Coder IQ1_M). No model/GGUF, engine executable, or personal paths are redistributed.
- One GPU (`gpu: 0`), localhost (`127.0.0.1:8080`), total context `65536` (including input **and** output).
- INT8 KV; `--expert-cache auto`; `--prefill auto`; `--spec 4`; `--spec-min-p 0.5`; `--resident-budget-gib 12`.
- Observed 2,152 expert cache slots in the Budget12 run. This is observed, not hard-coded in the JSON.

## Configuration fidelity

The file `budget12-sanitized-config.json` is the exact, unmodified output of a read-only exporter operating on the real Budget12 JSON. Its source hash was checked against earlier experiment receipts before it was sanitized. The sanitized file's SHA-256 is `3716B75CF4B77D7ED5857595DE3B4A0293B5F3F536FC5411D8F3501CE75B9BDE`. The original private JSON SHA-256 is `9F5DB7BC7065A18E8870DC503B7F8F60D695C24FF50883FDCF9AE2A3FF6C55EC`. The verification receipt is `budget12-export-receipt.json`.

The JSON cannot run as distributed because its paths contain placeholders, for example `<LOCAL_PACK_DIR>` and `<LOCAL_MODEL_SHARD>`; those deliberately have **no** filesystem meaning. Its exact performance also depends on the model files, disk/host load and driver.

## Safe adaptation to another PC (not an automatically executed script)

1. Install upstream Strata **v0.1.43** and the **Coder IQ1_M** model through upstream's official setup workflow, with its required pack, MTP draft and expert-profile assets. Consult [Strata installation](https://github.com/Niko1221/Strata/blob/v0.1.43/docs/INSTALL.md). Do not overwrite an existing functioning configuration.
2. Make a **separate copy** of `budget12-sanitized-config.json` outside this report. Locate the actual engine executable, pack directory, native GGUF shard, PLE GGUF shard, expert profile file, MTP directory, tokenizer, libraries, repository folder and engine log in **your own** installation. Replace only the matching placeholders with your verified local paths.
3. Review any GPU selection and port conflicts; GPU `0` and localhost port `8080` are the observed host's values, not universal requirements. Review `--spec`/MTP compatibility and driver/architecture with your installed release. Check that your chosen setup can support 65,536 total context and INT8 KV within available VRAM.
4. Verify locally that the selected engine is unmodified v0.1.43 and that **the adapted config's argument array** contains exactly the demonstrated operational flags. Use upstream's documented `--config` option to point your Strata start workflow to the **separate** adapted config, after inspecting the file. Do not assume a later Strata release has identical behavior.
5. Monitor available host RAM and GPU VRAM, background applications and pagefile/commit capacity. Our **external evaluation runner** required two passing samples, 10 seconds apart, of at least **6,144 MiB available host RAM and 256 MiB free GPU VRAM** before dispatching inference, with approximately 10-second monitoring and emergency thresholds of 1,024 MiB RAM and 128 MiB VRAM. These safeguards are **not** built into the config or upstream Strata and do **not** guarantee prevention of all out-of-memory events.
6. For a HIGH reasoning request, the **per-request** parameter `reasoning_budget_tokens: 6144` alongside `max_tokens: 8192` successfully produced final JSON on our 64I and 64J single-run evaluations. It is **not** in the engine's global config. Keep total prompt + requested output below the full 65,536-token context, with engine overhead and margin.
7. Run your **own** new, safe evaluation before making workload claims. This report does not authorize rerunning our sealed one-shot tests or deploying this configuration to financial/operational production.

## Example request fields only (not a benchmark command)

```json
{
  "model": "strata",
  "messages": [{"role": "user", "content": "Your own short test prompt"}],
  "reasoning_effort": "high",
  "reasoning_budget_tokens": 6144,
  "temperature": 0,
  "max_tokens": 8192,
  "stream": false
}
```

**Important limitations:** A published path-redacted JSON supports configuration inspection and adaptation, **not** a bit-for-bit runnable setup on a different PC; no cross-machine replication is claimed. Recorded prefill/decode metrics vary by workload and background load; the 64J prefill slowdown remains unexplained. Manual repository grades and strict synthetic field grades must not be conflated. Independent deterministic validation is recommended before trusting exact financial arithmetic.
