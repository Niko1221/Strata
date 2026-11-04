# Run notes

- Model: GSQ-RCO Coder IQ1_M, the **2026-10-17 build** of ISTA-DASLab's release (first shard
  `bfa3644d…`). The earlier build (`e11083ba…`) is superseded. The new shard is not drop-in: the
  Strata pack must be rebuilt from it. Replacing the shard under an existing pack produced empty
  answers (0/45, no tool calls) until the pack was rebuilt.
- Launched with `strata launch --model-path <dir>`; the pack was rebuilt on that start. Server
  already running when the runner attached, so the runner reports the engine options as not exposed.
- Engine argv: `--max-context 131072 --kv int8 --kv-resident 32768 --mtp <models>/mtp/rt
--spec 4 --spec-min-p 0.5 --vram-reserve-mib 700 --batch 3 --vision`
- DFlash2 is refused for this model by the drafting layer ("DFlash2 is for Qwen3.8-Flash-Next only");
  MTP is the supported drafter here.
- One request at a time from the client; the engine has three batch slots. No background GPU work;
  no request retried.
