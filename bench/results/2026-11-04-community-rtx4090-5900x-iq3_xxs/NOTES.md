# Run notes

- The server was already running when the runner started, launched from a launch receipt
  (`serve.server --engine strata --config <receipt>`). The runner therefore reports the engine
  options as not exposed; they are recorded here.
- Engine argv: `--max-context 131072 --kv int8 --kv-resident 32768 --draft dflash2
--draft-model <models-dir>/dflash2/qwen3.8-flash-next-dflash2.q4_0-selq4.pack
--draft-fallback none --draft-conf 0.5 --vram-reserve-mib 1408 --vision`
- Server capacity: one request at a time. No background GPU work during the run; no request was
  retried.
- The model files were hashed by the runner before the requests; they did not change during the run.
