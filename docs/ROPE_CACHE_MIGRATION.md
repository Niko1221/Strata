# Experimental RoPE to YaRN 4x saved-cache migration

Converts a saved ordinary-RoPE text session into approximate YaRN 4x history,
including MTP draft keys. Retained values and recurrent state still reflect the
original attention computations; this is not equivalent to a fresh YaRN prefill.

Both source and target engines require `STRATA_ROPE_TABLE=1` and:

```text
--experimental-rope-yarn4-cache --kv fp16 --conversation-cache-mib 0
```

Use one GPU, with batching and vision off. Keep the same model, MTP and execution
settings. Start the source with `--rope-scaling none`, save its session through
`POST /slots/0?action=save`, then start the target with a larger `--max-context`
and `--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144`.

On the target, send `POST /slots/0?action=migrate_yarn4` with:

```json
{"filename":"ordinary.sess","source_context":262400}
```

`source_context` is the source engine's allocation capacity. The original file is
retained; `ordinary.sess.yarn4` is written and restored. Existing destinations,
repeated conversion and incompatible sessions are refused. Success returns
`approximate_migrated_history: true`; subsequent saves preserve that marker.

Continue with the full recorded conversation and a new message so the boundary
is recomputed under YaRN. Migration reads all model files for identity and needs
host RAM for the saved state and changed keys. FP16 KV only; SYCL does not expose
this operation. Native protocol: `MIGRATE_YARN4 <source_context> <path>`.
