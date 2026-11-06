# Same-turn steering component

The optional `soft-cut` prompt mode cuts an active model attempt and continues the same agent turn. Version 2 requires `expectedTurn`, the numeric parent turn observed by the Host. The Harness compares it with the open turn synchronously, immediately before durable inbox insertion. A closed or different turn rejects the instruction without starting another turn. It does not cancel the turn, a running tool, or compaction. Wait, Send after, Interrupt, and full Stop keep their existing meanings. The desktop exposes this operation separately as **Steer current run**.

This change depends on a focused Harness source patch. The host refuses true steering on an older component; it never substitutes Stop and another prompt. The patch belongs to DeepSeek Harness, not the inference engine.

## Build from the reviewed source

Use a supported Node development installation with Node-API headers, pnpm, and the normal Harness build tools. A runtime-only Node bundled with the app has no development headers. No installed Strata app is changed by these commands.

The source pin is `571902028a75c64c1d767f4ae5fd099485f0ddf0` of `deepseek-ai/deepseek-harness`. `soft-cut.json` records the patch hash and the before/after hashes of every changed source file. The helper refuses different source files. It applies the source patch before compilation, and never rewrites compiled JavaScript.

The patch also includes Harness's generated persistence inventory and bilingual acknowledgement for the optional `steeringTurn` source field. Its classifier reports four optional additions with no required Session format migration.

```sh
# Obtain the separate Harness source. Do not run these in the Strata checkout.
harness_src=/tmp/strata-harness-source
git init "$harness_src"
git -C "$harness_src" fetch --depth=1 https://github.com/deepseek-ai/deepseek-harness.git 571902028a75c64c1d767f4ae5fd099485f0ddf0
git -C "$harness_src" checkout --detach FETCH_HEAD
node apps/desktop/scripts/harness-steering.mjs --src "$harness_src" --apply

# In that source checkout, use its frozen dependency versions and standard build.
cd "$harness_src"
corepack pnpm install --frozen-lockfile
corepack pnpm run build:artifacts
corepack pnpm run build:check
```

Run the focused source tests named in `soft-cut.json` before staging. Return to the Strata checkout, record the built module hashes, and use the existing component staging command:

```sh
node apps/desktop/scripts/harness-steering.mjs --src "$harness_src" --record
node apps/desktop/scripts/stage-harness.mjs --src "$harness_src" --out /tmp/strata-steering-harness
```

Staging verifies that the compiled modules match the recorded source build, and carries the patch identity and hashes into `BUNDLE.json`. The host uses that identity to enable `soft-cut`. Unpatched source builds still stage normally and do not advertise the operation.

## Verification limits

The host's `agent-steering.test.mjs` uses the actual Harness process and Gateway with a scripted HTTP model. Set `STRATA_HARNESS_DIR` to the staged component and run it with Node 24. It checks current-turn identity, durable ordered delivery, Gateway reconnect, incomplete output, running side effects, approval, and full cancellation. These are runtime integration tests; they do not measure a real model's response to steering. Harness source tests separately cover model-attempt, tool-scheduler, and compaction boundaries.

The desktop stores failed or uncertain instructions for review. Retry keeps the original request identity and expected numeric turn. The `strata-harness-soft-cut-v2` capability replaces v1; the Host refuses older components. A received acknowledgement must include the expected `turn`. Only a committed user message whose source carries the matching `steeringTurn` and request ID is marked delivered. The added optional source metadata is preserved in the existing extensible message source; queue and ordinary steer records keep their existing representation.

The regression sequence holds A's prepared steering RPC, completes A, opens B, then releases the RPC. Admission must reject A's correction, leave B's inbox and model requests untouched, and retain a failed Host receipt. Repeated identities acknowledge only their original admitted turn. Tests separately exercise closure without B and correct-turn duplicate delivery.
