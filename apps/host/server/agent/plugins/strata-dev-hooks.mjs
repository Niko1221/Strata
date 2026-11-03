/**
 * Strata Dev (docs/DEV_BUILD.md): the `on_tool_call` hook, as a Harness plugin beside strata-gate.mjs. The host inserts it into
 * the agent profile only when STRATA_DEV=1 (agent/profile.mjs); Live never loads this file.
 *
 * Every hook in dev/extensions/backend that exports `on_tool_call(call, ctx)` (or a Python hook that names it in its
 * `# strata-hooks:` line) sees each tool call before it runs: `{ name, callId, arguments, sessionId }`. A hook that returns
 * `{ deny: 'why' }` refuses the call (the model reads the reason); anything else lets it go on to the gate and the tool.
 * A hook that throws or takes too long is switched off for the rest of this agent's life and logged; the call goes on.
 *
 * It runs after the Strata gate (it is added later and not prepended), so in Manual a refused call never shows a card.
 */
import { createDevHooks } from '../../dev-hooks.mjs';

export const name = 'strata-dev-hooks';
export const inject = ['tools'];

/** @param {{ dir?: string, timeoutMs?: number, python?: { command: string, args?: string[] } }} config */
export function apply(ctx, config = {}) {
  if (!config.dir) return;
  const hooks = createDevHooks({
    dir: config.dir,
    settings: { timeoutMs: config.timeoutMs ?? 2000, disabled: config.disabled ?? [] },
    ...(config.python?.command ? { python: config.python } : {}),
    onLog: (line, level) => { try { (level === 'error' ? console.error : console.log)(`[strata-dev-hooks] ${line}`); } catch { /* logging never decides a call */ } },
  });
  const loading = hooks.load().catch(e => { try { console.error(`[strata-dev-hooks] ${e?.message ?? e}`); } catch { /* ignore */ } });

  ctx.on('tools/pre-execute', async (exec, next) => {
    try {
      await loading;
      const call = { name: String(exec.name ?? ''), callId: exec.callId ?? null, arguments: exec.arguments && typeof exec.arguments === 'object' ? exec.arguments : {}, sessionId: exec.agent?.session?.id ?? null };
      const verdict = await hooks.run('on_tool_call', call);
      if (verdict && typeof verdict.deny === 'string') return { kind: 'deny', reason: `${verdict.deny} (Strata Dev hook ${verdict.by})` };
    } catch (e) {
      try { console.error(`[strata-dev-hooks] ${e?.message ?? e}`); } catch { /* ignore */ }
    }
    return next();
  });
}
