import { auditInventory } from '../inventory-audit.mjs';
export const name = 'strata-workflows';
export const inject = ['tools', 'commands', 'sessions', 'llm'];
const REASON = 'Research only permits web_search and web_fetch while gathering, and no tools while writing the answer. Return the answer in chat; do not save files or ask to save them.';

export function apply(ctx) {
  const policies = new Map();
  const policyOf = session => {
    for (let depth = 0; session && depth < 8; depth++) {
      if (policies.has(session.id)) return policies.get(session.id);
      session = session.header?.parentSession ? ctx.sessions.get(session.header.parentSession) : null;
    }
    return null;
  };
  const denial = exec => {
    const policy = policyOf(exec.agent?.session);
    return policy && !(policy.phase === 'gather' && ['web_search', 'web_fetch'].includes(exec.name)) ? REASON : undefined;
  };
  // Hide capabilities from the model and independently reject stale/direct calls,
  // before approval prompts as well as after the extensible permission waterfall.
  ctx.on('tools/pre-execute', (exec, next) => denial(exec) ? { kind: 'deny', reason: REASON } : next(), { prepend: true });
  ctx.tools.guard(denial);
  // The Harness may register session-local delegation tools after a global
  // allowlist is installed. Filter the final per-request schema as well; the
  // execution guard above remains authoritative for stale/direct calls.
  const originalStream = ctx.llm.streamWithRegistration;
  if (typeof originalStream !== 'function') throw new Error('Research requires the pinned LLM request boundary.');
  const researchStream = function (options, prepared) {
    const policy = policyOf(ctx.sessions.get(options.sessionId));
    const scoped = policy ? { ...options, tools: (options.tools ?? []).filter(tool => policy.phase === 'gather' && ['web_search', 'web_fetch'].includes(tool.name)) } : options;
    return originalStream.call(this, scoped, prepared);
  };
  ctx.llm.streamWithRegistration = researchStream;
  ctx.commands.register({
    definitionId: 'strata-research', name: 'strata-research', description: 'Host-controlled research capability scope', input: { hint: 'gather, write or off' },
    handler: async ({ agent, rawInput }) => {
      if (!agent || !['gather', 'write', 'off'].includes(rawInput?.trim())) return { kind: 'error', text: 'Invalid research phase.' };
      const phase = rawInput.trim(), previous = policies.get(agent.session.id);
      if (phase === 'off') { previous?.release(); policies.delete(agent.session.id); return { kind: 'success', text: 'research off' }; }
      try {
        let release;
        await agent.ctx.inject(['tools'], scoped => { release = scoped.tools.restrict({ allow: phase === 'gather' ? ['web_search', 'web_fetch'] : [] }); });
        policies.set(agent.session.id, { phase, release });
        previous?.release();
        return { kind: 'success', text: `research ${phase}` };
      } catch { return { kind: 'error', text: 'Research tool restrictions could not be established. Nothing should be sent.' }; }
    },
  });
  ctx.tools.register({
    name: 'audit_inventory',
    description: 'After reading inventory.csv, expected.csv and requirements.csv, pass their complete exact CSV content here to compute version mismatches, duplicate/missing IDs and failed checks. A passed status is successful; no extra verification evidence is required. Use these calculated counts in the saved audit and final answer. This tool never reads or writes files.',
    parameters: { type: 'object', additionalProperties: false, required: ['inventory_csv', 'expected_csv', 'requirements_csv'], properties: Object.fromEntries(['inventory_csv', 'expected_csv', 'requirements_csv'].map(key => [key, { type: 'string', maxLength: 1048576 }])) },
    output: { schema: { type: 'object', additionalProperties: true }, render: (_args, value) => [{ type: 'text', text: JSON.stringify(value) }] },
    execute: async args => auditInventory(args),
    isConcurrencySafe: () => true,
  });
  ctx.on('dispose', () => { for (const entry of policies.values()) entry.release(); policies.clear(); if (ctx.llm.streamWithRegistration === researchStream) ctx.llm.streamWithRegistration = originalStream; });
}
