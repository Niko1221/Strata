/** Human-only bridge commands. A caller-delivered child never wakes a chat. */
export const name = 'strata-btw';
export const inject = ['subagents', 'agentPresets'];

export function apply(ctx) {
  const active = new Map();
  ctx.on('agent/created', async ({ agent }) => {
    if (ctx.agentPresets.composedPreset(agent.ctx) !== 'strata-btw') return;
    await agent.ctx.inject(['tools', 'commands'], scoped => {
      scoped.tools.restrict({ allow: [] });
      scoped.tools.guard(() => 'Side-question subagents cannot use tools.');
      if (agent.session.header.origin !== 'subagent') registerCommands(scoped.commands);
    });
  });
  const assertSideParent = agent => {
    if (ctx.agentPresets.composedPreset(agent.ctx) !== 'strata-btw') throw new Error('A dedicated side-question parent is required.');
  };
  function registerCommands(commands) {
  commands.register({
    definitionId: 'strata-btw-start', name: 'strata-btw-start', description: 'Start a tool-free side-question subagent',
    input: { hint: 'Host request' },
    handler: async ({ agent, rawInput }) => {
      try {
        assertSideParent(agent);
        if (active.has(agent.id)) throw new Error('A side question is already running.');
        const request = JSON.parse(rawInput);
        if (Object.keys(request).some(k => !['prompt', 'provider', 'model', 'effort'].includes(k)) || (request.effort !== undefined && request.effort !== 'off') || typeof request.prompt !== 'string' || !request.prompt.trim() || request.prompt.length > 75_000 ||
          typeof request.provider !== 'string' || !request.provider || request.provider.length > 200 || typeof request.model !== 'string' || !request.model || request.model.length > 256) throw new Error('Invalid side question.');
        const controller = new AbortController();
        const run = await ctx.subagents.startActivation({
          provider: 'spawn', label: 'Side question', delivery: 'caller', signal: controller.signal,
          request: { parent: agent, prompt: [{ type: 'text', text: request.prompt }], agentPreset: 'strata-btw', maxDepth: 1,
            agentOptions: { provider: request.provider, model: request.model, reasoningEffort: request.effort, maxTokens: 1536 } },
        });
        active.set(agent.id, { run, controller });
        run.result.catch(() => {});
        return { kind: 'success', text: JSON.stringify({ parentSessionId: agent.id, childSessionId: run.childId, mode: 'continuable' }) };
      } catch { return { kind: 'error', text: 'The side-question subagent could not start.' }; }
    },
  });
  commands.register({
    definitionId: 'strata-btw-end', name: 'strata-btw-end', description: 'Release this side-question subagent',
    handler: async ({ agent }) => {
      try {
        assertSideParent(agent);
        const entry = active.get(agent.id);
        if (entry) { entry.controller.abort(); await entry.run.dispose(); active.delete(agent.id); }
        return { kind: 'success' };
      } catch { return { kind: 'error', text: 'The side-question subagent could not be released.' }; }
    },
  });
  }
  ctx.on('dispose', async () => { await Promise.allSettled([...active.values()].map(async entry => { entry.controller.abort(); await entry.run.dispose(); })); active.clear(); });
}
