/**
 * Example backend hook for Strata Dev (docs/DEV_BUILD.md, "Backend-Hooks"). Only loaded when STRATA_DEV=1.
 * Edit it and press Reload in the Dev panel (or POST /api/dev/reload); a broken hook is switched off, the host keeps running.
 */
export const name = 'example';

/** The chat request before it runs. Return an object to replace it; return nothing to keep it. */
export function before_request(body, ctx) {
  ctx.log(`chat request: ${String(body.message ?? '').slice(0, 60)}`);
}

/** The finished answer: text, requestId, error, durationMs. */
export function after_response(result, ctx) {
  ctx.log(`answer: ${result.text.length} characters in ${result.durationMs} ms${result.error ? ` (error: ${result.error})` : ''}`);
}

/** A model became ready. */
export function on_model_load(model, ctx) {
  ctx.log(`model ready: ${model.label ?? model.id} on port ${model.port}`);
}

/** Every tool call of the agent (runs in the agent's process). Return { deny: 'why' } to refuse one. */
export function on_tool_call(call) {
  if (call.name === 'bash' && /\brm\s+-rf\s+\/(\s|$)/.test(String(call.arguments.command ?? ''))) return { deny: 'The example hook refuses rm -rf /.' };
}

/** Own routes: GET http://127.0.0.1:8792/dev/example/ping */
export const routes = {
  'GET /ping': ({ query }) => ({ pong: true, at: new Date().toISOString(), query }),
};
