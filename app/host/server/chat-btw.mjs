/** Validate a tool-free subagent side question. Main-agent state is never modified. */
import { validChatId } from './chat-store.mjs';

export function btwRequest(body) {
  if (!validChatId(body?.sessionId)) throw new Error('A valid chat is required for /btw.');
  if (typeof body.message !== 'string' || !body.message.trim() || body.message.length > 4000) throw new Error('Add a side question of 1–4000 characters after /btw.');
  const history = [];
  let size = 0;
  for (const row of (Array.isArray(body.history) ? body.history : []).slice(-40).reverse()) {
    if (!['user', 'assistant'].includes(row?.role) || typeof row.content !== 'string') continue;
    const content = row.content.slice(0, 8000);
    if (size + content.length > 64_000) break;
    history.unshift({ role: row.role, content }); size += content.length;
  }
  return { message: `Answer this side question briefly using the supplied conversation as context. You have no tools. Do not claim to change files, run actions or affect the main task.\n\nSide question: ${body.message.trim()}`, history };
}
