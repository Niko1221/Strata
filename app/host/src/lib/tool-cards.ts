import type { ToolCall } from './types';

/**
 * Whether a tool call is drawn as a card in the reply (#7184).
 *  - A failed call that the same call later made good (`superseded`, set by the host when the same tool with the same arguments worked later in
 *    the turn) is never drawn: it said "failed" about a step that went through.
 *  - With "Hide failed steps" on (Settings > General, default features.ts HIDE_FAILED_TOOLS_BY_DEFAULT) no failed call is drawn at all.
 *    The assistant still sees the error and tries again.
 *  - A call that waits for an answer or was denied is always drawn. In a reply with activity lines only those cards stay, as before.
 */
export function shownToolCard(t: Pick<ToolCall, 'state' | 'superseded'>, grouped: boolean, hideFailed: boolean): boolean {
  if (t.state === 'error' && (t.superseded || hideFailed)) return false;
  return !grouped || t.state === 'proposed' || t.state === 'denied' || t.state === 'error';
}
