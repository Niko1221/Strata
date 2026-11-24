/**
 * How far a local engine has got, as a short text that changes whenever it makes progress. The agent's stall watchdog
 * compares it with the last one it saw: while a long prompt is read the Harness hears nothing for minutes, but the
 * engine's /metrics (polled every 2 s) shows `live.prompt_read` moving (#3623). Null when the engine is idle or says nothing.
 *
 * `live.generated` counts every token the model samples, its hidden thinking included (the server counts them before it
 * sorts them into thinking and answer), so a model that thinks for a long time without showing a word keeps changing the
 * mark (#6794).
 */
export function engineProgressMark(instance) {
  const live = instance?.metrics?.live;
  if (!live || live.state === 'idle' || live.state === 'unloaded') return null;
  const slots = Array.isArray(live.slots) ? live.slots.map(s => `${s?.state}:${s?.prompt_read ?? ''}:${s?.generated ?? ''}`).join(',') : '';
  return `${live.state}|${live.prompt_read ?? ''}|${live.generated ?? ''}|${slots}`;
}

/** The longest a busy engine may go without its mark changing before the watchdog stops counting it (#6794). The mark moves once per prompt
 *  chunk while a prompt is read and once per token after it, so a quiet stretch this long means a slow chunk (a long agent prompt on a CPU
 *  or an SSD-streamed model reads one chunk in minutes), not a lost turn. */
export const ENGINE_QUIET_MS = 20 * 60_000;

/**
 * The watchdog's question for a local turn: does the engine's state say it is still working on this turn?
 * `turn` carries `heard` (when the Harness last spoke), `engineMark` and `engineMarkAt` (the last mark seen and when it changed); this updates them
 * and returns true when the turn is alive (`turn.lastActivity` is moved on a change). A busy engine whose mark has not changed counts for
 * `quietMs` more; an idle or silent one (mark null) counts for nothing, so the turn's normal limit applies. Nothing counts once the Harness
 * has been silent for `graceMs`: progress for another chat must not keep a lost turn alive for ever.
 */
export function engineKeepsTurnAlive(turn, mark, { now = Date.now(), graceMs, quietMs = ENGINE_QUIET_MS } = {}) {
  if (now - turn.heard >= graceMs || mark == null) return false;
  if (mark !== turn.engineMark) {
    turn.engineMark = mark; turn.engineMarkAt = now; turn.lastActivity = now;
    return true;
  }
  return now - (turn.engineMarkAt ?? turn.heard) < quietMs;
}
