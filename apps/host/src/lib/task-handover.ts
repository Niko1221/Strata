import { tr, localeTag } from './i18n';
import type { Task } from './tasks';

/**
 * "Continue in a new chat" on a task: the first message of the new chat. It carries what the task found, what it did and the files it made, in the
 * words the person reads, so the conversation can go on from the result (a finished task, or one that stopped and needs a person).
 */
const when = (iso?: string) => (iso ? new Date(iso).toLocaleString(localeTag(), { dateStyle: 'medium', timeStyle: 'short' }) : '');

/** The task's saved answer as its parts: the answer, the list of what it did, and the note that something needed a yes ("What it did" is how the host writes it, server/task-report.mjs). */
export function splitTaskOutput(output: string): { found: string; did: string[]; attention: string } {
  let text = String(output ?? '').trim();
  let attention = '';
  const note = /\n\n> ([\s\S]*)$/.exec(text);
  if (note) { attention = note[1].replace(/\n> /g, '\n').trim(); text = text.slice(0, note.index).trim(); }
  const cut = /\n\n---\n\n\*\*What it did\*\*\n\n([\s\S]*)$/.exec(text);
  const did = cut ? cut[1].split('\n').map(l => l.replace(/^-\s*/, '').trim()).filter(Boolean) : [];
  return { found: (cut ? text.slice(0, cut.index) : text).trim(), did, attention };
}

const MAX_STEPS = 10, MAX_STEP = 160;

export const taskHandoverTitle = (task: Task) => tr('Continuing from the task “{0}”', [task.title]);

export function taskHandoverMessage(task: Task): string {
  const failed = !!task.error;
  const ran = when(task.lastRunAt);
  const head = `**${taskHandoverTitle(task)}**${ran ? ` ${failed ? tr('(ran {0}, did not finish).', [ran]) : tr('(ran {0}, finished).', [ran])}` : ''}`;
  const parts = [head];
  if (failed) parts.push(`**${tr('What went wrong')}**\n\n${String(task.error).trim()}`);
  const { found, did, attention } = splitTaskOutput(task.output ?? '');
  if (found) parts.push(`**${tr('What it found')}**\n\n${found}`);
  if (attention) parts.push(`**${tr('Needs your attention')}**\n\n${attention}`);
  if (did.length) {
    // the steps of a long run are many and long: the first ones, shortened
    const shown = did.slice(0, MAX_STEPS).map(l => `- ${l.length > MAX_STEP ? `${l.slice(0, MAX_STEP - 1)}…` : l}`);
    if (did.length > MAX_STEPS) shown.push(`- ${tr('…and {0} more steps', [did.length - MAX_STEPS])}`);
    parts.push(`**${tr('What it did')}**\n\n${shown.join('\n')}`);
  }
  const files = (task.files ?? []).filter(f => f && typeof f.path === 'string' && f.path);
  if (files.length) parts.push(`**${tr('Files')}**\n\n${files.map(f => `- ${f.path}`).join('\n')}`);
  parts.push(tr('Please help me take it from here.'));
  return parts.join('\n\n');
}
