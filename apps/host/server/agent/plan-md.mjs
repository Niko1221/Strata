/**
 * The plan as a file the person can read, keep and open: a markdown checklist the app writes and ticks.
 *
 * Every plan change passes through the bridge's plan report (its `onPlan`), which calls writePlanFile with the chat's folder.
 * The file sits in ".strata/plans" in that folder and holds the plan's steps as checkboxes. The app rewrites it whole when the
 * plan changes, so a tick in the plan card is a tick in the file, and a plan the person approved is a plan they can read in
 * their own project. Nothing else writes it: a chat in Plan mode stays read-only, so the assistant never touches the file.
 *
 * Shape of the file (it is a plan document, not a data file: the plan store is the thing that is read back). The steps come last, so the
 * person reads the request and what was found first, and the checklist they can watch get ticked at the end:
 *
 *   <!-- Strata plan <id> -->
 *   # Pagoda page
 *
 *   approved · 1 of 4 steps done
 *
 *   What was asked for:
 *
 *   > A page of the pagoda at night.
 *
 *   ## What I found
 *
 *   `pagoda.html` does not exist yet. `style.css` already has the night palette.
 *
 *   ## Steps
 *
 *   - [x] 1. Read the folder
 *   - [ ] 2. Write pagoda.html
 *     One file, inline styles
 *     > Wrote pagoda.html, Changed style.css
 *
 * A file that already exists and does not carry this plan's marker line was written by someone else and is never overwritten.
 */

import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { basename, dirname, join } from 'node:path';

/** Where the plan files of a chat folder live. Hidden, like the app's own per-folder files. */
export const PLAN_DIR = join('.strata', 'plans');

const marker = id => `<!-- Strata plan ${id} -->`;

const BOX = { pending: '[ ]', running: '[ ]', done: '[x]', skipped: '[x]', failed: '[ ]' };
const NOTE = { running: ' _(in progress)_', skipped: ' _(skipped)_', failed: " _(didn't finish)_" };
const STATE = { proposed: 'waiting for you', approved: 'approved', rejected: 'not taken', running: 'running', paused: 'paused', done: 'done', stopped: 'stopped' };

/** A file name from the plan's title: its words kept, everything else a dash. */
export function planSlug(plan) {
  const s = String(plan?.title ?? '').toLowerCase().replace(/[^\p{L}\p{N}]+/gu, '-').replace(/^-+|-+$/g, '').slice(0, 48).replace(/-+$/g, '');
  return s || 'plan';
}

/** The plan as markdown: what was asked for and what was found first, then its steps last as boxes, ticked when the app marked one done. */
export function planMarkdown(plan) {
  const steps = Array.isArray(plan?.steps) ? plan.steps : [];
  const done = steps.filter(s => s.state === 'done' || s.state === 'skipped').length;
  const asked = String(plan?.request ?? '').replace(/\r\n?/g, '\n').trim();
  const found = String(plan?.context ?? '').replace(/\r\n?/g, '\n').trim();
  return [
    marker(plan.id),
    `# ${plan.title || 'Plan'}`,
    '',
    `${STATE[plan.state] ?? plan.state} · ${done} of ${steps.length} step${steps.length === 1 ? '' : 's'} done`,
    ...(asked ? ['', 'What was asked for:', '', ...asked.split('\n').map(l => `> ${l}`.trimEnd())] : []),
    ...(found ? ['', found] : []),
    '',
    '## Steps',
    '',
    ...steps.flatMap((s, i) => [
      `- ${BOX[s.state] ?? '[ ]'} ${i + 1}. ${s.text}${NOTE[s.state] ?? ''}`,
      ...(s.detail ? [`  ${s.detail}`] : []),
      ...(s.summary ? [`  > ${s.summary}`] : []),
    ]),
    '',
  ].join('\n');
}

/** The plan's file in the plans folder: the name the plan already uses (so a new title does not move it), else one from its title; a name another plan owns is stepped past. */
function planRel(folder, plan) {
  const names = [];
  if (plan.file) names.push(basename(String(plan.file)));
  const slug = planSlug(plan);
  for (let k = 1; k <= 12; k++) names.push(k === 1 ? `${slug}.md` : `${slug}-${k}.md`);
  for (const name of names.filter(Boolean)) {
    const abs = join(folder, PLAN_DIR, name);
    if (!existsSync(abs)) return join(PLAN_DIR, name);
    try { if (readFileSync(abs, 'utf8').includes(marker(plan.id))) return join(PLAN_DIR, name); } catch { /* unreadable: try the next name */ }
  }
  return join(PLAN_DIR, `${slug}-${String(plan.id).slice(0, 8)}.md`);
}

/**
 * Bring the chat folder's plan file up to date. Nothing is written when the text is already the same, and a file someone else
 * wrote under that name is left alone. Returns { rel, abs, created } or null when there is no folder, no room for it, or the
 * file is not ours: a plan file is a convenience, so a turn is never failed over it.
 */
export function writePlanFile(folder, plan) {
  if (!folder || !plan?.id || !Array.isArray(plan.steps) || !plan.steps.length) return null;
  try {
    const rel = planRel(folder, plan);
    const abs = join(folder, rel);
    const text = planMarkdown(plan);
    let before = null;
    if (existsSync(abs)) {
      before = readFileSync(abs, 'utf8');
      if (!before.includes(marker(plan.id))) return null;
      if (before === text) return { rel, abs, created: false };
    }
    mkdirSync(dirname(abs), { recursive: true });
    const tmp = `${abs}.tmp`;
    writeFileSync(tmp, text);
    renameSync(tmp, abs);
    return { rel, abs, created: before === null };
  } catch { return null; }
}
