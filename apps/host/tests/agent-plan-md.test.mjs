// The plan's checklist file in the chat's folder: the app writes it and ticks it, and the page can open it.
// The pure part always runs; the part through the Harness needs a built Harness (STRATA_HARNESS_DIR) and skips loudly without it.
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, realpathSync, rmSync, statSync, utimesSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';
import { DeliverableTracker } from '../server/agent/deliverables.mjs';
import { PLAN_DIR, planMarkdown, planSlug, writePlanFile } from '../server/agent/plan-md.mjs';
import { resolveChatFile } from '../server/chat-files.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;

const ID = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee';
/** A plan in the shape PlanStore keeps: the fields plan-md.mjs reads, with the step states the card shows. */
function fakePlan(state, steps) {
  return {
    id: ID, title: 'The Pagoda page', source: 'plan', state, request: 'Put the pagoda history in a panel',
    steps: steps.map((s, i) => ({ id: `s${i + 1}`, text: s[0], detail: s[1] ?? '', state: s[2], summary: s[3] ?? null })),
  };
}

function tmpWs(t) {
  const dir = mkdtempSync(join(tmpdir(), 'strata-planmd-'));
  const ws = join(dir, 'ws');
  mkdirSync(ws, { recursive: true });
  t.after(() => { try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  return ws;
}

test('the file is a checklist of the plan the card shows, in its order and state', () => {
  const md = planMarkdown(fakePlan('running', [
    ['Read the folder', 'which files are there', 'done'],
    ['Write the panel', '', 'running'],
    ['Skip the header', '', 'skipped'],
    ['Check the page', '', 'pending'],
  ]));
  const lines = md.split('\n');
  assert.match(lines[0], /^<!-- Strata plan aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee -->$/, 'the first line says whose file this is');
  assert.equal(lines[1], '# The Pagoda page');
  assert.equal(lines[3], 'running · 2 of 4 steps done', 'the count is of the ticked boxes, and a skipped step is ticked');
  assert.match(md, /- \[x\] 1\. Read the folder/);
  assert.match(md, /- \[ \] 2\. Write the panel _\(in progress\)_/);
  assert.match(md, /- \[x\] 3\. Skip the header _\(skipped\)_/, 'a step that was skipped is ticked and says so');
  assert.match(md, /- \[ \] 4\. Check the page/);
  assert.match(md, /^ {2}which files are there$/m, "a step's own words come under it");
  assert.match(md, /^What was asked for:$/m);
  assert.match(md, /^> Put the pagoda history in a panel$/m, 'the file says what the plan is for');
});

test('the file name comes from the title and stays a file name', () => {
  assert.equal(planSlug({ title: 'The Pagoda page!' }), 'the-pagoda-page');
  assert.equal(planSlug({ title: 'Wrocław Café — 2026' }), 'wrocław-café-2026', 'accents are kept, the rest of the word is not cut');
  assert.equal(planSlug({}), 'plan', 'a plan with no title still gets a file');
  assert.equal(planSlug({ title: '!!!' }), 'plan');
});

test('the file is written once, and only rewritten when the plan changed', (t) => {
  const ws = tmpWs(t);
  const first = writePlanFile(ws, fakePlan('proposed', [['Read the folder', '', 'pending'], ['Write the panel', '', 'pending']]));
  assert.equal(first.created, true);
  assert.equal(first.rel, join(PLAN_DIR, 'the-pagoda-page.md'));
  const abs = join(ws, first.rel);
  assert.ok(existsSync(abs), 'the file is in the chat folder');
  utimesSync(abs, new Date(0), new Date(0));
  const again = writePlanFile(ws, fakePlan('proposed', [['Read the folder', '', 'pending'], ['Write the panel', '', 'pending']]));
  assert.equal(again.rel, first.rel); assert.equal(again.created, false);
  assert.equal(statSync(abs).mtimeMs, 0, 'nothing was rewritten: the same plan leaves the file alone');
  assert.equal(existsSync(`${abs}.tmp`), false, 'the temporary file is renamed away');

  const ticked = writePlanFile(ws, fakePlan('running', [['Read the folder', '', 'done'], ['Write the panel', '', 'running']]));
  assert.equal(ticked.created, false);
  const md = readFileSync(abs, 'utf8');
  assert.match(md, /- \[x\] 1\. Read the folder/); assert.match(md, /- \[ \] 2\. Write the panel _\(in progress\)_/);
  assert.match(md, /running · 1 of 2 steps done/);
});

test('a file at that name that is not ours is left alone', (t) => {
  const ws = tmpWs(t);
  const dir = join(ws, PLAN_DIR);
  mkdirSync(dir, { recursive: true });
  const theirs = join(dir, 'the-pagoda-page.md');
  writeFileSync(theirs, '# My own notes\n\n- [ ] do the thing\n');
  const made = writePlanFile(ws, fakePlan('proposed', [['Read the folder', '', 'pending']]));
  assert.equal(made.rel, join(PLAN_DIR, 'the-pagoda-page-2.md'), 'the next name is used');
  assert.equal(readFileSync(theirs, 'utf8'), '# My own notes\n\n- [ ] do the thing\n', 'someone else\'s file is never touched');
  assert.match(readFileSync(join(ws, made.rel), 'utf8'), /^<!-- Strata plan /);
});

test('a plan with nothing in it, or a chat with no folder, gets no file', (t) => {
  const ws = tmpWs(t);
  assert.equal(writePlanFile(ws, { id: ID, title: 'x', steps: [] }), null, 'a plan with no steps has no checklist');
  assert.equal(writePlanFile(ws, { title: 'x', steps: [{ text: 'a', state: 'pending' }] }), null, 'a plan without an id is not ours to track');
  assert.equal(writePlanFile(null, fakePlan('proposed', [['a', '', 'pending']])), null, 'a chat with no folder has no file');
  assert.deepEqual(readdirSync(ws), [], 'nothing was created');
});

test('the file never becomes a deliverable card, and the page can still open it', (t) => {
  const ws = tmpWs(t);
  const tracker = new DeliverableTracker(ws);
  tracker.snapshot();
  const made = writePlanFile(ws, fakePlan('proposed', [['Read the folder', '', 'pending']]));
  assert.equal(tracker.check(made.abs), null, 'a folder starting with a dot never becomes a card, so the plan file does not crowd the chat');
  writeFileSync(join(ws, 'notes.md'), 'a real file\n');
  assert.equal(tracker.check(join(ws, 'notes.md'))?.kind, 'markdown', 'an ordinary file in the same folder does become one, so the rule above is the folder, not the format');
  assert.equal(resolveChatFile(ws, made.rel), realpathSync(made.abs), 'Open and the preview can reach the plan file through the chat folder');
});

async function rig(t) {
  const worker = await startScriptedWorker({});
  const dir = mkdtempSync(join(tmpdir(), 'strata-planmd-run-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const local = [{ id: 'inst-1', label: 'Scripted', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots: 2 }];
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => local, connections: () => [], env: process.env, startTimeoutMs: 90_000 });
  t.after(async () => {
    await bridge.close(); await worker.close();
    try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ }
  });
  const turn = (chatId, onFrame = () => {}, extra = {}) => ({ chatId, workspacePath: ws, mode: 'plan', onFrame, ...extra });
  return { bridge, ws, turn };
}

test('Plan mode: the turn ends with a real plan and its checklist file, and the model still wrote nothing', { skip }, async (t) => {
  const r = await rig(t);
  const out = [];
  await r.bridge.runTurn({ ...r.turn('md1', f => out.push(f)), message: 'SCRIPT:planreply' });
  const frame = out.find(f => f.type === 'plan');
  assert.ok(frame, 'a Plan-mode turn that ends with a numbered list sends a plan frame');
  assert.equal(frame.event, 'proposed');
  assert.equal(frame.plan.steps.length, 3);
  assert.equal(frame.plan.context, 'I looked around the folder first.', 'what the turn found by looking comes with the plan, not only its steps');
  const file = frame.plan.file;
  assert.equal(file, join(PLAN_DIR, 'the-pagoda-plan.md'), 'the frame carries the file of this plan');
  const abs = join(r.ws, file);
  const md = readFileSync(abs, 'utf8');
  assert.match(md, /^# The pagoda plan$/m);
  assert.equal((md.match(/^- \[ \] \d\. /gm) ?? []).length, 3, 'every step starts unticked');
  assert.match(md, /waiting for you · 0 of 3 steps done/, 'the file says in words what the card shows');
  const askedAt = md.indexOf('What was asked for:'), foundAt = md.indexOf('I looked around the folder first.'), stepsAt = md.indexOf('## Steps');
  assert.ok(askedAt > 0 && foundAt > askedAt && stepsAt > foundAt, 'the file is read top down: the ask, then what was found, then ## Steps last');
  assert.deepEqual(readdirSync(r.ws), ['.strata'], 'Plan mode is read-only: the only folder in the chat is the app\'s own');
  const stored = r.bridge.planState('md1');
  assert.equal(stored.id, frame.plan.id); assert.equal(stored.file, file, 'the plan keeps one file name, so a new title does not move it');
});

test('Plan mode: a turn with no numbered plan does not make one, and gets no file', { skip }, async (t) => {
  const r = await rig(t);
  const out = [];
  await r.bridge.runTurn({ ...r.turn('md2', f => out.push(f)), message: 'SCRIPT:echo hello' });
  assert.deepEqual(out.filter(f => f.type === 'plan'), [], 'no numbered steps, no plan');
  assert.equal(r.bridge.planState('md2'), null);
  assert.deepEqual(readdirSync(r.ws), [], 'no plan, no plan folder');
});

test('Plan mode then Start as goal: the same file gets ticked as the steps finish', { skip }, async (t) => {
  const r = await rig(t);
  await r.bridge.runTurn({ ...r.turn('md3'), message: 'SCRIPT:planreply' });
  const plan = r.bridge.planState('md3');
  const abs = join(r.ws, plan.file);
  const approved = r.bridge.planControl({ chatId: 'md3', planId: plan.id, action: 'approve' });
  assert.equal(approved.plan.file, plan.file, 'approving keeps the same file');
  assert.match(readFileSync(abs, 'utf8'), /approved · 0 of 3 steps done/);
  const edited = r.bridge.planControl({ chatId: 'md3', planId: plan.id, action: 'edit', steps: plan.steps.map(s => s.text) });
  assert.equal(edited.plan.context, plan.context, 'editing the steps keeps what was found');
  assert.equal(r.bridge.planState('md3').context, plan.context, 'and it is the stored plan that keeps it, not only the reply');
  const out = [];
  await r.bridge.planGoal({ chatId: 'md3', workspacePath: r.ws, mode: 'bypass', planId: plan.id, onFrame: f => out.push(f) });
  const md = readFileSync(abs, 'utf8');
  assert.equal((md.match(/^- \[x\] \d\. /gm) ?? []).length, 3, 'all three steps are ticked in the file');
  assert.match(md, /done · 3 of 3 steps done/);
  assert.match(md, /> /, 'a finished step leaves its summary in the file');
  assert.equal(r.bridge.planState('md3').file, plan.file, 'running the plan did not move or rename the file');
  assert.equal(out.filter(f => f.type === 'plan' && f.file !== undefined).length, 0, 'the frames the page gets are the plan frames, unchanged');
});
