import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { transform } from 'esbuild';

/**
 * Owner decision 12 (0.2.22): the chat toolbar's "Pause for update" button is shown only when an update is waiting, from what the app already
 * knows (the last update check kept on this computer, Control's own kept check). While the agents are paused it shows "Resume agents".
 * /pause-for-update and /resume-agents stay as commands.
 */

const src = p => readFileSync(new URL(p, import.meta.url), 'utf8');

async function load() {
  const { code } = await transform(src('../src/lib/update-waiting.ts'), { loader: 'ts', format: 'esm' });
  return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
}
const check = strataStatus => ({ checkedAt: '2026-11-25T08:00:00Z', strata: { status: strataStatus, canInstall: true, message: '' }, engine: { status: 'current', canInstall: false, message: '' } });

test('nothing known, or a check that found nothing, is no update waiting', async () => {
  const { strataUpdateWaiting } = await load();
  assert.equal(strataUpdateWaiting(null, null), false);
  assert.equal(strataUpdateWaiting(undefined, undefined), false);
  assert.equal(strataUpdateWaiting(check('current'), { engine: { updateAvailable: null } }), false);
  assert.equal(strataUpdateWaiting(check('error'), { engine: { updateAvailable: { engine: null, app: null } } }), false);
  assert.equal(strataUpdateWaiting(check('unavailable'), { engine: null }), false);
  assert.equal(strataUpdateWaiting(check('no-releases'), {}), false);
});

test('a kept check that found a newer Strata, or Control naming a newer app, is an update waiting', async () => {
  const { strataUpdateWaiting } = await load();
  assert.equal(strataUpdateWaiting(check('available'), null), true);
  assert.equal(strataUpdateWaiting(null, { engine: { updateAvailable: { engine: null, app: '0.2.22' } } }), true);
  assert.equal(strataUpdateWaiting(check('current'), { engine: { updateAvailable: { engine: null, app: '0.2.22' } } }), true);
});

test('a newer engine alone is not one: it does not end the app, so the agents have nothing to wait for', async () => {
  const { strataUpdateWaiting } = await load();
  assert.equal(strataUpdateWaiting(check('current'), { engine: { updateAvailable: { engine: '0.1.68', app: null } } }), false);
});

test('the toolbar renders the button only while an update waits or the agents are paused', () => {
  const bar = src('../src/components/ChatToolbar.tsx');
  assert.match(bar, /const updateWaiting = useStrataUpdateWaiting\(\);/);
  assert.match(bar, /\{\(paused \|\| updateWaiting\) && <Btn /, 'hidden unless an update waits or the pause is on (then it is Resume agents)');
  assert.equal((bar.match(/'Pause for update'/g) ?? []).length, 1, 'the label appears once, inside that condition');
  assert.match(bar, /tr\(paused \? 'Resume agents' : 'Pause for update'\)/);
  // The Btn element closes right after its label, with nothing else of the toolbar inside the condition.
  const start = bar.indexOf('{(paused || updateWaiting) && <Btn ');
  const end = bar.indexOf('</Btn>}', start);
  assert.ok(start > 0 && end > start);
  assert.doesNotMatch(bar.slice(start, end), /chat-agents|menuRef|Chat options/);
});

test('the hook reads only kept state: no network, no host call of its own', () => {
  const check_ = src('../src/lib/update-check.ts');
  const hook = check_.slice(check_.indexOf('export function useStrataUpdateWaiting'), check_.indexOf('/** The last known check result'));
  assert.match(hook, /useNative\(\)/);
  assert.match(hook, /useLastUpdateCheck\(useInstalledVersions\(\)\)/);
  assert.doesNotMatch(hook, /fetch\(|api\.|fetchUpdateCheck/);
});

test('the commands stay: /pause-for-update and /resume-agents, in the chat box and the command palette', () => {
  const chat = src('../src/screens/Chat.tsx');
  assert.match(chat, /text === '\/pause-for-update'/);
  assert.match(chat, /text === '\/resume-agents'/);
  assert.match(chat, /id: 'pause-for-update', label: '\/pause-for-update'/);
  assert.match(chat, /id: 'resume-agents', label: '\/resume-agents'/);
  assert.match(chat, /onPauseForUpdate=\{pauseForUpdate\}/);
  assert.match(chat, /onResumeAfterUpdate=\{resumeAfterUpdate\}/);
});
