import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { build } from 'esbuild';

const bundle = async (entry) => {
    const built = await build({
        stdin: { contents: `export * from '${entry}';`, resolveDir: process.cwd() },
        bundle: true,
        write: false,
        format: 'esm',
        platform: 'node',
    });
    return import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
};

const scroll = await bundle('./src/lib/subagent-scroll');
const send = await bundle('./src/lib/subagent-send');
const source = await readFile(new URL('../src/components/SubagentRow.tsx', import.meta.url), 'utf8');

test('a detached reader gets a count of new content; a following reader stays at zero', () => {
    assert.equal(scroll.newContentCount(4, 7, false), 3);
    assert.equal(scroll.newContentCount(4, 7, true), 0);
    assert.equal(scroll.newContentCount(7, 4, false), 0);
});

test('the helper scroller follows only at the bottom and lets go on an upward wheel', async () => {
    assert.match(source, /pinned\.current = distance <= FOLLOW_SLACK/);
    assert.match(source, /onScroll=/);
    assert.match(source, /onWheel=/);
    assert.match(source, /wheelLeavesBottom/);
    assert.match(source, /newContentCount/);
    // The old unconditional effect is gone.
    assert.doesNotMatch(source, /useEffect\(\(\) => \{\s*const el = scroller\.current;\s*if \(el\)\s*el\.scrollTop = el\.scrollHeight;/);
    // A new entry while detached must only update the jump affordance.
    assert.match(source, /setNewCount\(count => count \+ Math\.max\(1, newContentCount/);
});

test('a failed send keeps its text and stays retryable', () => {
    const mine = { id: 'local-1', clientId: 'local-1', text: 'Keep this sentence.', state: 'sending', at: 5 };
    const failed = send.markSendFailed(mine, 'The helper did not accept the message.');
    assert.equal(failed.text, 'Keep this sentence.');
    assert.equal(failed.state, 'failed');
    assert.equal(failed.error, 'The helper did not accept the message.');
    const retry = send.markSendRetrying(failed);
    assert.equal(retry.text, 'Keep this sentence.');
    assert.equal(retry.state, 'sending');
    assert.equal(retry.error, null);
    // The dialog uses the helper rather than filtering the optimistic line out.
    assert.match(source, /markSendFailed/);
    assert.match(source, /setDraft\(''\)/);
});
