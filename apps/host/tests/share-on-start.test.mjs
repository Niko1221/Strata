// The AppImage's `--lan` (STRATA_LAN=1): the network share comes up with Strata, before anyone opens a page.
// The shell turns the switch into STRATA_SHARE_ON_START=1 (apps/desktop/src-tauri/src/supervisor.rs); the host reads it here.
// Nothing is shared when the switch is not given, which is the default path (docs/SAFETY.md).
import test from 'node:test';
import assert from 'node:assert/strict';
import { createLocalShare, shareOnStartRequested, shareOnStartLine, NO_KEY_WARNING } from '../server/local-share.mjs';

const openaiLike = () => (req, res, path, method) => {
  if (path === '/v1/models' && method === 'GET') { res.writeHead(200, { 'content-type': 'application/json' }); res.end('{"object":"list","data":[]}'); return; }
  res.writeHead(404, { 'content-type': 'application/json' }); res.end('{"error":"No route"}');
};

test('nothing is shared at start unless the switch is on', () => {
  assert.equal(shareOnStartRequested({}), false);
  assert.equal(shareOnStartRequested({ STRATA_SHARE_ON_START: '' }), false);
  assert.equal(shareOnStartRequested({ STRATA_SHARE_ON_START: '0' }), false);
  assert.equal(shareOnStartRequested({ STRATA_SHARE_ON_START: 'browser' }), false);
});

test('the shell is not the only way in: the words it accepts are the words a person can set', () => {
  for (const value of ['1', 'on', 'TRUE', 'yes', 'lan', ' Lan ']) {
    assert.equal(shareOnStartRequested({ STRATA_SHARE_ON_START: value }), true, value);
  }
});

test('the line the host prints names the address to type on the other device', () => {
  const line = shareOnStartLine({ active: true, url: 'http://192.168.1.20:47832/v1', warning: null });
  assert.equal(line, '[strata] sharing the model on your network at http://192.168.1.20:47832/v1');
});

test('a share that starts without a key says so, in the same words the page uses', () => {
  const line = shareOnStartLine({ active: true, url: 'http://10.0.0.9:8792/v1', warning: NO_KEY_WARNING });
  const [first, second] = line.split('\n');
  assert.equal(first, '[strata] sharing the model on your network at http://10.0.0.9:8792/v1');
  assert.equal(second, `[strata] ${NO_KEY_WARNING}`);
});

test('the listener that comes up at start is the inference-only one, as big as a person left it', async () => {
  // the same createLocalShare index.mjs starts on STRATA_SHARE_ON_START=1: /v1 answers, no host control does
  const share = createLocalShare({ handle: openaiLike(), hasKey: () => false, host: '127.0.0.1', port: 0 });
  try {
    const started = await share.start();
    assert.equal(shareOnStartRequested({ STRATA_SHARE_ON_START: '1' }), true);
    const models = await fetch(`${started.url}/models`);
    assert.equal(models.status, 200);
    assert.equal((await fetch(started.url.replace(/\/v1$/, '') + '/api/serving')).status, 404, 'the share never answers the app APIs');
  } finally { await share.stop(); }
});
