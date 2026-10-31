/** The shared listener's API key check (serving.authorized): Bearer or x-api-key, compared in constant time like the engine's (#213). */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const home = mkdtempSync(join(tmpdir(), 'strata-serving-compare-'));
process.env.STRATA_HOME = home;
const serving = await import('../server/serving.mjs');

test('the API key passes as Bearer or x-api-key, and nothing else passes', () => {
  serving.init({ send() {} }, { port: 8080, host: '127.0.0.1' });
  serving.patch({ apiKey: 'sk-strata-0123456789abcdef01234567' });
  const ok = (headers) => serving.authorized({ headers });
  assert.equal(ok({ authorization: 'Bearer sk-strata-0123456789abcdef01234567' }), true);
  assert.equal(ok({ 'x-api-key': 'sk-strata-0123456789abcdef01234567' }), true);
  assert.equal(ok({}), false);
  assert.equal(ok({ authorization: 'Bearer ' }), false);
  assert.equal(ok({ authorization: 'Bearer sk-strata-0123456789abcdef01234566' }), false);
  assert.equal(ok({ authorization: 'Bearer sk-strata-0123456789abcdef0123456' }), false);       // a prefix of the key
  assert.equal(ok({ authorization: 'Bearer sk-strata-0123456789abcdef012345670' }), false);     // the key plus one character
  assert.equal(ok({ authorization: 'sk-strata-0123456789abcdef01234567' }), false);             // no "Bearer "
  assert.equal(ok({ authorization: 'Bearer wrong', 'x-api-key': 'sk-strata-0123456789abcdef01234567' }), true);
});
test.after(() => { try { rmSync(home, { recursive: true, force: true }); } catch { /* temp folder */ } });
