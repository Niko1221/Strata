import test from 'node:test';
import assert from 'node:assert/strict';
import {parseLoopbackUrl, openLoopbackUrl, openLoopbackCommand} from '../server/open-loopback.mjs';

test('parseLoopbackUrl accepts only 127.0.0.1 / ::1 with a validated port', () => {
  assert.deepEqual(parseLoopbackUrl('http://127.0.0.1:18500/'), {
    protocol: 'http', host: '127.0.0.1', port: 18500, path: '/', href: 'http://127.0.0.1:18500/',
  });
  assert.throws(() => parseLoopbackUrl('http://192.168.1.1:18500/'), /local 127\.0\.0\.1/);
  assert.throws(() => parseLoopbackUrl('http://example.com/'), /local 127\.0\.0\.1/);
  assert.throws(() => parseLoopbackUrl('http://127.0.0.1:80/'), /1024 to 65535/);
  assert.throws(() => parseLoopbackUrl('file:///tmp/x'), /local 127\.0\.0\.1/);
});

test('openLoopbackUrl never embeds credentials and records the browser command', () => {
  const recorded = [];
  const result = openLoopbackUrl('http://127.0.0.1:18501/', { recorder: recorded, platform: 'win32' });
  assert.equal(result.url, 'http://127.0.0.1:18501/');
  assert.deepEqual(recorded[0], ['rundll32.exe', 'url.dll,FileProtocolHandler', 'http://127.0.0.1:18501/']);
  assert.equal(JSON.stringify(recorded).includes('Bearer'), false);
  assert.equal(JSON.stringify(recorded).includes('api_key'), false);
  assert.deepEqual(openLoopbackCommand('http://127.0.0.1:19000/', 'darwin'), ['open', ['http://127.0.0.1:19000/']]);
});
