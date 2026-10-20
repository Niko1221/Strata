import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createModelProviders, normalizeModelManagement } from '../server/model-providers.mjs';

const draft = { kind: 'custom', name: 'Terminal server', baseUrl: 'http://inference.invalid/v1', protocol: 'openai-chat', models: [] };
test('SSH management is per connection, optional, persisted privately and independent of inference credentials', t => {
  const directory = mkdtempSync(join(tmpdir(), 'strata-ssh-settings-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const service = createModelProviders({ directory });
  const old = service.save(draft).connections[0];
  assert.equal(Object.hasOwn(old, 'modelManagement'), false, 'The existing API path keeps its saved shape.');
  const first = service.save({ ...draft, name: 'First server', modelManagement: { transport: 'ssh', host: 'garage-alias', user: 'alice', port: 2201, sourcePath: '~/apps/Strata' } }).connections.at(-1);
  const second = service.save({ ...draft, name: 'Second server', modelManagement: { transport: 'ssh', host: '2001:db8::1', user: 'bob', sourcePath: '/opt/Strata' } }).connections.at(-1);
  assert.equal(first.apiKeySet, false, 'SSH does not require an inference API key.');
  assert.deepEqual(first.modelManagement, { transport: 'ssh', host: 'garage-alias', user: 'alice', port: 2201, sourcePath: '~/apps/Strata' });
  assert.equal(second.modelManagement.user, 'bob');
  assert.equal(Object.hasOwn(second.modelManagement, 'port'), false, 'An omitted port honors SSH configuration.');
  const reopened = createModelProviders({ directory });
  assert.deepEqual(reopened.list(), service.list());
  const { modelManagement, ...legacyEditor } = first;
  const retained = reopened.save({ ...legacyEditor, baseUrl: 'http://new-inference.invalid/v1' }).connections.find(row => row.id === first.id);
  assert.deepEqual(retained.modelManagement, modelManagement, 'Older editors do not erase SSH configuration.');
  const cleared = reopened.save({ ...retained, modelManagement: null }).connections.find(row => row.id === first.id);
  assert.equal(Object.hasOwn(cleared, 'modelManagement'), false, 'Selecting API explicitly removes SSH settings.');
  const configured = reopened.save({ ...second, modelManagement: { transport: 'ssh', host: 'another-alias' } }).connections.find(row => row.id === second.id);
  assert.deepEqual(configured.modelManagement, { transport: 'ssh', host: 'another-alias' }, 'No username or port is invented.');
  assert.equal(readFileSync(join(directory, 'model-providers.json'), 'utf8').includes('privateKey'), false);
});

test('SSH configuration rejects option injection, credentials and malformed ports before saving', () => {
  const valid = { transport: 'ssh', host: 'server-alias' };
  for (const patch of [
    { host: '-oProxyCommand=bad' }, { host: 'alice@host' }, { host: 'host with spaces' }, { host: 'host\n-oAnything' },
    { host: 'ssh://host' }, { user: '-oBad' }, { user: 'user\nname' }, { port: 0 }, { port: 65536 },
    { port: 1.5 }, { port: '22' }, { port: null }, { sourcePath: 'relative/Strata' }, { sourcePath: '/tmp/Strata\0bad' },
    { password: 'never-store-this' }, { privateKey: 'never-store-this' }, { transport: 'shell' },
  ]) assert.throws(() => normalizeModelManagement({ ...valid, ...patch }), undefined, JSON.stringify(Object.keys(patch)));
  assert.deepEqual(normalizeModelManagement({ ...valid, host: ' ::1 ', user: '', sourcePath: '' }), { transport: 'ssh', host: '::1' });
  for (const value of [[], 'ssh', true, 12]) assert.throws(() => normalizeModelManagement(value));
});

test('only custom connections can opt in, and corrupted saved SSH settings fail visibly', t => {
  const directory = mkdtempSync(join(tmpdir(), 'strata-ssh-settings-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const service = createModelProviders({ directory });
  assert.throws(() => service.save({ kind: 'provider', provider: 'openai', apiKey: 'fixture', models: [{ id: 'fixture' }], modelManagement: { transport: 'ssh', host: 'server' } }), /custom/);
  service.save(draft);
  const file = join(directory, 'model-providers.json'), rows = JSON.parse(readFileSync(file, 'utf8'));
  rows[0].modelManagement = { transport: 'ssh', host: '-malformed' };
  writeFileSync(file, JSON.stringify(rows));
  assert.throws(() => createModelProviders({ directory }), /SSH host/);
  assert.equal(JSON.parse(readFileSync(file, 'utf8'))[0].modelManagement.host, '-malformed', 'Bad configuration is not silently erased.');
});
