// Independent checks run after the agent finishes; no generated files are edited.
import assert from 'node:assert/strict';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const folder = process.argv[2];
if (!folder) throw new Error('Usage: node check_fullstack_edges.mjs <fixture-folder>');
const load = name => import(pathToFileURL(path.resolve(folder, name)).href);
const {Database} = await load('db.js');
const {createApi} = await load('api.js');
const {createClient} = await load('client.js');
const results = [];
async function check(name, fn) {
  try { await fn(); results.push({name, passed: true}); }
  catch (error) { results.push({name, passed: false, error: error.message}); }
}
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
}
await check('shared database across API instances cannot oversell', async () => {
  const db = new Database(), apis = [createApi(db), createApi(db)];
  const settled = await Promise.allSettled(Array.from({length: 8}, (_, i) =>
    apis[i % 2].order({tenant: 'a'}, {quantity: 1})));
  assert.equal(settled.filter(r => r.status === 'fulfilled').length, 3);
  assert.equal(db.stock, 0);
  assert.equal(db.orders.length, 3);
});
await check('invalid quantities cannot mutate stock', async () => {
  const db = new Database(), api = createApi(db);
  for (const quantity of [NaN, Infinity, 0, -1, 0.5, '1', null, undefined]) {
    await assert.rejects(async () => api.order({tenant: 'a'}, {quantity}));
    assert.equal(db.stock, 3);
    assert.equal(db.orders.length, 0);
  }
});
await check('stale failure cannot overwrite a newer success', async () => {
  const first = deferred();
  let calls = 0;
  const client = createClient(() => ++calls === 1 ? first.promise : Promise.resolve(['new']));
  const old = client.refresh();
  await client.refresh();
  first.reject(Error('obsolete'));
  await old;
  assert.deepEqual(client.state.orders, ['new']);
  assert.equal(client.state.error, null);
  assert.equal(client.state.loading, false);
});
await check('stale success cannot overwrite a newer error', async () => {
  const first = deferred();
  let calls = 0;
  const client = createClient(() => ++calls === 1 ? first.promise : Promise.reject(Error('latest')));
  const old = client.refresh();
  await client.refresh();
  first.resolve(['obsolete']);
  await old;
  assert.deepEqual(client.state.orders, []);
  assert.equal(client.state.error, 'latest');
  assert.equal(client.state.loading, false);
});
await check('obsolete completion cannot clear active loading state', async () => {
  const first = deferred(), second = deferred();
  let calls = 0;
  const client = createClient(() => ++calls === 1 ? first.promise : second.promise);
  const old = client.refresh(), current = client.refresh();
  first.resolve(['obsolete']);
  await old;
  const loadingBeforeCurrentFinished = client.state.loading;
  second.resolve(['new']);
  await current;
  assert.equal(loadingBeforeCurrentFinished, true);
  assert.equal(client.state.loading, false);
  assert.deepEqual(client.state.orders, ['new']);
});
console.log(JSON.stringify({results, passed: results.every(r => r.passed)}, null, 2));
process.exitCode = results.every(r => r.passed) ? 0 : 1;
