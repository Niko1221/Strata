/** Actual host/share routing against a disposable remote API; no model acquisition. */
import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function listen(server) { await new Promise(resolve => server.listen(0, '127.0.0.1', resolve)); return server.address().port; }

test('host wires saved-connection management, opt-in, key checks and the restricted sharing listener', { timeout: 40_000 }, async t => {
  const directory = mkdtempSync(join(tmpdir(), 'strata-management-host-'));
  const requests = [];
  const remote = http.createServer(async (req, res) => {
    let raw = ''; for await (const chunk of req) raw += chunk;
    requests.push({ path: req.url, key: req.headers.authorization, body: raw ? JSON.parse(raw) : null });
    res.setHeader('content-type', 'application/json');
    if (req.url === '/v1/strata/models') return res.end(JSON.stringify({ supported: true, targetName: 'Remote fixture', capabilities: { downloads: true, imports: true }, models: [{ id: 'native-fixture', name: 'Fixture model', sizeGb: 7, state: 'available', readPictures: true }], jobs: [] }));
    if (req.url === '/v1/strata/models/plan') return res.end(JSON.stringify({ planId: 'fixture-remote-plan', targetName: 'Remote fixture', modelName: 'Fixture model', downloadGb: 7, diskFreeGb: 100, diskNeededGb: 8, readPictures: true, warnings: [], expiresAt: Date.now() + 300_000 }));
    if (req.url === '/v1/strata/models/downloads') return res.end(JSON.stringify({ job: { id: 'fixture-job', kind: 'download', name: 'Fixture model', state: 'running', summary: 'Synthetic download' } }));
    res.writeHead(404); res.end('{}');
  });
  const remotePort = await listen(remote);
  const reservation = http.createServer(); const port = await listen(reservation); await new Promise(resolve => reservation.close(resolve));
  const host = spawn(process.execPath, ['server/index.mjs'], { cwd: new URL('..', import.meta.url), env: { ...process.env,
    HOME: directory, STRATA_HOME: directory, STRATA_MODELS_DIR: join(directory,'models'), STRATA_COMPANION_DIR: join(directory,'companion'),
    STRATA_NATIVE_REPO: join(directory,'no-installation'), STRATA_BIND: '127.0.0.1', STRATA_PORT: String(port), STRATA_SHARE_PORT: '0',
    STRATA_REAL_ENGINE: '0', STRATA_AGENT: 'off', STRATA_NO_ACQUIRE: '1' }, stdio: 'ignore' });
  t.after(async () => {
    if (host.exitCode === null) { const ended = new Promise(resolve => host.once('exit',resolve)); host.kill('SIGTERM'); await Promise.race([ended,pause(2000)]); if(host.exitCode===null) { host.kill('SIGKILL'); await ended; } }
    remote.closeAllConnections(); await new Promise(resolve => remote.close(resolve)); rmSync(directory,{recursive:true,force:true});
  });
  const origin = `http://127.0.0.1:${port}`;
  async function call(path, method = 'GET', body, headers = {}) {
    const response = await fetch(origin + path, { method, headers: { 'content-type':'application/json', ...headers }, ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
    return { status: response.status, body: await response.json() };
  }
  let online = false;
  for (let attempt = 0; attempt < 100; attempt++) { try { if ((await call('/api/serving')).status === 200) { online = true; break; } } catch {} await pause(50); }
  assert.equal(online, true, 'isolated host started');
  assert.equal((await call('/api/serving')).body.access.remoteModelManagement ?? false, false);
  assert.equal((await call('/api/serving', 'PATCH', { remoteModelManagement:true })).status, 400);
  assert.equal((await call('/v1/strata/models')).status, 403);
  assert.equal((await call('/api/serving', 'PATCH', { apiKey:'fixture-admin-key' })).status,200);
  assert.equal((await call('/api/serving/share','POST',{enabled:true})).status,200);
  const sharing = (await call('/api/serving')).body.sharing;
  const share = `http://127.0.0.1:${new URL(sharing.url).port}`;
  assert.equal((await fetch(share + '/v1/strata/models',{headers:{authorization:'Bearer fixture-admin-key'}})).status,404);
  assert.equal((await call('/api/serving','PATCH',{remoteModelManagement:true},{origin:'https://other.example'})).status,403);
  assert.equal((await call('/api/serving','PATCH',{remoteModelManagement:true})).status,200);
  assert.equal((await call('/v1/strata/models')).status,401);
  assert.equal((await call('/v1/strata/models','GET',undefined,{authorization:'Bearer fixture-admin-key',origin:'https://other.example'})).status,403);
  const sharedManagement = await fetch(share + '/v1/strata/models',{headers:{authorization:'Bearer fixture-admin-key'}});
  assert.equal(sharedManagement.status,200); assert.equal((await sharedManagement.json()).supported,true);
  assert.equal((await fetch(share + '/api/native',{headers:{authorization:'Bearer fixture-admin-key'}})).status,404);
  const saved = await call('/api/model-providers','POST',{kind:'custom',name:'Other Strata',baseUrl:`http://127.0.0.1:${remotePort}/v1`,protocol:'openai-chat',apiKey:'fixture-remote-key',models:[{id:'fixture',name:'Existing model'}]});
  assert.equal(saved.status,200);
  const targets = await call('/api/model-management/targets');
  const target = targets.body.targets.find(value=>value.kind==='remote'); assert.ok(target); assert.equal(target.name,'Other Strata');
  assert.equal(JSON.stringify(targets.body).includes('fixture-remote-key'),false);
  const root = `/api/model-management/targets/${target.id}`;
  assert.equal((await call(root)).body.targetName,'Other Strata');
  const plan = await call(root + '/plan','POST',{modelId:'native-fixture',readPictures:true});
  assert.equal(plan.status,200); assert.notEqual(plan.body.planId,'fixture-remote-plan');
  assert.equal(requests.filter(value=>value.path.endsWith('/downloads')).length,0);
  const started = await call(root + '/downloads','POST',{planId:plan.body.planId});
  assert.equal(started.status,200); assert.equal(started.body.job.id,'fixture-job');
  assert.deepEqual(requests.at(-1).body,{planId:'fixture-remote-plan'}); assert.equal(requests.at(-1).key,'Bearer fixture-remote-key');
  assert.equal((await call(root + '/downloads','POST',{planId:plan.body.planId})).status,409);
  await call('/api/serving','PATCH',{remoteModelManagement:false});
  assert.equal((await fetch(share + '/v1/strata/models',{headers:{authorization:'Bearer fixture-admin-key'}})).status,404);
  await call('/api/serving/share','POST',{enabled:false});
});
