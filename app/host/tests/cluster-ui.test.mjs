import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mkdtempSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const src = p => readFileSync(new URL(p, import.meta.url), 'utf8');

test('Cluster screen is registered in router, app shell and sidebar', () => {
  assert.match(src('../src/lib/router.ts'), /'cluster'/);
  assert.match(src('../src/app/App.tsx'), /case 'cluster':/);
  assert.match(src('../src/app/App.tsx'), /screens\/Cluster/);
  assert.match(src('../src/app/Sidebar.tsx'), /route: 'cluster'/);
});

test('Cluster screen uses translated strings and the cluster API', () => {
  const screen = src('../src/screens/Cluster.tsx');
  assert.match(screen, /api\.get\('\/api\/cluster\/status'\)/);
  assert.match(screen, /api\.post\('\/api\/cluster\/node'/);
  assert.match(screen, /tr\('Create cluster'\)/);
  assert.match(screen, /tr\('Launch on cluster'\)/);
});

test('host exposes the cluster routes', () => {
  const host = src('../server/index.mjs');
  for (const route of ['/api/cluster/status', '/api/cluster/node']) {
    assert.ok(host.includes(route), route);
  }
});

test('adding a node probes its agent and records real resources', async () => {
  const token = 'probe-token';
  process.env.STRATA_HOME = mkdtempSync(join(tmpdir(), 'strata-cluster-probe-'));
  process.env.STRATA_CLUSTER_TOKEN = token;
  const server = createServer((req, res) => {
    if (req.url !== '/cluster/info' || req.headers['x-cluster-token'] !== token) {
      res.writeHead(401).end(); return;
    }
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ ok: true, hostname: 'test-node', os: 'TestOS', backend: 'cuda',
      gpu: 'RTX Test', vram_gib: 12, ram_gib: 64, cpu_cores: 8, status: 'healthy' }));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const mod = await import('../server/cluster.mjs');
    await mod.createCluster('probe-lab');
    const node = await mod.addNode('probe-lab', {
      id: 'local', address: '127.0.0.1', port: server.address().port, backend: 'cuda',
    });
    assert.equal(node.status, 'healthy');
    assert.equal(node.gpu, 'RTX Test');
    assert.equal(node.vram_gib, 12);
    assert.equal(node.ram_gib, 64);
    assert.equal(node.cpu_cores, 8);
    const again = await mod.refreshCluster('probe-lab');
    assert.equal(again.nodes.local.hostname, 'test-node');
  } finally {
    await new Promise(resolve => server.close(resolve));
  }
});

test('losing a node is recorded without breaking the cluster file', async () => {
  process.env.STRATA_HOME = mkdtempSync(join(tmpdir(), 'strata-cluster-loss-'));
  process.env.STRATA_CLUSTER_TOKEN = 'loss-token';
  const server = createServer((req, res) => {
    if (req.headers['x-cluster-token'] !== 'loss-token') { res.writeHead(401).end(); return; }
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ ok: true, hostname: 'will-vanish', os: 'TestOS',
      gpu: 'Test GPU', vram_gib: 8, ram_gib: 32, cpu_cores: 4, status: 'healthy' }));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const mod = await import('../server/cluster.mjs');
  await mod.createCluster('loss-lab');
  await mod.addNode('loss-lab', {
    id: 'vanishing', address: '127.0.0.1', port: server.address().port, backend: 'cuda',
  });
  await new Promise(resolve => server.close(resolve));
  const after = await mod.refreshCluster('loss-lab');
  assert.equal(after.nodes.vanishing.status, 'offline');
  const reread = await mod.listClusters();
  assert.equal(reread[0].name, 'loss-lab');
  assert.equal(reread[0].nodes.vanishing.status, 'offline');
});

test('cluster service creates a cluster and rejects bad input', async () => {
  const root = mkdtempSync(join(tmpdir(), 'strata-cluster-ui-'));
  process.env.STRATA_HOME = root;
  const mod = await import('../server/cluster.mjs');
  const created = await mod.createCluster('home-lab');
  assert.equal(created.name, 'home-lab');
  const node = await mod.addNode('home-lab', { id: 'desk', address: '192.168.1.10', port: 8791, backend: 'cuda', vram_gib: 24 });
  assert.equal(node.id, 'desk');
  const list = await mod.listClusters();
  assert.equal(list[0].nodes.desk.address, '192.168.1.10');
  await assert.rejects(() => mod.addNode('home-lab', { id: 'Bad Name', address: 'x' }));
  await mod.deleteCluster('home-lab');
});
