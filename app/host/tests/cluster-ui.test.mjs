import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mkdtempSync } from 'node:fs';
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
