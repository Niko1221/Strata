// Groundwork API for opt-in LAN clusters. Reads/writes the same JSON files as
// terminal/strata_control/cluster/registry.py. Nothing here is enabled by default.
import { promises as fs } from 'node:fs';
import nodePath from 'node:path';

const NAME = /^[a-z][a-z0-9_-]{0,63}$/;
const BACKENDS = new Set(['cuda', 'hip', 'sycl', 'metal', 'cpu']);

export function clusterRoot() {
  const base = process.env.STRATA_HOME || nodePath.join(process.env.USERPROFILE || process.env.HOME || '.', '.strata');
  return nodePath.join(base, 'clusters');
}

function validName(value) {
  const name = String(value ?? '');
  if (!NAME.test(name)) throw new Error('Use lowercase letters, digits, - or _ and start with a letter.');
  return name;
}

function fileFor(name) {
  return nodePath.join(clusterRoot(), `${validName(name)}.json`);
}

async function read(name) {
  try {
    return JSON.parse(await fs.readFile(fileFor(name), 'utf8'));
  } catch (error) {
    if (error.code === 'ENOENT') throw new Error('This cluster does not exist yet.');
    throw error;
  }
}

async function write(name, data) {
  const root = clusterRoot();
  await fs.mkdir(root, { recursive: true });
  const target = fileFor(name);
  const temporary = `${target}.tmp`;
  await fs.writeFile(temporary, `${JSON.stringify(data, null, 2)}\n`, 'utf8');
  await fs.rename(temporary, target);
}

export async function listClusters() {
  try {
    const names = (await fs.readdir(clusterRoot())).filter(name => name.endsWith('.json')).map(name => name.slice(0, -5)).filter(name => NAME.test(name)).sort();
    const clusters = [];
    for (const name of names) {
      try { clusters.push(await read(name)); } catch { /* skip a broken file */ }
    }
    return clusters;
  } catch (error) {
    if (error.code === 'ENOENT') return [];
    throw error;
  }
}

export async function createCluster(name) {
  const clean = validName(name);
  await fs.mkdir(clusterRoot(), { recursive: true });
  const target = fileFor(clean);
  try {
    await fs.writeFile(target, `${JSON.stringify({ name: clean, model: '', token_env: 'STRATA_CLUSTER_TOKEN', tls: true, nodes: {} }, null, 2)}\n`, { encoding: 'utf8', flag: 'wx' });
  } catch (error) {
    if (error.code === 'EEXIST') throw new Error('That cluster already exists.');
    throw error;
  }
  return { name: clean };
}

export function validateNode(raw) {
  const id = validName(raw.id);
  const address = String(raw.address ?? '').trim();
  if (!address || address.length > 253 || /\s/.test(address)) throw new Error('Enter a valid address.');
  const port = Number(raw.port ?? 8899);
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Port must be between 1 and 65535.');
  const backend = String(raw.backend ?? 'cuda');
  if (!BACKENDS.has(backend)) throw new Error('Unknown backend.');
  return { id, address, port, backend, gpu: String(raw.gpu ?? '').slice(0, 80), vram_gib: Number(raw.vram_gib) || 0, ram_gib: Number(raw.ram_gib) || 0, expert_slots: Number(raw.expert_slots) || 0, role: 'worker', token_env: 'STRATA_CLUSTER_TOKEN', status: 'unknown', last_seen: '' };
}

function clusterToken() {
  return process.env.STRATA_CLUSTER_TOKEN || '';
}

async function probe(node) {
  const token = clusterToken();
  try {
    const response = await fetch(`http://${node.address}:${node.port}/cluster/info`, {
      headers: token ? { 'X-Cluster-Token': token } : {},
      signal: AbortSignal.timeout(3000),
    });
    if (!response.ok) return { ...node, status: 'offline', error: `HTTP ${response.status}` };
    const info = await response.json();
    return {
      ...node,
      backend: info.backend || node.backend,
      gpu: info.gpu || '',
      vram_gib: Number(info.vram_gib) || 0,
      ram_gib: Number(info.ram_gib) || 0,
      cpu_cores: Number(info.cpu_cores) || 0,
      hostname: info.hostname || '',
      os: info.os || '',
      status: info.ok ? 'healthy' : 'degraded',
      last_seen: new Date().toISOString(),
      error: '',
    };
  } catch (error) {
    return { ...node, status: 'offline', error: error.message || 'unreachable' };
  }
}

export async function addNode(name, raw) {
  const node = validateNode(raw);
  const cluster = await read(name);
  if (cluster.nodes?.[node.id]) throw new Error('That node is already registered.');
  cluster.nodes = cluster.nodes || {};
  cluster.nodes[node.id] = await probe(node);
  await write(cluster.name, cluster);
  return cluster.nodes[node.id];
}

export async function refreshCluster(name) {
  const cluster = await read(name);
  const entries = Object.entries(cluster.nodes || {});
  const probed = await Promise.all(entries.map(async ([id, node]) => [id, await probe(node)]));
  cluster.nodes = Object.fromEntries(probed);
  await write(cluster.name, cluster);
  return cluster;
}

export async function removeNode(name, nodeId) {
  const cluster = await read(name);
  if (!cluster.nodes?.[nodeId]) throw new Error('That node is not registered.');
  delete cluster.nodes[nodeId];
  await write(cluster.name, cluster);
  return { removed: nodeId };
}

export async function deleteCluster(name) {
  await fs.unlink(fileFor(name));
  return { deleted: name };
}
