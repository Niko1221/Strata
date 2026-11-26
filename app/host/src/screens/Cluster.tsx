import { useEffect, useState } from 'react';
import { api } from '../lib/client';
import { tr } from '../lib/i18n';
import { Screen } from '../components/layout';
import { Btn, Empty, ErrorNotice, Panel, PanelHead } from '../components/ui';

type Node = {
  id: string; address: string; port: number; backend: string; gpu?: string;
  vram_gib?: number; ram_gib?: number; cpu_cores?: number; hostname?: string;
  os?: string; status?: string; last_seen?: string; error?: string;
};
type Cluster = { name: string; model?: string; tls?: boolean; nodes?: Record<string, Node> };

const statusText = (status?: string) => status === 'healthy' ? tr('Healthy')
  : status === 'degraded' ? tr('Degraded')
  : tr('Not reachable');

export function ClusterScreen() {
  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [error, setError] = useState('');
  const [ready, setReady] = useState(false);
  const [pending, setPending] = useState(false);
  const [name, setName] = useState('');
  const [nodeId, setNodeId] = useState('');
  const [address, setAddress] = useState('');
  const [port, setPort] = useState('8791');
  const [backend, setBackend] = useState('cuda');
  const [selected, setSelected] = useState('');

  async function refresh() {
    try {
      const data = await api.get('/api/cluster/status') as { clusters?: Cluster[] };
      setClusters(data?.clusters ?? []);
      setError('');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setReady(true);
    }
  }
  useEffect(() => { void refresh(); }, []);

  async function action(run: () => Promise<unknown>) {
    setPending(true);
    try { await run(); await refresh(); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setPending(false); }
  }

  const current = clusters.find(c => c.name === selected) ?? clusters[0];
  const nodes = Object.values(current?.nodes ?? {});
  const totals = nodes.reduce((acc, n) => ({
    vram: acc.vram + (Number(n.vram_gib) || 0),
    ram: acc.ram + (Number(n.ram_gib) || 0),
    healthy: acc.healthy + (n.status === 'healthy' ? 1 : 0),
  }), { vram: 0, ram: 0, healthy: 0 });

  return <Screen title={tr('Cluster')} sub={tr('Create a cluster to see its resources')}>
    <div className="stack">
      {error ? <ErrorNotice message={error}/> : null}
      <Panel><PanelHead title={tr('Create cluster')}/>
        <div className="row wrap">
          <input aria-label={tr('Name')} placeholder={tr('Name')} value={name}
                 onChange={e => setName(e.target.value)} />
          <Btn disabled={pending || !name.trim()} onClick={() => action(async () => {
            await api.post('/api/cluster', { name: name.trim() });
            setSelected(name.trim()); setName('');
          })}>{tr('Create cluster')}</Btn>
        </div>
      </Panel>

      {!ready ? <p className="muted">{tr('Loading…')}</p> : clusters.length === 0
        ? <Empty title={tr('No cluster yet')} body={tr('Create a cluster to see its resources')}/>
        : <>
          <div className="row wrap">
            {clusters.map(c => <button key={c.name} type="button" className="st-btn"
              aria-pressed={current?.name === c.name} onClick={() => setSelected(c.name)}>{c.name}</button>)}
            {current ? <Btn disabled={pending} onClick={() => action(() => api.post(
              `/api/cluster/${encodeURIComponent(current.name)}/refresh`, {}))}>{tr('Refresh')}</Btn> : null}
          </div>
          <Panel><PanelHead title={tr('Add node')}/>
            <div className="row wrap">
              <input aria-label={tr('Name')} placeholder={tr('Name')} value={nodeId} onChange={e => setNodeId(e.target.value)}/>
              <input aria-label={tr('Address')} placeholder={tr('Address')} value={address} onChange={e => setAddress(e.target.value)}/>
              <input aria-label={tr('Port')} placeholder={tr('Port')} value={port} onChange={e => setPort(e.target.value)}/>
              <select aria-label={tr('Backend')} value={backend} onChange={e => setBackend(e.target.value)}>
                {['cuda','hip','sycl','metal','cpu'].map(b => <option key={b} value={b}>{b}</option>)}
              </select>
              <Btn disabled={pending || !current || !nodeId.trim() || !address.trim()}
                onClick={() => action(() => api.post('/api/cluster/node', {
                  cluster: current!.name, id: nodeId.trim(), address: address.trim(),
                  port: Number(port), backend,
                }))}>{tr('Add node')}</Btn>
            </div>
            <p className="small muted">Run this on each machine (from the Strata terminal folder), with the same token as the host:
              <code> cd Strata/terminal </code>
              <code> $env:STRATA_CLUSTER_TOKEN='your-secret' </code>
              <code> python -m strata_control.cluster.agent --host 0.0.0.0 --port {port} </code>
              For a quick local test, add a node with address 127.0.0.1 and start the agent on this machine.</p>
          </Panel>
          <Panel><PanelHead title={tr('Resources')}/>
            <div className="summary-list">
              <div><span>{tr('Nodes')}</span><strong>{nodes.length}</strong></div>
              <div><span>VRAM</span><strong>{totals.vram} GB</strong></div>
              <div><span>RAM</span><strong>{totals.ram} GB</strong></div>
              <div><span>{tr('Connected')}</span><strong>{totals.healthy}</strong></div>
            </div>
            <div className="st-tablewrap"><table className="st-table">
              <thead><tr><th>{tr('Name')}</th><th>{tr('Address')}</th><th>{tr('OS')}</th>
                <th>CPU</th><th>GPU</th><th>VRAM</th><th>RAM</th>
                <th>{tr('Resources')}</th><th>{tr('Last seen')}</th></tr></thead>
              <tbody>{nodes.map(n => <tr key={n.id}>
                <td>{n.hostname || n.id}</td><td>{n.address}:{n.port}</td><td>{n.os || '—'}</td>
                <td>{n.cpu_cores || '—'}</td><td>{n.gpu || '—'}</td>
                <td>{n.vram_gib || 0}</td><td>{n.ram_gib || 0}</td>
                <td title={n.error || ''}>{statusText(n.status)}</td>
                <td>{n.last_seen ? new Date(n.last_seen).toLocaleString() : '—'}</td>
              </tr>)}</tbody>
            </table></div>
          </Panel>
          <Btn disabled={pending || totals.healthy === 0} onClick={() => setError(tr('Launch on cluster'))}>{tr('Launch on cluster')}</Btn>
        </>}
    </div>
  </Screen>;
}
