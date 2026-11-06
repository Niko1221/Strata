import { useEffect, useState } from 'react';
import { api } from '../lib/client';
import { tr } from '../lib/i18n';
import { useStore } from '../lib/store';

type NpuMetrics = {
  name?: string;
  index?: number;
  utilization_pct?: number | null;
  aicore_utilization_pct?: number | null;
  memory_used_bytes?: number | null;
  memory_total_bytes?: number | null;
  memory_free_bytes?: number | null;
  memory_allocated_bytes?: number | null;
  memory_reserved_bytes?: number | null;
  temperature_c?: number | null;
  power_w?: number | null;
  source?: string[];
};

type RemoteMetrics = {
  source?: string;
  time?: number;
  model?: string;
  device?: string;
  npu?: NpuMetrics;
  live?: { state?: string; running?: number; queued?: number; tok_s?: number; prefill_tok_s?: number };
  service?: { uptime_s?: number; completed?: number; failed?: number; busy?: boolean };
  engine?: { layers?: number; max_context?: number; expert_slots?: number; expert_cache_mib?: number; expert_budget_mib?: number; weight_dtype?: string };
  weights?: { hbm_weight_bytes?: number; weight_cache_hits?: number; weight_cache_misses?: number; network_bytes?: number; disk_hits?: number };
  totals?: { requests?: number; failed?: number; prompt_tokens?: number; output_tokens?: number };
};

const number = (value: number | null | undefined, digits = 1) => typeof value === 'number' && Number.isFinite(value)
  ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : '—';
const gb = (value: number | null | undefined) => typeof value === 'number' && Number.isFinite(value)
  ? `${(value / 1024 ** 3).toFixed(1)} GB` : '—';
const mb = (value: number | null | undefined) => typeof value === 'number' && Number.isFinite(value)
  ? `${(value / 1024 ** 2).toFixed(0)} MB` : '—';

function Reading({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return <div className="remote-metric-reading"><span>{tr(label)}</span><strong>{tr(value)}</strong>{detail ? <small>{tr(detail)}</small> : null}</div>;
}

export function RemoteMetricsPanel() {
  const remote = useStore(state => state.remoteModel);
  const providers = useStore(state => state.providers);
  const connection = providers?.connections.find(item => item.id === remote?.connectionId);
  const [metrics, setMetrics] = useState<RemoteMetrics | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let alive = true;
    if (!remote || !connection || connection.kind !== 'custom') {
      setMetrics(null);
      setError('');
      setLoading(false);
      return () => { alive = false; };
    }
    const read = async () => {
      setLoading(true);
      try {
        const value = await api.get<RemoteMetrics>(`/api/model-providers/${encodeURIComponent(connection.id)}/metrics`);
        if (alive) { setMetrics(value); setError(''); }
      }
      catch (reason) {
        if (alive) setError(reason instanceof Error ? reason.message : 'Metrics unavailable.');
      }
      finally {
        if (alive) setLoading(false);
      }
    };
    void read();
    const timer = window.setInterval(() => void read(), 3000);
    return () => { alive = false; window.clearInterval(timer); };
  }, [connection?.id, connection?.kind, remote?.connectionId, remote?.modelId]);

  if (!remote || !connection || connection.kind !== 'custom') return null;
  const npu = metrics?.npu;
  const live = metrics?.live;
  const service = metrics?.service;
  const engine = metrics?.engine;
  const weights = metrics?.weights;
  const memory = npu?.memory_used_bytes != null && npu.memory_total_bytes != null
    ? `${gb(npu.memory_used_bytes)} / ${gb(npu.memory_total_bytes)}` : gb(npu?.memory_used_bytes);
  const cache = engine?.expert_cache_mib != null && engine.expert_budget_mib != null
    ? `${number(engine.expert_cache_mib / 1024)} / ${number(engine.expert_budget_mib / 1024)} GB` : mb(weights?.hbm_weight_bytes);

  return <section className="st-card remote-metrics-panel" aria-label={tr("Remote Ascend metrics")}>
    <div className="section-toolbar">
      <div><h2 className="h4">{tr("Remote Ascend metrics")}</h2><span className="small muted">{tr(connection.name)} · {tr(remote.modelId)} · {tr("through Strata")}</span></div>
      <span className="xs muted">{tr(metrics ? `Updated ${new Date((metrics.time ?? 0) * 1000).toLocaleTimeString()}` : loading ? 'Reading…' : 'Not connected')}</span>
    </div>
    {error ? <p className="small inline-error" role="alert">{tr(error)}</p> : null}
    {metrics ? <>
      <div className="remote-metrics-grid">
        <Reading label="NPU utilization" value={npu?.utilization_pct == null ? '—' : `${number(npu.utilization_pct, 1)}%`} detail={npu?.name ?? metrics.device}/>
        <Reading label="NPU memory" value={memory} detail={npu?.memory_allocated_bytes != null ? `allocated ${gb(npu.memory_allocated_bytes)}` : undefined}/>
        <Reading label="Temperature" value={npu?.temperature_c == null ? '—' : `${number(npu.temperature_c, 0)} °C`} detail={npu?.power_w == null ? undefined : `${number(npu.power_w, 0)} W`}/>
        <Reading label="Generation" value={live?.tok_s == null ? '—' : `${number(live.tok_s, 1)} tok/s`} detail={live?.state ?? 'idle'}/>
        <Reading label="Prompt reading" value={live?.prefill_tok_s == null ? '—' : `${number(live.prefill_tok_s, 1)} tok/s`} detail={live?.running ? 'active request' : 'last live state'}/>
        <Reading label="Expert cache" value={cache} detail={engine?.weight_dtype ? `${engine.weight_dtype} · ${engine.layers ?? '—'} layers` : undefined}/>
      </div>
      <div className="remote-metrics-facts">
        <span>{tr("Requests")}: <strong>{number(service?.completed ?? metrics.totals?.requests, 0)}</strong></span>
        <span>{tr("Errors")}: <strong>{number(service?.failed ?? metrics.totals?.failed, 0)}</strong></span>
        <span>{tr("Weights cache hits")}: <strong>{number(weights?.weight_cache_hits, 0)}</strong></span>
        <span>{tr("Weights cache misses")}: <strong>{number(weights?.weight_cache_misses, 0)}</strong></span>
      </div>
      <p className="xs muted">{tr("Read from the Qwen /metrics endpoint through the saved model connection. The API key stays on the Strata host.")}</p>
    </> : <p className="small muted">{tr(loading ? 'Reading remote NPU metrics…' : 'No metrics returned by this model server.')}</p>}
  </section>;
}
