import { useEffect, useState } from 'react';
import { api } from '../lib/client';
import { tr } from '../lib/i18n';
import { useStore } from '../lib/store';
import { Select } from './ui';

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
  engine?: { layers?: number; max_context?: number; weight_cache_entries?: number; expert_cache_mib?: number; expert_budget_mib?: number; weight_dtype?: string };
  weights?: { hbm_weight_bytes?: number; weight_cache_hits?: number; weight_cache_misses?: number; network_bytes?: number; disk_hits?: number };
  totals?: { requests?: number; failed?: number; prompt_tokens?: number; output_tokens?: number };
  requests?: { id: string; time: number; state: string; prompt_tokens: number; output_tokens: number; prompt_ms?: number; decode_tok_s?: number; duration_s: number }[];
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
  const [selected, setSelected] = useState('');
  const connections = providers?.connections.filter(item => item.kind === 'custom') ?? [];
  const connection = connections.find(item => item.id === selected) ?? connections.find(item => item.id === remote?.connectionId) ?? connections[0];
  const [metrics, setMetrics] = useState<RemoteMetrics | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let alive = true;
    setMetrics(null);
    if (!connection) {
      setMetrics(null);
      setError('');
      setLoading(false);
      return () => { alive = false; };
    }
    let timer: ReturnType<typeof setTimeout>;
    const read = async () => {
      if (!alive) return;
      if (document.visibilityState === 'hidden') { timer = setTimeout(read, 3000); return; }
      setLoading(true);
      try {
        const value = await api.get<RemoteMetrics>(`/api/model-providers/${encodeURIComponent(connection.id)}/metrics`);
        if (alive) { setMetrics(value); setError(''); }
      }
      catch (reason) {
        if (alive) { setMetrics(null); setError(reason instanceof Error ? reason.message : 'Metrics unavailable.'); }
      }
      finally {
        if (alive) { setLoading(false); timer = setTimeout(read, 3000); }
      }
    };
    void read();
    return () => { alive = false; clearTimeout(timer); };
  }, [connection?.id, connection?.baseUrl]);

  if (!connection) return null;
  const npu = metrics?.npu;
  const live = metrics?.live;
  const service = metrics?.service;
  const engine = metrics?.engine;
  const weights = metrics?.weights;
  const last = metrics?.requests?.find(row => row.state === 'done');
  const decode = live?.running ? live.tok_s : last?.decode_tok_s;
  const prefill = live?.running ? live.prefill_tok_s : last?.prompt_ms ? last.prompt_tokens / (last.prompt_ms / 1000) : null;
  const memory = npu?.memory_used_bytes != null && npu.memory_total_bytes != null
    ? `${gb(npu.memory_used_bytes)} / ${gb(npu.memory_total_bytes)}` : gb(npu?.memory_used_bytes);
  const cache = engine?.expert_cache_mib != null && engine.expert_budget_mib != null
    ? `${number(engine.expert_cache_mib / 1024)} / ${number(engine.expert_budget_mib / 1024)} GB` : mb(weights?.hbm_weight_bytes);

  return <section className="st-card remote-metrics-panel" aria-label={tr("Remote Ascend metrics")}>
    <div className="section-toolbar">
      <div><h2 className="h4">远端昇腾 NPU 监控</h2><span className="small muted">{connection.name} · {metrics?.model ?? connection.models[0]?.id} · {metrics?.device}</span></div>
      {connections.length > 1 ? <Select ariaLabel="监控模型连接" value={connection.id} options={connections.map(c => ({value: c.id, label: c.name}))} onChange={setSelected}/> : null}
      <span className="xs muted">{tr(metrics ? `Updated ${new Date((metrics.time ?? 0) * 1000).toLocaleTimeString()}` : loading ? 'Reading…' : 'Not connected')}</span>
    </div>
    {error ? <p className="small inline-error" role="alert">{tr(error)}</p> : null}
    {metrics ? <>
      <div className="remote-metrics-grid">
        <Reading label="NPU 利用率" value={npu?.utilization_pct == null ? '—' : `${number(npu.utilization_pct, 1)}%`} detail={npu?.name ?? metrics.device}/>
        <Reading label="HBM 已用 / 总量" value={memory} detail={npu?.memory_allocated_bytes != null ? `本进程 Torch 分配 ${gb(npu.memory_allocated_bytes)}` : undefined}/>
        <Reading label="温度 / 功耗" value={npu?.temperature_c == null ? '—' : `${number(npu.temperature_c, 0)} °C`} detail={npu?.power_w == null ? undefined : `${number(npu.power_w, 1)} W`}/>
        <Reading label="生成速度" value={decode == null ? '—' : `${number(decode, 2)} tok/s`} detail={live?.running ? live.state : '最近完成的请求'}/>
        <Reading label="提示词处理速度" value={prefill == null ? '—' : `${number(prefill, 2)} tok/s`} detail={live?.running ? '当前请求' : '最近完成的请求'}/>
        <Reading label="权重缓存 / 预算" value={cache} detail={engine?.weight_dtype ? `${engine.weight_dtype} · ${engine.layers ?? '—'} 层` : undefined}/>
      </div>
      <div className="remote-metrics-facts">
        <span>{tr("Requests")}: <strong>{number(service?.completed ?? metrics.totals?.requests, 0)}</strong></span>
        <span>{tr("Errors")}: <strong>{number(service?.failed ?? metrics.totals?.failed, 0)}</strong></span>
        <span>{tr("Weights cache hits")}: <strong>{number(weights?.weight_cache_hits, 0)}</strong></span>
        <span>{tr("Weights cache misses")}: <strong>{number(weights?.weight_cache_misses, 0)}</strong></span>
      </div>
      <div className="st-tablewrap"><table className="st-table"><thead><tr><th>时间</th><th>状态</th><th>输入 token</th><th>输出 token</th><th>生成 tok/s</th><th>耗时 s</th></tr></thead><tbody>{metrics.requests?.slice(0, 12).map(row => <tr key={row.id}><td>{new Date(row.time * 1000).toLocaleTimeString()}</td><td>{row.state}</td><td>{row.prompt_tokens}</td><td>{row.output_tokens}</td><td>{number(row.decode_tok_s, 2)}</td><td>{number(row.duration_s, 2)}</td></tr>)}</tbody></table></div>
      <p className="xs muted">每 3 秒刷新；NPU 读数包含该卡所有进程，权重缓存和请求统计属于当前模型。无读数显示 —。</p>
    </> : <p className="small muted">{tr(loading ? 'Reading remote NPU metrics…' : 'No metrics returned by this model server.')}</p>}
  </section>;
}
