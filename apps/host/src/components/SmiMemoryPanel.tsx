/** SMI2000: the Monitor's "Memory and experts" card - system RAM, and where the model's experts are (VRAM, RAM, NVMe). */
import { localeTag, tr } from '../lib/i18n';
import { hasReading } from '../lib/monitor-hardware';
import { expertTiers, type ExpertTier as Tier } from '../lib/smi-expert-tiers';
import type { MonitorState } from '../lib/types';
import { ReferenceSpark } from './charts';
import { Bar } from './ui';

const number = (value: number | null | undefined, digits = 0) => hasReading(value) ? value.toLocaleString(localeTag(), { maximumFractionDigits: digits, minimumFractionDigits: digits }) : '—';
const clamp = (value: number, max: number) => Math.max(0, Math.min(max, value));

export function SmiMemoryPanel({ data }: { data: MonitorState }) {
    const hw = data.hardware, ramUsed = hw.ramUsedGb, ramTotal = hw.ramTotalGb;
    const ramPct = hasReading(ramUsed) && hasReading(ramTotal) && ramTotal > 0 ? clamp(ramUsed / ramTotal * 100, 100) : null;
    const tiers = expertTiers(data.expertCache);
    const rows = tiers ? [
        { id: 'vram', label: 'VRAM', tier: tiers.vram },
        { id: 'ram', label: 'RAM', tier: tiers.ram },
        { id: 'disk', label: 'NVMe disk', tier: tiers.disk },
    ] : [];
    const count = (t: Tier) => hasReading(t.count) ? `${t.approx ? '≈' : ''}${number(t.count)} ${tr('experts')}` : '—';
    return <section className="st-card smi-memory" aria-label={tr('Memory and experts')}>
      <h2 className="h4">{tr('Memory and experts')}</h2>
      <div className="smi-memory__grid">
        <div className="smi-memory__ram">
          <div className="monitor-bar-reading"><span>{tr('System RAM')}</span><span className="muted num">{hasReading(ramUsed) ? `${number(ramUsed, 1)} / ${number(ramTotal)} GB` : '—'}</span></div>
          {hasReading(ramPct) ? <Bar label={tr('System RAM')} value={ramPct} tone={ramPct > 92 ? 'danger' : 'accent'}/> : <div className="st-progress"/>}
          <div className="smi-memory__facts xs muted">
            <span>{hasReading(ramPct) ? `${number(ramPct)}% ${tr('used')}` : ''}</span>
            <span>{hasReading(ramUsed) && hasReading(ramTotal) ? `${number(Math.max(0, ramTotal - ramUsed), 1)} GB ${tr('free')}` : ''}</span>
            {tiers && hasReading(tiers.ram.gb) ? <span>{tr('Experts kept in RAM')}: {number(tiers.ram.gb, 1)} GB</span> : null}
          </div>
          <ReferenceSpark label={tr('System RAM')} data={data.history.map(s => s.ramUsedGb)} unit="GB" timestamps={data.history.map(s => s.t)} maximum={hasReading(ramTotal) ? ramTotal : undefined} tone="var(--st-info)"/>
        </div>
        <div className="smi-memory__experts">
          <div className="monitor-bar-reading"><span>{tr('Expert placement')}</span><span className="muted num">{tiers ? `${tr('Total')} ${number(tiers.totalCount)} ${tr('experts')}${hasReading(tiers.totalGb) ? ` · ${number(tiers.totalGb, 1)} GB` : ''}` : '—'}</span></div>
          <div className="smi-stack" role="img" aria-label={tr('Expert placement')}>{rows.map(r => <span key={r.id} data-tier={r.id} style={{ width: `${clamp(r.tier.pct, 100)}%` }}/>)}</div>
          {tiers ? <dl className="smi-memory__tiers">{rows.map(r => <div key={r.id} data-tier={r.id}><dt><i/>{tr(r.label)}</dt><dd className="num">{count(r.tier)}{hasReading(r.tier.gb) ? ` · ${number(r.tier.gb, 1)} GB` : ''} · {number(r.tier.pct, 1)}%</dd></div>)}</dl> : null}
          <p className="xs muted">{tr(tiers ? 'Experts in RAM were counted when the model loaded. Experts on disk are read on demand; the system file cache may hold some of them.' : 'This engine does not report where its experts are.')}</p>
        </div>
      </div>
    </section>;
}
