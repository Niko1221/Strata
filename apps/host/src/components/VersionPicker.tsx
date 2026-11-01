import { useEffect, useRef, useState } from 'react';
import { tr } from '../lib/i18n';
import { Icon } from './ui';
import { FloatingMenu } from './FloatingMenu';
import { cx } from '../lib/format';
import { nativeVersionLabel, versionNote, type CatalogVersion } from '../lib/native-catalog';
import { nvfp4Version } from '../lib/native-names';
import { fitOn, recommendedVersions, versionSections, PICK_NOTE, type FitHardware, type PcPick } from '../lib/native-fit';

/** The name a version goes by in the list: the quantization, with the maker only where the quantization alone would be ambiguous. */
export function versionName(v: CatalogVersion): string {
  switch (v.sourceFamily) {
    case 'swift': return `Swift ${v.model}`;
    case 'unsloth': return `Unsloth ${v.model}`;
    case 'nvfp4': return tr(nvfp4Version(v.model)?.name ?? v.model);
    case 'coder': return `Coder ${v.model}`;
    default: return v.model;
  }
}

/** Who made a version (Details only): the family name of the quantization, a proper name in every language. */
const MAKERS: Record<string, string> = { qwen: 'GSQ-RCO', coder: 'GSQ-RCO', swift: 'Swift', unsloth: 'Unsloth', gyro: 'Gyro', nvfp4: 'NVFP4' };

const gb = (n: number) => `${Number.isInteger(n) ? n : n.toFixed(1)} GB`;
const whole = (n: number) => `${Math.round(n)} GB`;

/** A short plain word for the version ("High quality"), empty where the name is all there is. */
function plainLabel(v: CatalogVersion): string {
  const label = nativeVersionLabel(v.model);
  return label && label !== v.model && label !== versionName(v) ? label : '';
}

/**
 * Where the versions are listed: one row per version, each exactly once. The name is the main thing, with a plain word beside it
 * ("High quality") and small tags (Recommended, Downloaded); under it, what it costs: the download and the memory it needs. The maker,
 * the RAM + VRAM split and the SSD figure are behind "Show details" (open from the start in Expert mode). Rows are in order of size
 * inside each model (by download size), and the pick for this PC comes first. With more than one model the sections carry the model name, small and quiet.
 */
export function VersionPicker({ versions, models, selectedId, hardware, onSelect, onOwnGguf, downloaded, showRecommended = true, moreLabel, pick, details: detailsOpen = false }: {
  versions: CatalogVersion[];
  /** The models the versions belong to, in the order shown (each version appears under exactly one); without it the versions are one list. */
  models?: { family: string; title: string; sizes: CatalogVersion[] }[];
  selectedId: string;
  hardware: FitHardware | null | undefined;
  onSelect: (id: string) => void;
  onOwnGguf: () => void;
  /** The versions already on this computer (ready to load): each gets a small "Downloaded" tag. */
  downloaded?: ReadonlySet<string>;
  /** False where the recommended ones are shown elsewhere (the first-run page): the menu then holds only the others. */
  showRecommended?: boolean;
  /** What the button says while the selected version is one of the recommended ones shown elsewhere. */
  moreLabel?: string;
  /** The engine's own pick for this PC (catalog.recommendation); it comes first, tagged Recommended. */
  pick?: PcPick | null;
  /** Start with the technical detail shown (Expert mode). */
  details?: boolean;
}) {
  const trigger = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  const [details, setDetails] = useState(detailsOpen);
  useEffect(() => setDetails(detailsOpen), [detailsOpen]);
  const selected = versions.find(v => v.id === selectedId);
  // Where the recommended ones are shown outside the menu, the menu holds only the others.
  const rows = recommendedVersions(versions, pick);
  const apart = new Set(showRecommended ? [] : rows.map(x => x.v.id));
  const pickId = rows.find(x => x.r.note === PICK_NOTE)?.v.id;
  const sections = versionSections(versions, models, apart, pickId);
  const shown = selected && !apart.has(selected.id) ? selected : undefined;

  const need = (v: CatalogVersion, technical: boolean) => {
    const fit = fitOn(v, hardware);
    if (technical) return v.multi_file && v.disk_gb ? tr("{0} RAM + VRAM, plus about {1} GB of disk", [gb(fit.memoryGb), Math.round(v.disk_gb)]) : tr("{0} RAM + VRAM", [gb(fit.memoryGb)]);
    return v.multi_file && v.disk_gb ? tr("Needs {0} of memory and about {1} GB of disk", [whole(fit.memoryGb), Math.round(v.disk_gb)]) : tr("Needs {0} of memory", [whole(fit.memoryGb)]);
  };
  const row = (v: CatalogVersion) => {
    const fit = fitOn(v, hardware);
    const on = v.id === selectedId;
    const word = plainLabel(v), note = versionNote(v.model);
    const slow = fit.known && fit.spillGb >= 1;
    const technical = [MAKERS[v.sourceFamily], need(v, true), slow ? tr("Uses SSD {0} GB", [Math.round(fit.spillGb)]) : ''].filter(Boolean).join(' · ');
    return <button key={v.id} role="menuitemradio" aria-checked={on} className={cx('version-row', on && 'is-on')} onClick={() => { onSelect(v.id); setOpen(false); }}>
      <span className="version-row__check">{tr(on ? <Icon name="check" size={14}/> : null)}</span>
      <span className="version-row__body">
        <span className="version-row__title"><strong>{versionName(v)}</strong>{tr(word ? <span className="version-row__word">{word}</span> : null)}{tr(v.id === pickId ? <span className="version-tag version-tag--pick">{tr("Recommended")}</span> : null)}{tr(downloaded?.has(v.id) ? <span className="version-tag version-tag--have">{tr("Downloaded")}</span> : null)}{tr(slow ? <span className="version-row__slow">{tr("Slower on this computer")}</span> : null)}</span>
        {tr(details ? <span className="version-row__tech">{technical}{tr(note ? <><br/>{note}</> : null)}</span> : null)}
      </span>
      <span className="version-row__size">{tr("{0} download", [gb(v.download_gb)])}</span>
    </button>;
  };
  // The open menu opens on the chosen version, not at the top of a long list.
  const menu = useRef<HTMLDivElement>(null);
  useEffect(() => { if (open) menu.current?.querySelector('.is-on')?.scrollIntoView({ block: 'nearest' }); }, [open]);
  return <>
    <button ref={trigger} type="button" className="version-trigger" aria-haspopup="menu" aria-expanded={open} aria-label={tr("Model version")} onClick={() => setOpen(v => !v)}>
      <strong>{shown ? versionName(shown) : tr(moreLabel ?? "Choose a version")}</strong>{tr(shown && downloaded?.has(shown.id) ? <span className="version-tag version-tag--have">{tr("Downloaded")}</span> : null)}
      <span className="version-trigger__need muted">{tr(shown ? need(shown, details) : '')}</span>
      <Icon name="chevron" size={14} className="version-trigger__chev"/>
    </button>
    {tr(open ? <FloatingMenu anchor={trigger.current} title={tr("Model version")} width={480} onClose={() => setOpen(false)}>
      <div className="version-menu" ref={menu}>
        {tr(sections.map(s => <div className="version-section" role="group" aria-label={s.title || tr("Model version")} key={s.id}>
          {tr(s.title ? <div className="version-section__title" aria-hidden="true">{s.title}</div> : null)}
          {tr(s.versions.map(row))}
        </div>))}
        <div className="version-foot">
          <button role="menuitem" className="version-foot__btn" onClick={() => { setOpen(false); onOwnGguf(); }}><Icon name="folder" size={14}/>{tr("Use your own GGUF…")}</button>
          <button role="menuitem" className="version-foot__btn" aria-pressed={details} onClick={() => setDetails(d => !d)}>{tr(details ? "Hide details" : "Show details")}</button>
        </div>
      </div>
    </FloatingMenu> : null)}
  </>;
}
