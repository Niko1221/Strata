"""The ab_engine versions report: one HTML file (release-style card and a sortable table) and a PNG of the card.

The PNG is a screenshot of the same HTML taken by a headless Edge/Chrome/Chromium, so there is one layout only.
"""
from __future__ import annotations

import os
import shutil
import statistics
import subprocess
import sys
from html import escape
from pathlib import Path

GREEN = "#16a34a"


def _vkey(v: str) -> tuple:
    return tuple(int(p) if p.isdigit() else 0 for p in str(v).split("."))


def card_pair(results: dict) -> tuple[dict, dict] | None:
    ran = sorted((a for a in results.get("arms", []) if a.get("status") == "ran"), key=lambda a: _vkey(a["version"]))
    return (ran[0], ran[-1]) if len(ran) >= 2 else None


def _median(arm: dict, req: str, key: str) -> float | None:
    vals = [r[key] for r in arm.get("rows", []) if r.get("req") == req and isinstance(r.get(key), (int, float))]
    return statistics.median(vals) if vals else None


def _start(arm: dict) -> float | None:
    vals = [s for s in arm.get("start_s") or [] if isinstance(s, (int, float))]
    return statistics.median(vals) if vals else None


def _rel(old, new) -> float | None:
    return (new / old - 1) * 100 if old and new is not None else None


def _pct(p: float | None) -> str:
    return "n/a" if p is None else f"{'+' if round(p) >= 0 else '-'}{abs(round(p))}%"


def _num(v, unit="") -> str:
    return "n/a" if v is None else f"{v:.1f}{unit}" if v < 100 else f"{v:.0f}{unit}"


def _card(results: dict, old: dict, new: dict) -> str:
    e = lambda s: escape(str(s))  # noqa: E731
    reqs = results.get("requests") or []
    dec = {q: (_median(old, q, "decode"), _median(new, q, "decode")) for q in reqs}
    gains = [g for g in (_rel(a, b) for a, b in dec.values()) if g is not None]
    top = max(dec_v for pair in dec.values() for dec_v in pair if dec_v is not None) if gains else 1
    bars = []
    for q, (a, b) in dec.items():
        bar = lambda v, cls: (f'<div class="bar {cls}"><i style="width:{(v or 0) / top * 100:.1f}%"></i>'  # noqa: E731
                              f'<b>{_num(v)}</b></div>')
        bars.append(f'<div class="pair"><span>{e(q)}</span><div>{bar(a, "old")}{bar(b, "new")}</div></div>')
    lp = (_median(old, "long", "prompt"), _median(new, "long", "prompt"))
    tune = f"tuned {results['tune']}" if results.get("tune") else "not tuned"
    chips = [f"{results.get('rounds', 0)} rounds", f"{len(reqs)} requests"]
    chips += [f"{a['version']} did not run" for a in results.get("arms", []) if a.get("status") != "ran"]
    tile = lambda label, big, small: (f'<div class="tile"><div class="label">{e(label)}</div>'  # noqa: E731
                                      f'<div class="value">{e(big)}</div><div class="small">{e(small)}</div></div>')
    return f"""<section id="card"><div class="glow"></div>
<h1>Strata <span class="ver">{e(old['version'])} → {e(new['version'])}</span></h1>
<p class="sub">{e(results.get('gpu', ''))} · {e(results.get('model', ''))} · default settings · {e(tune)}</p>
<div class="main"><div class="dark"><div class="label">Answers, up to</div><div class="big">{_pct(max(gains) if gains else None)}</div>
<div class="legend"><span class="k old"></span>{e(old['version'])} <span class="k new"></span>{e(new['version'])} <em>decode tok/s, median</em></div>
{''.join(bars)}</div>
<div class="tiles">{tile('Long prompt', _pct(_rel(*lp)), f'{_num(lp[0])} → {_num(lp[1])} tok/s prompt')}
{tile('Setup', f"{_num(old.get('setup_s'))} → {_num(new.get('setup_s'), ' s')}", 'first launch: pack, tuning, start')}
{tile('Start', f'{_num(_start(old))} → {_num(_start(new), " s")}', 'median restart')}</div></div>
<div class="foot"><div>{''.join(f'<span class="chip">{e(c)}</span>' for c in chips)}</div><div class="url">github.com/Niko1221/Strata</div></div>
</section>"""


def _rows(results: dict) -> str:
    e = lambda s: escape(str(s))  # noqa: E731
    out = []
    for a in results.get("arms", []):
        reqs = (results.get("requests") or []) if a.get("status") == "ran" else ["—"]
        status = "ran" if a.get("status") == "ran" else f"did not run: {a.get('reason', '')}"
        for q in reqs:
            d, p = _median(a, q, "decode"), _median(a, q, "prompt")
            out.append(f'<tr data-version="{e(a["version"])}" data-req="{e(q)}" data-decode="{"" if d is None else d}">'
                       f'<td>{e(a["version"])}</td><td>{e(q)}</td><td>{_num(d)}</td><td>{_num(p)}</td><td class="delta"></td>'
                       f'<td>{_num(a.get("setup_s"))}</td><td>{_num(_start(a))}</td><td>{e(status)}</td></tr>')
    return "\n".join(out)


CSS = """
body{margin:0;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;background:#f4f5f7;color:#111}
#card{position:relative;overflow:hidden;box-sizing:border-box;width:1600px;height:900px;padding:56px 72px;
 background:#f7f8f6;color:#0f172a;display:flex;flex-direction:column}
#card .glow{position:absolute;top:-320px;right:-260px;width:900px;height:900px;border-radius:50%;
 background:radial-gradient(circle,rgba(22,163,74,.35),rgba(22,163,74,0) 65%)}
#card h1{position:relative;margin:0;font-size:80px;font-weight:800;letter-spacing:-2px;line-height:1}
#card .ver{color:GREEN}
#card .sub{position:relative;margin:16px 0 34px;font-size:26px;color:#475569;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#card .main{position:relative;display:flex;gap:32px;flex:1;min-height:0}
#card .dark{flex:1.55;background:#0f172a;color:#e2e8f0;border-radius:28px;padding:34px 40px;display:flex;flex-direction:column}
#card .label{font-size:22px;text-transform:uppercase;letter-spacing:2px;color:#94a3b8;font-weight:600}
#card .big{font-size:104px;font-weight:800;color:#22c55e;line-height:1.05;margin:4px 0 12px}
#card .legend{font-size:18px;color:#94a3b8;margin-bottom:10px}
#card .legend em{font-style:normal;margin-left:14px}
#card .k{display:inline-block;width:14px;height:14px;border-radius:4px;margin:0 6px 0 10px;vertical-align:-1px}
#card .k.old,#card .bar.old i{background:#64748b}#card .k.new,#card .bar.new i{background:#22c55e}
#card .pair{display:flex;align-items:center;gap:18px;margin-top:10px}
#card .pair>span{width:80px;font-size:20px;color:#cbd5e1}
#card .pair>div{flex:1}
#card .bar{position:relative;height:22px;margin:3px 0;display:flex;align-items:center}
#card .bar i{display:block;height:100%;border-radius:6px;max-width:calc(100% - 80px)}
#card .bar b{margin-left:10px;font-size:18px;font-weight:600}
#card .tiles{flex:1;display:flex;flex-direction:column;gap:20px}
#card .tile{flex:1;background:#fff;border-radius:24px;padding:20px 30px;box-shadow:0 6px 24px rgba(15,23,42,.08);
 display:flex;flex-direction:column;justify-content:center}
#card .tile .label{font-size:18px;color:#64748b}
#card .value{font-size:52px;font-weight:800;line-height:1.1;white-space:nowrap}
#card .small{font-size:18px;color:#64748b}
#card .foot{position:relative;display:flex;justify-content:space-between;align-items:center;margin-top:26px}
#card .chip{display:inline-block;background:#fff;border:1px solid #d1d5db;border-radius:999px;padding:8px 20px;margin-right:12px;
 font-size:20px;color:#334155}
#card .url{font-size:22px;color:#475569;font-weight:600}
main{padding:24px 32px;font-size:15px}
table{border-collapse:collapse;margin-top:12px}
th,td{padding:6px 14px;border-bottom:1px solid #d1d5db;text-align:left}
th{cursor:pointer;user-select:none}
.pos{color:GREEN}.neg{color:#dc2626}
@media (prefers-color-scheme:dark){body{background:#0b1120;color:#e5e7eb}th,td{border-color:#334155}
 select{background:#1e293b;color:#e5e7eb}.neg{color:#f87171}.pos{color:#4ade80}}
""".replace("GREEN", GREEN)

JS = """
const tb=document.querySelector('tbody'),sel=document.getElementById('base');
function delta(){const base={};
 for(const r of tb.rows)if(r.dataset.version===sel.value)base[r.dataset.req]=parseFloat(r.dataset.decode);
 for(const r of tb.rows){const c=r.querySelector('.delta'),b=base[r.dataset.req],d=parseFloat(r.dataset.decode);
  if(isNaN(b)||isNaN(d)||r.dataset.version===sel.value){c.textContent='';c.className='delta';continue}
  const p=Math.round((d/b-1)*100);c.textContent=(p>=0?'+':'-')+Math.abs(p)+'%';c.className='delta '+(p>=0?'pos':'neg')}}
sel.addEventListener('change',delta);delta();
document.querySelectorAll('th').forEach((th,i)=>th.addEventListener('click',()=>{
 const dir=th.dataset.dir==='asc'?-1:1;th.dataset.dir=dir>0?'asc':'desc';
 const val=r=>r.cells[i].textContent,num=s=>s.trim()?Number(s.replace(/[%+]/g,'')):NaN;
 [...tb.rows].sort((a,b)=>{const x=val(a),y=val(b),nx=num(x),ny=num(y);
  return dir*(!isNaN(nx)&&!isNaN(ny)?nx-ny:x.localeCompare(y,undefined,{numeric:true}))}).forEach(r=>tb.appendChild(r))}));
"""


def build_html(results: dict) -> str:
    pair = card_pair(results)
    if pair:
        title = f"Strata {pair[0]['version']} → {pair[1]['version']}"
        card = _card(results, *pair)
    else:
        title, card = "Strata versions", "<p>Fewer than two versions ran, so there is no comparison card.</p>"
    versions = [a["version"] for a in results.get("arms", []) if a.get("status") == "ran"]
    opts = "".join(f'<option{" selected" if pair and v == pair[0]["version"] else ""}>{escape(v)}</option>' for v in versions)
    heads = "".join(f"<th>{h}</th>" for h in ("Version", "Request", "Decode", "Prompt", "Δ Decode", "Setup", "Start", "Status"))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>{CSS}</style></head><body>
{card}
<main><label>Base version <select id="base">{opts}</select></label>
<span> · decode and prompt in tok/s (median over rounds), setup and start in seconds · {escape(str(results.get('created', '')))}</span>
<table><thead><tr>{heads}</tr></thead><tbody>
{_rows(results)}
</tbody></table></main>
<script>{JS}</script></body></html>
"""


def find_browser() -> str | None:
    for name in ("msedge", "chrome", "chromium", "chromium-browser", "google-chrome"):
        if found := shutil.which(name):
            return found
    if os.name == "nt":
        for env in ("PROGRAMFILES(X86)", "PROGRAMFILES"):
            base = os.environ.get(env)
            for rel in (r"Microsoft\Edge\Application\msedge.exe", r"Google\Chrome\Application\chrome.exe"):
                if base and (p := Path(base) / rel).is_file():
                    return str(p)
    return None


def render_png(browser: str, html: Path, png: Path) -> bool:
    cmd = [browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1600,900",
           f"--screenshot={png}", html.resolve().as_uri() + "#card"]
    try:
        done = subprocess.run(cmd, capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0 and png.is_file()


def write_report(results: dict, out_dir: Path) -> tuple[Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    html = out_dir / "report.html"
    html.write_text(build_html(results), encoding="utf-8")
    if not card_pair(results):
        return html, None
    browser = find_browser()
    if not browser:
        print("No Edge, Chrome or Chromium found: wrote report.html only, no report.png.", file=sys.stderr)
        return html, None
    png = out_dir / "report.png"
    if render_png(browser, html, png):
        return html, png
    print("The browser could not take the screenshot: wrote report.html only.", file=sys.stderr)
    return html, None
