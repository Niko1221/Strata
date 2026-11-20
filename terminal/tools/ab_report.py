"""The ab_engine versions report: one HTML file (release-style card and a sortable table) and a PNG of the card.

The PNG is a screenshot of the same HTML taken by a headless Edge/Chrome/Chromium, so there is one layout only.
"""
from __future__ import annotations

import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from html import escape
from pathlib import Path

GREEN = "#16a34a"


def _vkey(v: str) -> tuple:
    return tuple(int(p) if p.isdigit() else 0 for p in str(v).split("."))


def card_pair(results: dict) -> tuple[dict, dict] | None:
    ran = sorted((a for a in results.get("arms", []) if a.get("status") == "ran"), key=lambda a: _vkey(a["version"]))
    return (ran[0], ran[-1]) if len(ran) >= 2 else None


def _pick(arm: dict, req: str, key: str, suite: str = "base") -> list[dict]:
    """Rows of one request in one suite (rows without a suite are base) that carry a number for key."""
    rows = [r for r in arm.get("rows", []) if r.get("req") == req and (r.get("suite") or "base") == suite
            and isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
    return [r for r in rows if r.get("round", 0) >= 1] or rows  # round 0 is the cold one; keep it only when it is all there is


def _median(arm: dict, req: str, key: str, suite: str = "base") -> float | None:
    vals = [r[key] for r in _pick(arm, req, key, suite)]
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
    top = max((v for pair in dec.values() for v in pair if v), default=0) or 1  # no positive value: all bars empty
    bars = []
    for q, (a, b) in dec.items():
        bar = lambda v, cls: (f'<div class="bar {cls}"><i style="width:calc((100% - 80px) * {max(v or 0, 0) / top:.4f})"></i>'  # noqa: E731
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
<p class="sub">{e(' · '.join(x for x in (results.get('gpu'), results.get('model'), 'default settings', tune) if x))}</p>
<div class="main"><div class="dark"><div class="label">Answers, up to</div><div class="big">{_pct(max(gains) if gains else None)}</div>
<div class="legend"><span class="k old"></span>{e(old['version'])} <span class="k new"></span>{e(new['version'])} <em>decode tok/s, median</em></div>
{''.join(bars)}</div>
<div class="tiles">{tile('Long prompt', _pct(_rel(*lp)), f'{_num(lp[0])} → {_num(lp[1])} tok/s prompt')}
{tile('Setup', f"{_num(old.get('setup_s'))} → {_num(new.get('setup_s'), ' s')}", 'first launch with one-time downloads and pack builds; oldest version runs first')}
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
#card .bar i{display:block;height:100%;border-radius:6px}
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
  if(!(b>0)||isNaN(d)||r.dataset.version===sel.value){c.textContent='';c.className='delta';continue}
  const p=Math.round((d/b-1)*100);c.textContent=(p>=0?'+':'-')+Math.abs(p)+'%';c.className='delta '+(p>=0?'pos':'neg')}}
sel.addEventListener('change',delta);delta();
document.querySelectorAll('th').forEach((th,i)=>th.addEventListener('click',()=>{
 const dir=th.dataset.dir==='asc'?-1:1;th.dataset.dir=dir>0?'asc':'desc';
 const val=r=>r.cells[i].textContent,num=s=>s.trim()?Number(s.replace(/[%+]/g,'')):NaN;
 [...tb.rows].sort((a,b)=>{const x=val(a),y=val(b),nx=num(x),ny=num(y);
  return dir*(!isNaN(nx)&&!isNaN(ny)?nx-ny:x.localeCompare(y,undefined,{numeric:true}))}).forEach(r=>tb.appendChild(r))}));
"""


SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # fixed order; a fifth version gets grey, never a reused colour
GREY, INK, INK2, GRID = "#898781", "#0f172a", "#475569", "#e2e8f0"
SUITE_TITLES = {"ladder": "Context ladder", "think": "Thinking on", "agent": "Agent turns"}

SUITE_CSS = """
.suite h2{margin:28px 0 4px}.suite p{margin:6px 0;color:#475569}.suite table{margin-top:8px}
tr.worse td{background:rgba(220,38,38,.12);color:#b91c1c}td.fast{color:GREEN;font-weight:700}
#ladder{position:relative;overflow:hidden;box-sizing:border-box;width:1600px;height:900px;padding:56px 72px;margin:16px 0;
 background:#f7f8f6;color:#0f172a;display:flex;flex-direction:column}
#ladder:target{position:fixed;left:0;top:0;z-index:99;margin:0}
#ladder h1{margin:0;font-size:72px;font-weight:800;letter-spacing:-2px;line-height:1}
#ladder .ver{color:GREEN}
#ladder .sub{margin:14px 0 26px;font-size:24px;color:#475569;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#ladder .charts{display:flex;gap:32px;flex:1;min-height:0}
#ladder .tile{flex:1;background:#fff;border-radius:24px;padding:20px 24px;box-shadow:0 6px 24px rgba(15,23,42,.08);
 display:flex;flex-direction:column;min-width:0}
#ladder .label{font-size:20px;text-transform:uppercase;letter-spacing:2px;color:#64748b;font-weight:600;margin-bottom:6px}
#ladder svg{width:100%;flex:1;min-height:0;font-family:inherit}
#ladder .foot{display:flex;justify-content:space-between;align-items:center;margin-top:22px}
#ladder .chip{display:inline-block;background:#fff;border:1px solid #d1d5db;border-radius:999px;padding:8px 20px;margin-right:12px;
 font-size:20px;color:#334155}
#ladder .url{font-size:22px;color:#475569;font-weight:600}
@media (prefers-color-scheme:dark){tr.worse td{color:#fca5a5}td.fast{color:#4ade80}.suite p{color:#94a3b8}}
""".replace("GREEN", GREEN)


def _e(s) -> str:
    return escape(str(s))


def _sorted_arms(results: dict) -> list[dict]:
    return sorted((a for a in results.get("arms", []) if a.get("status") == "ran"), key=lambda a: _vkey(a["version"]))


def _correct(arm: dict, suite: str, req: str | None = None) -> tuple[int, int]:
    """Rows with ok True over all rows (all rounds) of the suite, or of one request."""
    rows = [r for r in arm.get("rows", []) if (r.get("suite") or "base") == suite and (req is None or r.get("req") == req)]
    return sum(r.get("ok") is True for r in rows), len(rows)


def _tags(arms: list[dict], suite: str) -> list[str]:
    """Request tags of the suite in the order they were first answered."""
    seen: dict[str, None] = {}
    for a in arms:
        for r in a.get("rows", []):
            if (r.get("suite") or "base") == suite:
                seen.setdefault(r["req"], None)
    return list(seen)


def _worse(arms: list[dict], suite: str) -> dict[str, str]:
    """Version -> best version, for every version with a lower share of correct answers than the best one of this suite."""
    counts = [(a["version"], *_correct(a, suite)) for a in arms]
    counts = [c for c in counts if c[2]]
    if not counts:
        return {}
    best = max(counts, key=lambda c: c[1] / c[2])  # the first of the best ones; the share, as versions may have run different numbers of rounds
    return {v: best[0] for v, ok, n in counts if ok * best[2] < best[1] * n}


def _notes(results: dict, arm: dict, worse: dict[str, str]) -> list[str]:
    out = []
    if arm["version"] in worse:
        out.append(f"weniger richtig als v{worse[arm['version']]}")
    ctx, n = results.get("context"), arm.get("n_ctx")
    if isinstance(ctx, int) and isinstance(n, int) and n < ctx:
        out.append(f"anderer Kontext ({n} statt {ctx})")
    return out


def _fast(vals: dict[str, float | None], worse: dict[str, str]) -> set[str]:
    """The versions with the highest value, among the ones that are not worse; none when there is nothing to compare."""
    ok = {v: x for v, x in vals.items() if x is not None and v not in worse}
    return {v for v, x in ok.items() if x == max(ok.values())} if len(ok) >= 2 else set()


def _td(text, fast: bool = False) -> str:
    return f'<td class="fast">{_e(text)}</td>' if fast else f"<td>{_e(text)}</td>"


def _tr(cells: list[str], worse: bool = False) -> str:
    return f'<tr class="worse">{"".join(cells)}</tr>' if worse else f'<tr>{"".join(cells)}</tr>'


def _table(heads: list[str], rows: list[str]) -> str:
    return f"<table><thead><tr>{''.join(f'<th>{_e(h)}</th>' for h in heads)}</tr></thead><tbody>\n" + "\n".join(rows) + "\n</tbody></table>"


def _unrun(results: dict, suite: str) -> str:
    """One line per version that has no row in the suite, with the reason."""
    out = []
    for a in results.get("arms", []):
        if a.get("status") != "ran":
            out.append(f"v{_e(a['version'])}: did not run: {_e(a.get('reason', ''))}")
        elif not _correct(a, suite)[1]:
            out.append(f"v{_e(a['version'])}: no {suite} results" + (f" ({_e(a['suites_reason'])})" if a.get("suites_reason") else ""))
    return "".join(f'<p class="skipped">{x}</p>' for x in out)


def _nice(v: float) -> float:
    m = 10 ** math.floor(math.log10(v)) if v > 0 else 1
    return next(k * m for k in (1, 2, 2.5, 5, 10) if v <= k * m) if v > 0 else 1


def _chart(series: list[dict]) -> str:
    """One line chart as inline SVG: x = prompt tokens on a log scale, one y-axis, one line per version."""
    w, h, left, right, bottom = 672, 470, 60, 96, 52
    legend, x, row = [], 0, 0  # (series, x, row); a long legend wraps instead of running off the chart
    for s in series if len(series) >= 2 else []:
        name = s["name"] + (" (grey, no own colour)" if s["color"] == GREY else "")
        if x and left + x + 20 + 8 * len(name) > w:
            x, row = 0, row + 1
        legend.append((s, name, left + x, row))
        x += 20 + 8 * len(name) + 22
    top = 48 + 20 * row
    xs = [x for s in series for x, _y, _t in s["pts"]]
    ys = [y for s in series for _x, y, _t in s["pts"]]
    lo, hi = min([8192, *xs]), max([262144, *xs])
    ymax = _nice(max(ys, default=1) * 1.05)
    sx = lambda x: left + (math.log(x) - math.log(lo)) / (math.log(hi) - math.log(lo)) * (w - left - right)  # noqa: E731
    sy = lambda y: top + (h - top - bottom) * (1 - y / ymax)  # noqa: E731
    g = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">']
    t = 8192
    while t <= hi * 1.001:  # x ticks and grid
        if t >= lo * 0.999:
            g.append(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{top}" y2="{h - bottom}" stroke="{GRID}" stroke-width="1"/>'
                     f'<text x="{sx(t):.1f}" y="{h - bottom + 22}" text-anchor="middle" font-size="15" fill="{INK2}">{t // 1024}K</text>')
        t *= 2
    for i in range(5):
        v = ymax * i / 4
        g.append(f'<line x1="{left}" x2="{w - right}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{GRID}" stroke-width="1"/>'
                 f'<text x="{left - 8}" y="{sy(v) + 5:.1f}" text-anchor="end" font-size="15" fill="{INK2}">{v:g}</text>')
    g.append(f'<text x="{(left + w - right) / 2:.0f}" y="{h - 8}" text-anchor="middle" font-size="15" fill="{INK2}">prompt tokens (log scale)</text>')
    g.append(f'<text x="{left - 8}" y="{top - 14}" text-anchor="end" font-size="15" fill="{INK2}">tok/s</text>')
    labels = []
    for s in series:
        pts = s["pts"]
        g.append(f'<polyline fill="none" stroke="{s["color"]}" stroke-width="2" stroke-linejoin="round" '
                 f'points="{" ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y, _t in pts)}"/>')
        for x, y, tip in pts:
            g.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{s["color"]}" stroke="#fff" stroke-width="2">'
                     f'<title>{_e(tip)}</title></circle>')
        labels.append([sy(pts[-1][1]) + 5, sx(pts[-1][0]) + 12, s["name"]])
    labels.sort()
    for i in range(1, len(labels)):  # labels of lines that end close together must not overlap
        labels[i][0] = max(labels[i][0], labels[i - 1][0] + 17)
    g += [f'<text x="{x:.1f}" y="{y:.1f}" font-size="15" font-weight="600" fill="{INK}">{_e(n)}</text>' for y, x, n in labels]
    for s, name, x, row in legend:
        y = 10 + 20 * row
        g.append(f'<rect x="{x}" y="{y}" width="14" height="14" rx="3" fill="{s["color"]}"/>'
                 f'<text x="{x + 20}" y="{y + 12}" font-size="15" fill="{INK}">{_e(name)}</text>')
    return "".join(g) + "</svg>"


def _ladder_series(arms: list[dict], key: str) -> list[dict]:
    tags = sorted((t for t in _tags(arms, "ladder")), key=lambda t: int(t.split(".")[1].rstrip("k") or 0))
    have = [a for a in arms if _correct(a, "ladder")[1]]
    out = []
    for i, a in enumerate(have):
        pts = []
        for tag in tags:
            x, y = _median(a, tag, "prompt_n", "ladder"), _median(a, tag, key, "ladder")
            if x and y is not None:
                rounds = ", ".join(str(n) for n in sorted({r.get("round", 0) for r in _pick(a, tag, key, "ladder")}))
                pts.append((x, y, f"v{a['version']} · {tag.split('.')[1].upper()}: {_num(y)} tok/s (rounds {rounds})"))
        if pts:
            out.append({"name": f"v{a['version']}", "color": SERIES[i] if i < len(SERIES) else GREY, "pts": pts})
    return out


def _ladder_card(results: dict, arms: list[dict]) -> str:
    tiles = "".join(f'<div class="tile"><div class="label">{_e(title)}</div>{_chart(_ladder_series(arms, key))}</div>'
                    for title, key in (("Prompt tok/s", "prompt"), ("Decode tok/s", "decode")))
    sub = " · ".join(x for x in (results.get("gpu"), results.get("model"), f"context {results['context']}" if results.get("context") else "") if x)
    chips = "".join(f'<span class="chip">v{_e(a["version"])}</span>' for a in arms)
    return (f'<section id="ladder"><h1>Strata <span class="ver">context ladder</span></h1><p class="sub">{_e(sub)}</p>'
            f'<div class="charts">{tiles}</div><div class="foot"><div>{chips}</div><div class="url">github.com/Niko1221/Strata</div></div></section>')


def _ladder(results: dict, arms: list[dict]) -> str:
    have = [a for a in arms if _correct(a, "ladder")[1]]
    if not have:
        return ""
    worse = _worse(have, "ladder")
    summary = _table(["Version", "Correct", "Note"], [
        _tr([_td(f"v{a['version']}"), _td("richtig {}/{}".format(*_correct(a, "ladder"))), _td("; ".join(_notes(results, a, worse)))],
            a["version"] in worse) for a in have])
    tags = sorted(_tags(have, "ladder"), key=lambda t: int(t.split(".")[1].rstrip("k") or 0))
    heads, rows = ["Step"], []
    for a in have:
        heads += [f"v{a['version']} prompt", f"v{a['version']} decode", f"v{a['version']} correct"]
    for tag in tags:
        cells = [_td(tag.split(".")[1].upper())]
        prompt = {a["version"]: _median(a, tag, "prompt", "ladder") for a in have}
        decode = {a["version"]: _median(a, tag, "decode", "ladder") for a in have}
        fp, fd = _fast(prompt, worse), _fast(decode, worse)
        for a in have:
            v = a["version"]
            cells += [_td(_num(prompt[v]), v in fp), _td(_num(decode[v]), v in fd), _td("richtig {}/{}".format(*_correct(a, "ladder", tag)))]
        rows.append(_tr(cells))
    return (_ladder_card(results, have) + "<p>Prompt and decode in tok/s, median of the rounds from round 1 on when there is more than "
            "one round. Correct counts every answer of every round.</p>" + summary + _table(heads, rows))


def _think(results: dict, arms: list[dict]) -> str:
    have = [a for a in arms if _correct(a, "think")[1]]
    if not have:
        return ""
    worse = _worse(have, "think")
    rows = []
    for tag in _tags(have, "think"):
        mine = [a for a in have if _correct(a, "think", tag)[1]]
        fast = _fast({a["version"]: _median(a, tag, "decode", "think") for a in mine}, worse)
        for a in mine:
            v = a["version"]
            rows.append(_tr([_td(tag), _td(f"v{v}"), _td(_num(_median(a, tag, "decode", "think")), v in fast),
                             _td(_num(_median(a, tag, "predicted_n", "think"))), _td(_num(_median(a, tag, "wall_s", "think"))),
                             _td("richtig {}/{}".format(*_correct(a, "think", tag))), _td("; ".join(_notes(results, a, worse)))],
                            v in worse))
    tot = " · ".join(f"v{_e(a['version'])} richtig {'{}/{}'.format(*_correct(a, 'think'))}" for a in have)
    return (f"<p>Thinking on, greedy. Decode in tok/s (over reasoning and answer), generated tokens and wall time in seconds are medians "
            f"of the rounds from round 1 on when there is more than one round.</p><p>{tot}</p>"
            + _table(["Task", "Version", "Decode tok/s", "Tokens", "Wall s", "Correct", "Note"], rows))


def _agent(results: dict, arms: list[dict]) -> str:
    have = [a for a in arms if _correct(a, "agent")[1]]
    if not have:
        return ""
    worse = _worse(have, "agent")
    turns = _tags(have, "agent")
    per = {a["version"]: {t: _median(a, t, "decode", "agent") for t in turns} for a in have}
    med = {v: statistics.median([x for x in d.values() if x]) if any(d.values()) else None for v, d in per.items()}
    first = have[0]["version"]
    fast = _fast(med, worse)
    rows = []
    for a in have:
        v = a["version"]
        if v == first:
            ratio = "baseline"
        else:
            r = [per[v][t] / per[first][t] for t in turns if per[v][t] and per[first][t]]
            ratio = f"x{statistics.median(r):.3f}, {sum(x > 1 for x in r)}/{len(r)} turns faster" if r else "n/a"
        notes = _notes(results, a, worse)
        if a.get("left_out"):
            notes.append("left out, too long for the context: " + ", ".join(str(t) for t in a["left_out"]))
        rows.append(_tr([_td(f"v{v}"), _td(_num(med[v]), v in fast), _td(ratio), _td("richtig {}/{}".format(*_correct(a, "agent"))),
                         _td("; ".join(notes))], v in worse))
    return (f"<p>Median decode tok/s over all turns, and the median of the per-turn ratio to v{_e(first)}. Per turn: median of the "
            f"rounds from round 1 on when there is more than one round.</p>"
            + _table(["Version", "Decode tok/s (median)", f"Ratio to v{first}", "Correct", "Note"], rows))


def suite_head(results: dict) -> str:
    """The header line of a run with suites: context asked and served, texts, sampling."""
    if not results.get("suites"):
        return ""
    served = ", ".join(f"v{_e(a['version'])} {_e(a['n_ctx'])}" for a in results.get("arms", []) if a.get("n_ctx"))
    parts = [f"Suites: {_e(', '.join(results['suites']))}", f"context {_e(results.get('context') or 'n/a')}" + (f" (served: {served})" if served else ""),
             f"texts from git {_e(results.get('corpus') or 'n/a')}", "temperature 0, seed 7",
             "model code " + ("was run on this PC" if results.get("run_code") else "was not run")]
    return "<p>" + " · ".join(parts) + "</p>"


def suite_sections(results: dict) -> str:
    """One HTML section per chosen suite; empty when no suite ran."""
    chosen = [s for s in SUITE_TITLES if s in (results.get("suites") or [])]
    if not chosen:
        return ""
    arms = _sorted_arms(results)
    body = {"ladder": _ladder, "think": _think, "agent": _agent}
    out = [f"<style>{SUITE_CSS}</style>"]
    for s in chosen:
        out.append(f'<section class="suite" id="suite-{s}"><h2>{SUITE_TITLES[s]}</h2>{body[s](results, arms) or "<p>No results.</p>"}'
                   f"{_unrun(results, s)}</section>")
    return "\n".join(out)


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
    if results.get("untouched"):
        iso = "<p>Real Strata install untouched: no file changed during the run.</p>"
    else:
        iso = "<p>Files of the real Strata install changed during the run:</p><ul>" + "".join(
            f"<li><code>{escape(str(c))}</code></li>" for c in results.get("changed") or []) + "</ul>"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>{CSS}</style></head><body>
{card}
<main><label>Base version <select id="base">{opts}</select></label>
<span> · decode and prompt in tok/s (median over rounds), setup and start in seconds · {escape(str(results.get('created', '')))}</span>{suite_head(results)}
<table><thead><tr>{heads}</tr></thead><tbody>
{_rows(results)}
</tbody></table>{iso}{suite_sections(results)}</main>
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


def render_png(browser: str, html: Path, png: Path, anchor: str = "card") -> bool:
    # A private profile, so a running Edge/Chrome cannot take over the URL and exit without the screenshot.
    # Edge on Windows sometimes exits before the file is written, or not at all on the first try: wait a moment, try twice.
    for _attempt in range(2):
        with tempfile.TemporaryDirectory(prefix="strata-ab-browser-", ignore_cleanup_errors=True) as profile:
            cmd = [browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1600,900",
                   f"--user-data-dir={profile}", f"--screenshot={png}", html.resolve().as_uri() + "#" + anchor]
            try:
                subprocess.run(cmd, capture_output=True, timeout=120)
            except (OSError, subprocess.SubprocessError):
                continue
            for _ in range(20):
                if png.is_file():
                    return True
                time.sleep(0.5)
    return False


def write_report(results: dict, out_dir: Path) -> tuple[Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    html = out_dir / "report.html"
    html.write_text(build_html(results), encoding="utf-8")
    cards = [("card", "report.png")] if card_pair(results) else []
    if "ladder" in (results.get("suites") or []) and any(_correct(a, "ladder")[1] for a in results.get("arms", [])):
        cards.append(("ladder", "report-ladder.png"))
    if not cards:
        return html, None
    browser = find_browser()
    if not browser:
        print("No Edge, Chrome or Chromium found: wrote report.html only, no report.png.", file=sys.stderr)
        return html, None
    card_png = None
    for anchor, name in cards:
        if render_png(browser, html, out_dir / name, anchor):
            card_png = out_dir / name if anchor == "card" else card_png
        else:
            print(f"The browser could not take the screenshot: wrote report.html only, no {name}.", file=sys.stderr)
    return html, card_png
