"""Collect one test day's results (results/<DAY>/T0..T7) into
  REPORT.md      the community report in docs/COMMUNITY_BENCHMARKS.md's template (English, for upstream)
  SUMMARY_ja.md  a short Japanese summary (for EXPERIMENT_LOG.md)
A test that was not run is written as "not run"; fields only a person can know are marked TODO(operator).

    python3 summarize.py results/2026-10-20
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from pathlib import Path

DAY = Path(sys.argv[1]).resolve()
NR = "not run"


def latest(prefix: str) -> Path | None:
    """The last folder of a test (T2-speed, T2-speed-2, ... -> the highest number). T2-speed-q4 (STRATA_QUANT=q4) is
    another test: ask for it as latest("T2-speed-q4")."""
    c = [p for p in DAY.glob(prefix + "*") if p.is_dir() and re.fullmatch(re.escape(prefix) + r"(-\d+)?", p.name)]
    if not c:
        return None
    return max(c, key=lambda p: int(m.group(1)) if (m := re.search(r"-(\d+)$", p.name[len(prefix):])) else 1)


def read(p: Path | None, default=None):
    try:
        return p.read_text(encoding="utf-8") if p else default
    except OSError:
        return default


def jread(p: Path | None, default=None):
    t = read(p)
    try:
        return json.loads(t) if t else default
    except ValueError:
        return default


def env_line(env: str, header: str, n: int = 1) -> str:
    m = re.search(r"## " + re.escape(header) + r"\n((?:.*\n){%d})" % n, env or "")
    return m.group(1).strip() if m else "not recorded"


def results_table(readme: str | None) -> str:
    """The '## Results' table of a strata_bench README (the template's table)."""
    if not readme:
        return NR
    m = re.search(r"## Results\n\n(\|.*?\n)\n", readme, re.S)
    return m.group(1).strip() if m else "see the strata_bench README"


def med_range(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    return f"{statistics.median(xs):.1f} ({min(xs):.1f}-{max(xs):.1f})"


out, ja = [], []
t0, t1, t2, t3, t4, t5, t6, t7 = (latest(p) for p in
                                  ("T0-env", "T1-unit", "T2-speed", "T3-ab-numa", "T4-needle", "T5-quality",
                                   "T6-soak", "T7-quant-quality"))
env = read(t0 / "environment.txt", "") if t0 else ""
bench = jread(t2 / "warm" / next((p.name for p in (t2 / "warm").glob("*") if p.is_dir()), "") / "results.json") \
    if t2 and (t2 / "warm").exists() else None
mach = (bench or {}).get("machine", {})

out += [f"# Community benchmark on RTX 3060 12 GB with an AVX-only Xeon E5-2630 v2 (no AVX2/FMA), DDR3, PCIe Gen2 x4",
        "",
        f"Measured on {DAY.name} by TODO(operator). Strata built with `STRATA_ISA_FLOOR=avx` (docs/INSTALL.md, "
        "\"Older CPUs (experimental)\"), run on a real AVX-only CPU, which the docs list as not yet tested. "
        "Main limitation: TODO(operator).",
        "", "## Hardware and software", "",
        f"- GPU: {mach.get('gpu', 'NVIDIA GeForce RTX 3060')}, {mach.get('gpu_vram_mib', '12288')} MiB; "
        f"driver {mach.get('gpu_driver', 'see environment.txt')}",
        "- PCIe: " + env_line(env, "PCIe (sysfs) GPU and its root port", 2).replace("\n", "; "),
        "- GPU power limit: see `T0-env/environment.txt` (nvidia-smi query)",
        f"- CPU: 2 x Intel Xeon E5-2630 v2 (Ivy Bridge, 6 cores / 12 threads each, AVX + F16C, no AVX2/FMA), "
        "2 NUMA nodes; the engine bound to node 1 (the GPU's node) unless a test says otherwise",
        f"- RAM: {mach.get('ram_gib', '251.8')} GiB DDR3; read bandwidth measured 30.5 GB/s on node 1 "
        "(OpenMP stream read, 12 threads)",
        "- Storage: NVMe SSD (models and the n-gram table)",
        f"- OS: {mach.get('os', env_line(env, 'uname'))}",
        f"- Strata: commit {mach.get('strata_commit', '296b59d2')}, engine {mach.get('engine_version', '0.1.44.1')}, "
        "source build (CUDA 13.3 from pip wheels, g++-14, sm_86, `-DSTRATA_ISA_FLOOR=avx`); engine SHA-256 in "
        "`T0-env/environment.txt`",
        "- Background workloads: the GPU's other services were released for every test (a GPU lease); "
        "TODO(operator): anything else running",
        "- Network: disconnected during the tests", "",
        "## Model and configuration", "",
        "- ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, revision ed59f92, IQ2_XS "
        "(`Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-0000{1,2}-of-00002.gguf`); SHA-256 of every file in `manifest/`",
        "- Context 40,960; KV int8; expert cache auto; prefill auto; MTP spec 4 (min p 0.5); thinking off; "
        "temperature 0; setup's settings otherwise", "",
        "```text", (read(t2 / "config.json", "") if t2 else "").strip() or "see the config of each test", "```", "",
        "## Method", "",
        "- T1: `tools/test_*.py` one module at a time (`python -m unittest tools.<module>`), and the CTest suite of a "
        "separate build of the same source with `-DSTRATA_BUILD_TESTS=ON`",
        "- T2: `tools/strata_bench.py --long` (3 runs per test), twice: cold (model files dropped from the page cache "
        "with posix_fadvise DONTNEED, no sudo) and warm (started again right after)",
        "- T3: `tools/ab_engine.py`, arms both sockets / NUMA node 1, 3 rounds alternating",
        "- T4: `tools/needle_bench.py --lengths 4k,16k,32k --depths 10,50,90`",
        "- T5: the checks of the 2026-10-09 RTX 3060 report (HumanEval 0-39 with EvalPlus 0.3.1, 12 Japanese "
        "questions, 8 tool calls), same prompts and limits, non-streamed requests",
        "- T6: soak, back-to-back requests (every 4th a 17x19 check, others fresh ~512/4K/32K prompts, 512 tokens), "
        "GPU and RAM sampled every 10 s",
        "- T7 (optional): `tools/quant_quality.py` twice on IQ2_XS and once on UD-IQ4_XS", ""]

# ---- T2
out += ["## Results", "", "### Speed (T2)", ""]
if t2:
    for arm in ("cold", "warm"):
        d = t2 / arm
        rd = next((p for p in d.glob("*/README.md")), None) if d.exists() else None
        out += [f"**{arm}**", "", results_table(read(rd)), ""]
    dc = read(t2 / "drop-cache.txt")
    if dc:
        out += [f"Page cache before the cold run: {dc.strip()}", ""]
else:
    out += [NR, ""]

# ---- T3
out += ["### NUMA placement A/B (T3)", ""]
rows = [json.loads(x) for x in (read(t3 / "ab_engine.jsonl", "") or "").splitlines() if x.strip()] if t3 else []
if rows:
    out += ["| Request | Arm | Rounds | Decode tok/s median (range) | Prompt tok/s median (range) |",
            "| --- | --- | ---: | --- | --- |"]
    for req in sorted({r["req"] for r in rows}):
        for arm in sorted({r["arm"] for r in rows}):
            rs = [r for r in rows if r["req"] == req and r["arm"] == arm]
            out.append(f"| {req} | {arm} | {len(rs)} | {med_range([r.get('decode') for r in rs])} | "
                       f"{med_range([r.get('prompt') for r in rs])} |")
    out.append("")
else:
    out += [NR, ""]

# ---- correctness
out += ["## Correctness and limitations", "", "### Unit tests (T1)", ""]
if t1:
    py = read(t1 / "py-tests-result.txt", NR).strip()
    fails = [ln.split("\t") for ln in (read(t1 / "py-tests" / "summary.tsv", "") or "").splitlines()[1:]
             if ln.split("\t")[1] != "0"]
    ct = read(t1 / "ctest.log", "") or ""
    ctl = re.search(r"\d+% tests passed.*", ct)
    out += [f"- Python: {py}. Failing modules: " + (", ".join(f"`{f[0]}` ({f[3]})" for f in fails) or "none")
            + ". The same three modules fail on the unmodified upstream source at 296b59d on this host "
              "(checked 2026-10-16): `test_setup_choices`, `test_setup_engine_hash`, `test_setup_golden`.",
            f"- CTest: {ctl.group(0) if ctl else NR}"]
    failed = re.search(r"The following tests FAILED:\n((?:\s+.*\n)+)", ct)
    if failed:
        out += ["", "```text", failed.group(1).rstrip(), "```"]
    out.append("")
else:
    out += [NR, ""]

out += ["### Long-context recall (T4)", ""]
nd = jread(t4 / "needles.json") if t4 else None
if nd:
    out += ["| Length | Depth % | Prompt tokens | Found | Seconds |", "| --- | ---: | ---: | --- | ---: |"]
    out += [f"| {r.get('length')} | {r.get('depth')} | {r.get('prompt_tokens')} | {r.get('found', r.get('error', 'skipped'))} "
            f"| {r.get('seconds')} |" for r in nd]
    out += ["", f"Found {sum(bool(r.get('found')) for r in nd)} of {len(nd)}. A needle test measures recall on these "
                "inputs, not overall quality.", ""]
else:
    out += [NR, ""]

out += ["### Code, Japanese and tool checks (T5)", ""]
qs = jread(t5 / "summary.json") if t5 else None
ev = jread(t5 / "code" / "iq2_xs-samples-eval.json") if t5 else None
if qs:
    base = sum(r["base"]["status"] == "pass" for r in ev) if ev else None
    plus = sum(r["base_and_plus_pass"] for r in ev) if ev else None
    n = len(ev) if ev else 0
    out += ["| Machine | Code evaluated | Base pass | Base+plus pass | Japanese | Tool |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            f"| this machine (E5-2630 v2, AVX only) | {n} | {base}/{n} | {plus}/{n} | "
            f"{qs['japanese_pass']}/{qs['japanese_graded']} | {qs['tool_pass']}/{qs['tool_graded']} |",
            "| 2026-10-09 RTX 3060 report (i7-12700, AVX2) | 40 | 36/40 | 35/40 | 11/12 | 8/8 |", "",
            f"Code answers that reached the 1,024-token cap: {qs['code_at_cap']}. Request errors: {qs['request_errors']}. "
            "Differences from the 3060 report: context 40,960 here (131,072 for its code tasks, 8,192 for its small "
            "tasks), non-streamed requests, another CPU's rounding (IQ answers are not byte-reproducible across "
            "machines). HumanEval IDs 0-39 only, not the full 164-task score.", ""]
else:
    out += [NR, ""]

out += ["### Soak (T6)", ""]
sk = jread(t6 / "summary.json") if t6 else None
if sk:
    out += [f"- {sk['observed_minutes']} minutes, {sk['requests']} requests, {sk['failed']} failed; known-answer "
            f"checks {sk['checks_correct']}/{sk['checks']} correct",
            f"- GPU max temperature {sk.get('gpu_temp_max_c')} C, max power {sk.get('gpu_power_max_w')} W, max VRAM "
            f"{sk.get('vram_used_max_mib')} MiB; engine RSS first/last/max {sk.get('engine_rss_first_last_max_mib')} MiB; "
            f"lowest MemAvailable {sk.get('mem_available_min_gib')} GiB", "",
            "| Kind | Requests | Prompt tokens | Decode tok/s 1st half | 2nd half | Prompt tok/s 1st half | 2nd half |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for k, v in sk["by_kind"].items():
        out.append(f"| {k} | {v['n']} | {v['prompt_tokens_median']} | {v['decode_tps_first_half']} | "
                   f"{v['decode_tps_second_half']} | {v['prompt_tps_first_half']} | {v['prompt_tps_second_half']} |")
    out.append("")
else:
    out += [NR, ""]

out += ["### Quant yardstick (T7, optional)", ""]
if t7:
    out += ["```text", (read(t7 / "compare-repeat.txt", "") or NR).strip(), "```", "",
            "IQ2_XS against UD-IQ4_XS (not the official Q8_0 reference):", "",
            "```text", (read(t7 / "compare-vs-iq4.txt", "") or NR).strip(), "```", ""]
else:
    out += [NR, ""]

# ---- UD-IQ4_XS (STRATA_QUANT=q4): the same tests on the ~4-bit model, when they were run
q4 = {k: latest(k + "-q4") for k in ("T2-speed", "T3-ab-numa", "T4-needle", "T5-quality", "T6-soak")}
if any(q4.values()):
    out += ["### UD-IQ4_XS (~4-bit), same tests (`STRATA_QUANT=q4`)", "",
            "unsloth/Qwen3.8-Flash-Next-GGUF UD-IQ4_XS, context 40,960, node 1, otherwise as above. There is no "
            "RTX 3060 community row for this quantization to compare against.", ""]
    d = q4["T2-speed"]
    if d:
        for arm in ("cold", "warm"):
            rd = next((p for p in (d / arm).glob("*/README.md")), None) if (d / arm).exists() else None
            out += [f"**T2 {arm}**", "", results_table(read(rd)), ""]
    rows4 = [json.loads(x) for x in (read(q4["T3-ab-numa"] / "ab_engine.jsonl", "") or "").splitlines() if x.strip()] \
        if q4["T3-ab-numa"] else []
    if rows4:
        out += ["| T3 request | Arm | Rounds | Decode tok/s median (range) | Prompt tok/s median (range) |",
                "| --- | --- | ---: | --- | --- |"]
        for req in sorted({r["req"] for r in rows4}):
            for arm in sorted({r["arm"] for r in rows4}):
                rs = [r for r in rows4 if r["req"] == req and r["arm"] == arm]
                out.append(f"| {req} | {arm} | {len(rs)} | {med_range([r.get('decode') for r in rs])} | "
                           f"{med_range([r.get('prompt') for r in rs])} |")
        out.append("")
    nd4 = jread(q4["T4-needle"] / "needles.json") if q4["T4-needle"] else None
    if nd4:
        out += [f"- T4 needle: found {sum(bool(r.get('found')) for r in nd4)} of {len(nd4)}"]
    qs4 = jread(q4["T5-quality"] / "summary.json") if q4["T5-quality"] else None
    ev4 = jread(q4["T5-quality"] / "code" / "ud-iq4_xs-samples-eval.json") if q4["T5-quality"] else None
    if qs4:
        n4 = len(ev4) if ev4 else 0
        b4 = sum(r["base"]["status"] == "pass" for r in ev4) if ev4 else None
        p4 = sum(r["base_and_plus_pass"] for r in ev4) if ev4 else None
        out += [f"- T5: code base {b4}/{n4}, base+plus {p4}/{n4}; Japanese {qs4['japanese_pass']}/{qs4['japanese_graded']}; "
                f"tool {qs4['tool_pass']}/{qs4['tool_graded']}; answers at the 1,024-token cap {qs4['code_at_cap']}"]
    sk4 = jread(q4["T6-soak"] / "summary.json") if q4["T6-soak"] else None
    if sk4:
        out += [f"- T6 soak: {sk4['observed_minutes']} minutes, {sk4['requests']} requests, {sk4['failed']} failed; "
                f"checks {sk4['checks_correct']}/{sk4['checks']}; max VRAM {sk4.get('vram_used_max_mib')} MiB"]
    out.append("")

out += ["### Not tested", "",
        ("- Other quantizations than IQ2_XS and UD-IQ4_XS" if any(q4.values()) else
         "- Other quantizations than IQ2_XS (UD-IQ4_XS only in T7)") + "; Q2_0 cannot run without AVX-512 VNNI",
        "- Contexts above 40,960; images; `--batch` / concurrent clients; Windows",
        "- llama.cpp on the same machine", "- TODO(operator): anything skipped or stopped", ""]
(DAY / "REPORT.md").write_text("\n".join(out), encoding="utf-8")

# ---- Japanese summary
found = sum(bool(r.get("found")) for r in nd) if nd else 0
ja += [f"# Strataテスト結果 {DAY.name}（自動集計）", "",
       "| テスト | 結果 |", "|---|---|",
       f"| T1 単体 | {read(t1 / 'py-tests-result.txt', NR).strip() if t1 else NR}; ctest: "
       f"{(re.search(r'[0-9]+% tests passed.*', read(t1 / 'ctest.log', '') or '') or [NR])[0] if t1 else NR} |",
       f"| T2 速度 | {'REPORT.md の Speed を参照' if t2 else NR} |",
       f"| T3 NUMA A/B | {'REPORT.md の表を参照' if rows else NR} |",
       f"| T4 needle | {f'{found}/{len(nd)} 見つけた' if nd else NR} |",
       f"| T5 品質 | " + (f"HumanEval base {base}/{n}・plus {plus}/{n}、日本語 {qs['japanese_pass']}/{qs['japanese_graded']}、ツール {qs['tool_pass']}/{qs['tool_graded']}"
                         f"（3060報告: 36/40・35/40、11/12、8/8）" if qs else NR) + " |",
       f"| T6 soak | " + (f"{sk['observed_minutes']}分・{sk['requests']}件・失敗{sk['failed']}・確認問題 {sk['checks_correct']}/{sk['checks']}"
                          if sk else NR) + " |",
       f"| 4bit（UD-IQ4_XS） | " + ("REPORT.md の UD-IQ4_XS 節を参照（" + "、".join(k for k, v in q4.items() if v) + "）"
                                  if any(q4.values()) else NR) + " |",
       f"| T7 量子化品質 | {'REPORT.md を参照' if t7 else NR} |", ""]
(DAY / "SUMMARY_ja.md").write_text("\n".join(ja), encoding="utf-8")
print(f"wrote {DAY / 'REPORT.md'} and {DAY / 'SUMMARY_ja.md'}")
