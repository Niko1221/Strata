"""Keep the bound fork release gate plus upstream instance-level diagnostic APIs."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("H:/Strata/evidence/upstream-sync-20261116/spill-resolution.json")


def source(ref: str) -> str:
    return subprocess.check_output([
        "git", "-C", str(ROOT), "show", ref + ":tools/check_cuda_spills.py",
    ]).decode("utf-8").replace("\r\n", "\n")


def function(text: str, name: str) -> str:
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return "".join(text.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])


def change(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError("Reviewed spill merge anchor changed: " + old[:60])
    return text.replace(old, new, 1)


def main() -> None:
    if OUT.exists():
        raise ValueError("Retain spill resolution receipt")
    ours, upstream = source("6f13c755"), source("09b603886")
    measured_check = function(ours, "check").replace("def check(", "def check_measured(", 1)
    merged_main = function(ours, "main")
    merged_main = change(merged_main, "    a = ap.parse_args()\n", '''    ap.add_argument('--known-baseline', help='upstream per-instance diagnostic reference; not a bound release gate')
    ap.add_argument('--print-baseline', action='store_true', help='emit per-instance measurements; no verdict')
    ap.add_argument('-v', '--verbose', action='store_true')
    a = ap.parse_args()
    if (a.known_baseline or a.print_baseline) and (a.baseline or a.capture_baseline or a.list):
        ap.error('known-instance diagnostics cannot replace the source-bound gate or capture')
    if a.known_baseline and a.print_baseline:
        ap.error('select one diagnostic mode')
    known = None
    if a.known_baseline:
        try:
            known = load_baseline(a.known_baseline)
        except ValueError as error:
            print('check_cuda_spills: ' + str(error))
            return 2
''')
    merged_main = change(merged_main, "    rc = 0\n", "    rc = 0\n    found, seen = {}, {}\n    archs_seen = set()\n")
    merged_main = change(
        merged_main,
        "        else:\n            rc = max(rc, check(lib, rows, names, baseline))\n",
        '''        elif a.print_baseline:
            for _, instance, sm, stack, local, _, _ in find_violations(rows, names)[2]:
                old = found.setdefault(sm, {}).get(instance, (0, 0))
                found[sm][instance] = (max(stack, old[0]), max(local, old[1]))
        elif known is not None:
            archs_seen.update(sm for _, sm, _, _ in rows)
            rc = max(rc, check(lib, rows, names, known, seen, a.verbose))
        else:
            rc = max(rc, check_measured(lib, rows, names, baseline))
''',
    )
    merged_main = change(merged_main, "    return rc", '''    if a.print_baseline:
        print(json.dumps({sm: {'measured_with': toolkit_line(cuobjdump),
                              'instances': [{'name': name, 'stack': stack, 'local': local, 'reason': KNOWN_REASON}
                                            for name, (stack, local) in sorted(per.items())]}
                          for sm, per in sorted(found.items())}, indent=2))
    elif known is not None:
        for sm in sorted(archs_seen & set(known)):
            print('check_cuda_spills: baseline %s: %d known violations seen, %d listed' %
                  (sm, len(seen.get(sm, ())), len(known[sm])))
        print('check_cuda_spills: diagnostic reference only; source-bound release qualification NOT_RUN')
    return rc''')
    output = upstream.replace(function(upstream, "main"), measured_check + "\n\n" + merged_main)
    imports = '''import hashlib
try:
    from . import cuda_spill_baseline as measured
except ImportError:
    import cuda_spill_baseline as measured
'''
    output = change(output, "import sys\n", "import sys\n" + imports)
    output = change(output, "Known violations (tools/cuda_spill_baseline.json):", '''The fork CLI's default gate retains ALLOWED and --baseline remains a source/compiler-bound measurement.
--capture-baseline requires explicit reviewed provenance and never constitutes a verdict. Upstream known-instance
APIs remain available through --known-baseline (diagnostic reference only) and --print-baseline (measurement only).

Known violations (tools/cuda_spill_baseline.json):''')
    ast.parse(output)
    (ROOT / "tools/check_cuda_spills.py").write_text(output, encoding="utf-8", newline="\n")
    tests = ROOT / "tools/test_check_cuda_spills.py"
    test_source = tests.read_text(encoding="utf-8")
    # The upstream diagnostic tests keep all assertions and now explicitly select
    # their mode; release-bound baseline tests remain in test_cuda_spill_baseline.
    test_source = test_source.replace('["--baseline", str(self.path)]', '["--known-baseline", str(self.path)]')
    test_source = test_source.replace('["--baseline", str(Path(self.d.name, "none.json"))]',
                                      '["--known-baseline", str(Path(self.d.name, "none.json"))]')
    tests.write_text(test_source, encoding="utf-8", newline="\n")
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump({"fork": hashlib.sha256(ours.encode()).hexdigest(),
                   "upstream": hashlib.sha256(upstream.encode()).hexdigest(),
                   "result": hashlib.sha256(output.encode()).hexdigest(),
                   "policy": "retain_bound_release_gate_diagnostics_explicit"}, stream, indent=2)


if __name__ == "__main__":
    main()
