"""Real isolated compiler corpus and hostile-input checks. Requires the optional worker."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.tex_math import render

fixtures = json.loads((ROOT / "serve/fixtures/math.json").read_text())
fixtures += [
    {"name": "file-read", "source": r"\input{/etc/passwd}", "display": False, "invalid": True},
    {"name": "lua-file-read", "source": r"\directlua{tex.print(io.open('/etc/passwd'):read('*a'))}", "display": False, "invalid": True},
    {"name": "lua-execute", "source": r"\directlua{os.execute('touch /tmp/escape')}x", "display": False, "invalid": True},
    {"name": "lua-loop", "source": r"\directlua{while true do end}", "display": False, "invalid": True},
]
results = []
for fixture in fixtures:
    start = time.perf_counter()
    result = render({"source": fixture["source"], "display": fixture["display"]})
    evidence = {"name": fixture["name"], "ok": result["ok"], "reason": result.get("reason"),
                "ms": round((time.perf_counter() - start) * 1000, 2)}
    results.append(evidence)
    print(json.dumps(evidence), flush=True)
    if fixture.get("invalid"):
        assert not result["ok"], fixture
    else:
        assert result["ok"], fixture
        assert "<script" not in result["svg"] and "<image" not in result["svg"]
Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tex-worker-results.json").write_text(json.dumps(results, indent=2) + "\n")
