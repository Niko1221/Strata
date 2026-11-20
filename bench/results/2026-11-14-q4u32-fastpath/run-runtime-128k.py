# A.28 des Branchs feature_q4u32_as_an_int8_replacement_for_all: die 128K-Messung.
# Vorregistrierung: bench/results/2026-11-14-q4u32-fastpath/README.md, Abschnitt A.28.
#
# Was gemessen wird: die Prompt-Read-Rate des Engine-Logs fuer einen 128K-Nadel-Prompt
# (Tiefen 10/50/90 %), je Arm eine Engine, 30 s zwischen den Engine-Starts, ein GPU-Job zur
# Zeit.  Der Harness ist der unveraenderte terminal/tools/needle_bench.py, der Server der
# unveraenderte terminal/serve/server.py.
#
# Die Gates, die dieser Treiber selbst entscheiden kann:
#   G-R8  das VRAM-Budget des Prozesses (>= 14.000 MiB), die Zahl der Experten-Slots (>= 1.000)
#         und keine Paging-Warnung - sonst ist der Arm ungueltig (nicht langsam);
#   G-R6  jeder Arm muss "the prompt attention on matrix cores on" zeigen und darf die Zeile
#         nicht mit "STRATA_HIP_WMMA:" beginnen - der Schalter ist nicht gesetzt, der Default
#         des Builds laeuft;
#   R    der Harness muss alle drei Nadeln finden (er prueft das und endet mit Code 1, wenn nicht).
# Die inhaltlichen Kriterien (E1, E2, E3) entscheidet der Auswerfer, nicht der Treiber.
#
# Die Rohdaten (Logs, needles-JSON) liegen ausserhalb des Repos unter
# C:\\Users\\krone\\git\\Strata-data\\bench\\2026-11-14-q4u32-fastpath\\.
import hashlib
import json
import pathlib
import re
import signal
import subprocess
import sys
import time
import urllib.request

REPO = pathlib.Path(r"C:\Users\krone\git\q4u32_kv_quant_default_pr")   # der Baum unter Pruefung
PY = r"C:\Users\krone\git\kv-quant-strata-0.2\.venv\Scripts\python.exe"   # das Venv mit regex
CHECKOUT = pathlib.Path(r"C:\Users\krone\git\q4u32-merge-check-0217")    # der Baum mit dem gebauten Motor
DATA = pathlib.Path(r"C:\Users\krone\git\Strata-data")
OUT = DATA / "bench" / "2026-11-14-q4u32-fastpath"
LOGS = OUT / "logs"
URL = "http://127.0.0.1:7603"
PORT = "7603"

# (Tag, Config, erwartete 4-Bit-Route)
ARMS = [
    ("int8-2500", "needle-fastpath-128k-int8-2500.json", False),
    ("q4u32-2500", "needle-fastpath-128k-q4u32-2500.json", True),
    ("int8-auto", "needle-fastpath-128k-int8-auto.json", False),
    ("q4u32-auto", "needle-fastpath-128k-q4u32-auto.json", True),
]

RATE = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\)")
BUDGET = re.compile(r"Windows budgets (\d+) of this card's (\d+) MiB")
SLOTS = re.compile(r"expert cache (\d+) slots|-> (\d+) slots")
PAGING = re.compile(r"only (\d+) MiB of VRAM free")
# die Route-Zeilen des Motors
WMMA_ON = re.compile(r"the prompt attention on matrix cores on")
WMMA_SET = re.compile(r"STRATA_HIP_WMMA: the prompt attention")
Q4_ROUTE = re.compile(r"the matrix-core prompt path serves the 4-bit uniform grid")

MIN_BUDGET_MIB = 14000
MIN_SLOTS = 1000


def resolve(value):
    """Die Platzhalter der Configs mit echten Pfaden fuellen (die Repo-Configs tragen sie absichtlich).

    Die Ersetzung laeuft auf den Werten der JSON-Struktur, nicht auf dem Text: sonst kaemen die
    Backslashes der Pfade unescaped in die JSON.
    """
    if isinstance(value, str):
        return value.replace("<strata-checkout>", str(CHECKOUT)).replace("<strata-data>", str(DATA))
    if isinstance(value, list):
        return [resolve(v) for v in value]
    if isinstance(value, dict):
        return {k: resolve(v) for k, v in value.items()}
    return value


def slot_count(text: str) -> int:
    m = SLOTS.search(text)
    if not m:
        return -1
    return int(m.group(1) or m.group(2))


def last_run_text(path: pathlib.Path) -> str:
    """Der Text des LETZTEN Engine-Laufs in der Logdatei (der Motor haengt an)."""
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    i = text.rfind("engine started")
    return text[i:] if i >= 0 else text


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wait_up(seconds=1800) -> bool:
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            with urllib.request.urlopen(URL + "/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except OSError:
            time.sleep(3)
    return False


def stop(server) -> None:
    if server.poll() is None:
        server.send_signal(signal.CTRL_BREAK_EVENT)
        try:
            server.wait(timeout=180)
        except subprocess.TimeoutExpired:
            server.kill()


def strata_running() -> bool:
    """Das Ein-GPU-Job-Gate: eine Engine zur Zeit."""
    try:
        raw = subprocess.run(["tasklist", "/FI", "IMAGENAME eq strata.exe"], capture_output=True).stdout
    except OSError:
        print("tasklist failed; das Ein-GPU-Job-Gate kann nicht geprueft werden", flush=True)
        return True
    out = raw.decode("utf-8", errors="replace")
    return any(line.lower().lstrip().startswith("strata.exe") for line in out.splitlines())


def wait_engine_gone(seconds=300) -> bool:
    waited = 0
    while strata_running() and waited < seconds:
        time.sleep(5)
        waited += 5
    return not strata_running()


def write_config(cfg_name: str) -> pathlib.Path:
    """Die Repo-Config mit echten Pfaden in den Datenordner schreiben (die Rohdaten bleiben draussen)."""
    src = REPO / "bench" / "configs" / cfg_name
    cfg = resolve(json.loads(src.read_text(encoding="utf-8")))
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / cfg_name
    target.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return target


def start_server(cfg_path: pathlib.Path):
    return subprocess.Popen(
        [PY, str(REPO / "terminal" / "serve" / "server.py"), "--engine", "strata",
         "--config", str(cfg_path), "--port", PORT, "--no-api-key"],
        cwd=str(REPO / "terminal"), creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)


def vram_probe() -> bool:
    """Ein kurzer Start nur fuer das VRAM-Budget (der erste Start eines Laufs sieht zu wenig)."""
    cfg = write_config("needle-fastpath-128k-int8-2500.json")
    for attempt in (1, 2, 3):
        print(f"=== VRAM-Probe {attempt}: kurzer Start, nur das Budget lesen ===", flush=True)
        server = start_server(cfg)
        bud = -1
        if wait_up():
            text0 = last_run_text(LOGS / "a28-int8-2500.engine.log")
            m = BUDGET.search(text0)
            bud = int(m.group(1)) if m else -1
        stop(server)
        wait_engine_gone()
        print(f"VRAM-Probe {attempt}: Budget {bud} MiB (need >= {MIN_BUDGET_MIB})", flush=True)
        if bud >= MIN_BUDGET_MIB:
            time.sleep(30)
            return True
        time.sleep(90)
    return False


def arm_once(tag: str, cfg_name: str, q4_route: bool) -> int:
    cfg_path = write_config(cfg_name)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    exe = cfg["exe"]
    print(f"=== arm {tag}: config {cfg_name} ===", flush=True)
    print(f"arm {tag}: engine {exe} sha256 {sha256(exe)}", flush=True)
    print(f"arm {tag}: env {cfg['env']}", flush=True)
    if strata_running():
        print(f"arm {tag}: eine strata.exe laeuft schon; der Start wird verweigert (ein GPU-Job zur Zeit)", flush=True)
        return 1
    print(f"arm {tag}: starting server", flush=True)
    server = start_server(cfg_path)
    rc = 0
    if not wait_up():
        print(f"arm {tag}: der Server kam nicht hoch", flush=True)
        stop(server)
        return 1
    log = LOGS / f"a28-{tag}.engine.log"
    text0 = last_run_text(log)
    m = BUDGET.search(text0)
    bud = int(m.group(1)) if m else -1
    slots = slot_count(text0)
    page = PAGING.search(text0)
    print(f"arm {tag}: VRAM budget {bud} MiB (need >= {MIN_BUDGET_MIB}), expert cache {slots} slots "
          f"(need >= {MIN_SLOTS})", flush=True)
    if bud < MIN_BUDGET_MIB or slots < MIN_SLOTS or page:
        print(f"arm {tag}: G-R8 nicht erfuellt"
              + (f" (Paging-Warnung: {page.group(0)})" if page else "")
              + " - der Arm ist ungueltig, nicht langsam.  Stop.", flush=True)
        stop(server)
        wait_engine_gone()
        time.sleep(30)
        return -1
    print(f"arm {tag}: server up, running needle_bench (128k x depths 10/50/90)", flush=True)
    t0 = time.time()
    done = subprocess.run([PY, str(REPO / "terminal" / "tools" / "needle_bench.py"), "--url", URL,
                           "--lengths", "128k", "--depths", "10,50,90", "--timeout", "3600",
                           "--out", str(OUT / f"needles-a28-{tag}.json")], cwd=str(REPO))
    print(f"arm {tag}: needle_bench exit {done.returncode}, wall {int(time.time() - t0)} s", flush=True)
    stop(server)
    if not wait_engine_gone():
        print(f"arm {tag}: der Motor blieb nach dem Server-Ende laufen", flush=True)
        rc = rc or 1

    text = last_run_text(log)
    wmma = bool(WMMA_ON.search(text))
    wmma_set = bool(WMMA_SET.search(text))
    q4 = bool(Q4_ROUTE.search(text))
    rates = [(int(m.group(1)), float(m.group(4)) / 1000.0, float(m.group(5)))
             for m in RATE.finditer(text)]
    print(f"arm {tag}: Matrix-Core-Promptpfad: {'an' if wmma else 'AUS'}"
          f"{' (gesetzt, nicht erlaubt)' if wmma_set else ''} (erwartet an, per Default)", flush=True)
    print(f"arm {tag}: 4-Bit-Route: {'gezeigt' if q4 else 'nicht gezeigt'} (erwartet "
          f"{'gezeigt' if q4_route else 'nicht gezeigt'})", flush=True)
    print(f"arm {tag}: prompt read lines: {len(rates)}", flush=True)
    for n, secs, tps in rates:
        print(f"    prompt {n} tokens: {secs:.1f} s  ({tps:.1f} tok/s)", flush=True)
    if not wmma or wmma_set:
        rc = rc or 1
    if q4 != q4_route:
        rc = rc or 1
    if len(rates) != 3:
        rc = rc or 1
    if done.returncode != 0:
        rc = rc or 1
    if log.exists():
        print(f"arm {tag}: engine log sha256 {sha256(log)}", flush=True)
    time.sleep(30)   # die Kampagnenregel: 30 s zwischen den Engine-Starts
    return rc


def arm(tag: str, cfg_name: str, q4_route: bool) -> int:
    for attempt in (1, 2):
        rc = arm_once(tag, cfg_name, q4_route)
        if rc != -1:
            return rc
        print(f"arm {tag}: Versuch {attempt} hat G-R8 verfehlt; 90 s Wartezeit, dann ein zweiter Versuch", flush=True)
        time.sleep(90)
    print(f"arm {tag}: G-R8 nach zwei Versuchen nicht erfuellt - der Arm bleibt ungueltig", flush=True)
    return 1


LOGS.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
want = set(sys.argv[1:])
if not want or "noprobe" not in want:
    if not vram_probe():
        print("=== das VRAM-Budget kam nie ueber die Schwelle; die Arme waeren ungueltig - Abbruch ===", flush=True)
        sys.exit(1)
rc = 0
for tag, cfg_name, q4_route in ARMS:
    if want and tag not in want and "noprobe" not in want:
        continue
    rc = arm(tag, cfg_name, q4_route) or rc

print("=== SUMMARY ===", flush=True)
print(f"{'arm':>12}  {'prompt tok/s (10/50/90 %)':>28}  {'Median':>8}  {'WMMA':>5}  {'4-Bit':>6}  "
      f"{'VRAM MiB':>9}  {'Slots':>6}  {'Nadeln':>7}", flush=True)
for tag, _cfg, _q4 in ARMS:
    p = OUT / f"needles-a28-{tag}.json"
    log = LOGS / f"a28-{tag}.engine.log"
    text = last_run_text(log)
    rates = [float(m.group(5)) for m in RATE.finditer(text)]
    med = sorted(rates)[len(rates) // 2] if rates else float("nan")
    bud = BUDGET.search(text)
    cells = " ".join(f"{v:7.1f}" for v in rates) if rates else "-"
    found = "-"
    if p.exists():
        rows = json.loads(p.read_text(encoding="utf-8"))
        found = f"{sum(1 for r in rows if r.get('found'))}/{len(rows)}"
    print(f"{tag:>12}  {cells:>28}  {med:8.1f}  "
          f"{'an' if WMMA_ON.search(text) else 'aus':>5}  "
          f"{'ja' if Q4_ROUTE.search(text) else 'nein':>6}  "
          f"{(bud.group(1) if bud else '-'):>9}  {slot_count(text):>6}  recall {found}", flush=True)
sys.exit(rc)
