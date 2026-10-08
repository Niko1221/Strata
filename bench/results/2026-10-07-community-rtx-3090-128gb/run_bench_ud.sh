#!/usr/bin/env bash
# Dedicated-GPU Strata community benchmark for Unsloth UD-IQ4_XS (single RTX 3090 24 GB + 128 GB RAM).
# The IQ3_S model is the user's daily model (strata.service); for this run we stop it, start the UD
# server manually on loopback:8080 (its config has no host key, so server.py binds 127.0.0.1), run the
# 4K/32K/128K sweep + needle recall, then kill the UD server and restart strata.service (IQ3_S) + neighbours.
# The trap restores on any exit. A contamination guard aborts if any request other than the harness's
# own 16 reaches the engine during the sweep.
set -euo pipefail

ROOT=/home/jfd/repositories/Strata
OUT="$ROOT/bench/results/2026-10-07-community-rtx-3090-128gb"
DED="$OUT/ud-iq4_xs"
PY="$ROOT/.venv/bin/python"
CFG="$ROOT/strata-unsloth-ud-iq4_xs.json"
URL=http://127.0.0.1:8080
PACK=/home/jfd/repositories/Strata-data/packs/unsloth-ud-iq4_xs
ENGINE_LOG="$ROOT/strata-unsloth-ud-iq4_xs.log"
MODEL_DIR=/mnt/nvme1/models/unsloth-UD-IQ4_XS
MMPROJ=/mnt/nvme1/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf
NEIGHBOURS="comfyui openclaw-gateway"
EXPECTED_REQUESTS=16            # 1 warmup + 9 speed + 6 needle
MON_PID=""
UD_PID=""
STOPPED_NEIGHBOURS=0
STOPPED_STRATA=0

log(){ printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

req_count(){ curl -s --max-time 5 "$URL/v1/status" | "$PY" -c "import sys,json;print(json.load(sys.stdin)['activity']['requests'])" 2>/dev/null || echo -1; }

port_free(){ ! curl -s --max-time 3 "$URL/v1/status" >/dev/null 2>&1; }

restore(){
  local rc=$?
  trap - EXIT INT TERM
  log "restore: kill UD server, restart strata.service (IQ3_S) + neighbours"
  if [ -n "$UD_PID" ]; then
    kill "$UD_PID" 2>/dev/null || true
    for _ in $(seq 1 24); do port_free && break; sleep 2; done
    kill -9 "$UD_PID" 2>/dev/null || true
  fi
  [ -n "$MON_PID" ] && kill "$MON_PID" 2>/dev/null || true
  if [ "$STOPPED_STRATA" = 1 ]; then
    systemctl --user start strata || log "WARN: could not start strata (IQ3_S)"
  fi
  if [ "$STOPPED_NEIGHBOURS" = 1 ]; then
    systemctl --user start $NEIGHBOURS || log "WARN: could not start neighbours"
  fi
  "$PY" - "$OUT/state-after.json" <<'PYEOF' || true
import json,subprocess,sys,pathlib,urllib.request
names=["strata","comfyui","openclaw-gateway"]
out=subprocess.run(["systemctl","--user","is-active",*names],capture_output=True,text=True).stdout.split()
d={"units":dict(zip(names,out))}
try: d["status"]=json.load(urllib.request.urlopen("http://127.0.0.1:8080/v1/status",timeout=10))
except Exception as e: d["status_error"]=str(e)
pathlib.Path(sys.argv[1]).write_text(json.dumps(d,indent=2)+"\n")
PYEOF
  log "restore done (exit $rc)"
  exit $rc
}
trap restore EXIT INT TERM

wait_loaded(){ for _ in $(seq 1 180); do curl -s --max-time 5 "$URL/v1/status" | grep -q '"loaded": true' && return 0; sleep 5; done; return 1; }
wait_idle(){ for _ in $(seq 1 24); do [ "$(curl -s --max-time 5 "$URL/v1/status" | "$PY" -c "import sys,json;print(json.load(sys.stdin)['activity']['in_flight'])" 2>/dev/null || echo 1)" = "0" ] && return 0; sleep 5; done; return 0; }

mkdir -p "$DED"

# 1. state before
log "snapshot state-before"
"$PY" - "$OUT/state-before-ud.json" <<'PYEOF'
import json,subprocess,sys,pathlib,hashlib,urllib.request
names=["strata","comfyui","openclaw-gateway"]
out=subprocess.run(["systemctl","--user","is-active",*names],capture_output=True,text=True).stdout.split()
d={"units":dict(zip(names,out))}
d["build_json"]=json.loads(pathlib.Path("/home/jfd/repositories/Strata/engine/BUILD.json").read_text())
d["iq3_s_config_sha256"]=hashlib.sha256(pathlib.Path("/home/jfd/repositories/Strata/strata-iq3_s.json").read_bytes()).hexdigest()
d["ud_config_sha256"]=hashlib.sha256(pathlib.Path("/home/jfd/repositories/Strata/strata-unsloth-ud-iq4_xs.json").read_bytes()).hexdigest()
try: d["engine_version"]=json.load(urllib.request.urlopen("http://127.0.0.1:8080/v1/status",timeout=10)).get("engine")
except Exception: d["engine_version"]=None
pathlib.Path(sys.argv[1]).write_text(json.dumps(d,indent=2)+"\n")
PYEOF

# 2. go dedicated: stop neighbours + strata (IQ3_S) to free VRAM/RAM
LOG_OFFSET=$(stat -c %s "$ENGINE_LOG" 2>/dev/null || echo 0)
log "stop neighbours + strata (IQ3_S); engine.log offset $LOG_OFFSET"
systemctl --user stop $NEIGHBOURS
STOPPED_NEIGHBOURS=1
systemctl --user stop strata
STOPPED_STRATA=1
for _ in $(seq 1 24); do port_free && break; sleep 2; done

# 3. start the UD server manually on loopback:8080 (config has no host key -> 127.0.0.1)
log "start UD server (loopback:8080)"
nohup "$PY" "$ROOT/serve/server.py" --engine strata --config "$CFG" --port 8080 > "$OUT/ud-server.out" 2>&1 &
UD_PID=$!
echo "$UD_PID" > "$OUT/ud-server.pid"
wait_loaded || { log "ERROR: UD server did not load within 900 s"; exit 1; }
log "engine version now: $(curl -s --max-time 5 "$URL/v1/status" | grep -oE '"engine": *"[^"]*"')"
log "model now: $(curl -s --max-time 5 "$URL/v1/status" | grep -oE '"model": *"[^"]*"')"
wait_idle

# 4. measure, with a contamination guard around the sweep
R0=$(req_count); log "request count before sweep: $R0"
"$PY" "$OUT/monitor2.py" "$DED/telemetry.jsonl" &
MON_PID=$!
log "benchmark sweep (4096,32768,128000 x 3)"
"$PY" "$OUT/benchmark.py" --root "$ROOT" --pack "$PACK" --url "$URL" --out "$DED" --targets 4096,32768,128000 --runs 3
log "needle recall (32k,128k x depths 10,50,90)"
"$PY" "$ROOT/tools/needle_bench.py" --url "$URL" --lengths 32k,128k --depths 10,50,90 --out "$DED/needles.json"
kill "$MON_PID" 2>/dev/null || true; MON_PID=""
R1=$(req_count); log "request count after sweep: $R1 (expected +$EXPECTED_REQUESTS)"
"$PY" - "$DED/contamination.json" "$R0" "$R1" "$EXPECTED_REQUESTS" <<'PYEOF'
import json,sys
r0,r1,exp=map(int,sys.argv[2:5]); foreign=(r1-r0)-exp
json.dump({"requests_before":r0,"requests_after":r1,"expected":exp,"foreign":foreign,"clean":foreign==0},
          open(sys.argv[1],"w"),indent=2)
print("contamination:",json.dumps({"delta":r1-r0,"expected":exp,"foreign":foreign,"clean":foreign==0}))
PYEOF
[ "$((R1-R0))" = "$EXPECTED_REQUESTS" ] || { log "ERROR: contamination detected (foreign $((R1-R0-EXPECTED_REQUESTS))); discarding run"; exit 1; }
log "extract engine.log slice"
tail -c "$((LOG_OFFSET+1))" "$ENGINE_LOG" > "$DED/engine.log" 2>/dev/null || true

# 5. memory summary (engine PSS + page cache, not just MemTotal-MemAvailable)
"$PY" - "$DED/telemetry.jsonl" "$DED/memory-summary.json" <<'PYEOF'
import json,sys,pathlib
rows=[json.loads(l) for l in pathlib.Path(sys.argv[1]).read_text().splitlines() if l.strip()]
def gu(r):
    try: return int(r["gpu"].split(",")[0].strip())
    except Exception: return None
def peak(fn):
    v=[fn(r) for r in rows]; v=[x for x in v if x is not None]; return max(v) if v else None
G=1048576.0
def sm(r,k):
    e=r.get("engine");  s=e.get("smaps") if e else None
    return s.get(k) if s else None
out={
 "samples":len(rows),
 "gpu_used_mib_peak":peak(lambda r:gu(r)),
 "ram_used_gib_peak":round(peak(lambda r:(r["memory"]["MemTotal_KiB"]-r["memory"]["MemAvailable_KiB"]) if "MemTotal_KiB" in r.get("memory",{}) else None)/G,2),
 "ram_raw_used_gib_peak":round(peak(lambda r:(r["memory"]["MemTotal_KiB"]-r["memory"]["MemFree_KiB"]) if "MemFree_KiB" in r.get("memory",{}) else None)/G,2),
 "cached_gib_peak":round(peak(lambda r:r["memory"].get("Cached_KiB"))/G,2),
 "shmem_gib_peak":round(peak(lambda r:r["memory"].get("Shmem_KiB"))/G,2),
 "engine_pss_gib_peak":round(peak(lambda r:sm(r,"Pss_KiB"))/G,2),
 "engine_pss_anon_gib_peak":round(peak(lambda r:sm(r,"Pss_Anon_KiB"))/G,2),
 "engine_pss_file_gib_peak":round(peak(lambda r:sm(r,"Pss_File_KiB"))/G,2),
 "swap_used_kib_first":(rows[0]["memory"].get("SwapTotal_KiB",0)-rows[0]["memory"].get("SwapFree_KiB",0)) if rows and rows[0].get("memory") else None,
 "swap_used_kib_peak":peak(lambda r:r["memory"].get("SwapTotal_KiB",0)-r["memory"].get("SwapFree_KiB",0) if r.get("memory") else None),
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(out,indent=2)+"\n"); print("memory-summary:",json.dumps(out))
PYEOF

# 6. provenance: shard SHA-256 from setup's verified .done files + mmproj hash
log "record provenance (shard hashes from .done, hash mmproj)"
"$PY" - "$MODEL_DIR" "$MMPROJ" "$OUT/model-provenance-ud-iq4_xs.json" <<'PYEOF'
import hashlib,json,sys,pathlib
mdir=pathlib.Path(sys.argv[1]); mmproj=pathlib.Path(sys.argv[2])
out={"model_dir":str(mdir),"files":{}}
for p in sorted(mdir.glob("*.gguf")):
    done=p.with_suffix(p.suffix+".done")
    sha=None
    if done.exists():
        line=done.read_text().strip()
        sha=line.split()[-1] if line.split()[-1] else None
    out["files"][p.name]={"bytes":p.stat().st_size,"sha256":sha,"source":"setup .done"}
h=hashlib.sha256(); n=0
with mmproj.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1<<22), b""): h.update(chunk); n+=len(chunk)
out["files"][mmproj.name]={"bytes":n,"sha256":h.hexdigest(),"source":"hashed"}
pathlib.Path(sys.argv[3]).write_text(json.dumps(out,indent=2)+"\n")
PYEOF

# 7. scrubbed config for publication (loopback host)
"$PY" - "$CFG" "$OUT/strata-unsloth-ud-iq4_xs.json" <<'PYEOF'
import json,sys,pathlib
c=json.loads(pathlib.Path(sys.argv[1]).read_text()); c["host"]="127.0.0.1"
pathlib.Path(sys.argv[2]).write_text(json.dumps(c,indent=1)+"\n")
PYEOF

log "DONE"
