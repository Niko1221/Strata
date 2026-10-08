#!/usr/bin/env bash
# IQ3_S elastic-VRAM community benchmark: leave 10 GiB free for Flux/image generation.
# ComfyUI stays running idle; openclaw-gateway is stopped to keep the Strata request guard clean.
set -euo pipefail

ROOT=/home/jfd/repositories/Strata
OUT="$ROOT/bench/results/2026-10-07-community-rtx-3090-128gb"
DED="$OUT/iq3_s-10gb-elastic"
PY="$ROOT/.venv/bin/python"
CFG="$ROOT/strata-iq3_s.json"
URL=http://127.0.0.1:8080
PACK=/home/jfd/repositories/Strata-data/packs/iq3_s
ENGINE_LOG="$ROOT/strata-iq3_s.log"
NEIGHBOURS="openclaw-gateway"
RESERVE_MIB=10240
EXPECTED_REQUESTS=16
MON_PID=""
STOPPED_NEIGHBOURS=0
HOST_CHANGED=0
ORIG_HOST=""

log(){ printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

req_count(){ curl -s --max-time 5 "$URL/v1/status" | "$PY" -c "import sys,json;print(json.load(sys.stdin)['activity']['requests'])" 2>/dev/null || echo -1; }

restore(){
  local rc=$?
  trap - EXIT INT TERM
  log "restore: host binding + neighbours"
  if [ "$HOST_CHANGED" = 1 ] && [ -n "$ORIG_HOST" ]; then
    "$PY" -c "import json,sys;p='$CFG';c=json.load(open(p));c['host']=sys.argv[1];json.dump(c,open(p,'w'),indent=1)" "$ORIG_HOST" || log "WARN: host revert failed"
    systemctl --user restart strata || log "WARN: strata restart after host revert failed"
  fi
  if [ "$STOPPED_NEIGHBOURS" = 1 ]; then
    systemctl --user start $NEIGHBOURS || log "WARN: could not start neighbours"
  fi
  systemctl --user is-active --quiet strata || systemctl --user start strata || log "WARN: strata not active"
  [ -n "$MON_PID" ] && kill "$MON_PID" 2>/dev/null || true
  "$PY" - "$OUT/state-after-10gb.json" <<'PYEOF' || true
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

log "snapshot state-before"
"$PY" - "$OUT/state-before-10gb.json" <<'PYEOF'
import json,subprocess,sys,pathlib,hashlib,urllib.request
names=["strata","comfyui","openclaw-gateway"]
out=subprocess.run(["systemctl","--user","is-active",*names],capture_output=True,text=True).stdout.split()
d={"units":dict(zip(names,out))}
d["build_json"]=json.loads(pathlib.Path("/home/jfd/repositories/Strata/engine/BUILD.json").read_text())
d["config_sha256"]=hashlib.sha256(pathlib.Path("/home/jfd/repositories/Strata/strata-iq3_s.json").read_bytes()).hexdigest()
try: d["status"]=json.load(urllib.request.urlopen("http://127.0.0.1:8080/v1/status",timeout=10))
except Exception as e: d["status_error"]=str(e)
pathlib.Path(sys.argv[1]).write_text(json.dumps(d,indent=2)+"\n")
PYEOF

log "git HEAD: $(git -C "$ROOT" rev-parse --short HEAD) ($(git -C "$ROOT" describe --tags 2>/dev/null))"
cmake --build "$ROOT/build" -j"$(nproc)"
cmake --build "$ROOT/build-vision" -j"$(nproc)"
cp -f "$ROOT/build/strata" "$ROOT/engine/strata.tmp"; mv -f "$ROOT/engine/strata.tmp" "$ROOT/engine/strata"
cp -f "$ROOT/build-vision/bin/strata-vision" "$ROOT/engine/strata-vision.tmp"; mv -f "$ROOT/engine/strata-vision.tmp" "$ROOT/engine/strata-vision"
"$PY" - <<'PYEOF'
import json,pathlib,sys
sys.path.insert(0,"/home/jfd/repositories/Strata"); import setup
p=pathlib.Path("/home/jfd/repositories/Strata/engine/BUILD.json"); meta=json.loads(p.read_text())
meta.update(version=setup.source_version(), src=setup.source_hash(setup.ENGINE_SOURCES),
            vision_src=setup.source_hash(setup.VISION_SOURCES), source="local")
p.write_text(json.dumps(meta,indent=1)+"\n"); print("BUILD.json:",json.dumps(meta))
PYEOF

ORIG_HOST=$("$PY" -c "import json;print(json.load(open('$CFG')).get('host','0.0.0.0'))")
log "original host: $ORIG_HOST -> binding 127.0.0.1 for the run"
"$PY" -c "import json;p='$CFG';c=json.load(open(p));c['host']='127.0.0.1';json.dump(c,open(p,'w'),indent=1)"
HOST_CHANGED=1
LOG_OFFSET=$(stat -c %s "$ENGINE_LOG" 2>/dev/null || echo 0)
log "stop openclaw-gateway only; keep comfyui idle (engine.log offset $LOG_OFFSET)"
systemctl --user stop $NEIGHBOURS
STOPPED_NEIGHBOURS=1
log "restart strata (loopback, ComfyUI idle)"
systemctl --user restart strata
wait_loaded || { log "ERROR: strata did not load within 900 s"; exit 1; }
log "engine version now: $(curl -s --max-time 5 "$URL/v1/status" | grep -oE '"engine": *"[^"]*"')"
wait_idle

R0=$(req_count); log "request count before sweep: $R0"
log "reserve $RESERVE_MIB MiB for Flux/image generation"
"$PY" - "$URL" "$DED/vram-reserve.json" "$RESERVE_MIB" <<'PYEOF'
import json,sys,time,urllib.error,urllib.request
url,out,reserve=sys.argv[1],sys.argv[2],int(sys.argv[3])
last=None
for attempt in range(12):
    req=urllib.request.Request(url.rstrip('/')+'/v1/vram', data=json.dumps({"reserve_mib":reserve}).encode(),
                               headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            last=json.load(response)
            if last.get("status") == "ok":
                break
    except urllib.error.HTTPError as exc:
        last={"status":"http_error","code":exc.code,"body":exc.read().decode(errors="replace")[:1000]}
        if exc.code == 409:
            time.sleep(5)
            continue
        raise
    except Exception as exc:
        last={"status":"error","error":str(exc)}
        time.sleep(5)
        continue
    time.sleep(2)
pathlib=__import__("pathlib").Path
pathlib(out).write_text(json.dumps(last,indent=2)+"\n")
if not last or last.get("status") != "ok":
    raise SystemExit(f"vram reserve failed: {last}")
free=last.get("vram_free_mib")
if free is None or free < reserve - 512:
    raise SystemExit(f"vram_free_mib {free} is below reserve {reserve}")
print("vram-reserve:", json.dumps(last))
PYEOF

"$PY" "$OUT/monitor-elastic.py" "$DED/telemetry.jsonl" "$URL" &
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

curl -s --max-time 10 "$URL/v1/status" > "$DED/final-status.json" || true
curl -s --max-time 10 "$URL/metrics" > "$DED/final-metrics.json" || true
log "extract engine.log slice"
tail -c "$((LOG_OFFSET+1))" "$ENGINE_LOG" > "$DED/engine.log" 2>/dev/null || true

log "memory summary"
"$PY" - "$DED/telemetry.jsonl" "$DED/memory-summary.json" "$DED/vram-reserve.json" <<'PYEOF'
import json,sys,pathlib
rows=[json.loads(l) for l in pathlib.Path(sys.argv[1]).read_text().splitlines() if l.strip()]
reserve=json.loads(pathlib.Path(sys.argv[3]).read_text()) if pathlib.Path(sys.argv[3]).exists() else {}
def num(x):
    return x if isinstance(x,(int,float)) else None
gpu_used=[num(r.get("gpu",{}).get("used_mib")) for r in rows]
gpu_used=[x for x in gpu_used if x is not None]
gpu_free=[num(r.get("gpu",{}).get("total_mib",0))-num(r.get("gpu",{}).get("used_mib")) for r in rows if isinstance(r.get("gpu"),dict)]
gpu_free=[x for x in gpu_free if x is not None]
ram=[r["memory"]["MemTotal_KiB"]-r["memory"]["MemAvailable_KiB"] for r in rows if "MemTotal_KiB" in r.get("memory",{})]
swap=[r["memory"].get("SwapTotal_KiB",0)-r["memory"].get("SwapFree_KiB",0) for r in rows if r.get("memory")]
pss=[r["engine"]["smaps"]["Pss_KiB"] for r in rows if r.get("engine") and r["engine"].get("smaps") and "Pss_KiB" in r["engine"]["smaps"]]
out={
  "samples":len(rows),
  "gpu_used_mib_peak":max(gpu_used) if gpu_used else None,
  "gpu_free_mib_min":min(gpu_free) if gpu_free else None,
  "ram_used_gib_peak":round(max(ram)/1048576,2) if ram else None,
  "engine_pss_gib_peak":round(max(pss)/1048576,2) if pss else None,
  "swap_used_kib_first":swap[0] if swap else None,
  "swap_used_kib_peak":max(swap) if swap else None,
  "vram_reserve":{k:reserve.get(k) for k in ("reserve_mib","expert_slots","expert_slots_full","expert_cache_mib","expert_cache_full_mib","vram_free_mib","prompt_chunk")},
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(out,indent=2)+"\n")
print("memory-summary:",json.dumps(out))
PYEOF

if [ ! -s "$OUT/model-provenance.json" ]; then
  log "hash model files"
  "$PY" - "/mnt/nvme1/models/IQ3_S" "$OUT/model-provenance.json" <<'PYEOF'
import hashlib,json,sys,pathlib
d=pathlib.Path(sys.argv[1])
files=["Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf","Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf","mmproj-Qwen3.8-Flash-Next-BF16.gguf"]
out={}
for f in files:
    p=d/f; h=hashlib.sha256(); n=0
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1<<22), b""): h.update(chunk); n+=len(chunk)
    out[f]={"bytes":n,"sha256":h.hexdigest()}
pathlib.Path(sys.argv[2]).write_text(json.dumps({"model_dir":str(d),"files":out},indent=2)+"\n")
PYEOF
fi

"$PY" - "$CFG" "$DED/strata-iq3_s.json" <<'PYEOF'
import json,sys,pathlib
c=json.loads(pathlib.Path(sys.argv[1]).read_text()); c["host"]="127.0.0.1"
pathlib.Path(sys.argv[2]).write_text(json.dumps(c,indent=1)+"\n")
PYEOF

log "DONE"
