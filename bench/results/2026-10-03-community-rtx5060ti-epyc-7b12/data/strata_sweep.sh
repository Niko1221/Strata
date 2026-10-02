#!/usr/bin/env bash
# 参数扫描编排：读清单文件（可运行中追加），逐项起服务→测→杀，结果落 TSV。
# 清单格式（每行一项，| 分隔）：
#   label|max_context|额外引擎参数(空格分隔,可为空)|env键值(逗号分隔,可为空)|vision模式(none|gpu|cpu)
# 用法: strata_sweep.sh <清单文件> [起始序号]
set -u
cd /home/ai-agent/DSHW/strata-study || exit 1
ITEMS=$1
START=${2:-1}
BIN=/home/ai-agent/DSHW/engine-builds/v-strata-0.1.35-120a-u2-20261003/bin
W=/home/ai-agent/DSHW/work/strata-sweep
mkdir -p "$W"
TABLE=$W/sweep.tsv
[ -f "$TABLE" ] || printf 'idx\tlabel\tctx\tload_s\tdecode\tprefill\taccept\texpert_slots\tvram_free\tenc\tgpu_hits\tfile_blobs\tfile_mb\tkv_hit_pct\tkv_ram_mib\tresident_gib\tnotes\n' >> "$TABLE"
GGUF=/tank/models/unsloth/Qwen3.8-Flash-Next.gguf

free_gpu() {
  pid=$(ss -ltnpH 'sport = :8087' 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
  [ -n "$pid" ] && kill -TERM "-$pid" 2>/dev/null
  pkill -x strata 2>/dev/null; pkill -x strata-vision 2>/dev/null; pkill -x strata-vision-c 2>/dev/null
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$used" ] && [ "$used" -lt 600 ] && return 0
    sleep 2
  done
}

idx=0
while IFS='|' read -r label ctx extra envs vismode; do
  [ -z "${label:-}" ] && continue
  case "$label" in \#*) continue;; esac
  idx=$((idx+1))
  [ "$idx" -lt "$START" ] && continue
  echo "########## [$idx] $label (ctx=$ctx vision=$vismode) $(date '+%T')"
  python3 - "$label" "$ctx" "$extra" "$envs" "$vismode" "$W" <<'PY'
import json, sys
label, ctx, extra, envs, vismode, W = sys.argv[1:7]
BIN = "/home/ai-agent/DSHW/engine-builds/v-strata-0.1.35-120a-u2-20261003/bin"
GGUF = "/tank/models/unsloth/Qwen3.8-Flash-Next.gguf"
args = ["--pack", "/home/ai-agent/DSHW/work/strata-packs/unsloth-q4", "--native", GGUF, "--ple-gguf", GGUF,
        "--kv", "q4_0", "--mtp", "/home/ai-agent/DSHW/work/strata-data/mtp/rt",
        "--expert-profile", "/home/ai-agent/DSHW/strata-study/data/expert-profile.bin",
        "--spec", "4", "--spec-min-p", "0.5"]
if extra.strip():
    args += extra.split()
args += ["--max-context", str(ctx)]
cfg = {"exe": BIN + "/strata", "args": args, "cwd": "/home/ai-agent/DSHW/strata-study",
       "tokenizer": "/home/ai-agent/DSHW/work/strata-packs/unsloth-q4/tokenizer",
       "model_name": "qwen3.8-flash-next-ud-q4_k_xl",
       "log": f"{W}/{label}-engine.log", "host": "127.0.0.1", "port": 8087}
if envs.strip():
    cfg["env"] = dict(kv.split("=", 1) for kv in envs.split(","))
if vismode == "gpu":
    args.append("--vision")
    cfg["vision"] = {"exe": BIN + "/strata-vision", "mmproj": "/tank/models/unsloth/mmproj-Qwen3.8-FN.gguf",
                     "model": GGUF, "gpu": True, "max_tokens": 1024}
elif vismode == "cpu":
    args.append("--vision")
    cfg["vision"] = {"exe": BIN + "/strata-vision-cpu", "mmproj": "/tank/models/unsloth/mmproj-Qwen3.8-FN.gguf",
                     "model": GGUF, "gpu": False, "threads": 64, "max_tokens": 300}
json.dump(cfg, open(f"{W}/{label}.json", "w"), indent=2, ensure_ascii=False)
print("  args:", " ".join(args))
PY
  free_gpu
  : > "$W/$label-engine.log"
  t0=$(date +%s)
  setsid nohup python3 -m serve.server --engine strata --config "$W/$label.json" --host 127.0.0.1 --port 8087 \
      > "$W/server-$label.out" 2>&1 &
  ready=0
  for i in $(seq 1 200); do
    curl -s --max-time 3 http://127.0.0.1:8087/health 2>/dev/null | grep -q '"loaded": true' && { ready=1; break; }
    # 引擎已死就别再空等：启动失败（例如 expert-cache 超显存）会立刻 exit。
    # 用精确进程名 pgrep -x strata 判活，不做 cmdline 匹配；给前 18s 启动窗口。
    if [ "$i" -gt 6 ] && ! pgrep -x strata >/dev/null 2>&1; then
      echo "  ✗ 引擎进程已退出，提前结束等待(第${i}轮)"; break
    fi
    sleep 3
  done
  load=$(( $(date +%s) - t0 ))
  if [ "$ready" != "1" ]; then
    echo "  ✗ 未就绪(${load}s)"; tail -3 "$W/$label-engine.log"
    printf '%s\t%s\t%s\t%s\tFAILED\n' "$idx" "$label" "$ctx" "$load" >> "$TABLE"; continue
  fi
  echo "  ✓ 就绪 ${load}s → 测量"
  python3 /home/ai-agent/DSHW/work/strata_sweep_measure.py 8087 "$W/$label-engine.log" "$W/$label.res.json" \
      > "$W/$label.measure.out" 2>&1
  python3 - "$idx" "$label" "$ctx" "$load" "$W/$label.res.json" <<'PY' >> "$TABLE"
import json, sys
idx, label, ctx, load, p = sys.argv[1:6]
r = json.load(open(p)); e = r.get("engine", {}) or {}
et = r.get("expert_tiers", {}) or {}; kv = r.get("kv_stream", {}) or {}
row = [idx, label, ctx, load, r.get("decode_med"), r.get("prefill_med"), r.get("accept_med"),
       e.get("expert_slots"), e.get("vram_free_mib"), e.get("arena_mib") or r.get("resident_gib"),
       et.get("gpu_hits_this_req"), et.get("file_blobs"), et.get("file_mb_read"),
       kv.get("vram_hit_pct"), kv.get("ram_mib_read"), r.get("resident_gib"), ""]
print("\t".join(str(x) if x is not None else "" for x in row))
PY
  echo "  结果: $(tail -1 "$W/$label.measure.out" | cut -c1-220)"
  free_gpu
done < "$ITEMS"
echo "########## 扫描完成 $(date '+%F %T')"
column -t -s $'\t' "$TABLE"
