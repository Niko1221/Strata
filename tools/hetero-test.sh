#!/usr/bin/env bash
# tools/hetero-test.sh - the hetero multi-GPU test harness (Fase 14, pre-PR: for OTHER rigs).
#
# Runs the roles/batching/pipelining variants on YOUR rig, compares the greedy token streams
# bit-exactly, and writes OUT/report.md ready to paste into the upstream issue.
#
#   tools/hetero-test.sh BUILD_DIR "BASE_FLAGS" OUT_DIR [MAX_NEW] [REPEATS]
#     BUILD_DIR     directory holding `strata` (and `strata-device`)
#     BASE_FLAGS    the flags that load YOUR pack, quoted, in the model's config order, e.g.
#                   "--pack DIR --spec 4 --mtp DIR" (a native (IQ) pack: add --native/--ple-gguf/
#                   --prefill/--kv/--mmap-experts exactly as the engine demands)
#     OUT_DIR       where the logs, the probe tokens and report.md land
#     MAX_NEW       greedy tokens per run [60]  |  REPEATS per variant [2]
#   env: HETERO_TEST_TOKENS=95 HETERO_TEST_SEED=7 HETEROS_TEST_VOCAB=151936
#
# The roles' promise: PLACEMENT never changes the output, only the measured time.  So every
# compare is on the GREEDY token stream:
#   - a variant's repeats must be identical among themselves (deterministic per plan);
#   - the roles/batching variants must match the SAME-residency reference (equal expert slots -
#     forced `--expert-cache N` when the runs' autos differ, exactly the F13 gate rule);
#   - the AUTO-residency runs (no forced cache) are NOT compared for exactness - their differing
#     slot counts are the "is it worth it" signal (more cache on the role split = fewer CPU-pool
#     reads on expert-heavy workloads).
# Probe input = deterministic random token ids (the engine build has no tokenizer: ids ARE the input).
set -u

BUILD="${1:?build dir}"; BASE="${2:?base flags (quoted)}"; OUT="${3:?out dir}"
MAX_NEW="${4:-60}"; REPEATS="${5:-2}"
STRATA="$BUILD/strata"
[ -x "$STRATA" ] || { echo "no engine at $STRATA"; exit 2; }
mkdir -p "$OUT"

NT="${HETERO_TEST_TOKENS:-95}"; SEED="${HETERO_TEST_SEED:-7}"; VOCAB="${HETERO_TEST_VOCAB:-151936}"

# --- the greedy probe (deterministic; same ids every run - the variants must not move them) ---
if [ ! -f "$OUT/tokens.txt" ]; then
  python3 - "$OUT/tokens.txt" "$NT" "$SEED" "$VOCAB" <<'PY'
import random, sys
out, n, seed, vocab = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
random.seed(seed)
ids = [random.randrange(1000, max(2500, vocab - 2000)) for _ in range(n)]
open(out, "w").write(" ".join(map(str, ids)) + "\n")
PY
  echo "probe: $NT tokens, seed $SEED, vocab $VOCAB -> $OUT/tokens.txt"
fi

# --- the rig: device count and names from the engine's own report ---
DEVLINES=$("$STRATA-device" 2>/dev/null | grep -E "^device [0-9]+:" | sort -u)
NGPU=$(printf '%s\n' "$DEVLINES" | grep -c . || true)
case "$BASE" in *--mtp*) HAS_MTP=1 ;; *) HAS_MTP=0 ;; esac
echo "rig: $NGPU GPU(s); roles need --mtp: $HAS_MTP"

# --- the variants (name:base-flags); the forced-equal-residency run joins after the autos ran ---
run_one() {  # NAME EXTRA-FLAGS
  local name="$1" vflags="$2"
  for i in $(seq 1 "$REPEATS"); do
    local log="$OUT/run-$name-$i.log"
    # the flags are word-split on purpose (a tester's own BASE_FLAGS vocabulary)
    "$STRATA" $BASE --tokens-file "$OUT/tokens.txt" --max-new "$MAX_NEW" $vflags > "$log" 2>&1
    echo "V$name[$i] rc=$?  $(grep -a 'tokens per round' "$log" | tail -1)"
  done
}

V() {
  local n="$1"; shift
  VNAMES+=("$n"); VFLAGS+=("$*")
}

VNAMES=(); VFLAGS=()
V 0 ""

if [ "$NGPU" -ge 2 ] && [ "$HAS_MTP" = "1" ]; then
  V 1 "--main-device 0 --draft-device 1"
  V 2 "--main-device 1 --draft-device 0"
  V 3 "--main-device 0 --draft-device 1 --draft-prefill-parallel"
  V 4 "--main-device 0 --draft-device 1 --draft-chain-batch"
  V 5 "--main-device 0 --draft-device 1 --draft-prefill-parallel --draft-chain-batch"
  V 6 "--auto-roles"
elif [ "$HAS_MTP" = "1" ]; then
  V 7 "--draft-chain-batch"   # the batch works on ONE GPU too
fi

# --- run: V0/V1 first (an autos' reference), the rest after, the forced-equal-residency run last ---
for idx in "${!VNAMES[@]}"; do run_one "${VNAMES[$idx]}" "${VFLAGS[$idx]}"; done

# --- the exactness proof row: only when the autos' slot counts can differ and no forced cache is in BASE ---
FORCE_NAME=""; FORCE_FLAGS=""
if ! printf '%s' "$BASE" | grep -qE -- "--expert-cache[= ][0-9]+"; then
    if [ "$NGPU" -ge 2 ] && [ "$HAS_MTP" = "1" ]; then
      RES=$(grep -a "expert cache .* slots" "$OUT/run-0-1.log" | grep -oE "[0-9]+ slots" | head -1 | cut -d' ' -f1)
      if [ -n "${RES:-}" ]; then
        FORCE_NAME="1f"; FORCE_FLAGS="--main-device 0 --draft-device 1 --expert-cache $RES"
        echo "exactness row: V$FORCE_NAME = V1 at V0's residency ($RES slots)"
        run_one "$FORCE_NAME" "$FORCE_FLAGS"
        [ "$REPEATS" -gt 1 ] && run_one "$FORCE_NAME" "$FORCE_FLAGS"
      fi
    fi
fi

# --- the report ---
VNAMES+=("${FORCE_NAME:-}"); VFLAGS+=("${FORCE_FLAGS:-}")

python3 - "$OUT" "$REPEATS" <<'PY'
import hashlib, pathlib, re, sys
out = pathlib.Path(sys.argv[1]); repeats = int(sys.argv[2])
DUMP = re.compile(r"^\s*(prompt|output)\s*:\s*([0-9][0-9\s,]*)$")
def stream(text, kind):
    for line in text.splitlines():
        m = DUMP.match(line)
        if m and m.group(1) == kind:
            return [int(t) for t in re.split(r"[\s,]+", m.group(2).strip()) if t]
    return None
def grab(text, pat):
    m = re.search(pat, text)
    return m.group(1) if m else ""
TOKROUND = r"([\d.]+) tokens per round"     # hoisted: a backslash regex cannot sit inside an f-string
CACHE = r"expert cache (\d+) slots"

rows, streams, slots = [], {}, {}
for log in sorted(out.glob("run-*.log")):
    name, i = log.stem[4:].rsplit("-", 1)
    text = log.read_text(errors="replace")
    s = stream(text, "output")
    streams.setdefault(name, {})[i] = s
    n_slots = grab(text, r"expert cache (\d+) slots") or grab(text, r"-> (\d+) slots")
    if name not in slots and n_slots: slots[name] = n_slots
    rows.append((name, i, log.with_suffix("").name, stream(text, "prompt"),
                 n_slots, grab(text, r"speculation\s+(\S.*?tokens per round)"),
                 grab(text, r"mtp\s+([\d.]+) ms/round"), text.count("ERR ")))
names = sorted({r[0] for r in rows if r[0]}, key=lambda n: (n != "0", n))
def md5(s): return hashlib.md5(",".join(map(str, s)).encode()).hexdigest()[:12] if s else "-"

_rig = ""
for seed_log in ("run-1-1.log", "run-0-1.log"):   # the role lines print in the roles runs (V1 first)
    if (out / seed_log).exists():
        t = (out / seed_log).read_text(errors="replace")
        _rig = " | ".join(re.findall(r"role device \d+: ([^\n]+)", t))
        if _rig: break
rep = ["# Hetero multi-GPU test report (auto-generated by tools/hetero-test.sh)", "",
       f"rig: `{_rig or 'see logs'}`", ""]
rep.append("| variant | run | tokens | accepted / offered | tok/round | ms/round drafting | expert slots | stream md5 | rc/ERRs |")
rep.append("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    name, i = r[0], r[1]
    s = streams[name].get(i)
    acc = r[5].split(" drafts accepted ")[-1] if " drafts accepted " in r[5] else r[5]
    rep.append(f"| {name} | {i} | {len(s) if s else 0} | {acc} | {grab(r[5], TOKROUND) or '-'} "
               f"| {r[6] or '-'} | {r[4] or '-'} | {md5(s)} | {r[7]} |")
rep.append("")
rep.append("## bit-exactness verdicts (the promise: placement never changes the output)")
ref = "1f" if "1f" in streams else ("0" if "0" in streams else None)
ok = fail = 0
def cmp_streams(a_name, b_name, why):
    global ok, fail
    ref_s = streams.get(b_name, {}).get("1")
    for i in sorted(streams.get(a_name, {})):
        s = streams[a_name][i]
        refuse = f"{b_name}/1 ({why})"
        if ref_s is None or s is None:
            rep.append(f"- V{a_name}[{i}] vs {refuse}: **INCOMPLETE** (no stream)"); fail += 1
        elif s == ref_s:
            rep.append(f"- V{a_name}[{i}] vs {refuse}: **PASS** ({md5(s)})"); ok += 1
        else:
            first = next((k for k, (x, y) in enumerate(zip(s, ref_s)) if x != y), min(len(s), len(ref_s)))
            rep.append(f"- V{a_name}[{i}] vs {refuse}: **FAIL** (first difference at {first})"); fail += 1
# 1) repeats equal
for n in names:
    seen = {md5(streams[n][i]) for i in streams[n] if streams[n][i] is not None}
    if len(seen) > 1:
        rep.append(f"- V{n} repeats differ among themselves: **FAIL** (determinism)"); fail += 1
    elif len(seen):
        rep.append(f"- V{n} repeats identical: PASS"); ok += 1
# 2) each variant vs a reference of the SAME expert residency (the F13 rule): prefer the same-
#    placement auto reference (V1), then the baseline (V0), then the forced-residency V1f
for n in names:
    if n in ("0", "1f"): continue
    ref_n = None
    for cand in ("1", "0", "1f"):
        if cand == n or cand not in slots: continue
        if not slots.get(n) or slots[cand] == slots[n]: ref_n = cand; break
    if ref_n is None:
        rep.append(f"- V{n}: no same-residency reference available - SKIPPED by design "
                   "(the determinism check above still covers it; the auto slot count is the "
                   "placement's VRAM dividend)")
    else:
        cmp_streams(n, ref_n, f"(same expert residency, {slots[n]} slots, vs V{ref_n})")
# 3) the exactness proof row vs the baseline (equal residency)
f1f = slots.get("1f") or "?"
if "1f" in streams and "0" in streams: cmp_streams("1f", "0", f"V1 at V0's (--expert-cache {f1f})")
if "1" in streams and "0" in streams:
    s0 = grab((out/'run-0-1.log').read_text(errors="replace"), r"expert cache .* -> (\d+) slots") or grab((out/'run-0-1.log').read_text(errors="replace"), r"expert cache (\d+) slots")
    s1 = grab((out/'run-1-1.log').read_text(errors="replace"), r"expert cache (\d+) slots")
    if s0 and s1 and s0 != s1:
        rep.append(f"- NOTE: V0's auto residency ({s0} slots) != V1's auto residency ({s1} slots): the roles moved "
                   "the drafter off the main device and freed its VRAM for the expert cache - the runs are NOT "
                   "compared bit-exact (by design, the F13 rule); V1f above is the equal-residency proof.")
rep.append("")
verdict = f"**{ok} PASS / {fail} FAIL**" + (" - the roles are exact on this rig" if fail == 0 else " - INVESTIGATE (a difference is a finding, not a failure of the idea)")
rep.append(verdict)
rep.append("")
rep.append("Timings live in the table above; decode tok/s per run: see each log's last lines.  Paste this "
           "report (or attach it) into the issue - it is the rig's evidence.")
(out / "report.md").write_text("\n".join(rep) + "\n")
print("\n".join(rep[-6:]))
print(f"report: {out/'report.md'}")
PY
echo "logs: $OUT/run-*.log  report: $OUT/report.md"