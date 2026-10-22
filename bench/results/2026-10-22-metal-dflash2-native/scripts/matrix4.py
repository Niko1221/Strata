# matrix.py: every server config x every spec_min_p on the same workloads; one row per (config, p) in matrix.tsv
import json, os, re, subprocess, sys, time, urllib.request
ST = "<strata>"; B = "<data>/dflash2/bench"; TUNED = "<data>/dflash2/tuned/DFlash2-v2-Q8_0.gguf"
SHORT = ["Write a Python function that parses an ISO 8601 duration string like P3DT4H12M into seconds, with tests.",
         "Summarize the causes and consequences of the 1997 Asian financial crisis in about 300 words.",
         "Explain how a B-tree insert splits nodes, then give a short C++ implementation of insert for order 4."]
LONG = "```cpp\n" + open(ST + "/src/core/mtp.cpp").read()[:90000] + "\n```\n\n"
LQ = ["Rewrite the function record_top2 from the file above with a comment on every line. Output only code.",
      "Now list every environment variable the file reads and say in one line what each one does."]
EXE = "<strata>/build-native/strata"   # v4: both arms on the native-only build (main 9de04cd4 + PR 83774137)
def cfg(name, base, spec, arena, suffix=0, dflash=None):
    c = json.load(open(f"{ST}/{base}")); a = c["args"]; c["exe"] = EXE; c["env"].pop("STRATA_DFLASH_NATIVE", None)
    a[a.index("--spec") + 1] = str(spec); a[a.index("--suffix-draft") + 1] = str(suffix)
    if dflash: a[a.index("--dflash") + 1] = dflash
    c["env"]["STRATA_METAL_ARENA_GB"] = str(arena)
    p = f"{B}/cfg-{name}.json"; json.dump(c, open(p, "w"), indent=2); return p, c["log"]
M, D = "strata-iq3_s.json", "strata-dflash.json"
PM, PD = [0.5, 0.7, 0.85], [0.5, 0.6, 0.75, 0.85]
CONFIGS = [  # v4, same day, interleaved MTP / DFlash2 on one binary
    ("v4-mtp-s3-a42", M, 3, 42, 0, None, [0.7]), ("v4-df-s4-a42", D, 4, 42, 0, TUNED, [0.6, 0.75]),
    ("v4-df-s3-a42", D, 3, 42, 0, TUNED, [0.6]), ("v4-mtp-s3-a42-r", M, 3, 42, 0, None, [0.7]),
    ("v4-df-s4-a42-r", D, 4, 42, 0, TUNED, [0.6, 0.75]),
]
only = sys.argv[1:]  # optional config names
SEEN = [0]
def req(msgs, pm):
    SEEN[0] += others()
    b = json.dumps({"model": "x", "messages": msgs, "max_tokens": 600, "temperature": 0, "strata_tune": {"spec_min_p": pm}}).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8092/v1/chat/completions", b, {"Content-Type": "application/json"}), timeout=3600))
def last(log):
    l = [x for x in open(log) if "generated in" in x][-1]
    m = re.search(r"(\d+) generated in (\d+) ms.*drafts accepted (\d+) of (\d+)", l); return int(m[1]), int(m[2]), int(m[3])
def others():   # engines on this Mac besides ours (any --serve not started by swap4.sh)
    ps = subprocess.run(["ps", "-Ao", "command"], capture_output=True, text=True).stdout.splitlines()
    return sum(1 for l in ps if "--serve" in l and "strata" in l) - 1
out = open(f"{B}/matrix-v4.tsv", "a")
for name, base, spec, arena, sfx, dfl, plist in CONFIGS:
    if only and name not in only: continue
    path, log = cfg(name, base, spec, arena, sfx, dfl)
    r = subprocess.run([f"{B}/swap4.sh", path], capture_output=True, text=True)
    if "ready" not in r.stdout: out.write(f"{name}\tSTART FAILED\n"); out.flush(); continue
    swap0 = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split()[5]
    for pm in plist:
        n0 = others(); SEEN[0] = 0
        g = t = a = 0
        try:
            for p in SHORT:
                req([{"role": "user", "content": p}], pm); x = last(log); g += x[0]; t += x[1]; a += x[2]
            short = g / (t / 1000); spw = g / max(1, g - a)
            msgs = [{"role": "user", "content": LONG + LQ[0]}]; j = req(msgs, pm); c0 = last(log)
            msgs += [j["choices"][0]["message"], {"role": "user", "content": LQ[1]}]; req(msgs, pm); c1 = last(log)
            row = f"{name}\t{pm}\t{short:.1f}\t{spw:.2f}\t{c0[0]/(c0[1]/1000):.1f}\t{c1[0]/(c1[1]/1000):.1f}"
        except Exception as e:
            row = f"{name}\t{pm}\tERROR {e}"
        swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split()[5]
        if n0 or SEEN[0] or others(): row += "\tCONTAMINATED (another engine ran)"
        out.write(row + f"\tswap {swap0}->{swap}\n"); out.flush(); print(row, flush=True)
subprocess.run(["sh", "-c", f"kill $(cat {B}/swap4.pid)"])
