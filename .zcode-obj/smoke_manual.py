import subprocess, sys, time, hashlib, io
sys.path.insert(0, "F:/Strata-dev/tools")
import grammar_acceptance as G

proc = subprocess.Popen([G.RIG["exe"], "--serve", *G.engine_args(8192)],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=open("F:/Strata-dev/.zcode-smoke-err.log", "w", encoding="utf-8"),
                        cwd="F:/Strata-dev")
proc.stdin = io.TextIOWrapper(proc.stdin, encoding="utf-8", newline="\n")
proc.stdout = io.TextIOWrapper(proc.stdout, encoding="utf-8")
sha = hashlib.sha256(b'root ::= "ab"').hexdigest()[:16]
sent = []
def send(line):
    sent.append(line)
    proc.stdin.write(line + "\n"); proc.stdin.flush()
t0 = time.time()
lines = []
def read_until(pred, timeout=900):
    while time.time() - t0 < timeout:
        l = proc.stdout.readline()
        if not l:
            print("ENGINE DIED after", sent[-1] if sent else "start"); return None
        lines.append(l.rstrip()); print("OUT:", l.rstrip()[:160], flush=True)
        if pred(l.rstrip()):
            return l.rstrip()
    print("TIMEOUT"); return None
read_until(lambda l: l.startswith("READY") or l.startswith("INFO"))
send(f'GRAMMAR {sha} 13 root ::= "ab"')
read_until(lambda l: l.startswith("GRAMMAR"))
send("GRAMMAR 1111111111111111 19 root ::= notdefined")
read_until(lambda l: l.startswith("GRAMMAR"))
send(f"GEN 8 seed=7 temperature=0.8 grammar={sha} 1,2,3")
read_until(lambda l: l.startswith("DONE") or l.startswith("ERR"))
send("GEN 4 grammar=deadbeefdeadbeef 1,2,3")
read_until(lambda l: l.startswith("ERR"))
send(f"GEN 4 grammar={sha} spec_min_p=0.5 1,2,3")
read_until(lambda l: l.startswith("ERR"))
proc.kill()
print("SMOKE-DONE")
