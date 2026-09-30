#!/usr/bin/env python3
"""Confronto diretto llama.cpp <-> strata-dense, stessi id di token, greedy, niente template.

Uso (dalla cartella Strata, con il venv che ha llama_cpp):
    python compare_llama.py /percorso/Qwen3.8-27B-UD-Q4_K_M.gguf
    python compare_llama.py MODEL.gguf "The capital of France is" 8

llama.cpp gira su CPU (-ngl 0) per non contendere la VRAM a strata-dense: lento ma bastano pochi token.
"""
import os, subprocess, sys, tempfile

gguf = sys.argv[1]
text = sys.argv[2] if len(sys.argv) > 2 else "The capital of France is"
n_new = int(sys.argv[3]) if len(sys.argv) > 3 else 8
exe = os.environ.get("STRATA_DENSE_EXE", "./build-dense/strata-dense")

# ---------- llama.cpp (binari: llama-tokenize e llama-cli; LLAMA_BIN_DIR=cartella che li contiene, se non sono nel PATH)
import re, shutil
bindir = os.environ.get("LLAMA_BIN_DIR", "")
def tool(name):
    p = os.path.join(bindir, name) if bindir else shutil.which(name)
    if not p or not os.path.exists(p):
        sys.exit("non trovo %s: esporta LLAMA_BIN_DIR=/cartella/con/i/binari di llama.cpp" % name)
    return p

def tokenize(t):
    out = subprocess.run([tool("llama-tokenize"), "-m", gguf, "-p", t, "--ids", "--no-bos", "--log-disable"],
                         capture_output=True, text=True).stdout
    m = re.search(r"\[([0-9,\s]*)\]", out)
    if not m:
        sys.exit("llama-tokenize: output non riconosciuto:\n" + out)
    return [int(x) for x in m.group(1).split(",") if x.strip()]

ids = tokenize(text)
print("prompt ids:", ids)
comp = "llama-completion" if (shutil.which("llama-completion") or os.path.exists(os.path.join(bindir, "llama-completion"))) else "llama-cli"
r = subprocess.run([tool(comp), "-m", gguf, "-p", text, "-n", str(n_new), "--temp", "0", "-ngl", "0",
                    "-c", "256", "-no-cnv", "--no-warmup"], capture_output=True, text=True)
if not r.stdout.strip():
    print("ATTENZIONE: %s non ha scritto niente su stdout. stderr (ultime righe):\n%s" % (comp, r.stderr[-800:]))
ref_text = r.stdout[len(text):] if r.stdout.startswith(text) else r.stdout
print("llama.cpp greedy text:", repr(ref_text))
ref = tokenize(text + ref_text)[len(ids):][:n_new]     # id ricavati dal testo: bastano per il confronto
print("llama.cpp greedy ids :", ref)

# ---------- strata-dense
err = tempfile.NamedTemporaryFile("w+", suffix=".err", delete=False)
env = dict(os.environ, STRATA_DENSE_DEBUG="%d:%d" % (len(ids) - 1, len(ids) + n_new + 2))
p = subprocess.Popen([exe, "--serve", "--native", gguf, "--context", "256"], stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE, stderr=err, text=True, env=env)
while True:
    line = p.stdout.readline()
    if not line:
        sys.exit("strata-dense e' uscito prima di READY; vedi " + err.name)
    if line.startswith("READY"):
        break
p.stdin.write("GEN %d temperature=0 top_k=1 %s\n" % (n_new, ",".join(map(str, ids))))
p.stdin.flush()
got = []
while True:
    line = p.stdout.readline()
    if not line or line.startswith("DONE"):
        break
    if line.startswith("T "):
        got.append(int(line.split()[1]))
p.stdin.write("QUIT\n"); p.stdin.flush(); p.wait(timeout=30)
print("strata-dense greedy ids:", got)
err.seek(0)
logit_lines = [l.strip() for l in err if "logits" in l]
print("strata-dense first-step:", logit_lines[0] if logit_lines else "(nessuna riga logits)")
print("\nRISULTATO:", "IDENTICI" if got == ref else "DIVERSI (primo id diverso alla posizione %d)" %
      next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), min(len(got), len(ref))))
