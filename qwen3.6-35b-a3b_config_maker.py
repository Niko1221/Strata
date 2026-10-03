#!/usr/bin/env python3
"""Writes qwen36.json for `python -m serve.server --engine strata --config qwen36.json`.

  python qwen3.6-35b-a3b_config_maker.py model.gguf                      # senza MTP (come prima)
  python qwen3.6-35b-a3b_config_maker.py model.gguf --mtp                 # MTP, variante scelta in automatico
  python qwen3.6-35b-a3b_config_maker.py model.gguf --mtp --mtp-k 4 --mtp-variant 3   # variante fissata a mano

Con --kv int8 --kv-unified la KV cache occupa circa la meta' (un solo buffer).
Con --mtp (senza --mtp-variant) lo script lancia il probe dell'engine (--selftest-mtp) su un testo tokenizzato qui, legge la
variante migliore e la scrive nel JSON. Serve la GPU libera: chiudi prima il server.
Eseguire dalla radice di Strata (usa tools/strata_tokenizer.py).
"""
import argparse, json, os, re, subprocess, sys

sys.path.insert(0, "tools")
import strata_tokenizer as T

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("gguf", help="percorso del file .gguf di Qwen3.6-35B-A3B")
ap.add_argument("--max-context", type=int, default=32768)
ap.add_argument("--mtp", action="store_true", help="attiva la testa MTP (decodifica speculativa)")
ap.add_argument("--mtp-k", type=int, default=3, help="bozze per giro, 1..7 (default 3)")
ap.add_argument("--mtp-variant", type=int, default=None, help="0..7; se omesso con --mtp viene scelta da --selftest-mtp")
ap.add_argument("--probe-tokens", type=int, default=200, help="token generati dal probe MTP (default 200)")
ap.add_argument("--kv", choices=["fp16", "int8"], default="fp16", help="formato della KV cache: int8 usa ~47%% di memoria in meno (default fp16)")
ap.add_argument("--kv-unified", action="store_true", help="un unico buffer GPU per K e V di tutti i layer")
ap.add_argument("--ckpt-max", type=int, default=12, help="snapshot del prompt tenuti in RAM (~62 MB l'uno, 0 = spenti; default 12)")
ap.add_argument("--ckpt-every", type=int, default=8192, help="un snapshot ogni N token di prompt (default 8192)")
ap.add_argument("--exe", default=os.path.abspath("build/src/qwen36/strata-qwen36"))
ap.add_argument("--out", default="qwen36.json")
a = ap.parse_args()

if not 1 <= a.mtp_k <= 7:
    ap.error("--mtp-k deve essere tra 1 e 7")
if a.mtp_variant is not None and not 0 <= a.mtp_variant <= 7:
    ap.error("--mtp-variant deve essere tra 0 e 7")
if not os.path.exists(a.exe):
    print(f"ATTENZIONE: {a.exe} non esiste ancora (compila con --target strata-qwen36 o usa --exe)", file=sys.stderr)

gguf = os.path.abspath(a.gguf)
tok = T.Tokenizer.from_gguf(gguf)
eos = [e[0] for e in (tok.encode(s, parse_special=True) for s in ("<|im_end|>", "<|endoftext|>")) if len(e) == 1]

PROBE_TEXT = ("Scrivi una storia dettagliata su un vecchio faro sulla costa e sul guardiano che ogni sera accende la luce. "
              "Descrivi il mare, il vento, i gabbiani e le persone del villaggio, con dialoghi naturali. "
              "Write a clear explanation of how a hash table works, with an example in Python and its time complexity.")

def pick_variant():
    """Runs the engine's MTP probe and returns the best variant, or None if MTP is unusable with this GGUF."""
    ids = tok.encode(PROBE_TEXT)
    ids_file = os.path.abspath("mtp_probe_ids.txt")
    open(ids_file, "w").write(",".join(map(str, ids)))
    cmd = [a.exe, "--native", gguf, "--max-context", "4096", "--mtp", "--selftest-mtp", str(a.probe_tokens), ids_file]
    print("probe MTP:", " ".join(cmd), flush=True)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except (OSError, subprocess.TimeoutExpired) as e:
        print("probe MTP non riuscito:", e, file=sys.stderr)
        return None
    print(r.stdout)
    if "MTP is off" in r.stderr or "MTP head" not in r.stderr:
        print("questo GGUF non ha una testa MTP utilizzabile; MTP disattivato.\n" + r.stderr[-600:], file=sys.stderr)
        return None
    m = re.search(r"MIGLIORE: --mtp-variant (\d) \(([\d.]+)%\)", r.stdout)
    if not m:
        print("probe MTP senza risultato:\n" + r.stderr[-600:], file=sys.stderr)
        return None
    if float(m.group(2)) < 20.0:
        print(f"la variante migliore ha solo {m.group(2)}%: i tensori MTP non corrispondono; MTP disattivato.", file=sys.stderr)
        return None
    return int(m.group(1))

if a.mtp and a.mtp_variant is None:
    a.mtp_variant = pick_variant()
    if a.mtp_variant is None:
        a.mtp = False
args = ["--native", gguf, "--max-context", str(a.max_context), "--eos-ids", ",".join(map(str, eos))]
if (a.ckpt_max, a.ckpt_every) != (12, 8192):
    args += ["--ckpt-max", str(a.ckpt_max), "--ckpt-every", str(a.ckpt_every)]
if a.kv != "fp16":
    args += ["--kv", a.kv]
if a.kv_unified:
    args += ["--kv-unified"]
if a.mtp:
    args += ["--mtp", "--mtp-k", str(a.mtp_k), "--mtp-variant", str(a.mtp_variant)]
cfg = {
    "exe": a.exe,
    "args": args,
    "tokenizer": os.path.abspath("qwen36/tokenizer"),
    "model_name": "qwen3.6-35b-a3b",
    "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
}
json.dump(cfg, open(a.out, "w"), indent=2)
print(f"scritto {a.out}, eos = {eos}, KV = {a.kv}{' unificata' if a.kv_unified else ''}, MTP = " + (f"on (k={a.mtp_k}, variante {a.mtp_variant})" if a.mtp else "off"))
print(f"avvio: python -m serve.server --engine strata --config {a.out} --port 8080 --open")
