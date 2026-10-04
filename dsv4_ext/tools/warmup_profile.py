#!/usr/bin/env python3
"""Builds the dsv4 expert profile by driving the real CUDA server with a few prompts.

Run it from the dsv4_ext directory:   python3 warmup_profile.py
"""
import array
import os
import signal
import struct
import subprocess
import sys
import time

import requests

# --- CONFIGURAZIONE ---
MODEL_PATH = sys.argv[1]
PORT = "8095"
CTX = "4096"
# 2. Estrae la cartella di base (basepath) in cui si trova il modello
BASE_PATH = os.path.dirname(MODEL_PATH)

# 3. Costruisce il nuovo percorso di output unendo il BASE_PATH al nome del file
PROFILE_OUTPUT = os.path.join(BASE_PATH, "dsv4_experts.bin")
LOG_FILE = "engine_warmup.log"
START_TIMEOUT_S = 900      # model load with --expert-ram all is ~85 s once the mmap fix is in
REQUEST_TIMEOUT_S = 900    # 150 tokens at a few tok/s, plus a cold start on the first prompts

# Representative prompts: the profile only knows what these activate. Use the kind of text you will really send.
PROMPTS = [
    "ciao, come stai oggi?",
    "Scrivi un algoritmo di ordinamento QuickSort in Python molto ottimizzato.",
    "Riassumi le principali differenze tra una Mixture of Experts (MoE) e un modello denso classico.",
    "Traduci questo testo in inglese: Il sistema di intelligenza artificiale locale richiede molta memoria VRAM ed ottimizzazione dei pesi.",
    "Risolvi questa equazione: 3x + 5 = 20. Mostra i passaggi matematici.",
    "Agisci come un feed RSS parser ed estrai le entità nominate da una news di tecnologia.",
    "Explain in simple terms how a hash table handles collisions.",
    "Scrivi una breve mail formale per chiedere un preventivo a un fornitore.",
]


def stop_group(proc):
    """Kill the server AND the engine it spawned (they share the session started below)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def start_server():
    for p in (PROFILE_OUTPUT, PROFILE_OUTPUT + ".tmp"):
        if os.path.exists(p):
            os.remove(p)   # a stale file would make the final check report success for nothing
    cmd = [
        "python3", "tools/serve_dsv4.py", "--cuda",     # without --cuda this runs the CPU-emulated engine
        "--model", MODEL_PATH, "--port", PORT, "--ctx", CTX,
        "--expert-vram-pct", "80",
        "--expert-ram", "all", "--ram-cache-mib", "0",
        "--expert-profile-save", PROFILE_OUTPUT,
    ]
    print("Avvio del server (CUDA) per la profilazione...")
    log = open(LOG_FILE, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < START_TIMEOUT_S:
        if proc.poll() is not None:
            sys.exit(f"Il server si e' fermato (codice {proc.returncode}): vedi {LOG_FILE}")
        try:
            requests.get(f"http://127.0.0.1:{PORT}/", timeout=2)
            print(f"Server pronto dopo {time.time() - t0:.0f}s")
            return proc
        except requests.exceptions.RequestException:
            time.sleep(3)
    stop_group(proc)
    sys.exit(f"Il server non e' partito entro {START_TIMEOUT_S}s: vedi {LOG_FILE}")


def run_prompts():
    url = f"http://127.0.0.1:{PORT}/v1/chat/completions"
    for i, prompt in enumerate(PROMPTS, 1):
        payload = {"model": "deepseek-v4-flash", "max_tokens": 150,
                   "messages": [{"role": "user", "content": prompt}]}
        t = time.time()
        print(f"Prompt {i}/{len(PROMPTS)}: {prompt[:50]}...")
        try:
            r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_S)
            print(f"   HTTP {r.status_code} in {time.time() - t:.1f}s")
        except requests.exceptions.RequestException as e:
            print(f"   errore: {e}")


def wait_for_profile(t0, timeout_s):
    """True once the file exists, was written after t0 and has more than the 28-byte header."""
    end = time.time() + timeout_s
    while time.time() < end:
        try:
            st = os.stat(PROFILE_OUTPUT)
            if st.st_mtime >= t0 and st.st_size > 28:
                return True
        except FileNotFoundError:
            pass
        time.sleep(1)
    return False


def summarize(path):
    with open(path, "rb") as f:
        if f.read(8) != b"DSV4PROF":
            return None
        _ver, nl, ne, _hash = struct.unpack("<IIIQ", f.read(20))
        counts = array.array("Q")
        counts.frombytes(f.read())
    if len(counts) != nl * ne:
        return None
    return nl, ne, sum(1 for c in counts if c), len(counts)


def main():
    t0 = time.time()
    proc = start_server()
    ok = False
    try:
        run_prompts()
        # Engine that saves after every request (the serve_loop.cpp patch): the file is already there.
        ok = wait_for_profile(t0, 20)
        if not ok:
            # Stock engine: it saves only when its stdin reaches EOF or it reads QUIT. SIGINT/SIGHUP/SIGKILL on the
            # ENGINE would skip the save, so terminate ONLY the python parent: its pipe closes, the engine sees EOF,
            # finishes what it is doing and writes the profile.
            print("Chiudo il server python: il motore salva il profilo quando vede la pipe chiusa...")
            proc.send_signal(signal.SIGTERM)
            ok = wait_for_profile(t0, 180)
    finally:
        stop_group(proc)

    left = subprocess.run(["pgrep", "-f", "dsv4_run"], capture_output=True, text=True).stdout.split()
    if left:
        print(f"ATTENZIONE: processi dsv4_run ancora attivi (PID {', '.join(left)}): liberali con kill, tengono ~90 GiB.")

    if not ok:
        sys.exit(f"Profilo non scritto: vedi {LOG_FILE} (cerca 'profile saved').")
    info = summarize(PROFILE_OUTPUT)
    if info is None:
        sys.exit(f"{PROFILE_OUTPUT} esiste ma non e' un profilo DSV4 valido.")
    nl, ne, seen, total = info
    print(f"Profilo scritto: {PROFILE_OUTPUT}")
    print(f"  layer {nl}, expert/layer {ne}, expert attivati almeno una volta: {seen}/{total} ({100.0 * seen / total:.1f}%)")
    print("\nServer definitivo:")
    print(f"python3 tools/serve_dsv4.py --cuda --model {MODEL_PATH} --port {PORT} --ctx {CTX} \\")
    print(f"  --expert-vram-pct 80 --expert-ram all --ram-cache-mib 0 --expert-adapt-swaps 8 \\")
    print(f"  --expert-profile {PROFILE_OUTPUT} --expert-profile-save {PROFILE_OUTPUT}")


if __name__ == "__main__":
    main()
