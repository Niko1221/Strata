import subprocess
import time
import requests
import os
import signal

# --- CONFIGURAZIONE ---
MODEL_PATH = "/home/user/workspaces/rag_rss/models/llm_models/models/DeepSeekV4/DeepSeek-V4-Flash-UD-IQ1_M-00001-of-00003.gguf"
PORT = "8095"
CTX = "4096"
# Il motore genera un file binario finale con questo flag
PROFILE_OUTPUT = "/home/user/workspaces/rag_rss/models/llm_models/models/DeepSeekV4/dsv4_experts.bin"
LOG_FILE = "engine_warmup.log"

# Batteria di prompt differenziati per attivare ed evidenziare gli esperti "hot" corretti
PROMPTS = [
    "Scrivi un algoritmo di ordinamento QuickSort in Python molto ottimizzato.",
    "Riassumi le principali differenze tra una Mixture of Experts (MoE) e un modello denso classico.",
    "Traduci questo testo in inglese: Il sistema di intelligenza artificiale locale richiede molta memoria VRAM ed ottimizzazione dei pesi.",
    "Risolvi questa equazione: 3x + 5 = 20. Mostra i passaggi matematici.",
    "Agisci come un feed RSS parser ed estrai le entità nominate da una news di tecnologia."
]

def start_server_for_profiling():
    print("🚀 Avvio del server per la profilazione degli esperti...")
    
    # Passiamo il flag nativo di salvataggio al motore tramite --engine-arg
    cmd = [
        "python3", "tools/serve_dsv4.py",
        "--model", MODEL_PATH,
        "--port", PORT,
        "--ctx", CTX,
        "--expert-ram", "all", # Carica temporaneamente in RAM per la calibrazione
        "--engine-env", "DSV4_EMULATED_VRAM_MIB=22000",
        "--engine-arg", f"--expert-profile-save",
        "--engine-arg", PROFILE_OUTPUT
    ]
    
    log_out = open(LOG_FILE, "w")
    process = subprocess.Popen(cmd, stdout=log_out, stderr=log_out, text=True)
    
    # Attesa che l'endpoint OpenAI mock del server sia attivo
    url = f"http://127.0.0.1:{PORT}/v1/chat/completions"
    for i in range(30):
        try:
            print(f"⏳ In attesa del server (tentativo {i+1}/30)...")
            requests.get(f"http://127.0.0.1:{PORT}/", timeout=2)
            break
        except requests.exceptions.RequestException:
            time.sleep(3)
    else:
        print("❌ Il server ha impiegato troppo tempo ad avviarsi. Controlla engine_warmup.log")
        process.terminate()
        exit(1)
        
    return process

def run_evaluation_prompts():
    url = f"http://127.0.0.1:{PORT}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    print("\n🧠 Invio dei prompt di valutazione per generare hit sugli esperti...")
    for idx, prompt in enumerate(PROMPTS):
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150
        }
        start_time = time.time()
        try:
            print(f"💬 Prompt {idx+1}/{len(PROMPTS)}: '{prompt[:45]}...'")
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                print(f"   ✅ Risposta ricevuta in {time.time() - start_time:.2f}s")
            else:
                print(f"   ⚠️ Risposta del server anomala: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Errore di connessione: {e}")
            
def main():
    server_process = start_server_for_profiling()
    
    try:
        run_evaluation_prompts()
        print("\n🎯 Valutazione completata.")
    finally:
        print("🛑 Arresto del server per forzare la scrittura del profilo (atomic save)...")
        # Inviamo un SIGINT (Ctrl+C) pulito per permettere al ciclo C++ di uscire e salvare il file
        server_process.send_signal(signal.SIGINT)
        try:
            server_process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            print("⚠️ Il server non si è chiuso in tempo, forzo l'arresto...")
            server_process.kill()
            
        print("💾 Controllo del file di output...")
        if os.path.exists(PROFILE_OUTPUT) and os.path.getsize(PROFILE_OUTPUT) > 0:
            print(f"🎉 Successo! Profilo generato correttamente in: {PROFILE_OUTPUT}")
            print("\n👉 Ora puoi lanciare il server definitivo per tenere solo gli esperti HOT in VRAM con:")
            print(f"python3 tools/serve_dsv4.py --model {MODEL_PATH} --port {PORT} --ctx {CTX} \\")
            print(f"  --expert-ram profile --expert-profile {PROFILE_OUTPUT} --expert-vram-pct 80 --expert-adapt-swaps 8 --engine-env DSV4_EMULATED_VRAM_MIB=22000")
        else:
            print("❌ Errore: Il file di profilo non è stato generato o è vuoto. Verifica il contenuto di 'engine_warmup.log'.")

if __name__ == "__main__":
    main()

