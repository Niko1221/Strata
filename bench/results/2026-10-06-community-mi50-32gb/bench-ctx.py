import json, sys, time, random, urllib.request
URL = "http://127.0.0.1:8080/v1/chat/completions"
target = int(sys.argv[1])            # tokens aproximados do prompt
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 11
random.seed(seed)
words = "rede fibra cliente contrato roteador antena sinal latência suporte fatura plano cobertura instalação modem ponto técnico visita chamado velocidade link dedicado".split()
def filler(n, tag):
    return "\n".join(f"[{tag}-{i}] O registro {random.randint(1000,9999)} informa: " + " ".join(random.choice(words) for _ in range(random.randint(10, 18))) + "." for i in range(n))
n = int(target / 33.85)
codes = {k: str(random.randint(100000, 999999)) for k in ("ALFA", "BETA", "GAMA")}
cuts = (int(n * 0.10), int(n * 0.50), int(n * 0.90))
parts = [filler(cuts[0], "A"), f"\n\nIMPORTANTE: o código do cofre ALFA é {codes['ALFA']}.\n\n",
         filler(cuts[1] - cuts[0], "B"), f"\n\nIMPORTANTE: o código do cofre BETA é {codes['BETA']}.\n\n",
         filler(cuts[2] - cuts[1], "C"), f"\n\nIMPORTANTE: o código do cofre GAMA é {codes['GAMA']}.\n\n",
         filler(n - cuts[2], "D")]
doc = "".join(parts)
def ask(q, max_tokens):
    body = {"model": "x", "messages": [{"role": "user", "content": doc + "\n\n" + q}], "max_tokens": max_tokens, "temperature": 0.0, "reasoning_effort": "none"}
    t = time.time()
    r = urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3000)
    d = json.loads(r.read())
    return time.time() - t, d["usage"], (d["choices"][0]["message"].get("content") or "").strip()
dt, u, txt = ask("Responda apenas com os três códigos dos cofres ALFA, BETA e GAMA, nessa ordem, separados por vírgula.", 60)
hits = [c in txt for c in (codes["ALFA"], codes["BETA"], codes["GAMA"])]
print(f"[1] prompt={u['prompt_tokens']} tok tempo={dt:.0f}s prefill~{u['prompt_tokens']/dt:.0f} tok/s agulhas(10/50/90%)={hits} resp={txt[:60]!r}", flush=True)
dt, u, txt = ask("Escreva um resumo de cerca de 200 palavras sobre os temas mais frequentes do texto acima.", 300)
print(f"[2] prompt={u['prompt_tokens']} gerados={u['completion_tokens']} tempo={dt:.0f}s", flush=True)
print("DONE", flush=True)
