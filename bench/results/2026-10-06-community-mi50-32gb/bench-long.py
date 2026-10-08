import json, time, urllib.request, random
URL="http://127.0.0.1:8080/v1/chat/completions"
random.seed(7)
words="rede fibra cliente contrato roteador antena sinal latência suporte fatura plano cobertura instalação modem ponto técnico visita chamado velocidade link dedicado".split()
def filler(n_sent, tag):
    out=[]
    for i in range(n_sent):
        s=" ".join(random.choice(words) for _ in range(random.randint(10,18)))
        out.append(f"[{tag}-{i}] O registro {random.randint(1000,9999)} informa: {s}.")
    return "\n".join(out)
def run(n_sent, needle_at=0.5):
    secret=str(random.randint(100000,999999))
    pre=filler(int(n_sent*needle_at),"A"); post=filler(n_sent-int(n_sent*needle_at),"B")
    doc=pre+f"\n\nIMPORTANTE: o código de acesso do cofre é {secret}.\n\n"+post
    q="\n\nResponda apenas com o código de acesso do cofre mencionado no texto acima."
    body={"model":"x","messages":[{"role":"user","content":doc+q}],"max_tokens":40,"temperature":0.0,"reasoning_effort":"none"}
    t=time.time(); r=urllib.request.urlopen(urllib.request.Request(URL,json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=1500)
    d=json.loads(r.read()); dt=time.time()-t
    u=d["usage"]; txt=(d["choices"][0]["message"].get("content") or "").strip()
    ok = secret in txt
    pt=u["prompt_tokens"]
    print(f"prompt={pt} tok  tempo={dt:.1f}s  prefill~{pt/dt:.0f} tok/s  agulha={'OK' if ok else 'FALHOU'}  resp={txt[:40]!r}", flush=True)
for n in (60, 250, 500, 1000, 2000, 3500):
    run(n)
print("DONE", flush=True)
