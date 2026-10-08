import json, time, urllib.request, random
URL="http://127.0.0.1:8080/v1/chat/completions"
random.seed(99)
words="rede fibra cliente contrato roteador antena sinal latência suporte fatura plano cobertura instalação modem ponto técnico visita chamado velocidade link dedicado".split()
def filler(n):
    return "\n".join(f"[R-{i}] O registro {random.randint(1000,9999)} informa: "+" ".join(random.choice(words) for _ in range(random.randint(10,18)))+"." for i in range(n))
def run(n):
    doc=filler(n)
    q="\n\nCom base no texto acima, escreva um resumo de cerca de 250 palavras sobre os temas mais frequentes."
    body={"model":"x","messages":[{"role":"user","content":doc+q}],"max_tokens":350,"temperature":0.0,"reasoning_effort":"none"}
    t=time.time(); r=urllib.request.urlopen(urllib.request.Request(URL,json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=1500)
    d=json.loads(r.read()); u=d["usage"]
    print(f"ctx={u['prompt_tokens']} gerados={u['completion_tokens']} total={time.time()-t:.0f}s", flush=True)
for n in (1000, 2000): run(n)
print("DONE", flush=True)
