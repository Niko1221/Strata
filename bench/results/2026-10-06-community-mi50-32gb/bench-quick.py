import json, time, urllib.request
URL="http://127.0.0.1:8080/v1/chat/completions"
def ask(prompt, max_tokens=300, effort="none"):
    body={"model":"x","messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,"temperature":0.0,"reasoning_effort":effort}
    t=time.time(); r=urllib.request.urlopen(urllib.request.Request(URL,json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=900)
    d=json.loads(r.read()); dt=time.time()-t
    u=d.get("usage",{}); txt=d["choices"][0]["message"].get("content") or ""
    return dt,u,txt
prompts=[("curto-pt","Explique em 3 frases o que é um modelo MoE."),
 ("codigo","Escreva em Python uma função quicksort com comentários, apenas o código."),
 ("conta","Um taco e uma bola custam R$ 1,10 juntos. O taco custa R$ 1,00 a mais que a bola. Quanto custa a bola? Explique brevemente."),
 ("redacao","Escreva um texto de 250 palavras sobre a história do café.")]
for name,p in prompts:
    dt,u,txt=ask(p,400)
    ct=u.get("completion_tokens",0); pt=u.get("prompt_tokens",0)
    print(f"{name}: {dt:.1f}s prompt={pt} comp={ct} -> {ct/dt:.1f} tok/s (inclui prefill) | {txt[:90]!r}")
