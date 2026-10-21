#!/usr/bin/env python3
"""bench3d.py - bateria 3D com o Blender da VM (5.2.1, sem tela): o modelo escreve um script bpy para cada tarefa de
tarefas3d.py; o Blender executa numa cena vazia e um verificador confere medidas, posicoes, rotacoes, materiais, hierarquia,
modificadores, camera, luz e animacao. Nota = tarefas que passaram (uma tentativa, temperatura 0, raciocinio no padrao do servidor).
  bench3d.py validar           as solucoes de referencia passam e a cena vazia falha em todas (confere os verificadores)
  bench3d.py rodar <tag>       pergunta ao servidor de teste (8090) e grava resultados/<tag>/bench3d/
"""
import json, os, re, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

S = Path(__file__).resolve().parent; T = Path(os.environ.get("TESTES_DIR", "<data>/testes-021"))
BLENDER = os.environ.get("BLENDER", str(Path.home() / ".local/bin/blender"))
URL = os.environ.get("TESTES_URL", "http://127.0.0.1:8090")
sys.path.insert(0, str(S)); import tarefas3d  # noqa: E402

SISTEMA = ("Você escreve scripts Python para o Blender 5.2 (módulo bpy). O script roda sem interface (blender -b) numa cena "
           "vazia: não há cubo, câmera nem luz padrão. As unidades são metros; o chão é o plano XY (z = 0) e +Z aponta para "
           "cima. Crie só o que for pedido: nada de chão, câmera ou luz a mais, a menos que o pedido diga. Responda com um "
           "único bloco ```python``` contendo o script completo.")


def executa(tid, codigo):
    with tempfile.TemporaryDirectory() as d:
        sc, out = Path(d) / "s.py", Path(d) / "r.json"; sc.write_text(codigo, encoding="utf-8")
        try:
            p = subprocess.run(["nice", "-n", "5", BLENDER, "-b", "--factory-startup", "-t", "4", "--python", str(S / "executor3d.py"),
                                "--", tid, str(sc), str(out)], capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            return {"tarefa": tid, "erro": "tempo esgotado (180 s)", "passou": False, "detalhe": ""}
        if not out.exists():
            return {"tarefa": tid, "erro": "o Blender nao gravou o resultado: " + (p.stderr or p.stdout)[-600:], "passou": False, "detalhe": ""}
        return json.loads(out.read_text(encoding="utf-8"))


def validar():
    ruins = 0
    for tid, cat, _, _ in tarefas3d.TAREFAS:
        a, b = executa(tid, tarefas3d.REF[tid]), executa(tid, "pass")
        ok = a["passou"] and not b["passou"]; ruins += not ok
        print(f"{'OK ' if ok else 'RUIM'} {tid:18s} ref={a['passou']} vazio={b['passou']}  {a['detalhe'][:90]} {a['erro'] or ''}"[:220])
    print(f"{len(tarefas3d.TAREFAS) - ruins}/{len(tarefas3d.TAREFAS)} verificadores conferidos"); return ruins == 0


def codigo_de(texto):
    blocos = re.findall(r"```(?:python|py)?\s*\n(.*?)```", texto or "", re.S)
    return blocos[-1] if blocos else (texto or "")


def rodar(tag):
    key = (T / "configs/.api_key-teste").read_text().strip(); d = T / "resultados" / tag / "bench3d"; d.mkdir(parents=True, exist_ok=True)
    linhas = []
    for tid, cat, pedido, _ in tarefas3d.TAREFAS:
        body = {"model": "t", "max_tokens": 8192, "temperature": 0,
                "messages": [{"role": "system", "content": SISTEMA}, {"role": "user", "content": pedido}]}
        t0 = time.time()
        try:
            rq = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                        {"Content-Type": "application/json", "Authorization": "Bearer " + key})
            resp = json.load(urllib.request.urlopen(rq, timeout=1200)); msg = resp["choices"][0]["message"]
            texto, uso, fim = msg.get("content") or "", resp.get("usage", {}), resp["choices"][0].get("finish_reason")
        except Exception as e:  # noqa: BLE001
            texto, uso, fim = "", {}, f"erro HTTP: {e}"
        dt = time.time() - t0; codigo = codigo_de(texto)
        (d / f"{tid}.resposta.md").write_text(texto, encoding="utf-8"); (d / f"{tid}.py").write_text(codigo, encoding="utf-8")
        r = executa(tid, codigo) if codigo.strip() else {"tarefa": tid, "erro": "sem codigo na resposta", "passou": False, "detalhe": ""}
        r.update(categoria=cat, segundos=round(dt, 1), tokens=uso.get("completion_tokens"), fim=fim); linhas.append(r)
        print(f"{'PASSOU' if r['passou'] else 'falhou'} {tid:18s} {dt:6.1f}s {r['detalhe'][:80]} {(r['erro'] or '')[:80]}".replace("\n", " "), flush=True)
    cats = {}
    for x in linhas:
        c = cats.setdefault(x["categoria"], [0, 0]); c[0] += x["passou"]; c[1] += 1
    res = {"tag": tag, "passou": sum(x["passou"] for x in linhas), "total": len(linhas), "por_categoria": cats, "tarefas": linhas}
    (d / "resultado.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"3D {tag}: {res['passou']}/{res['total']}", cats)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "validar":
        sys.exit(0 if validar() else 1)
    elif cmd == "rodar":
        rodar(sys.argv[2])
    else:
        print(__doc__); sys.exit(1)
