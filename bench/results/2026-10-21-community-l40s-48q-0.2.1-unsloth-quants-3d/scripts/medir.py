#!/usr/bin/env python3
"""medir.py - medidas do servidor de TESTE (porta 8090). Grava tudo em resultados/<tag>/.
# Copia de testes-020/scripts/medir.py para a bateria 0.2.1 (TESTES_DIR=<data>/testes-021). Antes: copia de strata-0145/scripts/medir.py para a bateria 0.2.0: TESTES_DIR=<data>/testes-020, STRATA_SRC=testes-q6-q8/strata-src (os mesmos prompts de 10/10).
# Copia de testes-q6-q8/scripts/medir.py para a v0.1.45: roda com TESTES_DIR=<data>/strata-0145 e STRATA_SRC=testes-q6-q8/strata-src (v0.1.42),
# para os prompts (montados de arquivos do codigo) serem os MESMOS dos testes de 10/10 e a KL comparar com eles; venv-eval de testes-q6-q8.

Rodar com o python do venv da producao (so executado, nada e instalado):
    PY=<data>/strata/.venv/bin/python
    $PY medir.py info       <tag>                 # /health, /v1/models e as linhas de inicio do log (slots, VRAM livre)
    $PY medir.py velocidade <tag> [8k,32k,128k,256k,500k]   # leitura do prompt (frio) e geracao de 512 tokens
    $PY medir.py agulhas    <tag> [32k,128k,262k,400k,500k] # tools/needle_bench.py, profundidades 10/50/90 %
    $PY medir.py visao      <tag> [imagem ...]   # imagem sintetica (3 formas) e, se dadas, imagens suas
    $PY medir.py kl-coletar <tag>                 # 60 prompts fixos, 1 token; o motor grava os logits (ver abaixo)
    $PY medir.py kl-comparar <ref> <tag> ...      # KL(ref || tag), mesmo top-1, sobreposicao top-5
    $PY medir.py resumo                           # tabela de todos os resultados -> resultados/RESUMO.md

kl-coletar precisa do motor iniciado com o despejo de logits ligado:
    STRATA_ENV="STRATA_DUMP_FIRST_LOGITS=<data>/testes-q6-q8/resultados/<tag>/logits/l" alternar.sh teste <tag>
(STRATA_DUMP_FIRST_LOGITS, src/program/generate.cpp:10688: os logits completos da 1a janela de cada pedido).
"""
import base64, io, json, os, random, re, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path

T = Path(os.environ.get("TESTES_DIR", "<data>/testes-q6-q8"))          # na VM das A2: /root/a2 (kit vm-a2)
SRC = Path(os.environ.get("STRATA_SRC", str(T / "strata-src"))); RES = T / "resultados"
URL = os.environ.get("TESTES_URL", "http://127.0.0.1:8090")
KEY = (T / "configs/.api_key-teste").read_text().strip()
sys.path.insert(0, str(SRC / "tools"))


def req(path, body=None, timeout=3600):
    h = {"Authorization": "Bearer " + KEY}
    if body is not None:
        h["Content-Type"] = "application/json"
    r = urllib.request.Request(URL + path, json.dumps(body).encode() if body is not None else None, h)
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.load(f)


def ultimo():                                   # o pedido mais recente em /metrics (a lista vem do mais novo ao mais velho)
    return max(req("/metrics?requests=all")["requests"], key=lambda x: x.get("time", 0))


def salva(tag, nome, dados):
    d = RES / tag; d.mkdir(parents=True, exist_ok=True)
    (d / nome).write_text(json.dumps(dados, indent=1, ensure_ascii=False))
    print("gravado", d / nome)


def tokens(s):
    return int(float(s[:-1]) * 1024) if s.lower().endswith("k") else int(s)


def info(tag):
    h, m = req("/health"), req("/v1/models")
    f = T / "logs" / f"servidor-{tag}.log"
    if not f.exists():                                   # resultado com nome diferente do config (extras A/B)
        f = max((T / "logs").glob("servidor-*.log"), key=lambda x: x.stat().st_mtime)
    log = f.read_text(errors="replace")
    log = log[log.rfind("=== "):]
    linhas = [l for l in log.splitlines() if re.search(r"expert cache|VRAM free|KV streaming|rope|ready:|experts? (in|into) RAM|GiB|resident|budget|mmap|layer split|GPU [01]", l)]
    print(json.dumps(h), "\n".join(linhas[-15:]), sep="\n")
    salva(tag, "info.json", {"health": h, "models": m, "log": linhas})


def velocidade(tag, lens="8k,32k,128k,256k,500k"):
    from needle_bench import haystack, CHARS_PER_TOKEN
    linhas = []
    for L in lens.split(","):
        n = int(tokens(L) * 0.98)
        sal = f"[sessao {random.randrange(10**9)}]\n"          # texto novo a cada pedido: nada reaproveitado do cache
        txt = sal + haystack(int(n * CHARS_PER_TOKEN))
        body = {"model": "t", "max_tokens": 512, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": txt + "\n\nResuma em portugues, em uns 20 topicos, o que esse texto "
                              "contem. Seja detalhado."}]}
        t0 = time.time(); r = req("/v1/chat/completions", body); dt = time.time() - t0
        u = ultimo()
        row = {"alvo": L, "prompt_tokens": r["usage"]["prompt_tokens"], "saida": r["usage"]["completion_tokens"],
               "total_s": round(dt, 1), "prompt_tok_s": round(u["prompt_read"] / u["prompt_ms"] * 1000) if u.get("prompt_ms") else None,
               "decode_tok_s": u.get("decode_tok_s"), "hit_rate": u.get("hit_rate"), "pcie_share": u.get("pcie_share"),
               "reaproveitado": u.get("reused"), "drafts": f"{u.get('drafts_accepted')}/{u.get('drafts_offered')}"}
        print(row, flush=True); linhas.append(row)
    salva(tag, "velocidade.json", linhas)


def agulhas(tag, lens="32k,128k,262k,400k,500k"):
    d = RES / tag; d.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(SRC / "tools/needle_bench.py"), "--url", URL, "--api-key", KEY, "--lengths", lens,
                    "--depths", "10,50,90", "--timeout", "3600", "--out", str(d / "agulhas.json")], check=False)


def visao(tag, imagens=()):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (640, 480), "white"); g = ImageDraw.Draw(im)
    g.ellipse((60, 150, 220, 310), fill=(220, 0, 0)); g.rectangle((260, 160, 400, 300), fill=(0, 0, 220))
    g.polygon([(480, 310), (560, 150), (620, 310)], fill=(0, 170, 0)); b = io.BytesIO(); im.save(b, "PNG")
    casos = [("sintetica", "image/png", b.getvalue(), "How many objects are in this image? Give shape and colour of each, left to right.")]
    for p in imagens:
        casos.append((Path(p).name, "image/png" if p.lower().endswith(".png") else "image/jpeg", Path(p).read_bytes(),
                      "Descreva esta imagem em detalhes, em portugues."))
    out = []
    for nome, mime, data, q in casos:
        body = {"model": "t", "max_tokens": 3000, "reasoning_budget_tokens": 1000, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(data).decode()}},
            {"type": "text", "text": q}]}]}
        t0 = time.time(); r = req("/v1/chat/completions", body); a = r["choices"][0]["message"].get("content") or ""
        row = {"imagem": nome, "s": round(time.time() - t0, 1), "resposta": a}
        if nome == "sintetica":
            row["ok"] = all(w in a.lower() for w in ("red", "blue", "green", "circle", "square", "triangle"))
        print(nome, row.get("ok", ""), row["s"], "s:", a[:200].replace("\n", " "), flush=True); out.append(row)
    salva(tag, "visao.json", out)


def imagens_detalhe():
    """24 imagens sinteticas com resposta conhecida (semente fixa), para ver se o modelo enxerga DETALHE:
    texto pequeno (9/12/16/24 px, com texto cinza de distracao), contar pontos, ler uma celula de tabela 6x6,
    cor por posicao numa grade 3x3. Devolve [(tipo, png, pergunta, resposta_certa)]."""
    from PIL import Image, ImageDraw, ImageFont
    F = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    rnd = random.Random(4321); out = []
    def png(im):
        b = io.BytesIO(); im.save(b, "PNG"); return b.getvalue()
    for tam in (9, 12, 16, 24):
        f = ImageFont.truetype(F, tam)
        for _ in range(2):
            cod = "".join(rnd.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(8))
            im = Image.new("RGB", (1280, 960), "white"); g = ImageDraw.Draw(im)
            for _ in range(60):
                g.text((rnd.randrange(0, 1150), rnd.randrange(0, 930)), "".join(rnd.choice("abcdefghijkmn ") for _ in range(12)),
                       fill=(165, 165, 165), font=f)
            x, y = rnd.randrange(40, 900), rnd.randrange(40, 880); t = f"CODE: {cod}"
            g.rectangle(g.textbbox((x, y), t, font=f), fill="white"); g.text((x, y), t, fill="black", font=f)
            out.append((f"texto{tam}px", png(im), "Somewhere in this image there is a line that starts with 'CODE:'. "
                        "Reply with only the 8 characters that follow 'CODE:', nothing else.", cod))
    for _ in range(6):
        n = rnd.randint(7, 23); im = Image.new("RGB", (1024, 768), "white"); g = ImageDraw.Draw(im); pts = []
        while len(pts) < n:
            x, y = rnd.randrange(30, 994), rnd.randrange(30, 738)
            if all((x - a) ** 2 + (y - b) ** 2 > 50 ** 2 for a, b in pts):
                pts.append((x, y)); g.ellipse((x - 12, y - 12, x + 12, y + 12), fill=(30, 60, 200))
        out.append(("contar", png(im), "How many blue dots are in this image? Reply with only the number.", str(n)))
    f = ImageFont.truetype(F, 18)
    for _ in range(6):
        im = Image.new("RGB", (900, 700), "white"); g = ImageDraw.Draw(im)
        tab = [[rnd.randint(100, 999) for _ in range(6)] for _ in range(6)]
        for i in range(7):
            g.line((50, 50 + i * 100, 850, 50 + i * 100) if False else (50, 50 + i * 100, 50 + 6 * 130, 50 + i * 100), fill="black")
            g.line((50 + i * 130, 50, 50 + i * 130, 650), fill="black")
        for r in range(6):
            for c in range(6):
                g.text((50 + c * 130 + 40, 50 + r * 100 + 38), str(tab[r][c]), fill="black", font=f)
        r, c = rnd.randrange(6), rnd.randrange(6)
        out.append(("tabela", png(im), f"This image shows a 6x6 table of numbers. What number is in row {r + 1}, column {c + 1} "
                    "(rows counted from the top, columns from the left, starting at 1)? Reply with only the number.", str(tab[r][c])))
    cores = {"red": (220, 0, 0), "green": (0, 160, 0), "blue": (0, 0, 220), "yellow": (240, 220, 0), "orange": (250, 140, 0),
             "purple": (130, 0, 160), "black": (0, 0, 0), "pink": (250, 120, 190)}
    lugares = {"top-left": (0, 0), "top-right": (0, 2), "bottom-left": (2, 0), "bottom-right": (2, 2), "center": (1, 1)}
    for _ in range(4):
        im = Image.new("RGB", (600, 600), "white"); g = ImageDraw.Draw(im)
        grade = [[rnd.choice(list(cores)) for _ in range(3)] for _ in range(3)]
        for r in range(3):
            for c in range(3):
                g.rectangle((30 + c * 190, 30 + r * 190, 30 + c * 190 + 150, 30 + r * 190 + 150), fill=cores[grade[r][c]])
        lug = rnd.choice(list(lugares)); r, c = lugares[lug]
        out.append(("cor-posicao", png(im), f"The image shows a 3x3 grid of colored squares. What color is the square in the {lug} "
                    "position? Reply with one word.", grade[r][c]))
    return out


def visao_detalhe(tag):
    """as 24 imagens de imagens_detalhe(): resposta (temperatura 0, sem raciocinio), acerto e os logits do 1o token
    (kl-mapa-visao.json; comparar com KL_MAPA=kl-mapa-visao.json medir.py kl-comparar <ref> <tag>)"""
    ld = RES / tag / "logits"; ld.mkdir(parents=True, exist_ok=True)
    antes = len(list(ld.glob("l.*"))); linhas, mapa = [], []
    for i, (tipo, data, q, certo) in enumerate(imagens_detalhe()):
        body = {"model": "t", "max_tokens": 32, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}},
                    {"type": "text", "text": q}]}]}
        t0 = time.time(); r = req("/v1/chat/completions", body); a = (r["choices"][0]["message"].get("content") or "").strip()
        if tipo in ("contar", "tabela"):
            n = re.findall(r"\d+", a); ok = bool(n) and n[0] == certo
        else:
            ok = certo.lower() in a.lower().replace(" ", "")
        linhas.append({"tipo": tipo, "certo": certo, "resposta": a[:80], "acertou": ok, "s": round(time.time() - t0, 1)})
        mapa.append({"prompt": i, "arquivo": f"l.{antes + i}", "dominio": tipo})
        print(tipo, certo, "|", a[:40], "|", "OK" if ok else "ERRO", flush=True)
    salva(tag, "visao-detalhe.json", linhas); salva(tag, "kl-mapa-visao.json", mapa)
    print("acertos:", sum(x["acertou"] for x in linhas), "de", len(linhas))


def prompts_qualidade():
    """60 prompts fixos (mesma ordem em todos os modelos): 24 perguntas + 36 continuacoes de texto/codigo do repositorio."""
    qs = json.loads((T / "scripts/prompts-qualidade.json").read_text())
    fontes = sorted((SRC / "docs").glob("*.md"))[:6] + sorted((SRC / "serve").glob("*.py"))[:3] + sorted((SRC / "src/core").glob("*.cpp"))[:3]
    rnd = random.Random(1234)
    for f in fontes:
        t = f.read_text(errors="replace")
        for _ in range(3):
            i = rnd.randrange(0, max(1, len(t) - 3000))
            qs.append("Continue o texto abaixo exatamente de onde ele para, sem comentarios:\n\n" + t[i:i + 2500])
    return qs


def prompts_dominios(n=40):
    """160 continuacoes rotuladas, n por assunto: python (biblioteca padrao do Python da maquina), cpp (codigo do Strata),
    ingles (docs do Strata sem blocos de codigo), portugues (README.pt-BR.md). Mesma escolha em todo modelo (semente fixa).
    Serve para ver se a compressao estraga mais um assunto (ex.: Python) do que outro."""
    import sysconfig
    fontes = {"python": sorted(Path(sysconfig.get_paths()["stdlib"]).glob("*.py")),
              "cpp": sorted((SRC / "src").rglob("*.cpp")),
              "ingles": [f for f in sorted((SRC / "docs").glob("*.md"))],
              "portugues": [SRC / "README.pt-BR.md"]}
    out = []
    for dom, fs in fontes.items():
        txt = []
        for f in fs:
            t = f.read_text(errors="replace")
            if dom == "ingles":
                t = re.sub(r"```.*?```", "", t, flags=re.S)
            if len(t) > 3000: txt.append(t)
        rnd = random.Random(f"dominio-{dom}")
        for k in range(n):
            t = txt[rnd.randrange(len(txt))]; i = rnd.randrange(0, len(t) - 2600)
            i = t.find("\n", i) + 1
            out.append((dom, "Continue the text below exactly where it stops, with no comments:\n\n" + t[i:i + 2000]))
    return out


def kl_coletar(tag, conjunto="geral"):           # conjunto: geral (60 prompts) ou dominios (160, por assunto)
    ld = RES / tag / "logits"
    if not ld.is_dir():
        sys.exit(f"{ld} nao existe: inicie o teste com STRATA_ENV=\"STRATA_DUMP_FIRST_LOGITS={ld}/l\" (ver o topo)")
    antes = len(list(ld.glob("l.*")))
    mapa = []
    lista = [("geral", p) for p in prompts_qualidade()] if conjunto == "geral" else prompts_dominios()
    for i, (dom, p) in enumerate(lista):
        req("/v1/chat/completions", {"model": "t", "max_tokens": 1, "temperature": 0,
                                     "chat_template_kwargs": {"enable_thinking": False},
                                     "messages": [{"role": "user", "content": p}]})
        mapa.append({"prompt": i, "arquivo": f"l.{antes + i}", "dominio": dom})
    salva(tag, "kl-mapa.json" if conjunto == "geral" else f"kl-mapa-{conjunto}.json", mapa)


def kl_comparar(ref, *tags):                     # KL_MAPA=kl-mapa-dominios.json para o conjunto por assunto
    import numpy as np
    def dist(tag, arq):
        x = np.fromfile(RES / tag / "logits" / arq, dtype=np.float32).astype(np.float64)
        x -= x.max(); lp = x - np.log(np.exp(x).sum()); return lp
    MAPA = os.environ.get("KL_MAPA", "kl-mapa.json"); suf = MAPA[len("kl-mapa"):-len(".json")]
    mr = json.loads((RES / ref / MAPA).read_text())
    for tag in tags:
        mt = json.loads((RES / tag / MAPA).read_text())
        kls, top1, top5 = [], 0, []
        for a, b in zip(mr, mt):
            lr, lt = dist(ref, a["arquivo"]), dist(tag, b["arquivo"])
            pr = np.exp(lr); kls.append(float((pr * (lr - lt)).sum()))
            top1 += int(lr.argmax() == lt.argmax())
            top5.append(len(set(np.argsort(-lr)[:5]) & set(np.argsort(-lt)[:5])) / 5)
        k = np.array(kls)
        res = {"ref": ref, "tag": tag, "n": len(kls), "kl_media": float(k.mean()), "kl_mediana": float(np.median(k)),
               "kl_p90": float(np.percentile(k, 90)), "kl_max": float(k.max()), "mesmo_top1": f"{top1}/{len(kls)}",
               "top5_sobreposicao": float(np.mean(top5))}
        doms = [a.get("dominio", "geral") for a in mr]
        if len(set(doms)) > 1:
            res["por_dominio"] = {d: {"n": sum(x == d for x in doms),
                                      "kl_media": float(np.mean([k[j] for j, x in enumerate(doms) if x == d])),
                                      "kl_mediana": float(np.median([k[j] for j, x in enumerate(doms) if x == d]))}
                                  for d in dict.fromkeys(doms)}
        print(res); salva(tag, f"kl{suf}-vs-{ref}.json", res)


def _chat(body, timeout=7200):
    """um pedido nao-streaming; devolve (resposta, segundos no cliente, ultimo pedido de /metrics)"""
    t0 = time.time(); r = req("/v1/chat/completions", body, timeout); dt = time.time() - t0
    return r, dt, ultimo()


def _linha(r, dt, u, **extra):
    t = r.get("timings") or {}
    row = {"prompt_tokens": r["usage"]["prompt_tokens"], "gerados": r["usage"]["completion_tokens"],
           "reaproveitados": u.get("reused"), "lidos": u.get("prompt_read"),
           "prompt_tok_s": t.get("prompt_per_second") or (round(u["prompt_read"] / u["prompt_ms"] * 1000, 1) if u.get("prompt_ms") else None),
           "decode_tok_s": t.get("predicted_per_second") or u.get("decode_tok_s"),
           "prompt_ms": u.get("prompt_ms"), "decode_ms": u.get("decode_ms"), "total_s": round(dt, 2),
           "ttft_s": round(u["prompt_ms"] / 1000, 2) if u.get("prompt_ms") else None, "finish": u.get("finish"),
           "hit_rate": u.get("hit_rate"), "pcie_share": u.get("pcie_share"),
           "drafts": [u.get("drafts_accepted"), u.get("drafts_offered")],
           "ssd_mb": u.get("file_mb"), "ssd_blobs": u.get("file_blobs"), "ram_blobs": u.get("ram_blobs")}
    row.update(extra); return row


def sweep(tag, lens="4096,32768,128000", runs="3"):
    """padrao docs/COMMUNITY_BENCHMARKS.md e bench_sweep.py (relatorio 786K): texto sintetico repetido com marcador de
    revisao diferente por corrida (nada reaproveitado), 256 tokens, temperatura 0, raciocinio desligado, nao-streaming"""
    from needle_bench import haystack, CHARS_PER_TOKEN
    linhas = []
    for L in [int(x) for x in lens.split(",")]:
        for k in range(int(runs)):
            txt = f"[revisao {tag}-{L}-{k}-{random.randrange(10**9)}]\n" + haystack(int(L * 0.97 * CHARS_PER_TOKEN))
            body = {"model": "t", "max_tokens": 256, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": txt + "\n\nSummarize the text above in detail."}]}
            r, dt, u = _chat(body); row = _linha(r, dt, u, alvo=L, corrida=k + 1)
            print(L, k + 1, row["prompt_tok_s"], row["decode_tok_s"], row["hit_rate"], flush=True); linhas.append(row)
    salva(tag, "sweep.json", linhas)


def longo(tag, frac="0.97"):
    """contexto quase cheio: 3 codigos a 10/50/90 % (como bench/results/2026-10-09-rope-yarn-long-context), leitura do
    prompt do zero; depois uma continuacao que reaproveita o prompt inteiro e gera 512 tokens (geracao com o contexto cheio)"""
    from needle_bench import haystack, CHARS_PER_TOKEN, WORDS
    ctx = int(req("/health")["max_context"]); alvo = int(ctx * float(frac))
    import zlib; rnd = random.Random(zlib.crc32(tag.encode()))
    codes = [f"{rnd.choice(WORDS).upper()}-{rnd.randint(100, 999)}" for _ in range(3)]
    base = haystack(int(alvo * CHARS_PER_TOKEN))
    for tent in range(6):
        txt = base
        for d, c in sorted(zip((90, 50, 10), reversed(codes)), reverse=True):
            cut = txt.rfind("\n", 0, int(len(txt) * d / 100)) + 1
            txt = txt[:cut] + f"\nCode number {(10, 50, 90).index(d) + 1} is {c}.\n" + txt[cut:]
        msgs = [{"role": "user", "content": txt + "\n\nReturn the three code numbers 1, 2 and 3 in order, separated by |. Nothing else."}]
        try:
            r, dt, u = _chat({"model": "t", "max_tokens": 64, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}, "messages": msgs})
            break
        except urllib.error.HTTPError as e:            # passou do contexto: corta 3 % e tenta de novo
            if e.code != 400 or tent == 5: raise
            base = base[:int(len(base) * 0.97)]
    resp = (r["choices"][0]["message"].get("content") or "").strip()
    certos = sum(c in resp for c in codes); ordem = resp.replace(" ", "") == "|".join(codes)
    leitura = _linha(r, dt, u, contexto=ctx, codigos=codes, resposta=resp[:200], certos=f"{certos}/3", ordem_exata=ordem)
    print("leitura", leitura["prompt_tokens"], leitura["prompt_tok_s"], leitura["certos"], flush=True)
    msgs += [{"role": "assistant", "content": resp}, {"role": "user", "content": "Now write a detailed summary of the whole text, about 400 words."}]
    r2, dt2, u2 = _chat({"model": "t", "max_tokens": 512, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}, "messages": msgs})
    geracao = _linha(r2, dt2, u2)
    print("geracao", geracao["reaproveitados"], geracao["decode_tok_s"], flush=True)
    salva(tag, "longo.json", {"leitura": leitura, "geracao_no_contexto_cheio": geracao})


def evalplus(tag, ids="todos", dataset="humaneval"):   # ids "" = o conjunto inteiro; dataset humaneval ou mbpp
    """HumanEval+ (EvalPlus 0.3.1, como o relatorio da RTX 3060): tarefas 0-39, uma geracao gulosa, base e plus"""
    ids = "" if ids in ("todos", "all") else ids
    d = RES / tag / ("evalplus" if dataset == "humaneval" else f"evalplus-{dataset}"); d.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OPENAI_API_KEY": KEY}; V = Path("<data>/testes-q6-q8/venv-eval/bin")
    subprocess.run([str(V / "evalplus.codegen"), "--model", "t", "--dataset", dataset, "--backend", "openai",
                    "--base_url", URL + "/v1", "--greedy", "--root", str(d)] + (["--id_range", f"[{ids}]"] if ids else []),
                   env=env, cwd=d, check=False, stdout=open(d / "codegen.log", "w"), stderr=subprocess.STDOUT)
    amostras = sorted(x for x in d.rglob("*.jsonl") if ".raw." not in x.name)   # a versao limpa (o .raw. tem as cercas ```)
    if not amostras: print("evalplus: sem amostras (ver codegen.log)"); return
    subprocess.run([str(V / "evalplus.evaluate"), "--dataset", dataset, "--samples", str(amostras[-1])],
                   env=env, cwd=d, check=False, stdout=open(d / "evaluate.log", "w"), stderr=subprocess.STDOUT)
    print("evalplus:", (d / "evaluate.log").read_text()[-400:])


def lmeval(tag, tarefas="gsm8k,ifeval", limite="100", gen="", nome="lm-eval"):
    """lm-evaluation-harness pela API (local-chat-completions: so tarefas geradas), amostra fixa de N por tarefa.
    gen: --gen_kwargs (ex. max_gen_toks=4096; o padrao do lm-eval e 256, corta o raciocinio do BBH e do MATH); nome: a pasta"""
    if tarefas == "gsm8k,ifeval" and not gen:    # o GSM8K com o limite padrao (256) corta respostas certas antes do "####":
        lmeval(tag, "gsm8k", limite, "max_gen_toks=4096", "lm-eval-gsm8k-4k")   # vai a parte, com 4096; o IFEval ja tem 1280
        tarefas = "ifeval"
    d = RES / tag / nome; d.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OPENAI_API_KEY": KEY}
    subprocess.run(["<data>/testes-q6-q8/venv-eval/bin/lm_eval", "--model", "local-chat-completions", "--model_args",
                    f"model=t,base_url={URL}/v1/chat/completions,num_concurrent=1,max_retries=3,tokenized_requests=False",
                    "--tasks", tarefas, "--apply_chat_template", "--limit", limite, "--seed", "1234",
                    "--output_path", str(d), "--log_samples"] + (["--gen_kwargs", gen] if gen else []), env=env, cwd=d, check=False,
                   stdout=open(d / "lm-eval.log", "w"), stderr=subprocess.STDOUT)
    print("lm-eval:", "\n".join(l for l in (d / "lm-eval.log").read_text().splitlines() if "|" in l)[-1500:])


def resumo():
    import statistics as st
    J = lambda p: json.loads(p.read_text()) if p.exists() else None
    def med(v): v = [x for x in v if x is not None]; return (st.median(v), min(v), max(v)) if v else None
    fmt = lambda m, d=0: "-" if not m else (f"{m[0]:.{d}f} ({m[1]:.{d}f}-{m[2]:.{d}f})")
    tags = sorted(p.name for p in RES.iterdir() if p.is_dir())
    L = ["# Resumo das medidas (gerado por scripts/medir.py resumo)", "",
         "## Velocidade, sweep do guia (3 corridas, mediana e faixa; 256 tokens, temperatura 0, raciocinio desligado)", "",
         "| teste | 4K prompt tok/s | 4K geracao | 32K prompt | 32K geracao | 128K prompt | 128K geracao | acerto cache 128K |",
         "|---|---|---|---|---|---|---|---|"]
    for t in tags:
        sw = J(RES / t / "sweep.json")
        if not sw: continue
        c = []
        for Lx in (4096, 32768, 128000):
            rs = [r for r in sw if r["alvo"] == Lx]
            c += [fmt(med([r["prompt_tok_s"] for r in rs])), fmt(med([r["decode_tok_s"] for r in rs]), 1)]
        c.append(fmt(med([r["hit_rate"] for r in sw if r["alvo"] == 128000]), 3))
        L.append(f"| {t} | " + " | ".join(c) + " |")
    L += ["", "## Contexto quase cheio (1 corrida: leitura do zero + 3 codigos; depois geracao reaproveitando tudo)", "",
          "| teste | contexto | prompt lido | prompt tok/s | TTFT s | codigos | geracao tok/s no contexto cheio | acerto cache |",
          "|---|---:|---:|---:|---:|---|---:|---:|"]
    for t in tags:
        lg = J(RES / t / "longo.json")
        if not lg: continue
        a, b = lg["leitura"], lg["geracao_no_contexto_cheio"]
        L.append(f"| {t} | {a['contexto']:,} | {a['prompt_tokens']:,} | {a['prompt_tok_s']} | {a['ttft_s']} | {a['certos']}"
                 f"{' ordem exata' if a['ordem_exata'] else ''} | {b['decode_tok_s']} | {b['hit_rate']} |")
    L += ["", "## Partida (log do servidor)", ""]
    for t in tags:
        i = J(RES / t / "info.json")
        if i: L.append(f"- **{t}**: " + " / ".join(x.strip()[:110] for x in i["log"] if re.search(r"expert cache|VRAM free|KV streaming", x))[:400])
    L += ["", "## Qualidade: KL do 1o token contra a referencia (60 prompts; menor = mais perto)", "",
          "| teste | ref | KL media | mediana | p90 | max | mesmo top-1 | top-5 |", "|---|---|---:|---:|---:|---:|---|---:|"]
    for f in sorted(RES.glob("*/kl-vs-*.json")):
        r = J(f); L.append(f"| {r['tag']} | {r['ref']} | {r['kl_media']:.4f} | {r['kl_mediana']:.4f} | {r['kl_p90']:.4f} | {r['kl_max']:.3f} | {r['mesmo_top1']} | {r['top5_sobreposicao']:.2f} |")
    dom = sorted(RES.glob("*/kl-dominios-vs-*.json"))
    if dom:
        L += ["", "## Qualidade por assunto: KL media do 1o token (160 continuacoes, 40 por assunto; menor = mais perto)", "",
              "| teste | ref | python | cpp | ingles | portugues |", "|---|---|---:|---:|---:|---:|"]
        for f in dom:
            r = J(f); pd = r.get("por_dominio", {})
            L.append(f"| {r['tag']} | {r['ref']} | " + " | ".join(f"{pd[d]['kl_media']:.4f}" if d in pd else "-" for d in ("python", "cpp", "ingles", "portugues")) + " |")
    L += ["", "## Avaliacoes (raciocinio desligado)", ""]
    for t in tags:
        e = RES / t / "evalplus" / "evaluate.log"
        if e.exists():
            L.append(f"- **{t} HumanEval+**: " + " ".join(x.strip() for x in e.read_text().splitlines() if "pass@1" in x or "humaneval" in x.lower())[-300:])
        for f in sorted((RES / t / "lm-eval").rglob("results_*.json")) if (RES / t / "lm-eval").exists() else []:
            for task, v in J(f)["results"].items():
                L.append(f"- **{t} {task}**: " + ", ".join(f"{k}={x:.3f}" for k, x in v.items() if isinstance(x, float) and "stderr" not in k))
    vd = [t for t in tags if (RES / t / "visao-detalhe.json").exists()]
    if vd:
        tipos = ["texto9px", "texto12px", "texto16px", "texto24px", "contar", "tabela", "cor-posicao"]
        L += ["", "## Visao com detalhe (24 imagens com resposta conhecida; acertos por tipo)", "",
              "| teste | " + " | ".join(tipos) + " | total |", "|---|" + "---:|" * (len(tipos) + 1)]
        for t in vd:
            v = J(RES / t / "visao-detalhe.json")
            L.append(f"| {t} | " + " | ".join(f"{sum(x['acertou'] for x in v if x['tipo'] == k)}/{sum(x['tipo'] == k for x in v)}"
                                              for k in tipos) + f" | {sum(x['acertou'] for x in v)}/{len(v)} |")
        for f in sorted(RES.glob("*/kl-visao-vs-*.json")):
            r = J(f); pd = r.get("por_dominio", {})
            L.append(f"- KL {r['tag']} contra {r['ref']}: media {r['kl_media']:.4f}, mesmo top-1 {r['mesmo_top1']}; por tipo: "
                     + ", ".join(f"{d} {v['kl_media']:.4f}" for d, v in pd.items()))
    L += ["", "## Visao (imagem sintetica: 3 formas e cores)", ""]
    for t in tags:
        v = J(RES / t / "visao.json")
        if v: L.append(f"- {t}: {'OK' if v[0].get('ok') else 'FALHOU'} em {v[0]['s']} s")
    falhas = [t for t in tags if (RES / t / "FALHOU").exists()]
    if falhas: L += ["", "## Falhas", ""] + [f"- {t}: {(RES / t / 'FALHOU').read_text().strip()}" for t in falhas]
    (RES / "RESUMO.md").write_text("\n".join(L) + "\n"); print(f"RESUMO.md: {len(L)} linhas")


if __name__ == "__main__":
    cmd, *a = sys.argv[1:] or ["help"]
    fn = {"info": info, "velocidade": velocidade, "agulhas": agulhas, "visao": lambda t, *i: visao(t, i),
          "kl-coletar": kl_coletar, "visao-detalhe": visao_detalhe, "sweep": sweep, "longo": longo, "evalplus": evalplus, "lmeval": lmeval, "kl-comparar": kl_comparar, "resumo": resumo}.get(cmd)
    if fn is None:
        print(__doc__); sys.exit(1)
    fn(*a)
