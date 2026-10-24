#!/usr/bin/env python3
"""gerar-configs.py - configs de teste da bateria 0.2.0 (porta 8090), todos copiados do config da producao atual
(<data>/strata/strata-q8_0-fp16-1m-0145.json, so lido). Nao mexe na producao.

Mesmas opcoes para todos (KV fp16, MTP --spec 4, KV streaming 32.768, cache de conversas, visao na GPU, reserva 700 MiB),
contexto 262.144 (nativo, sem YaRN) para a comparacao entre modelos; o bloco "env" e o que o setup da 0.2.0 escreve para
arquivos Unsloth (setup.py unsloth_default_env, todos os experts na RAM):
  UD-Q5/Q6: STRATA_MMQ_KQ=1;  todos os UD-*: STRATA_HC_Q8=1;  todos: STRATA_ROUTE_TAIL_SKIP=0

Escreve em configs/:
  strata-teste-020-<m>-256k.json      m = ud-q2_k_xl ud-q3_k_xl ud-q4_k_xl ud-q5_k_xl ud-q6_k_xl q8_0  (motor 0.2.0)
  strata-teste-ref0145-q8_0-256k.json  a referencia: o mesmo Q8_0 no motor e servidor 0.1.45 de producao
  strata-teste-020-<m>-1m.json         1.048.576 com YaRN 4 (as opcoes da producao) para o modelo escolhido (--longo m)
Chave da API de teste: configs/.api_key-teste (criada aqui, nunca impressa).
"""
import json, os, secrets, sys
from pathlib import Path

T = Path("<data>/testes-020"); CFG = T / "configs"
NOVO = {"exe": Path("<data>/strata-020/engine"), "src": Path("<data>/strata-020/control")}
V145 = {"exe": Path("<data>/strata-0145/engine"), "src": Path("<data>/strata-0145/strata-src")}
ATUAL = Path("<data>/strata/strata-q8_0-fp16-1m-0145.json")
GGUF = {"ud-q2_k_xl": T / "modelos/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf",
        "ud-q3_k_xl": T / "modelos/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf",
        "ud-q4_k_xl": Path("<data>/Strata-data/models/unsloth-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"),
        "ud-q5_k_xl": T / "modelos/UD-Q5_K_XL/Qwen3.8-Flash-Next-UD-Q5_K_XL-00001-of-00006.gguf",
        "ud-q6_k_xl": Path("<data>/testes-q6-q8/modelos/UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf"),
        "q8_0": Path("<data>/testes-q6-q8/modelos/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf")}
base = json.loads(ATUAL.read_text())


def env_setup(m):
    e = {}
    if m in ("ud-q5_k_xl", "ud-q6_k_xl"):
        e["STRATA_MMQ_KQ"] = "1"
    if m.startswith("ud-"):
        e["STRATA_HC_Q8"] = "1"
    e["STRATA_ROUTE_TAIL_SKIP"] = "0"
    return e


def opcao(a, k, v=None):                     # troca o valor de k (ou tira k e seu valor com v=None)
    if k not in a:
        return
    i = a.index(k)
    if v is None:
        del a[i:i + 2]
    else:
        a[i + 1] = v


def config(m, versao, tag, ctx):
    c = json.loads(json.dumps(base)); a = c["args"]
    pack = (T / "packs" / m) if versao is NOVO else Path("<data>/strata-0145/packs/q8_0")
    for k, v in {"--pack": str(pack), "--native": str(GGUF[m]), "--mtp": "<data>/strata-0145/mtp/rt",
                 "--expert-profile": str(versao["src"] / "data/expert-profile.bin"), "--max-context": str(ctx)}.items():
        opcao(a, k, v)
    if ctx <= 262144:
        opcao(a, "--rope-scaling"); opcao(a, "--rope-scale")
    c.update(exe=str(versao["exe"] / "strata"), cwd=str(versao["src"]), tokenizer=str(pack / "tokenizer"), env=env_setup(m),
             port=8090, host="127.0.0.1", api_key=chave, model_name=f"qwen3.8-flash-next-teste-{tag}",
             log=str(T / "logs" / f"engine-{tag}.log"))
    if versao is NOVO:
        c["lib_dirs"] = [str(NOVO["exe"])] + c.get("lib_dirs", [])
    c["vision"].update(exe=str(versao["exe"] / "strata-vision"), model=str(GGUF[m]))
    c.pop("aliases", None)
    return c


def grava(tag, c):
    p = CFG / f"strata-teste-{tag}.json"
    p.write_text(json.dumps(c, indent=1) + "\n"); os.chmod(p, 0o600); print("escrito", p)   # contem a chave de teste


CFG.mkdir(exist_ok=True); kf = CFG / ".api_key-teste"
if not kf.exists():
    kf.write_text(secrets.token_urlsafe(32) + "\n"); os.chmod(kf, 0o600)
chave = kf.read_text().strip()
for m in GGUF:
    grava(f"020-{m}-256k", config(m, NOVO, f"020-{m}-256k", 262144))
grava("ref0145-q8_0-256k", config("q8_0", V145, "ref0145-q8_0-256k", 262144))
if "--longo" in sys.argv:
    m = sys.argv[sys.argv.index("--longo") + 1]
    grava(f"020-{m}-1m", config(m, NOVO, f"020-{m}-1m", 1048576))
