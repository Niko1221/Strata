#!/usr/bin/env python3
"""gerar-configs.py - configs de teste da bateria 0.2.1 (porta 8090). Copia de testes-020/scripts/gerar-configs.py com o motor e o
servidor do 0.2.1 (strata-021: motor 0.1.46, control 0.1.46.2) e os packs feitos pela ferramenta do 0.2.1 (testes-021/packs).
Mesmas opcoes do 0.2.0 para a comparacao: copiadas do config da producao (so lido), contexto 262.144 sem YaRN, KV fp16, MTP,
visao na GPU, reserva 700 MiB, e o "env" que o setup grava para arquivos Unsloth (setup.py unsloth_default_env, igual no 0.2.1):
  UD-Q5/Q6: STRATA_MMQ_KQ=1;  todos os UD-*: STRATA_HC_Q8=1;  todos: STRATA_ROUTE_TAIL_SKIP=0

Escreve em configs/:
  strata-teste-021-<m>-256k.json   m = ud-q2_k_xl ud-q3_k_xl ud-q4_k_xl ud-q5_k_xl ud-q6_k_xl q8_0
  strata-teste-ab020-q4.json / strata-teste-ab021-q4.json   UD-Q4_K_XL no 0.2.0 e no 0.2.1 para as partidas alternadas
A chave de teste e a mesma da bateria 0.2.0 (configs/.api_key-teste, nunca impressa).
"""
import json, os
from pathlib import Path

T = Path("<data>/testes-021"); CFG = T / "configs"; M = Path("<data>/testes-020/modelos")
V021 = {"exe": Path("<data>/strata-021/engine"), "src": Path("<data>/strata-021/control"), "packs": T / "packs"}
V020 = {"exe": Path("<data>/strata-020/engine"), "src": Path("<data>/strata-020/control"),
        "packs": Path("<data>/testes-020/packs")}
ATUAL = Path("<data>/strata/strata-q8_0-fp16-1m-0145.json")
GGUF = {"ud-q2_k_xl": M / "UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf",
        "ud-q3_k_xl": M / "UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf",
        "ud-q4_k_xl": Path("<data>/Strata-data/models/unsloth-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"),
        "ud-q5_k_xl": M / "UD-Q5_K_XL/Qwen3.8-Flash-Next-UD-Q5_K_XL-00001-of-00006.gguf",
        "ud-q6_k_xl": Path("<data>/testes-q6-q8/modelos/UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf"),
        "q8_0": Path("<data>/testes-q6-q8/modelos/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf")}
base = json.loads(ATUAL.read_text())
chave = (CFG / ".api_key-teste").read_text().strip()


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


def config(m, versao, tag):
    c = json.loads(json.dumps(base)); a = c["args"]; pack = versao["packs"] / m
    for k, v in {"--pack": str(pack), "--native": str(GGUF[m]), "--mtp": "<data>/strata-0145/mtp/rt",
                 "--expert-profile": str(versao["src"] / "data/expert-profile.bin"), "--max-context": "262144"}.items():
        opcao(a, k, v)
    opcao(a, "--rope-scaling"); opcao(a, "--rope-scale")
    c.update(exe=str(versao["exe"] / "strata"), cwd=str(versao["src"]), tokenizer=str(pack / "tokenizer"), env=env_setup(m),
             port=8090, host="127.0.0.1", api_key=chave, model_name=f"qwen3.8-flash-next-teste-{tag}",
             log=str(T / "logs" / f"engine-{tag}.log"))
    c["lib_dirs"] = [str(versao["exe"])] + c.get("lib_dirs", [])
    c["vision"].update(exe=str(versao["exe"] / "strata-vision"), model=str(GGUF[m]))
    c.pop("aliases", None)
    return c


def grava(tag, c):
    p = CFG / f"strata-teste-{tag}.json"
    p.write_text(json.dumps(c, indent=1) + "\n"); os.chmod(p, 0o600); print("escrito", p)   # contem a chave de teste


for m in GGUF:
    grava(f"021-{m}-256k", config(m, V021, f"021-{m}-256k"))
grava("ab020-q4", config("ud-q4_k_xl", V020, "ab020-q4"))
grava("ab021-q4", config("ud-q4_k_xl", V021, "ab021-q4"))
