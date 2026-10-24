#!/usr/bin/env python3
"""montar-publicacao.py - (bateria 0.2.1) monta a pasta do PR (bench/results/<PASTA>) em publicacao/<PASTA>, a partir de resultados/,
logs/, configs/ e scripts/. So copia e sanitiza: caminhos locais viram <data>/<home>, o usuario e o dominio da instituicao
saem, a chave de teste sai dos configs. Os logits brutos (centenas de MB) ficam de fora (TRIMMED.md). No fim procura
qualquer resto de nome, dominio ou chave e para com erro se achar.
"""
import gzip, json, re, shutil, subprocess, sys, urllib.request
from pathlib import Path

T = Path("<data>/testes-021"); R = T / "resultados"; PASTA = "2026-10-21-community-l40s-48q-0.2.1-unsloth-quants-3d"
D = T / "publicacao" / PASTA
TAGS = ["021-ud-q2_k_xl-256k", "021-ud-q3_k_xl-256k", "021-ud-q4_k_xl-256k", "021-ud-q5_k_xl-256k", "021-ud-q6_k_xl-256k",
        "021-q8_0-256k"] + [f"ab{v}-q4-r{i}" for i in (1, 2, 3) for v in ("020", "021")]
CHAVE = (T / "configs" / ".api_key-teste").read_text().strip()
TROCAS = [("<data>/", "<data>/"), ("<home>", "<home>"), ("<user>", "<user>"),
          ("<group>", "<group>"), ("<user>", "<user>"), ("<domain>", "<domain>"), (CHAVE, "<test-key>")]


def limpa(s):
    for a, b in TROCAS:
        s = s.replace(a, b)
    return s


def texto(src, dst, gz=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    s = limpa(Path(src).read_text(errors="replace"))
    if gz:
        with gzip.open(str(dst) + ".gz", "wt") as f:
            f.write(s)
    else:
        dst.write_text(s)


if D.exists():
    shutil.rmtree(D)
D.mkdir(parents=True)
for tag in TAGS:
    src = R / tag
    if not (src / "FEITO").exists() or (src / "ADIADO").exists():
        print("sem resultados (pulado):", tag); continue
    out = D / "results" / tag
    for f in src.glob("*.json"):
        texto(f, out / f.name)
    for f in ("vram-pronto.txt", "ram-pronto.txt", "vram-final.txt", "ram-final.txt", "tailskip-log.txt"):
        if (src / f).exists():
            texto(src / f, out / f)
    for f in (src / "bench").glob("*/*"):
        if f.suffix in (".md", ".json", ".txt"):
            texto(f, out / "strata_bench" / f.name)
    for sub in src.glob("lm-eval*"):
        for f in sub.rglob("*.json*"):
            texto(f, out / sub.name / f.name, gz=f.name.startswith("samples_"))
        if (sub / "lm-eval.log").exists():
            s = (sub / "lm-eval.log").read_text(errors="replace")
            texto_l = "\n".join(l for l in s.splitlines() if "Requesting API" not in l)
            (out / sub.name).mkdir(parents=True, exist_ok=True); (out / sub.name / "lm-eval.log").write_text(limpa(texto_l) + "\n")
    if (src / "evalplus").exists():
        texto(src / "evalplus" / "evaluate.log", out / "evalplus" / "evaluate.log")
        for f in (src / "evalplus").rglob("*.jsonl"):
            texto(f, out / "evalplus" / f.name, gz=True)
        for f in (src / "evalplus").rglob("*eval_results.json"):
            texto(f, out / "evalplus" / f.name, gz=True)
    if (src / "bench3d").exists():   # 3D: resultado, e por tarefa a resposta do modelo e o script que o Blender rodou
        for f in (src / "bench3d").iterdir():
            texto(f, out / "bench3d" / f.name)
        if (src / "bench3d.log").exists():
            texto(src / "bench3d.log", out / "bench3d" / "bench3d.log")
    cfg = T / "configs" / f"strata-teste-{tag}.json"
    if tag.startswith("ab"):
        cfg = T / "configs" / f"strata-teste-{tag[:-3]}.json"
    if cfg.exists():
        c = json.loads(cfg.read_text()); c["api_key"] = "<test-key>"
        (D / "configs").mkdir(exist_ok=True); (D / "configs" / f"{tag}.json").write_text(limpa(json.dumps(c, indent=1)) + "\n")
    for nome in (f"engine-{tag}.log", f"servidor-{tag}.log", f"engine-{tag[:-3]}.log", f"servidor-{tag[:-3]}.log"):
        if (T / "logs" / nome).exists():
            texto(T / "logs" / nome, D / "logs" / nome, gz=True)
for f in ("pipeline-eventos.log", "eventos.log", "noite.log") + tuple(p.name for p in (T / "logs").glob("pack-*.log")):
    texto(T / "logs" / f, D / "logs" / f)
texto(Path("<data>/strata-021/scripts/baixar.sh"), D / "scripts" / "baixar.sh")
for f in ("pipeline.sh", "noite.sh", "alternar.sh", "medir.py", "gerar-configs.py", "empacotar.sh", "bench3d.py", "executor3d.py", "tarefas3d.py",
          "gerar-relatorio.py", "montar-publicacao.py", "manifesto.txt", "prompts-qualidade.json"):
    if (T / "scripts" / f).exists():
        texto(T / "scripts" / f, D / "scripts" / f)
# manifesto dos GGUF: tamanhos e SHA-256 do Hub na revisao fixa (os arquivos locais sao essa revisao)
REV = "766911a6b7369840a91dbcd95f9f997acaab6cd6"
api = f"https://huggingface.co/api/models/unsloth/Qwen3.8-Flash-Next-GGUF/tree/{REV}?recursive=true"
arv = json.load(urllib.request.urlopen(api, timeout=60))
q = ("UD-Q2_K_XL", "UD-Q3_K_XL", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0")
man = {"repository": "unsloth/Qwen3.8-Flash-Next-GGUF", "revision": REV,
       "files": [{"path": x["path"], "size": x["size"], "sha256": (x.get("lfs") or {}).get("oid")} for x in arv
                 if x.get("type") == "file" and x["path"].endswith(".gguf") and any(x["path"].startswith(m + "/") for m in q)]}
(D / "model-manifest.json").write_text(json.dumps(man, indent=1) + "\n")
subprocess.run([sys.executable, str(T / "scripts" / "gerar-relatorio.py")], check=True, stdout=subprocess.DEVNULL)
shutil.copy(T / "publicacao" / "summary.json", D / "summary.json"); shutil.copy(T / "publicacao" / "tabelas.md", D / "tables.md")
# procura restos
ruim = re.compile("|".join(["tal" + "is", "ude" + "sc", "/CER" + "ES", "00469" + "10"]) + "|" + re.escape(CHAVE), re.I)   # em pedacos, para nao achar a si mesmo
achou = []
for f in D.rglob("*"):
    if f.is_file():
        s = gzip.open(f, "rt", errors="replace").read() if f.suffix == ".gz" else f.read_text(errors="replace")
        for m in ruim.finditer(s):
            achou.append(f"{f.relative_to(D)}: ...{s[max(0, m.start() - 40):m.end() + 40]!r}...")
if achou:
    print("RESTOS ENCONTRADOS:"); print("\n".join(achou[:30])); sys.exit(1)
tam = sum(f.stat().st_size for f in D.rglob("*") if f.is_file())
print(f"ok: {sum(1 for f in D.rglob('*') if f.is_file())} arquivos, {tam / 1e6:.1f} MB em {D}")
