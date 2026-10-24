#!/usr/bin/env python3
"""gerar-relatorio.py - junta os resultados da bateria 0.2.0 em tabelas Markdown (em ingles, para o PR do Strata) e num
resumo JSON. So le resultados/; escreve publicacao/tabelas.md e publicacao/summary.json. Pode rodar de novo a qualquer hora.
"""
import glob, json, re
from pathlib import Path

T = Path("<data>/testes-020"); R = T / "resultados"; OUT = T / "publicacao"
MODELOS = [("UD-Q2_K_XL", "ud-q2_k_xl", 73.5), ("UD-Q3_K_XL", "ud-q3_k_xl", 83.8), ("UD-Q4_K_XL", "ud-q4_k_xl", 103.7),
           ("UD-Q5_K_XL", "ud-q5_k_xl", 147.4), ("UD-Q6_K_XL", "ud-q6_k_xl", 157.5), ("Q8_0", "q8_0", 175.3)]
EXTRAS = [("Q8_0 on 0.1.45 (reference)", "ref0145-q8_0-256k"), ("UD-Q4_K_XL, tail skip at the engine default", "020-ud-q4_k_xl-256k-tailskip7")]


def J(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def lm(tag, sub, task, metric):
    for f in sorted(glob.glob(str(R / tag / sub / "*" / "results_*.json")), reverse=True):
        r = J(f)["results"].get(task)
        if r:
            for k, v in r.items():
                if k.split(",")[0] == metric.split(",")[0] and (len(metric.split(",")) == 1 or k == metric):
                    return v
    return None


def evalplus(tag):
    try:
        p = re.findall(r"pass@1:\s+([\d.]+)", (R / tag / "evalplus" / "evaluate.log").read_text())
        return [float(x) for x in p[:2]] if len(p) >= 2 else None
    except OSError:
        return None


def bench(tag):
    f = glob.glob(str(R / tag / "bench" / "*" / "README.md"))
    if not f:
        return {}
    t = Path(f[0]).read_text(); out = {}
    for nome, col, chave in (("short chat", 6, "chat_decode"), ("4K prompt, fresh", 5, "p4k"), ("16K prompt, fresh", 5, "p16k"),
                             ("agent session, follow-up turns", 6, "agent_decode")):
        m = re.search(r"^\| " + re.escape(nome) + r" \|(.*)$", t, re.M)
        if m:
            out[chave] = m.group(1).split("|")[col - 1].strip().split(" ")[0]
    m = re.search(r"Cache hit over the follow-up turns: ([\d.]+) %", t)
    out["agent_cache_hit"] = m.group(1) if m else None
    return out


def partida(tag):   # a primeira partida do teste (arquivos frios), do log de eventos do alternar.sh
    for l in (T / "logs" / "eventos.log").read_text().splitlines():
        m = re.search(r"\] " + re.escape(tag) + r" pronto em (\d+) s", l)
        if m:
            return int(m.group(1))
    return None


def cache(tag):
    i = J(R / tag / "info.json") or {}
    s = "\n".join(i.get("log", []))
    ram = re.search(r"experts loaded: ([\d.]+) GiB", s); gpu = re.search(r"expert cache \((\d+) experts, ([\d.]+) GiB", s)
    return (float(ram.group(1)) if ram else None, int(gpu.group(1)) if gpu else None, float(gpu.group(2)) if gpu else None)


def linha(tag):
    v = J(R / tag / "velocidade.json") or []; ag = J(R / tag / "agulhas.json") or []
    kl = J(R / tag / "kl-vs-020-q8_0-256k.json") or {}; kld = J(R / tag / "kl-dominios-vs-020-q8_0-256k.json") or {}
    vram = (R / tag / "vram-pronto.txt").read_text().split(",")[0].strip() if (R / tag / "vram-pronto.txt").exists() else None
    vis = J(R / tag / "visao-detalhe.json")
    return {"tag": tag, "start_s": partida(tag), "ram_experts_gib": cache(tag)[0], "gpu_cache_experts": cache(tag)[1],
            "gpu_cache_gib": cache(tag)[2], "vram_ready": vram,
            "decode": {r["alvo"]: r["decode_tok_s"] for r in v}, "prompt": {r["alvo"]: r["prompt_tok_s"] for r in v},
            "drafts": {r["alvo"]: r["drafts"] for r in v}, "bench": bench(tag),
            "needles": f"{sum(a['found'] for a in ag)}/{len(ag)}" if ag else None,
            "kl": kl.get("kl_media"), "kl_median": kl.get("kl_mediana"), "kl_p90": kl.get("kl_p90"), "top1": kl.get("mesmo_top1"),
            "kl_domains": kld.get("kl_media"), "top1_domains": kld.get("mesmo_top1"), "kl_by_domain": kld.get("por_dominio"),
            "humaneval_plus": evalplus(tag), "gsm8k_4096": lm(tag, "lm-eval-gsm8k-4k", "gsm8k", "exact_match,strict-match"),
            "gsm8k_256": lm(tag, "lm-eval", "gsm8k", "exact_match,strict-match"),
            "ifeval_prompt_strict": lm(tag, "lm-eval", "ifeval", "prompt_level_strict_acc"),
            "ifeval_inst_strict": lm(tag, "lm-eval", "ifeval", "inst_level_strict_acc"),
            "math_geometry": lm(tag, "lm-eval-espacial", "minerva_math_geometry", "math_verify"),
            "bbh_geometric_shapes": lm(tag, "lm-eval-bbh", "bbh_cot_zeroshot_geometric_shapes", "exact_match,flexible-extract"),
            "bbh_navigate": lm(tag, "lm-eval-bbh", "bbh_cot_zeroshot_navigate", "exact_match,flexible-extract"),
            "vision_detail": {k: f"{sum(r['acertou'] for r in vis if r['tipo'] == k)}/{sum(r['tipo'] == k for r in vis)}"
                              for k in dict.fromkeys(r["tipo"] for r in vis)} if vis else None}


def p(x, d=1):
    return "-" if x is None else f"{x * 100:.{d}f}"


def n(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


linhas = [(nome, linha(f"020-{m}-256k"), gib) for nome, m, gib in MODELOS]
extras = [(nome, linha(t), None) for nome, t in EXTRAS if (R / t / "FEITO").exists() and not (R / t / "ADIADO").exists()]
L = ["### Speed (Strata 0.2.0, 262,144-token context, one request at a time)", "",
     "| Model | GGUF size (GiB) | Cold start (s) | Experts in GPU cache | Decode tok/s 8K / 32K / 128K | Prompt tok/s 8K / 32K / 128K | strata_bench short chat / agent decode | strata_bench 4K / 16K prompt |",
     "|---|---:|---:|---|---|---|---|---|"]
for nome, r, gib in linhas + extras:
    b = r["bench"]
    L.append(f"| {nome} | {gib if gib else '-'} | {r['start_s'] or '-'} | {r['gpu_cache_experts'] or '-'} ({r['gpu_cache_gib'] or '-'} GiB) | "
             + " / ".join(str(round(r["decode"].get(k, 0))) if k in r["decode"] else "-" for k in ("8k", "32k", "128k")) + " | "
             + " / ".join(str(r["prompt"].get(k, "-")) for k in ("8k", "32k", "128k")) + f" | {b.get('chat_decode', '-')} / {b.get('agent_decode', '-')} | "
             + f"{b.get('p4k', '-')} / {b.get('p16k', '-')} |")
L += ["", "### Quality against Q8_0 on the same engine (first-token logits)", "",
      "| Model | KL mean, 60 prompts | KL median / p90 | same top-1 | KL mean, 160 prompts by domain | same top-1 | needles at 128K |",
      "|---|---:|---|---|---:|---|---|"]
for nome, r, _ in linhas + extras:
    L.append(f"| {nome} | {n(r['kl'])} | {n(r['kl_median'])} / {n(r['kl_p90'])} | {r['top1'] or '-'} | {n(r['kl_domains'])} | {r['top1_domains'] or '-'} | {r['needles'] or '-'} |")
L += ["", "### Task accuracy (lm-evaluation-harness 0.4.13 and EvalPlus over the HTTP API, greedy)", "",
      "| Model | HumanEval+ base / plus (164) | GSM8K strict, 4096 tokens (100) |  IFEval prompt / instruction strict (100) | MATH geometry, math_verify (100) | BBH geometric_shapes (250) | BBH navigate (250) |",
      "|---|---|---|---|---|---|---|"]
for nome, r, _ in linhas + extras:
    he = r["humaneval_plus"]
    L.append(f"| {nome} | {p(he[0]) + ' / ' + p(he[1]) if he else '-'} | {p(r['gsm8k_4096'], 0)} | "
             f"{p(r['ifeval_prompt_strict'], 0)} / {p(r['ifeval_inst_strict'], 1)} | {p(r['math_geometry'], 0)} | {p(r['bbh_geometric_shapes'])} | {p(r['bbh_navigate'])} |")
vis = [(nome, r["vision_detail"]) for nome, r, _ in linhas + extras if r["vision_detail"]]
if vis:
    tipos = list(vis[0][1])
    L += ["", "### Image input (synthetic images, vision encoder on the GPU)", "", "| Model | " + " | ".join(tipos) + " |", "|---|" + "---|" * len(tipos)]
    L += [f"| {nome} | " + " | ".join(v.get(t, "-") for t in tipos) + " |" for nome, v in vis]
OUT.mkdir(exist_ok=True)
(OUT / "tabelas.md").write_text("\n".join(L) + "\n")
(OUT / "summary.json").write_text(json.dumps({nome: r for nome, r, _ in linhas + extras}, indent=1) + "\n")
print("\n".join(L))
