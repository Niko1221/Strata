#!/usr/bin/env python3
"""MTP (multi-token-prediction / speculative decoding) benchmark for the Strata engine.

Based on mtp-bench.py by Aman Gupta https://gist.github.com/am17an

Two things had to change to benchmark Strata properly:

1. IDENTITY. Strata's /props includes a "model_path" key, which makes a llama.cpp-style
   probe mislabel Strata as llama-server. Strata is therefore detected FIRST, from
   /health -> {"service": "strata"} (authoritative), with /props "build_info"
   ("Strata 0.1.48") and /metrics -> engine.version as corroboration. Strata's engine
   block also reports HOW it drafts (engine.drafting = "mtp", plus spec/mtp_max/lookup/kv),
   which is printed before benchmarking.

2. TRANSPORT. Strata has no llama.cpp legacy endpoint: POST /completion returns 404.
   Its dialects are /v1/chat/completions (OpenAI), /v1/messages (Anthropic), /v1/responses.
   The benchmark sends OpenAI chat requests (messages[] + max_tokens + temperature + seed)
   and reads r["timings"], which Strata emits in llama.cpp's own field names, including
   draft_n / draft_n_accepted (serve/server.py: request_timings). So the accept-rate
   arithmetic is unchanged, and the llama.cpp path is kept for engines that do have it.

Strata extras: the conversation cache is on by default, so per-request cache_n (prompt
tokens the cache already held) is reported - prompt tok/s is not comparable across runs
without it. /metrics "totals" is snapshotted before and after the run, giving the
server's own counts for this run's window (requests, output_tokens, drafts_offered,
drafts_accepted) as an independent check on the per-request numbers.
"""

import argparse, json, sys, time
from urllib import request

PROMPTS = [
    {"name": "code_python",      "prompt": "Write a Python function that returns the n-th Fibonacci number using memoization. Include a docstring."},
    {"name": "code_cpp",         "prompt": "Write a C++ template function `clamp(x, lo, hi)` that returns x clamped to [lo, hi]. No std::clamp."},
    {"name": "explain_concept",  "prompt": "Explain how speculative decoding works in large language model inference, in three short paragraphs."},
    {"name": "summarize",        "prompt": "Summarize in two sentences: The Industrial Revolution began in Britain in the late 18th century, transforming manufacturing through mechanization, steam power, and the factory system. It spread to continental Europe and North America during the 19th century."},
    {"name": "qa_factual",       "prompt": "Q: What are the four fundamental forces of physics?\nA:"},
    {"name": "translation",      "prompt": "Translate to French: 'The quick brown fox jumps over the lazy dog.'"},
    {"name": "creative_short",   "prompt": "Write a four-line poem about an old lighthouse."},
    {"name": "stepwise_math",    "prompt": "Solve step by step: A train leaves station A at 60 km/h. Two hours later, a second train leaves the same station on the same track at 90 km/h. How long until the second train catches the first?"},
    {"name": "long_code_review", "prompt": (
        "You are reviewing a backend service that has been suffering intermittent latency spikes "
        "in production. Below is the relevant code and a description of the system. After reading "
        "carefully, produce a structured review with three sections: (1) likely root causes ranked "
        "by probability, (2) concrete code or configuration changes you would make first, "
        "(3) what telemetry you would add to confirm the diagnosis.\n\n"
        "System description: a Python FastAPI service in front of a Postgres 15 database, deployed "
        "as four replicas behind an nginx load balancer. Each request reads a user record, fetches "
        "their last 50 events from a partitioned events table, computes an aggregate score, writes "
        "the score back to the user row, and returns a JSON response. Average payload is 4 KB. "
        "p50 latency is 35 ms; p99 spikes to 1.8 seconds approximately every 90 seconds in a "
        "regular pattern. The spikes correlate with elevated Postgres CPU but not with elevated "
        "Postgres connection count. The application pool is sized at 20 connections per replica. "
        "PgBouncer is in front of Postgres in transaction pooling mode with a pool size of 50.\n\n"
        "Code excerpt — the hot endpoint:\n"
        "```python\n@app.post('/score/{user_id}')\nasync def score(user_id: int, payload: ScoreRequest):\n"
        "    async with db.transaction() as tx:\n        user = await tx.fetchrow(\n"
        "            'SELECT id, tier, last_score FROM users WHERE id = $1 FOR UPDATE',\n            user_id,\n        )\n"
        "        if user is None:\n            raise HTTPException(404)\n        events = await tx.fetch(\n"
        "            'SELECT type, weight, ts FROM events '\n            'WHERE user_id = $1 ORDER BY ts DESC LIMIT 50',\n            user_id,\n        )\n"
        "        new_score = compute_score(user['tier'], events, payload.signals)\n"
        "        await tx.execute(\n            'UPDATE users SET last_score = $1, updated_at = now() WHERE id = $2',\n            new_score, user_id,\n        )\n"
        "        await tx.execute(\n            'INSERT INTO score_history (user_id, score, ts) VALUES ($1, $2, now())',\n            user_id, new_score,\n        )\n"
        "    await cache.set(f'score:{user_id}', new_score, ex=300)\n"
        "    metrics.histogram('score.latency_ms').observe((time.time() - start) * 1000)\n"
        "    return {'user_id': user_id, 'score': new_score}\n```\n\n"
        "Schema notes: `users` is ~50M rows, `events` is partitioned by month with ~2B rows total "
        "and a btree index on `(user_id, ts DESC)`. `score_history` is unpartitioned, ~800M rows, "
        "with a single index on `user_id`. Postgres autovacuum is at default settings. There is "
        "a nightly batch job that rebuilds materialized views starting at 02:00 UTC; spikes occur "
        "throughout the day, not just during the batch window. Connection pooling metrics show "
        "PgBouncer waiting connections occasionally hit 8-12 during spikes but never saturate. "
        "CPU on the FastAPI replicas stays below 30% even during spikes. Network round-trip time "
        "between the application and Postgres is consistently 0.4 ms.\n\nBegin your review now."
    )},
]

# --- identity probing ---------------------------------------------------------------------------
# Strata:  /health -> {"service": "strata", "model": ...}, /props -> build_info "Strata 0.1.48",
#          /metrics -> engine.version, engine.drafting, spec, mtp_max, lookup, kv, totals
# llama.cpp: /props (model_path, default_generation_params), /server_info
# vLLM: /version ; Ollama: /api/version, /api/tags ; TGI: /info ; generic: /v1/models
PROBE_ENDPOINTS = ("/health", "/props", "/server_info", "/version", "/api/version",
                   "/api/tags", "/info", "/metrics", "/v1/models")

# Strata engine settings worth reporting in a speculative-decoding benchmark.
STRATA_ENGINE_KEYS = ("model", "version", "drafting", "drafting_note", "spec", "mtp_max", "lookup",
                      "spec_min_p", "kv", "kv_resident", "context", "max_context", "expert_slots",
                      "expert_cache_mib", "pool_workers", "pcie_frac", "prefill_cap",
                      "conversation_cache_slots", "conversation_cache_mib", "logit_bias")

def _get(url, timeout):
    """GET and JSON-decode; never raises. Returns a dict, a list, or {'__error__': ...}."""
    try:
        with request.urlopen(request.Request(url, method="GET"), timeout=timeout) as r:
            body = r.read()
    except Exception as e:
        return {"__error__": f"{type(e).__name__}: {e}"}
    try:
        return json.loads(body)
    except Exception:
        return {"__error__": f"non-JSON response ({len(body)} bytes)"}

def _ok(d):
    return isinstance(d, dict) and "__error__" not in d and "__raw__" not in d

def _short_name(path):
    return path.rstrip("/").split("/")[-1] if path else None

def _model_from_models(doc):
    """Pull model ids out of an OpenAI-style /v1/models payload."""
    if not isinstance(doc, dict):
        return None, []
    data = doc.get("data") or doc.get("models") or []
    ids = []
    for m in data:
        if isinstance(m, dict):
            mid = m.get("id") or m.get("name") or m.get("model")
            if mid:
                ids.append(str(mid))
        elif isinstance(m, str):
            ids.append(m)
    return (ids[0] if ids else None), ids

def probe_target(base, timeout):
    """Identify the inference engine, the served LLM, and the request dialect to benchmark."""
    ep = {p: _get(base + p, timeout) for p in PROBE_ENDPOINTS}
    engine = engine_version = model = model_path = engine_homepage = dialect = None
    engine_settings, notes = {}, []
    served = _model_from_models(ep.get("/v1/models"))[1]

    health, props, sinfo = ep.get("/health"), ep.get("/props"), ep.get("/server_info")
    version, ovar, tags = ep.get("/version"), ep.get("/api/version"), ep.get("/api/tags")
    tgi, models, metrics = ep.get("/info"), ep.get("/v1/models"), ep.get("/metrics")

    # --- Strata first: its /props carries a model_path, so a llama.cpp-shaped probe would
    # label it llama-server. /health "service" is authoritative.
    if _ok(health) and str(health.get("service", "")).lower() == "strata":
        engine = "Strata"
        # canonical repo, from Strata's own source: serve/server.py issue-report strings and
        # setup.py PREBUILT_URL (github.com/Niko1221/Strata/releases/latest/download/)
        engine_homepage = "https://github.com/Niko1221/Strata"
        dialect = "strata-chat"
        info = metrics.get("engine", {}) if _ok(metrics) else {}
        engine_version = info.get("version") or (props.get("build_info") if _ok(props) else None)
        model = (health.get("model") or (props.get("model_alias") if _ok(props) else None)
                 or _model_from_models(models)[0])
        model_path = props.get("model_path") if _ok(props) else None
        engine_settings = {k: info[k] for k in STRATA_ENGINE_KEYS if k in info}
        if _ok(metrics) and _ok(metrics.get("hardware_static")):
            hs = metrics["hardware_static"]
            engine_settings["gpu_name"] = hs.get("gpu_name")
            engine_settings["gpu_vram_total_mib"] = hs.get("gpu_vram_total_mib")
            engine_settings["cpu_name"] = hs.get("cpu_name")
        if _ok(props) and _ok(props.get("default_generation_settings")):
            engine_settings["sampling_defaults"] = props["default_generation_settings"].get("params")
            engine_settings["n_ctx"] = props["default_generation_settings"].get("n_ctx")
        notes.append(f"/health: service=strata model={health.get('model')} "
                     f"max_context={health.get('max_context')} loaded={health.get('loaded')}")
        notes.append(f"request dialect: POST /v1/chat/completions (Strata has no llama.cpp "
                     f"POST /completion — that route 404s)")
        if engine_settings.get("drafting"):
            notes.append(f"drafting method: {engine_settings.get('drafting')}"
                         f"{' — ' + engine_settings['drafting_note'] if engine_settings.get('drafting_note') else ''}"
                         f" (spec={engine_settings.get('spec')} mtp_max={engine_settings.get('mtp_max')}"
                         f" lookup={engine_settings.get('lookup')})")
    # --- llama.cpp llama-server
    elif _ok(sinfo) and any(k in sinfo for k in ("build_commit", "build_info", "model_path", "slot_info", "total_slots")):
        engine, dialect = "llama.cpp llama-server", "llama-completion"
        engine_homepage = "https://github.com/ggml-org/llama.cpp"
        engine_version = sinfo.get("build_commit") or sinfo.get("build_info")
        model = _short_name(sinfo.get("model_path"))
        model_path = sinfo.get("model_path")
        notes.append(f"/server_info: model_type={sinfo.get('model_type')} total_slots={sinfo.get('total_slots')}")
    elif _ok(props) and any(k in props for k in ("model_path", "default_generation_params", "tokenizer.ggml.model")):
        engine, dialect = "llama.cpp llama-server", "llama-completion"
        engine_homepage = "https://github.com/ggml-org/llama.cpp"
        engine_version = props.get("version")
        model = _short_name(props.get("model_path"))
        model_path = props.get("model_path")
        notes.append(f"/props: total_params={props.get('total_params')} tokenizer={props.get('tokenizer.ggml.model')}")
    elif _ok(version) and any(k in version for k in ("vllm_version", "version")):
        engine, dialect = "vLLM", "openai-chat"
        engine_homepage = "https://github.com/vllm-project/vllm"
        engine_version = version.get("vllm_version") or version.get("version")
        model = _model_from_models(models)[0]
        notes.append(f"/version: {json.dumps(version)[:160]}")
    elif _ok(ovar) and "version" in ovar:
        engine, dialect = "Ollama", "openai-chat"
        engine_homepage = "https://github.com/ollama/ollama"
        engine_version = ovar.get("version")
        model = _model_from_models(tags)[0] or _model_from_models(models)[0]
        notes.append(f"/api/version: {json.dumps(ovar)[:120]}")
    elif _ok(tgi) and any(k in tgi for k in ("model_id", "model_sha", "model_dtype", "model_device_type")):
        engine, dialect = "Text Generation Inference (TGI)", "openai-chat"
        engine_homepage = "https://github.com/huggingface/text-generation-inference"
        engine_version = tgi.get("version")
        model = tgi.get("model_id")
        notes.append(f"/info: {json.dumps(tgi)[:160]}")
    else:
        model = _model_from_models(models)[0]
        if model is not None:
            engine, dialect = "OpenAI-compatible server (engine not identified)", "openai-chat"
            notes.append("/v1/models responded, no engine-specific metadata endpoint")
        elif _ok(sinfo) and "version" in sinfo:
            engine = f"unidentified server (version {sinfo['version']})"
        elif _ok(version) and "version" in version:
            engine = f"unidentified server (version {version['version']})"

    # Model fallbacks, independent of engine detection.
    if model is None:
        model = _model_from_models(models)[0] or _model_from_models(tags)[0]
    if model is None and _ok(tgi):
        model = tgi.get("model_id")
    if model is None:
        model = model_path and _short_name(model_path)

    probe_summary = {p: ("ok: " + ",".join(sorted(d.keys())[:8]) if _ok(d) else d.get("__error__", "unavailable"))
                     for p, d in ep.items()}
    return {"engine": engine, "engine_version": engine_version, "model": model,
            "model_path": model_path, "engine_homepage": engine_homepage, "dialect": dialect,
            "served_models": served, "engine_settings": engine_settings, "notes": notes,
            "probe": probe_summary}

def gather_identity(args):
    meta = {"url": args.url, "engine": None, "engine_version": None, "model": None,
            "model_path": None, "engine_homepage": None, "dialect": args.dialect,
            "served_models": [], "engine_settings": {}, "notes": [], "probed": False}
    if args.no_probe:
        meta["notes"] = ["--no-probe: metadata endpoints not queried"]
        meta["dialect"] = args.dialect if args.dialect != "auto" else "openai-chat"
        meta["notes"].append(f"--no-probe: assuming dialect {meta['dialect']} "
                            "(use --dialect llama-completion for llama.cpp's /completion)")
    else:
        meta.update(probe_target(args.url.rstrip("/"), args.probe_timeout))
        meta["probed"] = True
        meta["dialect"] = args.dialect if args.dialect != "auto" else (meta["dialect"] or "llama-completion")
    if args.engine:
        meta["engine"] = args.engine
        meta["notes"].append(f"--engine override: {args.engine}")
    if args.model:
        meta["model"] = args.model
        meta["notes"].append(f"--model override: {args.model}")
    meta["engine"] = meta["engine"] or ("unknown (no recognised metadata endpoint)" if meta["probed"]
                                        else "unknown (--no-probe)")
    meta["model"] = meta["model"] or ("unknown (no model metadata endpoint)" if meta["probed"]
                                      else "unknown (--no-probe)")
    return meta

def print_identity(meta, label=""):
    pre = f"[{label}] " if label else ""
    ver = f"  (version {meta.get('engine_version')})" if meta.get("engine_version") else ""
    print(f"{pre}Inference engine : {meta.get('engine')}{ver}")
    print(f"{pre}LLM / model      : {meta.get('model')}")
    print(f"{pre}Request dialect  : {meta.get('dialect')}")
    if meta.get("model_path"):
        print(f"{pre}Model path       : {meta['model_path']}")
    if meta.get("served_models") and len(meta["served_models"]) > 1:
        print(f"{pre}Served models    : {', '.join(meta['served_models'])}")
    if meta.get("url"):
        print(f"{pre}Endpoint         : {meta['url']}")
    if meta.get("engine_homepage"):
        print(f"{pre}Engine project   : {meta['engine_homepage']}")
    st = meta.get("engine_settings") or {}
    if st:
        print(f"{pre}Engine settings  : " + ", ".join(f"{k}={st[k]}" for k in
                ("drafting", "spec", "mtp_max", "lookup", "spec_min_p", "kv", "context",
                 "expert_slots", "pool_workers", "gpu_name") if st.get(k) not in (None, "")))
    for n in meta.get("notes", []):
        print(f"{pre}note: {n}")

def post(url, payload, retries=2):
    """POST with retry on the transient admission/drain answers (429/503 + Retry-After)."""
    for attempt in range(retries + 1):
        req = request.Request(url, data=json.dumps(payload).encode(),
                              headers={"Content-Type": "application/json"}, method="POST")
        try:
            with request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())
        except Exception as e:
            code = getattr(e, "code", None)
            if code in (429, 503) and attempt < retries:
                time.sleep(1.5)
                continue
            raise

def build_payload(meta, p, args):
    """Request body for the benchmark prompt, in the dialect the engine actually speaks."""
    dialect = meta["dialect"]
    if dialect == "llama-completion":
        return f"{meta['url']}/completion", {"prompt": p["prompt"], "n_predict": args.n_predict,
                                             "temperature": args.temperature, "seed": args.seed,
                                             "cache_prompt": False, "stream": False}
    body = {"messages": [{"role": "user", "content": p["prompt"]}], "max_tokens": args.n_predict,
            "temperature": args.temperature, "seed": args.seed, "stream": False}
    if meta.get("model"):
        body["model"] = meta["model"]
    if dialect == "strata-chat":
        if args.no_thinking:
            body["enable_thinking"] = False
        if args.effort:
            body["reasoning_effort"] = args.effort
    return f"{meta['url']}/v1/chat/completions", body

def metrics_totals(meta, args):
    """Strata's cumulative counters, for a before/after delta over the run's window."""
    if meta["engine"] != "Strata":
        return None
    d = _get(meta["url"].rstrip("/") + "/metrics", max(args.probe_timeout, 10))
    return d.get("totals") if _ok(d) and _ok(d.get("totals")) else None

def totals_delta(before, after):
    if not before or not after:
        return None
    keys = ("requests", "output_tokens", "drafts_offered", "drafts_accepted", "reused", "prompt_ms", "decode_ms")
    d = {k: round(after.get(k, 0) - before.get(k, 0), 1) for k in keys}
    d["server_accept_rate"] = round(d["drafts_accepted"] / d["drafts_offered"], 4) if d["drafts_offered"] else None
    d["server_decode_tok_s"] = round(d["output_tokens"] / (d["decode_ms"] / 1000), 1) if d["decode_ms"] else None
    return d

def run(args):
    args.url = args.url.rstrip("/")
    # Identify what we are about to benchmark, and announce it *before* benchmarking.
    meta = gather_identity(args)
    print("=" * 78)
    print("MTP benchmark — subject under test")
    print_identity(meta)
    print("=" * 78)
    print(f"Running {len(PROMPTS)} prompts (max_tokens={args.n_predict}, temperature={args.temperature}, "
          f"seed={args.seed}) via {meta['dialect']}\n")
    totals_before = metrics_totals(meta, args)

    out = {"identity": meta, "results": []}
    for p in PROMPTS:
        url, payload = build_payload(meta, p, args)
        t0 = time.time()
        r = post(url, payload)
        wall = time.time() - t0
        t = r.get("timings", {}) or {}
        rec = {"name": p["name"], "wall_s": round(wall, 3),
               "predicted_n": t.get("predicted_n"), "predicted_per_second": t.get("predicted_per_second"),
               "draft_n": t.get("draft_n", 0), "draft_n_accepted": t.get("draft_n_accepted", 0)}
        rec["accept_rate"] = round(rec["draft_n_accepted"] / rec["draft_n"], 4) if rec["draft_n"] else None
        if meta["dialect"] in ("strata-chat", "openai-chat"):
            ch = (r.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            rec["finish_reason"] = ch.get("finish_reason")
            # Strata's conversation cache: prompt tokens it already held (llama.cpp cache_n).
            rec["cache_n"] = t.get("cache_n", (r.get("usage") or {}).get("prompt_tokens_details", {})
                                   .get("cached_tokens") if isinstance((r.get("usage") or {}).get("prompt_tokens_details"), dict) else None)
            rec["prompt_n"] = t.get("prompt_n")
            rec["prompt_per_second"] = t.get("prompt_per_second")
            # A reasoning model spends its token budget on reasoning_content, not on content.
            rec["reasoning_only"] = (msg.get("content") in (None, "")) and bool(msg.get("reasoning_content"))
        out["results"].append(rec)
        ar = f"{rec['accept_rate']:.3f}" if rec["accept_rate"] is not None else "n/a"
        pps = rec["predicted_per_second"] if rec.get("predicted_per_second") else 0.0
        extra = f" cache={rec.get('cache_n')}" if meta["dialect"] != "llama-completion" else ""
        print(f"  {rec['name']:<18} pred={rec['predicted_n']:>4} draft={rec['draft_n']:>4} "
              f"acc={rec['draft_n_accepted']:>4} rate={ar} tok/s={pps:.1f}{extra}")
    td  = sum(x["draft_n"] or 0 for x in out["results"])
    ta  = sum(x["draft_n_accepted"] or 0 for x in out["results"])
    tp  = sum(x["predicted_n"] or 0 for x in out["results"])
    tw  = sum(x["wall_s"] for x in out["results"])
    out["aggregate"] = {"n_requests": len(out["results"]), "total_predicted": tp, "total_draft": td,
                        "total_draft_accepted": ta, "aggregate_accept_rate": round(ta/td, 4) if td else None,
                        "wall_s_total": round(tw, 2)}
    if meta["dialect"] != "llama-completion":
        out["aggregate"]["total_cache_n"] = sum(x.get("cache_n") or 0 for x in out["results"])
    delta = totals_delta(totals_before, metrics_totals(meta, args))
    if delta:
        out["aggregate"]["server_delta"] = delta
        out["aggregate"]["per_request_accept_rate"] = out["aggregate"]["aggregate_accept_rate"]
        out["aggregate"]["server_accept_rate"] = delta["server_accept_rate"]
        agree = "matches server counters" if delta["drafts_offered"] == td and delta["drafts_accepted"] == ta \
                else "MISMATCH with server counters"
        out["notes"] = out.get("notes", []) + [f"per-request draft counters {agree} "
                                              f"(server: {ta}->{delta['drafts_accepted']} accepted of "
                                              f"{td}->{delta['drafts_offered']} offered)"]
    print(f"\nBenchmark results — engine: {meta['engine']}"
          f"{'' if not meta.get('engine_version') else ' ' + str(meta['engine_version'])} | "
          f"LLM: {meta['model']} | dialect: {meta['dialect']}")
    print("Aggregate:", json.dumps(out["aggregate"], indent=2))
    for n in out.get("notes", []):
        print("note:", n)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2); print("Wrote", args.out)

def identity_from_file(path):
    try:
        d = json.load(open(path))
    except Exception as e:
        return {"engine": f"unreadable ({type(e).__name__})", "model": "unknown",
                "url": path, "served_models": [], "notes": []}
    m = d.get("identity")
    if m:
        m.setdefault("url", d.get("url"))
        m.setdefault("served_models", [])
        m.setdefault("notes", [])
        return m
    # Files written by the original mtp-bench.py have no identity block.
    return {"engine": "unknown (legacy output)", "model": "unknown (legacy output)",
            "url": d.get("url") or path, "served_models": [], "notes": [f"no identity block in {path}"]}

def diff(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    print("=" * 78)
    print("Comparing two benchmark runs — subject identity reported before results")
    print_identity(identity_from_file(a), label="A")
    print("-" * 78)
    print_identity(identity_from_file(b), label="B")
    print("=" * 78)
    keys = ["aggregate_accept_rate", "total_predicted", "total_draft", "total_draft_accepted", "wall_s_total"]
    for k in ("total_cache_n", "per_request_accept_rate", "server_accept_rate"):
        if k in A["aggregate"] or k in B["aggregate"]:
            keys.append(k)
    print(f"{'metric':<24} {'A':>14} {'B':>14} {'delta':>10}")
    for k in keys:
        va, vb = A["aggregate"].get(k), B["aggregate"].get(k)
        if va is None or vb is None: print(f"{k:<24} {str(va):>14} {str(vb):>14}"); continue
        d = vb - va
        s = f"{d:>+10.4f}" if isinstance(d, float) else f"{d:>+10}"
        print(f"{k:<24} {va:>14} {vb:>14} {s}")
    by_a = {x["name"]: x for x in A["results"]}
    print("\n{:<20} {:>8} {:>8} {:>8}".format("prompt", "A", "B", "delta"))
    for rb in B["results"]:
        ra = by_a.get(rb["name"]) or {}
        ar = ra.get("accept_rate") or 0; br = rb.get("accept_rate") or 0
        print(f"{rb['name']:<20} {ar:>8.3f} {br:>8.3f} {br-ar:>+8.3f}")

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8090")
ap.add_argument("--out")
ap.add_argument("--diff", nargs=2)
ap.add_argument("--engine", help="override the detected inference engine name")
ap.add_argument("--model", help="override the detected LLM / model name")
ap.add_argument("--dialect", choices=["auto", "strata-chat", "openai-chat", "llama-completion"],
                default="auto", help="request shape; auto = from the detected engine")
ap.add_argument("--n-predict", type=int, default=192, help="tokens to generate per prompt")
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--no-thinking", action="store_true", help="Strata: send enable_thinking=false")
ap.add_argument("--effort", help="Strata: reasoning_effort (low, medium, high, xhigh)")
ap.add_argument("--no-probe", action="store_true", help="skip the engine/model metadata probes")
ap.add_argument("--probe-timeout", type=float, default=10.0, help="timeout (s) for metadata endpoint probes")
a = ap.parse_args()
if a.diff: diff(*a.diff)
else: run(a)
