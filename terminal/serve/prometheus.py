"""GET /metrics in the Prometheus text format, with vLLM's metric names, so the dashboards and alerts written for a
vLLM server read a Strata server unchanged.  Everything comes from the server's own record (Service.metrics(), the
same dict the JSON /metrics returns) plus the per-request histograms observed when a request finishes: the latencies
(TTFT, ITL, e2e), the token counts (prompt, generated, computed KV), the stage timings (prefill, decode, time per
output token) and the finish reason on `vllm:request_success_total`.

What vLLM has no name for is the JSON's own facts under `strata:`, named after the JSON's keys (live.tok_s ->
strata:live_tok_s, totals.decode_ms -> strata:totals_decode_seconds_total, hardware.gpu_util -> strata:gpu_util,
conversation_cache.bytes -> strata:conversation_cache_used_bytes), so the two formats say the same thing and a panel
can be moved from the Monitor tab to Grafana by its key."""
import threading

# vLLM's buckets for the request latencies (seconds)
BUCKETS = (0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0,
           80.0, 160.0, 640.0, 2560.0)
CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def _buckets_125(top: int) -> tuple:
    """vLLM's 1-2-5 per decade up to `top` (the engine's context, 262144 here), with `top` as the last boundary."""
    out, decade = [], 1
    while True:
        for m in (1, 2, 5):
            v = m * decade
            if v >= top:
                return tuple(out + [top])
            out.append(v)
        decade *= 10


# vLLM's request-token buckets, reaching the engine's context (issue #6062)
TOKEN_BUCKETS = _buckets_125(262144)

# Strata's request finish values, under vLLM's finished_reason names (cancel/disconnect are client aborts; a reply
# that ended in tool calls is a normal stop). The rest (stop, length, error, timeout) keep their own name.
FINISH_REASON = {"cancel": "abort", "disconnect": "abort", "tool_calls": "stop"}


def _finish_reason(finish):
    return FINISH_REASON.get(finish, finish) if finish else None


def wants_prometheus(accept: str, query: str) -> bool:
    """Prometheus asks with Accept: text/plain or application/openmetrics-text; a person can add ?format=prometheus.
    Anything else (the Monitor tab, curl without a header) keeps the JSON."""
    return "format=prometheus" in (query or "") or "openmetrics" in (accept or "") or "text/plain" in (accept or "")


class Histogram:
    def __init__(self, buckets=BUCKETS):
        self.buckets = buckets
        self.counts = [0] * len(buckets)
        self.sum = 0.0
        self.n = 0

    def observe(self, v: float, times: int = 1):
        if v is None or times <= 0:
            return
        for i, b in enumerate(self.buckets):
            if v <= b:
                self.counts[i] += times
        self.sum += v * times
        self.n += times


class Latencies:
    """Per-finished-request histograms: the latencies (TTFT, ITL, e2e), the token counts (prompt, generated, computed
    KV) and the stage timings (prefill, decode, time per output token), plus a finish-reason counter."""

    def __init__(self):
        self.lock = threading.Lock()
        self.ttft, self.itl, self.e2e = Histogram(), Histogram(), Histogram()
        self.prompt_tokens = Histogram(TOKEN_BUCKETS)
        self.generation_tokens = Histogram(TOKEN_BUCKETS)
        self.prefill_kv_computed = Histogram(TOKEN_BUCKETS)
        self.prefill, self.decode, self.tpot = Histogram(), Histogram(), Histogram()
        self.finish = {}

    def observe(self, ttft_s, output_tokens, decode_s, total_s, *, prompt_tokens=None, computed_tokens=None,
                prefill_s=None, finish=None):
        with self.lock:
            self.ttft.observe(ttft_s)
            if output_tokens and output_tokens > 1 and decode_s:
                # the mean gap, once per gap: the engine reports the decode time, not each token's
                self.itl.observe(decode_s / (output_tokens - 1), output_tokens - 1)
            self.e2e.observe(total_s)
            self.prompt_tokens.observe(prompt_tokens)
            self.generation_tokens.observe(output_tokens)
            self.prefill_kv_computed.observe(computed_tokens)
            self.prefill.observe(prefill_s)
            self.decode.observe(decode_s)
            if output_tokens:
                # vLLM's time per output token: (e2e - ttft) / (generated - 1); a one-token reply has no gap
                gap = (total_s - ttft_s) / (output_tokens - 1) if output_tokens > 1 and ttft_s is not None else 0.0
                self.tpot.observe(gap)
            if finish:
                self.finish[finish] = self.finish.get(finish, 0) + 1

    def snapshot(self):
        with self.lock:
            snap = {k: (h.buckets, list(h.counts), h.sum, h.n) for k, h in
                    (("ttft", self.ttft), ("itl", self.itl), ("e2e", self.e2e),
                     ("prompt_tokens", self.prompt_tokens), ("generation_tokens", self.generation_tokens),
                     ("prefill_kv_computed", self.prefill_kv_computed),
                     ("prefill", self.prefill), ("decode", self.decode), ("tpot", self.tpot))}
            snap["finish"] = dict(self.finish)
            return snap


def _num(v, default=0):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else default


def render(m: dict, lat: dict) -> str:
    """The text for one scrape.  `m` is Service.metrics(), `lat` Latencies.snapshot()."""
    model = str((m.get("engine") or {}).get("model") or "strata").replace("\\", "\\\\").replace('"', '\\"')
    lab = f'model_name="{model}"'
    live, totals = m.get("live") or {}, m.get("totals") or {}
    out = []

    def metric(name, kind, help_, value):
        out.append(f"# HELP {name} {help_}\n# TYPE {name} {kind}\n{name}{{{lab}}} {value}")

    busy = live.get("state") in ("reading", "generating")
    ctx = _num((m.get("engine") or {}).get("max_context"), 0)
    used = _num(live.get("prompt_tokens")) + _num(live.get("generated")) if busy else 0
    # "parallel": live.running / live.waiting count the requests together (waiting: for a slot or the control lines)
    metric("vllm:num_requests_running", "gauge", "Requests reading their prompt or generating.",
           _num(live.get("running"), int(busy)))
    metric("vllm:num_requests_waiting", "gauge", "Requests waiting for their turn.",
           _num(live.get("queued")) + _num(live.get("waiting")))
    metric("vllm:kv_cache_usage_perc", "gauge", "The running request's share of the context (1 = full).",
           round(min(1.0, used / ctx), 4) if ctx else 0)
    metric("vllm:prompt_tokens_total", "counter", "Prompt tokens of the finished requests.",
           _num(totals.get("prompt_tokens")))
    metric("vllm:generation_tokens_total", "counter", "Generated tokens.", _num(totals.get("output_tokens")))
    out.append("# HELP vllm:request_success_total Finished requests.\n# TYPE vllm:request_success_total counter")
    finish = {}
    for reason, n in (lat.get("finish") or {}).items():  # cancel + disconnect both become abort: count them together
        reason = _finish_reason(reason)
        if reason:
            finish[reason] = finish.get(reason, 0) + n
    if finish:
        for reason, n in sorted(finish.items()):
            out.append(f'vllm:request_success_total{{finished_reason="{reason}",{lab}}} {n}')
    else:                                               # nothing observed yet: still expose the family
        out.append(f"vllm:request_success_total{{{lab}}} {_num(totals.get('requests'))}")
    metric("vllm:prefix_cache_queries_total", "counter", "Prompt tokens looked up in the prompt cache.",
           _num(totals.get("prompt_tokens")))
    metric("vllm:prefix_cache_hits_total", "counter", "Prompt tokens the prompt cache already held.",
           _num(totals.get("reused")))
    metric("vllm:spec_decode_num_draft_tokens_total", "counter", "MTP draft tokens offered.",
           _num(totals.get("drafts_offered")))
    metric("vllm:spec_decode_num_accepted_tokens_total", "counter", "MTP draft tokens accepted.",
           _num(totals.get("drafts_accepted")))
    metric("vllm:num_preemptions_total", "counter", "Preempted requests (Strata does not preempt).", 0)
    for name, key, help_ in (("vllm:time_to_first_token_seconds", "ttft", "Time to the first token."),
                             ("vllm:inter_token_latency_seconds", "itl", "Time between two tokens."),
                             ("vllm:e2e_request_latency_seconds", "e2e", "A request from start to end."),
                             ("vllm:request_prompt_tokens", "prompt_tokens", "Prompt tokens per finished request."),
                             ("vllm:request_generation_tokens", "generation_tokens",
                              "Generated tokens per finished request."),
                             ("vllm:request_prefill_kv_computed_tokens", "prefill_kv_computed",
                              "Prompt tokens actually computed, not reused, per finished request."),
                             ("vllm:request_prefill_time_seconds", "prefill",
                              "Prompt read time per finished request."),
                             ("vllm:request_decode_time_seconds", "decode",
                              "Generation time per finished request."),
                             ("vllm:request_time_per_output_token_seconds", "tpot",
                              "Time per output token ((e2e - ttft) / (generated - 1)), per finished request.")):
        buckets, counts, total, n = lat[key]
        out.append(f"# HELP {name} {help_}\n# TYPE {name} histogram")
        for b, c in zip(buckets, counts):
            out.append(f'{name}_bucket{{{lab},le="{b}"}} {c}')
        out.append(f'{name}_bucket{{{lab},le="+Inf"}} {n}')
        out.append(f"{name}_sum{{{lab}}} {round(total, 6)}\n{name}_count{{{lab}}} {n}")
    strata(m, out, lab)
    return "\n".join(out) + "\n"


def strata(m: dict, out: list, lab: str):
    """The JSON's facts vLLM has no name for, under `strata:` and the JSON's key."""
    live, totals = m.get("live") or {}, m.get("totals") or {}
    hw, eng = m.get("hardware") or {}, m.get("engine") or {}
    reqs = m.get("requests") or []

    typed = set()

    def metric(name, kind, help_, value, labels=""):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return                                       # unknown here (no GPU telemetry, an older engine): no sample
        if name not in typed:                            # one HELP / TYPE per family, before its first sample
            typed.add(name)
            out.append(f"# HELP strata:{name} {help_}\n# TYPE strata:{name} {kind}")
        out.append(f"strata:{name}{{{lab}{labels}}} {value}")

    for state in ("unloaded", "idle", "reading", "generating"):
        metric("live_state", "gauge", "1 for what the engine is doing (live.state).", int(live.get("state") == state),
               f',state="{state}"')
    metric("live_tok_s", "gauge", "Decode rate over the last seconds (live.tok_s).", live.get("tok_s") or 0)
    metric("live_prefill_tok_s_mean", "gauge", "Prompt reading rate of the running request (live.prefill_tok_s_mean).",
           live.get("prefill_tok_s_mean") or 0)
    metric("live_prompt_read", "gauge", "Prompt tokens read so far by the running request (live.prompt_read).",
           live.get("prompt_read") or 0)
    for state in ("idle", "reading", "generating"):
        slots = live.get("slots")
        if isinstance(slots, list) and slots:
            metric("live_slots", "gauge", "The batch slots in each state (live.slots).",
                   sum(1 for x in slots if isinstance(x, dict) and x.get("state") == state), f',state="{state}"')
    metric("engine_max_context", "gauge", "The engine's context (engine.max_context).", eng.get("max_context"))
    metric("totals_prompt_seconds_total", "counter", "Time spent reading prompts (totals.prompt_ms).",
           round((totals.get("prompt_ms") or 0) / 1000, 3))
    metric("totals_decode_seconds_total", "counter", "Time spent generating (totals.decode_ms).",
           round((totals.get("decode_ms") or 0) / 1000, 3))
    last = reqs[0] if reqs else {}
    metric("last_hit_rate", "gauge", "The last request's expert cache hit rate (requests[0].hit_rate).",
           last.get("hit_rate"))
    metric("last_decode_tok_s", "gauge", "The last request's decode rate (requests[0].decode_tok_s).",
           last.get("decode_tok_s"))
    gpus = hw.get("gpus") or ([{"index": 0, "util": hw.get("gpu_util"), "mem_used": hw.get("gpu_mem_used"),
                                "mem_total": hw.get("gpu_mem_total"), "temp": hw.get("gpu_temp"),
                                "power": hw.get("gpu_power")}] if hw.get("gpu_util") is not None else [])
    for name, key, help_ in (("gpu_util", "util", "GPU busy, percent (hardware.gpu_util)."),
                             ("gpu_mem_used_bytes", "mem_used", "GPU memory in use (hardware.gpu_mem_used)."),
                             ("gpu_mem_total_bytes", "mem_total", "GPU memory (hardware.gpu_mem_total)."),
                             ("gpu_temp_celsius", "temp", "GPU temperature (hardware.gpu_temp)."),
                             ("gpu_power_watts", "power", "GPU power draw (hardware.gpu_power).")):
        for g in gpus:                                   # a family's samples together, one per card
            metric(name, "gauge", help_, g.get(key), f',gpu="{g.get("index")}"')
    metric("cpu", "gauge", "CPU busy, percent (hardware.cpu).", hw.get("cpu"))
    metric("ram_used_bytes", "gauge", "RAM in use (hardware.ram_used).", hw.get("ram_used"))
    metric("ram_total_bytes", "gauge", "RAM (hardware.ram_total).", hw.get("ram_total"))
    cc = m.get("conversation_cache") or {}             # #6062: the conversation cache (the prefix cache for hybrids)
    if cc.get("enabled"):                              # only when --conversation-cache-mib is on, like the Monitor card
        budget = cc.get("budget_mib")
        metric("conversation_cache_budget_bytes", "gauge",
               "Conversation-cache budget (conversation_cache.budget_mib).",
               budget * (1 << 20) if isinstance(budget, int) else None)
        metric("conversation_cache_used_bytes", "gauge",
               "Conversation-cache bytes parked now (conversation_cache.bytes).", cc.get("bytes"))
        metric("conversation_cache_slots", "gauge",
               "Conversation-cache slots (conversation_cache.slots).", cc.get("slots"))
        metric("conversation_cache_parked", "gauge",
               "Conversations parked now (conversation_cache.parked).", cc.get("parked"))
        metric("conversation_cache_evictions_total", "counter",
               "Parked conversations evicted (conversation_cache.evictions).", cc.get("evictions"))
        metric("conversation_cache_parks_total", "counter",
               "Conversations parked (conversation_cache.parks).", cc.get("parks"))
        metric("conversation_cache_restores_total", "counter",
               "Parked conversations restored (conversation_cache.restores).", cc.get("restores"))
        metric("conversation_cache_reused_tokens_total", "counter",
               "Prompt tokens the conversation cache gave back (conversation_cache.reused_tokens).",
               cc.get("reused_tokens"))
