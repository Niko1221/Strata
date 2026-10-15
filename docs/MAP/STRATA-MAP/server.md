# Strata map: server (`serve/`, `chat.py`)

Source: Strata commit `296b59d2`. Paths are relative to the Strata repo root.

## 1. Overview

- The server is a pure-Python HTTP layer (stdlib `ThreadingHTTPServer`, no FastAPI) plus a tiny terminal client (chat.py).
- It speaks OpenAI Chat Completions, Anthropic Messages, OpenAI Responses (stateless) and image generation.
- It renders prompts with a Jinja chat template (serve/chat_template.jinja), tokenizes them, and drives the native engine (`strata --serve`, in src/program/generate.cpp) as a **child process over a line-based stdin/stdout protocol** (`GEN`/`GENI`/`BGEN`/`STOP`/`SAVE`/`RESTORE`/`VRAM` in; `T`/`PP`/`RESUME`/`DONE`/`ERR`/`BDONE` out).
- It parses the streamed text back into reasoning, content and Qwen XML-style tool calls.
- Prefix/KV reuse is mostly done **inside the engine** (live session + conversation checkpoints + parked conversations).
- The Python side picks batch slots by held prefix, routes replicas by prefix hash, and offers the explicit `strata_prefix` pin.

## 2. File table (non-test files; serve/ also has 43 test_*.py files)

| path | lines | purpose |
|---|---|---|
| serve/server.py | 6976 | Everything: engine subprocess wrapper (StrataEngine), Service (queue, prompt build, run loop), HTTP handler, main()/CLI |
| serve/frontend.py | 1433 | Text-only layer: ChatTemplate, OpenAI/Anthropic → template messages, tool list, streaming OutputParser (think + tool calls) |
| serve/responses.py | 690 | OpenAI Responses API (POST /v1/responses), stateless, converted to the chat path; Assembler for typed SSE events |
| serve/chat_template.jinja | 184 | Qwen-style ChatML template (`<\|im_start\|>`, `<think>`, `<tool_call><function=..><parameter=..>`) |
| serve/chat_golden.json | 197 | golden renders for the template (test data) |
| serve/structured.py | 218 | response_format JSON/json_schema: prompt once, validate after; no grammar decoding |
| serve/mcp.py | 676 | MCP client (stdio + streamable HTTP) so the web app's chats can call MCP tools server-side |
| serve/mcp_fake_server.py | 172 | fake MCP server for tests |
| serve/replicas.py | 421 | data-parallel replicas: N engines, Router by 64-token-block prefix hash + load |
| serve/images.py | 471 | /v1/images/generations via stable-diffusion.cpp `sd-cli` |
| serve/telemetry.py | 505 | GPU/CPU sampler for the web Monitor tab |
| serve/prometheus.py | 160 | /metrics in Prometheus format with vLLM metric names |
| serve/runconfig.py | 179 | GET/POST /config: whitelisted keys of strata-<model>.json |
| serve/logit_bias.py | 52 | request-scoped logit_bias normalization |
| serve/session_save_retry.py | 84 | bounded SAVE retries after engine cache reclaim |
| serve/cache_drop.py | 70 | opt-in page-cache drop on exit (STRATA_DROP_CACHE_ON_EXIT=1) |
| serve/cpu_affinity.py | 68 | split cores between Linux instances |
| serve/keepawake.py | 94 | Windows prevent_sleep while busy |
| serve/winjob.py | 97 | Windows job object so children die with the server |
| serve/__init__.py | 1 | docstring only |
| serve/web/; serve/fixtures/ | dirs | web app static files; test fixtures (not mapped) |
| chat.py | 116 | stdlib terminal chat client against a running server (`--port`, `--think`, /image, /reset) |
| src/program/generate.cpp | 14182 | native engine; its `--serve` block (stdin command loop) is the server-side endpoint of the pipe |

## 3. Where it does X

**HTTP routing:**

- serve/server.py:4993 `make_handler(svc)`.
- `do_GET` at serve/server.py:5249 (/fonts, /web, /metrics, /metrics/prometheus, /v1/models & /models serve/server.py:5369, /v1/status serve/server.py:5409, /v1/images/files/).
- `do_POST` serve/server.py:5415 → `_post` serve/server.py:5421: /settings, /config, /load, /unload, /v1/load, /v1/unload, /v1/vram serve/server.py:5494, /slots/<n>?action=, then /v1/responses serve/server.py:5507, /v1/chat/completions serve/server.py:5518 → `_openai`, /v1/messages serve/server.py:5520 → `_anthropic`, /v1/messages/count_tokens, /v1/images/generations.
- The query string is stripped (Claude Code posts `/v1/messages?beta=true`).
- `/v1/responses/<id>` always returns 404 (nothing is stored).
- Server class: `Server(ThreadingHTTPServer)` serve/server.py:6096; `serve()` serve/server.py:6360.

**Auth / host checks** — `_authorized` (in handler), `key_matches` serve/server.py:6324, `api_key_of` serve/server.py:6307, `host_allowed` serve/server.py:6283, `origin_allowed` serve/server.py:6340. Default bind is 127.0.0.1. /load, /unload and /slots accept requests only from Strata's own page (`_own_page`).

**Request → messages:**

- serve/frontend.py:521 `openai_to_messages`, serve/frontend.py:590 `anthropic_to_messages`, serve/responses.py:128 `input_messages`.
- These normalize content parts, images and tool results into template messages plus kwargs.
- Reasoning effort becomes template kwargs (`effort_kwargs` serve/frontend.py:138, `EFFORT` serve/frontend.py:134).
- Server-side effort sentences are in serve/server.py:136-144.
- `_stable_tool_order` serve/frontend.py:449 keeps the tool order fixed across requests (LRU of 64, `_TOOL_ORDERS_MAX` serve/frontend.py:446) so the rendered prefix stays the same and the cache can hit.

**Prompt build** — `Service.render_prompt` serve/server.py:3772, `encode_prompt` serve/server.py:3790, `prepare` serve/server.py:3848. Template: `ChatTemplate` serve/frontend.py:40, `render` serve/frontend.py:57. Literal `<think>`, `<tool_call>` and vision tags inside user content are swapped for private-use codepoints (`LITERAL_TAGS` serve/frontend.py:253, `mark_think_literals` serve/frontend.py:332, `unmark_think_literals` serve/frontend.py:365) so user text cannot inject control tokens.

**Request lifecycle / scheduling:**

- `Service.begin_request` serve/server.py:3615 records the request.
- `Service.fifo = threading.Lock()` serve/server.py:3042 is the engine's single turn: one request at a time unless `--batch`/"parallel" is on.
- Queued requests send SSE keep-alives every `QUEUE_BEAT_S`=10 s (serve/server.py:214) and leave the queue on client disconnect.
- `Service.run` serve/server.py:4065 → `_run` serve/server.py:4076 merges sampling defaults, caps the thinking budget (`ANSWER_RESERVE_MIN` serve/server.py:150), and builds the `OutputParser`, Detokenizer and StopMatcher.
- It then iterates `engine.generate` and yields ("event", Event) / ("done", {...}).
- It also does reasoning-loop detection (`reasoning_repeat_coverage` serve/server.py:168, LOOP_* constants serve/server.py:161-165) and recovery (`focused_recovery_prompt` serve/server.py:195).
- Output formatting: `openai_chunks` serve/server.py:4736 / `openai_collect` serve/server.py:4806, `anthropic_events` serve/server.py:4868 / `anthropic_collect` serve/server.py:4938, `responses.Assembler` serve/responses.py:468.
- MCP tool loop: `run_with_mcp` serve/server.py:4617.

**Engine hand-off (→ engine-core.md, model-spec-ngram-platform.md via src/program/generate.cpp):**

- `StrataEngine` serve/server.py:738 spawns the `strata` binary (`engine_binary` serve/server.py:2484, args from `engine_args` serve/server.py:2573, `parallel_args` serve/server.py:2609, env `child_env` serve/server.py:2788 / `hip_env_guard` serve/server.py:2816).
- `generate` serve/server.py:1854 writes `GEN <max_new><sampling keys> <id,id,...>` (or `GENI.. <embeddings file>` for images) to stdin.
- It then reads `T <id>`, `PP pos total ms tok/s`, `RESUME n` and `DONE..`/`ERR`.
- On early exit it sends `STOP` and drains to DONE with a timeout.
- A `None` is yielded every 10 s as a heartbeat.
- Batched path: `generate_batched` serve/server.py:1477 (BGEN/BT/BDONE/BADM/BSTOP lines).
- Native side: protocol comment at src/program/generate.cpp:6511-6530; command loop at src/program/generate.cpp:10322-10636 (BSTOP src/program/generate.cpp:10328, VRAM src/program/generate.cpp:10334, BGEN src/program/generate.cpp:10349 rewritten to `GEN 1`, SAVE/RESTORE src/program/generate.cpp:10412, GEN/GENI src/program/generate.cpp:10635); STOP is read by a stdin thread at src/program/generate.cpp:9483; `RESUME` printed src/program/generate.cpp:11214; `T` printed src/program/generate.cpp:12420, src/program/generate.cpp:12713; `PP` src/program/generate.cpp:8540, src/program/generate.cpp:11342, src/program/generate.cpp:11403.
- Note: the protocol comment says "Decoding is greedy" (src/program/generate.cpp:6530), but sampling keys are sent (`sampling_keys` serve/server.py:1132).

**Prefix caching / KV reuse:**

- (a) Engine (engine-core.md): a request continues from the live session or the longest **conversation checkpoint** its prompt starts with (src/program/generate.cpp:10919-10925, `ConvCheckpoint`).
- Parked conversations can be restored (src/program/generate.cpp:11107-11115), and a `req_pin` path exists (src/program/generate.cpp:11019).
- Flags: `--prompt-cache N` (default 6, ~118 MB), `--prompt-cache-every N` (default 16384), `--conversation-cache-mib/-slots/-min-tokens/-min-free-mib`, `--memory-limit-mib` (help text src/program/generate.cpp:783-812).
- Code in src/core/conversation_state.cpp, src/core/conversation_snapshot.cpp, src/core/conversation_memory.cpp, src/core/conversation_file.cpp.
- (b) `strata_prefix` request field → `Service.resolve_prefix` serve/server.py:3946 turns {messages:K}|{message:I,chars:N}|{tokens:N} into `pin=N`.
- The engine keeps that checkpoint pinned.
- It is ignored when the request has images.
- (c) Batch slots: `pick_slot` serve/server.py:1807 picks the free slot whose held tokens are the longest prefix of the prompt.
- With batch groups the minimum is `SLOT_PREFIX_MIN`=512 (serve/server.py:1805).
- (d) Replicas: serve/replicas.py:122 `Router` hashes 64-token blocks (`BLOCK` serve/replicas.py:32) and routes to the matching replica, weighed against `LOAD_TOKENS`=3000.
- (e) Prefix stability helpers: `_env_to_user` serve/frontend.py:392 (config key `env_to_user`) and `pin_billing_stamp` serve/frontend.py:564 (Anthropic billing header pinned).

**Tool-call parsing:**

- serve/frontend.py:873 `OutputParser` (`feed` serve/frontend.py:1161, `finish` serve/frontend.py:1383, `_finish_call` serve/frontend.py:1416).
- It streams over the text deltas and splits `<think>…</think>` reasoning, content and `<tool_call>` blocks (markers serve/frontend.py:672-679).
- `parse_tool_call` serve/frontend.py:837 parses Qwen-XML `<function=NAME><parameter=P>VALUE</parameter></function>` and JSON-decodes values per the tool schema.
- `json_tool_call` serve/frontend.py:812 is a fallback for JSON-in-fence calls.
- Tool calls inside the reasoning are accepted only if declared, at line start, outside code fences, and closed.
- Up to `RCALL_MAX`=32768 chars are held while waiting for the close.
- `tool_choice_of` serve/frontend.py:733 and `forced_call` serve/frontend.py:756 implement tool_choice; a forced call is prefilled into the prompt (`force`).

**Structured output** — serve/structured.py:83 `prepare_format` (adds an instruction to the prompt), serve/structured.py:192 `validated_json`; serve/server.py:4842 `structured_chunks`. A failed validation is an error, never a retry.

**Streaming detok / stops** — `Detokenizer` serve/server.py:2897 (incremental, fixed cost per token), `stop_strings` serve/server.py:2928, `StopMatcher` serve/server.py:2948.

**Sessions (save/restore)** — HTTP `/slots/<n>?action=save|restore` → `Service.slot_action` serve/server.py:3111 → `StrataEngine.session_file` serve/server.py:2029 → engine `SAVE`/`RESTORE` (src/program/generate.cpp:10412). `--slot-save-path` sets the directory (`slot_save_dir` serve/server.py:3013, `slot_filename_problem` serve/server.py:2988). On-disk format is in src/core/conversation_file.cpp:38-46 (format v1, `kMagic` STRSESS\x01; engine-core.md).

**Load/unload, VRAM, vision** — `ensure_loaded` serve/server.py:3316, `load` serve/server.py:3441, `unload` serve/server.py:3486, idle unload serve/server.py:3511, `vram` serve/server.py:3403 (→ engine `VRAM`, NVIDIA `--vram-elastic` only per src/program/generate.cpp:9721). `Vision` serve/server.py:2204 is a separate strata-vision process that writes embeddings files for GENI.

**See also:** engine-core.md (src/program/generate.cpp `--serve` loop, src/core/session.cpp, src/core/conversation_state.cpp, src/core/conversation_snapshot.cpp, src/core/conversation_memory.cpp, src/core/conversation_file.cpp checkpoints); prefill.md (prompt chunks/PP lines, src/prefill); speculative-decoding.md (MTP draft accepted/offered counts reported in DONE); moe-experts.md (expert cache, VRAM command, PCIe-offload stats in DONE serve/server.py:1097); intel-sycl-backend.md (`STRATA_SYCL_BIN` serve/server.py:2490 picks the SYCL binary); apple-metal-backend.md; model-spec-ngram-platform.md (qwen35moe `strata-q36` engine family; family name in `default_model_name` serve/server.py:6618-6620); build-system.md (engine binary path); tests-bench-docs.md (serve/test_*.py use MockEngine serve/server.py:277 or fake engine scripts, no GPU).

## 4. Key types, env vars, CLI flags

**Types:**

| Name | Where / notes |
|---|---|
| `Engine` (Protocol) | serve/server.py:272 |
| `MockEngine` | serve/server.py:277 |
| `StrataEngine` | serve/server.py:738 |
| `Service` | serve/server.py:3026 |
| `Vision` | serve/server.py:2204 |
| `ReplicaEngine` | serve/replicas.py:173 |
| `Router` | serve/replicas.py:122 |
| `ChatTemplate` | serve/frontend.py:40 |
| `OutputParser` | serve/frontend.py:873 |
| `ToolCall` | serve/frontend.py:659 |
| `Event` | serve/frontend.py:666 |
| `McpHub` | serve/mcp.py:494 |
| `Assembler` | serve/responses.py:468 |

**Errors:**

| Name | Where / notes |
|---|---|
| `EngineDied` | serve/server.py:303 |
| `EngineStarting` | serve/server.py:307 |
| `ClientGone` | serve/server.py:312 |
| `ModelBusy` | serve/server.py:317 |
| `EngineSilent` | serve/server.py:321 |
| `EngineNonFinite` | serve/server.py:327 |
| `SessionRefused` | serve/server.py:334 |
| `EngineStuck` | serve/server.py:351 |
| `GpuBusy` | serve/server.py:356 |

**CLI (serve/server.py main serve/server.py:6623):**

| Name | Where / notes |
|---|---|
| `--engine` | serve/server.py:6625 (`mock` or `strata`) |
| `--config` | serve/server.py:6626 |
| `--host` | serve/server.py:6628 |
| `--script` | serve/server.py:6631 |
| `--port` | serve/server.py:6634 |
| `--gpu` | serve/server.py:6635 |
| `--tokenizer` | serve/server.py:6637 |
| `--open` | serve/server.py:6639 |
| `--fit-max-tokens` | serve/server.py:6640 |
| `--api-key` | serve/server.py:6643 |
| `--mcp-config` | serve/server.py:6646 |
| `--lazy` | serve/server.py:6649 |
| `--api-monitor` | serve/server.py:6650 |
| `--idle-unload` | serve/server.py:6653 |
| `--min-free-vram-mib` | serve/server.py:6657 |
| `--before-load` | serve/server.py:6660 |
| `--slot-save-path` | serve/server.py:6662 |
| `--replicas` | serve/server.py:6665 |

**Env:**

| Name | Where / notes |
|---|---|
| `STRATA_API_KEY` | serve/server.py:6643 |
| `STRATA_ALLOWED_HOSTS` | serve/server.py:6824 |
| `STRATA_MAX_BODY_MIB` | serve/server.py:4979 |
| `STRATA_HTTP_BACKLOG` | serve/server.py:6103 |
| `STRATA_ENGINE_STALL_S` | serve/server.py:256 (90) |
| `STRATA_ENGINE_READY_S` | serve/server.py:268 (900) |
| `STRATA_VISION_READY_S/_ENCODE_S` | serve/server.py:266-267 (300) |
| `STRATA_START_RETRIES` | serve/server.py:528 |
| `STRATA_REQUEST_LINES` | serve/server.py:804 |
| `STRATA_BATCH_TRACE` | serve/server.py:721 |
| `STRATA_BATCH_MTP` | serve/server.py:2624 |
| `STRATA_PARALLEL_SOLO` | serve/server.py:1467 |
| `STRATA_REPLICA_START` | serve/server.py:6556 |
| `STRATA_SYCL_BIN` | serve/server.py:2490 |
| `STRATA_KEEP_EMPTY_TURNS` | serve/frontend.py:62 |
| `STRATA_DEBUG` | serve/mcp.py:131 |
| `STRATA_DROP_CACHE_ON_EXIT` | serve/cache_drop.py:16 |
| `STRATA_KV_GROW_HOLD` | engine, src/program/generate.cpp:11144 |

**Config keys (strata-<model>.json):**

| Name | Where / notes |
|---|---|
| `parallel` | serve/server.py:2615 |
| `mcp_servers` / `mcpServers` | serve/server.py:11; serve/server.py:6646 |
| `env_to_user` | serve/server.py:6868 |
| `api_key` | serve/server.py:6814 |
| `cors_origins` | serve/server.py:6821 |
| `host` | serve/server.py:6676 |
| `vision` | serve/server.py:2582 (object with `elastic`, ...) |
| `reasoning` | effort defaults (`lift_reasoning_effort_arg` serve/server.py:6436, `sampling_defaults_from_config` serve/server.py:6469) |

## 5. Gotchas / constants / surprises

- One engine turn: `Service.fifo` (serve/server.py:3042) serializes everything without `--batch`. HTTP is threaded, but the GPU is not concurrent by default.
- The engine is a **child process over text pipes**. Token ids are sent as one comma list per request (`GEN`), so a very long prompt is one huge stdin line.
- `CTX_SLACK`=8 (serve/server.py:216): the engine refuses prompt+max_new+8 > context, and the server mirrors this check.
- Silence and stall watchdogs: `ENGINE_SILENCE_S`=300 (serve/server.py:224), PP_CHUNK_MAX=32768, PP_FLOOR_TOK_S=50, PP_SLACK=3 (serve/server.py:231-233). The config key `engine_silence_s` sets the silence limit (0 = wait forever).
- Non-finite logits (`DONE.. nonfinite`) → 503. The engine then drops every cache and the next prompt is read from 0.
- Prefix reuse needs **byte-identical rendered prefixes**: tool order is stabilized, and env/billing blocks are moved or pinned. Changing the template or tool JSON silently kills cache hits.
- `strata_prefix` is ignored for image requests. `pick_slot` prefix logic is skipped for pipelined groups (`slot_gs`).
- Responses API is stateless: `previous_response_id`-style retrieval is 404.
- No grammar-constrained decoding. JSON mode is prompt plus validation only.
- Hardcoded Qwen/ChatML markers: IM_END `<|im_end|>` serve/server.py:72, IMAGE_PAD serve/server.py:73, the tool XML in serve/frontend.py:672-679. The `--serve` turn token is configurable on the engine side (src/program/generate.cpp:652).
- Default model name "qwen3.8-flash-next" (Service.__init__ serve/server.py:3027). Family mapping: qwen4exp → qwen3.8-flash-next.
