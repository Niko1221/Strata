# Strata: All Options — Environment Variables, CLI Switches, Commands

The complete reference for every knob Strata exposes: the engine's CLI flags, its environment
variables (grouped by graphics-card vendor), the server layer's options, the installer's options,
the run-config JSON keys, and the runtime commands. Each entry lists accepted values, the default,
what it does, and when to touch it.

**How to read the entries**

- **Values / default** are taken from the code that parses them. `0|1` switches treat any set value
  as "on" only where the code says so; most compare against the string `1`.
- **Gate** = which hardware the switch can affect. Strata compiles one CUDA/HIP codebase for
  NVIDIA *and* AMD, so many switches live in shared code and are gated at runtime by architecture
  (`gfx12` = RDNA4, `gfx1151` = RDNA3.5, `gfx906` = Vega, `sm_70` = Volta, …). The Intel port is a
  separate `sycl/` tree.
- **Tags**: `[tunable]` user-serviceable · `[debug]` diagnostics/logging · `[ab]` A/B benchmark arm
  (leave alone in production) · `[internal]` plumbing, do not touch.
- `(docs: FILE)` = the option is explained further in that file under `docs/`.

**How options reach the engine**

1. **CLI flags** — on the engine command line (`strata … --flag value`). The server passes the
   `args` list from the run config verbatim.
2. **Environment variables** — in the shell, in the systemd unit, or in the run config's `env`
   block (`"env": {"STRATA_X": "1"}`), which the server exports into the engine process.
3. **Run config JSON** (`strata-<model>.json`) — the server's own keys plus `args`/`env` for the
   engine. `setup.py` writes it; the web Settings page edits a documented subset (see
   [Web UI](#web-ui-settings-surface)).
4. **Per-request keys** — a few knobs work per request via `sampling.strata_tune`
   (`pcie_frac`, `spec_min_p`, `aux_cpus`), `strata_checkpoint`, `strata_prefix`,
   `experimental_speed_projection` (`cvec=0|1`). See [Runtime commands](#runtime-commands).

---

## 1. Engine CLI flags

Single binary, one mode flag: `strata generate …` runs one prompt; `strata … --serve` runs the
interactive engine that `serve/server.py` drives over stdio. `strata --help` prints this list.

### Model, pack, and PLE

| Flag | Values / default | What it does | Best for |
|---|---|---|---|
| `--pack DIR` | dir (`pack/full`) | The pack directory to run. | Every run. |
| `--tokens LIST` | token ids | The prompt as comma-separated token ids (required; the build has no tokenizer). | Scripted runs. |
| `--tokens-file PATH` | path | Pretokenized prompt from a file (commas or whitespace). | Long prompts. |
| `--ple-gguf PATH` | path | The required PLE/n-gram shard (original second GGUF). **Without it layer 1's PLE is silently skipped and every number downstream changes.** | Any real run. |
| `--no-ple` | flag | Diagnostic ablation of the PLE layer. | A/B only. |
| `--ple-io direct\|mmap\|ram` | `direct` | N-gram table reads: `direct` = unbuffered SSD (never in RAM/cache), `mmap` = A/B arm, `ram` = whole table locked in RAM at start (no SSD read on the token path). | `ram` on big-RAM boxes for lowest token latency. |
| `--ple-row-cache N` | int (1048576; 0=off) | Bounded cache of fetched 90 B rows. | Tuning PLE hit rate. |
| `--ple-inflight N` | int (256) | Outstanding SSD reads for the PLE table. | Slow SSDs. |
| `--ple-delay-us U` | µs | Fault injection: each row read completes no earlier than U µs. | Testing only. |
| `--ple-sync-submit` | flag | A/B arm: submit table reads on the token thread (default: an I/O thread). | A/B only. |
| `--no-ple-prefetch` | flag | A/B arm: read the 16 PLE rows one at a time (default: one PrefetchVirtualMemory call). | A/B only. |
| `--embd-gguf PATH` | path | Token embedding from this GGUF (BF16, mapped host memory, no VRAM) instead of `--native`'s. | VRAM recovery. |
| `--native-head-gguf PATH` | path | Native Q5_K output head from model shard 1; requires `--stream-token`. | Native-path runs. |
| `--native-dense-gguf PATH` | path (repeatable) | Native GDN/QSA/shared projections from each source model shard. | Native-path runs. |

### GPU selection and multi-GPU

| Flag | Values / default | What it does | Best for |
|---|---|---|---|
| `--gpu LIST` | indices | Run on these GPUs (nvidia-smi/PCI indices; default: every visible one). Sets `CUDA_VISIBLE_DEVICES`/`CUDA_DEVICE_ORDER` inside the engine. | Pinning the model to a card. |
| `--layer-split SPEC` | layer numbers | Split model layers across GPUs (see docs/MULTI_GPU.md). | Two+ cards. |
| `--trim-stage-weights` | flag | With an explicit `--layer-split`: each GPU loads only its own layers' dense weights. | Tight VRAM splits. |
| `--pipeline-windows N` | 0/1/2 (auto-on for 2+ AMD gfx12 stages) | Overlap the two split stages: 1 = prompt's short reads, 2 = decode too. `STRATA_PIPELINE_AUTO=0` or `N=0` turns it off. | Layer-split decode. |
| `--expert-cache-device1/2/3 N\|auto` | int/auto | Fill CUDA1/2/3 with N experts (or as many as fit). | Multi-GPU expert caches. |
| `--expert-cache-remote-placement stripe\|layer` | `stripe` | Distribute expert ranks (stripe) or whole layers across CUDA1..3. | Multi-GPU caches. |
| `--remote-expert-opt` | flag | Remote-expert decode optimizations (serve mode). | Multi-GPU serve. |
| `--peer-device N` | index | A second GPU as an adaptive expert-cache tier over P2P; it also computes its experts' prompt rows. Not with `--layer-split`/`--expert-cache-device*`. | Big expert packs on two cards. |
| `--peer-reserve-mib N` | MiB (600) | VRAM the peer tier leaves free on its card. | Peer with other workloads. |
| `--peer-slots N` | int (fits) | Expert slots on the peer. | Peer tuning. |
| `--peer-adapt-swaps N` | int (=--adapt-swaps) | Peer cache swaps per adaptation step. | Peer tuning. |
| `--peer-prefill-rows N` | int (half chunk×top-k; 0=off) | Prompt rows per layer the peer computes. | Peer prefill tuning. |

### KV cache and context

| Flag | Values / default | What it does | Best for |
|---|---|---|---|
| `--kv fp16\|int8\|q4_0\|k8v4` | `fp16` | KV storage: `int8` = codes + fp16 scale per 64, half the VRAM; `q4_0` = 4-bit after Hadamard rotation (half of int8, slightly lower precision); `k8v4` = INT8 K (exact scores) + rotated Q4_0 V, 816 B/cell. | Long contexts: `int8` default above 8K, `k8v4` for more room. |
| `--kv-resident N` | cells (0=all in VRAM; min 20240) | KV streaming: keep N cells per QSA layer in VRAM, whole K/V in pinned RAM; freed VRAM goes to expert slots. | Huge contexts on small cards. |
| `--kv-grow` | flag | K/V takes VRAM only for cells requests reach; expert cache holds the rest and gives slots back as context grows (one GPU, with `--expert-profile`; `STRATA_KV_GROW=1/0` also). | Mixed short/long traffic. |
| `--max-context N` | int (4096) | KV/state capacity. | Every run. |
| `--rope-scaling none\|linear\|yarn` | model file's keys, else none | Context extension past the trained length (llama.cpp types). Fixed at startup. | Long-context models. |
| `--rope-scale F` | float (model's, else 1) | Extension factor for linear/yarn. | With `--rope-scaling`. |
| `--rope-freq-base N` / `--rope-freq-scale F` | float (0=model) | Raw ggml RoPE knobs: frequency base / angle shrink. | Fine RoPE tuning. |
| `--yarn-orig-ctx N` | int (0=262144) | Trained context the YaRN correction targets. | YaRN. |
| `--yarn-ext-factor F --yarn-attn-factor F --yarn-beta-fast F --yarn-beta-slow F` | floats (-1/1/32/1) | YaRN's knobs. | YaRN. |
| `--stream-token` | flag | Enqueue token work on the session stream (experimental). | Experimental. |
| `--check-logits` | flag | Copy and check all logits in the stream-token path. | Debugging. |

### Expert cache, CPU pool, RAM modes

| Flag | Values / default | What it does | Best for |
|---|---|---|---|
| `--expert-cache N` | int (0) | Keep N expert blobs resident in VRAM, compute their rows on the GPU. Measured 54.4% hits at 4096 slots with `--expert-cache-per-layer`. | Decode speed on roomy VRAM. |
| `--expert-cache-per-layer` | flag | Give each layer its own slots (shared-counter default measured 2.97% hits; per-layer 21.4% at 8/layer, 70.4% at 64). | Any expert cache. |
| `--expert-profile P` | path | Pre-load the VRAM tier from a `profile.bin` (tools/make_profile.py) instead of admitting on first use. | Warm starts. |
| `--expert-profile-save P` | path | `--serve`: save what the adaptive tier learned, on QUIT and every `--expert-profile-save-every` MIN minutes (10; 0=QUIT only). | Persisting learned routing. |
| `--adapt-min-gain F` | float (1.5) | Swap an expert in only when its decayed routing count beats the victim by >F. Higher = fewer, surer swaps. | Swap churn tuning. |
| `--adapt-async 1` | 0/1 (0) | `--serve`, resident/arena mode, one GPU: adaptive swaps advance between verify windows instead of blocking one. Not bit-exact run to run. | Decode throughput (opt-in). |
| `--mmap-experts` | flag | Opt OUT of the resident expert arena, back to mmap/MapViewOfFile (cold 71.97 vs warm 34.78 ms/token). | Low-RAM A/B, page-cache setups. |
| `--shared-expert-arena FILE` | path | Linux: back the resident arena with one MAP_SHARED file (put it on /dev/shm). | Multi-process arena sharing. |
| `--resident-cpu-experts` | flag | With mmap + static profile: keep the GPU cache's complement resident in RAM; adaptive swaps exchange them, no re-reads. | RAM-rich boxes. |
| `--resident-experts` | flag | Setup's low-RAM resident mode: `--mmap-experts --resident-cpu-experts`, page-locked where possible, 4 GiB headroom. | Low-RAM PCs (setup writes it). |
| `--no-pool` | flag | Skip the CPU expert pool (GPU-only floor). | Measuring the GPU floor. |
| `--pool-workers N` | int (0=all physical cores − host's) | CPU expert pool worker count (hybrid CPUs with `--pool-affinity auto/p-cores`: P-cores − 1). | Pool sizing; sweep before claiming gains. |
| `--pool-tasks N` | 0..4096 (0=3/thread) | Batched CPU expert tasks per GU/Down phase. | Pool throughput tuning. |
| `--pool-affinity all\|auto\|p-cores` | `all` | Worker affinity: `all` = one per physical core; `auto` = hybrid CPUs, P-cores first; `p-cores` = P-cores + siblings only. | Hybrid (P/E-core) CPUs. |
| `--host-core first\|last` | `first` | The host thread's core. Windows routes GPU interrupts to one logical CPU (usually the first); a spinning host there waits for them. Moves threads only. Not on hybrid CPUs. (`STRATA_HOST_CORE` also.) | Windows latency. |
| `--aux-cpus SET` | off/auto/list | Where non-pool, non-host threads run: `off` (default), `auto` (host core's SMT siblings, then spare cores), or a CPU list like `24,26`. (`STRATA_AUX_CPUS` also; per-request `strata_tune aux_cpus` beats it.) | SMT-heavy hosts. |
| `--no-host-worker` | flag | A/B arm: host thread does not join the drain (5 workers + idle core). | A/B only. |
| `--no-hit-poke` | flag | A/B arm: don't poke the driver after the hit-path launch (without the poke the GPU work waits for the next driver entry). | A/B only. |

### Speculation (MTP, lookup drafts)

| Flag | Values / default | What it does | Best for |
|---|---|---|---|
| `--spec N` | int (setup: 4) | MTP drafter's verify window: tokens proposed per check. `--suffix-draft` lets it grow by 2 up to 8 where repeats are likely. | Decode speed. |
| `--spec-min-p P` | float (setup: 0.5) | How sure the draft layer must be to extend a window by another guess. `--calibrate` measures it per PC. | Accept-rate tuning (measured: 0 = 9.6% slower on R9700). |
| `--mtp-max-t M` | int (0=--spec) | Cap MTP windows at M tokens; longer ones come from suffixes. | Draft-shape tuning. |
| `--suffix-draft N` | int (3; 0=MTP only) | Prompt-lookup drafting from an earlier repeat of the last N+ tokens when it pays. | Repetitive traffic. |
| `--lookup-chain K` | int (0=off) | After MTP drafts, add up to K prompt-lookup drafts continuing them (window ≤8). | Long repeats. |
| `--lookup-chain-min M` | int (3) | Shortest context match `--lookup-chain` extends. | With `--lookup-chain`. |
| `--mtp-hnorm pooled\|stream` | `pooled` | Draft layer's hidden-input norm: one RMS over all four streams, or one per stream (llama.cpp's form). | Parity A/B. |
| `--mtp-q4 all\|proj\|head` | off | Draft layer's projections and/or draft head as 4-bit copies. | VRAM for MTP. |
| `--mtp-draft-vocab FILE` | path (`<mtp>/draft_vocab.bin`) | Draft head's token subset. | CJK/en draft vocab. |
| `--coupled-draft` / `--no-coupled-draft` | flag | Coupled draft sampling for the MTP drafter under sampling (`STRATA_SPEC_COUPLED`). | Sampling acceptance rates. |
| `--spec-follow PATH` | path | Benchmarks: emit a given continuation and accept matching drafts, so speculation settings compare on the same text. | A/B only. |
| `--window-hashes PATH` | path | Per verify window: index, position, size, 64-bit hash of final residual rows + argmaxes. Two builds doing the same arithmetic write the same file. | Parity checks. |

### Prompt processing (prefill)

| Flag | Values / default | What it does | Best for |
|---|---|---|---|
| `--prefill CHUNK` | int/auto (auto ≤8192) | Batched prompt processing in chunks; `auto` = largest chunk the expert cache can lend; `auto:16384`/`auto:32768` (or `STRATA_PREFILL_AUTO_MAX`) allow bigger. | Long-prompt throughput (32768 ≈ +20% on R9700). |
| `--short-read N` | tokens (64; 0=off) | `--serve`: read ≤N fresh tokens through the decode windows instead of the batched prompt path. | Short follow-up messages. |
| `--prefill-head-alias` | flag | `--serve`: unload the native output head while each fresh prompt is read, release prompt buffers, reload before decode. | VRAM headroom on long prompts. |
| `--prompt-cache N` | int (6) | `--serve`: keep N conversation checkpoints between requests (~118 MB RAM each; 0 = re-read every prompt). | Multi-turn latency. |
| `--prompt-cache-every N` | tokens (16384; 0=off) | `--serve`: also checkpoint every N fresh prompt tokens. | Long-session resume. |
| `--prompt-cache-tail` | flag | `--serve`, single GPU: one extra checkpoint near the prompt's end at a chunk boundary. | Fast final-turn resume. |
| `--turn-token ID` | token (248045) | `--serve`: token that opens a chat turn (`<\|im_start\|>`). | Non-Qwen tokenizers. |
| `--tail-role-token ID` | token (-1=off) | `--serve`: a turn of this role right before the last is left out of the conversation checkpoint (a trailing effort turn). | Effort-turn cache hygiene. |

### Conversation cache and sessions (`--serve`)

| Flag | Values / default | What it does | Best for |
|---|---|---|---|
| `--conversation-cache-mib N` | MiB (0=off) | RAM budget for parked conversations. | Multi-session serving. |
| `--conversation-cache-slots N` | int (4) | At most N parked conversations. | Session caps. |
| `--conversation-cache-min-free-mib N` | MiB (2560) | RAM floor (MemAvailable, cgroup limits, `--memory-limit-mib`) when parking/saving/restoring. | Shared hosts. |
| `--conversation-cache-min-tokens N` | tokens (0) | Park only conversations ≥N tokens, so short side requests don't evict long ones. | Mixed traffic. |
| `--memory-limit-mib N` | MiB (0=none) | Total RAM this engine's container/cgroup may use when the container can't see its real limit (`STRATA_MEMORY_LIMIT_MIB`; flag wins). | Containers. |
| `--session-min-free-mib N` | MiB (4096; 0=off) | Disk space a SAVE must leave free. | Session saves. |
| `--session-save-reclaim` | flag | Reclaim unpinned host caches if SAVE needs RAM. | RAM-tight saves. |
| `--vram-elastic` | flag | `--serve`, NVIDIA opt-in (#533): expert cache in segments (`--vram-segment-mib`, 512) so `POST /v1/vram` can give VRAM back between requests and retake it. (docs/VRAM_ELASTIC.md) | Sharing a GPU with games/apps. |

### Batching (`--serve`)

| Flag | Values / default | What it does | Best for |
|---|---|---|---|
| `--batch N` / `--slots N` | 2..8 | Up to N requests decode together in batch slots, each with its own session (VRAM like the main one). (docs/BATCHING.md) | Concurrent users. |
| `--batch-mtp` | flag | `--batch` + `--mtp --spec`: each slot also verifies one MTP proposal per window (`STRATA_BATCH_MTP=1` same). One GPU or layer split; not with `--batch-groups`. | Batched speculation. |
| `--batch-groups G` | int | `--batch` with a layer split: slots in G groups pipelined through the GPUs. | Multi-GPU batching. |

### Native paths and precision (mostly `[ab]`/experimental)

| Flag | Notes |
|---|---|
| `--native SHARD1` | Every full-context native path at once (stream-token, GR MMVF, BF16, head, dense + PLE key, MoE combine, GDN, router, QSA, indexer, RoPE, PLE postops, CPU q8_0 contract). Individual flags below stay for A/B. |
| `--native-bf16`, `--native-bf16-extra` | CUDA-oracle SSM/router/indexer (and PLE/shared gate) BF16 projections. |
| `--gr-fp32-activations`, `--gr-native-mmvf` | CUDA-oracle GR activation precision; pinned GR norm/projections (implies FP32). |
| `--native-moe-combine`, `--native-gdn`, `--native-qsa`, `--native-qsa-indexer`, `--native-rope`, `--native-ple-postops`, `--native-ple-key`, `--native-router`, `--native-head-gguf` | Pinned native path per subsystem; see `--help` for exact scope. |
| `--native-flash-attn-short` | Diagnostic pinned vector attention; `--max-context ≤256`. |
| `--cpu-oracle-q8-0` | Pinned CPU expert quantization and dot reduction. |
| `--expert-cache-cpu-order` | GPU expert reduction matching CPU order. |
| `--no-fused-gr` | Six-kernel hyper-connection read + separate write (native). |
| `--no-token-graph` | Two graphs per layer (host launches each) instead of one per token. |
| `--no-capture` | Run layers directly instead of replaying graphs. |
| `--shared-late` | Shared expert after the CPU pool (default: overlapped). |
| `--keep-canonical` | Also load canonical copies of natively served tensors (more VRAM). |

### Sampling and output

| Flag | Values / default | What it does |
|---|---|---|
| `--greedy` | flag | Argmax (the default). |
| `--seed S` | int | Enable sampling with this Philox seed. |
| `--top-k N --top-p F --temperature F` | | Standard sampling knobs. |
| `--max-new N` | int (16) | Tokens to generate. |
| `--dump-logits PATH` / `--logits-stride N` | | Raw logits per position; stride N>1 requires `--max-new 1`. |
| `--dump-residual / --dump-layers / --dump-halves / --dump-routing / --dump-mixed PATH` | | Bisection/debug dumps (head, per-layer, halves, routed experts, post-attention residual). |
| `--control-vector-scaled FILE:SCALE[,...]` | path:float | Control vector on the residual stream (llama.cpp format; `--control-vector FILE` = scale 1). `--serve`: requests switch it (`cvec=0\|1`). |
| `--control-vector-layer-range A B` | ints (1..last) | Layers the control vector follows (inclusive). |
| `--cvec-mode add\|project` | `add` | `h += s·v` (add) or `h -= s(h·v)v` with unit v (project). |
| `--cvec-dir per-layer\|single:L` | `per-layer` | Each layer's own direction, or layer L's everywhere (project). |

### Measurement and diagnostics

| Flag | Notes |
|---|---|
| `--stats` | Per-stage breakdown. |
| `--stage-timing` | Per-stage KERNEL-COUNT shares (not a time profile; pair with `--gpu-only-full`). |
| `--gpu-stages` | Capture the layer as three graphs (mixer / ffn+router / post), time from outside the capture. Prints and exits. |
| `--graph-only` | MEASURE: replay the 48 `pre` graphs only — NOT the GPU floor. |
| `--gpu-only-full` | MEASURE: pre+post for all 48 layers + LM head, no pool. The true per-token GPU floor. |
| `--sync-every-layer` | Debug: synchronize after every layer. |
| `--vision` | `--serve` takes images (embeddings from strata-vision). (docs/IMAGES.md) |

---

## 2. Server CLI (`python -m serve.server`)

| Flag | Default | What it does |
|---|---|---|
| `--engine mock\|strata` | `mock` | Which engine to drive. |
| `--config PATH` | — | Engine config JSON (exe, args, cwd, tokenizer, model_name, …). |
| `--host` | config/127.0.0.1 | Listen address (`0.0.0.0` = LAN). |
| `--port` | 8095 | API port. |
| `--script` | (appendable) | Extra script path. |
| `--gpu LIST` | — | GPU(s) as nvidia-smi numbers them; several for a layer split. |
| `--tokenizer PATH` | `pack/full/tokenizer` | Tokenizer directory. |
| `--open` | off | Open the chat page in the browser when the model is ready. |
| `--fit-max-tokens` | off | Clamp `max_tokens` to the remaining context instead of 400. |
| `--api-key KEY` | `STRATA_API_KEY` | Require this key from clients. |
| `--mcp-config FILE` | — | MCP servers in Claude Desktop's format. (docs/MCP_SERVER.md) |
| `--lazy` | off | Start unloaded; first request loads the engine (text only). |
| `--api-monitor` | off | `/api-monitor`: keep the last 100 requests in memory. |
| `--idle-unload SECONDS` | none | Unload after this long without requests. |
| `--min-free-vram-mib N` | none | Load an unloaded model only when this much VRAM is free, else 503. |
| `--before-load CMD` | — | Command run before the model loads again (e.g. unload something else). |
| `--slot-save-path DIR` | none | Directory for session slot save/restore. |
| `--replicas N` | none | N independent engines. |

Server env vars: see [Server layer](#8-server-layer-serve).

---

## 3. Installer CLI (`setup.py` / `setup.sh` / `SETUP.bat`)

| Flag | Default | What it does |
|---|---|---|
| `--family` | asked | `qwen` = Qwen3.8-Flash-Next, `swift` = Swift 1.5 (`qwen36` = download only). |
| `--model` | asked | Model within the family. |
| `--context` | asked | Context length to configure. |
| `--rope-scaling none\|linear\|yarn` | — | RoPE extension past the trained 262144. |
| `--rope-scale F` | context/trained ≥1 | Extension factor. |
| `--kv int8\|q4_0\|k8v4` | int8 above 8K | KV cache precision. |
| `--vision yes\|no\|none\|gpu\|cpu` | asked | Image reading (encoder on GPU/CPU). |
| `--vision-tokens N` | 1024 GPU / 300 CPU | Max image tokens per picture. |
| `--experimental-speed-projection on\|off\|GGUF` | off | EXPERIMENTAL control vector (data/experimental-speed-projection). |
| `--port` | 8080 | Server port. |
| `--gpu` | asked | One GPU (nvidia-smi numbering; AMD: as setup lists). |
| `--gpus` | — | Several GPUs sharing one model. |
| `--layer-split` | — | With `--gpus`: where each later GPU's layers start ("18", "16,32"). |
| `--no-remote-expert-opt` | off | With 2+ GPUs: omit `--remote-expert-opt` (#578). |
| `--host` | 127.0.0.1 | `0.0.0.0` = also serve the LAN. |
| `--api-key` | none | Require key(s) (comma-separated for several). |
| `--no-browser` / `--browser` | browser on | Open the chat page when ready. |
| `--data-dir` | `Strata-data/` | Where model files go (~70–120 GB). |
| `--models-dir` | `<data>/models` | GGUF location. |
| `--gguf-dir` | — | Use GGUF files you already have. |
| `--yes` | off | Accept recommended answers. |
| `--setup` | — | Install another model / change settings. |
| `--no-start` | — | Install only, don't start. |
| `--update` | — | Update engine, packages, model settings as a start would. |
| `--rollback-engine` | — | Restore the engine an update replaced (engine/.previous). |
| `--build` | — | Compile the engine instead of the prebuilt. |
| `--source auto\|huggingface\|modelscope` | auto | Model download source. |
| `--inspect SOURCE VARIANT` | — | What a GGUF is and whether Strata runs it, headers only. |
| `--cuda 12\|13\|auto` | auto (13) | NVIDIA: CUDA toolkit for the engine. (`STRATA_CUDA`) |
| `--prebuilt URL` | PREBUILT_URL | Where the ready-made engine is. (`STRATA_PREBUILT_URL`) |
| `--check` | — | Only check this PC and exit. |
| `--calibrate` | — | Tune engine settings for this PC (~15–30 min), then start. |
| `--draft-vocab cjk\|en` | cjk | Draft layer's token subset. |
| `--low-ram auto\|on\|off\|resident\|mmap` | auto | Experts from file instead of RAM copy. |
| `--resident-budget-gib N` | RAM−24GB (40 on 64) | GiB of experts kept in RAM (UD-Q4_K_XL / UD-IQ4_XS). |
| `--vram-reserve-mib N` | engine default | VRAM left free for other programs. |
| `--parallel N` | 1 | Up to N requests decode together (batch slots). |
| `--replicas N` | — | N independent engines on equal GPU groups. |
| `--kv-streaming auto\|on\|off` | auto | From 64K context: KV in RAM, attention window in VRAM. |
| `--backend cuda\|hip\|sycl` | cuda | cuda = NVIDIA, hip = AMD, sycl = Intel Arc. |

---

## 4. Run-config JSON keys (`strata-<model>.json`)

The server reads this at start; `setup.py` writes it; the web Settings page edits a subset.
Keys seen by `serve/server.py`:

`exe`, `args`, `env`, `cwd`, `log`, `backend`, `gpu`, `gpu_order`, `hip_ordinal`, `host`,
`port`, `api_key`, `cors_origins`, `trusted_origins`, `allowed_hosts`, `model_name`,
`tokenizer`, `aliases`, `sampling`, `reasoning_effort`, `reasoning_budget_tokens`,
`reasoning_loop_recovery`, `reasoning_close_retry`, `effort_position`, `anthropic_thinking`,
`fit_max_tokens`, `min_tokens`, `max_tokens`, `repeat_stop_tokens`, `tool_call_recovery`,
`threads`, `parallel`, `replicas`, `replica_cpus`, `replica_load_tokens`, `vision`,
`images`, `vram_elastic`, `vram_segment_mib`, `elastic`, `elastic_mib`, `elastic_idle_s`,
`min_free_vram_mib`, `idle_unload_s`, `lazy_load`, `engine_silence_s`, `api_monitor`,
`open_browser`, `prevent_sleep`, `slot_save_path`, `layer_split`, `split_skip_if_fits`,
`expert_profile_save`, `expert_profile_save_every`, `env_to_user`, `before_load`,
`codex_thread_titles`, `codex_compaction_cache`, `lib_dirs` (extra library dirs, e.g. ROCm
runtime paths).

`args` is the engine CLI list; `env` exports environment variables into the engine process —
that is the supported way to set every `STRATA_*` switch below.

---

## 5. Runtime commands

- **Web UI**: `/` chat + Settings; `/monitor` live stats; `/api-monitor` last 100 requests
  (needs `api_monitor`).
- **`GET`/`POST /config`** — the Settings surface (see [Web UI](#web-ui-settings-surface)).
- **`POST /v1/vram`** (`VRAM <reserve_mib>`) — with `--vram-elastic` (NVIDIA): give VRAM back to
  other programs between requests and retake it later. (docs/VRAM_ELASTIC.md)
- **`/slots/N?action=save|restore`** — session slot save/restore (needs `slot_save_path`).
- **Per-request keys** (in `sampling`): `strata_tune: {pcie_frac, spec_min_p, aux_cpus}` —
  measured per-request tuning without restart; `strata_checkpoint: false` — one-shot call, no
  conversation checkpoint; `strata_prefix: {tokens: N}` — pin the checkpoint at a shared prefix;
  `strata_img_keys: [hex…]` — match pictures by key; `experimental_speed_projection: true|false`
  — toggle the control vector per request (`cvec=0|1`).
- **Engine stdio control lines** (server↔engine): `GEN`/`BGEN` requests, `T` tokens, `DONE`,
  `BADM`, `STOP`, `QUIT`.

---

## 6. Tools (commands)

User-facing tools in `tools/`:

| Tool | What it is |
|---|---|
| `tools/calibrate.py` | The `--calibrate` engine: measures per-PC settings (`--spec-min-p`, PCIe fractions). |
| `tools/make_profile.py` / `make_profile_mix.py` | Build `profile.bin` expert profiles for `--expert-profile`. |
| `tools/mtp_fetch.py` / `mtp_pack.py` / `draft_vocab.py` | Fetch/pack the MTP draft layer and draft vocab. (`STRATA_MTP_REVISION`, `STRATA_SOURCE`, `HF_ENDPOINT`, `MODELSCOPE_ENDPOINT`) |
| `tools/strata_bench.py` | Throughput benchmark client. |
| `tools/strata_inspect.py` | Inspect packs/models. |
| `tools/strata_mcp.py` | MCP server bridge. (docs/MCP_SERVER.md) |
| `tools/needle_bench.py` | Long-context needle test (`--lengths 256k`). |
| `tools/embd_bf16_pack.py` | BF16 embedding pack for `--embd-gguf`. |
| `tools/nvfp4_convert.py` / `nvfp4_verify.py` | NVFP4 quant conversion (NVIDIA). (docs/NVFP4.md) |
| `tools/hip/tune_hipblaslt.cpp` | hipBLASLt tuning-table calibrator (build target `tune_hipblaslt`); `--case dtype,T,N,K,ldy --tuning-out PATH`. Shipped tables: `tools/hip/gfxNNNN-hipblaslt-<ver>.txt`. |
| `tools/image/qimage.py` + `tools/image/sdcpp_native` | Qwen-Image generation. (docs/IMAGES.md) |
| `tools/vision/strata_vision.cpp` | Vision encoder server for `--vision`. |

Everything else under `tools/` is test/soak/parity harnesses.

---

## 7. Environment variables by vendor

The engine reads 511 `STRATA_*` environment variables (plus 20 generic ones), every production site in the tree. Most are A/B arms, diagnostics, or
plumbing; the `[tunable]` ones are the user-serviceable set. Set them in the run config's
`env` block, the systemd unit, or the shell before starting the engine.

### 7.1 NVIDIA-specific (35)

Switches that only exist in, or are gated to, the NVIDIA (CUDA) path: SM-generation tables,
NVFP4, PDL, WDDM, CUDA runtime behavior.

**`STRATA_ADAPT_SIGNAL`** — `host` / `copy` (default: stream-ordered 4-byte copy on Linux, host function on Windows) — chooses how the adaptive overlap's per-layer "copies landed" signal is delivered: a stream-ordered copy into pinned memory, or a host callback function. Gate: Windows WDDM defaults to host (unflushed command buffers stall the copy signal, #2188); Linux defaults to copy. Best for: set `=copy` on Windows if layers report "copies never landed", `=host` on Linux if driver callback threads are starved. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_DBG_GDN`** — `1` (default: off) — forces the per-layer GDN commit kernel path (and the debug kernel variant) instead of the batched multi-layer commit. Gate: CUDA GDN verify kernels. Best for: leave at default; kernel-development diagnostics. Tags: [debug]

**`STRATA_DEFERRED_REGISTER`** — non-empty non-`0` value (default: off) — registers the host expert arena per layer as it is filled (pipelined with the load) instead of one whole-arena cudaHostRegister before it, cutting ~2.5 s of startup at 33 GiB. Gate: Windows only (Linux keeps the whole-arena pin, #253); only when no pin cap, no shared arena. Best for: faster startup on Windows; measured ~10% slower decode on an RTX 5070, so it stays opt-in. Tags: [ab]

**`STRATA_DENSE_F32`** — `1` (default: off) — runs the large dense GEMMs (qkv, query, gate, output projections) as `cublasSgemm` on FP32 copies of the FP16 operands, which beat cuBLAS's FP16-operand kernels on Pascal. Gate: NVIDIA sm_60–sm_69 only (Pascal: Tesla P40/P4, GTX 10 series); ignored on other cards. Best for: set `=1` on a P40-class card — measured 360-432 → 464-474 tok/s on 7K prompts. Tags: [ab] (docs: docs/OLDER_GPUS.md, docs/COMMUNITY_BENCHMARKS.md)

**`STRATA_DEQUANT_WIDE`** — `0`/`1` (default: on for sm_61, card default elsewhere) — switches the Q4_K/Q5_K/Q6_K/IQ4_XS dequantization between the wide kernel (a thread per 8 values, 16-byte stores; 10→227 GB/s on a P40) and the old thread-per-32-values kernel. Gate: NVIDIA CUDA; on by default on sm_61 (Tesla P40), measured there. Best for: `=0` only to A/B the dequant path on Pascal. Tags: [ab] (docs: docs/OLDER_GPUS.md)

**`STRATA_DF_BRANCH`** — `1` (default: off) — runs the mixer's query side and K/V side as side-branch streams that fork/join around the block scores instead of one stream. Gate: NVIDIA CUDA graph path; off by default because on Linux (open modules, GSP) a branched window graph stalled inside the graph when an NVML query arrived between requests (#905, 2x RTX 3090 / 2x RTX 2080 Ti). Best for: leave at default; `=1` to experiment with mixer parallelism. Tags: [tunable]

**`STRATA_DF_PDL`** — `1`/`2` (default: off) — captures the verify window graph with programmatic dependent launch edges (griddepcontrol) between kernels; `=2` restricts to a single programmatic predecessor per node. Gate: NVIDIA sm_90+ (Hopper and newer) with the kernel built for PTX/binary ≥ 9.0; opt-in pending A/B. Best for: leave at default; benchmark arm for window-graph launch latency. Tags: [ab]

**`STRATA_DF_PLE`** — `1` (default: off) — runs the window's post-projection steps (norms, history advance, commit snapshots) in one batched pass over the window's rows, bitwise-equal to the per-token loop. Gate: NVIDIA CUDA batched decode path (needs the batched KV path, not batch-rec). Best for: leave at default; A/B kernel-latency experiments. Tags: [ab]

**`STRATA_DF_QB_Q4`** — `1` (default: off) — extends the batched per-token query path (GEMVs/norms/RoPE as one launch) to a Q4_0 KV cache as well, since the query rotation is row-wise; `0` keeps Q4_0 token-by-token. Gate: NVIDIA CUDA, batched decode with native QSA and RoPE. Best for: `=1` when running a Q4_0 KV cache on the batched path. Tags: [tunable]

**`STRATA_DMA_BATCH`** — `0`/`1`/`2` (default: 0) — submits a verifier group's independent expert uploads through cudaMemcpyBatchAsync (mode 1) and additionally sets CUDA 13.4's PreferOverlapWithCompute flag (mode 2, a scheduling hint). Gate: NVIDIA CUDA 13.0+ only, with `--pcie-mode dma`; modes 2 on CUDA 13.0-13.3 behave like 1. Best for: A/B batched PCIe uploads on CUDA 13; leave at 0 elsewhere. Tags: [ab] (docs: docs/BATCHED_DMA.md, docs/DETAILS.md)

**`STRATA_DUMMY_NODES`** — integer k (default: 0) — measurement only: injects k empty kernel nodes per layer into the window graph to measure what one graph node costs. Gate: CUDA graph capture path. Best for: leave at default; graph-overhead measurement. Tags: [debug]

**`STRATA_DUMP_MOE_INPUT`** — file path (default: unset) — dumps the first prefill chunk's MoE input for the chosen layer (T,N,K header, T×N floats, T×K expert ids) so kernel products can be tested on real rows. Gate: CUDA prefill path. Best for: leave unset; parity testing (e.g. `mmq_nvfp4_parity --real`). Tags: [debug] (docs: docs/NVFP4.md)

**`STRATA_DUMP_MOE_LAYER`** — layer index (default: 0) — selects which MoE layer's input STRATA_DUMP_MOE_INPUT dumps. Gate: same as STRATA_DUMP_MOE_INPUT. Best for: pair with the dump var; diagnostics only. Tags: [debug] (docs: docs/NVFP4.md)

**`STRATA_EMB_REUSE_ACCOUNT`** — `1` (default: off) — lets the prompt planner count the VRAM freed by the half-output buffer sharing the embedding buffer, growing the auto chunk and borrowed cache slots where VRAM is the limit (RTX 3060 IQ3_XXS: chunk 6400→6656, prompt +3.9%) at the cost of different prompt-path rounding. Gate: CUDA prompt path; opt-in until KL-measured. Best for: `=1` on VRAM-limited cards wanting a bigger prompt chunk. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_IO_PF_STAGE`** — `1` (default: off) — instead of warming the page cache, stages every uncached expert of a layer in a buffer and makes the layer wait for them. Gate: Linux mapped file tier with STRATA_IO_PREFETCH=1. Best for: `=1` only on 32 GB-class boxes (RTX 3060 32 GB: +25.7%; slower at 16 GB and on iGPUs). Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_MMQ_RESIDENT_SORT_NE`** — `1` (strcmp == "1"; default: off) — requests the resident sort for MMQ prompt packing on native-layout expert packs (slot/src/off built identically for both layouts); measured on a GP100 (sm_60) it cut prompt times with bitwise-identical bits. Gate: NVIDIA sm_60 measured, MMQ prompt path with native layout. Best for: A/B on older NVIDIA cards running MMQ prompts. Tags: [ab]

**`STRATA_PARTIAL_PIN`** — integer (atoi; default: off) — Registers the hottest prefix of the RAM expert budget with the GPU driver so the GPU computes a share of cache misses over PCIe; measured no faster on the reference RTX 5070 and changes those experts' numerics, so it is off by default. Gate: all GPUs, no gate; most relevant on WDDM Windows boxes. Best for: leave off; experiment with `--pcie-frac` setups. Tags: [tunable] (docs: docs/UNSLOTH_Q4.md)

**`STRATA_PCIE_AUTO_CANON`** — `1` (default: on by default on Linux CUDA; Windows needs the opt-in; not applicable on HIP) — lets the canonical Q2_0 pack's PCIe share be probed and refined by PcieAuto from the CPU pool's speed instead of keeping its constant 0.2 start. Gate: NVIDIA CUDA only (Windows opt-in because an RTX 5070 was mixed: code +2.5%, short answer -3.3%). Best for: enable on Windows to auto-calibrate the canonical pack's share. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_PF_PEER_HOST`** — `0`/`1` (default: on) — when the peer GPU has no P2P, routes the prompt's activations and result rows through the mapped-host buffers (as the layer split's helper does); `=0` refuses the peer and keeps the prompt path on the primary GPU. Gate: multi-GPU prefill peer path without P2P, needs the MMQ prompt path, not for NVFP4 experts. Best for: leave at default; `=0` to force the old refuse behavior. Tags: [tunable]

**`STRATA_PIN_BY_NEED`** — `1` (default: off, opt-in) — pins expert blobs by need instead of by prefix: the arena registers the profile's non-GPU-resident experts by rank, hottest first, until the driver refuses, so blobs past Windows' ~30 GiB page-lock budget still get a PCIe share. Gate: Windows WDDM single GPU with `--expert-profile`, no layer split, peer GPU, or helper caches. Best for: Windows boxes where the shared page-lock budget caps prefix pinning. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_PREFILL_CPU_SHARE_MAX`** — integer >= 1024 (default: 3072; 1024 when the share runs on its default arming) — the prompt-chunk size below which chunks are staged after routing so they can hand the CPU a share, instead of streaming every expert. Gate: same paths as STRATA_PREFILL_CPU_SHARE. Best for: `=1024` keeps the old limit; `=3072` extends the share to larger chunks (lost 2% at 2,048 on an RTX 5070). Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_PREFILL_GROUP_GATHER`** — integer (atoi; default: on) — Batches stream gathers into groups (one gather/wait/record per group) to kill the per-expert ~10 µs WDDM launch bubbles; `=0` returns to one gather per expert. Gate: MMQ native-pack streaming path (stream_all + MMQ), all GPUs. Best for: leave at default; A/B arm. Tags: [ab]

**`STRATA_PREFILL_NVFP4`** — `w4a8` (default), `w4a4`, `fp16` (docs also measure `w4a4x2`) — picks how the batched prompt path multiplies NVFP4 experts (weights-only FP8 activations, FP4 activations, or the exact FP16 path); an unknown value exits with an error. Gate: NVIDIA NVFP4 expert files (Blackwell-era tensor cores). Best for: A/B prompt speed against first-token KL (w4a8 KL 0.0068, w4a4x2 0.0056). Tags: [ab] (docs: docs/NVFP4.md)

**`STRATA_PROMPT_ATTN_OLD`** — any value (default: unset) — Runs prompt QSA attention through the decode kernel (32 queries at a time) instead of the tensor-core whole-chunk kernel. Gate: all GPUs; documented for Volta (V100) and older-GPU recipes. Best for: set on pre-sm_80 cards or for A/B parity checks. Tags: [tunable] (docs: docs/OLDER_GPUS.md, docs/NVIDIA_V100.md)

**`STRATA_PROMPT_ATTN_Q4`** — `0`/`1` (default: on) — `=0` disables the mode-4 tensor-core prompt attention for Q4_0 KV, falling back to the old kernel. Gate: NVIDIA sm_80+ (Turing excluded from mode 4). Best for: leave at default; A/B arm. Tags: [ab]

**`STRATA_PROMPT_DIRECT`** — `1` (default: off, opt-in) — reads the prompt path's experts unbuffered (bypassing the page cache) while decode keeps the file cache; the same bytes either way (RTX-class box: 4K prompt 152 -> 315 tok/s, 20K 217 -> 500, decode +2-15%). Gate: Linux file tier. Best for: SSD-bound prompt reads on big MoE files. Tags: [tunable] (docs: docs/QUANTS.md)

**`STRATA_ROUTE_PRIOR`** — float lambda (default: 0 = off) — opt-in re-pick of the top-10 experts with a logit bonus for GPU-resident experts (weights stay the router's); it changes the output (adds speed, doubles the KL). Gate: MoE with 512 experts and top-10 routing, resident table present (CUDA route_prior kernel). Best for: A/B speed vs quality on cache-limited cards. Tags: [ab] (docs: docs/DETAILS.md)

**`STRATA_SELECT_OLD`** — any value (default: unset) — Runs QSA block scores on the warp kernel instead of the 3xTF32 tensor-core scorer. Gate: prompt-path QSA selection, all GPUs (tensor scorer needs sm_80+). Best for: leave unset on modern cards; set on older GPUs or for parity. Tags: [tunable]

**`STRATA_SELECT_SIMT`** — `1` (default: off) — on cards without TF32 tensor cores (compute capability < 8) uses the tiled FP32 SIMT QSA block scorer instead of the default warp scorer; its summation order differs, so output bytes change. Gate: NVIDIA pre-sm_80 (Pascal/Volta/Turing). Best for: A/B on older cards (does not enable TF32 on Turing). Tags: [ab] (docs: docs/OLDER_GPUS.md)

**`STRATA_SM60_TABLE`** — `0`/`1` (default: on) — on sm_60 (GP100 / Tesla P100) selects the interleaved matvec table (kIlRows_60) and expert mode 8, bitwise-equal to native_mmvq's kernels (P100: down 116 -> 94 us, IQ3_S gate/up 360 -> 229 us); `=0` keeps native_mmvq's kernels and the CUDA layout. Gate: NVIDIA sm_60 only. Best for: leave on; `=0` is the A/B arm. Tags: [tunable]

**`STRATA_SM61_TABLE`** — `0`/`1` (default: on) — on sm_61 (Tesla P40/P4, GTX 10 series, native dp4a) selects the measured interleaved table (kIlRows_61) and expert mode 8, bitwise-equal (P40: IQ3_S 390 -> 255 us in an 18-group call); `=0` keeps native_mmvq's kernels and the CUDA layout. Gate: NVIDIA sm_61 only. Best for: leave on; `=0` restores the 0.1.43.1 kernels for A/B. Tags: [tunable] (docs: docs/OLDER_GPUS.md)

**`STRATA_SM70_TABLE`** — `1` (default: off, opt-in) — enables the decode kernels measured on Volta: the interleaved 2-4 column matvec with a Volta rows table, expert mode 8, and the latency-hidden norm/up, all bitwise-equal to the defaults (V100: decode +9.8% and +15.0%, tokens bit-identical). Gate: NVIDIA sm_70 (V100) only. Best for: V100 owners; becomes default once confirmed on final code. Tags: [tunable] (docs: docs/NVIDIA_V100.md, docs/COMMUNITY_BENCHMARKS.md)

**`STRATA_SPEC_PROB_GATE`** — `top` (default: unset) — gates a further draft on q's top probability instead of the drawn token's probability (RTX 3060: drafts longer windows whose later guesses are kept less often, decoding slower). Gate: STRATA_SPEC_PROB path. Best for: A/B the drafting gate. Tags: [ab]

**`STRATA_TOPK_ACTIVE_ANY`** — `1` (default: off) — Test hook: forces the Turing active-count QSA top-k dispatch on any CUDA card so `qsa_topk_active_parity` can check it anywhere. Gate: NVIDIA CUDA QSA select. Best for: tests only. Tags: [debug]

**`STRATA_TOPK_CAPACITY_GUARD`** — any value (default: unset) — Restores 0.1.32's capacity rule for QSA top-k (skips the active-count prefill bound) on non-Turing CUDA devices. Gate: NVIDIA CUDA QSA select. Best for: leave unset; A/B arm. Tags: [ab]

### 7.2 AMD-specific (37)

Switches that only exist in, or are gated to, the AMD (HIP/ROCm) path: gfx-architecture
defaults, hipBLASLt tuning, WMMA on RDNA matrix cores, ROCm runtime behavior. Shipped
hipBLASLt tuning tables: `tools/hip/gfxNNNN-hipblaslt-<version>.txt` (see docs/AMD_HIP.md).

**`STRATA_ARENA_MMAP`** — `1` (default: off) — maps a native pack's expert arena read-only from its `experts.bin` via POSIX mmap instead of reading it into locked RAM, for PCs whose GPUs hold most experts but whose RAM is small (~1 GB -> 25 GB available while serving). Gate: Linux only (disabled on Windows); the AMD-HIP path additionally raises `GPU_PINNED_MIN_XFER_SIZE` to dodge a ROCclr page-locking leak. Best for: small-RAM machines with big GPU expert residency. Tags: [tunable] (docs: docs/DETAILS.md, docs/AMD_HIP.md)

**`STRATA_COPY_HIP_PRIO`** — non-zero (default: off) — creates the verify copy stream at high stream priority so ROCm gives it its own hardware queue, preventing `--pcie-mode dma` copies from queueing behind the window graph and deadlocking on its flag. Gate: AMD HIP only (excluded on gfx906). Best for: set `=1` on AMD if DMA copies time out waiting behind the window graph. Tags: [tunable]

**`STRATA_DENSE_MMQ`** — `1` (default: off) — runs the prompt's dense GGUF projections through the int8 MMQ kernels instead of dequantize + hipBLAS (RX 6900 XT: 4K prompt 436 -> 757 tok/s); needs K a multiple of 256, and greedy text diverges from the default path. Gate: AMD HIP only (`__HIPCC__` with STRATA_PREFILL_MMQ). Best for: prompt speed on RDNA2 cards with quantized dense weights. Tags: [tunable] (docs: docs/AMD_HIP.md, docs/DETAILS.md)

**`STRATA_FUSE_CONVAB`** — non-zero (default: off) — fuses the GDN conv and the a/b projections into one kernel launch. Gate: AMD HIP builds only; measured -0.3% on gfx1201, so it stays opt-in. Best for: leave at default; kernel A/B experiments. Tags: [ab]

**`STRATA_FUSE_HITS`** — `0`/`1` (default: on on gfx12, off on other AMD, never on CUDA) — folds the moe_hit_add step into the combine kernel, saving a graph node. Gate: AMD HIP; on by default on gfx12 (RDNA4) where a dependent kernel costs ~2.4 µs. Best for: leave at default; force on/off for A/B. Tags: [tunable]

**`STRATA_GFX12_CHUNK`** — `0`/`1` (default: on on gfx12 Linux) — lets a bare `--prefill auto` take 16384-token chunks instead of the 8192 cap (+0.5..+5.8% on R9700, identical residuals); `=1` keeps 16384 even when free RAM can't cover the extra loan. Gate: AMD gfx1200/gfx1201 (RDNA4), Linux, bare auto prefill only. Best for: `=0` to keep the 8192 cap on RAM-tight boxes. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_GFX906_ATTN_QUERY_SWIZZLE`** — `1` (default: off) — swizzles the decode-attention query layout on gfx906, community-measured on MI50 (PR #1661). Gate: AMD gfx906-only build (compiled in only there; needs kv_mode 1 and off with STRATA_ATTN_LANECELL=1). Best for: MI50 owners benchmarking `=1`. Tags: [tunable] (docs: docs/AMD_HIP.md)

**`STRATA_GFX906_ATTN_QUERY_SWIZZLE_TRACE`** — set (default: off) — prints one line naming the query-swizzle's active state, n_q, cap, kv_mode, and lane_cell on the first decode-attention call. Gate: AMD gfx906-only build. Best for: diagnostics when testing the swizzle. Tags: [debug]

**`STRATA_GFX906_ATTN_REDUCE12`** — `1` (default: off) — the reduce12 variant of the decode-attention chunk kernel on gfx906 (PR #1718), community-measured on MI50. Gate: AMD gfx906-only build; requires STRATA_GFX906_ATTN_QUERY_SWIZZLE=1 (INT8 pools, warp scoring, 64-lane wavefront). Best for: MI50 owners benchmarking the swizzle path. Tags: [tunable] (docs: docs/AMD_HIP.md)

**`STRATA_GFX906_ATTN_REDUCE12_TRACE`** — `1` (default: off) — prints one line confirming the reduce12 kernel was selected. Gate: AMD gfx906-only build, inside the query-swizzle path. Best for: diagnostics only. Tags: [debug]

**`STRATA_HCD_EXACT`** — `1` (default: off) — the hyper-connection down projection (N 320, K 10240) as a WMMA kernel reproducing hipBLASLt solution 1176/1177's summation order bit-for-bit, used on chunks of STRATA_PF_SWITCH_MIN_T+ tokens; runs only where the table makes hipBLASLt pick one of those two. Gate: AMD (hipBLASLt WMMA, Strix Halo tuning). Best for: prompt speed with exact bits on gfx1151. Tags: [ab] (docs: docs/STRIX_HALO.md)

**`STRATA_HIPBLASLT_TUNING`** — `<path to a calibration table>` (default: unset = hipBLASEx) — enables the hipBLASLt prefill GEMM path with a per-architecture solution table (the file's second line names the arch and ROCm version; a mismatch is refused). Gate: AMD HIP only. Best for: point it at the shipped `tools/hip/<arch>-hipblaslt-<version>.txt` for big prompt gains (1.5-22x per GEMM). Tags: [tunable] (docs: docs/AMD_HIP.md, docs/AMD_HIP_PERFORMANCE.md, docs/STRIX_HALO.md)

**`STRATA_HIPBLASLT_VERBOSE`** — presence (default: off) — prints the hipBLASLt summary (launches, fallbacks, unique fallback shapes) at teardown and a line per uncached shape at resolve time. Gate: AMD HIP, only with a tuning table. Best for: check `fallbacks=0` after changing a table; leave unset. Tags: [debug] (docs: docs/AMD_HIP.md)

**`STRATA_HIP_ADAPT_KERNEL_COPY`** — `1` (default: off) — copies the adaptive tier's host-to-device swaps with a kernel on the refill stream instead of the SDMA copy engine — the #884 workaround for the 2x RX 6900 XT hang with MMQ prompts + adaptive swaps; needs a mapped, 16-byte-aligned source. Gate: AMD HIP only. Best for: set when SDMA + adaptive swaps hang; keeps SDMA for the prompt read. Tags: [tunable] (docs: docs/AMD_HIP.md, docs/DETAILS.md)

**`STRATA_HIP_PROMPT_F16`** — `1` (default: off) — runs the prompt's 16-bit GEMMs FP16 in and out (rocBLAS is tuned for that on gfx103x: ~2x faster long prompts); not bit-identical — an FP16 rounding of each GEMM output. Gate: AMD HIP only; the engine prints a tip on gfx103x. Best for: RDNA2 prompt speed. Tags: [tunable] (docs: docs/AMD_HIP.md)

**`STRATA_HIP_ROUTER_FAST`** — presence (default: off on gfx906) — on gfx906 (wave64) runs the wave32-shaped S6 router top-10 kernel on request; on other HIP cards the S6 kernel is the default and this variable does nothing. Gate: AMD gfx906 only. Best for: A/B the router kernel on gfx906. Tags: [ab]

**`STRATA_HIP_ROUTER_OLD`** — presence (default: off) — runs the portable one-wavefront router top-10 kernel instead of the fast S6 kernel. Gate: AMD HIP (non-gfx906, where S6 is the default). Best for: fall back to the portable kernel to A/B. Tags: [ab] (docs: docs/AMD_HIP.md)

**`STRATA_HIP_WMMA`** — `1` (default: off) — the prompt path's QSA attention on RDNA4/RDNA3 matrix cores (v_wmma_f32_16x16x16_f16, FP16 hi+lo halves): ~30% faster prompts on gfx12 with `--kv int8` (R9700 4K 1,784 -> 2,427 tok/s), and it selects the 96-slot prefill ring; the output's last bits change, so the engine only hints at it. Gate: AMD gfx11/gfx12 only. Best for: prompt speed on RDNA4 with int8 KV. Tags: [tunable] (docs: docs/AMD_HIP.md, docs/STRIX_HALO.md)

**`STRATA_MTP_HIP_PRIO`** — `0`/`1` (default: on only on gfx12) — creates the MTP drafter's stream at the highest priority so ROCm gives it its own hardware queue and the draft round stops waiting behind the verifier's commit graph (R9700: draft wall 5.8 -> 3.5 ms per window). Gate: AMD gfx12 only by default. Best for: leave at default; force on/off to A/B the priority-queue trick on other cards. Tags: [ab]

**`STRATA_MTP_SIDE_STREAM`** — `0`/`1` (default: on only on gfx103x, off on other HIP cards) — creates a non-blocking side stream for the MTP drafter under HIP; gfx103 is the exception that creates it (RX 6800 under Windows gained prompt speed), other HIP cards skip it since an R9700 lost prompt speed with it. Gate: AMD HIP only, gfx103x by default. Best for: `=0` to turn it off on gfx103x, `=1` to try it on another HIP card. Tags: [ab] (docs: docs/AMD_HIP.md)

**`STRATA_PCIE_AUTO_SPLIT`** — `0`/`1` (default: off, opt-in) — PcieAuto also refines a two-stage layer split's shares: the pool speed is read once and each stage takes the share its own probed PCIe link gives. Gate: AMD HIP only, two-stage splits (three or more stages skip it). Best for: try on a two-RDNA4-card split (2x R9700: code +8-10%). Tags: [ab] (docs: docs/DETAILS.md)

**`STRATA_PF_BK64`** — `0`/`1` (default: on; `=0` forces the BK 32 kernel) — selects the BK-64 WMMA prefill GEMM kernel for wide shapes (N >= 1024, K % 64 == 0) on gfx11; the A/B arm runs the BK-32 kernel with the same bits. Gate: AMD gfx11 only (`strata_wmma_gfx11_device`). Best for: A/B kernel selection on gfx11 prompt GEMMs. Tags: [ab]

**`STRATA_PF_FUSED_KQ`** — `0`/`1` (default: 0) — On top of `STRATA_PF_FUSED=1`, extends the fused prompt-expert kernels to K-quant formats (Q4_K/Q5_K gate/up, gfx11 kernels only); output differs from the FP16 path at rounding level. Gate: AMD HIP gfx11 (the K-quant coverage is `__HIPCC__`-only). Best for: leave off; try on Strix with a UD-Q4_K_XL pack. Tags: [tunable] (docs: docs/STRIX_HALO.md, docs/QUANTS.md, docs/AMD_HIP.md)

**`STRATA_PF_GEMM`** — `0`/`1` (default: 0) — Runs the prompt's FP16 projections on the 128×256 WMMA GEMM kernel (once chunks reach `STRATA_PF_SWITCH_MIN_T` tokens) instead of hipBLASLt. Gate: AMD HIP only (the call site is inside `#ifdef STRATA_USE_HIP`). Best for: set `=1` on Strix/RDNA per the STRIX_HALO recipe. Tags: [tunable] (docs: docs/STRIX_HALO.md, docs/AMD_HIP.md)

**`STRATA_PF_OCC`** — integer N (default: 2× the queried occupancy) — Sets blocks per WGP for the fused prompt-expert persistent grid on AMD, where the occupancy query undercounts (a WGP is two CUs); results are unchanged, only grid size. Gate: AMD HIP (gfx1151/gfx12 WGP code path). Best for: leave at default; experiment knob (gfx1151 measured 2 as best). Tags: [tunable]

**`STRATA_PF_OCC_W`** — integer N > 0 (default: unset = per-variant table) — overrides the blocks-per-multiprocessor occupancy cap for every fused prefill occupancy variant; the built-in table gives 3 to gate/up IQ2_S and IQ2_XS on gfx12 (R9700: 4K 8.8 -> 8.2 ms) and 4 elsewhere. Gate: AMD (gfx12 measured, gfx11 keeps 4). Best for: experiment knob for occupancy tuning. Tags: [ab]

**`STRATA_PF_PAD`** — `0`/`1` (default: 0) — With `STRATA_PF_GEMM=1`, dequantizes weights and pads FP16 activations with row stride K+64 halves so GEMM rows stop camping on memory channels (19→29 TFLOPS measured); bitwise same products. Gate: AMD HIP only (`#ifdef STRATA_USE_HIP`), chunks ≥ `STRATA_PF_SWITCH_MIN_T`. Best for: set `=1` with `STRATA_PF_GEMM=1` on Strix. Tags: [tunable] (docs: docs/STRIX_HALO.md)

**`STRATA_PREFILL_AUTO_MAX`** — integer (atoll; 8192/16384/32768, default: 8192) — Lets a bare `--prefill auto` take chunks above 8192 tokens (values snap to 8192/16384/32768). Gate: all GPUs, no gate (Windows HIP keeps a 6144 cap for bare auto). Best for: raise to 16384/32768 on roomy cards, same as `--prefill auto:N`. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_PREFILL_MMQ`** — integer (atoi; default: on when built) — Runtime switch for the MMQ prompt-expert path (needs the MMQ build, e.g. `-DSTRATA_PREFILL_MMQ=ON`); `=0` falls back to the FP16 path. Gate: all GPUs; the AMD builds document it as required for HIP prompt experts. Best for: leave at default when built. Tags: [tunable] (docs: docs/AMD_HIP.md, docs/STRIX_HALO.md, docs/AMD_HIP_PERFORMANCE.md)

**`STRATA_PREFILL_RING`** — integer slots (default: auto from ring bytes; 96 with HIP WMMA attention on streaming chunks) — Sets the prompt path's streamed-expert ring slot count. Gate: all GPUs; the 96-slot default with `STRATA_HIP_WMMA=1` is AMD RDNA4-specific. Best for: set `=96` on RDNA4 WMMA setups, `=384` on pinned-RAM DMA rigs. Tags: [tunable] (docs: docs/AMD_HIP.md, docs/AMD_HIP_PERFORMANCE.md, docs/INTEL.md, docs/MULTI_GPU.md)

**`STRATA_PREFILL_STREAM_MIN`** — integer tokens (atoll; default: 1024) — Chunk size from which every non-resident expert streams (and fused prompt experts apply); below it only routed experts move. Gate: all GPUs, streaming prompt path. Best for: set `=128` on Strix per the STRIX_HALO recipe. Tags: [tunable] (docs: docs/STRIX_HALO.md)

**`STRATA_QFUSE`** — integer (atoi; default: off, auto-set to 1 by the gfx1151 arch defaults) — Activation q8_1 images are written directly by their producers (e.g. the GDN output norm) instead of a separate quantize pass; same bytes. Gate: verify path, all GPUs; enabled by default only on AMD gfx1151. Best for: leave to the arch defaults. Tags: [tunable] (docs: docs/STRIX_HALO.md)

**`STRATA_ROUTER_BLOCK`** — any value (default: unset) — On gfx906, forces the router top-10 kernel away from the one-wavefront register-argmax kernel (n_expert ≤ 64×RW_PER) to the block kernel. Gate: AMD gfx906 only. Best for: leave unset; A/B arm. Tags: [ab] (docs: docs/AMD_HIP.md)

**`STRATA_SELECT_WMMA`** — any non-`0` value (default: off) — Opts into the gfx12 (RDNA4) WMMA QSA block scorer; it selects slightly differently from the warp kernel (254/256 queries the same). Gate: AMD gfx12 only (`sel_gfx12_device()`); every other target stays on the warp kernel. Best for: set `=1` on R9700-class cards per the Strix/RDNA4 recipes. Tags: [tunable] (docs: docs/STRIX_HALO.md, docs/AMD_HIP.md)

**`STRATA_SPLIT_CHUNK16K`** — `1` (default: off) — keeps the gfx12 16384 prefill-chunk ceiling on a layer split; by default splits cap chunks at 8192 so the prompt runs as a pipeline across cards (2x R9700: pp32k +14%, 4x +19%). Gate: AMD gfx12 multi-GPU layer split. Best for: leave off; `=1` restores the single-card chunk size. Tags: [tunable] (docs: docs/MULTI_GPU.md)

**`STRATA_TAIL_SKIP_MAX_CACHED`** — percent (default: 90; 60 on a HIP layer split; `101` = never) — moves the cached-expert line below which the route tail skip gets no default (with nearly every expert cached, the few misses are cheap and the skip costs draft acceptance). Gate: same paths as STRATA_ROUTE_TAIL_SKIP. Best for: tune when the skip's auto-off line fires at the wrong point. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_WMMA_BF16`** — `0` to exclude bf16 (default: on) — With `STRATA_WMMA_GEMM=1`, allows the RDNA3/3.5 WMMA dense GEMM for BF16 shapes; `=0` keeps those on hipBLASLt. Gate: AMD HIP gfx11 WMMA path (`#ifdef STRATA_USE_HIP`). Best for: leave at default; A/B arm. Tags: [ab]

### 7.3 Intel-specific (68)

Switches in the Intel SYCL port (`sycl/`): Arc Alchemist/Battlemage, Xe Matrix Engines (XMX),
Level Zero queues, doorbell synchronization. See docs/INTEL.md, docs/INTEL_ARC.md,
docs/A770_SPEED_SPEC.md.

**`STRATA_ARENA_ALIAS_CHECK`** — `0`/unset (default: on) — Enables a guard that checks the SYCL device-memory arena for aliasing (overlapping allocations); `=0` skips the check. Gate: all SYCL (arena allocator in `sycl_queue.hpp`). Best for: leave at default; set `=0` only when reproducing the aliasing bug needs the guard off. Tags: [debug] (docs: docs/INTEL.md)

**`STRATA_CHECK_SWAPS`** — any non-`0` value (default: off) — Test aid: after each expert eviction verifies the exchange buffer matches the model file's bytes, and after the run that every expert is still held (GPU cache or RAM) with no stale blobs. Gate: SYCL expert swap/eviction path in `generate.cpp`. Best for: leave at default; use when testing the VRAM/RAM expert tiering for data loss. Tags: [debug]

**`STRATA_CHUNKED_FILL`** — `0`/unset (default: on) — Performs large device zero-fills as compute kernels in chunks of at most 256 MiB; on the Arc Pro B70 (xe driver) a single multi-GiB `memset` runs on the blitter engine and times out with a CAT error/GT reset, kernels do not. Gate: all SYCL large fills (`big_fill_zero`). Best for: leave at default; `=0` reverts to plain `q.memset`. Tags: [tunable]

**`STRATA_COPY_ENGINE`** — `0`/`1`/unset (default: on for Battlemage cards only) — Puts prompt-staging memcpys on a dedicated Level Zero copy-engine queue; forced on other GPUs it hung the A750 prompt path and gained only 1.4% there, so Alchemist stays off. Gate: SYCL prompt staging copy queue (needs the Level Zero v2 adapter). Best for: leave at default; `=1` to force on a non-Battlemage card, `=0` to disable. Tags: [tunable] (docs: docs/INTEL.md)

**`STRATA_DBG_DRAFT`** — any value (default: off) — Prints each speculative-decoding window the drafter built, what the target model said, and how many tokens were accepted, to stderr. Gate: SYCL speculative decoding loop in `generate.cpp`. Best for: leave at default; set when debugging draft/accept behavior. Tags: [debug]

**`STRATA_DOORBELL_CHECK`** — `1`/unset (default: off) — Opt-in full host-side checksum of the whole doorbell payload (re-read from uncached host memory until ready) before CPU-expert layers consume it; cost 15-20% of decode on an Arc A750, guarding against a stale payload not seen there. Gate: Level Zero doorbell sync (`doorbell_publish_res`, `doorbell_wait_payload`). Best for: leave at default; `=1` when chasing doorbell payload staleness. Tags: [debug]

**`STRATA_DOORBELL_PLAIN`** — `0`/`1`/unset (default: plain store when the driver is i915) — Chooses plain stores vs. the mapped `sys_store` when publishing doorbell payload/sequence values; the launchers read it host-side and pass it into the kernel. Gate: Level Zero doorbell publish path. Best for: leave at default; override only when a driver's coherent mapped-memory stores misbehave. Tags: [tunable]

**`STRATA_DUMP_SEL`** — a file path (default: off) — Dumps the selected KV cells of every prompt position of the first QSA layer of the last chunk (header, ids, widths) to a binary file, the input to the grouped-gather study. Gate: SYCL QSA prompt attention in `prefill.cpp`. Best for: leave at default; set when studying QSA selection overlap. Tags: [debug] (docs: docs/INTEL.md)

**`STRATA_FILL_SERIAL`** — any value (default: off) — Uses the old serial expert-fill form (read, copy, wait per expert) instead of the default pipelined fill (worker-thread file reads overlapping batch copies), which took 120 s vs 41 s. Gate: SYCL GGUF `--stream-experts` VRAM-cache fill. Best for: leave at default; `=1` only to reproduce the old fill. Tags: [tunable] (docs: docs/INTEL.md)

**`STRATA_GDN_XE`** — `0`/`1`/unset (default: on for Alchemist Intel GPUs with 16-lane sub-groups) — Selects the Xe-optimized GDN recurrence kernel over the column kernels; measured 478-480 → 511-518 tok/s at 32K on Alchemist with identical output. Gate: SYCL prefill GDN recurrence (`gdn_recurrence`). Best for: leave at default; `=1` to try the kernel on another Intel GPU (verify with `gdn_parity --selftest`), `=0` to force the column kernels. Tags: [tunable] (docs: docs/INTEL.md)

**`STRATA_GGUF_PLAN_READ`** — `1`/unset (default: off) — Switches the stream-all prompt expert streaming to plan-ordered reads instead of the GGUF source's own order. Gate: SYCL stream-all prompt path in `prefill.cpp`. Best for: leave at default; set when A/B-testing expert read ordering. Tags: [ab]

**`STRATA_GR_DOWN_DIRECT`** — `0`/unset (default: on, direct kernel) — Chooses the direct multi-token GR down kernel; `=0` falls back to the tiled kernel, which measured 13-19% slower at 3-6 tokens. Gate: SYCL fused GR down step (`fused_gr.dp.cpp`). Best for: leave at default; `=0` to restore the old tiled path. Tags: [tunable] (docs: docs/INTEL.md)

**`STRATA_GR_DOWN_SLICED`** — `0`/unset (default: on) — Uses the sliced GR down kernel (GR read 108.6 → 76.5 µs at 6 tokens); `=0` restores the direct kernel, whose output is bitwise original when a near-tie flips. Gate: SYCL fused GR down step. Best for: leave at default; `=0` when bitwise parity with the old kernel matters. Tags: [tunable] (docs: docs/INTEL.md)

**`STRATA_GR_LARGE_GRF`** — `0`/`1`/unset (default: on for Alchemist only) — Runs the sliced down and multi-token up kernels in large-GRF mode (256 registers per thread); on the A770 the up kernel spills ~3.4 KB/thread at 128, and large GRF halves resident threads per Xe core, so it is only defaulted where measured. Gate: SYCL fused GR kernels on Intel GPUs. Best for: leave at default; force `=1`/`=0` when tuning register pressure on a new card. Tags: [tunable]

**`STRATA_GR_MORE_GROUPS`** — `0`/`1`/unset (default: on for Alchemist only) — Uses more, smaller work-groups for the multi-token GR read (480 groups for the sliced down, 320 for the 8-column up); A770 gr_bench 117-180 → 87-149 µs, decode ~28.3 → ~30.0 tok/s, sums unchanged. Gate: SYCL fused GR multi-token kernels. Best for: leave at default; force on/off when benchmarking group shapes on other Intel GPUs. Tags: [tunable]

**`STRATA_GR_NORM_SPLIT`** — `0`/unset (default: on) — Splits the GR norm step across multiple work-groups per token; `=0` runs one work-group per token. Gate: SYCL fused GR norm step. Best for: leave at default; `=0` to A/B the single-group form. Tags: [tunable]

**`STRATA_GR_UP_COLS`** — `4`/`8`/`16`/`32`/`64` (default: 16) — Sets columns per block in the multi-token GR up kernel: 8/4 run 320/640 blocks with 4/2 rows per warp, attacking the latency-bound 16-column form (B70: 23.7 µs for 6.55 MB, 46% of bandwidth); all sums identical. Gate: SYCL fused GR up kernel. Best for: leave at default; try 8 on Xe2 when tuning. Tags: [tunable]

**`STRATA_GR_UP_ILP`** — `1`/unset (default: off) — Enables instruction-level-parallelism staging in the multi-token GR up kernel. Gate: SYCL fused GR up kernel. Best for: leave at default; `=1` when experimenting with up-kernel scheduling. Tags: [ab]

**`STRATA_HC_CHECK`** — any value (default: off) — Opts into upstream's hyper-connection read self-test/variants on the SYCL port, where they self-crash on the B70; unset runs the plain hyper-connection read. Gate: SYCL fused_gr hyper-connection variant selection. Best for: leave at default; `=1` only when testing upstream HC variants. Tags: [debug] (docs: docs/INTEL.md)

**`STRATA_HOST_UNCACHED`** — `0`/unset (default: on) — Allocates polled host buffers (`host_malloc_polled`) with Level Zero's `BIAS_UNCACHED` flag so CPU/GPU doorbell polling sees fresh data; `=0` falls back to normal cached host memory. Gate: all SYCL polled host allocations on the Level Zero backend. Best for: leave at default; `=0` only to test cached-memory coherence behavior. Tags: [tunable]

**`STRATA_KQ_GU_LANES`** — lane count (default: follows `STRATA_GU_LANES`) — Overrides the lane split for K-quant/Q8_0 gate-up rows only (Unsloth files); on the B70 16 lanes read Q4_K/Q5_1 at 439 GB/s vs 426 at 8, but the reduction order changes bits, so it stays opt-in. Gate: SYCL native expert gate/up kernels (`iq_kernels.dp.cpp`). Best for: leave at default; set when tuning lane width for K-quant experts. Tags: [tunable]

**`STRATA_KQ_SPLIT`** — `0`/unset (default: on) — Enables the multi-entry split in the K-quant/Q5_1/Q8_0 lane kernels (one weight read serves multiple entries); `=0` runs one weight read per entry, same output bits either way. Gate: SYCL native expert gate/up kernels. Best for: leave at default; `=0` to A/B the unsplit kernel. Tags: [ab]

**`STRATA_KV_WIDE_HOST`** — `0`/unset (default: on) — Uses the wide host-pool path in `kv_append` when writing K/V into host/staging pools. Gate: SYCL KV append in prefill. Best for: leave at default; `=0` to fall back to the narrow path. Tags: [tunable] (inferred from code)

**`STRATA_LEND_MIRROR`** — `0` / slot count (default: last 10% of cache slots, max 1,200) — Mirrors the experts in the tail of the VRAM cache into the pinned host mirror so prompt-borrowed ("lent") experts stream by DMA instead of ~2.3 GB/s preads (4K prompt refill 7.2 → 3.3 s). Gate: SYCL prompt prefill-borrow path in `generate.cpp`. Best for: leave at default; `=0` to disable, `=N` to size the mirrored tail. Tags: [tunable] (docs: docs/INTEL.md)

**`STRATA_MIRROR_MIB`** — MiB (default: free RAM minus 4 GiB) — Caps the pinned host mirror that holds every expert without a VRAM slot at startup; a single `malloc_host` above a few GB fails on the A750 (3,000 worked, 8,192+ did not), and multi-GPU setups should set `=0` since the mirror goes unused. Gate: SYCL GGUF expert mirroring in `generate.cpp`. Best for: lower it on cards with small pinned-allocation limits; `=0` to disable mirroring. Tags: [tunable] (docs: docs/INTEL.md, docs/INTEL_ARC.md)

**`STRATA_MMVQ_A2`** — `0`/unset (default: on) — Enables the aligned-load path in the wide Q6_K MMVQ kernel (coder decode 44.9 → 54 tok/s, identical output); `=0` restores the old misaligned path. Gate: SYCL native Q6_K expert decode kernel. Best for: leave at default; `=0` to A/B the old loads. Tags: [tunable] (docs: docs/INTEL.md)

**`STRATA_MMVQ_PF`** — N (default: 0, off) — Issues N weight-load steps ahead (software prefetch) in the wide Q6_K kernel. Gate: SYCL native Q6_K expert decode kernel. Best for: leave at default; experiment when tuning Q6_K bandwidth. Tags: [ab]

**`STRATA_MMVQ_RPW`** — `0`/`1`/`2`/`4` (default: 0) — Bench knob for the row-warp MMVQ kernel: rows per warp (1/2/4) or `0` for the shared kernel, which measured equal at warm clocks. Gate: SYCL native MMVQ decode launch. Best for: leave at default; use when benchmarking row-warp shapes. Tags: [ab]

**`STRATA_MMVQ_SG`** — `16`/`1`/`2`/`3`/`32` (default: 32) — Chooses Xe2-native SIMD16 sub-groups for the decode MMVQ kernels: 16 = all, 1 = IQ4_XS only, 2 = short outputs (n_out ≤ 1024) only, 3 = both, 32 = CUDA-shaped 32-lane warps; SIMD16 wins up to 1.45x on IQ4_XS in isolation but loses 10-25% on large K-quant projections, so the default keeps warps. Gate: SYCL native MMVQ decode kernels. Best for: leave at default; experiment on Xe2. Tags: [ab] (docs: docs/INTEL.md)

**`STRATA_MMVQ_WIDE`** — `0`/unset (default: on) — Routes Q6_K decode through the wide 16-byte-load kernel (2.6-3.4x the misaligned kernels, decode 44.9 → 54 tok/s, identical tokens); `=0` uses the old shared kernel. Gate: SYCL native Q6_K MMVQ dispatch. Best for: leave at default; `=0` to A/B the old kernel. Tags: [tunable] (docs: docs/INTEL.md)

**`STRATA_MMVQ_WIDE_32`** — `0`/unset (default: on) — Routes Q8_0/IQ4_NL decode through the wide32 kernel; `=0` restores the old small/multi kernels. Gate: SYCL native MMVQ dispatch for 32-byte-block formats. Best for: leave at default; `=0` to restore the old Q8_0/IQ4_NL kernels. Tags: [tunable] (docs: docs/INTEL.md)

**`STRATA_MMVQ_WIDE_K`** — `0`/unset (default: on) — Routes Q4_K/Q5_K/IQ4_XS decode through the wide kernel; `=0` sends them through the multi kernel (the old path). Gate: SYCL native MMVQ dispatch for K-quant/IQ4_XS. Best for: leave at default; `=0` to restore the old path. Tags: [tunable] (docs: docs/INTEL.md)

**`STRATA_MMVQ_WIDE_K_RPW`** — `1`/`2`/`4` (default: 1) — Rows per sub-group in the Q4_K/Q5_K/IQ4_XS wide kernel. Gate: SYCL native wide MMVQ kernel. Best for: leave at default; tune when benchmarking wide K-quant kernels. Tags: [ab]

**`STRATA_MMVQ_WIDE_MAXC`** — N (default: 8 with aligned loads, 4 without) — Maximum column count (window width) for which the wide Q6_K kernel is chosen. Gate: SYCL native Q6_K MMVQ dispatch gate. Best for: leave at default; adjust with `STRATA_MMVQ_A2` when re-benchmarking the wide kernel's crossover. Tags: [ab]

**`STRATA_MMVQ_WIDE_MIN`** — N (default: 0 with aligned loads, 4096 without) — Minimum output width (n_out) for which the wide Q6_K kernel is chosen. Gate: SYCL native Q6_K MMVQ dispatch gate. Best for: leave at default; adjust when re-benchmarking the wide kernel's crossover. Tags: [ab]

**`STRATA_MMVQ_WIDE_RPW`** — `1`/`2`/`4` (default: 1) — Rows per sub-group in the wide Q6_K kernel. Gate: SYCL native wide Q6_K kernel launch. Best for: leave at default; tune when benchmarking the Q6_K wide kernel. Tags: [ab]

**`STRATA_PF_DQ_FUSED`** — `0`/unset (default: on) — Dequantizes a native pack layer's gate/up and down matrices in one fused launch (same values); `=0` runs the two separate dequant launches. Gate: SYCL prefill native-pack dequant. Best for: leave at default; `=0` to A/B the split launches. Tags: [ab]

**`STRATA_PF_XMX`** — `0`/`1`/`2` (default: 0, off) — Enables the XMX joint-matrix grouped expert GEMM in prefill (sub-group 8 kernels on DG2/Alchemist, 8x16x16 sub-group 16 in an Xe2 build); mode 2 holds a group of staged experts for lookahead. Built for Battlemage it was correct to 6e-7 but 11-15% slower, so it stays off. Gate: A770 XMX prefill MoE GEMM. Best for: leave at default; `=1`/`=2` to experiment on Arc A-series. Tags: [ab] (docs: docs/INTEL.md)

**`STRATA_PF_XMX_GU_MAX`** — N (default: 250) — Cap on mean rows per expert for using the fused XMX gate/up grouped GEMM. Gate: XMX prefill MoE dispatch. Best for: leave at default; tune with `STRATA_PF_XMX_ROWS` when re-benchmarking. Tags: [ab]

**`STRATA_PF_XMX_ROWS`** — N (default: 200) — Rows-per-expert crossover where the fused XMX grouped launch beats dequantize + oneMKL (measured: fused wins below ~200 rows/expert, gate/up 0.67-0.76x at 320). Gate: XMX prefill MoE GEMM path selection. Best for: leave at default; move when re-measuring on a new card. Tags: [tunable]

**`STRATA_PLE_TRACE`** — any value (default: off) — Emits a timing trace of the PLE (prompt row) reader: job issue, first submit, last completion, exposing read stalls such as the 226 ms first-decode-round stall. Gate: SYCL PLE reader (`ple_reader.cpp`). Best for: leave at default; set when diagnosing prompt-read stalls. Tags: [debug] (docs: docs/INTEL.md)

**`STRATA_PREFILL_FIRST`** — token count, `0` = off (default: 256 only when the VRAM cache holds every expert, else 0) — Runs a short first prompt chunk so the GPU starts while the rest of the rows are read; with experts streaming over PCIe a short chunk costs a whole extra stream pass, hence the conditional default. Gate: SYCL prefill chunking. Best for: leave at default; set explicitly when tuning prompt chunking (the boundary can shift GEMM rounding). Tags: [tunable] (docs: docs/INTEL.md)

**`STRATA_PREFILL_STREAM_ALL`** — `0`/`1`/unset (default: on only when the VRAM cache misses under 10% of expert pairs) — Makes the prompt walk copy every non-resident expert of a layer (routed or not); with a big host tier that is several times the routed traffic and a 2,184-token prompt fell ~570 → 254 tok/s, so it is gated. Gate: SYCL prefill staging walk. Best for: leave at default; force on/off for debugging. Tags: [tunable] (docs: docs/INTEL.md)

**`STRATA_PREFILL_SYNC`** — any value (default: off) — Debug switch: waits for the GPU at every prefill phase mark and logs the phase that finished, so a hang names the stage that never completes. Gate: SYCL prefill phase profiler. Best for: leave at default; set when a prefill hangs. Tags: [debug] (docs: docs/INTEL.md)

**`STRATA_PROFILE_PRIOR_ENTRIES`** — float (default: 800000) — Sizes the exponential (λ=2) prior mass added to the shipped expert-profile ranking, in routed-entry equivalents; 60k lost 36% on off-distribution text, 800k kept +13-19% on the trained kind. Gate: SYCL expert profile ranking build in `generate.cpp`. Best for: leave at default; tune when adapting the profile to a new workload. Tags: [tunable]

**`STRATA_PROMPT_ATTN_XMX`** — `1`/`32` (default: 0, off) — Opt-in XMX joint-matrix QSA prompt attention: `=1` uses 64-cell chunks (120 KB local memory), `=32` 32-cell chunks; in the full matrix it measured 23-33% slower than the FP32 fallback on long prompts. Gate: A770 XMX QSA prompt attention. Best for: leave at default; experiment only on Arc A-series. Tags: [ab] (docs: docs/INTEL.md)

**`STRATA_REFILL_SERIAL`** — any value (default: unset) — Refills each stage's lent slots stage after stage (0.1.30/0.1.31 behavior) instead of issuing every stage's copies first on its own link. Gate: multi-stage layer split, all GPUs. Best for: leave unset; A/B arm. Tags: [ab]

**`STRATA_RING_TIMEOUT_S`** — seconds ≥ 1 (default: 20) — How long the host waits for a layer's doorbell before giving the verify window up; cards that JIT-compile kernels (e.g. an A750 needing FP64 emulation) can take longer on the first window. Gate: SYCL verify loop doorbell wait (`verify.cpp`). Best for: leave at default; raise on JIT-heavy first windows. Tags: [tunable]

**`STRATA_SPIN_MAX`** — positive count (default: driver-dependent, 20,000 on xe, 2,000,000 elsewhere) — Bounds every device doorbell spin; an unbounded spin makes the xe driver time the queue out and reset the GT node by node, wedging the card until reboot. Gate: all SYCL doorbell device spins. Best for: leave at default; tune only when a driver's flag latency needs a longer bound. Tags: [tunable]

**`STRATA_SPLIT_HEADROOM`** — `0`/unset (default: on) — Among multi-GPU layer placements within the headroom tolerance of the best predicted time, picks the one whose tightest card keeps the most VRAM beyond its experts (a full xe card silently moves buffers to system memory). Gate: multi-GPU layer-split placement in `generate.cpp`. Best for: leave at default; `=0` keeps the old first-best pick. Tags: [tunable] (docs: docs/INTEL_ARC.md)

**`STRATA_SPLIT_HEADROOM_TOL`** — fraction (default: 0.002 = 0.2%) — Width of the predicted-time band around the best multi-GPU placement within which headroom decides. Gate: multi-GPU layer-split placement. Best for: leave at default; widen when a near-tie picks a bad split. Tags: [tunable] (docs: docs/INTEL_ARC.md)

**`STRATA_SYCL_A770_FAST`** — `1`/unset (default: off) — Experimental opt-in (PR #1710) for the A770 router and 64-bit paired doorbell payload transfers (two floats per PCIe transaction, flags stay 32-bit). Gate: A770 doorbell payload path. Best for: leave at default; `=1` only to test the fast path. Tags: [ab]

**`STRATA_SYCL_GEMM_NO_STAGE`** — any value (default: off) — Negative control: never stage GEMM operands that sit past 4 GiB in the arena (an A770 oneMKL bug); staging is already off by default on Battlemage, where the bug did not show and the copy cost ~10% of a prompt. Gate: SYCL prefill oneMKL GEMM operand staging. Best for: leave at default; set as a test control. Tags: [debug] (docs: docs/INTEL.md)

**`STRATA_SYCL_GEMM_STAGE`** — `1`/unset (default: off on Battlemage, on elsewhere for far operands) — Re-enables staging of GEMM operands past 4 GiB on Battlemage, working around the A770 oneMKL bug where it applies. Gate: SYCL prefill oneMKL GEMM operand staging. Best for: leave at default; `=1` only if a Battlemage card shows the far-operand bug. Tags: [tunable] (docs: docs/INTEL.md)

**`STRATA_USAGE_DUMP`** — a file path (default: off) — Device-side routing counts every routed expert during the run and writes the usage table to the path, the measurement used to build expert profiles (+5% to +64% decode from profiles). Gate: SYCL resident-plan mirror path in `generate.cpp`. Best for: leave at default; set when collecting expert-usage data. Tags: [debug] (docs: docs/INTEL.md)

**`STRATA_VERIFY_ALL_RESIDENT`** — integer (atoi; default: on) — `=0` disables the all-resident fast path (zero-doorbell graph when every routed expert sits in VRAM). Gate: verify path with hit tables, all GPUs. Best for: leave at default; A/B arm. Tags: [ab]

**`STRATA_VERIFY_ALL_SLOTS`** — any value (default: off) — Debug: reads every filled VRAM expert slot back after the profile fill and compares it byte-for-byte with the GGUF, reporting the first differing byte; this is how the arena aliasing bug was found. Gate: SYCL expert-cache fill in `generate.cpp`. Best for: leave at default; set when chasing corrupted expert blobs. Tags: [debug] (docs: docs/INTEL.md)

**`STRATA_VERIFY_DEBUG`** — any value (default: unset) — Prints verify-window timing (staging/GPU ms) and per-layer residual ladders (the SYCL port allocates the R/M debug buffers with it). Gate: verify path, all GPUs. Best for: set to trace decode divergence. Tags: [debug] (docs: docs/INTEL.md)

**`STRATA_VERIFY_DUMP`** — a directory (default: off) — With `STRATA_VERIFY_ALL_SLOTS`, writes the first bad slot's `.got` and `.want` blobs into the directory for offline diffing. Gate: SYCL slot-verification debug path. Best for: leave at default; set alongside `STRATA_VERIFY_ALL_SLOTS` when investigating corruption. Tags: [debug] (docs: docs/INTEL.md)

**`STRATA_VERIFY_EAGER`** — any value (default: off) — Runs the verify window eagerly on the queue instead of as a captured graph, with host-clocked per-stage stamps so the profiler reports ms per stage; inflates absolute times and fails on some builds (payload-never-arrived errors). Gate: SYCL verify window execution. Best for: leave absent from acceptance runs; set only for stage profiling. Tags: [debug] (docs: docs/A770_SPEED_SPEC.md, docs/INTEL.md)

**`STRATA_VERIFY_FIND`** — any value (default: off) — With `STRATA_VERIFY_ALL_SLOTS`, searches the GGUF for whose expert blob the wrong bytes actually are, naming the aliased (layer, expert, slot). Gate: SYCL slot-verification debug path. Best for: leave at default; set when a slot mismatch is found. Tags: [debug] (docs: docs/INTEL.md)

**`STRATA_VERIFY_LEAN`** — `0`/unset (default: on) — Prunes the ~9 dead graph nodes per layer-window (doorbell publish, flag waits, plan copy, PCIe-share staging) that nobody answers under `STRATA_VERIFY_NO_HOST`, worth ~1.4 ms and -2% decode; only applies with a device plan and no host/step hooks. Gate: SYCL verify graph construction. Best for: leave at default; `=0` to A/B the full graph. Tags: [tunable] (docs: docs/INTEL.md)

**`STRATA_VERIFY_NO_HOST`** — non-`0`/`0` (default: off; the `strata-sycl.sh` wrapper sets 1) — The GPU takes its own plan for an all-resident cache and the host waits for the whole window instead of per-layer doorbell rings; required on `xe` cards when experts are not all in VRAM (host flag stores are invisible across the bus → NaN logits), and the engine refuses to start if any expert is in neither VRAM nor the mirror. Gate: SYCL verify window host handshake. Best for: follow the wrapper; set `=0` on i915 Arc A-series with CPU experts. Tags: [tunable] (docs: docs/INTEL.md, docs/INTEL_ARC.md, docs/A770_SPEED_SPEC.md, docs/DETAILS.md)

**`STRATA_VERIFY_PIPE`** — `0`/unset (default: on) — In stepped mode, runs the plan-copy + VRAM segment of the window in a helper thread (optionally pinned to a CPU) pipelined with the host's CPU-expert serving. Gate: SYCL stepped verify window segments. Best for: leave at default; `=0` to serialize segments. Tags: [tunable]

**`STRATA_VERIFY_PROFILE`** — any value (default: unset) — Allocates per-layer GPU-side stage stamps for the verify window (the GPU half of decode timing); known to break requests on the A770 SYCL build. Gate: verify path, all GPUs. Best for: set with `STRATA_DECODE_TIMING=1` for decode profiling; avoid on the Intel port. Tags: [debug] (docs: docs/DETAILS.md, docs/QUANTS.md, docs/OLDER_GPUS.md, docs/A770_SPEED_SPEC.md)

**`STRATA_VERIFY_SPLIT`** — `0`/unset (default: on) — In stepped capture, adds the second step-hook boundary (plan-copy/VRAM segment split) so the window is captured as pipelinable segments; `=0` captures without that split. Gate: SYCL stepped verify window capture. Best for: leave at default; `=0` to A/B segment layout. Tags: [ab]

**`STRATA_WARM_GRAPHS`** — `0`/unset (default: on) — Captures every window graph and the drafter's graphs once at load instead of on first use, so the first request no longer pays 250-290 ms of captures. Gate: SYCL graph capture at startup in `generate.cpp`. Best for: leave at default; `=0` to restore lazy capture. Tags: [tunable] (docs: docs/INTEL.md)

**`STRATA_WINDOW_SYNC`** — `0`/`1`/`2` (default: 1 on i915 and Alchemist cards, 0 elsewhere) — Drains the device queues before launching a verify window: 1 always, 0 never, 2 only on the first window or a >20 ms gap; the drain prevents the never-starting first window (hang) seen on Alchemist. Gate: SYCL verify window launch. Best for: leave at default; `=0` only on Battlemage where overlap is safe. Tags: [tunable]

### 7.4 Cross-vendor engine switches (288)

Read in the shared CUDA/HIP engine code — they apply to **NVIDIA and AMD** (and, where the
SYCL port mirrors the site, to Intel). Each entry's Gate line says which architecture the
runtime predicate selects. This is the bulk of the surface: expert cache, KV streaming,
prefill, speculation, CPU pool, sessions, multi-GPU split.

**`STRATA_ADAPT_CALLBACK_WARMUP`** — `0`/`1` (default: on) — on Linux, runs a no-op GPU host callback at startup so the driver creates its callback thread before the host core pins itself to one CPU, keeping that thread off the pinned verifier CPU. Gate: Linux only (Windows CPU Sets are per-thread and not inherited). Best for: leave at default; set `=0` only to debug callback-thread placement. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_ADAPT_JOB_CPU`** — `<logical cpu>` or `-1` (default: unset, uses the spare SMT sibling of the last two-thread core) — pins the adaptive tier's job thread (its memcpys) to a logical CPU; `-1` means no pinning, and an explicit value overrides `--aux-cpus` placement. Gate: all GPUs (CUDA/HIP/SYCL), host CPU placement, no GPU gate. Best for: fixing the adaptive tier's thread placement on SMT machines when the default spare CPU is wrong; leave at default otherwise. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_ADAPT_LAG`** — integer >= 1 (default: 1) — how many decode windows old the adaptive tier's swaps must be before a verify window takes them; lag 2 defers the copy wait one window further (Tesla P100 +3.5%, RTX 3060 -1.3%, so it stays opt-in). Gate: all GPUs (CUDA/HIP/SYCL), adaptive tier decode only; batch paths, --pipeline-windows and the peer tier are not lagged. Best for: A/B on slow-PCIe single-GPU rigs; leave at 1 otherwise. Tags: [ab] (docs: docs/DETAILS.md, docs/MULTI_GPU.md)

**`STRATA_ADAPT_NOWAIT`** — `1` (default: off; `e[0] == '1'`) — lets a verify window start before the adaptive tier's expert copies have landed (the 0.1.37 behaviour, #463's A/B arm). Gate: all GPUs (CUDA/HIP/SYCL), adaptive tier. Best for: A/B benchmarking the copy-wait stall; leave unset in normal runs. Tags: [ab] (docs: docs/DETAILS.md)

**`STRATA_ADAPT_OVERLAP`** — `1` (default: off, opt-in on every backend) — makes the adaptive tier's expert copies land in layer order while the next verify window runs, hiding PCIe swap time behind the window with identical token selection. Gate: shared CUDA/HIP, all backends; disabled with page-locked swap sources, STRATA_DMA_BATCH, pipeline windows, or STRATA_ADAPT_LAG/NOWAIT. Best for: turn on (`=1`) when expert swaps on a slow PCIe link dominate decode; leave off on WSL2/RTX 50-series where it measured slower. Tags: [tunable] (docs: docs/DETAILS.md, docs/OLDER_GPUS.md)

**`STRATA_ADAPT_OVERLAP_RESIDENT`** — `1` (default: off) — allows adaptive overlap in the resident-RAM swap mode, where the copies share RAM bandwidth with the CPU experts. Gate: shared CUDA/HIP, adaptive tier in resident RAM mode. Best for: try `=1` on machines with fast RAM (measured +7% on an RTX A4000, -2.6% on an RTX 3060). Tags: [tunable]

**`STRATA_ADAPT_OVERLAP_SLOW_US`** — integer microseconds (default: 0 = off) — test hook that sleeps each per-layer copy by N microseconds to simulate a slow PCIe link and force the verify windows to wait for copies. Gate: no vendor gate (test hook in the overlap path). Best for: leave at default; developers testing overlap behavior only. Tags: [debug]

**`STRATA_ARENA_LOCK`** — `0` to disable (default: on) — keeps the host expert arena resident by locking it through the process working set when full cudaHostRegister pinning is not possible (unpinned, Windows trims it under memory pressure); large-page backing is already resident so the lock is skipped. Gate: shared CUDA/HIP (and SYCL port), mainly Windows; no arch gate. Best for: leave at default; `=0` is the A/B arm measuring OS paging effects on the CPU pool. Tags: [ab]

**`STRATA_ARENA_PIN_GIB`** — integer GiB, `0`, or `auto` (default: unset; 29 GiB default on Windows) — caps how much of the arena is cudaHostRegister-pinned in slices; `auto` lets the Windows shared-memory budget set the cap (#243), `0` pins the whole arena on any OS. Gate: shared CUDA/HIP (SYCL port identical); on Linux the whole arena is normally registered. Best for: raise/lower when the pinned arena starves the GPU contexts or the WDDM shared-memory budget. Tags: [tunable] (docs: docs/SECOND_GPU.md, docs/MULTI_GPU.md, docs/DETAILS.md, docs/PREFILL_HEAD_ALIAS.md)

**`STRATA_ARENA_RELEASE`** — `0` to keep (default: release) — with a file-backed (mmap) arena, hands the VRAM-held experts' pages back to the OS after the VRAM fill, since left mapped and referenced they crowd every other allocation into swap. Gate: file-backed arena path (Linux mmap), shared CUDA/HIP/SYCL. Best for: leave at default; `=0` keeps them mapped as the A/B arm. Tags: [ab]

**`STRATA_ARENA_THP`** — `1` (default: off) — forces the transparent-huge-page request (MADV_HUGEPAGE) on the pinned arena even when less than half of it could be backed by currently-free 2 MiB pages, a case the engine skips because kernel compaction can cost minutes per GiB. Gate: Linux only (THP/madvise). Best for: leave at default on fragmented RAM boxes; `=1` only when you know THP defrag is cheap. Tags: [tunable]

**`STRATA_ARGMAX_MULTI`** — `0` to disable (default: on) — greedy sampling uses the multi-CTA-cluster argmax kernel (one cluster per token, sm_90+) when there are no penalties; `=0` always runs the one-block 1024-thread kernel. Gate: NVIDIA sm_90+ cluster path (falls back elsewhere); SYCL port has the same switch. Best for: leave at default; `=0` to A/B the sampler kernels. Tags: [ab] (docs: docs/DETAILS.md)

**`STRATA_ATTN_LANECELL`** — `1` (default: off) — decode QSA attention computes the score phase one cell per thread (bit-identical scores, no shuffle reductions) — the S25 decode layout experiment. Gate: shared CUDA/HIP/SYCL, QSA decode attention; on gfx906 it disables the query-swizzle variant. Best for: A/B the attention score layout on Strix Halo / older cards. Tags: [ab] (docs: docs/STRIX_HALO.md)

**`STRATA_ATTN_PRE75`** — `0`/`1` (default: auto by device) — selects PR #540's pre-sm_75 QSA attention kernel (fewer shuffles, bit-exact): it is the default on NVIDIA cards below sm_75 and on AMD gfx103x; `=0` runs the standard kernel, `=1` forces it on another wave32 AMD card. Gate: NVIDIA <sm_75 and AMD gfx103x by default. Best for: A/B the attention kernel on V100/RDNA2 cards. Tags: [ab] (docs: docs/OLDER_GPUS.md, docs/NVIDIA_V100.md)

**`STRATA_AUX_CPUS`** — `auto`, a CPU list (`24,26`), or a range (`24-27`) (default: unset) — declares which CPUs the process may use for helper threads (SMT siblings of the host core first, then cores with no pool worker); equivalent to `--aux-cpus`. Gate: CPU-affinity feature, Linux; a CPU sharing a core with a pool worker is never taken. Best for: set `auto` or an explicit spare-core list to give copy/IO helpers CPUs that don't fight the pinned workers. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_AUX_STAGER`** — `0` (default: pinned) — `=0` leaves the prefill copy (stager) threads on the CPUs where they were created instead of moving them to the aux CPU set. Gate: CPU-affinity feature, only meaningful with STRATA_AUX_CPUS/--aux-cpus set. Best for: set `=0` when the spare CPU set is tiny so 32 stager threads don't pile onto one CPU and slow a long-prompt read. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_BATCH_DECODE_SHARE`** — float >= 0 (default: 0.5) — in --batch mode, when a long prompt is read chunk by chunk while other slots decode, the slots get to decode for this share of the time each prompt chunk took, so a long prompt slows the others instead of stopping them. Gate: shared CUDA/HIP/SYCL, --batch server path, no vendor gate. Best for: tune the fairness between a long prompt read and concurrent decoding slots. Tags: [tunable] (docs: docs/BATCHING.md)

**`STRATA_BATCH_MTP`** — any non-empty non-`0` value (default: off) — enables `--batch-mtp` from the environment: each batch slot also verifies one MTP proposal per window; needs --batch 2+, --mtp, --spec T>=2 and --serve, and on a layer split the pipeline wins and turns it off. Gate: shared CUDA/HIP/SYCL, batched MTP serving; one GPU (no layer split) on the SYCL port. Best for: enabling batched MTP from the server environment; pair with `--batch-groups 1` / `STRATA_BATCH_MTP_SERIAL=1` for serial windows. Tags: [tunable] (docs: docs/BATCHING.md)

**`STRATA_BATCH_MTP_SERIAL`** — `1` (default: off) — keeps `--batch-mtp` running as one serial batch group (serial windows) instead of letting the engine pipeline groups and drop `--batch-mtp`. Gate: multi-GPU layer-split batch mode with `--batch-mtp`. Best for: set `=1` (like `--batch-groups 1`) for about two streams, where serial MTP groups measured better than pipelining. Tags: [tunable] (docs: docs/BATCHING.md)

**`STRATA_BATCH_YIELD_CHUNKS`** — integer chunks (default: 8; `0` = one run) — size of a decode piece in a pipelined layer-split batch read: a chunk is decoded in pieces of this many chunks so a cooperative preemption (BYIELD) can be taken between pieces. Gate: batch mode with a multi-GPU layer split. Best for: leave at default; lower for faster preemption of long generations, `0` to disable piecewise decoding. Tags: [tunable]

**`STRATA_BF16_TC`** — `0`/`1`/`2` (default: auto by arch) — chooses the BF16 projection GEMM path: 0 = cuBLAS BF16, 1 = convert to FP16 and use FP16 tensor cores with FP32 accumulation, 2 = FP32 (Pascal); the FP16 path is the default on sm_70-74, off on sm_75+, and `=2` is a test mode that runs the FP16 path on any card. Gate: NVIDIA 7.x cards (sm_80+ keeps native BF16); applies to the CUDA/HIP and SYCL gemm. Best for: `=1` on Turing (RTX 20, prompt +15-18%) and V100; not bitwise identical to cuBLAS BF16. Tags: [tunable] (docs: docs/OLDER_GPUS.md, docs/NVIDIA_V100.md)

**`STRATA_CACHE_MESSAGE_BOUNDARY`** — non-zero int (default: off) — adds a prompt-cache checkpoint at message boundaries: takes the last turn-token boundary that starts a prefix of at least 8,192 tokens with a tail of at most 1,024 as an extra snapshot point for cache reuse. Gate: shared CUDA/HIP/SYCL, single-GPU with prompt caching enabled and a valid turn token. Best for: turn on for chat servers that resume at message boundaries. Tags: [tunable] (docs: docs/MESSAGE_BOUNDARY_CACHE.md)

**`STRATA_CKPT_REREAD`** — presence (default: off) — checkpoint self-check: instead of restoring a conversation checkpoint, re-reads its tokens from position 0 in one batched run and compares token for token (also forces every read through the batched path), exposing anything the checkpoint missed. Gate: shared CUDA/HIP/SYCL, session-checkpoint path. Best for: validating checkpoint fidelity (use with --adapt-swaps 0); never set in normal serving. Tags: [debug]

**`STRATA_COMMIT_BATCH`** — `0`/`1` (default: on) — commits every GDN layer's conv history and state in two kernel launches total instead of two per layer (~70 fewer graph nodes, bitwise-identical arithmetic). Gate: shared CUDA/HIP; always off on Metal (the Mac engine keeps the per-layer loop). Best for: leave at default; `=0` only to A/B the per-layer commit loop. Tags: [tunable]

**`STRATA_COMMIT_SYNC`** — presence (default: off) — forces the verifier's window commit to run synchronously by disabling the async-commit path (`set_commit_async` is neutered while it is set). Gate: shared CUDA/HIP/SYCL, verify path. Best for: A/B the async commit for timing or reproducibility; leave unset. Tags: [ab]

**`STRATA_COMMIT_TIMING`** — `1` (default: off) — brackets each verify commit with CUDA events and prints accumulated commit latency every 300 commits. Gate: CUDA/HIP event timing, no vendor gate. Best for: leave at default; diagnostics when commit latency is suspected. Tags: [debug]

**`STRATA_CVEC_FUSE`** — `1` (default: off) — fuses a steered layer's write, control vector and next norm into one pass during prefill (S23 experiment). Gate: shared CUDA/HIP/SYCL, prefill hyper-connection path. Best for: A/B the fused prefill pass on Strix Halo; leave off elsewhere. Tags: [ab] (docs: docs/STRIX_HALO.md)

**`STRATA_CVEC_FUSE_CHECK`** — integer N (default: 0) — on the first N fused cvec writes, also runs the three-kernel path on a copy of R and reports whether R, the row scales and the BF16 image are bit-identical. Gate: shared CUDA/HIP/SYCL, prefill, only with STRATA_CVEC_FUSE on. Best for: validating the fused kernel; costs extra copies, leave at 0. Tags: [debug]

**`STRATA_DBG_CPU_GATE`** — `1` (default: off) — prints each layer's CPU-gate reading (non-resident expert count, ms per expert) and the resulting share/do-not-share decision. Gate: prefill CPU-sharing path, no vendor gate. Best for: diagnostics when tuning CPU expert sharing. Tags: [debug]

**`STRATA_DBG_FEAT`** — file path (default: unset) — research hook that dumps, per MoE layer and chunk, the routed expert ids and weights (plus router-input hidden state for STRATA_DBG_FEAT_LAYERS layers) in binary to the given file. Gate: MoE prefill path; research hook, never on by default. Best for: leave unset; routing research only. Tags: [debug]

**`STRATA_DBG_FEAT_LAYERS`** — comma-separated layer list (default: `3,4,15,16,27,28,39,40`) — which MoE layers also get their router-input hidden state (bf16) dumped by STRATA_DBG_FEAT. Gate: research hook, layers below 64. Best for: leave at default; research only. Tags: [debug]

**`STRATA_DBG_FORCE_SLOW_LAYER`** — `1` (default: off) — test switch that makes layer 0 count as a layer the fused prefill path does not take, exercising the buffer-size backstop so it reports a clear error instead of hanging. Gate: shared CUDA/HIP/SYCL, prefill fused-layout path. Best for: testing the backstop only; never set in normal runs. Tags: [debug]

**`STRATA_DBG_NAN`** — presence (default: off) — NaN hunting: reports the first non-finite head logits, the first prefill layer whose MoE produced non-finite values, the state the prompt leaves for the token path, and (HIP) non-finite FP16 GEMM outputs past 65504; costs a sync per check. Gate: shared CUDA/HIP/SYCL (the FP16 GEMM count is HIP-only). Best for: tracking down where NaNs enter; leave unset (it slows the run). Tags: [debug] (docs: docs/INTEL.md, docs/AMD_HIP.md)

**`STRATA_DBG_ROUTE_OVERRIDE`** — file path (default: unset) — research hook that replaces the MoE routing of the layers listed in the file with the file's expert ids and weights, replaying recorded routing. Gate: MoE prefill path; research hook. Best for: leave unset; routing research paired with STRATA_DBG_FEAT dumps. Tags: [debug]

**`STRATA_DECODE_TIMING`** — presence / `1` (default: off) — decode profiler: after each request the log gets one `strata decode timing:` line splitting the per-window verify time into the wait for expert copies, pool/host/plan/activation-queue/jobs/run phases, plus misses, entries, hits and PCIe experts. Gate: shared CUDA/HIP/SYCL, decode path. Best for: finding the decode bottleneck (pair with STRATA_VERIFY_PROFILE); leave off. Tags: [debug] (docs: docs/DETAILS.md, docs/OLDER_GPUS.md, docs/QUANTS.md)

**`STRATA_DEC_BATCH`** — `0` to disable (default: on) — batches the decode/verify window's per-token work: the GPU's row-copy zeroes the rows it computed itself, the shared expert runs one multi-row GEMV with fused sigmoid+scale for n_tok>1, and it gates the shared-expert gate/scale fusions (S26). Gate: shared CUDA/HIP/SYCL, decode/verify window path. Best for: leave at default; `=0` falls back to the per-token loops as the A/B arm. Tags: [ab]

**`STRATA_DISJOINT_ADAPT`** — non-zero int (default: off) — the adaptive tiers exclude from the primary GPU's promotion candidates any expert a helper GPU's cache currently holds (asked live via RemoteExperts::holds), keeping the two caches disjoint instead of duplicating an expert. Gate: shared CUDA/HIP/SYCL, multi-GPU with remote expert caches. Best for: turn on for multi-GPU expert-cache efficiency. Tags: [tunable] (docs: docs/DISJOINT_EXPERT_CACHE.md)

**`STRATA_DRAFT_TIMING`** — presence / `1` (default: off) — instruments the MTP drafter: CUDA events per draft step plus host-side timings (staging, graph launch, PLE prefetch, spin waits) reported as round/step/gap/wall/tail milliseconds, and syncs the prefill batch pass to time it. Gate: shared CUDA/HIP/SYCL, MTP draft path. Best for: profiling draft-round latency; costs syncs, leave off. Tags: [debug]

**`STRATA_DUMP_FIRST_LOGITS`** — `<path>` (default: unset) — writes the first verify window's logits (the prompt's last token over the state the prompt path left) to `<path>` (serve: `<path>.<n>` per request) so prompt paths — a peer card's share, a layer split — can be compared by KL. Gate: shared CUDA/HIP/SYCL, generate and serve paths. Best for: diagnosing prompt-path output divergence; leave unset. Tags: [debug]

**`STRATA_EARLY_REMOTE_CONTEXTS`** — `1` (default: off) — creates every secondary GPU's context at startup (like CUDA1's) before GPU0's weights and the host arena are allocated, so WSL2's ~1 GiB driver pinned/mapped budget is not already spent when a later context is created (fixes `cudaSetDevice(2) failed: out of memory`). Gate: multi-GPU; the problem is WSL2/NVIDIA. Best for: set on WSL2 multi-GPU runs that OOM at context creation. Tags: [tunable]

**`STRATA_EMULATE_CC`** — compute-capability integer 75–119 (default: 0 = off) — makes the engine report and behave as a different NVIDIA compute capability (kernel routing, shared-memory attributes), e.g. `=80` to run A100 paths. Gate: NVIDIA CUDA only. Best for: leave unset; development testing of other cards' code paths. Tags: [internal]

**`STRATA_EXCHANGE_ROTATE`** — `1` (strcmp == "1"; default: off) — an adaptive swap hands buffer ownership over (rotating the pinned exchange buffers) instead of copying the evicted blob into the RAM copy; needs equal-size blobs and a fully page-locked complement. Same tokens, fewer host copies; works with --adapt-async. Gate: shared CUDA/HIP/SYCL, adaptive tier with pinned complement. Best for: A/B the swap path (multi-GPU bench: 139.6 vs 142.8 tok/s arms). Tags: [ab] (docs: docs/EXCHANGE_ROTATION.md, docs/MULTI_GPU.md, docs/DETAILS.md)

**`STRATA_EXPERT_V2`** — `1` (default: off) — the S26 grouped decode-expert kernel for IQ3_S/IQ4_NL (gate/up type 21, down 20, n_embd 2560, n_ff 640), with the optional STRATA_TSUM single-butterfly warp sums. Gate: AMD (gfx1151/Strix Halo tuning); the SYCL port prints "AMD-only experiments, ignored on SYCL" and skips it. Best for: decode expert speed on AMD Strix Halo. Tags: [ab] (docs: docs/STRIX_HALO.md, docs/QUANTS.md)

**`STRATA_EXPERT_V2K`** — `0`/`1` (default: on on gfx12, off elsewhere) — the S27 grouped decode-expert kernels for UD-Q4_K_XL packs (Q4_K/Q5_K gate/up, Q5_1/Q8_0 down); every output is the old kernels' bits (R9700 decode +1.2% story, +0.8% code). Gate: AMD gfx12 by default, other AMD cards opt-in; the SYCL port prints "AMD-only experiments, ignored on SYCL" and skips it. Best for: leave at default; `=0` A/Bs the kernel on gfx12. Tags: [ab] (docs: docs/QUANTS.md)

**`STRATA_EXP_MODE`** — integer mode (default: unset = the card's default; SYCL port defaults to 7) — picks the native expert-set kernel/layout for VRAM-resident routed experts in decode/verify windows: mode 8 puts the codebook grid and group activations in shared memory with fused SwiGLU+q8_1 (Volta sm_70, gfx906), mode 7 is the AMD dp4a LDS layout, `0` the plain CUDA layout. Gate: shared CUDA/HIP (mode 8 on sm_70 and gfx906, mode 7 on other AMD); SYCL port takes the integer directly. Best for: `=0` to fall back to the CUDA layout on an old card; leave unset otherwise. Tags: [tunable] (docs: docs/OLDER_GPUS.md, docs/NVIDIA_V100.md, docs/AMD_HIP.md)

**`STRATA_FETCH_PVM`** — `0` to disable (default: on) — issues one Windows PrefetchVirtualMemory call for every expert slice about to be copied so the memory manager reads them in large queued requests and the copies find the pages resident. Gate: Windows only (kernel32 API). Best for: leave at default; `=0` is the A/B arm. Tags: [ab]

**`STRATA_FETCH_THREADS`** — integer > 0 (default: 8) — number of threads that fetch a layer's missing experts from the GGUF/experts file while the engine answers, in the low-RAM mmap-experts mode. Gate: shared CUDA/HIP/SYCL, file-backed expert source. Best for: raise on fast NVMe with many cores; 16 measured no faster. Tags: [tunable] (docs: docs/DETAILS.md, docs/UNSLOTH_Q4.md)

**`STRATA_FILE_RELEASE`** — `1` (strcmp == "1"; default: off) — lets FileExpertSource trim the full file-backed pages of experts after their GPU uploads complete (including slots lent to prefill and refilled), freeing RAM. Gate: Windows only. Best for: low-RAM Windows machines serving from experts.bin or direct GGUF views. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_FILL_AHEAD`** — integer >= 0 (default: 256) — how many profile fill pairs to advise ahead during the expert-cache fill so mapped-file reads overlap the copies; `0` turns the read-ahead off. Gate: shared CUDA/HIP/SYCL, mapped (non-unbuffered) file tier. Best for: tune on slow drives; leave at default otherwise. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_FS_AHEAD`** — `0`/`1` (default: on when STRATA_FS_SLOTS>0) — feeds the next layer's router predictions into the Foresight swap space so predicted experts are staged, not just the ones that just missed. Gate: shared CUDA/HIP, only with the Foresight swap space on. Best for: leave at default; `=0` is the A/B arm (misses only). Tags: [ab] (docs: docs/DETAILS.md)

**`STRATA_FS_SLOTS`** — integer N (default: 0 = off) — the Foresight swap space: N VRAM slots per layer refilled on a copy stream with the experts that just missed and the ones the next layer's router predicts, so a cache miss can run on the GPU instead of the CPU. Gate: shared CUDA/HIP; refuses to run beside STRATA_VERIFY_DEVICE_PLAN. Best for: set N>0 on file/pinned-RAM tiers to cut CPU expert work. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_FUSED_SWIGLU_Q81`** — `0` to disable (default: on) — fuses SwiGLU and the q8_1 activation quantize into one pass in the shared expert's path. Gate: shared CUDA/HIP/SYCL, shared expert. Best for: leave at default; `=0` is the A/B arm. Tags: [ab]

**`STRATA_FUSE_HEAD_GR`** — `1` (default: off) — fuses the head's gate/read (GR) work into the verify window's last-layer post stage (needs the last layer on this device, the head loaded, and no cvec covering it); also read in the MTP path. Gate: shared CUDA/HIP/SYCL, verify/MTP path. Best for: A/B the fused head pass; leave off by default. Tags: [ab]

**`STRATA_FUSE_WAIT`** — `0`/`1` (default: on on gfx12, off on other AMD, never on CUDA) — fuses the post-wait row copies into the waiting kernel (same reads, one kernel node fewer). Gate: AMD HIP; measured on gfx12 only. Best for: leave at default; force off/on for A/B. Tags: [tunable]

**`STRATA_GDN_CHUNKED`** — `1`/`2` (default: off) — computes the DeltaNet prompt recurrence in 32-token chunks (WY form, FP32, slightly different bits, KL 0.002-0.009); `=2` skips the 128-SM requirement to measure small cards. Gate: NVIDIA CUDA sm_80+ only (not HIP); needs 128 SMs for a win (RTX 5090: 1.8x on the recurrence; slower on 28-48 SM cards). Best for: `=1` on 128+ SM cards with long prompts. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_GDN_COMMIT_SPLIT`** — `0` to disable (default: on) — batches the GDN per-layer state commit of a verify window into one multi-layer kernel instead of one commit per layer; per-layer kernels are also forced by `STRATA_GDN_SPLIT=1` or `STRATA_DBG_GDN=1`. Gate: shared CUDA/HIP/SYCL, GDN verify commit path. Best for: leave at default; `=0` A/Bs the batched commit kernel. Tags: [ab]

**`STRATA_GDN_CONVL2`** — non-zero int (default: off) — runs the prompt's GDN convolution through the fused L2-tiled kernel (gdn_conv_l2_kernel + history kernel, convolution and norms fused) instead of the plain tiled kernel, for chunks above the tile size. Gate: shared CUDA/HIP/SYCL, prefill GDN recurrence. Best for: A/B the prompt recurrence on Strix Halo. Tags: [ab] (docs: docs/STRIX_HALO.md)

**`STRATA_GDN_CONV_SERIAL`** — presence (default: off) — forces the old serial walk for the prompt's GDN convolution (the one-block-per-column kernel) at any chunk size — the A/B against the tiled kernels. Gate: shared CUDA/HIP/SYCL, prefill GDN convolution. Best for: A/B only; leave unset. Tags: [ab]

**`STRATA_GDN_HEAD`** — non-zero int (default: on on gfx1151, off on other cards) — the Aurora (S23) prompt GDN recurrence with four lanes per column and the grid-stride norm, bit-identical to the old kernel. Gate: AMD builds only (arch default sets it on gfx1151); the SYCL port picks its Xe variant independently. Best for: leave at default; `=0` falls back to the earlier kernels. Tags: [ab] (docs: docs/STRIX_HALO.md)

**`STRATA_GDN_KEYHEAD`** — `0` to disable (CUDA/HIP default: on, chosen per device by SM count and occupancy on sm_80+) — runs the prompt GDN recurrence per key head (the value heads of one key head in a single thread), 1.28-1.57x on 4080 SUPER/3090 grids. Gate: NVIDIA Ampere+ by SM-count thresholds; on Intel Xe it is opt-in `=1` and measured 8% slower. Best for: leave at default; `=0` A/Bs the kernel. Tags: [ab] (docs: docs/INTEL.md)

**`STRATA_GDN_NOY`** — non-zero int (default: off) — the quad GDN recurrence's out-norm loop skips the intermediate FP32 y image and writes the BF16 output directly. Gate: shared CUDA/HIP/SYCL, prefill GDN quad recurrence. Best for: A/B on Strix Halo. Tags: [ab] (docs: docs/STRIX_HALO.md)

**`STRATA_GDN_PIPELINE`** — `0` to disable (default: on) — pipelines the prompt GDN recurrence through the key-head kernel (value heads of one key head in one thread, same bits) instead of the plain per-head kernel. Gate: shared CUDA/HIP/SYCL, prefill GDN recurrence (the key-head path is gated by SM occupancy on NVIDIA; Xe takes its own launch). Best for: leave at default; `=0` is the A/B arm. Tags: [ab]

**`STRATA_GDN_PP`** — `1` or `2` (default: 0) — picks the pipelined quad GDN recurrence kernel: 1 = the pipelined quad kernel, 2 = two columns per lane (gdn_rec_quad2c_kernel). Gate: shared CUDA/HIP/SYCL, quad recurrence variant (the quad kernels are AMD-only in the CUDA/HIP tree). Best for: `=2` is the Strix Halo setting. Tags: [ab] (docs: docs/STRIX_HALO.md)

**`STRATA_GDN_REC_HEADS`** — presence (default: off) — forces the old one-block-per-head GDN recurrence kernel, disabling the pipelined, head and Xe variants — the A/B arm. Gate: shared CUDA/HIP/SYCL, prefill GDN recurrence. Best for: A/B only. Tags: [ab]

**`STRATA_GDN_SPLIT`** — `1` (default: off) — splits the GDN verify step over four blocks per head (state-replay kernel plus separate norm), 8-19% faster per commit call and 3-13% for 2+-row windows on gfx1151, bitwise equal; also forces per-layer commit kernels. Gate: shared CUDA/HIP/SYCL, GDN verify step. Best for: A/B on Strix Halo. Tags: [ab] (docs: docs/STRIX_HALO.md)

**`STRATA_GFX1151_DEFAULTS`** — `0` (default: on for gfx1151) — master switch for the gfx1151 (Strix Halo "Aurora") table of bitwise-exact tuned kernel defaults (GDN_HEAD, GDN_PP, GDN_CONVL2, GDN_NOY, CVEC_FUSE, ...); `=0` turns the whole table off and the startup line names what was set. Gate: AMD gfx1151 only; nothing applies to CUDA or other AMD archs. Best for: `=0` to debug a regression in the tuned defaults. Tags: [tunable] (docs: docs/STRIX_HALO.md)

**`STRATA_GFX12_DEFAULTS`** — `0` (default: on for gfx12) — master switch for the gfx12 (RDNA4) defaults: the prompt-half kernel table (GDN_HEAD, GDN_PP, GDN_CONVL2, GDN_NOY, CVEC_FUSE), Q8_0 hyper-connection projections, and the 16384 prefill chunk. Gate: AMD gfx12 only; a CUDA build never consults it. Best for: `=0` to revert everything gfx12-specific. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_GROUPED_PAIR_MIN_HITS`** — integer (default: 0 = always the new kernels) — the minimum hits per token for which the new two-rows-per-warp grouped expert kernels are used; below it the engine takes the old per-hit kernels (fewer blocks than an RTX 5090's SMs at one hit). Bitwise the same either way. Gate: shared CUDA/HIP/SYCL, grouped MoE expert kernels. Best for: tune on cards with many SMs via `--bench`; leave at 0 otherwise. Tags: [tunable]

**`STRATA_GROUP_COPY`** — `1` (default: off) — keeps the existing group-plan buffer when it is too small for the chunk instead of reallocating a bigger mapped pinned one. Gate: shared CUDA/HIP/SYCL, prefill group planning. Best for: A/B the pinned-buffer growth; leave off. Tags: [ab]

**`STRATA_GR_DOWN_MAX4`** — non-zero int (default: off) — fused GR down launches of up to 4 tokens hold 4 tokens' sums per thread instead of the full window maximum; same bits, and the RTX 5070 A/B did not reproduce the bench win (neutral on the Intel B70). Gate: shared CUDA/HIP/SYCL, fused GR kernels. Best for: A/B on a new card; leave off. Tags: [ab] (docs: docs/INTEL.md)

**`STRATA_GR_FAST`** — `0`/`1` (default: on on HIP; on CUDA on Volta, opt-in elsewhere) — chooses the latency-hidden hyper-connection read kernel (norm loads in flight, 8 lanes per row) over the plain one; bitwise identical. Gate: AMD HIP on by default; CUDA on by default only on Volta (V100-SXM2: 50.9→46.7 µs a read), opt-in on other NVIDIA cards (slower on P40 below 5 tokens). Best for: `=0` to A/B the plain read. Tags: [tunable] (docs: docs/OLDER_GPUS.md, docs/NVIDIA_V100.md, docs/AMD_HIP.md)

**`STRATA_GR_NORM_FAST`** — `0`/`1` (default: on on gfx12, off on other AMD) — the register-based split norm kernel for the hyper-connection read. Gate: AMD HIP; on only for gfx12 (RDNA4) until another arch is measured. Best for: leave at default; force off/on for A/B. Tags: [tunable]

**`STRATA_GR_SPLIT`** — `1` (default: off) — splits the down projection along K (8 rows × 5 slices of 2048) in the multi-token hyper-connection read, ~3% per verify window, but the fixed-order finish sums K slices in a different order, so multi-token reads are no longer bitwise equal to single-token calls. Gate: AMD gfx906-only build. Best for: MI50 owners trading bitwise parity for ~3%. Tags: [tunable] (docs: docs/AMD_HIP.md)

**`STRATA_GR_UNFUSED`** — `1` (default: off) — keeps the FP32 copy of the normalized rows (separate gr_norm + gr_mix kernels) instead of the fused GR prefill path — the F-1 A/B arm. Gate: shared CUDA/HIP/SYCL, prefill GR path. Best for: A/B only. Tags: [ab]

**`STRATA_GR_V3`** — non-zero int (default: off) — the two-kernel split GR read, which sums in another order: equal within float rounding, not bit-for-bit (parity tests compare it with a 2e-6 relative tolerance). Gate: shared CUDA/HIP/SYCL, fused GR read; on Intel it measured -4 to -6%. Best for: A/B the GR read; leave off. Tags: [ab] (docs: docs/INTEL.md)

**`STRATA_HC_Q8`** — `1`/`0`/`auto` (default: on on AMD gfx12 unless STRATA_GFX12_DEFAULTS=0, on by default on Metal, off elsewhere) — reads the hyper-connection projections from Q8_0 bytes (the GGUF's own Q8_0, or a copy quantized at load) for the verify read; on gfx12 the R9700 decode gains +2.5-4% with KL near the noise floor. Gate: AMD gfx12 default, Metal default, SYCL port opt-in `=1`. Best for: leave at default per platform; `=0` restores the exact BF16 read. Tags: [tunable] (docs: docs/STRIX_HALO.md, docs/QUANTS.md, docs/MAC.md)

**`STRATA_HC_Q8_INJECT`** — `1` (default: off) — asks for a Q8_0 copy of the F32 inject weights too; by default the read takes the pack's BF16-exact rows for those. Gate: shared CUDA/HIP/SYCL, only with STRATA_HC_Q8 on. Best for: leave off; `=1` A/Bs the inject read. Tags: [ab]

**`STRATA_HC_SPLIT`** — `0`/`1`/`2` (default: unset = staged) — picks the fused GR (hyper-connection) read variant: 0 = the plain read, 1 = at most the split kernel, 2/unset = the newest staged variant that the per-device fused_gr_check accepts. Gate: shared CUDA/HIP/SYCL, fused GR read. Best for: leave unset; use the numbers to force a variant in parity tests. Tags: [tunable]

**`STRATA_HC_UPMIX`** — `1` (default: off) — fuses the hyper-connection up projection with its mix epilogue into one kernel (gr_upmix) on prompt chunks of STRATA_PF_SWITCH_MIN_T+ tokens with Bf16InF32 up weights. Gate: shared CUDA/HIP/SYCL, prefill. Best for: Strix Halo prompt config. Tags: [ab] (docs: docs/STRIX_HALO.md)

**`STRATA_HC_UPMIX_CHECK`** — integer N (default: 0) — on the first N upmixes, also runs the default projection+mix pair and compares it against the fused gr_upmix output. Gate: shared CUDA/HIP/SYCL, prefill, only with STRATA_HC_UPMIX on. Best for: validating the fused kernel; leave at 0. Tags: [debug]

**`STRATA_HEAD_MIX_MULTI`** — `0` to disable (CUDA/SYCL default: on; HIP default: off, opt-in `=1`) — the head's hyper-connection read for all of a window's tokens in one launch set instead of per-token gr_read; bitwise the same sums. Gate: on for CUDA and the SYCL port, opt-in on HIP. Best for: leave at default; flip to A/B the head read per backend. Tags: [ab]

**`STRATA_HOST_CORE`** — `last` (default: first) — runs the host thread on the last physical core and gives the workers the first cores; Windows routes a GPU's interrupts to the first core, where a host spinning on GPU flags would wait for them. Gate: host CPU placement (the Windows case; shared CUDA/HIP/SYCL code). Best for: set on Windows boxes whose GPU interrupts land on core 0. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_IDX_FP16_CHECK`** — presence (default: off) — research check: re-selects every query from pooled indexer keys rounded to FP16 and prints the cumulative agreement with the FP32 selection after each chunk (the KV-streaming design's open question); syncs per layer. Gate: shared CUDA/HIP/SYCL, QSA prefill indexer. Best for: design studies only. Tags: [debug]

**`STRATA_INDEXER_PER_TOKEN`** — presence (default: off) — restores the old per-token QSA indexer appends (one launch per token) instead of the batched three-launch append of a whole chunk; same end state. Gate: shared CUDA/HIP/SYCL, QSA prefill. Best for: A/B the append batching. Tags: [ab]

**`STRATA_IO_PF_AHEAD`** — `0`/`1` (default: on) — with file-tier I/O prefetch, reads the router-predicted experts of the next layers ahead of demand; `=0` keeps only the demand reads. Gate: Linux mapped file tier with STRATA_IO_PREFETCH=1. Best for: leave at default; `=0` is the A/B arm. Tags: [ab] (docs: docs/DETAILS.md)

**`STRATA_IO_PF_THREADS`** — integer (default: 8) — number of I/O threads doing the whole-blob preads for file-tier prefetch. Gate: Linux mapped file tier with STRATA_IO_PREFETCH=1. Best for: raise on fast NVMe with many spare cores. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_IO_PREFETCH`** — `1` (default: off) — Linux file-tier I/O path: the layer's uncached experts and the router-predicted ones are read by I/O threads with whole-blob `pread`s into the page cache instead of page faults; identical bytes, identical output. Gate: Linux mapped file tier only, unavailable with unbuffered reads. Best for: `=1` when the file tier's page-fault reads bottleneck decode. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_IO_PREFETCH_DEPTH`** — integer layers (default: 2 when I/O prefetch is on) — how many layers ahead the routing-aware prefetch reads predicted experts. Gate: Linux file tier with STRATA_IO_PREFETCH / Foresight lookahead. Best for: raise for deeper read-ahead on fast drives, lower to cut wasted reads. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_IO_STATS`** — `1` (default: off) — adds the page-cache hit/miss split (mincore) and the read-ahead used/unused counts to the per-request "file tier I/O" log line. Gate: file tier I/O path. Best for: turn on to measure whether prefetch pays off. Tags: [debug] (docs: docs/DETAILS.md)

**`STRATA_IO_THREADS`** — integer 1-64 (default: 4 on Windows, 16 on Linux) — the number of issuing threads for direct unbuffered file reads; on Linux each thread does one blocking pread, so the count is the queue depth. Gate: shared CUDA/HIP/SYCL, host file I/O, no GPU gate. Best for: raise to 32-64 on fast NVMe (the PLE row reader's 466 -> 330 ms); beyond the drive's IOPS nothing changes. Tags: [tunable] (docs: docs/AMD_HIP_PERFORMANCE.md, docs/INTEL.md)

**`STRATA_IQ_STAGE_GRID`** — `0` to disable (default: on) — stages the 64-bit i-quant codebook tables into shared memory in the grouped IQ expert kernels. Gate: shared CUDA/HIP/SYCL, IQ kernels. Best for: leave at default; `=0` is the A/B arm. Tags: [ab]

**`STRATA_IQ_STAGE_GRID_MMVQ`** — `1` (default: off) — also stages the codebook in the single-matrix mmvq kernels, where the 2-8 KB staging per 128-thread block made IQ2_XS/IQ2_S/IQ1_M 10-20% slower on the RTX 5070; bitwise the same either way. Gate: shared CUDA/HIP/SYCL, native_mmvq path. Best for: A/B on a card with fast shared memory. Tags: [ab]

**`STRATA_KEEP_MAPPING`** — `1` (default: off) — keeps the mapped view of experts.bin open when the file tier runs unbuffered; by default it is closed because an open mapping makes NTFS serve the unbuffered reads one at a time. Gate: unbuffered file tier (matters on NTFS/Windows). Best for: `=1` as the A/B arm to test keeping the mapping. Tags: [ab]

**`STRATA_KV_GROW`** — `1`/`0` (default: off; the `--kv-grow` flag is the same switch) — elastic K/V: the K/V takes VRAM only for the cells requests reach, and the profiled expert cache gives slots back as the context grows. Gate: shared CUDA/HIP/SYCL; needs one GPU, an expert profile, the whole K/V in VRAM, every expert in RAM, VMM available, and no --batch / --vram-elastic / --peer-device. Best for: long contexts on one GPU with a profile. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_KV_GROW_FLOOR`** — integer > 0 (default: computed) — the number of expert-cache slots the elastic K/V leaves untouched as its floor (the cache keeps this many; the K/V grows into new VRAM only above it). Gate: elastic K/V path (--kv-grow). Best for: tests and tuning the growth boundary; leave unset. Tags: [debug]

**`STRATA_KV_GROW_HOLD`** — `1` (default: off) — with elastic K/V, keeps the K/V pool grown to the longest parked conversation at a request boundary instead of trimming it back to the incoming request alone. Gate: serve-mode elastic K/V (kvg) with parked sessions. Best for: `=1` when conversations alternate lengths and re-growing the pool is costly. Tags: [tunable]

**`STRATA_KV_GROW_INIT`** — integer > 0 (default: 16384) — the number of K/V cells mapped into VRAM at startup when elastic K/V is on. Gate: elastic K/V path. Best for: raise for servers that start long contexts immediately. Tags: [tunable]

**`STRATA_KV_GROW_STEP`** — integer > 0 (default: built-in) — the cell step the elastic K/V grows by when requests need more room. Gate: elastic K/V path. Best for: tune the growth granularity; leave unset. Tags: [tunable]

**`STRATA_KV_HOST_DMA`** — `1` (default: off) — the streamed KV append writes only the staging pool and resident slots, and one DMA copies the chunk's blocks to the host copy, so no GPU kernel writes RAM over PCIe; the bytes readers see are the same. Gate: AMD HIP only (never on CUDA). Best for: the documented fix for AMD KV-streaming prompt problems. Tags: [tunable] (docs: docs/TROUBLESHOOTING.md)

**`STRATA_KV_MAIN_OFF`** — presence (default: off) — A/B switch: keeps the main layers' K/V fully resident (skips the streaming ring for them); STRATA_KV_RING_OFF is the drafter-side twin. Gate: shared CUDA/HIP/SYCL, KV streaming (--kv-resident). Best for: A/B the streaming split; leave unset. Tags: [ab]

**`STRATA_KV_PREFETCH`** — `1`/`0` (default: off) — stages the next streamed layer's KV prefix on its own stream, with a release event protecting the staging pool's last reader and a ready event ordering the next attention consumer. Gate: shared CUDA/HIP/SYCL, KV-streaming prefill. Best for: prompt speed with KV streaming. Tags: [tunable] (docs: docs/KV_PREFETCH.md)

**`STRATA_KV_RING_OFF`** — presence (default: off) — A/B switch: keeps the MTP drafter's K/V fully resident, skipping its streaming ring; STRATA_KV_MAIN_OFF is the main-layers twin. Gate: shared CUDA/HIP/SYCL, KV streaming (--kv-resident). Best for: A/B the drafter streaming; leave unset. Tags: [ab]

**`STRATA_KV_ROT`** — `1` (default: off) — runs INT8 K/V through the Hadamard rotation that `--kv q4_0` already uses; first-token KL to fp16 K/V improved on an NVFP4 pack (0.0066 -> 0.0051) but not on IQ2_XS (0.0022 -> 0.0054). Gate: shared CUDA/HIP/SYCL, with `--kv int8`; part of the cache-key digest. Best for: try on packs whose quantization benefits from rotation. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_KV_STAGE_OWN`** — presence (default: off) — A/B switch: the streamed session's KV staging pool gets its own allocation instead of borrowed expert slots, so a streamed run lends the prompt path exactly the slots a resident run does and the comparison isolates the KV placement. Gate: shared CUDA/HIP/SYCL, KV-streaming prefill. Best for: clean resident-vs-streamed A/Bs only. Tags: [ab]

**`STRATA_LAYER_SPLIT_GATE`** — any value (presence check; default: off) — prints the multi-GPU layer-split gate's decision data to stderr (K, chunk, and per-device lb/cnt/used/cap/room/need/slots) when the engine picks a prefill chunk for a layer split. Gate: shared CUDA/HIP, multi-GPU layer-split path, no vendor gate. Best for: diagnosing why a particular layer split/chunk was chosen; leave unset otherwise. Tags: [debug]

**`STRATA_LAYER_SPLIT_PROBE`** — `0`/`1` (default: on; `=0` skips) — runs an exact startability probe of the chosen layer split against live free-memory figures, confirming every participant can lend a 512-token chunk or afford its own buffers with 512 MiB headroom before serve. Gate: multi-GPU with auto layer split, no vendor gate. Best for: leave at default; set `0` to skip the pre-flight probe. Tags: [internal]

**`STRATA_LFUSE`** — `1` (default: off) — the S26 layer fusion in the verify window: fewer launches around the shared expert and the KV append (applies with the batched decode path, native MoE combine, 2-8 entries). Gate: shared CUDA/HIP/SYCL, decode/verify window. Best for: A/B on Strix Halo. Tags: [ab] (docs: docs/STRIX_HALO.md, docs/INTEL.md)

**`STRATA_LFUSE_GATE`** — `0` to disable (default: on) — selects the gate part of the STRATA_LFUSE fusion; only meaningful with STRATA_LFUSE=1. Gate: shared CUDA/HIP/SYCL, verify window. Best for: A/B the fusion's parts. Tags: [ab]

**`STRATA_LFUSE_PAIR`** — `0` to disable (default: on) — selects the paired-kernel part of the STRATA_LFUSE fusion; only meaningful with STRATA_LFUSE=1. Gate: shared CUDA/HIP/SYCL, verify window. Best for: A/B the fusion's parts. Tags: [ab]

**`STRATA_LOGPOS`** — `<path>` (default: unset) — quant yardstick: teacher-forces the prompt through the verify windows and appends each token's log-probability (plus top-1 agreement columns) to the file; forces the serial window loop and enables head sampling on prompt windows. Gate: shared CUDA/HIP/SYCL (Metal too). Best for: KL/top-1 measurement runs; never set in serving. Tags: [debug] (docs: docs/QUANTS.md, docs/UNSLOTH_Q4.md)

**`STRATA_LOGPOS_EXTRA`** — token id (default: 248046, `<|im_end|>`) — the extra token whose log-probability is appended as the last column of each STRATA_LOGPOS line, so a chat model's end-of-turn mass can be taken out of the measurement. Gate: with STRATA_LOGPOS. Best for: set to the model's own end-of-turn token. Tags: [debug]

**`STRATA_LOGPOS_TOPK`** — integer 0-256 (default: 0) — appends the K most likely tokens and their log-probabilities (`id:logprob`) to each STRATA_LOGPOS line for a top-k KL comparison against another engine (20 for llama.cpp runs, 64 for the quant yardstick). Gate: with STRATA_LOGPOS. Best for: set with STRATA_LOGPOS for KL harnesses. Tags: [debug] (docs: docs/QUANTS.md, docs/UNSLOTH_Q4.md)

**`STRATA_LOOKAHEAD`** — `0` to disable (default: on in the warming file mode) — routing-aware prefetch: while the CPU works on a layer, a thread applies the next layer's router to this layer's MoE input and asks the OS to read the predicted experts' pages; pages only, the answers are the same (UD-Q4_K_XL: mean +14%). Gate: shared CUDA/HIP/SYCL, file-tier expert source. Best for: leave on in the low-RAM file mode; `=0` is the A/B arm. Tags: [tunable] (docs: docs/DETAILS.md, docs/UNSLOTH_Q4.md)

**`STRATA_LOOKAHEAD_K`** — integer (default: 10) — the number of experts per token the router look-ahead predicts when warming the file tier's pages (the k of the top-k guess). Gate: shared CUDA/HIP/SYCL, with routing-aware prefetch on. Best for: match the model's top-k routing (default 10); leave at default. Tags: [tunable] (docs: docs/DETAILS.md, docs/UNSLOTH_Q4.md)

**`STRATA_LOOKAHEAD_STATS`** — `0`/`1` (default: off; `atoi != 0`) — measurement-only mode that runs the router look-ahead expert predictor on any tier and scores it against real routing, logging a `strata lookahead stats:` line every 100 windows (recall/precision of top-k guesses, PCIe share). Gate: MoE models with Bf16InF32 router weights, no vendor gate. Best for: measuring expert-prediction quality; not a performance knob. Tags: [debug] (docs: docs/DETAILS.md)

**`STRATA_LOOKUP_CHAIN_FIXED`** — presence (default: off) — removes the acceptance-probability policy gate from `--lookup-chain` drafts: the proposed chain length is offered as-is instead of being trimmed by the MTP acceptance model. Gate: shared CUDA/HIP/SYCL, MTP decode with --lookup-chain. Best for: A/B the chain policy; leave unset. Tags: [ab]

**`STRATA_MEMORY_LIMIT_MIB`** — nonnegative integer MiB (default: 0 = no limit) — sets the RAM guard: the total memory this engine's container/cgroup may use (for parking and session save/restore); the `--memory-limit-mib` flag wins over it. Gate: no vendor gate (host RAM accounting). Best for: set when the container cannot see its own cgroup limit. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_MMQ_KQ`** — `1` to enable (default: off; the strata-q36 engine sets it to 1 for its K-quant family) — routes prompt-time products of K-quant experts (Q4_K/Q5_K/Q6_K/Q8_0) through llama.cpp's MMQ kernels from the strata_kq library instead of dequantizing to the FP16 path. Gate: shared CUDA/HIP (library opened only for models with these experts; on AMD the code object loads on first use). Best for: prompt speed on K-quant MoE files; opt-in because MMQ rounds products differently from the FP16 path. Tags: [ab] (docs: docs/QUANTS.md, docs/UNSLOTH_Q4.md, docs/UNSLOTH_Q6.md)

**`STRATA_MMVF_ROWS`** — `0`/`1` (default: 0) — Routes multi-token BF16 GEMV (2–8 tokens, n_out≥64) through the experimental "rows" kernel instead of the default multi kernel. Gate: same code in CUDA/HIP and SYCL, no vendor gate in code; A770 benchmarks measured the rows path 1.4–6x slower and rejected it as a default. Best for: leave at default; A/B benchmark arm only. Tags: [ab] (docs: docs/A770_SPEED_SPEC.md, docs/STRIX_HALO.md)

**`STRATA_MMVQ_IL`** — `0`/`1` (default: on; `=0` keeps native_mmvq's multi-column kernels) — switches the dense projections of decode/verify windows (2-4 tokens) to the interleaved small-batch matvec reading interleaved q8_1 activation copies; outputs are bitwise the same, only speed differs. Gate: shared CUDA/HIP, per-card rows table (tables measured for P40, V100). Best for: leave at default; set `0` to fall back to the plain kernel on a card without a good table. Tags: [tunable] (docs: docs/MMVQ_IL_TABLE.md, docs/NVIDIA_V100.md, docs/OLDER_GPUS.md)

**`STRATA_MMVQ_IL_ROWS`** — `"type:ncols:r0,r1,r2,r3,r4;..."` (default: unset) — overrides rows of the interleaved-matvec table without rebuilding: type is the ggml type (23/12/13/14), ncols 2-4, and the five r values (0/1/2/4) are rows-per-warp for the five n_out row classes. Gate: shared CUDA (native_mmvq path). Best for: measuring a table for a new card and A/B-ing decode with it set vs unset. Tags: [tunable] (docs: docs/MMVQ_IL_TABLE.md)

**`STRATA_MMVQ_WAVE`** — `0`/`1` (default: wave layout on) — On gfx906 the MMVQ kernel runs one wavefront per row with a 64-lane butterfly; `=0` falls back to the CUDA one-block-per-row layout. Gate: AMD gfx906 (wave64) only — the code is compiled under `STRATA_HIP_GFX906`. Best for: leave at default; use `=0` for parity tests (the `mmvq_multi_parity_cuda_layout` ctest). Tags: [ab] (docs: docs/AMD_HIP.md)

**`STRATA_MTP_BATCH`** — `0`/`1` (default: on) — Batches the MTP drafter's prompt K/V appends through the prefill path; `=0` declines it so the drafter runs its own pass. Gate: all GPUs (CUDA/HIP/SYCL), no gate. Best for: leave at default; `=0` to A/B the drafter's own pass. Tags: [ab] (docs: docs/MULTI_GPU.md)

**`STRATA_MTP_BATCH_F16`** — `0`/`1` (default: 0) — Forces the batched draft K/V projections to FP16 GEMMs instead of the default Q8_1×Q8_0 MMQ integer dot products (which keep K/V close to the drafter's own mmvq math). Gate: all GPUs, requires the MMQ build (`mmq::built()`); no arch gate. Best for: leave at default; A/B arm for K/V fidelity/speed. Tags: [ab]

**`STRATA_MTP_BATCH_RING`** — `0`/`1` (default: on) — Lets a ring-mode (KV streaming) drafter K/V state take the batched appends; `=0` sends ring drafters back to their own pass. Gate: all GPUs, no gate; only relevant when the drafter's KV mode is ring. Best for: leave at default; A/B arm for ring KV streaming. Tags: [ab] (docs: docs/MULTI_GPU.md)

**`STRATA_MTP_CATCHUP_ALL`** — any non-`0` value (default: off) — Catches the drafter's K/V up over the whole verified window, rejected rows included, instead of only accepted rows. Gate: all GPUs (CUDA/HIP/SYCL), no gate. Best for: leave at default; experiment if drafter acceptance looks off. Tags: [tunable]

**`STRATA_MTP_FULL_HEAD`** — `0`/`1` (default: 0) — Diagnostic: the drafter drafts over the whole vocabulary (the main head) instead of the `draft_vocab.bin` subset, to measure what the subset costs in acceptance; also disables the drafter-on-first-stage pipeline optimization. Gate: all GPUs, no gate. Best for: leave at default; measurement only. Tags: [debug]

**`STRATA_MTP_PREFILL_SYNC`** — integer (atoi; default: on under HIP, off under CUDA) — Syncs the drafter's prefill K/V queue per group; on gfx1201/ROCm 7.2.4 the unsynced queue launch path intermittently hangs, so HIP defaults to syncing. Gate: default-on for AMD HIP, default-off for NVIDIA CUDA; `=1` forces syncing anywhere. Best for: set `=1` if prompt prefill hangs on AMD. Tags: [tunable]

**`STRATA_MTP_PRIORITY`** — integer (atoi; default: on) — Gives the MTP draft stream the highest CUDA stream priority when pipeline-windows forcing shares its card. Gate: NVIDIA CUDA only (HIP and Metal keep default priority; the SYCL port drops queue priority entirely — DPCT substitution). Best for: leave at default; `=0` to test unprioritized scheduling. Tags: [tunable]

**`STRATA_MTP_PROFILE`** — `1` (default: off; checks `e[0] == '1'`) — turns on MTP graph profiling: allocates a device-side stamp buffer per graph slot and accumulates per-slot timing sums for the draft/verify graphs. Gate: shared CUDA/HIP, MTP path. Best for: profiling MTP window timing; leave off in normal runs. Tags: [debug]

**`STRATA_MTP_SHARED_BRANCH`** — integer (atoi; default: on for CUDA, forced off for HIP) — Runs the MoE shared expert on a side stream branch beside the router and routed experts, joined before the combine. Gate: NVIDIA CUDA only (HIP keeps one stream). Best for: leave at default; `=0` to A/B single-stream. Tags: [ab]

**`STRATA_MTP_STAGE0`** — `0`/`1` (default: on when eligible) — puts the MTP drafter on the first stage's card for a two-stage pipelined decode so its chain runs while that card is idle, instead of sharing the last card with the verifier (stage-1 window 11 -> 25 ms when shared). Gate: multi-GPU pipelined decode (`pipeline_windows >= 2`, batch 0), no vendor gate. Best for: `=0` keeps the drafter on the last card — the A/B arm for multi-GPU MTP. Tags: [ab] (docs: docs/MULTI_GPU.md)

**`STRATA_MTP_TOP2`** — `0`/`1` (default: 0) — Diagnostic for tree drafts: records each draft step's runner-up token from the head logits on the host, to measure what a second draft branch would have caught; slow. Gate: all GPUs (CUDA/HIP/SYCL), no gate. Best for: leave at default; acceptance research only. Tags: [debug]

**`STRATA_MULTI_BLOCK_ARGMAX`** — integer (atoi; default: on) — Uses multi-block argmax/top-k head kernels on NVIDIA; the HIP build returns false and never uses them, so the variable has no effect there. Gate: NVIDIA CUDA only. Best for: leave at default; `=0` to A/B single-block head ops. Tags: [ab]

**`STRATA_NAN_GUARD`** — `0`/`1` (default: on) — the verifier's non-finite guard: records a kernel that checks each window's output for NaN/inf; `=0` records no kernel at all (read once at graph capture). Gate: shared CUDA/HIP, verify path. Best for: leave on; set `0` to A/B whether the guard changes answers (measured byte-identical on an RTX 3080). Tags: [debug] (docs: docs/DETAILS.md)

**`STRATA_NAN_INJECT`** — `<n>[,<n>...]` window numbers (default: unset) — debug switch that makes the NaN guard report the n-th windows of the process (counted from 1 over prompt-tail, decode, and batch windows) as non-finite, exercising the whole recovery path without a real fire. Gate: shared CUDA/HIP, verify path. Best for: testing the guard/retry path only. Tags: [debug] (docs: docs/DETAILS.md)

**`STRATA_NORM_ROPE`** — any non-`0` value (default: off on AMD, on by default for CUDA) — Opts into the fused per-head RMSNorm+rope kernel on AMD HIP (opt-in until rope_parity check 6 passes there); CUDA enables the fusion without it. Only applies when head_dim is 128/256 and n_rot is 64. Gate: AMD HIP opt-in; CUDA on by default; SYCL shares the CUDA logic. Best for: set `=1` on AMD once parity is verified. Tags: [tunable]

**`STRATA_NO_ARENA_THP`** — any value (default: unset) — Skips the `MADV_HUGEPAGE` request for the expert arena on any THP setting, avoiding minutes of kernel compaction on fragmented machines. Gate: host memory path (Windows/Linux arena code in CUDA/HIP and SYCL), not GPU-specific. Best for: set on a fragmented Linux box whose model load stalls in memory compaction. Tags: [tunable] (docs: docs/TROUBLESHOOTING.md)

**`STRATA_NO_BATCH_KV_STEP`** — any non-`0` value (default: off) — Falls back to per-token K norm/rope/K-V append in the drafter step instead of the batched all-T-rows version. Gate: all GPUs (CUDA/HIP/SYCL), no gate. Best for: leave at default; A/B arm for the batched KV step. Tags: [ab]

**`STRATA_NO_IQ256`** — any value (default: unset) — Disables the AVX-2 IQ2_XXS/IQ2_S expert gather kernels, sending those rows to ggml-cpu vec_dot; also gates the opt-in single-token IQ3_S AVX-2 path. Gate: CPU expert kernels (AVX-2 machines), not a GPU gate. Best for: leave unset; set to test ggml-cpu rounding parity. Tags: [ab]

**`STRATA_NO_LARGEPAGES`** — any value (default: unset) — Skips the large-page arena attempt (Windows `MEM_LARGE_PAGES`, Linux hugetlb 2 MB) and the THP request, allocating the arena with 4 KB pages in the same run. Gate: host memory path on Windows and Linux, all GPU backends. Best for: A/B the page-backing path without rebooting, or when large pages cause trouble. Tags: [ab] (docs: docs/TROUBLESHOOTING.md)

**`STRATA_NO_MMVQ_ROWS2`** — any non-`0` value (default: off) — Keeps one row per block in the multi-column MMVQ kernel for large K, disabling the two-rows-per-block layout; `=0`/unset enables two rows per block for every K. Gate: all GPUs (CUDA/HIP/SYCL), no gate. Best for: leave at default; A/B arm for MMVQ block layout. Tags: [ab]

**`STRATA_NO_MULTI_GR`** — any non-`0` value (default: off) — Keeps the per-token GR (gated-RMSNorm/read) calls instead of the multi-token fused GR kernel, in the drafter step and the GR kernels. Gate: all GPUs (CUDA/HIP/SYCL), no gate. Best for: leave at default; A/B arm for the fused multi-token GR. Tags: [ab]

**`STRATA_NO_NORM_ROPE`** — any non-`0` value (default: off) — Disables the fused RMSNorm+rope kernel everywhere, keeping the two separate kernels. Gate: all GPUs (CUDA/HIP/SYCL); overrides `STRATA_NORM_ROPE` on AMD. Best for: set `=1` to debug rope/norm parity. Tags: [debug]

**`STRATA_OLD_GROUPED`** — `0`/`1` (default: 0) — Forces the grouped MoE expert kernels back to the old one-warp-per-row kernels instead of the new two-rows-per-block ones. Gate: all GPUs (CUDA/HIP/SYCL), no gate. Best for: leave at default; A/B arm for grouped-kernel benchmarks. Tags: [ab]

**`STRATA_ONE_TOKEN_COMMIT`** — integer (atoi; default: on for CUDA, opt-in for HIP, hard-off in the SYCL port) — Enables the one-token self-commit path in verify; the HIP build requires `=1` to opt in and the SYCL port always returns false. Gate: NVIDIA CUDA default-on, AMD HIP opt-in, Intel SYCL disabled. Best for: leave at default; `=1` to test on AMD. Tags: [ab]

**`STRATA_OWNED_PRICE`** — `exact` (default: unset) — Prices the prompt path's own buffers by their real allocation (`Prefill::bytes_needed_owned`) instead of 0.1.39's rule (160 + chunk×680/1024 MiB) when sizing the expert cache, giving the cache the difference. Gate: cache-sizing code, all GPUs, no gate. Best for: try `=exact` when the auto cache sizing over-reserves. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_PARTIAL_PIN_GIB`** — float GiB (default: 24) — Caps how many GiB `STRATA_PARTIAL_PIN` registers with the driver; over-registering left the driver unable to page-lock the prompt path's buffers afterwards. Gate: only read when `STRATA_PARTIAL_PIN` is on; all GPUs. Best for: tune down if driver page-locking fails. Tags: [tunable] (docs: docs/UNSLOTH_Q4.md)

**`STRATA_PA_FAST`** — `0`/`1` (default: 0) — Runs the WMMA prompt-attention kernel in its "fast" variant (single FP16 q and p) once the query count reaches `STRATA_PF_SWITCH_MIN_T`. Gate: WMMA prompt-attention path (CUDA/HIP/SYCL); documented for AMD Strix/RDNA. Best for: set `=1` on Strix/RDMA per the STRIX_HALO recipe. Tags: [tunable] (docs: docs/STRIX_HALO.md)

**`STRATA_PCIE_FRAC_DEFAULT`** — `old` (default: unset = new measured share) — `=old` keeps the 0.1.41 PCIe-share rule; the new default uses measured base shares (0.2 / 0.55 native / 0.25 NVFP4) that PcieAuto refines. Gate: CUDA on Linux by default, HIP only on gfx1200/gfx1201 (RDNA4); unused with a layer split, `--batch`, a peer GPU, all experts in VRAM, or SYCL. Best for: opt out of the new share default when it measures slower on your machine. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_PCIE_MIN1`** — `0`/`1` (default: off, opt-in) — when a layer has >= 2 expert misses, sends at least one of them over PCIe (the computed share rounds to 0 at 1-2 misses, idling the lane); it only flies if the blob is pinned, otherwise it stays a CPU miss. Gate: shared CUDA/HIP PCIe expert streaming. Best for: try on PCIe-limited boxes (RTX 3090: prompt read +5-7% at 56K-117K tokens). Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_PEER_DIRECT`** — `0`/`1` (default: on) — launches peer-GPU expert products directly into the caller's output buffer when one is supplied, instead of the staged row path. Gate: multi-GPU peer experts, no vendor gate. Best for: leave at default; `=0` forces the staged path for A/B. Tags: [internal]

**`STRATA_PEER_HOT`** — float fraction (default: 0.45 when a peer device exists) — Multi-GPU: gives the peer card a share f of the hot expert pairs by moving every f-th of the top-ranked pairs past the primary's slot count, so both cards compute routed experts every layer. Gate: multi-GPU only (`--peer-device` ≥ 1), CUDA/HIP/SYCL. Best for: tune 0.3–0.6 on a two-card rig; `=0` disables. Tags: [tunable]

**`STRATA_PEER_HOT_AT`** — integer rank (default: 8700) — How many top-ranked expert pairs `STRATA_PEER_HOT` redistributes (≈ the primary's resident slot count). Gate: multi-GPU only, read together with `STRATA_PEER_HOT`. Best for: leave at default unless the profile's hot set differs from the primary's capacity. Tags: [tunable]

**`STRATA_PEER_NO_P2P`** — `0`/`1` (default: off) — test knob that pretends the GPU pair has no P2P access (overrides `cudaDeviceCanAccessPeer`), forcing peer expert activations/results through copies. Gate: NVIDIA multi-GPU peer experts. Best for: exercising the no-P2P fallback path in tests. Tags: [debug]

**`STRATA_PF_FUSED`** — `0`/`1` (default: on for NVIDIA sm_80+ with Q2_0 packs; opt-in `=1` on HIP; gfx12 defaults on for IQ3_S/IQ3_XXS only) — Runs the prompt-path MoE experts on fused int8 matrix-core kernels instead of MMQ; `=0` keeps MMQ, `=1` opts in the native IQ packs (and K-quant packs with `STRATA_PF_FUSED_KQ`). Gate: NVIDIA compute capability ≥ 80; AMD gfx11/gfx12 WMMA kernels (gfx12 IQ3 default-on); SYCL mirrors the CUDA logic. Best for: leave at default; `=1` for native-pack fused prompt experts, `=0` for byte-identical old bits. Tags: [tunable] (docs: docs/DETAILS.md, docs/AMD_HIP.md, docs/STRIX_HALO.md, docs/MULTI_GPU.md)

**`STRATA_PF_FUSED_NATIVE`** — `0`/`1` (default: on) — `=0` keeps the native IQ packs on MMQ even when `STRATA_PF_FUSED=1` is set, isolating the fused-native-kernel arm. Gate: follows `STRATA_PF_FUSED` (NVIDIA sm_80+, AMD gfx11/gfx12). Best for: leave at default; A/B arm. Tags: [ab] (docs: docs/AMD_HIP.md)

**`STRATA_PF_FUSED_TILE`** — `64`/`128` (default: auto) — Forces the fused IQ prompt-expert GEMM tile width (128 rows when a expert averages 56–112 rows, else 64); other values are ignored. Gate: fused IQ kernels (NVIDIA sm_80+, AMD gfx11/gfx12). Best for: leave at default; tuning knob for fused-kernel benchmarks. Tags: [tunable]

**`STRATA_PF_HCDOWN`** — `0`/`1` (default: 0) — Fuses the hyper-connection read's down and inject projections into one WMMA GEMM over padded bf16 activations on chunks of `STRATA_PF_SWITCH_MIN_T`+ tokens; rounding-level difference from hipBLASLt (quality-gated). Gate: WMMA path (HIP-focused S23 recipe; code compiled for CUDA/HIP/SYCL). Best for: leave off; Strix tuning. Tags: [tunable]

**`STRATA_PF_PEER_COMPACT`** — integer (atoi; default: on) — Uses the compact layer-sized peer-prefill buffer layout (no row cap) instead of the layer-sized path that copies over P2P. Gate: multi-GPU peer prefill only (`--peer-device`); CUDA/HIP/SYCL. Best for: leave at default; A/B arm. Tags: [ab]

**`STRATA_PF_PEER_F16`** — `0`/`1` (default: on; needs N % 8 == 0) — keeps the peer-prefill accumulation sums in FP16 (half the buffer) instead of FP32. Gate: multi-GPU prefill peer path without P2P (compact sums), no vendor gate. Best for: leave at default; `=0` forces FP32 sums. Tags: [internal]

**`STRATA_PF_PEER_GATHER`** — `0`/`1` (default: on) — builds the gather lists (tokens, starts, rows) for the peer-prefill sums path so results are gathered per expert. Gate: multi-GPU prefill peer path (sums mode), no vendor gate. Best for: leave at default. Tags: [internal]

**`STRATA_PF_PEER_OUT_PIPE`** — integer (atoi; default: on) — Pipes the peer prefill's output copies on a dedicated stream per group; `=0` does one copy after the last group. Compact mode always pipes. Gate: multi-GPU peer prefill only. Best for: leave at default; A/B arm. Tags: [ab]

**`STRATA_PF_PEER_RING`** — integer (atoi; default: 48) — Number of staging slots in the peer-prefill expert streaming ring (used when `STRATA_PF_PEER_STREAM` share > 0). Gate: multi-GPU peer prefill only. Best for: leave at default; tuning knob. Tags: [internal]

**`STRATA_PF_PEER_STREAM`** — float fraction (default: 0.35) — Share of the peer prefill's experts streamed over PCIe through the staging ring; 0 disables streaming. Gate: multi-GPU peer prefill only, chunks ≥ the stream-all minimum. Best for: tune 0.25–0.5 on a two-card rig. Tags: [tunable]

**`STRATA_PF_PEER_SUMS`** — `0`/`1` (default: on) — enables the accumulation-sums buffers in the compact peer-prefill path when there is no P2P. Gate: multi-GPU prefill peer path without P2P, compact mode, no vendor gate. Best for: leave at default. Tags: [internal]

**`STRATA_PF_PEER_XSTREAM`** — `0`/`1` (default: on) — runs the host-route peer-prefill copies on their own non-blocking stream (overlapping with compute); `=0` puts them on the compute stream. Gate: multi-GPU prefill peer path without P2P, no vendor gate. Best for: leave at default; `=0` to serialize copies for A/B. Tags: [ab]

**`STRATA_PF_STEP_SYNC`** — `0`/`1` (default: 0) — Diagnosis: syncs the compute and copy streams after each named prefill step and logs any step over 250 ms, so a hang is attributed to a step; slower. Gate: all GPUs (CUDA/HIP/SYCL), no gate. Best for: set `=1` to find which prefill step hangs, then remove. Tags: [debug] (docs: docs/TROUBLESHOOTING.md)

**`STRATA_PF_SWITCH_MIN_T`** — integer token count (default: 0) — Gates the rounding-level prompt-path switches (PF_GEMM, PF_PAD, PA_FAST, HC fusions) to chunks of N+ tokens so short prompts keep default numerics. Gate: mainly the AMD HIP WMMA paths; read in CUDA/HIP/SYCL. Best for: set `=4096` with the Strix fast recipe. Tags: [tunable] (docs: docs/STRIX_HALO.md, docs/AMD_HIP.md)

**`STRATA_PIN_BY_NEED_SKIP`** — integer N (default: computed from the GPU expert-cache capacity) — forces the count R of top-ranked pairs taken as GPU-resident (skipped by pin-by-need) instead of estimating it from the expert-cache size. Gate: same as STRATA_PIN_BY_NEED. Best for: tuning the resident/pin boundary. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_PIN_GUARD`** — `0`/`1`/`2` (default: 1) — controls the Linux page-lock guard that checks free memory (MemFree, cgroup-aware host_available) before pinning expert copies; `=0` turns the check off, `=2` page-locks the complement in paced steps with each step re-checked. Gate: Linux only (pinning path). Best for: leave at default; `=0` if the guard blocks pinning on a machine that can actually take it. Tags: [tunable] (docs: docs/TROUBLESHOOTING.md, docs/DETAILS.md)

**`STRATA_PIN_RESERVE_GIB`** — number of GiB (default: the resident headroom, at least 2 GiB; docs cite 4 GiB) — sets how much RAM must stay free during each step of the paced page-locking of expert copies (the guard's reserve). Gate: Linux pinning path. Best for: raise it if page-locking pushes the host toward OOM; leave at default otherwise. Tags: [tunable] (docs: docs/TROUBLESHOOTING.md, docs/DETAILS.md)

**`STRATA_PIPELINE_ADAPT_ASYNC`** — `0`/`1` (default: on) — with `=0` the adaptive tier in the pipelined multi-window loop becomes the blocking tier (waits for in-flight windows) instead of running async. Gate: multi-GPU serve with `--adapt-async 1`; off beside `--pipeline-windows 2` in arena mode. Best for: set `=0` to A/B the blocking adaptive tier. Tags: [ab] (docs: docs/MULTI_GPU.md, docs/DETAILS.md)

**`STRATA_PIPELINE_AUTO`** — `0`/`1` (default: on for HIP on gfx1200/gfx1201; NVIDIA builds keep it opt-in) — auto-enables pipelined decode windows on a multi-GPU serve when the split has at least the minimum stage count (two stages pay off only with the drafter on the first card). Gate: AMD HIP (RDNA4 decode defaults) by default; CUDA opt-in. Best for: `=0` turns the auto pipeline off; `--pipeline-windows` overrides it. Tags: [tunable] (docs: docs/MULTI_GPU.md)

**`STRATA_PIPELINE_AUTO_MIN_STAGES`** — integer >= 2 (default: 2 when the drafter sits on the first card, else 3) — sets the minimum number of split stages before the auto pipeline engages. Gate: multi-GPU auto pipeline (same as STRATA_PIPELINE_AUTO). Best for: force 2 stages on a two-card split for A/B. Tags: [tunable] (docs: docs/MULTI_GPU.md)

**`STRATA_PIPELINE_DEBUG`** — any non-`0` value (default: off) — Master gate: every other `STRATA_PIPELINE_*` test/tuning variable (THETA, FORCE_MISS, SWITCH, LOG, TRACE, …) is read only when this is on; otherwise defaults apply. Gate: pipeline-windows scheduling code, all GPUs. Best for: leave off; pipeline experiments only. Tags: [debug] (docs: docs/MULTI_GPU.md)

**`STRATA_PLE_AUTO_MMAP`** — integer (atoi; default: 0) — Restores the 0.1.42 behavior of auto-switching a slow `--ple-io direct` probe to mmap; by default the probe only warns and keeps direct. Gate: host PLE table IO, all GPUs. Best for: leave off (the bar is NVMe-calibrated and mmap was slower on SATA/Windows). Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_PLE_BATCH`** — integer (atoi; default: on) — Batches the PLE block's key/value projections over the chunk/verify-window rows (multi-row kernels, weights read once); `=0` keeps the per-token block. Gate: all GPUs, requires PLE native postops/BF16 keys. Best for: leave at default; A/B arm. Tags: [ab] (docs: docs/STRIX_HALO.md)

**`STRATA_PLE_PROBE`** — any value (default: unset) — Skips the startup probe that measures PLE direct-read rows/s (and its slow-read warning). Gate: host PLE table IO, all GPUs. Best for: set to skip the probe on slow starts. Tags: [debug] (docs: docs/DETAILS.md)

**`STRATA_POOL_TRACE`** — any value (default: unset) — Prints a per-layer "pool trace" line for each expert-pool stage transition to stderr. Gate: all GPUs (CUDA/HIP/SYCL), no gate. Best for: set only when debugging expert-pool stalls. Tags: [debug]

**`STRATA_PREFILL_BF16X2`** — integer (atoi; default: 0) — Splits prompt-path activations into BF16 high+low parts for extra precision: 1 includes the hyper-connection's (slower), 2 all but the hyper-connection's; changes the prompt path's numbers, so opt-in. Ignored on FP16-IO stages. Gate: BF16 prompt path (CUDA/HIP/SYCL), no arch gate. Best for: leave at default; precision experiments. Tags: [ab]

**`STRATA_PREFILL_CPU_SHARE`** — `auto`, a fixed share `0.0`-`1.0`, or `0` (default: `auto` on single-GPU CUDA for chunks under 1,024 tokens; off on HIP, layer splits, batch) — lets the CPU pool compute a share of the experts during short prompt chunks, measuring per-layer times to balance both sides (`auto`) at which sharing pays. Gate: NVIDIA CUDA single GPU, serve without `--batch`; skipped by default when RAM is short. Best for: `=0` restores 0.1.40.3's exact bytes; `=auto`/a number forces the share on. Tags: [tunable] (docs: docs/DETAILS.md, docs/OLDER_GPUS.md)

**`STRATA_PREFILL_DUMP_R`** — file path (default: unset) — Debug: appends the final residual row of every 64th token (position + fp32 row) to the named binary file, for A/B quality of a prompt path. Gate: all GPUs, no gate. Best for: set only for quality comparisons. Tags: [debug]

**`STRATA_PREFILL_DUMP_R_ALL`** — file path (default: unset) — Appends every position's final multi-stream residual as BF16 rows (position order, appended across chunks) for draft-layer distillation data. Gate: all GPUs, no gate. Best for: set only when collecting distillation data. Tags: [debug]

**`STRATA_PREFILL_EQUAL`** — `0`/`1` (default: 0) — Reads a prompt segment in chunks of equal size (ceil(tokens/n) rounded to 256) instead of full chunks plus a short last one; changes the rounding, so off by default. Gate: all GPUs, chunk-choice code (serve lend, generate prompt). Best for: try `=1` on lend/split setups where chunk count drives cost. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_PREFILL_HELP`** — integer (atoi; default: 0) — Layer-split multi-GPU: the idle stage's GPU lends a share of a one-chunk prompt's streamed experts (computed on its MMQ path); rounds differently from the default, so opt-in. Gate: layer-split (`--layer-split`) with native packs on the MMQ prompt path; not with fused prompt kernels or `--peer-device`. Best for: set `=1` on a two-card layer-split rig with short prompts. Tags: [tunable] (docs: docs/MULTI_GPU.md)

**`STRATA_PREFILL_HELPER_SOURCE`** — `0`/`1` (default: off) — makes the prompt path read the experts a helper cache (remote expert cache on another GPU) already holds from that cache, via a RemotePrefillSource, instead of the local source. Gate: single GPU with helper caches, no layer split, no batch slots. Best for: enable when helper caches are populated and prompts re-read the same experts. Tags: [tunable]

**`STRATA_PREFILL_HELP_FRAC`** — float 0–1 (default: auto, ~0.45 falling to 0 by ~3.3K tokens) — Fixes the idle-card help share for every prompt length instead of the measured T-dependent rule. Gate: only with `STRATA_PREFILL_HELP=1` (layer-split multi-GPU). Best for: tune the share on a two-card rig. Tags: [tunable] (docs: docs/MULTI_GPU.md)

**`STRATA_PREFILL_ISSUER`** — integer (atoi; default: on) — Runs the expert stream's copies on a dedicated host issuer thread so the kernel-launching thread never waits behind an unpinned host copy; `=0` issues inline. Gate: all GPUs, streaming prompt path. Best for: leave at default; A/B arm. Tags: [ab]

**`STRATA_PREFILL_LEND_PCT`** — integer percent (atoi; default: 90 with ≥90% pinned copies, else 85) — Caps how much of a card's expert slots the auto planner lends to the prompt path. Gate: multi-GPU/lend planning, all GPUs. Best for: leave at default; tuning knob for lend sizing. Tags: [tunable] (docs: docs/MULTI_GPU.md)

**`STRATA_PREFILL_STREAM_AHEAD`** — `0`/`1` (default: on) — streams expert uploads ahead of each layer's routing so the copy engine keeps working through the attention halves; `=0` keeps the previous routed-only upload schedule. Gate: shared CUDA/HIP prefill streaming, no vendor gate. Best for: leave at default; `=0` is the A/B arm. Tags: [ab]

**`STRATA_PREFILL_TIMING`** — any value (default: unset) — Prints a per-phase GPU timeline of the prompt path (embed, attention, gemms, gather, wait copy, …), including the peer card's timeline on multi-GPU. Gate: all GPUs (CUDA/HIP/SYCL). Best for: set to find the prompt bottleneck. Tags: [debug] (docs: docs/OLDER_GPUS.md, docs/AMD_HIP.md)

**`STRATA_PROMPT_ATTN_V1`** — any value (default: unset) — Debug control: runs the first version of the int8-K prompt attention (same accuracy, different summation order) to measure how far the model amplifies an FP32-level change; Turing always takes v1 (no cp.async before sm_80). Gate: NVIDIA int8-KV path (CUDA/HIP/SYCL). Best for: set only for numerical-control experiments. Tags: [debug]

**`STRATA_Q36_BLAS_F32`** — any value (presence check; default: off) — runs the strata-q36 engine's FP32 GEMMs (router, GDN alpha/beta) through cuBLAS/hipBLAS SGEMM instead of the built-in deterministic split-K kernel. Gate: strata-q36 engine (CUDA/HIP), no arch gate. Best for: A/B vendor BLAS vs the custom kernel. Tags: [ab]

**`STRATA_Q36_FIXED_SLOTS`** — any value (presence check; default: off) — disables the expert-slot rebalancing (slots keep their original placement) so runs are deterministic for tests. Gate: strata-q36 hybrid mode. Best for: tests and byte-comparison runs. Tags: [debug]

**`STRATA_Q36_KEEP_PACKED`** — any value (presence check; default: off) — keeps the packed Q8_0 copies registered in VRAM even when freeing them would let the full experts fit, skipping the un-pack step. Gate: strata-q36 engine VRAM budgeting. Best for: A/B the packed-vs-full placement decision. Tags: [ab]

**`STRATA_Q36_NO_MMQ`** — any value (presence check; default: off) — turns off the MMQ prompt kernels in the strata-q36 engine, forcing the FP16 dequant prompt path. Gate: strata-q36 engine. Best for: A/B prompt path quality/speed. Tags: [ab]

**`STRATA_Q36_NO_MTP_CPU`** — `1` (presence check; default: off) — refuses the mode that keeps the MTP block's experts on the CPU (with a shrunk attention buffer and small reserve), using the plain hybrid mode for every layer. Gate: strata-q36 engine with an MTP block. Best for: disable the tight-VRAM MTP-to-CPU trick when it misbehaves. Tags: [tunable]

**`STRATA_Q36_PLACE_DEBUG`** — any value (presence check; default: off) — prints the chunk-rule's placement numbers (free GiB with packed copies, main experts GiB) to stderr. Gate: strata-q36 engine. Best for: diagnosing why a chunk size was chosen. Tags: [debug]

**`STRATA_Q36_STREAM_ALL`** — `1` (presence check; default: off) — streams the whole expert layer from host memory over PCIe instead of taking VRAM-slotted experts device-to-device (the first, PCIe-bound version). Gate: strata-q36 hybrid streaming. Best for: A/B the streaming strategy on small cards. Tags: [ab]

**`STRATA_Q6P_GRID`** — integer (atoi; default: 640) — Grid size (persistent workgroups) for the persistent Q6_K MMVQ kernel. Gate: all GPUs (CUDA/HIP/SYCL), no gate. Best for: leave at default; tuning knob for Q6_K decode. Tags: [tunable]

**`STRATA_Q6P_SELFTEST`** — `1`/`2` (default: off) — With `STRATA_Q6_PACKED=1`, runs a bitwise self-check of the packed Q6_K kernel against the GGUF-layout kernels (1–4 columns) plus paired timings; `2` adds the grid sweep. Gate: all GPUs (CUDA/HIP/SYCL), no gate. Best for: set only when validating the packed path. Tags: [debug]

**`STRATA_Q6_PACKED`** — `1` (exact string; default: off) — Packs Q6_K weights into a kernel-friendly layout (extra memory) so decode MMVQ runs the packed/persistent kernels. Gate: all GPUs (CUDA/HIP/SYCL), shape-eligibility only (n_in%256, n_out≥2048). Best for: Strix decode recipe knob. Tags: [tunable] (docs: docs/STRIX_HALO.md)

**`STRATA_Q8_PACKED`** — `1` (exact string; default: off) — Packs Q8_0 weights into a persistent-kernel layout for decode MMVQ (shape-eligibility gated). Gate: all GPUs (CUDA/HIP/SYCL). Best for: Strix decode recipe knob. Tags: [tunable] (docs: docs/STRIX_HALO.md)

**`STRATA_QFUSE_HC`** — `0`/`1` (default: on only on gfx12 under HIP) — restricts the fused q8_1 activation images to the hyper-connection read's consumer (not the GDN recurrence). Gate: AMD gfx12 by default (the CUDA branch is not visible in context). Best for: force off/on to A/B the fusion on other cards. Tags: [ab] (inferred from code)

**`STRATA_QFUSE_HC_LOCAL`** — `0`/`1` (default: on) — under STRATA_QFUSE_HC picks the 32-column local up-projection kernel; `=0` puts the atomic-tail up kernel back. Gate: shared CUDA/HIP fused_gr path, applies where QFUSE_HC is on. Best for: A/B the up-kernel variant. Tags: [ab]

**`STRATA_QSA_CLUSTER`** — integer (atoi; default: on) — Uses the thread-block-cluster top-k kernel (CL_N CTAs per query) for small-query QSA selection; `=0` falls back to one-CTA kernels. Gate: NVIDIA sm_90+ only (compiled out on HIP). Best for: leave at default; `=0` to turn the decode kernel off. Tags: [ab] (docs: docs/DETAILS.md)

**`STRATA_QSA_DUMP`** — file path (default: unset) — Appends every QSA layer's selected cells for the prompt's last positions (records of layer/pos0/T/cap + cells) for tools/qsa_locality.py locality studies. Gate: all GPUs, prompt path. Best for: set for sparse-attention research. Tags: [debug]

**`STRATA_QSA_DUMP_LAST`** — integer positions (atoll; default: 4096) — How many trailing prompt positions `STRATA_QSA_DUMP` records. Gate: only with `STRATA_QSA_DUMP` set. Best for: tune dump size. Tags: [internal]

**`STRATA_QSA_EARLY_EXIT`** — integer (atoi; default: on) — Lets QSA block-score CTAs exit early when past a query's block count; `=0` keeps the old staging (every CTA loads the queries first), same score bits. Gate: multi-query block-score kernel (CUDA/HIP/SYCL). Best for: leave at default; A/B arm. Tags: [ab]

**`STRATA_QSA_WARP`** — `1`/`select`/`attn` (default: unset) — A/B arm forcing the pre-sm_80 warp kernels for QSA block selection (`select`) or prompt attention (`attn`) on any card, as RTX 20 runs them. Gate: NVIDIA CUDA (cc override); the SYCL port reads it but its tensor path is XMX-based. Best for: leave unset; older-GPU parity testing. Tags: [ab]

**`STRATA_READ_AHEAD`** — `0`/`1` (default: on) — asks the OS to read ahead mmap'd model files in 128 KiB steps (madvise WILLNEED / posix_fadvise) so sequential loader reads find pages cached. Gate: Linux/POSIX (macOS has no posix_fadvise; not Windows). Best for: `=0` turns the read-ahead off if it thrashes the page cache. Tags: [tunable] (docs: docs/TROUBLESHOOTING.md, docs/DETAILS.md)

**`STRATA_REFILL_BLOCKING`** — any value (default: unset) — Makes the lent-slot refill run blocking (issue+wait per part) instead of the pipelined form. Gate: multi-GPU lend refill, all GPUs. Best for: leave unset; debugging arm. Tags: [debug]

**`STRATA_REMOTE_SPIN`** — `0` to disable (default: on) — Sets the remote-expert helper device's CUDA context to spin-wait scheduling (faster wake-up on the CPU pool's critical path); `=0` keeps the driver default. Gate: NVIDIA CUDA only (HIP keeps its default; SYCL has no equivalent — DPCT drop). Best for: leave at default on CUDA helper cards. Tags: [tunable]

**`STRATA_REMOTE_ZEROCOPY`** — `0` to disable (default: on) — The remote-expert helper reads input and writes compact rows directly in the pinned host buffers (mapped device pointers), saving two PCIe copies per layer; `=0` copies. Gate: remote-expert helper path, CUDA/HIP/SYCL (SYCL uses direct host pointers). Best for: leave at default; `=0` if mapped pointers misbehave. Tags: [tunable]

**`STRATA_RESIDENT_HEADROOM_GIB`** — float GiB (atof; default: 4) — How much available RAM the resident-expert mode leaves free when sizing its pinned budget. Gate: host RAM sizing for `--resident-experts`/`--resident-budget-gib`, all GPUs. Best for: lower a little if the resident set is short; raise if the machine swaps. Tags: [tunable] (docs: docs/DETAILS.md, docs/AMD_HIP.md, docs/TROUBLESHOOTING.md)

**`STRATA_RESIDENT_PIN`** — `0` (default: pinned) — Keeps the resident expert copy pageable instead of page-locked (the `--resident-cpu-experts` form; the CPU then computes every miss). Gate: host RAM resident mode, all GPUs. Best for: set `=0` when pinning starves the OS. Tags: [tunable] (docs: docs/TROUBLESHOOTING.md, docs/DETAILS.md, docs/UNSLOTH_Q4.md)

**`STRATA_RING_BYTES`** — `0` to disable (default: on) — Sizes the prompt ring by bytes (fewer ring slots, more cache slots, larger auto chunk) instead of 0.1.39's fixed ring/loan/chunk choice; changes a long prompt's bits. Gate: native-pack prompt path, all GPUs. Best for: leave at default; `=0` to restore 0.1.39 behavior. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_ROPE_TABLE`** — `0`/`1` (default: 0) — Uses the precomputed rope cos/sin table in the native rope kernels; its angles differ from fast-math in the last bits, so outputs move. Gate: native rope kernels (CUDA/HIP/SYCL), no arch gate. Best for: leave at default; parity experiments. Tags: [ab]

**`STRATA_ROUTE_PER_TOKEN`** — any value (default: unset) — Forces verify-window MoE routing to the per-token `moe_route` path instead of the batched multi-column BF16 projection + single top-k. Gate: batched routing requires native BF16 projections (CUDA/HIP/SYCL); documented for AMD. Best for: leave unset; A/B arm for window routing. Tags: [ab] (docs: docs/AMD_HIP.md)

**`STRATA_ROUTE_PROJ_PER_TOKEN`** — any value (default: unset) — Inside the batched routing path, runs the router projection as one GEMV per token instead of the multi-column kernel. Gate: batched routing path, all GPUs. Best for: leave unset; kernel A/B arm. Tags: [ab]

**`STRATA_ROUTE_RESIDENT`** — float margin (default: unset = off) — after the router, biases the chosen ids/weights toward GPU-resident experts within a margin, before the plan reads them; changes the output. Gate: MoE with 512 experts, top-10 routing, single GPU. Best for: A/B residency-biased routing (see STRATA_ROUTE_RESIDENT_RANKS). Tags: [ab] (docs: docs/DETAILS.md)

**`STRATA_ROUTE_RESIDENT_MTP`** — `1` (default: off; experimental, needs STRATA_ROUTE_RESIDENT=<margin>) — gives the MTP draft's router the same residency bias, using the last layer's residency row as its table. Gate: single GPU with MTP and no pipeline stages. Best for: A/B (docs say it is not faster than plain STRATA_ROUTE_RESIDENT). Tags: [ab] (docs: docs/DETAILS.md)

**`STRATA_ROUTE_RESIDENT_RANKS`** — `lo-hi` (default: 6-9) — sets which routing ranks the residency bias may move (used by STRATA_ROUTE_RESIDENT and the MTP variant). Gate: same as STRATA_ROUTE_RESIDENT. Best for: tuning the bias's rank window. Tags: [tunable]

**`STRATA_ROUTE_TAIL_SKIP`** — `0`-`9` (default: 7 on CUDA builds and HIP gfx1200/gfx1201; `0` = off) — a missed expert that every token of a verify window routes at rank N or lower is neither copied nor computed (answers differ slightly from 0.1.41, KL ~0.003-0.006; `=0` restores the old bytes). Gate: CUDA by default, HIP RDNA4; off with `--batch` slots, a peer, or a helper GPU; no default when every expert is in VRAM or nearly all are cached. Best for: the main decode-speed knob on cache-limited cards (RTX 5070: +47-55% decode); `=0` for byte-exact comparison. Tags: [tunable] (docs: docs/DETAILS.md, docs/QUANTS.md, docs/OLDER_GPUS.md, docs/NVFP4.md)

**`STRATA_RSS_TRACE`** — any value (default: unset) — Prints the process's mapped-file RSS (the arena) at each startup step from /proc/self/status. Gate: Linux host code (CUDA/HIP/SYCL), not GPU-specific. Best for: set when debugging arena page faults. Tags: [debug]

**`STRATA_S2_SWIGLU_Q8`** — `0` to disable (default: on) — Fuses SwiGLU and the scaled Q8_0 quantize of its result into one kernel in the grouped S2 experts; `=0` keeps the two launches. Gate: grouped S2 expert path (CUDA/HIP/SYCL), fast path with x_scales. Best for: leave at default; A/B arm. Tags: [ab]

**`STRATA_SCORES_MULTI`** — integer (atoi; default: on) — Uses the multi-query QSA block-scores kernel for small decode batches; `=0` falls back to the per-query grid. Gate: QSA select path (CUDA/HIP/SYCL). Best for: leave at default; A/B arm. Tags: [ab]

**`STRATA_SEL_OVERLAP`** — any value (default: unset) — Debug: measures how much neighbouring queries' QSA selections overlap (union of selected cells vs sum of widths) for the first QSA layer. Gate: prompt path, all GPUs. Best for: set only for sparse-attention research. Tags: [debug]

**`STRATA_SESSION_BUFFERED`** — `1` (default: off) — forces session save/restore files through buffered I/O instead of O_DIRECT (the path taken where direct I/O is refused). Gate: filesystem layer (Windows always uses buffered I/O). Best for: set when direct I/O fails on the session disk. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_SESSION_GOLDEN_PRINT`** — any value (presence check; default: off) — test-only: prints the golden session file's size and hash so the fixed sample's bytes can be checked for drift. Gate: session-file test code. Best for: running the conversation-file test. Tags: [debug]

**`STRATA_SH_FORK_LATE`** — `0` to disable (default: on) — With CPU experts in the window, forks the shared expert's side stream only after the doorbell is published (so it stops competing for the GPU's first microseconds); `=0` forks at the top. Gate: verify path with shared-expert stream (CUDA/HIP/SYCL). Best for: leave at default; A/B arm. Tags: [ab]

**`STRATA_SH_STREAM`** — `1`/`0` (default: on for CUDA, off for HIP except the gfx1151 arch table which sets 1) — Runs the shared expert on a second stream beside the routed experts in verify; it slows some AMD cards (gfx1030) and pays on gfx1151 (+1.8–6.7% decode). Gate: default on NVIDIA CUDA, default off AMD HIP; output identical either way. Best for: try `=1` on Linux RX 6000; gfx1151 gets it automatically. Tags: [tunable] (docs: docs/STRIX_HALO.md, docs/AMD_HIP.md, docs/DETAILS.md)

**`STRATA_SNAPSHOT_FULL_CAPTURE`** — any value (default: unset) — Diagnostic: skips the conversation-KV retention step at snapshot restore, for full-capture comparisons. Gate: serve snapshot code, all GPUs. Best for: set only for diagnostics. Tags: [debug] (docs: docs/DETAILS.md)

**`STRATA_SNAPSHOT_VERIFY`** — any value (default: unset) — Logs the draft-prefill path/mode/cells and verifies the restored draft KV hash after a snapshot restore. Gate: serve snapshot code with MTP, all GPUs. Best for: set when debugging snapshot restore. Tags: [debug]

**`STRATA_SPEC_COUPLED`** — any non-`0` value (default: off) — For sampled requests, the draft layer samples with the target's own chain and random draw instead of taking its most likely token (acceptance rises; speed gain is not universal). Gate: spec/MTP drafting, all GPUs; greedy requests untouched. Best for: try with `STRATA_SPEC_GUMBEL=1` on sampling rigs. Tags: [tunable] (docs: docs/DETAILS.md, docs/MULTI_GPU.md)

**`STRATA_SPEC_DEPTH`** — any value (presence check; default: off) — at serve shutdown prints speculative-draft acceptance tested/accepted counts per draft depth (depths 1-5) for each request. Gate: MTP/speculative decode path, no vendor gate. Best for: measuring draft acceptance by depth. Tags: [debug] (docs: docs/DETAILS.md)

**`STRATA_SPEC_MIN_TEMP`** — float x (default: 0 = every sampled request) — limits coupled/probabilistic sampled drafting to requests with temperature >= x; below x the drafter keeps argmax drafts (sampled drafting gains at 1.0, loses below ~0.7). Gate: sampled-drafting path (SPEC_PROB / SPEC_COUPLED). Best for: set ~0.9 to apply sampled drafting only to hot requests. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_SPEC_PROB`** — `1` (default: off) — speculative rejection sampling: the draft layer samples from its own distribution q and the check accepts with probability min(1, p/q), so every token is distributed exactly as without drafts (also switches on the coupled drafter's machinery). Gate: MTP drafter with sampled requests (temperature > 0), no vendor gate. Best for: distribution-exact sampled drafting; measured not consistently faster. Tags: [ab] (docs: docs/DETAILS.md, docs/MULTI_GPU.md)

**`STRATA_SPEC_PROB_DT`** — float x > 0 (default: 1) — runs the drafter's sampling chain at x times the request's temperature, changing the acceptance sum min(p, q) of rejection sampling. Gate: STRATA_SPEC_PROB path. Best for: tuning draft distribution sharpness. Tags: [tunable]

**`STRATA_SPLIT_COVER_B`** — float exponent (atof; default: 3.0, curve used at ≥4 GPUs) — Sets the exponent of the coverage curve the layer-split auto planner uses to price routed-mass coverage; setting it also forces the curve at any GPU count. Gate: multi-GPU layer-split planning. Best for: tune for a model whose routing is sharper/flatter than the fitted rig. Tags: [tunable]

**`STRATA_SPLIT_LINK_ABS`** — `1` (default: off) — scores a layer split's per-stage miss cost with 0.1.41's absolute 20 GB/s link rule instead of the relative-to-fastest-stage scale (which only kicks in from a 4x link gap). Gate: multi-GPU layer-split auto placement. Best for: A/B the split cost model on mixed rigs. Tags: [ab]

**`STRATA_SPLIT_MISS_MS`** — float ms (atof; default: 190) — Scales the per-miss cost the layer-split auto placement search feeds its coverage curve. Gate: multi-GPU layer-split planning. Best for: tune on rigs whose links are much slower/faster than the fitted one. Tags: [tunable]

**`STRATA_SPLIT_MTP_BATCH`** — `1` (default: off) — on a layer split, fills the draft layer's prompt K/V in batches on the last stage's Prefill (a few large MMQ products per chunk) instead of the drafter's own per-8-row pass; drafts may change, the target's tokens never do. Gate: multi-GPU layer split with `--mtp`. Best for: A/B prompt-time draft K/V on splits. Tags: [ab] (docs: docs/MULTI_GPU.md)

**`STRATA_SPLIT_NEAR_TIE_PCT`** — float percent (default: 4.0) — widens/narrows the near-tie window in auto layer-split choice: priced splits within this percent of the best are re-ranked by boundary distance from the chosen one. Gate: multi-GPU layer-split auto on mixed rigs with <= 3 stages. Best for: leave at default; tune when auto split boundaries look off. Tags: [tunable]

**`STRATA_SPLIT_OWN`** — `1`/`auto`/`0` (default: unset = borrow) — Layer split: `=1` gives every card its own prompt buffers (output differs from the borrow default), `=auto` does it only where the buffers are ≤12% of each card's VRAM. Gate: multi-GPU layer split (`--layer-split`), all GPUs. Best for: try `=1` for prompt speed on roomy cards. Tags: [tunable] (docs: docs/MULTI_GPU.md)

**`STRATA_SPLIT_OWN_BUFFERS`** — `0` to disable (default: on) — In a layer split, a stage whose cache holds every expert of its layers keeps its prompt buffers as its own instead of borrowing cache slots; `=0` makes every stage borrow (0.1.30/0.1.31). Gate: multi-GPU layer split only. Best for: leave at default; A/B arm. Tags: [ab]

**`STRATA_SPLIT_OWN_PLACE`** — `reserve` (default: unset) — Makes the layer-split placement search price stage room with the reserve rule. Gate: multi-GPU layer-split planning. Best for: leave unset; planner experiments. Tags: [internal]

**`STRATA_SPLIT_RING`** — integer slots (atoi; default: 96 when ≥75% of pairs are resident, else the pinned-share rule) — Sets the prompt path's streamed ring slots under a layer split (chosen before buffer sizing). Gate: multi-GPU layer split with resident RAM mode. Best for: set `=384` on big rigs per the MULTI_GPU notes. Tags: [tunable] (docs: docs/MULTI_GPU.md)

**`STRATA_SPLIT_SMALL_MAX`** — integer tokens (atoll; default: = SPLIT_SMALL_OWN's S) — Requests of at most M prompt tokens read in the small S-token chunks on the owning stages (nothing lent) instead of borrowing for one bigger chunk. Gate: only with `STRATA_SPLIT_SMALL_OWN` set, multi-GPU split. Best for: tune alongside SPLIT_SMALL_OWN. Tags: [tunable]

**`STRATA_SPLIT_SMALL_OWN`** — integer tokens (atoll; default: unset) — Reserves S-token chunks of each stage's own cache slots so short prompts lend, stream, and refill nothing; their experts stay non-resident (CPU/PCIe). Gate: multi-GPU layer split with resident RAM mode. Best for: set for latency-sensitive short prompts on splits. Tags: [tunable]

**`STRATA_SPLIT_TIMING`** — any value (default: unset) — Prints per-stage verify host timing per window (GPU wait, pool+plan, host staging, commit) cumulatively since start. Gate: multi-GPU split serve path. Best for: set to find split decode host bottlenecks. Tags: [debug]

**`STRATA_SSD_KEEPALIVE`** — float ms (atof, clamped 0–10000; default: 100, 0 = off) — Reads one PLE page after this long without a read, keeping SSDs that stall 50–150 ms after ~250 ms idle awake. Gate: PLE direct-IO path with its IO thread, all GPUs. Best for: set `=0` on drives that don't need it; raise on aggressive power-saving SSDs. Tags: [tunable]

**`STRATA_SSD_KEEPALIVE_WINDOW`** — float seconds (atof, clamped 1–86400; default: 60) — How long after the last PLE row request the keepalive reads continue. Gate: only with PLE direct keepalive active. Best for: leave at default. Tags: [internal]

**`STRATA_STAGER_BATCH`** — integer 1..kMaxBatch (default: 8 with the file/SSD profile, 1 otherwise) — how many neighbouring expert blobs each prompt-stager thread claims and reads as one request; `=1` is the A/B arm (one blob per request, as before). Gate: prefill stager reading GGUF files in place, no vendor gate. Best for: A/B prompt-read batching. Tags: [ab]

**`STRATA_STAGER_RING`** — integer (atoi, clamped 2–256; default: 16, 4×threads for GGUF-in-place reads) — Depth of the prompt stager's staging-buffer ring for streaming expert blobs. Gate: prompt stager, all GPUs. Best for: set `=128` with 32 threads on SSD GGUF reads. Tags: [tunable] (docs: docs/UNSLOTH_Q4.md, docs/DETAILS.md)

**`STRATA_STAGER_SLEEP`** — `1`/`0` (default: sleep on Linux, spin on Windows) — makes the prompt-staging threads wait by sleeping instead of a yield spin (Linux spinners starved a pinned host thread: DGX Spark 8K prompt 91 -> 1,222 tok/s with sleeping; Windows spinning is 1.2% faster). Output is the same either way. Gate: OS (Linux/Windows). Best for: `=1` on Windows when sharing the machine matters (whole-machine CPU 77-90% -> 21-25%); `=0` forces spinning on Linux. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_STAGER_SSD`** — `1`/`0` (default: auto, via a mincore page-residency sample) — forces the stager's SSD profile (32 threads, 128-deep ring) or the RAM profile (4 threads) when a native pack is read in place from GGUF shards (the wrong profile cost 4-7x on a warm page cache). Gate: Linux page-cache probe, in-place GGUF reads. Best for: support/A/B when auto-detection picks the wrong profile. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_STAGER_THREADS`** — integer (atoi, clamped 1–32; default: 32 for cold GGUF-in-place, else 2–4) — Host copy threads the prompt stager uses for unpinned blobs (many reads in flight on SSD page faults). Gate: prompt stager, all GPUs. Best for: leave to the auto profile; `STRATA_STAGER_SSD=1/0` forces it. Tags: [tunable] (docs: docs/DETAILS.md, docs/INTEL.md, docs/UNSLOTH_Q4.md)

**`STRATA_STAGE_PIN`** — `0`/`1` (default: unset = pin only while the host has RAM to spare; docs mark it opt-in `=1` after a release-gate corruption finding) — page-locks the expert stage buffers that feed the cache-fill copies, avoiding the driver's bounce-buffer copy. Gate: shared CUDA/HIP. Best for: leave off until the stage-recycling bug is fixed; `=0` keeps buffers pageable. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_STAGE_TRIM`** — any non-`0` value (default: off) — Layer split: each GPU loads only its own layers' dense weights instead of a full model copy, freeing VRAM for the expert cache (output can shift slightly). Gate: multi-GPU layer split, CUDA and HIP alike; explicit `--layer-split` for CUDA0. Best for: set `=1` (or `--trim-stage-weights`) on split rigs. Tags: [tunable] (docs: docs/MULTI_GPU.md, docs/BATCHING.md)

**`STRATA_STARTUP_UNBUFFERED`** — `0` to disable (default: on) — opens the experts.bin file tier unbuffered at startup (Windows), before any mmap views exist. Gate: Windows only, single experts.bin file. Best for: `=0` restores buffered startup reads. Tags: [tunable]

**`STRATA_STATE_HASH`** — any value (default: unset) — Debug: fingerprints every part of the session state over its live positions (and stale tail cells) after each finished request, for parity gates. Gate: serve path with prompt cache, all GPUs. Best for: set only for state-parity debugging. Tags: [debug]

**`STRATA_STATE_HASH_GDN`** — any value (default: unset) — Debug: prints per-GDN-layer hashes of the GDN recurrent states as the prompt path leaves them, to find which layer diverges first. Gate: all GPUs (CUDA/HIP/SYCL). Best for: set with `STRATA_STATE_HASH` for GDN divergence hunts. Tags: [debug]

**`STRATA_TEST_CACHE_FAIL`** — integer N (atoi; default: 0) — Test hook: makes the first N expert-cache opens fail like an out-of-commit cudaMalloc, exercising the shrink-retry path. Gate: cache sizing code, all GPUs. Best for: tests only. Tags: [debug]

**`STRATA_TEST_PAGEABLE`** — `1` (default: off) — Test hook: forces every pinned host allocation's pageable fallback to fail, so the LXC-memlock-limit fallback path runs under ASan/MALLOC_CHECK_. Gate: prefill/pinned-ring code, all GPUs. Best for: tests only. Tags: [debug]

**`STRATA_TEST_PINNED_SHARE`** — float fraction (default: measured from the source) — tests: overrides the pinned share of expert bytes so the Windows-like ring rule can be exercised on other hosts. Gate: prefill ring sizing, test use. Best for: tests only. Tags: [internal]

**`STRATA_TEST_SLOW_COPY_US`** — integer microseconds (default: 0 = no delay) — tests: every 1 MiB slice of a checkpoint copy takes N extra microseconds, simulating an OS that paged the checkpoint out to the drive, to exercise the watchdog's beat logic. Gate: checkpoint copy path, test use. Best for: tests only. Tags: [debug]

**`STRATA_TEST_UNMAP_GGUF`** — `1` (default: off) — tests on Linux: closes the GGUF mmap views so the engine runs the unmapped file-tier path that Windows takes. Gate: Linux, unbuffered reads active. Best for: tests only. Tags: [debug]

**`STRATA_TEST_VERIFY_STALL`** — integer N (atoll; default: 0) — Test hook: withholds the last layer's flag in the N-th verify window so the GPU spins on a flag nobody raises, exercising the bounded window wait and release. Gate: verify path, all GPUs. Best for: tests only. Tags: [debug]

**`STRATA_TG_FLUSH_US`** — integer µs (atoi; default: 2000) — Poll interval after launching the token graph, so WDDM submits it promptly and faults get noticed. Gate: token-graph session path (CUDA/HIP/SYCL), WDDM-motivated. Best for: leave at default. Tags: [internal]

**`STRATA_TOPK_OLD`** — any value (default: unset) — Runs the original QSA block top-k kernel instead of the register-keys/cluster kernels (same ids). Gate: QSA select, CUDA/HIP/SYCL. Best for: leave unset; A/B arm. Tags: [ab]

**`STRATA_TOPK_STREAM`** — `0` to disable (default: on) — On the Turing active-count prefill path above ~22,528 blocks, uses the wide coalesced top-k kernel instead of the register kernel (same ids); `=0` restores the register kernel. Gate: NVIDIA CUDA (compiled out on HIP), sm_75 band. Best for: leave at default; `=0` per TROUBLESHOOTING if the wide kernel misbehaves. Tags: [tunable] (docs: docs/TROUBLESHOOTING.md, docs/DETAILS.md)

**`STRATA_TRACE`** — `1`/`2` (default: unset) — Prints a stderr line per startup step (VRAM free), per request step, and per prompt chunk; `=2` also syncs at each mark to pin a sticky CUDA fault to the step that caused it. Gate: all GPUs (CUDA/HIP/SYCL). Best for: set `=1` to see where a request stalls, `=2` for sticky launch failures. Tags: [debug]

**`STRATA_TRACE_ADAPT`** — any value (default: unset) — Logs why each adaptive-residency round did or did not swap (whether the previous round's async expert copies had landed). Gate: serve adapt path, all GPUs. Best for: set when decode residency churn looks random. Tags: [debug]

**`STRATA_TSUM`** — `1` (default: off; set by the gfx1151 arch table) — Uses the single transposed-butterfly warp sum (S26) in the MMVQ/BF16-rows/GR/expert kernels; bitwise the same values. Gate: CUDA/HIP/SYCL kernels; enabled by default on AMD gfx1151. Best for: leave to the arch defaults. Tags: [tunable] (docs: docs/STRIX_HALO.md)

**`STRATA_UMA_HEADROOM_GIB`** — integer GiB 0–1024 (default: 6) — How much of MemAvailable the unified-memory expert-cache plan leaves for the system on integrated GPUs (Linux). Gate: UMA/integrated-GPU Linux path (Strix Halo, DGX Spark), CUDA/HIP/SYCL. Best for: lower it on a tight Strix box. Tags: [tunable] (docs: docs/STRIX_HALO.md)

**`STRATA_UNBUFFERED_LOAD`** — `1`/`0` (default: auto) — Forces (`=1`) or forbids (`=0`) unbuffered (O_DIRECT-style) reads of the expert file tier; auto decides from RAM headroom. Ignored without a resident budget. Gate: Windows and Linux file-tier code, all GPUs. Best for: leave auto; `=1` only when the file cache evicts the document's experts. Tags: [tunable] (docs: docs/TROUBLESHOOTING.md, docs/QUANTS.md)

**`STRATA_VERIFY_ARENA`** — any value (default: unset) — Prints the loaded expert arena's FNV-1a checksum and size, for any loader. Gate: arena loading code, all GPUs. Best for: set for load-integrity checks. Tags: [debug]

**`STRATA_VERIFY_ASYNC_LAUNCH`** — `1` (default: off) — runs the verifier's instrumented CUDA graph launch on a separate host thread so its caller can keep serving the CPU experts' doorbells while the blocking launch waits. Gate: NVIDIA CUDA (not HIP, not gfx906 builds). Best for: A/B doorbell latency during graph launches. Tags: [ab] (docs: docs/DETAILS.md)

**`STRATA_VERIFY_COMPLEMENT`** — any value (presence check; default: off) — tests: checksums (FNV-1a per 1 GiB piece) the RAM complement copy and the profile-filled cache slots to verify the bytes read back. Gate: expert-source/verify test path. Best for: integrity tests only. Tags: [debug]

**`STRATA_VERIFY_DEVICE_PLAN`** — integer (atoi; default: off) — Lets the GPU plan resident layers itself (E-6) instead of the host; exact, neutral on 1–2 GPU rigs. Gate: verify path, all GPUs; the Intel SYCL port's setup scripts set it `=1` (with `STRATA_VERIFY_NO_HOST=1`) on Arc/xe. Best for: leave off on CUDA/HIP; keep on the SYCL port. Tags: [tunable] (docs: docs/INTEL.md, docs/INTEL_ARC.md)

**`STRATA_VERIFY_NODES`** — `1`/`2` (default: unset) — Lists the verify window's captured graph nodes (type/kernel counts; `=2` also the first nodes in graph order on HIP), including the draft graphs. Gate: CUDA and HIP graph capture (SYCL prints segment counts). Best for: set for graph-capture debugging. Tags: [debug]

**`STRATA_VERIFY_QDEDUP`** — integer (atoi; default: off) — Deduplicates the verify window's activation quantization work. Gate: verify path, all GPUs. Best for: leave off; A/B arm. Tags: [ab]

**`STRATA_WATCHDOG_IO_S`** — integer seconds (default: 10x STRATA_WATCHDOG_S, i.e. 600) — extra silent-step time the watchdog allows while the OS is measurably reading the file tier (major faults advancing), so a slow page-in from a paged-out checkpoint is not killed as a stall; `0` = no allowance. Gate: serve watchdog, no vendor gate. Best for: raise on very slow storage; leave at default. Tags: [tunable]

**`STRATA_WATCHDOG_S`** — integer seconds (atoi; default: 60) — Ends the engine with a stall report when a request's heartbeat stops this long (a flag nobody will raise); `=0` turns the watchdog off. Gate: serve path, all GPUs. Best for: leave at default; raise on very slow drives (see `STRATA_WATCHDOG_IO_S`). Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_WDDM_BUDGET`** — `0` to disable (default: on) — Applies the WDDM-aware free-memory budget correction in HIP's mem_get_info (the "leaves M MiB" line); `=0` reports raw figures. Gate: AMD HIP on Windows (WDDM). Best for: leave at default. Tags: [tunable] (docs: docs/AMD_HIP.md)

**`STRATA_WMMA_GEMM`** — any non-`0` value (default: off) — Opts into the RDNA3/RDNA3.5 WMMA dense GEMM kernels (FP16 and BF16) ahead of hipBLASLt for T≥16, beta 0/1; unsupported shapes fall through. Gate: AMD HIP gfx11 only (compiled out on CUDA). Best for: try on Strix/RDNA3.5 prompt GEMMs. Tags: [tunable]

### 7.5 Apple Metal (36)

The Metal backend (`metal/`). See docs/MAC.md.

**`STRATA_METAL_ALLOC_LOG`** — set (any value) to enable (default: off) — logs every Metal allocation of 16 MB or more to stderr with a running total, for watching memory budgets on small Macs. Best for: debugging OOM/pressure issues on the Metal backend; leave unset otherwise. Tags: [debug]

**`STRATA_METAL_ARENA_CHUNK_GB`** — float GB (default: computed, at least 8 GB) — sets the size of each streaming-arena chunk buffer, overriding the engine's computed size (max of 8 GB and a quarter of the expert total plus four times the biggest class). Best for: leave at default; tune only when debugging arena behavior on the Metal streaming path. Tags: [internal]

**`STRATA_METAL_ARENA_GB`** — float GB (default: auto-sized from free RAM) — sets the expert arena size directly, overriding the automatic sizing; the engine warns when the value exceeds the resident-footprint cap. Best for: forcing a smaller arena on a memory-constrained Mac, or benchmarking; most users should leave it unset. Tags: [tunable] (docs: docs/MAC.md)

**`STRATA_METAL_ARENA_STEP_GB`** — float GB, minimum 64 MB (default: 1.5 GB) — sets the step size in which the streaming arena is filled, each step first checking that macOS still has the memory for it. Best for: leave at default; smaller steps make the fill back off more gracefully under memory pressure. Tags: [tunable]

**`STRATA_METAL_CLASS_WASTE`** — float fraction (default: 0.15) — maximum fraction of a slot a member may waste when the layers' expert slot sizes are merged into shared LRU size classes; a larger value means fewer classes and less memory tied up in per-class minimums. Best for: leave at default; raise slightly to shrink arena overhead on models with many slot sizes. Tags: [tunable]

**`STRATA_METAL_CTAB`** — `0`/unset (default: on) — controls the routed gate/up quant grids kept in constant memory (bitwise identical results, 3-7% faster on the M5 Max); `=0` falls back to the threadgroup copy. Best for: leave at default; set to 0 only to A/B-test kernel performance. Tags: [tunable]

**`STRATA_METAL_DRAFT_VOCAB`** — path, or `full` (default: `<model>/rt/draft_vocab.bin`, else the repository's default draft_vocab.bin) — selects the draft-token vocabulary subset file used by the MTP draft layer; `full` disables the subset. Best for: leave unset; point at a custom vocab produced by the setup's --draft-vocab step, or use `full` to draft over the whole vocabulary. Tags: [internal]

**`STRATA_METAL_EXPERT_PROFILE`** — path (default: `<data>/expert-profile.bin`) — selects the expert-usage profile file that ranks experts for the streaming arena fill. Best for: leave unset; point at an alternative profile when experimenting with arena ranking. Tags: [internal]

**`STRATA_METAL_FUSE`** — `0`/unset (default: on) — toggles the decode fusion where the GDN step writes the prepared y planes itself and a window within the selection budget skips the QSA score and top-k; `=0` turns it off (same tokens, slower). Best for: leave at default; disable only for debugging or benchmarking. Tags: [tunable] (docs: docs/MAC.md)

**`STRATA_METAL_GEMV_NR`** — `1` or `2` (default: 1) — for multi-token windows, shares each x load across NR rows in the GEMV tile (sum order unchanged, so results are identical). Best for: leave at default; try 2 when tuning multi-token decode GEMV throughput. Tags: [tunable]

**`STRATA_METAL_IGNORE_REPACK`** — `1`/unset (default: unset) — starts the Metal engine even while a model repack or setup's preparation is running beside it; by default the engine refuses to start in that situation because it froze a 128 GB Mac twice. Best for: leave unset and wait for the repack to finish; set to 1 only when you are sure the concurrent process is safe. Tags: [tunable] (docs: docs/MAC.md)

**`STRATA_METAL_KEEP_FREE_GB`** — float GB (default: engine-chosen) — sets how much memory is kept free for macOS when the streaming arena is sized. Best for: raise it on a Mac that runs other heavy work beside the engine; leave at default otherwise. Tags: [tunable] (docs: docs/MAC.md)

**`STRATA_METAL_KEEP_HOST_WEIGHTS`** — `1`/unset (default: unset) — keeps the host code's own device copies of the weights (2-3 GB) instead of purging them back to macOS when the streaming arena is built. Best for: leave unset to reclaim the memory; set to 1 only for debugging the host path. Tags: [debug] (docs: docs/MAC.md)

**`STRATA_METAL_KPROF`** — integer T (default: off) — test tooling: once, after the 20th cached window of exactly T tokens with the head, prints per-kernel GPU times for that window. Best for: kernel-level GPU profiling; leave unset in normal use. Tags: [debug] (docs: docs/MAC.md)

**`STRATA_METAL_LAZY_FOLD`** — `0`/unset (default: on) — toggles the lazy GDN fold at commit: the kept prefix is folded into the state by gdn_step over only the kept tokens, with cells past the prefix rewritten when those positions run again; `=0` turns the fusion off (same tokens, slower). Best for: leave at default; disable only for debugging. Tags: [tunable] (docs: docs/MAC.md)

**`STRATA_METAL_LEAVE_SHARE`** — float fraction (default: built-in kLeaveShare, ~0.4) — on Macs with 64 GB+, the share of RAM left to macOS and its file cache when the resident-footprint cap is computed. Best for: leave at default; it exists for measuring the cap's effect. Tags: [debug] (docs: docs/MAC.md)

**`STRATA_METAL_LOGPOS`** — file path (default: off) — appends per-position sampling records (target token, logprob margin, top candidates) for head windows to the given file in append mode. Best for: offline analysis of token-level model behavior; leave unset. Tags: [debug]

**`STRATA_METAL_MTP_CHAIN`** — `0`/unset (default: on when the embedding type is supported) — runs the MTP draft steps as a chained kernel with one GPU submission per round; `=0` falls back to one submission per draft step. Best for: leave at default; set to 0 to compare draft-step submission strategies. Tags: [tunable] (docs: docs/MAC.md)

**`STRATA_METAL_MTP_GGUF`** — file path (default: `<rt>/../mtp-q2_0.gguf`) — selects the MTP draft model in GGUF form (produced by tools/mtp_pack.py) loaded beside the rt directory. Best for: leave unset; point at an alternative MTP pack when experimenting. Tags: [internal]

**`STRATA_METAL_MTP_Q8`** — `0`/unset (default: on) — quantizes the MTP draft layer's BF16 matrices to Q8_0 on load; `=0` keeps them in BF16. Best for: leave at default; set to 0 to test draft quality/speed with BF16 matrices. Tags: [tunable] (docs: docs/MAC.md)

**`STRATA_METAL_PARTIAL`** — `1`/unset (default: unset) — opts into gdn_step's partial commit, where a verify window commits its first row in place so commit(1) needs no fold; measured 0.3-0.8% slower on the M5 Max, so it stays off by default. Best for: leave unset; try 1 only when benchmarking decode variants. Tags: [ab] (docs: docs/MAC.md)

**`STRATA_METAL_PREFETCH`** — integer (default: on) — streaming prefetch of predicted layers ahead of the decode; any value <= 0 turns it off (and frees the prediction buffers). Best for: leave at default; disable only when debugging the streaming path. Tags: [internal]

**`STRATA_METAL_PROFILE`** — set (any value) to enable (default: off) — accumulates per-window-type timing (host prep, graph run, GPU time, patch/finish) and prints the breakdown to stderr when the window is destroyed. Best for: Metal performance analysis; leave unset in normal use. Tags: [debug] (docs: docs/MAC.md)

**`STRATA_METAL_PROMPT`** — `window`/unset (default: unset) — forces prompt processing through the decode windows, the slow but exact fallback (~190 tok/s), instead of the fast prompt path. Best for: leave unset; use when the fast prompt path needs to be ruled out as a source of differences. Tags: [debug] (docs: docs/MAC.md)

**`STRATA_METAL_PROMPT_CHUNK`** — integer rows (default: 4096 on streaming Macs with 96 GB+, 1024 on smaller, 2048 on resident models) — sets the prompt chunk size; each chunk makes one pass over the experts not in the arena, so longer chunks stream them fewer times (4096 gave 1,100 tok/s vs 700 at 1024 on the M5 Max). Best for: tune on streaming Macs with lots of long prompts; leave at default otherwise. Tags: [tunable] (docs: docs/MAC.md)

**`STRATA_METAL_PROMPT_EXPERTS`** — `v2` or `v3` (default: v3 on the M5's cooperative-matrix layout, else v2) — selects which expert GEMM kernel generation (gate/up and down) the prompt GEMM path uses. Best for: leave unset; set only when A/B-testing prompt kernels. Tags: [ab]

**`STRATA_METAL_PROMPT_GEMM`** — `0`/unset (default: on) — toggles the batched GEMM prompt path on streaming Macs (used for prompts of at least STRATA_METAL_PROMPT_GEMM_MIN rows); `=0` falls back to the window path. Best for: leave at default; disable to compare prompt throughput or rule out the GEMM path. Tags: [tunable]

**`STRATA_METAL_PROMPT_GEMM_MIN`** — integer rows (default: 64) — the minimum prompt length for which the streaming GEMM prompt path is used; shorter prompts go through decode windows. Best for: leave at default; raise it if short prompts are faster through the window path. Tags: [tunable]

**`STRATA_METAL_PROMPT_MIN`** — integer rows (default: 32) — on a non-streaming (resident) model, the minimum prompt length for the dedicated prompt path; shorter prompts are processed through decode windows. Best for: leave at default. Tags: [internal]

**`STRATA_METAL_PROMPT_QSA`** — `v1` or `flash` (default: flash on the M5's cooperative-matrix layout, else v1) — selects the QSA (sparse-attention scoring) kernel variant for the prompt GEMM path. Best for: leave unset; set only when A/B-testing prompt kernels. Tags: [ab]

**`STRATA_METAL_ROUTE`** — `1`/unset (default: unset) — opts into the fused dispatch of the BF16 router GEMV and moe_route beside the shared expert's GEMVs (bitwise results); measured 0.3-0.5% slower end to end on the M5 Max, so it stays off. Best for: leave unset; try 1 only when benchmarking decode variants. Tags: [ab] (docs: docs/MAC.md)

**`STRATA_METAL_SKIP`** — comma-separated words (default: off) — runs dispatches whose pipeline name contains any of the words as empty grids, to measure the marginal cost of a kernel class; produces WRONG answers, timing only. Best for: kernel-cost profiling; never use for real inference. Tags: [debug]

**`STRATA_METAL_STREAM`** — `1` or `0` (default: auto by size) — `=1` streams the experts from the SSD even when the pack fits in the resident footprint cap; `=0` never streams; the default decides by comparing mapped weights plus ~12 GiB against the cap. Best for: force streaming to test arena behavior, or force resident on a Mac that is borderline. Tags: [tunable] (docs: docs/MAC.md)

**`STRATA_METAL_STUBS`** — `log`/unset (default: unset) — with unset, calling an operation that has no Metal implementation yet prints a message and exits with code 86; `=log` only logs the call and keeps going. Best for: `log` when exploring which code paths hit missing kernels; leave unset in normal use. Tags: [debug]

**`STRATA_METAL_TRACE`** — set (any value) to enable (default: off) — prints a per-window trace: the picked token and its margin over the runner-up (a near tie flags a possible backend flip), plus per-prompt-chunk timings. Best for: diagnosing token-level differences between backends; leave unset. Tags: [debug] (docs: docs/MAC.md)

**`STRATA_METAL_WIRE_ARENA`** — `0` or `1` (default: on for Macs with 64 GB+) — wires the filled expert arena with mlock so macOS cannot page it into the compressor (unwired, an idle arena froze a Mac); `=0` leaves it pageable. Best for: leave at default; set 0 only on small Macs (the 32 GB M4 was measured unwired) or for memory-behavior experiments. Tags: [tunable] (docs: docs/MAC.md)

### 7.6 CPU-only (23)

CPU expert kernels (AVX2/AVX-512/VNNI) and the CPU expert pool.

**`STRATA_CPU_YMM`** — any value (default: off) — Forces the 256-bit (YMM) CPU expert kernel; unset uses the AVX-512 (ZMM) kernel, which is one unpack + one VNNI dot per block and beats the compute-bound 256-bit path (~39 vs 52 GB/s). Gate: CPU expert decode kernels (`src/kernels/cpu/expert.cpp`). Best for: leave at default; set to test the AVX2-width kernel (sum order differs, last-bit changes). Tags: [ab]

**`STRATA_FORCE_AVX2`** — `1`/unset (default: off) — Reports the CPU as AVX2-only, disabling the AVX-512 dispatch paths in the CPU expert layout. Gate: CPU kernel ISA selection (`expert_layout.cpp`). Best for: leave at default; set on CPUs with buggy AVX-512 or to test the AVX2 kernels. Tags: [tunable]

**`STRATA_FORCE_ISA`** — `avx2`/`avx`/`sse`/`sse4.2`/`none` (default: unset = full capability) — Makes the engine's own dispatch act as if the CPU stopped at that ISA level (for tests and older-CPU builds); ggml-cpu is unaffected and runs what it was compiled for. Gate: CPU kernel ISA capability check. Best for: leave unset; use for testing degraded-ISA builds (token streams may differ from the normal build). Tags: [debug] (docs: docs/INSTALL.md)

**`STRATA_IQ256_GATHER`** — `0`/`1`/unset (default: unset = auto) — Forces the 256-bit gather path for IQ codebook lookups on/off; unset lets the CPU's measured gather speed decide. Gate: CPU IQ expert kernels (`expert_layout.cpp` gather probe). Best for: leave at default; set when probing gather performance on a new CPU. Tags: [ab]

**`STRATA_IQ3S_MT1`** — `1`/unset (default: off) — Opt-in: on AVX2-only machines gives IQ3_S the multi-token gate/up kernel even for a single token (faster there: 0.34 → 0.19 ms per expert's rows), at the cost of changing a lone token's IQ3_S rounding. Gate: CPU native IQ3_S expert dispatch (`native_expert.cpp`). Best for: leave at default; `=1` on non-AVX-512 CPUs chasing decode speed. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_IQ_GATHER`** — any value (default: off) — Builds the IQ3 grids with one AVX-512 gather instead of scalar lookups; measured 3-5% slower on Zen 4 (microcoded gathers), so it stays opt-in. Gate: CPU IQ AVX-512 kernels (`iq_avx512.cpp`). Best for: leave at default; try on CPUs with fast gathers. Tags: [ab]

**`STRATA_IQ_MT_MIN`** — N (default: 2) — Sets the token count from which the multi-token CPU expert kernels run (ggml's single-token dot below); the default is fastest but an expert's rows round differently alone than in a group, so greedy output can depend on drafting. Gate: CPU native expert gate/up and down dispatch. Best for: set `=1` for byte-reproducible greedy output (measured -1..-3% decode on IQ3_S). Tags: [tunable] (docs: docs/DETAILS.md, docs/BATCHING.md, docs/OLDER_GPUS.md, docs/MULTI_GPU.md, docs/RESEARCH_RUNS.md)

**`STRATA_IQ_PREFETCH`** — bytes, `0` = off (default: 2048) — Software prefetch distance ahead in the IQ expert row kernels (AVX1/AVX2/AVX-512); worth -2-4% gate/up time on memory-bound IQ3_S decode. Gate: CPU IQ expert kernels. Best for: leave at default; tune (or `=0`) on CPUs where prefetching hurts. Tags: [tunable]

**`STRATA_KQ256`** — `1`/unset (default: off) — Opt-in multi-token AVX2 kernels for Q4_K/Q5_1/Q8_0 expert rows; bit-exact with ggml's dots but measured no faster in the engine (weights stay in L1 across ggml's per-token calls), though docs report 11-20% off Q4_K at 3-4 tokens. Gate: CPU native K-quant expert dispatch. Best for: leave at default; `=1` on K-quant-heavy models with multi-token windows. Tags: [tunable] (docs: docs/QUANTS.md, docs/UNSLOTH_Q4.md)

**`STRATA_NO_AVXVNNI`** — non-`0`/unset (default: off) — Disables the AVX-VNNI forms of the i-quant, IQ4_NL and Q2_0 row kernels on CPUs that have them (Alder Lake, Sapphire Rapids+); same bits, 4-25% faster rows when enabled. Gate: CPU expert kernel ISA dispatch. Best for: leave at default; set on CPUs with broken VNNI. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_NO_IQ128`** — any value (default: off) — Disables the 128-bit (AVX1) older-CPU kernels for IQ and Q2_0 rows, which only run on CPUs without AVX2 (1.2-3.1x ggml's scalar dots there). Gate: CPU native expert dispatch on pre-AVX2 CPUs. Best for: leave at default; set to force the ggml-cpu fallback on old machines. Tags: [tunable]

**`STRATA_NO_IQ4NL`** — any value (default: off) — Disables the AVX2 multi-token IQ4_NL down kernel (decodes nibbles once per block instead of per token), falling back to ggml-cpu's single-token dot. Gate: CPU native IQ4_NL down dispatch. Best for: leave at default; set to A/B the ggml path. Tags: [ab]

**`STRATA_NO_IQ512`** — any value (default: off) — Drops an AVX-512 CPU to the AVX2 multi-token IQ expert kernels (with `STRATA_NO_IQ256` it drops to ggml-cpu's vec_dot). Gate: CPU native IQ expert dispatch. Best for: leave at default; set on CPUs with slow AVX-512 (e.g. Zen 4 downclocking). Tags: [tunable]

**`STRATA_NO_NVFP4_256`** — any value (default: off) — Disables the 256-bit NVFP4 expert kernels (bit-equal to ggml's AVX2 dot, 2.0-2.4x it at 4-8 tokens), falling back to ggml-cpu. Gate: CPU native NVFP4 dispatch on AVX2 CPUs. Best for: leave at default; set to A/B the ggml path. Tags: [ab]

**`STRATA_NO_NVFP4_512`** — any value (default: off) — Disables the 512-bit NVFP4 expert kernels (ggml's arithmetic, blocks decoded once per window; 1.8x at one token, 3.7x at seven), falling back to ggml-cpu. Gate: CPU native NVFP4 dispatch on AVX-512 CPUs. Best for: leave at default; set to A/B the ggml path. Tags: [ab] (docs: docs/NVFP4.md)

**`STRATA_NO_Q8K_AVX1`** — `1`/unset (default: off) — Keeps ggml's scalar Q8_K activation quantizer on pre-AVX2 CPUs instead of the byte-identical AVX1 copy. Gate: CPU native Q8_K activation quantization (older-CPU tier). Best for: leave at default; set to A/B the reference quantizer. Tags: [ab]

**`STRATA_NO_Q8K_AVX2`** — `1`/unset (default: off) — Keeps ggml's scalar Q8_K activation quantizer (~3 µs per token and layer) on any CPU instead of the byte-identical AVX2 quantizer. Gate: CPU native Q8_K activation quantization. Best for: leave at default; set to A/B the reference quantizer. Tags: [ab]

**`STRATA_NVFP4_PREFETCH`** — bytes, `0` = off (default: 2048) — Software prefetch distance in the NVFP4 row kernels, where rows stream through 4 KB pages and hardware prefetchers stop at page boundaries. Gate: CPU NVFP4 expert kernels (AVX2/AVX-512). Best for: leave at default; tune when benchmarking NVFP4 bandwidth. Tags: [tunable]

**`STRATA_POOL_SPIN_US`** — microseconds ≥ 0 (default: built-in) — Test knob for how long CPU expert-pool workers spin before sleeping. Gate: CPU expert pool worker loop (`pool.cpp`). Best for: leave at default; use in pool latency tests. Tags: [debug]

**`STRATA_POOL_UMWAIT`** — cycles (default: 0, off) — Parked expert-pool workers wait with UMWAIT on the epoch cache line (waking on the host's publish or the cycle timeout) in the light C0.1 state instead of spinning, giving laptop power budget back to the GPU; needs WAITPKG (MSVC x64 builds only for now). Gate: CPU expert pool worker parking. Best for: set on laptops with WAITPKG CPUs to share power with the GPU. Tags: [tunable]

**`STRATA_POOL_UMWAIT_C02`** — `1`/unset (default: off = C0.1) — Uses the deeper C0.2 wait state for parked pool workers (slower wake) instead of the light C0.1. Gate: CPU expert pool UMWAIT state. Best for: leave at default; set for maximum power saving. Tags: [tunable]

**`STRATA_POOL_UMWAIT_ESC`** — TSC cycles (default: 0, off) — Experimental escalating wait: parked pool workers use C0.1 for the first N cycles, then C0.2, since a long park is a GPU phase where a slow wake costs little. Gate: CPU expert pool UMWAIT state. Best for: leave at default; experiment with UMWAIT power tuning. Tags: [ab]

**`STRATA_Q2_BITPLANE`** — `1`/unset (default: off) — Opt-in bit-plane row kernel for Q2_0 experts on CPUs with AVX2 but no AVX-512; changes the last bits. Gate: CPU Q2_0 expert row kernels. Best for: leave at default; `=1` on AVX2-only CPUs chasing Q2_0 decode speed. Tags: [tunable] (docs: docs/DETAILS.md)

### 7.7 Generic environment variables (10)

Non-`STRATA_*` variables the engine and server read: CUDA/HIP runtime switches, download
endpoints, library paths.

**`APPDATA`** — Windows directory (standard Windows variable) — on Windows, setup and the MCP app read their settings file from `%APPDATA%\Strata\settings.json` (falling back to `~\AppData\Roaming`), which remembers the data folder, last-used settings, and per-model configs. Best for: leave at the Windows default; set only to relocate per-user state. Tags: [internal] (docs: docs/INSTALL.md, docs/DETAILS.md)

**`CUDA_LAUNCH_BLOCKING`** — integer (default: unset) — the CUDA runtime's blocking-launch switch; the engine detects a non-zero value and warns that verify windows cannot run with blocking launches (the first window never ends), so the batched prompt path runs instead. Best for: set to 1 only when debugging CUDA errors synchronously; never for normal serving. Tags: [debug]

**`CUDA_MODULE_LOADING`** — `EAGER`/`LAZY` (engine default: EAGER) — the engine sets this to EAGER when unset so every CUDA kernel's code loads with the context, before the expert cache takes the free VRAM (lazy loading once OOM'd mid-prompt on a 12 GB card); costs ~30 MB of VRAM. Best for: leave unset; set LAZY only to reclaim that ~30 MB on a very tight card. Tags: [internal]

**`GPU_MAX_HW_QUEUES`** — integer (ROCm runtime variable) — the AMD HSA runtime's limit on hardware queues; Strata's verify trace prints its value alongside HIP_HOST_COHERENT and HSA_ENABLE_SDMA when diagnosing #649-class stalls. Best for: leave unset; tune only when following ROCm guidance for queue-related hangs. Tags: [debug] (inferred from code)

**`HIP_HOST_COHERENT`** — ROCm runtime variable (default: unset) — the AMD runtime's host-coherency switch; Strata's verify trace prints its value (with HSA_ENABLE_SDMA and GPU_MAX_HW_QUEUES) when diagnosing host/GPU coherency stalls. Best for: leave unset; set only per ROCm guidance when debugging verify-path hangs. Tags: [debug] (inferred from code)

**`HSA_ENABLE_SDMA`** — ROCm runtime variable (default: unset) — the AMD HSA runtime's SDMA-copy switch; Strata's verify trace prints its value alongside HIP_HOST_COHERENT and GPU_MAX_HW_QUEUES when diagnosing #649-class stalls. Best for: leave unset; set (e.g. 0) only per ROCm guidance when debugging AMD copy-path hangs. Tags: [debug] (inferred from code)

**`ProgramFiles`** — Windows directory (standard Windows variable) — the server's GPU telemetry looks for `nvml.dll` at `%ProgramFiles%\NVIDIA Corporation\NVSMI\nvml.dll` when the plain library name is not found. Best for: leave at the Windows default. Tags: [internal]

**`SYCL_UR_USE_LEVEL_ZERO_V2`** — `0`/unset (default: v2 adapter) — the oneAPI runtime's Level Zero adapter choice; `=0` selects the legacy adapter, and Strata's copy-engine path skips its work when the legacy adapter is in use (it already puts memcpy on the copy engine). Best for: leave unset; set 0 when debugging Intel GPU queue behavior on the v1 adapter. Tags: [debug] (docs: docs/INTEL.md)

**`SystemDrive`** — Windows drive letter (standard Windows variable) — the server's vision work directory falls back to the system drive's root when temp paths contain spaces (which break the engine's file-name parsing), and setup reads paging-file settings relative to it. Best for: leave at the Windows default. Tags: [internal]

**`XDG_CONFIG_HOME`** — directory (default: ~/.config) — on Linux, setup and the MCP app read their settings file from `$XDG_CONFIG_HOME/strata/settings.json`, which remembers the data folder and last-used settings. Best for: leave unset; set to relocate per-user config. Tags: [internal] (docs: docs/INSTALL.md, docs/DETAILS.md)


---

## 8. Server layer (`serve`)

Environment variables the Python server (`serve/server.py` and friends) reads directly:

**`FAKE_MCP_LOG`** — file path (default: off) — the fake MCP test server appends each handled event as a JSON line to this file. Best for: the serve test suite only; leave unset. Tags: [debug]

**`STRATA_ALLOWED_HOSTS`** — comma-separated host names (default: empty) — extra host names the server answers to in its Host-header check (ports ignored; `".example.com"` allows subdomains; `"*"` turns the check off); an empty written `api_key` or a set one also disables the check. Best for: set when clients reach the server by a name the config does not already allow (e.g. a llama-swap entry); leave unset on localhost. Tags: [tunable] (docs: docs/LLAMA_SWAP.md, docs/DETAILS.md)

**`STRATA_API_KEY`** — key string, comma-separated for several (default: empty, no key) — requires the key on `/v1/*` requests via `Authorization: Bearer` or `x-api-key`; setting it also turns off the Host check. Best for: set before exposing the server beyond localhost; several keys (one per client) let you withdraw one alone. Tags: [tunable] (docs: docs/DETAILS.md)

**`STRATA_BATCH_TRACE`** — set (any value) to enable (default: off) — prints `[batch-trace]` lines with the thread id for each batch-scheduling event to stderr. Best for: diagnosing request batching/parallelism in the server; leave unset. Tags: [debug]

**`STRATA_DEBUG`** — set (any value) to enable (default: off) — verbose server diagnostics: the raw decoded model output after each reply, one compact line per request (messages, tools, max_tokens) for diagnosing blank/empty turns, and echoed MCP server stderr lines. Best for: diagnosing odd replies or MCP servers; leave unset in production. Tags: [debug]

**`STRATA_HTTP_BACKLOG`** — integer (default: 256, floor 5) — the listen backlog of the server's HTTP socket; the stock 5 caused "connection reset by peer" for bursts of 30-40 clients. Best for: leave at default; raise only if a client burst outruns 256 queued connections. Tags: [tunable]

**`STRATA_KEEP_EMPTY_TURNS`** — `1`/unset (default: unset) — keeps completed empty assistant turns in the rendered conversation as examples for the next reply; by default they are dropped from rendering (the final message is always kept). Best for: set to 1 to restore the old rendering if a client depends on empty turns. Tags: [tunable] (docs: docs/TROUBLESHOOTING.md, docs/DETAILS.md)

**`STRATA_MAX_BODY_MIB`** — integer MiB (default: 256; 0 or an unreadable value keeps the default) — the largest request body the server reads into memory; a larger Content-Length is answered 413 before being read. Best for: leave at default (a million-token conversation fits with room); raise for unusually huge payloads. Tags: [tunable]

**`STRATA_PARALLEL_SOLO`** — `0`/unset (default: on) — lets a request decoding in a slot that is now alone (nothing busy, nobody waiting) return to the solo path with its MTP drafts; `=0` keeps it in its slot. Best for: leave at default; disable when debugging slot scheduling. Tags: [tunable] (docs: docs/BATCHING.md)

**`STRATA_REPLICA_START`** — `serial`/unset (default: parallel threads) — starts the engine replicas one at a time instead of all at once. Best for: set to serial when parallel engine starts strain the GPU or the model file cache. Tags: [tunable]

**`STRATA_REQUEST_LINES`** — set (any value) to enable (default: off) — starts a follower thread that echoes the engine log's request lines (per-request stats) into the server's log, once per log file across restarts. Best for: watching per-request engine behavior in the log; leave unset. Tags: [debug]

**`STRATA_START_RETRIES`** — integer (default: 3; 0 = never) — how often to retry an engine start that dies within 30 s with only a GPU-runtime failure (e.g. ROCm hsa_exception while the previous engine's queues are still being torn down). Best for: leave at default; raise on hosts with slow queue teardown, set 0 to fail fast. Tags: [tunable]

**`STRATA_SYCL_BIN`** — path (default: `build-sycl-aot/strata` relative to the launcher's repo) — the real engine binary behind the Intel `strata-sycl.sh` launcher, used when the server reads the engine's option strings for a feature check. Best for: set when the SYCL engine lives at a non-default build directory (e.g. `build-sycl/sycl/strata`). Tags: [tunable] (docs: docs/INTEL_ARC.md)


---

## 9. Installer layer (`setup.py`)

Environment variables `setup.py` / the installer reads (most have a CLI twin in section 3):

**`CC`** — compiler command (default: CMake's pick) — names the C compiler for a from-source build; setup passes it as `-DCMAKE_C_COMPILER` so an already-configured build folder picks it up too. Best for: set alongside CXX when the default compiler is too old for the toolchain. Tags: [tunable] (docs: docs/TROUBLESHOOTING.md)

**`CUDAHOSTCXX`** — compiler command (default: the value of CXX) — nvcc's host compiler (-ccbin) for CUDA builds, passed as `-DCMAKE_CUDA_HOST_COMPILER`. Best for: leave unset (it follows CXX); set only when nvcc needs a different host compiler than the C++ build. Tags: [tunable] (docs: docs/TROUBLESHOOTING.md)

**`CXX`** — compiler command (default: CMake's pick) — names the C++ compiler for a from-source build (e.g. `CXX=g++-14 ./setup.sh`); setup passes it as `-DCMAKE_CXX_COMPILER` so an existing build folder takes it too. Best for: set when the default compiler is too old or a specific one is required. Tags: [tunable] (docs: docs/TROUBLESHOOTING.md, docs/INTEL_ARC.md, docs/AMD_HIP.md)

**`HF_ENDPOINT`** — URL (default: https://huggingface.co) — the Hugging Face host setup and the fetch tools download from (a mirror such as https://hf-mirror.com); the same pinned revisions and SHA-256 checks apply on any host, and the MTP draft layer comes from there too. Best for: set to a fast regional mirror; leave unset otherwise. Tags: [tunable] (docs: docs/INSTALL.md)

**`LD_LIBRARY_PATH`** — directory list (standard Linux variable) — setup prepends the ROCm wheel folders and the Vulkan SDK loader to it when compiling/running the HIP and Vulkan backends, and the engine's runtime loader searches it for backend libraries. Best for: leave at your system default; export the SDK paths yourself only when starting the engine by hand (e.g. Strix Halo). Tags: [internal] (docs: docs/STRIX_HALO.md)

**`MODELSCOPE_ENDPOINT`** — URL (default: https://www.modelscope.cn) — the ModelScope host setup and the fetch tools download from when the source is ModelScope. Best for: leave unset; point at a regional mirror when the default is slow. Tags: [tunable]

**`PATH`** — directory list (standard variable) — setup prepends the ROCm bin/llvm-bin and the Vulkan SDK bin when building; the MCP config finds `npx`/`uvx`/`python` servers on it; starting the engine by hand on Windows may need `engine\rocm\bin` added. Best for: leave at your system default; extend only for manual starts per the backend docs. Tags: [internal] (docs: docs/AMD_HIP.md, docs/STRIX_HALO.md, docs/MCP_SERVER.md)

**`ROCM_PATH`** — directory (default: /opt/rocm) — where setup looks for a system ROCm (hipcc, hipBLAS, HIP dev files) for the AMD backend; a system ROCm 7+ there is used, otherwise TheRock wheels go into `.venv`. Best for: leave unset with a standard install; point at a non-standard ROCm root (e.g. a Strix Halo SDK). Tags: [tunable] (docs: docs/AMD_HIP.md, docs/STRIX_HALO.md)

**`STRATA_CUDA`** — `12`, `13`, or `auto` (default: auto = CUDA 13) — environment default for setup's `--cuda`: which CUDA toolkit's prebuilt engine to install; CUDA 12 (experimental) targets cards older than CUDA 13 supports (Pascal, Volta) and runs with older drivers (Windows 528+, Linux 525+). Best for: set to 12 on Pascal/Volta cards; leave unset otherwise. Tags: [tunable] (docs: docs/OLDER_GPUS.md)

**`STRATA_EXECV`** — set (any value; non-Windows) to enable (default: off) — setup replaces its own process with the server via execv instead of spawning a child, so in Docker the server is PID 1 and `docker stop`'s SIGTERM reaches it. Best for: set in container images; leave unset on normal Linux starts. Tags: [internal]

**`STRATA_EXPERIMENTAL_GFX1103`** — `1`/unset (default: unset) — opt-in, unsupported: admits the Radeon 780M / 760M / 740M iGPU (Ryzen 7040/8040, gfx1103) as a unified-memory AMD card like Strix Halo, using the portable kernels (no WMMA). Best for: leave unset; set to 1 only on such an iGPU, accepting that it is unvalidated. Tags: [tunable]

**`STRATA_EXPERIMENTAL_SM60`** — `1`/unset (default: unset) — lists Pascal (6.x) and Volta (7.0) NVIDIA cards as usable and lets setup pick the experimental CUDA 12 engine built with `-DSTRATA_EXPERIMENTAL_SM60=ON`; naming the card with --gpu/--gpus or --cuda 12 has the same effect. Best for: set on a PC whose only NVIDIA GPU is Pascal/Volta. Tags: [tunable] (docs: docs/OLDER_GPUS.md, docs/NVIDIA_V100.md)

**`STRATA_ISA_FLOOR`** — `avx` or `none` (default: chosen from the CPU) — forces the experimental older-CPU engine build on any PC: `avx` for Sandy/Ivy Bridge and Bulldozer, `none` for SSE4.2+POPCNT (Nehalem/Westmere); unset picks automatically (empty = the normal AVX2 engine). Best for: leave unset; set only to test the older build on a newer PC. Tags: [tunable] (docs: docs/INSTALL.md)

**`STRATA_NVCC`** — path to nvcc (default: newest toolkit found) — makes setup use exactly this CUDA compiler and no other (e.g. when a newer toolkit beside it cannot build on this PC, or CUDA 13.2.0/13.2.1 must be avoided for an RTX 50 card). Best for: pin a known-good toolkit when setup picks a bad one; leave unset otherwise. Tags: [tunable] (docs: docs/TROUBLESHOOTING.md, docs/OLDER_GPUS.md)

**`STRATA_PREBUILT_URL`** — URL folder or local folder (default: Strata's release location) — environment default for `--prebuilt`: where the ready-made engine archives are fetched from. Best for: point at a local folder or mirror for offline installs; leave unset otherwise. Tags: [tunable]

**`STRATA_ROCM_INDEX`** — index URL (default: per-architecture TheRock index) — overrides the wheel index setup picks ROCm/TheRock wheels from for the card's architecture family. Best for: leave unset; point at a specific index when a family's default is broken. Tags: [internal] (docs: docs/AMD_HIP.md)

**`STRATA_ROCM_VERSION`** — exact version string (default: per-family pins, e.g. 7.10.0a20251120) — forces one exact ROCm/TheRock wheel version with no lookup of newer builds in the same line. Best for: leave unset; pin a version when a nightly regresses on your card. Tags: [tunable] (docs: docs/AMD_HIP.md)

**`STRATA_SKIP_SHA256`** — `1`/unset (default: unset) — skips verifying the downloaded ready-made engine against the SHA-256 GitHub publishes; setup warns loudly and installs unverified files. Best for: leave unset; use only if you insist on installing a file whose checksum cannot be fetched. Tags: [tunable] (docs: docs/TROUBLESHOOTING.md)

**`STRATA_SOURCE`** — `auto`, `huggingface`, `modelscope`/`ms` (default: auto = Hugging Face) — where model files are downloaded from; ModelScope (mainland China) serves the repository's current files, each checked against ModelScope's published SHA-256, and setup never switches hosts by itself. Best for: set to modelscope when Hugging Face is slow or unreachable. Tags: [tunable] (docs: docs/INSTALL.md)

**`STRATA_VCVARS`** — path to vcvars64.bat (default: found via vswhere) — picks the Windows Visual Studio C++ environment by hand when vswhere cannot use the installed one (e.g. only Visual Studio 2026 present, which CUDA 13 rejects). Best for: leave unset; point at a CUDA-compatible vcvars64.bat when builds fail to find a compiler. Tags: [tunable]

**`VULKAN_SDK`** — directory (default: auto-detected) — points at an installed Vulkan SDK (headers + glslc) for compiling the Vulkan image generator (sd-cli); if unset, setup uses the distribution's packages or unpacks the LunarG SDK into third_party (Linux, no sudo). Best for: set when a Vulkan SDK is already installed in a non-standard place. Tags: [tunable]


---

## 10. Web UI settings surface (and the vendor-gating plan)

Today the Settings card edits the keys in `serve/runconfig.py::EDITABLE` via `GET/POST /config`:
sampling defaults, reasoning budget, aliases, idle unload, lazy load, `vram_reserve_mib`
(the only engine-arg key). Each entry is `(key, kind, help)`; kind `("arg", "--flag")` writes
into the config's `args` list.

**Planned: expose the full option set, shown per vendor.**

1. **New kind `("env", "STRATA_X")`** in `runconfig.py` — reads/writes `cfg["env"][NAME]`
   (the server already exports `env` into the engine). Add `("arg", flag)` entries for CLI
   switches. `check()` validates types; `apply()` already handles add/remove.
2. **Vendor metadata per entry**: extend the tuple with a `vendor` field
   (`nvidia|amd|intel|all|cpu`) and a `tier` (`tunable` only — never surface `[ab]`/`[debug]`
   by default). `view()` filters the list by the detected backend.
3. **Vendor detection server-side**: the config's `backend` key (`cuda|hip|sycl`) plus the
   engine's startup line (gfx1201 / SM version / Arc). Expose `GET /config` →
   `{"vendor": "amd", "arch": "gfx1201", "keys": [...]}`; the web UI renders only what applies.
4. **Arch-aware gating beyond vendor**: entries carry an optional `arch` regex
   (`gfx12`, `sm_75+`); the server hides entries whose gate can't match the running card
   (e.g. hide `STRATA_HIPBLASLT_TUNING` on NVIDIA, `STRATA_SM60_TABLE` on AMD).
5. **Restart semantics**: engine args/env apply on next engine start; the UI shows the
   "used from the next start" note (already in `view()`), and a Restart button can reuse the
   server's existing restart path.
6. **Safety**: keep the `EDITABLE` allowlist model — only entries listed in `runconfig.py`
   are writable; secrets (api_key, host) stay out.

Suggested first tier to surface (all `[tunable]`): `vram_reserve_mib`, `--prefill auto:32768`,
`--spec`, `--spec-min-p`, `--suffix-draft`, `--kv`, `--kv-grow`, `--expert-cache`,
`--expert-cache-per-layer`, `--pool-workers`, `--conversation-cache-mib`, `--prompt-cache`,
`--batch`, AMD: `STRATA_HIPBLASLT_TUNING`, `STRATA_HIP_WMMA`, `STRATA_SELECT_WMMA`,
`STRATA_ATTN_PRE75`, `STRATA_GFX12_DEFAULTS`, `STRATA_MTP_PREFILL_SYNC`; NVIDIA:
`STRATA_VRAM_ELASTIC` path, `STRATA_SM*_TABLE` picks, NVFP4 switches; Intel:
`STRATA_SYCL_A770_FAST`, XMX switches.

---

*Generated from the v0.1.44-era tree (branch `r9700-upstream` lineage): engine `--help`,
every production `getenv` site in `src/`, `include/`, `serve/`, `setup.py`, `sycl/`, `metal/`,
and the `docs/` corpus. Test/bench-only variables (in `tests/`, `bench/`, `*_parity.cpp`,
`*_bench.cpp`) are excluded from the vendor sections.*
