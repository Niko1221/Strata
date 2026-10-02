# Pool: several PCs as one

Strata's pool has two ways to use more than one PC. Pick one in the **Pool** tab, on each PC.

| | **Share requests** (routing) | **Split layers** (coordinator + workers) |
| --- | --- | --- |
| Each PC holds | the whole model, as it would alone | only its own layers (experts, dense weights, context) |
| One chat | as fast as one PC | about as fast as one PC (the PCs take turns per token) |
| Two chats at once | both run at full speed, one per PC | one waits for the other |
| What it adds | throughput, and a spare when a PC is off | room: a model, context or expert cache too big for one PC |
| Network | only the request and the answer | every token crosses it; wired is best |
| A PC goes away | the others carry on alone | the pool stops until it is back |

**Neither makes one chat faster.** Each token passes through the 48 layers in order and needs the token before it.
Two GPUs can work on one token only by taking turns (the layer split) or by exchanging results at every layer, which
over a network costs more time than it saves.

## Share requests (routing mode)

Every PC runs the whole model as an ordinary Strata server, with its own conversation cache. A chat request that
reaches **any** of them runs on the PC that suits it:

- **The PC that holds the conversation.** A PC keeps a chat's context (its prompt cache and parked chats), so the
  next turn runs there and reads only the new part. Moving a chat means reading it all again on the other PC, which
  takes minutes for a long one. The router recognises a conversation from its messages (a chain of hashes; the
  prompt-caching markers clients move from turn to turn are ignored).
- **If that PC is busy** and another is idle, the chat moves only when the other PC would read at most *Move a busy
  PC's conversation* tokens again (8,000 by default, about 10 seconds of reading). A long chat waits for its own PC.
- **A new conversation** goes to an idle PC: the **primary** first, then the PC it arrived at, then any other. When
  every PC is busy, it joins the shortest queue.
- **A PC whose context window is too small for the request is skipped.** Each PC keeps its own window (set in its own
  config) and tells its own clients that number. A PC with a 192K window never gets a 230K-token chat; it can still
  send one to a PC with a 256K window. You do not need to make the windows equal.
- **When the other PCs cannot be reached**, each PC answers its own requests and checks for the others every few
  seconds. A request in flight to a PC that stops answering before it starts is run on this PC instead.

Switching between PCs never loses context: the API is stateless, so every request carries the whole conversation.
The answers are the same model's. The sampling defaults (temperature, thinking budget, ...) are those of the PC that
runs the request, so keep the configs alike; the Pool tab marks a PC whose defaults differ.

Setting it up, on **each** PC: choose **Share requests**, give every PC the same **pool secret**, and add the other
PCs' web app addresses (`192.168.1.20` or `laptop:8080`; they also appear under *Found on your network*). Turn on
**This PC is the primary** on the PC new conversations should prefer. Apply. The model keeps running (switching
between *Off* and *Share requests* does not restart it). Point your apps at any PC; the response header
`X-Strata-Pool-Ran-On` names the PC that answered a routed request.

The PCs prove themselves to each other with the pool secret: a routed request carries an HMAC over the time, a nonce,
the path and the body. A routed request skips the receiving PC's own API key; nothing else does.

## Split layers (coordinator + workers)

The model is cut **by layers**. Each PC runs a contiguous range of the 48 layers on its own GPU, and for those layers
it keeps:

- **their experts**, in its own RAM and in its own VRAM cache, and
- **their share of the context**: the KV cache and recurrent state of those layers for every token.

So the pool holds more experts on GPUs than any one of its PCs could, and a long context is divided between them. A
PC whose RAM could not hold the whole model can still take part: it only holds its own layers' experts.

```
 Coordinator (your main PC)              Worker (another PC)
 ┌──────────────────────────┐            ┌──────────────────────────┐
 │ layers 0..K-1            │  rows ──►  │ layers K..47             │
 │ + output head, sampler   │  ◄── rows  │ its experts: RAM + VRAM  │
 │ + draft layer (MTP)      │            │ its KV / GDN state       │
 │ + the server you chat to │            └──────────────────────────┘
 └──────────────────────────┘
```

The **coordinator** is the PC you chat with. It runs the first layers, the output head, the sampler and the draft
layer (speculative decoding), and every decision of the serve loop: checkpoints, the conversation cache, sampling.
A **worker** is a plain function of what it receives. It gets each verify window's residual rows (12,804 floats per
token, about 300 KB for a 6-token window), runs its layers, and sends the rows back. It does the same with each prompt
chunk. It keeps its own layers' state and checkpoints under the ids the coordinator gives it.

This is pipeline parallelism, the same cut as Strata's [multi-GPU layer split](MULTI_GPU.md), with a network in
place of pinned RAM. A token crosses the network once out and once back per verify window, not once per layer. The
experts of one layer are never split across PCs: that would cost a network round trip at each of the 48 layers for
every token.

### What you need

- Two or more PCs on the same local network, each with a supported NVIDIA card (RTX 20 or newer, 8 GB or more).
- **The same model and size installed on every PC** (for example the Coder IQ1_M on both). Each PC reads its own
  layers from its own copy. The coordinator refuses a worker whose model differs, and says how.
- A wired network is best. At 1 Gbit/s a decode window adds about 3-5 ms (two crossings of ~300 KB plus latency).
  A 6,144-token prompt chunk is ~250 MB each way in exact mode, about 2 s at 1 Gbit/s, and it overlaps with the
  coordinator's own work. Wi-Fi works, but slower.
- An engine with the pool on each PC: one built from this version of Strata or later. An older engine says so in the
  Pool tab; `START-HERE.bat --build` compiles this checkout's engine (10-20 minutes, once). Share requests (routing)
  works with any engine: only the server takes part.

### Setting it up

1. **On every PC:** install Strata with the same model (`START-HERE.bat`). Then run `POOL-FIREWALL.bat` as
   administrator, once. It opens TCP
   7701 (the engines), UDP 7702 (the PCs finding each other) and TCP 8080 (the app) on private networks.
2. **On the worker PC:** open the app, go to **Pool**, choose **Worker**, and click **Apply**. The chat on that PC
   turns off and its GPU waits for a coordinator.
3. **On the coordinator PC:** open **Pool** and copy the **pool secret**. Paste it into the worker's Pool tab and
   Apply there too. Every PC of a pool must have the same secret.
4. **On the coordinator PC:** choose **Coordinator**. Add the worker: it shows up under *Found on your network*, or type
   its address (`192.168.1.20`, or `laptop:7701`). Click **Apply**. The model restarts. The coordinator connects to
   the worker and places the split. Both PCs then load their layers at the same time, which takes a minute or two.
   The **Layer map** shows which PC runs which layers, and how many experts each holds in VRAM.

Then chat on the coordinator, or point your apps at it, as before (`http://<coordinator>:8080/v1`).

With more than one worker, the order in the Workers table is the layer order. Use the arrows to change it.

## Testing it on one PC first

`POOL-SELFTEST.bat` checks the whole pool path on a single PC. Close Strata first, so the GPU is free.
The test runs the model alone. Then it runs the same model split between a coordinator and a worker on this same GPU,
connected over `127.0.0.1`, and uses the same prompts. It prints both answers side by side, how many tokens agree,
and the decode speed of each run.

Both engines share one GPU, so each gets a small expert cache and a short context (`--cache 400 --ctx 8192` by
default). The answers need not match token for token. A GPU-computed expert rounds slightly differently from a
CPU-computed one, and the pair caches different experts than the single engine. The answers should still make sense
and mostly agree.

```
POOL-SELFTEST.bat                         (or: .venv\Scripts\python tools\pool_selftest.py strata-<model>.json)
POOL-SELFTEST.bat --split 30 --wire f16   other options: --ctx, --cache, --chunk, --max-new, --skip-single
```

## Settings (Pool tab)

| Setting | What it does |
| --- | --- |
| **Role** | *Off*: this PC runs the model alone, as Strata. *Share requests*: routing mode (above). *Split: coordinator*: it uses the workers listed. *Split: worker*: it lends its GPU and RAM to a coordinator. |
| **Pool secret** | Proves to a worker that a coordinator belongs to the pool, and the other way round (HMAC-SHA256 challenge, both ways). The secret itself never crosses the network. |
| **Worker port** | The TCP port a worker's engine listens on (7701). |
| **Split** | *Automatic* (the default) weighs each PC's free VRAM after its dense weights, its free RAM for its layers' experts, and its GPU's speed. It uses the same cost model as the multi-GPU split, plus a network hop per PC. *Manual*: the first layer of each worker, e.g. `30`, or `16,32` for two workers. The coordinator keeps at least layers 0 and 1. |
| **Network precision** | *Exact* (f32) sends 32-bit rows, so the answers match one PC's exactly. *f16* / *bf16* halve the bytes on a slow network, with a small rounding difference. The last worker's prompt rows feed only the draft layer, so they always travel as f16. Drafts are only guesses that the model checks, so this never changes the output. |

The settings live in `strata-<model>.pool.json` next to the model's config. The role is kept between starts.
`run-<model>.bat --role off` (or `coordinator` / `worker`) overrides it for one start.

## What each PC holds

Each PC loads only its own layers: their experts in RAM, their dense weights and their part of the expert cache in
VRAM, and their part of the context (KV and state). A worker leaves out the embeddings and the output head, and the
coordinator leaves out the workers' layers. So the VRAM that the other layers' dense weights would take goes to
experts instead, and the pool as a whole caches more of them than any one PC.

## Which PCs make good workers

The pool runs its layers one after the other, so a token waits for every PC's part in turn. A worker helps when its
layers run about as fast as the coordinator's would, and when its RAM and VRAM let the pool cache experts that would
otherwise miss. A much slower GPU adds its slowness to every token, and the network adds about 1.5 ms each way.

For example, a GTX 1050 (4 GB, Pascal) takes around 10 ms per layer, where an RTX 3060 takes about 1.5 ms. Even one
layer on it costs more time than the experts it could cache save. Its prompt processing is slow as well, because
Pascal has no tensor cores. So the engine is built for Turing (RTX 20xx) and newer, and a GPU like that is better left
out of the pool. The automatic split would give it as few layers as it can.

## When a worker stops, or you change something

- A worker that loses its coordinator **keeps its layers loaded**. When the coordinator comes back with the same
  settings (it restarted, or it was unloaded while idle), the worker continues at once. If the coordinator asks for
  other layers, the worker restarts to load them and the coordinator waits for it.
- If the coordinator cannot reach a worker, the Pool tab says why: not reachable, a different model, a different
  secret, or busy with another coordinator. **Restart engine** tries again.
- If a worker fails during a request, the request ends with an error and the next request starts the pool again.

## What is not supported yet

- **Images** (vision), the **experimental speed projection** (control vectors) and the **low-RAM modes**. The Pool tab
  says when your setup has one of them on.
- **Conversation parking** (whole-chat snapshots in RAM). The per-chat checkpoints (`--prompt-cache`) work across
  the pool: each worker keeps its own layers' part.
- **Several GPUs in one pool PC.** Each pool PC uses one GPU, and the pool is the split.

## Engine flags

`run-<model>.bat` gets these from the Pool tab, so you do not normally type them.

```
--pool-peers H:P[,H:P..]     coordinator: the workers, in layer order (default port 7701)
--pool-split auto|K1[,K2..]  where each worker's layers start
--pool-listen [H:]P          worker: wait for a coordinator here
--pool-secret S              the shared secret (also STRATA_POOL_SECRET; the app passes it in the environment)
--pool-wire f32|f16|bf16     the rows on the network (f32: exact, the default)
--pool-draft-wire F          the last worker's prompt rows (default f16; they feed only the draft layer)
--pool-timeout-s S           a worker's reply (default 300)
--pool-wait-s S              connecting to and loading the workers at start (default 900)
```

## How it is built

| Part | Where |
| --- | --- |
| Frames, control messages, row codecs (f32/f16/bf16) | `include/strata/pool/protocol.hpp`, `src/pool/protocol.cpp` |
| Sockets, SHA-256 / HMAC handshake | `include/strata/pool/net.hpp`, `src/pool/net.cpp` |
| The worker's handshake and its "busy" answer | `src/pool/worker.cpp` |
| The split search | `src/pool/split.cpp` |
| The coordinator's link: workers, the head, prompt rows | `src/pool/link.cpp` |
| The verifier's and prompt path's hand-off to the network | `Verifier::set_link` / `set_headless`, `Prefill::remote_next` / `set_headless` |
| Each node's experts in RAM for its range only | `ArenaExpertSource::set_layer_range` |
| The worker's serve loop, and the coordinator's hooks | `src/program/generate.cpp` (search for `POOL`) |
| The server's side: roles, the worker supervisor, discovery | `serve/pool.py`; the Pool tab: `serve/web/pool.js` |

Tests (no GPU needed): `strata-pool-test` covers hashing, the wire formats, frames, the handshake and the split
search. `strata-pool-link-test` runs the coordinator against scripted workers: the secret, the model check,
reloading, busy workers, prompt rows through two workers, and the lazily acknowledged commits.
`python -m unittest serve.test_pool` covers the config, the supervisor, discovery, and switching a running server
between roles.

The protocol in one paragraph: one TCP connection per worker. The worker sends `HELLO` (its model, its GPU, its free
VRAM and RAM, and a nonce). The coordinator answers `AUTH` (an HMAC of the nonce under the secret, plus its own nonce),
and the worker answers `AUTH_OK` the same way. Then `CONFIG` gives the worker its layer range and the settings that
must match (context, KV format, window size, rope), and the worker loads and answers `READY`. After that each request
gets exactly one reply, in order: `VERIFY` gets `VERIFY_ROWS`, and `PREFILL` gets `PREFILL_ROWS`. `COMMIT`, `RESET`,
`CKPT_SAVE`, `CKPT_RETAIN` and `END_REQUEST` are acknowledged lazily. The coordinator reads those acknowledgements
before its next request, so a commit costs no round trip of its own.
