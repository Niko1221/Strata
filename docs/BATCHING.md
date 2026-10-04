# Several conversations at once (batch slots)

By default Strata serves **one request at a time**: the others wait in the server's queue. With `--batch N`
the engine keeps up to N conversations open and decodes them **together**: every verify window then carries one
token of each conversation, so the dense weights, the shared expert and the head are read once per window for
all of them. Combined with a layer split across several GPUs and `--batch-groups`, the cards also work on
different conversations at the same time instead of waiting for each other.

It is opt-in and changes nothing when the options are absent.

## Turning it on

Add the options to the `args` list of the model's config (`strata-<model>.json`) and restart:

```
"args": [ ..., "--batch", "8", "--batch-groups", "4", "--trim-stage-weights" ],
"layer_split": "12,24,36"
```

| Option | What it does |
| --- | --- |
| `--batch N` (2..8) | up to N conversations decoded together. Each slot gets its own state (a session carved like the stage's own: GDN recurrence, QSA K/V and indexer, PLE history) on every GPU of the split. |
| `--batch-groups G` | with a layer split: the N slots in G groups that flow through the GPUs as a pipeline (GPU k runs one group while GPU k+1 runs another). G must divide N. 1 = all slots in one window, GPU after GPU. |
| `--trim-stage-weights` | with an **explicit** `--layer-split` (e.g. `12,24,36`, not `auto`): every GPU loads only the dense weights of its own layers instead of the whole model's. The VRAM this frees goes to the expert cache - which is what makes batching pay, since several conversations together touch more distinct experts than one. Useful without `--batch` too. |

The server needs nothing else: it reads `--batch` / `--batch-groups` from the engine's arguments.

### Choosing the settings: `tools/autoconfig.py`

```
python3 tools/autoconfig.py --config strata-<model>.json               # rules from the machine, writes .auto.json
python3 tools/autoconfig.py --config strata-<model>.json --calibrate   # + starts the engine with each candidate
```

It reads the GPUs (count, VRAM, PCIe link), the RAM and the model's layer count, and writes a copy of the config:

- **Split**: one card, none; several, an explicit split with each card's share of the layers proportional to its
  VRAM, and `--trim-stage-weights`. A card whose link runs narrower than it can (x8 of x16) is named: every
  hand-off crosses it.
- **Slots**: `--batch 2 x cards` (at most 8) and one pipeline group per card; `--batch 4` on one card.
- **Context**: with KV streaming every sequence (the solo session and each slot) holds its K/V in pinned host
  memory, ~263 bytes per layer per token (int8). The largest context whose K/V for all sequences stays under a
  quarter of the RAM; if even 32K does not fit, fewer slots.
- **VRAM reserve**: 700 MiB, plus the slot drafters' K/V with `--batch-spec` (~175 MiB per slot at 262K).
- **Parking**: `--conversation-cache-mib` at 8 % of the RAM, 8 GiB at most.

The rules come from a few machines. `--calibrate` measures the candidates (the rules, the rules with
`--batch-spec 2`, the rules without pipeline groups) at 1 request (solo path), 2, 4 and all slots, and keeps the
one with the best mean relative rate; each candidate is an engine start, so it takes several minutes per candidate.

## How the server uses the slots

- **One request alone** runs on the usual solo path (verify windows with MTP drafts): the fastest single stream.
- **When a second request arrives**, the first is stopped (`STOP`) and continues in a batch slot with its prompt
  plus what it generated so far - the engine's prompt cache holds exactly that, so nothing is read again - and the
  new request is admitted next to it.
- **Each admission** reads the request's prompt through the usual prompt path (prompt cache and conversation
  checkpoints included) and produces its first token there; the state is then copied into the slot. Admissions
  are taken one at a time; the slots already decoding pause meanwhile.
- Slots are assigned so that consecutive requests land in different pipeline groups.
- A client that disconnects stops its slot (`BSTOP`); the others go on.

## Exactness

A batch row's arithmetic is the single-token window's, so with greedy decoding **every conversation of a batch
produces exactly the tokens it produces alone** - verified token by token for 8 concurrent conversations of 150
tokens, with and without the pipeline (`tools/batch_test.py`). Two settings make that comparison exact:

- `STRATA_IQ_MT_MIN=1` (the multi-token CPU expert kernels for every group, as for the solo path's own
  exactness tests), and
- `--pcie-frac 0`: the PCIe share of the missed experts is chosen per window from the window's misses, so the
  same expert can run on the GPU in one window and on the CPU in another, which rounds differently. With a share,
  outputs stay coherent but can drift apart after some tokens, as they would between two solo runs whose windows
  differ.

Sampled requests (temperature, top_p, top_k, min_p, seed) are drawn row by row with the solo window's
counter-based draw (Philox(seed, position)).

## Limits (for now)

- Batch windows carry no MTP drafts: a conversation in a slot decodes one token per window (the solo path keeps
  its drafts, which is why a request alone is not put in a slot).
- Repetition / frequency / presence penalties are not applied in batch windows.
- `--batch-groups` needs every stage on its own GPU.
- The slot sessions take VRAM (each one like the stage's own session; a few hundred MB per slot at 64K context
  with KV streaming) and pinned RAM for their streamed K/V.

## Measured

A 4-GPU layer split (4 x 16 GB, PCIe Gen3), IQ3_S, `--batch 8 --batch-groups 4 --trim-stage-weights`, through
the HTTP server, 400 tokens per answer, temperature 0.7:

| Concurrent requests | Per request | Total |
| ---: | ---: | ---: |
| 1 | 123 tok/s (solo path) | 120 tok/s |
| 2 | 57 tok/s | 113 tok/s |
| 4 | 51 tok/s | 205 tok/s |
| 8 | 45 tok/s | 360 tok/s |

With the patches below on engine 0.1.38 and parking on, through the service: 8 requests at temperature 0 -> 369
tok/s, at 0.7 -> 358 tok/s.

`--trim-stage-weights` alone raised the share of experts held in VRAM on that machine from 76-85 % to 84-100 %
per card.

## Together with conversation parking

`--conversation-cache-mib N --conversation-cache-slots K` (DETAILS.md) works with the layer split too: a request
whose conversation was parked is restored on every stage before its admission, so an agent and its sub-agents, or
several chats that alternate, come back without reading their history again. Measured on the same 4-GPU split,
two long conversations alternating through the HTTP server: the first turns took 3.9 s and 5.7 s to the first token
(their prompts read), the follow-ups 0.53 s and 0.46 s.

## Testing

Three scripts drive a built engine or a running server; each exits non-zero on a failure.

| Script | What it checks |
| --- | --- |
| `tools/batch_test.py` | the same prompts alone (`GEN`) and together in the batch slots (`BGEN`): every slot's greedy tokens equal its solo tokens; prints the aggregate rate. `--batch-groups` in `--extra` tests the pipeline, `--keys "temperature=0.7"` the sampled rows. |
| `tools/parking_test.py` | a follow-up to a conversation decodes the same tokens whether its state stayed live or came back from the parking cache (with a layer split: every stage's image). |
| `tools/early_close_test.py` | a client that stops reading a streamed answer early (alone, and with a second request running) does not leave its tokens to the next request (server). |

For exact comparisons pass `--pcie-frac 0 --adapt-every 1000000` (and the scripts set `STRATA_IQ_MT_MIN=1`):

```
python3 tools/batch_test.py --exe engine/strata --config strata-<model>.json --batch 8 --n 8 \
    --extra "--layer-split 12,24,36 --trim-stage-weights --batch-groups 4 --pcie-frac 0 --adapt-every 1000000"
python3 tools/parking_test.py --exe engine/strata --config strata-<model>.json \
    --extra "--layer-split 12,24,36 --conversation-cache-mib 8192 --conversation-cache-slots 4 --pcie-frac 0"
STRATA_KEY=<key> python3 tools/early_close_test.py http://127.0.0.1:8080
```

## Engine protocol (`--serve`)

On top of `GEN` / `GENI`:

| Line | Direction | Meaning |
| --- | --- | --- |
| `BGEN <slot> <max_new> [keys] <ids>` | in | read the prompt (as `GEN 1`), then continue in `<slot>` |
| `BGENI <slot> <max_new> [keys] <file> <ids>` | in | the same with images |
| `BADM <slot> <1/0>` | out | after the admission's `DONE`: 1 = it continues in the slot, 0 = it ended |
| `BT <slot> <id>` | out | a token of that slot |
| `BDONE <slot> <generated> <stop/length/cancel> <ms>` | out | the slot is free again |
| `BSTOP <slot>` | in | end that slot at its next window |

`tools/batch_test.py` drives the engine directly: the same prompts alone, then together, compared token by token,
and the aggregate rate.
