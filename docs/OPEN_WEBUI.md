# Running Strata behind Open WebUI

[Open WebUI](https://github.com/open-webui/open-webui) is a chat interface for several users, with document upload,
RAG, tools and voice. Strata works as an ordinary OpenAI-compatible connection in it. This page is what we needed to run
it for a small office. We tested Open WebUI 0.11.4 (venv install, same machine as Strata) and Strata 0.1.40 with Unsloth
UD-IQ4_XS, 262,144-token context and vision on the CPU encoder, on 2x RTX 4060 Ti 16 GB in a layer split with a
Threadripper PRO 3975WX.

## How many people can share it

Strata serves **one request at a time** unless you enable `"parallel"` (see [BATCHING.md](BATCHING.md)). That is one
request, not one user: all users stay connected and can chat at once, and only their requests take turns in the
server's queue. That does not mean less work gets done. On our machine Strata decodes ~55 tok/s alone, while
a 27B dense model we ran before on the same cards reached ~14 tok/s per request with 3 slots. With one slot the total
output was higher, and only the waiting moved around.

So whether it fits an office depends on **your** speed, not ours. Estimate it with two numbers from your own runs (the
[community reports](../bench/results/COMMUNITY.md) give you typical values):

- **one answer** takes about *answer tokens / decode tok/s*. Chat answers are mostly 300-500 tokens;
- **one long document** takes about *prompt tokens / prompt tok/s* before the first word. Nobody else is served while
  it is read.

| decode tok/s | 400-token answer | prompt tok/s at 128K | 100,000-token document |
| ---: | ---: | ---: | ---: |
| ~30 (1x 4060 Ti, Haswell + DDR3, IQ3_S) | ~13 s | not measured | not measured |
| ~55 (our 2x 4060 Ti, UD-IQ4_XS) | ~7 s | ~2,200 | ~45 s |
| ~60 (2x RX 6900 XT, IQ3_S) | ~7 s | ~820 | ~2 min |
| ~165-180 (RTX 5090, IQ2_XS) | ~2-3 s | ~5,800 | ~17 s |

Someone who asks while another answer is being written waits that long on top of their own. If your machine reaches
numbers like ours, the waits stay short. On a slower machine, every wait grows in proportion to how much slower it is.
In one office of about ten people (measured on 273 real requests with an earlier model) **at most two requests ever
overlapped**, so short chat answers rarely collided. Long documents are what users notice.

`"parallel": 2` does not raise the total on a layer split today. Each request in a batch slot decodes without MTP
drafts (#857), so the second person sees text sooner but both finish about when they would have anyway. On
**one GPU**, 0.1.40's `--batch-mtp` keeps the drafts in the batch. It does not support a layer split yet (BATCHING.md).

## The connection

Admin Panel -> Settings -> Connections -> OpenAI API:

- URL `http://127.0.0.1:8080/v1`, plus the API key if you set `api_key` in `strata-<model>.json`.
- The model then appears under the name Strata reports (`/v1/models`). Create a **model** on top of it in Open WebUI
  (Workspace -> Models) for the system prompt, tools and the settings below, and give that one to your users. Models
  and tools are private when created: share them (Access) or users will not see them.

If Open WebUI runs on another machine or in a container, Strata's Host and Origin checks apply. See "The trap: Host ...
is not allowed" in [LLAMA_SWAP.md](LLAMA_SWAP.md) and `STRATA_ALLOWED_HOSTS` / `api_key` in [DETAILS.md](DETAILS.md).
We have not tested that setup.

## Trap 1: images, and capabilities that are all on

Open WebUI marks a new model as able to do everything: vision, image generation, code interpreter and more. It does
not read Strata's `/v1/models` (`"input_modalities": ["text"]` without the encoder). If your Strata was set up without
images and a user pastes a picture, the request fails with a 400:

```text
this server was started without the vision encoder (run setup again and choose 'vision'), so it cannot read images
```

Because Open WebUI resends the picture with every later turn, **every following message in that chat fails too**. It
looks like the chat "hangs" on documents. Fix it in either of two ways:

- turn **Vision** off in the model's capabilities (Workspace -> Models -> the model -> Capabilities), and also image
  generation if you have no image backend; or
- give Strata the encoder. Setup can install it (`--vision cpu` or `gpu`). By hand, it is the `vision` section of the
  config plus `--vision` in `args`, and both are needed: without the flag the engine refuses images
  (`this engine was started without --vision`). On the CPU encoder we measured an 800x500 picture as 330 prompt tokens
  and 9 s for the whole answer, with exact text transcription.

## Trap 2: the task model queues behind the answers

Open WebUI sends small background requests (chat titles, tags, follow-up suggestions, search queries) to its **task
model**: Admin Panel -> Settings -> Interface -> **External Task Model**. Strata is an external (OpenAI-API)
connection, so the "Local Task Model" setting does not apply to it. If the external one is unset, or names a model that
is not available, Open WebUI uses the chat's own model, which is Strata.

With one slot, these requests queue like any other: the title of a new chat arrives after the answer, and the next
user waits for it too. That is acceptable for a few people. The alternative is a small model on another server (we
used qwen2.5:3b on Ollama), but it needs its own VRAM next to Strata's expert cache. On cards already full, the
2.8 GB it took cost Strata expert-cache slots (5.6 -> 8.3 GiB on one card when we removed it).

## Trap 3: context compaction and attachment limits

Open WebUI's context compaction (Admin Panel -> Settings -> Interface -> Context Compaction; its token threshold is
`CONTEXT_COMPACTION_TOKEN_THRESHOLD` on the first start) summarises old turns once a chat passes its threshold. The threshold has to fit **Strata's** context, not the
default. Ours is 120,000 with a 262,144 context, which leaves room for the answer and the next attachment. The
setting lives in Open WebUI's database once saved, so changing the environment variable later has no effect.

With RAG set to full context, an attachment goes whole into the prompt. A large one is then a long prompt read, the
"long document" row of the table above, and everyone waits. Decide how large an attachment may be before it is
handled another way (retrieval, or a tool that reads it in parts). We put the limit at 80,000 tokens per document and
100,000 for all attachments together.

## Trap 4: thinking effort

Open WebUI does not send `reasoning_effort` unless you set it in the model's advanced parameters. Without it, Strata
uses its own default. The Chat settings saved on Strata's page (`<config>.shared-settings.json`) also apply to other
apps, and the server says so at start (`other apps use the Chat settings: reasoning_effort=medium`). With `high`,
short context-compaction requests ran out of tokens while still thinking and came back empty. `medium` fixed that for
us.

## Measured with this setup

- Decode ~55 tok/s at 4K-128K prompts, prompt 880 / 1,917 / 2,221 tok/s at 4K / 32K / 128K
  ([report](../bench/results/2026-10-05-community-2x-rtx-4060ti/)).
- Needles found 6/6 up to 248,345 prompt tokens. A 248K-token prompt was read in ~108 s.
- With an Italian draft vocabulary (proposed as `--draft-vocab it` in #1259), Italian answers decode +14.6% on this
  machine.
