# Local Qwen3.8-27B dense native trial

The Windows CUDA 13 / Blackwell runner uses Strata's own loader and CUDA operations.
Choose **Use a local GGUF**, select the original file, check it, then choose
**Prepare and load**. A plausible dense Qwen3.8-27B model is identified from its
GGUF metadata, including renamed and fine-tuned files. A preview does not prove
that the complete tensor layout will load or that its answers are reliable.
The runner checks the complete model when it loads. The original file stays in
place. No replacement model, MoE pack or tokenizer download is made.

The signed native engine component must include the covered `strata-q38.exe`
helper. Control verifies the complete parent component inventory before selecting
it. An older component without that helper reports an error. Explicit PrismML
and TurboQuant adapter selections remain available through their existing routes.

The Terminal uses the same registration and serving configuration:

```powershell
strata models import 'H:/models/original.gguf' --plan
strata models import 'H:/models/original.gguf' --native-dense --context 4096 --kv auto --output-language ja
strata launch --model-path 'H:/models/original.gguf' --context 4096 --kv auto --output-language ja --yes --no-tui
```

Use `strata launch --model-path 'H:/models/original.gguf' --plan` to inspect the
launch without writing a configuration. Re-importing with no explicit choices
preserves previously saved choices. New registrations start at 4,096 tokens,
`auto` KV and Japanese output. In the app, the unchanged selection means keep
the saved choice, or use that default for a new model. Manage models can load a
registered dense model again; normal native instance controls stop and unload it.

Explicit `user_args` and hand-edited raw arguments take priority over registration
defaults for context, KV and language. When `args_written` records generated
arguments, unchanged generated values still follow the scalar settings. Invalid
or unsupported explicit arguments fail validation. Saved and running scalar
reports are reconciled with the effective arguments, including `model-default`.

Supported requested KV formats are `auto`, `f16`, `k8v4`, `spin8_k8v4`,
`f16_turbo2`, `f16_turbo3` and `f16_turbo4`. The native planner tries F16, then
Turbo4, then Spin8 for `auto`. It keeps the requested context and original weights.
If none fits the resource reserves, loading fails. The running engine's INFO and
service metrics report the effective KV format, allocation and selection reason.
Saved `auto` remains `auto` for future loads. Neither the app nor the Terminal
predicts admission from the preview.

The app offers 98,304 and 131,072 tokens as explicit requests. Full-input quality
and a five-minute first response at those lengths remain unqualified. Capacity
with a short prompt is different from processing a full window. Sequential
prefill can take minutes even on much shorter input; no speedup is promised.
MTP tensors, image input, multiple native sessions, prefix reuse, arbitrary
rollback and native private KV persistence are unsupported. Semantic memory and
workflow learning keep their existing settings. Each native request resets its
session; the service requires that capability receipt before accepting the child.

## Japanese output preference

`output_language: "ja"` adds a Japanese instruction before rendering the model's
original embedded chat template. It covers the final answer and any reasoning
the model visibly emits. It gives no access to hidden reasoning. Existing system
messages and tools remain present. `model-default` disables both that instruction
and the sampler constraint. These settings are specific to the dense native path;
another engine cannot claim it applied these constraints.

The native CPU sampler applies `ja-finite-v1`, a finite suppression of these
24 code points: `這們讓嗎與會說國體學發夠麼邊從來應對關實點處臺灣`.
It checks original GPT-2 byte-BPE tokens, including characters split across token
boundaries, before greedy or sampled selection and before streaming. A character
in literal user, tool or system input remains available for names, quotations or
code. Generated assistant history does not grant an exception. A bounded 24-bit
mask carries these exceptions; it contains no private input text.

This list includes old Japanese forms and does not classify the whole language.
Ordinary shared Japanese kanji, kana, ASCII code, JSON, control tokens and names
such as 𠮷 remain available. The finite list cannot exclude every Chinese sentence
or every traditional form. There is no weight modification, training, OpenCC
conversion or broad Han-character ban. If the constraint leaves no permitted
finite candidate, generation fails explicitly. State resets on each request,
cancellation, error and reload. Forced-logit diagnostics require `model-default`
and report that constraints are off.

The UI keeps the user's selected locale and sets the document language to it.
Font fallback can display a shared Han code point with a different glyph shape;
that is separate from the actual generated Unicode code point. UI language does
not select the model's output profile.

The current source integration has fixture and numerical regression coverage.
Actual checkpoint language quality, installed application behaviour and release
publication require their own recorded acceptance; source tests alone do not
establish those outcomes.
