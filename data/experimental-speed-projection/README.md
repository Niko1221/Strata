# experimental-speed-projection (EXPERIMENTAL, off by default)

`Qwen3.8-Flash-Next-experimental-speed-projection.gguf` is a 480 KB control vector for Qwen3.8-Flash-Next: one unit
direction of 2,560 values per layer (`direction.1` .. `direction.47`, GGUF `controlvector` format). The JSON beside it
is the configuration its authors selected and their scores.

**What it does.** Turned on, Strata removes each direction from the residual stream after layers 4-44
(`h -= (h . v) v` on every hyper-connection stream), the same operation as llama.cpp's `--cvec-mode project` in the
package's patches. The package's own documentation describes the vector as a **refusal-direction projection**: the
model declines far fewer requests (it reports 1 of 50 vs 50 of 50 on its test set), and removing refusals removes a
safety behaviour - you are responsible for what the model writes with it on. It also shifts the model's predictions
slightly on ordinary text (measured here: `bench/results/2026-09-27-esp/`). It is not an optimization in the engine:
on the same text it costs 0.2-0.4% per token; a chat's speed with it on depends on the text the model writes.

**Turning it on.** `START-HERE.bat --setup` asks (default: off), or pass `--experimental-speed-projection on`. With it
loaded, the web app's Sampling drawer and the API field `"experimental_speed_projection": false` switch it off per
request. See `docs/DETAILS.md`, "Experimental speed projection".

**Origin and license.** A local package of the vector (the published original uses filenames containing "refusal";
the bytes are the same), made from Qwen3.8-Flash-Next activations, so the Qwen Community License 1.0 of the model
applies to it. Made for the original Qwen3.8-Flash-Next (not Swift 1.5).

## A second, separate choice: `...-l8-33.gguf` (layers 8-33)

`Qwen3.8-Flash-Next-experimental-speed-projection-l8-33.gguf` (268 KB, GGUF `controlvector`, one unit direction of
2,560 values per layer, `--control-vector-layer-range 8 33`, strength 1.0, `--cvec-mode project --cvec-dir per-layer`)
is a second vector for the **original** Qwen3.8-Flash-Next in any quantization setup offers: the `qwen` (ISTA),
`unsloth` and `nvfp4` families (the GPTQ pack, `NVFP4`; not NVIDIA's checkpoint, see below). It is a separate, opt-in choice beside the vector above, which is
unchanged (same file, same option, layers 4-44, `qwen` and `coder`). Setup asks for it with its own question (default
no) or `--experimental-projection-l8-33 on`; not for the Coder or Swift 1.5. **The two are never on at once**: both
options on the command line stop setup naming both flags, setup asks about this one only after the other is answered
no, re-running setup replaces one with the other (a config is recognised by its vector file), and the server refuses a
config whose `args` load both. It is the same mechanism and the same per-request switch as the vector above
(`"experimental_speed_projection": false`, or the web app's switch, turns whichever one is loaded off). The JSON beside
it has the per-layer strengths and scores.

**Origin.** Made by sergqwer (github.com/sergqwer/strata-nvfp4, `data/uncensor/qwen-nvfp4-gptq-uncensored.gguf`,
sha256 `a2ec3e99...46b5`) from the activations of that fork's GPTQ all-NVFP4 quantization of Qwen's BF16 weights:
per layer, the difference of the mean residual at the 7 assistant-header positions of 320 harmful and 320 harmless
prompts (mlabonne's harmful_behaviors / harmless_alpaca, chat template, thinking on), summed over the 4
hyper-connection streams, unit-normalized. About 30 variants (layer window, strength, direction source) were scored on
held-out prompts. Layers 8-33: strength past layer 33 is what made the model end its thinking early. Because it comes
from Qwen3.8-Flash-Next activations, the model's Qwen Community License 1.0 applies to it.

**Measured** (on the fork's 0.1.41/0.1.42-era code, RTX 5090, 2026-10-10; not re-run on 0.1.45 beyond the check
in the PR). Against the stock model of the same quantization. KL is teacher-forced on 5,153 rows of code, prose and
the model's own replies (exact mode); the "</think> shift" is the mean change of log p(`</think>`) over the model's
own thinking (above 0 = ends thinking sooner); refusals are counted by a phrase classifier on the first 300 characters
(`tools/refusal.py` in the fork), 80 new tokens, greedy.

GPTQ NVFP4 pack, this vector against the vector above (layers 4-44):

| | this (8-33) | above (4-44) |
|---|---|---|
| refusals, validation set, thinking off | 4/96 | 4/96 |
| refusals, held-out 104 EN, thinking off | 3/104 (1 real, the rest disclaimers followed by an answer) | 3/104 |
| refusals, held-out 104 EN, thinking on | 0/104 | 2/104 |
| KL, all rows | 0.0259 | 0.0433 |
| KL, the model's thinking text | 0.0144 | 0.0298 |
| </think> shift, thinking text (nats) | +0.019 | +0.092 |
| thinking tokens on a long agentic request, mean of 5 greedy runs (stock 1,699) | 1,947, 0 loops | 1,825, 0 loops |

(The stock model refuses 96/96 and 102/104 of the same prompts with thinking off.)

Other quantizations (this vector against the above, both against that quantization's own stock; KL and shift on a
smaller subset of 4,818 thinking rows plus the base texts; held-out 104 EN prompts, thinking off):

| quantization | refusals this / above (stock) | KL, base texts this / above | </think> shift, thinking this / above | thinking-length runs (3, greedy) |
|---|---|---|---|---|
| ISTA Q2_0 | 1/104 / 1/104 (30/30 of a 30-prompt subset) | 0.016 / 0.027 | +0.027 / +0.087 | stock 965, LOOP, 1,766; this 1,306, 1,350, 906 |
| ISTA IQ2_XS | 1/104 / 1/104 (30/30) | 0.017 / 0.031 | -0.003 / +0.123 | stock and this loop on 3/3 and 2/3 (the quantization loops by itself) |
| Unsloth UD-Q4_K_XL | 4/104 / 4/104 (30/30) | 0.028 / 0.042 | +0.050 / +0.163 | stock 888, 1,183, 1,731; this 1,204, 1,255, 1,343 |

Not offered for NVIDIA's own checkpoint (`NVFP4-NVIDIA`): the NVIDIA Open Model License does not allow reducing that
model's safeguards, so setup leaves it off there and says why. Not measured: the other ISTA and Unsloth sizes. The fork's and this table's numbers come from its own harness; the 104
held-out prompts are small, so differences of 1-2 refusals are noise.

**Safety.** Off unless chosen at setup. It removes the model's refusal behaviour (a safety behaviour): you are
responsible for what the model writes with it on.
