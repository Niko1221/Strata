# ShinkaEvolve + NVIDIA NIM tuning setup (2026-11-20, OpenCode)

## Decision change

On 2026-11-19 the project had selected decision A: strict byte-identical FP32
logits (`2026-11-19_q38-performance-assessment_OpenCode.md:61-66`).
On 2026-11-20 the user explicitly changed this to a **tolerance-based numerical
contract** and requested mathematical/algorithmic speedups plus an
AI-native search using ShinkaEvolve with an NVIDIA NIM key. This record
supersedes decision A for future kernel work; any change must still be checked
against a documented tolerance and forced-token logits.

## Why ShinkaEvolve

ShinkaEvolve (SakanaAI) combines LLM proposals with an evolutionary archive.
The installed copy is `shinka` 0.1.0 from
`C:\Users\downl\Desktop\ShinkaEvolve-OSINT-main\ShinkaEvolve-OSINT-main`, which
includes a `local/<model>@<base_url>?api_key_env=<ENV>` OpenAI-compatible model
format (`shinka/local_openai_config.py`) and an NVIDIA NIM route in the egress
allow-list (`shinka/network_policy.py`). That makes NIM usable without code
changes to Shinka itself.

## Credentials

No `NVIDIA_API_KEY` was present in the Codex process environment. The key was
found in `C:\Users\downl\Documents\New project\hermes-agent\.hermes\.env`
(key name `NVIDIA_API_KEY`); the value was never printed or committed. Launch
scripts read it from that file at run time. The Codex `OPENAI_API_KEY` is a
placeholder (`sk-...`, length 6) and returns HTTP 401 from api.openai.com.

## Model selection

NIM `/v1/models` was queried with the Hermes key on 2026-11-20. The user chose
MiniMax 3.1; the available NIM id is `minimaxai/minimax-m3.1-flash`.
Embedding model: `nvidia/nemotron-3-embed-1b` (also listed by NIM).
Model strings in `shinka.yaml` use the `local/...@https://integrate.api.nvidia.com/v1?api_key_env=NVIDIA_API_KEY` form.

## Task scaffold (all under the approved temp directory)

- `C:\Users\downl\AppData\Local\Temp\opencode\shinka-q38\initial.py` — tuning policy with a tight `EVOLVE-BLOCK-START/END` region; emits JSON {chunk, tile_mb, kv, fp32_accum, fp8}
- `...\evaluate.py` — runs the pinned 9B fixture with warm-up, compares token output to a baseline policy, and writes `metrics.json` / `correct.json`
- `...\shinka.yaml` — 6 generations, 2 islands, `max_parallel_jobs: 1` (serial GPU), MiniMax M3.1 Flash for proposals/meta/novelty, NIM embed model
- `...\run_evo.py` — task-specific system message with known negative results (tiles/chunks, FP8, FP32 accum, alternative KV modes)
- `...\smoke.cmd` — loads the NIM key and smoke-tests `evaluate.py`

## Known negative results encoded in the task prompt

- Tile 64/128/256 MiB and chunk 512/256: no gain (571-581 ms vs 565-585 ms baseline)
- FP8: 616 ms, slower; plus field-reported SM120 low-precision hang risk
- FP32 accumulation: 832-913 ms vs 590-638 ms FP16 (measured this session)
- k8v4 / turbo / spin8: per-token attention path, slower for the 1024-token fixture
- INT8 projection pilots, fused WMMA, 8-block unroll: rejected in earlier records

## Safety / verification gates

- GPU benchmarks run serially; no user process is stopped.
- RAM 3 GiB / commit 4 GiB / VRAM reserve 768 MiB / C: 5 GiB / H: 10 GiB floors unchanged.
- Any candidate that changes tokens under `--seed 42` is scored invalid by the evaluator.
- The evaluator does not modify engine source or the working tree.

## Literature cross-check (web, 2026-11-20)

- ShinkaEvolve official getting-started guide confirms the dynamic model form
  `local/<model>@<base_url>?api_key_env=<ENV>` and the `EVOLVE-BLOCK-START/END`
  contract used in `initial.py`: https://sakanaai.github.io/ShinkaEvolve/getting_started
- The evolutionary loop is: select parent(s) from island archive -> LLM proposes
  a patch (diff/full/crossover) -> evaluate -> persist result -> migrate top
  solutions between islands. `combined_score` is the primary fitness.
- `max_evaluation_jobs` / `max_parallel_jobs` must stay at 1 for this GPU task:
  concurrent engine evaluations would violate the serial-GPU and VRAM-floor
  discipline recorded in `_docs/2026-11-20_q38-tensorcore-tuning_OpenCode.md`.
- Broader context (not adopted here): FunSearch/AlphaEvolve-style program search
  and LLM kernel generation use the same evaluate-and-archive structure; the
  installed Shinka fork is the applicable implementation for this workspace.

## Status

Scaffold created and configured for NIM/MiniMax M3.1 Flash. Smoke test was
launched but its engine process stalled; the cause is being isolated by
reproducing the evaluator's exact environment with a manual engine run.
Evolution was not yet started. Results and any adopted policy change must be
recorded in a follow-up implementation log.
