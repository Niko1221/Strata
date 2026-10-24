# Community benchmark: L40S-48Q (vGPU, 48 GB), Strata 0.2.1, six Unsloth quantizations of Qwen3.8-Flash-Next, with a 3D task suite in Blender

This is the follow-up to [the 0.2.0 folder](../2026-10-20-community-l40s-48q-0.2.0-unsloth-quants/). It uses the same machine, model files, run config and method. The runs took place overnight, 2026-10-20 to 21, on Linux. Three things are new:
- the same six GGUFs on **Strata 0.2.1**;
- **interleaved 0.2.0 / 0.2.1 starts** on UD-Q4_K_XL, so the two versions are compared under the same conditions;
- **a 26-task 3D suite**. The model writes a `bpy` script, Blender 5.2.1 runs it headless in an empty scene, and a checker measures the result.

**Short answer.**
- **0.2.1 gives the same speed and the same answers as 0.2.0 on this card.**
  - In the interleaved pairs, decode is 94.2 against 89.9 tok/s at 8K (+5%, about the spread between runs) and 101.6 against 102.5 at 32K.
  - Between the two versions, the first-token KL of Q8_0 is 0.012 (60 prompts) and 0.018 (160 prompts). That is the size of the 0.1.45 → 0.2.0 difference.
- **The quantization ranking does not change.** UD-Q4_K_XL is still the knee: KL 0.031 / 0.057 against Q8_0, and about 30% faster.
- **3D: every quantization passes 24 to 26 of the 26 tasks.**
  - UD-Q5_K_XL and UD-Q6_K_XL pass all 26.
  - Each of the five failures is a call to a `bpy` attribute that does not exist in Blender 5.2. None is a wrong position, size or rotation.

**Strata 0.2.2 came out this morning** (2026-10-21 11:10 UTC), after these runs. Its notes list engine changes for Gyro packs, AMD and Intel cards, multi-GPU roles and two-socket NUMA. They list nothing for a plain GGUF on one NVIDIA card. We have not run it.

## What changed from the 0.2.0 runs

- **Packages:** `strata-control-0.1.46.2-linux-x86_64.tar.gz` and `strata-engine-0.1.46-linux-x86_64-cuda-cuda12.tar.gz`, checked against `SHA256SUMS.txt` ([scripts/baixar.sh](scripts/baixar.sh)).
- **Packs:** rebuilt with the 0.2.1 `tools/iq_pack.py`. The packs are byte-identical to the 0.2.0 ones. We still set `STRATA_GGUF_PY`, because the control package still does not include `gguf-py`. 0.2.1 looks in more places (the install folders, `STRATA_INSTALL_DIR`), and its message now says to run `setup.py` once or set the variable. For #3269 that is the fix. We did not run `setup.py`, so our install, unpacked and run with `serve/server.py`, still needs the variable.
- **Config:** the same args and the same setup `env` as in 0.2.0. Only the engine, the control folder and the pack paths changed ([configs/](configs/)).
- **RAM:** these runs had 503 GiB, the 0.2.0 runs 251 GiB. The VM memory was on the GPU's NUMA node; see the 0.2.0 Limits. Cold starts are shorter here (33-69 s against 39-198 s) because every model now fits in the page cache. Speed is not affected, since every expert is in RAM either way.
- **Not repeated on 0.2.1:** BBH (it did not separate the quantizations on 0.2.0), the 0.1.45 reference and the 1M start check.

## Results

### Speed (Strata 0.2.1, 262,144-token context)

| Model | Decode tok/s 8K / 32K / 128K | Prompt tok/s 8K / 32K / 128K | strata_bench chat / agent decode | strata_bench 4K / 16K prompt |
|---|---|---|---|---|
| UD-Q2_K_XL | 130 / 126 / 118 | 1,428 / 3,434 / 3,417 | 136.9 / 149.8 | 2,736 / 3,465 |
| UD-Q3_K_XL | 111 / 114 / 110 | 1,471 / 3,485 / 3,416 | 125.5 / 125.5 | 2,710 / 3,482 |
| UD-Q4_K_XL | 103 / 99 / 92 | 1,324 / 2,825 / 2,734 | 96.8 / 106.6 | 1,728 / 2,895 |
| UD-Q5_K_XL | 94 / 87 / 86 | 1,208 / 2,308 / 2,179 | 87.6 / 96.8 | 1,258 / 2,297 |
| UD-Q6_K_XL | 83 / 86 / 78 | 1,173 / 2,043 / 1,926 | 81.9 / 85.4 | 1,094 / 2,026 |
| Q8_0 | 74 / 74 / 69 | 1,017 / 1,708 / 1,615 | 69.8 / 81.3 | 901 / 1,687 |

### The same model and config, 0.2.0 → 0.2.1 (one session each)

| Model | Decode 8K / 32K / 128K | Prompt 32K / 128K | HumanEval+ plus | GSM8K | IFEval prompt | MATH geometry |
|---|---|---|---|---|---|---|
| UD-Q2_K_XL | 126 / 122 / 111 → 130 / 126 / 118 | 3,194 / 3,425 → 3,434 / 3,417 | 89.0 → 89.6 | 98 → 97 | 88 → 83 | 95 → 96 |
| UD-Q3_K_XL | 107 / 108 / 105 → 111 / 114 / 110 | 3,484 / 3,417 → 3,485 / 3,416 | 87.2 → 87.8 | 99 → 98 | 83 → 83 | 96 → 93 |
| UD-Q4_K_XL | 98 / 99 / 96 → 103 / 99 / 92 | 2,828 / 2,739 → 2,825 / 2,734 | 91.5 → 89.0 | 99 → 99 | 84 → 82 | 95 → 96 |
| UD-Q5_K_XL | 89 / 90 / 83 → 94 / 87 / 86 | 2,288 / 2,167 → 2,308 / 2,179 | 89.6 → 90.9 | 98 → 99 | 82 → 84 | 96 → 96 |
| UD-Q6_K_XL | 82 / 82 / 78 → 83 / 86 / 78 | 2,037 / 1,928 → 2,043 / 1,926 | 89.0 → 90.9 | 99 → 99 | 86 → 83 | 96 → 97 |
| Q8_0 | 75 / 73 / 72 → 74 / 74 / 69 | 1,703 / 1,613 → 1,708 / 1,615 | 89.0 → 90.9 | 99 → 99 | 86 → 82 | 97 → 95 |

Single sessions move by up to about 5% in decode and 5 points on a 100-item task, in both directions. The pairs below are the comparison to trust.

### Interleaved starts, UD-Q4_K_XL

The order was 0.2.0, 0.2.1, 0.2.0, 0.2.1, 0.2.0, 0.2.1, each start with its own fresh server.

| Version | Decode 8K, runs 1 / 2 / 3 | Decode 32K, runs 1 / 2 / 3 | Prompt 8K / 32K, mean |
|---|---|---|---|
| 0.2.0 | 91.3 / 90.7 / 87.6 (mean 89.9) | 102.1 / 99.3 / 106.1 (mean 102.5) | 1,678 / 2,797 |
| 0.2.1 | 93.9 / 97.8 / 90.9 (mean 94.2) | 101.0 / 103.0 / 100.9 (mean 101.6) | 1,825 / 2,817 |

The 8K prompt mean of 0.2.0 includes one slow first run (1,305 tok/s). Without it, the two versions are at 1,865 and 1,825.

### Quality against Q8_0 (same engine, first-token logits)

| Model | KL, 60 prompts | median / p90 | same top-1 | KL, 160 prompts | same top-1 | needles 128K |
|---|---:|---|---|---:|---|---|
| UD-Q2_K_XL | 0.221 | 0.173 / 0.365 | 51/60 | 0.406 | 118/160 | 3/3 |
| UD-Q3_K_XL | 0.116 | 0.068 / 0.291 | 54/60 | 0.144 | 135/160 | 3/3 |
| UD-Q4_K_XL | 0.031 | 0.014 / 0.080 | 54/60 | 0.057 | 143/160 | 3/3 |
| UD-Q5_K_XL | 0.016 | 0.009 / 0.049 | 57/60 | 0.029 | 150/160 | 3/3 |
| UD-Q6_K_XL | 0.010 | 0.006 / 0.026 | 59/60 | 0.020 | 151/160 | 3/3 |
| Q8_0 | - | - | - | - | - | 3/3 |

**Across versions:**

| Comparison | KL, 60 prompts | same top-1 | KL, 160 prompts | same top-1 |
|---|---:|---|---:|---|
| Q8_0, 0.2.1 against 0.2.0 | 0.012 | 56/60 | 0.018 | 153/160 |
| UD-Q4_K_XL 0.2.1 against Q8_0 0.2.0 | 0.034 | 56/60 | - | - |
| UD-Q2_K_XL 0.2.1 against Q8_0 0.2.0 | 0.217 | 52/60 | - | - |

The 0.2.0 values were 0.037 for UD-Q4_K_XL and 0.202 for UD-Q2_K_XL. The 0.2.1 values against the 0.2.0 Q8_0 fall in between, so the quantization error is the same on both versions. The UD-Q5/Q6 values are within the version-to-version difference of Q8_0 itself.

### Task accuracy

| Model | HumanEval+ base / plus (164) | GSM8K strict, 4096 tokens (100) | IFEval prompt / instruction strict (100) | MATH geometry, math_verify (100) |
|---|---|---|---|---|
| UD-Q2_K_XL | 93.3 / 89.6 | 97 | 83 / 87.7 | 96 |
| UD-Q3_K_XL | 91.5 / 87.8 | 98 | 83 / 87.7 | 93 |
| UD-Q4_K_XL | 92.7 / 89.0 | 99 | 82 / 87.1 | 96 |
| UD-Q5_K_XL | 93.9 / 90.9 | 99 | 84 / 89.0 | 96 |
| UD-Q6_K_XL | 93.9 / 90.9 | 99 | 83 / 88.3 | 97 |
| Q8_0 | 93.3 / 90.9 | 99 | 82 / 87.7 | 95 |

The settings match the 0.2.0 runs: thinking on at the server default, and `max_gen_toks=4096` for GSM8K and MATH. Image results for UD-Q4_K_XL and Q8_0 are identical to 0.2.0 ([tables.md](tables.md)).

### 3D tasks in Blender 5.2.1

**How it runs:**
- Each task is one prompt in Portuguese. There is one attempt per task, at temperature 0, with up to 8,192 output tokens and thinking at the server default.
- The system prompt tells the model three things: it writes for Blender 5.2 `bpy`, run headless in an empty scene (no default cube, camera or light), in metres with +Z up, and it answers with one `python` block.
- `scripts/bench3d.py` takes the last `python` block. It runs the block with `blender -b --factory-startup` in an empty scene, with a 180 s limit. `scripts/executor3d.py` then runs the task's checker from `scripts/tarefas3d.py`.
- The checkers read the evaluated scene, with a tolerance of 0.02 m. They check world bounding boxes, positions, rotations and directions, parenting, modifiers (and whether they were applied), materials, collections, camera aim, lights, keyframes and the render output.
- `bench3d.py validar` checks the suite itself. Every reference solution in `tarefas3d.py` passes its checker, and an empty script fails all 26.

**The 26 tasks, by category:**
- **measures (5):** a cube on the floor, a row of spheres, a box with applied scale, a cone pointing to +X, and a cylinder lying along X;
- **spatial relations (8):** a sphere resting on a cube, a rotated stack, a ring of cylinders, a table, a grid, stairs, a room, and chairs around a table;
- **camera and light (2);**
- **materials and scene (5):** a material, a collection, text, subdivision, and a render to a file;
- **hierarchy and modifiers (5):** parenting, array, mirror, a boolean hole, and a car with four wheels;
- **animation (1).**

| Model | passed | measures | relations | camera, light | material, scene | hierarchy, modifiers | animation | failed tasks | median s per task |
|---|---|---|---|---|---|---|---|---|---:|
| UD-Q2_K_XL | **24/26** | 5/5 | 8/8 | 2/2 | 5/5 | 4/5 | 0/1 | mirror, animation | 30.2 |
| UD-Q3_K_XL | **25/26** | 5/5 | 8/8 | 2/2 | 4/5 | 5/5 | 1/1 | collection | 27.2 |
| UD-Q4_K_XL | **25/26** | 5/5 | 8/8 | 2/2 | 5/5 | 4/5 | 1/1 | mirror | 42.6 |
| UD-Q5_K_XL | **26/26** | 5/5 | 8/8 | 2/2 | 5/5 | 5/5 | 1/1 | - | 30.9 |
| UD-Q6_K_XL | **26/26** | 5/5 | 8/8 | 2/2 | 5/5 | 5/5 | 1/1 | - | 35.4 |
| Q8_0 | **25/26** | 5/5 | 8/8 | 2/2 | 4/5 | 5/5 | 1/1 | text | 45.0 |

**Every failure is an API error**, raised before the checker runs:
- **mirror** (UD-Q2, UD-Q4): `MirrorModifier.use_axis_x`. The attribute is `use_axis[0]`.
- **animation** (UD-Q2): `Action.fcurves`. Blender 5's layered actions no longer have it.
- **collection** (UD-Q3): `name in bpy.data.objects` called with an object instead of a string.
- **text** (Q8_0): `TextCurve.align_horizontal`. The attribute is `align_x`.

Every task that measures geometry passed on every quantization. With one attempt per task, a single failure is within noise, so this suite does not separate the quantizations either. It does show that UD-Q2_K_XL builds correct 3D scenes about as reliably as Q8_0. Each model's answer and the script Blender ran are in `results/<run>/bench3d/`.

## Limits

- **One machine, a vGPU, one request at a time.** The speed rows are single requests per length, inside one server session per model. Only the UD-Q4_K_XL pairs repeat.
- **Same KL limits as in 0.2.0:** the KL covers the prompt path only, and there is no same-engine repeat.
- **The 3D suite is ours and small:** 26 tasks, one attempt each, checked against geometric tolerances. It tests correct `bpy` for simple scenes, not modelling quality.
- **0.2.2 not run.** See above.

## Files

| File | Contents |
|---|---|
| `tables.md`, `summary.json` | every table above, generated by `scripts/gerar-relatorio.py` |
| `results/021-<model>-256k/` | as in 0.2.0, plus `bench3d/` (`resultado.json`, and per task `<task>.resposta.md` with the model's answer and `<task>.py` with the script Blender ran) and `kl-*-vs-020-q8_0-256k.json` for the cross-version KL |
| `results/ab020-q4-r*/`, `results/ab021-q4-r*/` | the interleaved starts |
| `configs/`, `engine-logs/`, `scripts/` | as in 0.2.0; the scripts include `bench3d.py`, `executor3d.py` and `tarefas3d.py` (tasks, checkers, reference solutions) |

Local paths are `<data>` and `<home>`. The comments in the scripts and the 3D prompts are in Portuguese.
