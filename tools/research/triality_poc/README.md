# Independent paper PoC

This directory is an offline research import. It does not enable a Strata,
Prism or TheTom runtime feature. The production default path is unchanged.

## Provenance

The supplied archive is `triality_llm_paper_poc_20261023.zip`, SHA-256
`f9358b4eec96aa302df926a3a7dbf1b8e5401263cea0ada25612f0c0bae1d009`.
All 61 manifest payloads passed verification. Its manuscript matches the adjacent
supplied manuscript byte for byte, SHA-256
`d734e0308e96a1c338e2f9e2f9dfa0873c8a431a77d1d0474ffbf080bc95f89b`.
`provenance/intake.json` records the full manifest. `provenance/imported_files.json`
maps the 42 byte-identical imported files to their archive paths and hashes.

Immutable originals are stored outside Git at
`H:/Strata/reference/triality-paper-20261023`. Extraction rejects existing targets,
traversal, Windows path ambiguities, case collisions and special files; each
extracted file is read-only. This prevents accidental edits, not deliberate
tampering. The bundled `.pt` files were extracted as opaque bytes only. They were
never loaded and are excluded from Git. No notebook or archive instructions ran.

The original package modules, two original test files and paper are unchanged.
The original package metadata is an inert provenance snapshot under
`provenance/supplied`; no dependency installation or project configuration ran.

## Reference codec and diagnostics

`triality_poc.reference_absmax.CODEC_ID` is
`paper-reference-absmax-group8-v1`. The named functions
`quantize_reference_absmax` and `dequantize_reference_absmax` expose the supplied
codec: group-eight absmax, input-dtype scales, `2**bits - 1` signed levels and
little-endian packed codes. The original `core.quantize` API remains available
for reproducibility. This is not a Prism, TheTom or TurboQuant binary format.

The supplied code already implements matrix-free Fisher weighted centring,
score-KL checks and a shared-noise shrinkage experiment with disjoint calibration
and evaluation sketches. The tangent least-squares oracle in `validate.py`
remains offline. `tests/test_dense_diagnostics.py` independently compares small
dense Fisher matrices and full cross-key covariance against those contracts,
and checks the KL identity and reference quantiser levels. No duplicate
diagnostic implementation was added to the runtime.

## Supplied Linux CPU evidence

`evidence/supplied_linux_cpu` is supplied evidence, not a new run on this PC.
It used Linux, Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5 and pytest 9.0.2.
It reports 25 tests passed and one CUDA skip, 12 numerical checks and seven
targeted mutation detections. The three seeds train independent tiny synthetic
teachers. They do not measure natural-language quality, Bonsai, the large Strata
Q2_0, CUDA throughput or production cache behaviour.

| Supplied arm | Teacher vocabulary KL, mean ± seed SD | NLL | Pooled PPL |
| --- | ---: | ---: | ---: |
| Exact teacher | 0 | 0.756865 | 2.131584 |
| Single 2-bit K | 0.169926 ± 0.042527 | 0.933274 | 2.542822 |
| Single 6-bit K | 0.0000306 ± 0.0000058 | 0.757004 | 2.131880 |
| Triality 3 × 2-bit K | 0.027913 ± 0.009309 | 0.786904 | 2.196586 |
| Generic 3 × 2-bit K | 0.033571 ± 0.007736 | 0.793094 | 2.210224 |
| Triality 3 × 2-bit K, V 3-bit | 0.032356 ± 0.009608 | 0.791292 | 2.206246 |
| Triality 3 × 2-bit K, trained Q | 0.023622 ± 0.005632 | 0.777320 | 2.175633 |
| Generic 3 × 2-bit K, trained Q | 0.029545 ± 0.005193 | 0.784560 | 2.191443 |
| Trained Q, exact KV | 0.003564 ± 0.001494 | 0.754845 | 2.127282 |

The Triality Q-training paired mean KL reduction is 0.004291 nat/token, but the
exploratory three-seed 95% paired t interval is [-0.004898, 0.013481], including
zero. Statistical superiority is not established. Single 6-bit K wins this
fixture on fidelity and payload: 5,376 bytes versus 6,400 for three 2-bit K
views, including FP32 V and group scales per layer and sequence. Shared profile
bytes and scratch are additional, so nominal bit counts are not total capacity.

![Supplied Linux CPU synthetic vocabulary KL for three independent teachers](paper/figures/tiny_vocab_kl.png)

## Fresh Windows CPU verification

Fresh logs are separate under `evidence/windows_cpu_20261023_final`; the initial
attempt and its failures are retained under `evidence/windows_cpu_20261023`.
Public snapshots replace local user/interpreter/workspace paths with documented
placeholders. Raw logs remain in the local H: evidence directory.

The existing interpreter is Python 3.12.10, PyTorch 2.5.1+cu121, NumPy 2.2.6 and
pytest 8.3.5. No packages were installed. Child processes explicitly set
`CUDA_VISIBLE_DEVICES=-1` and disable unrelated pytest plugins and parent
conftest discovery. This is CPU qualification only; the installed CUDA wheel
does not establish support for the PC's sm_120 GPU.

The default SDPA route produces NaNs in the tiny-model FP64 fixture on this
environment. Its failure is retained. A separate explicit CPU `SDPBackend.MATH`
check runs the unchanged reference tests. The latter cannot qualify the default
dispatch, GPU execution or an exact reproduction of the supplied environment.
The pre-existing pynvml deprecation warning remains visible in the logs.

| Fresh Windows CPU check | Result |
| --- | --- |
| Unchanged reference suite, default SDPA | 24 passed, 1 failed, 1 CUDA skipped |
| Unchanged reference suite, explicit MATH SDPA | 25 passed, 1 CUDA skipped |
| New dense-oracle and safe-intake fixtures | 17 passed |
| Original numerical checks | 12 / 12 passed |
| Original seven targeted source mutations | 7 / 7 detected |

The context-managed MATH rerun is in
`reference_pytest_math_sdpa_context_managed.txt`. An earlier compatibility probe
passed its tests but left a context open during interpreter shutdown; its cleanup
warning is retained in the preceding log. The harness now closes that context.
The overall native reproduction status remains false because of default SDPA.

From this directory, using an existing interpreter with the listed packages:

```powershell
& $Python scripts/reproduce.py --math-sdpa-check --out $NewLocalEvidenceDirectory
```

The output directory must not exist. This command never installs dependencies,
loads supplied checkpoints or starts training. A failing default-route check
keeps the aggregate result false even when explicit MATH compatibility passes.
Use `intake.py --help` for verification/extraction; that standalone tool requires
Python 3.12 or later and rejects an existing reference destination.
