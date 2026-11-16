# Q38 delivery legacy CUDA evidence selection

This bounded task starts at primary main
74a0bd425f82b440cff3844e7f0f46a3147d5622. It selects the complete actual
remeasured CUDA profile and the freshly completed ten-case legacy GPU proof.
The root controller observed GPU completion, exit 0 and drained Job 0 before
selection. This worker ran CPU boundary checks only. Stable installation and
publication remain 0.5.9; releaseQualified=false.

The selected profile is byte-identical to
H:/Strata/evidence/upstream-sync-20261116/native-q38-legacy-refresh-r1/candidate-measurements.json,
SHA-256 1db8c735238f98167cb676eec672a69586c7ee70e73a42e949a5eb53de8bd621.
All 451 resource entries and amounts match the retained historical profile and
base Git source exactly. No entries were added or removed. The actual source
digest before and after the rebuild/capture is
39e2edfabca66d1a6467d7fb8769b7948c7e361d14debac75465b73a806c1c07,
replacing profile binding
1c7811ccd9ebe8a853469b468ea196757a6d95332fe030b3e5f8d1512aedeaef.
Compiler key remains
3d02d9aa3a340546a098972927f76942ce6b1bb54348a6699d26fc9dc9195d65;
vendored ggml remains
vendored-sha256:00d79121e61703168b5f335234d8b391edd3192560b5bc98216a9c8fe8401af1.

This is a real five-job rebuild and compiler-resource recapture. Six actual
outputs changed: strata.exe, strata-q36.exe, strata_kernels.lib,
strata_engine.lib, strata_prefill.lib and strata_qwen35.lib. All eleven current
executable/library outputs match after.json; all nine profile library rows
match current actual bytes. The four changed measured-library SHA-256 values
are d62d3ff2033a84ffbedf96d92c7b7d7bcfb8c2fa0d2afb54daae2e730ed4f44b,
fad710b18e7fb34c1ed15130c28293953934ef5135fc258d6d6aee21a5f636ef,
290ed735be311901df2ab1acf6b32abd87214da0aa146a90e3d9508ac711fe71 and
86c825910a408315687bffef94a390917aafe6e09707a488c1d2391b8031c0c9,
respectively. This is not source-only relabelling. Old profiles and proofs remain.

The fresh GPU binding selects actual strata.exe SHA-256
b00e4ee065740fb9b25b57d946103096d4af4078989b316c6c62077e3c62d73f and
strata-q36.exe SHA-256
2eb40da0d574031eaa685939dbaacbb1eb1335d7c35159b674fa1a4f5a88b858.
Both before/after hashes and the native digest are unchanged during the proof.
The four selected GPU_PINS are binding
62abfb79bec6a56693a5b8ab65546d16edcec0d1debdfe49313d861877a09788,
report 24c5dae74cadff5b91eac19a66f334f935bc21598250b13b608844ef8ed4bc73,
guard e885bbbe1f8418d1b94cdd8cd5c8429ea6fed366a4e500555e9b0a1734042a5e
and JUnit b239f5cfcbebf8b77a88d5252b9dc3f3f60f4eff5f0001219693b0d4f7054e8f.
Their paths are the refresh directory's binding.json, parent gpu-q38-legacy-r1.json,
parent receipts/gpu-q38-legacy-r1.json and parent gpu-q38-legacy-r1.xml.

Actual JUnit contains the exact ten cases, without failures or skips. Execution
was serial one under a 900-second outer guard. GPU admission/minimum free memory
was 14822/10810 MiB, above the unchanged 768 MiB reserve. Guard RAM admission
was 13279764480 bytes and sampled minimum was 10702970880 bytes. Native/observed
exit codes were 0, termination was confirmed and Job active count was 0. No
diagnostic Job cap or RAM exception was used; RAM, commit and disk floors held.
Refresh and capture guards also passed, drained and retained their nominal floors.

Only the baseline JSON, four GPU_PINS values, directly corresponding acceptance
fixtures/tests and this record change. Existing source/binary equality checks
in final_native remain intact. The separately compiled Q38 runner still belongs
to 6ee425434c98277555332c7934e1ac51436eb198; later delivery/consumer HEADs are
not relabelled as its compilation source. Q38 adapter tests and actual Japanese
model evidence are independent of this ten-case legacy numerical proof.

The guarded focused check passed 42 tests, with 120 deselected and no skips,
in 3.80 seconds. Final checks passed the same 42 tests in 3.48 seconds after
AST-preserving formatting limited to changed test functions. Tests check
complete profile selection, exact source/compiler/ggml and library binding,
the four fresh artifact hashes, stale/missing/altered artifacts, changed
executables, malformed Jobs, exact JUnit case coverage and all existing
resource floors. Fatal Ruff checks, unchanged production AST outside GPU_PINS,
unchanged 451 entry amounts and diff whitespace checks passed.

Commands use the existing custom guard with offline uv and the existing
Python 3.12 interpreter. Prefix is
`C:/Users/downl/.local/bin/uv.exe run --no-project --offline --python C:/Users/downl/AppData/Local/Programs/Python/Python312/python.exe python -B _docs/upstream-sync-20261116/check.py`.
First suffix is
`--seconds 300 native-q38-legacy-bindings-focused-r1 python -B -m pytest -q _docs/stable-060-20261116/test_acceptance.py -k 'selected_profile or frozen_gpu or junit_exact' --tb=short -p no:cacheprovider --junitxml=H:/Strata/evidence/upstream-sync-20261116/native-q38-legacy-bindings-focused-r1.xml`.
Final suffix is `--seconds 300 native-q38-legacy-bindings-final-r1 python -B -c CODE`;
the exact formatting/AST/Ruff/pytest code vector is retained in that guard's
JSON receipt. Both owned Jobs ended exit/native 0 with active count 0, no cap
or nominal RAM breaches. The final receipt SHA-256 is
da8e9ebd718de05847ac91cf54b975a7b4c605b1312fa374431e5d58c6864f64.

Applied controls are the canonical PC prompt, PC/Codex/project AGENTS, common,
implementation-start, skill-selection, application and Python SOPs; applied
skills are milspec-codex-standard, documentation-and-adrs and review-only
agent-self-review-evolution. The custom guarded runner uses the existing pinned
interpreter through offline uv; no dependencies were resolved or installed.
No formal MILSPEC certification is asserted. Self-review-event: ticket_completed.
No governance or memory update was needed. No formal checks failed in this task.

The unchanged broad 312-test historical replay was not repeated because only
measured evidence selection changed. Generic strict spill thresholds and their
historical failures remain intact. This task does not build/sign components,
install, publish, stop other runtimes or qualify general model quality,
full-window inference, performance or GPU concurrency. Rollback is a scoped
revert of this selection commit, preserving actual measurements and old proofs.
The controller must obtain independent review and continue final delivery gates.
