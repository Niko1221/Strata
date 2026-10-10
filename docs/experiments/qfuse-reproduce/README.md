# Reproduce the commit regression

Build the pinned current upstream as the parent. Apply `git apply --unidiff-zero isolated-guard.patch`
to a separate checkout of that same parent and build the corrected arm. Edit `plan.json`
to point to those two binaries, your Q8 pack/native GGUF, MTP weights, Python
environment, and client source checkout. The paths are recorded examples.
Run each listed case with `python case.py LABEL`. The parent-on negative
control is expected to fail the arithmetic-function check. The other six
cases should pass. Compare the three token pairs listed in the result JSON.
No network listener is opened; requests use engine stdin/stdout.
