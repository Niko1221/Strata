# Extended local tools validation

Author: **Francisco Angulo de Lafuente** · [ORCID](https://orcid.org/0009-0001-1634-7063)

All **22 profile, reader and mix checks passed** inside the official extended
tools suite on contribution revision `6d4bea0e44acba39f2679d75a13de008546a1ab1`.
The extended run passed 1,675 checks. Its six failing cases also failed on the
unmodified upstream base in the same Windows/Python installation, with identical
failure messages after replacing checkout paths.

| Scope | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Profile/reader/mix subset, included in the row below | 22 | 0 | 0 |
| Official top-level tools suite on the contribution | 1,675 | 6 | 78 |
| The same six failure cases selected on the upstream base | 0 | 6 | 0 |

The records are [head-cases.json.gz](head-cases.json.gz),
[base-cases.json.gz](base-cases.json.gz) and [summary.json](summary.json).
They preserve every testcase outcome in the two JUnit files. The extraction
omits host metadata and traceback bodies and replaces machine-specific paths;
it is not the original XML. Original XML hashes and the exact extraction method
are recorded. [convert_junit.py](convert_junit.py) is the executed extractor.

The contribution run used the repository's unchanged command
`python tools/check --only py-tools`, with explicit scratch/result paths and
`--keep`. The control collected the base's top-level `tools/test_*.py` files and
used pytest `-k` to execute only the six named failing cases. It did not rerun the
entire base suite. The selected names and environment are in the summary.

## Observations

- The overall extended tools step failed. The six cases concern a chat-template
  hash, package-install stamp tests, and SYCL build-guard shell paths. Reproduction
  on the base establishes that these observed failures are present without this
  contribution in this environment; it does not establish that every CI runner
  has the same failures.
- This was Windows/Python 3.13.7 with pytest 9.1.1 and NumPy 2.5.4. The GitHub
  workflow specifies Python 3.12. These are local results, separate from GitHub's
  maintainer-controlled CI status.
- The 78 skips cover platform-specific cases and unavailable optional dependencies;
  their per-case reasons remain in the extracted record. They are not counted as passes.
- An existing file-recognition fixture temporarily allocated large, non-sparse
  fake GGUF files on this Windows filesystem. Those temporary files were cleaned
  by the test. No model inference or GPU computation was performed. Reproducing
  the full suite on Windows requires a scratch volume with sufficient capacity;
  the observed allocation and run time are recorded in the summary.
- All [profile measurements](../README.md) and [frozen live-pilot gates](../live-pilot/README.md)
  remain unchanged. This validation does not establish a runtime speedup or upstream acceptance.
