# Windows exact RoPE comparison: scalar math evaluation

The Windows hosted tools check reported a one-ULP RoPE table difference in all
seven Qwen-Image 2.1 cases. Token positions matched exactly. The Linux and macOS
checks and the preceding local Clang check passed. This record separates the
mathematical diagnosis, the small source correction, and the validation still
required on the hosted GCC compiler.

## Diagnosis and evidence

The independent reference remains the unchanged extracted sd.cpp source at
`2988060a4338a5dcd94721c964a8e87003c1ac8b`. Its original-file hashes, extraction
hashes and Git blob identities remain checked by `tools/test_dit_rope.py`.
The historical glibc table hashes and the exact byte comparisons are unchanged.

Hosted run `43791970812`, Windows job `133052276655`, reported table differences
of 266, 266, 266, 266, 1034, 1290 and 1292 elements. All seven first differences
were at index 2053: production `0xbf13067d`, reference `0xbf13067e`, maximum ULP
distance 1. The initial angle was the float value `2.5298221111297607`.

Private CPU probes using the installed Windows UCRT `sinf`/`cosf` and an actual
x87 `FSINCOS` function reproduced all seven difference counts, first indices,
first bit patterns and maximum ULP distances. This isolates a difference between
the two math implementations. It does not establish hosted generated assembly
or model/image quality. Each probe ran through the existing owned-Job guard
with unchanged RAM, commit and disk floors. All probe Jobs drained to zero.

The hosted image was Windows 2022 `20261112.366.7`. Its inventory lists GCC 14.2.0;
the pinned image installation script selects niXman MinGW, x86_64, POSIX threads,
SEH exceptions and UCRT. MinGW runtime v12 implements `sincosf` using x87.
GCC can combine separate scalar sin/cos calls into that combined routine.

Production used one local scalar angle, which permits this combination. The
reference materializes an angle matrix and reloads it between output stores.
The source difference and measured math signatures support compiler substitution
as the cause; direct hosted compiler validation remains required.

## Change

`src/image/dit/qi21_rope.cpp` now reads a materialized volatile float separately
for each of the four existing trig expressions. The separate reads preserve the
individual scalar evaluations instead of allowing one shared input to select a
combined math routine with different rounding. Positions, frequency arithmetic,
output layout and public interfaces are unchanged.

`tools/test_dit_rope.py` includes the compiler executable and its existing runtime
provenance in exact-comparison assertion messages. This makes the information
visible even when pytest captures stdout. Compiler selection, compile flags,
reference arithmetic, hashes, OS coverage, mutation checks and acceptance
conditions are unchanged. No numerical tolerance was introduced.

## Validation and limits

Local Clang 22.1.8 post-change validation passed all three existing tests with
zero skips, including the seven exact position/table cases, pinned-source
integrity and the bit-mutation/permutation rejection checks. Python AST parsing
also passed. Evidence is retained in private `rope-windows-fix-v1` artifacts and
the owned-Job `rope-windows-fix-clang-v1` receipt. Hosted GCC validation is pending;
it must pass all seven exact cases before this Windows failure is called
resolved. No GPU or real-model test is claimed by this correction.

An attempted local acquisition of the matching compiler archive was incomplete.
The primary release advertises 91,518,830 bytes and provides no API asset digest.
The initial guarded transfer timed out after 540 seconds and drained its Job.
Four subsequent 1 MiB HTTP ranges were validated, but larger ranges stalled.
Only the identified private download worker was stopped; the guard confirmed
termination and zero remaining Job processes. Partial data and failure receipts
were retained. No acquired compiler was extracted or executed, and no local GCC
result is claimed. A hash computed during acquisition would establish local
identity, not independent verification against a publisher checksum.

The preceding full native build completed before this source correction. Its
receipt does not cover the corrected C++ file; source binding and the relevant
incremental rebuild must be refreshed separately. Existing release qualification
limits and resource floors remain unchanged.

## Standards and sources

The compatible canonical pre-implementation prompt, PC/project instructions,
MILSPEC engineering/start-gate/application/C++/Python procedures, systematic
debugging and primary-source research guidance were applied. This record makes
no formal MILSPEC certification claim.

- [Pinned Windows runner inventory](https://github.com/actions/runner-images/blob/win22/20261112.366/images/windows/Windows2022-Readme.md).
- [Pinned MinGW installation script](https://github.com/actions/runner-images/blob/win22/20261112.366/images/windows/scripts/build/Install-Mingw64.ps1).
- [MinGW runtime v12 combined trig source](https://github.com/mingw-w64/mingw-w64/blob/v12.0.0/mingw-w64-crt/math/x86/cossin.c).
- [Maintainer explanation of GCC combination and UCRT implementations](https://sourceforge.net/p/mingw-w64/mailman/message/59124411/).
- [GCC volatile access semantics](https://gcc.gnu.org/onlinedocs/gcc/Volatiles.html).
