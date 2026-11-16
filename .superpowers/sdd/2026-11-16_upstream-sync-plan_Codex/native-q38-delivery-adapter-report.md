# Q38 delivery adapter implementation

Status: implemented with bounded CPU verification; actual formal build, signing,
installation and publication remain controller operations. Base was clean primary
main `963b6a19b09b0d48e323e7a7312e7be87ddd3350`. No subagents were used. No H native
snapshot, installed profile, model weight or original evidence file was edited.

The existing release controller, owned Job, resource admission, production engine
packager, signer, feed, NSIS and anonymous readback contracts remain in use. This
change supplies their new Q38 input. It does not replace the delivery machinery.

## Source and artifact binding

The compiled Q38 source remains `6ee425434c98277555332c7934e1ac51436eb198`.
Its executable is the retained
`H:/Strata/evidence/native-dense-own-cuda13-20261117/build-language/Release/strata-q38.exe`,
SHA-256 `706867da2b9ad1a1debf09b0ac2f02ce3921412382e167ff4d1211efb7012af7`.
Later app/Control source is a distinct consumer; it is not asserted as the native
compile commit. Root owns the final reviewed delivery commit and fresh receipts.

`q38-input.py` checks the immutable language manifest SHA-256
`7204b8c66d84ebd7fff932fd4b71fdd1a1d6ef4cf2acddf9f4f3f6433324a0b1` and full compiler
manifest SHA-256 `40851d465307d73a8c3844222112be5bfab87ca23c070923539c3bab53be4fcb`.
The former binds 385 native entries and 14 mixed successful/failed historical
receipts. The latter binds 1,237 files, 624 compiler-observed dependencies, no
missing paths and the actual toolchain. Failed historical UI attempts remain
failed. Only the explicitly successful native final build, CTest and CLI receipts
are required for this executable reuse. Their hashes and drained Jobs are checked.
The mutable historical report pointer is not relabelled or rewritten.

Every one of the 385 current source-relative paths, size and exact byte hashes is
compared with the compiled manifest before staging. The CMake cache and generated
embedded profile are checked against their original hashes. The portable
`q38-build.json` records the original compile commit, executable hash, full native
source map, CUDA13.3/MSVC versions, five jobs, actual SM120, static CUDA runtime,
compiler options, profile hashes and attributed format pins. CMakeCache's initial
architecture cache value is 75; the project's explicit `set(... 120)` and generated
CUDA projects actually select compute_120/sm_120. No SM75 qualification is implied.
Full SDK/header inventories, absolute personal paths, private receipts and model
weights stay out of the distributed portable record and payload.

PQ2_0 mathematics retain PrismML 0.4.1 pin
`1965755ceee6738b9a04ceaf5d30e6305d6ccd00`. Turbo tables retain
`643956891dfdf51aecdbd05ca508dd21e21e8195`; linear/ternary references retain
Prism `73bfe6cc485554822f80c473a3c0c4f25e80f41b` and Strata ggml reference
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. Both format licences and Spin8's SHA256 /
rotate-bits notices are staged through the existing notice machinery. Q38 has no
llama/ggml runtime dependency. Legacy native and standalone runtime dependencies
and their independent attribution remain intact.

The actual Q38 PE imports require MSVCP140.dll, VCRUNTIME140.dll and
VCRUNTIME140_1.dll in addition to OS libraries. They are absent from CUDA bin/x64.
The adapter takes only those actual required files from the exact installed
VS2022 Community `VC/Redist/MSVC/14.44.35112/x64/Microsoft.VC143.CRT` directory,
records each hash, checks each copied byte and preserves existing Microsoft terms.
It never falls back to System32 or an unbounded DLL search. The production
packager then closes all PE dependencies of strata.exe, strata-q36.exe and
strata-q38.exe. A CPU test with the real Q38 and CRT files has zero uncovered
imports. This is dependency inspection, not an installed inference claim.

## Changes and direct verification

`native-input.py` still builds the legacy configured C cache with five jobs. It
uses a fresh attempt-local `package-input` directory, stages the legacy selection,
adds the frozen Q38 executable/record/format notices/CRT files and probes all three
actual versions before packaging. The external native binding now covers all
three executable hashes, both build records and the original Q38 compiled commit.
`verify.mjs` requires those fields; `verifyNativeBuild` validates Q38 sources,
embedded profile, notice hashes and covered runtime bytes whenever the helper or
record is present. Missing/changed payload, provenance, profile, commit, version,
notice or required DLL is rejected. The existing parent signed inventory covers
the helper and all siblings. Delivery quiescence now recognises strata-q38.exe.

`release.py` dependencies now reports `dependencySourceRoot`, preserving the
phase's true source-checkout identity when results are merged. The CMake parser
reads individual cache records, so blank lines/comments cannot swallow PORTABLE,
ENABLE_CUDA or ARCH keys. The original failed dependencies-r2 and native-input-r1
attempts were not changed or passed retroactively; execution requires fresh attempts.

All following tests used the existing custom guard/interpreter, the documented
Python SOP custom-wrapper exception. No dependency was installed. The wrapper was
`C:/Users/downl/Documents/ChatGPT/Strata/_docs/upstream-sync-20261116/check.py` and
Python was `C:/Users/downl/AppData/Local/Programs/Python/Python312/python.exe`.
Complete logs and receipts are under
`H:/Strata/evidence/upstream-sync-20261116/receipts/`.

```powershell
& $py $guard --seconds 180 q38-delivery-python-final $py -B -m pytest -q _docs/stable-060-20261116/test_q38_delivery.py _docs/stable-060-20261116/test_native_feed.py
& $py $guard --seconds 180 q38-delivery-node-final node --test --test-concurrency=1 apps/host/tests/native-build-identity.test.mjs apps/host/tests/release-integrity.test.mjs
& $py $guard --seconds 120 q38-delivery-notices-final $py -B -m pytest -q _docs/stable-060-20261116/test_release.py -k 'inplace or preserved_input or occupied_target'
```

Final Python passed 22/22, including the real frozen binary/CRT staging, actual JS
consumer, PE closure, source-identity collision, real-layout cache parser and an
actual ephemeral Ed25519 signed parent resolved through production Control.
Changing its covered provenance sibling prevented resolution. No production key
was opened. Receipt SHA-256 is
`5f030deadae07d5c8176a8e65a56fa8238da2ed5d18bad9c025202758c779e46`.
Final Node passed 9/9, including native/source/resource-profile boundaries and
missing/altered Q38 executable, provenance, commit, version, profile, notices and
runtime library checks. Receipt SHA-256 is
`611504f50919f792ce378b8ca50958f3875aa2f5ade369712f9dd8cd608d73d5`.
Both Jobs drained with native exit zero, active-after zero, no nominal floor
breaches and minimum available RAM respectively 13,967,757,312 and 13,970,018,304
bytes. Selected notice and in-place dependency regressions passed4/4 (34 unrelated
cases deselected); receipt SHA-256 is
`afe210de4066567fc232b79d9396dd11653c87edf3fa57a55efe6995395a80dc`.
Their Job drained with zero breaches and minimum available RAM13,904,207,872bytes.

Initial q38-delivery-node-r1 failed because the new licence filename expression
did not admit the digit in q38. The production expression was corrected and r2,
r3 and final passed; the original failed evidence remains. Earlier direct Python
r1/r2 and adapter-r1 also passed. Black at line length 100 and Flake8 with the
existing compact-style line accommodations E501/W503 passed for the two new Python
modules. `git diff --check` passed. No numerical tolerance, RAM/commit/GPU/disk
reserve or spill/resource allowance was raised.

## Actual model boundary and controller work

README, install/release docs and the dense guide distinguish the candidate from
installed/public stable 0.5.9. They describe metadata-based original GGUF trials,
owned execution with attributed format ports, auto and explicit KV, embedded
no-training Japanese profile and finite ja-finite-v1 suppression, literal input
preservation, model-default override and unsupported features. Existing Hermes
memory, workflow learning, tool/key/approval choices and prior engines remain.

Root's separate final Japanese receipt is
`H:/Strata/evidence/native-dense-own-cuda13-20261117/japanese-actual-acceptance-r2.json`,
SHA-256 `1d58994b09236b5b2c7d6fbadab4e086657a769e6bc460925a3fb26cc432847e`.
It retains consumer963 / compiled6ee distinction and unchanged predeclared cases.
Soya passes four of four overall and manual Japanese review. PQ passes the
finite-script/raw/visible/service-wire/reset mechanics but only two of four
overall tasks: bare names instead of a Japanese sentence and an omitted code
fence. PQ overall acceptance did not pass. Those format failures remain open.
Historical hardware-source near-budget IQ4_XS/F16, short auto/Turbo4 at131072 and
soya's 1434-token/364856.9ms prefill remain separate measurements, with no proven
real-model speedup. They do not establish practical full-input performance.

The legacy broad CUDA baseline source digest necessarily includes added Q38
source. Its existing binding differs even if legacy measured libraries are
unchanged. This worker did not edit the baseline, native source, snapshots or
acceptance.py. Root owns a fresh guarded legacy refresh and measured source-only
binding or focused regression before formal native-input. No allowance relaxation
is authorised by a digest-only change. Root owns the actual formal build, final
version/closure probes, signed composition/feed, final-source CI, installed Q38
acceptance, protected standard overwrite and public anonymous readback.

No native compiler, GPU model, installed application, signing or publication was
run by this worker. Practical98304/131072 and <=300s first-response remain active
follow-up goals after packaging. Scalar prefill remains; batched prefill, full
graphs and MTP are deferred. No general quality, universal Japanese output,
full-window, GPU-concurrency or Authenticode claim is made; releaseQualified=false.

## Self-review

The completion and failed-new-test triggers invoked the self-review skill in
review-only mode. The observed filename-validator defect is covered directly by
the production licence fixture; actual PE inspection additionally found the
previously absent CRT closure before a clean-PC launch. No AGENTS, skill, SOP or
memory change is warranted. Existing explicit producer/consumer tests are the
narrow prevention measure. Reverting this scoped commit is the source rollback;
all previous runtime and evidence assets remain untouched.
