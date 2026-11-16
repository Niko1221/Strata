# Hakua Windows release qualification

## 0.6.0 owned Q38 source candidate

The new Windows CUDA 13 / Blackwell dense GGUF runner uses Strata-owned loading, forward execution and CUDA operations. It reuses Strata's GDN kernel and attributed PrismML/ggml format mathematics, Spin8 and Turbo scalar tables, without a llama runtime dependency. PrismML 0.4.1 PQ2_0 weights and F16-K/Turbo2/3/4-V or Spin8 K8V4 caches share this native path. The original standalone PrismML 0.4.1 and TurboQuant 0.4.0 packages retain their separate identities and evidence. Their old signatures do not qualify the combined native executable.

The Q38 binary was compiled from 6ee425434c98277555332c7934e1ac51436eb198 with SHA-256 706867da2b9ad1a1debf09b0ac2f02ce3921412382e167ff4d1211efb7012af7. Later app/Control commits are separate consumers. A portable q38-build.json binds the compiled source closure, embedded profile, format pins, toolchain options, executable and runtime imports. The full private compiler/SDK inventory is retained as evidence and is not shipped. Production packaging retains strata.exe and strata-q36.exe and must close all PE dependencies before signing. The native component covers Q38 and its provenance; the original GGUF weights remain separate.

Earlier hardware-source actual checks passed short PQ and mixed-IQ mechanics and sampled answers, including IQ4_XS/F16 at 1,024 capacity with four finite teacher-forced rows. PQ auto at 131,072 capacity selected F16/Turbo4 and processed short requests; it did not process a full window. Soya at 2,048 capacity needed 364,856.9 ms to prefill 1,434 tokens. No real-model speedup was demonstrated. Later language-source checks are separate: the current app consumer passed four predeclared Soya Japanese cases and manual review, including visible Japanese reasoning and preserved executable literal code. The later PQ run passed finite-script, visible-language, raw/service-wire and reset checks but only two of four overall tasks; it returned bare names instead of a Japanese sentence and omitted a requested code fence. PQ overall format acceptance did not pass. Earlier failed language attempts remain retained. These sampled checks do not establish general answer quality or universal language/format compliance.

Sequential prefill, no native image input, no MTP, no full CUDA graphs, no batched prefill and no native private KV persistence remain explicit limits. Practical 98,304/131,072-token input and a first response within five minutes remain follow-up targets. The RAM, commit, GPU and disk reserves are unchanged. Installed-app acceptance, final-source hosted CI, standard overwrite and public anonymous readback remain separate delivery gates. Current installed/public stable remains 0.5.9; releaseQualified=false. Windows Authenticode signing remains absent. Component and feed signatures are distinct. See [the native trial guide](QWEN38_DENSE_NATIVE.md).

## Stable 0.5.9 workflow safety release

Repairs Windows local-engine physical-memory containment, research capability
restrictions and inventory CSV comparison. App, Terminal, Control and native
source versions are coordinated at 0.5.9. Fresh builds, component signatures,
actual-model workflow checks, exact-source CI, standard overwrite and anonymous
release readback are tracked individually in the
[0.5.9 delivery record](../_docs/2026-11-13_stable-059-workflow-safety_Codex.md).
Formal v0.5.9 is published as GitHub Latest and the latest alias. All seventeen assets matched anonymous size and SHA-256 readback. All eight exact-source CI workflows and thirteen Checks jobs passed. The final standard overwrite matched 29558 managed files and preserved 3789 user files during installation.

Actual-model checks used a fresh isolated installed-app profile and explicit PrismML 0.4.1 with a Qwen3.5 9B GGUF. They passed /btw, skill learning and automatic application, two saved audits with counts 1/0/0/0, four real searches and four successful page reads with no workspace writes. A genuine pressure stop at 8447000576 bytes available was preserved; no RAM threshold was changed before reloading and observing UI answer 42.

The generated research answer incorrectly implied that Copy automatically supplies Clone, and one fetched page was the historical Rust 1.8.0 book. General answer accuracy is not qualified. The Automatic runtime could not load this trunk-only GGUF; explicit PrismML selection was required. These limits and the actual protective stop remain in public QUALITY.json. The normal user profile, all ten existing skill entries, tools/keys/approvals and model settings were restored; the model is unloaded.
The failed initial 0.5.8 real-model attempt is retained in its
[verification report](../_docs/2026-11-13_real-model-workflows_Codex.md).
General model quality, full-window inference and GPU concurrency remain
unqualified; releaseQualified=false. Historical measurements retain their
original versions below.

## Historical stable 0.5.8 — coordinated versions and repaired CI

App, Terminal, Control and native engine are all 0.5.8, with newly built Control
and native signed payloads. The four retained extended context/compaction/search
guard failures are repaired. The actual pinned Harness regression group passed
43 tests with no failures or skips, using private fixture profiles and a
synthetic model API. Existing `/btw` subagents, `/deepresearch`, workflow skill
learning/application and RustXNG 0.31.1 Full remain available.

All eight [exact-source workflows](https://github.com/zapabob/Strata/actions?query=head_sha%3Abb4c57d4a2a2ed0b8d269f120760892ed9f7038a)
passed, including all thirteen [Checks jobs](https://github.com/zapabob/Strata/actions/runs/43069304339)
and all five backend compile jobs. Metal builds genuine arm64 strata/strata_ds4
on an Intel macOS 26 runner and verifies their Mach-O architectures. This is
compile/link evidence; Mac inference was not tested. No required gate was waived.

The standard per-user NSIS overwrite verified 29,555 managed files and preserved
all 3,727 inventoried user files. Models were not copied; selected original model
hashes were checked. Thirteen packaged self-tests, six NSIS cleanup cases and
sixteen actual installed WebView2 cases passed with no console errors. WebView2
workflow checks use synthetic model replies and real RustXNG Full search.

Fresh native verification passed twenty-one numerical cases and sixteen actual
short model requests at 4096 capacity, including bounded/legacy comparison and
restart. The measured machine is RTX 5060 Ti 16 GB / 32 GB RAM, CUDA 13.3.73,
sm_120. Full-window generation, general/low-precision model quality, autonomous
research quality and GPU concurrency remain unqualified. `releaseQualified=false`
is explicit. The installer has no Windows Authenticode signature; authenticated
component/feed signatures are independent.

[Formal v0.5.8](https://github.com/zapabob/Strata/releases/tag/v0.5.8) was GitHub
Latest and the mutable latest alias at publication; 0.5.9 supersedes it. All seventeen public assets were downloaded
anonymously and matched their frozen sizes and SHA-256 values. The tar.7z contains
the same installer, signed components, instructions, source/quality records,
RustXNG corresponding source and checksums. Models and personal data are excluded.
Read the [delivery record](../_docs/2026-11-11_stable-058-context-ci_Codex.md) and its public-safe verification receipts for
exact source commits, byte identities, cleanup and retained failed attempts.
Historical releases, immutable tags and assets remain unchanged. Earlier
measurements and publication statements below retain their original dates.

## Historical stable app 0.5.7 — subagents and workflow skills

The stable zapabob fork app adds real tool-free `/btw` child agents, private
learning from completed workflows and automatic application of matching skills.
It retains `/deepresearch` and RustXNG 0.31.1 Full. The app's stable tag does not
change the independently pinned native 0.5.5 or Control 0.5.6 qualification.

[Stable delivery evidence](../_docs/2026-11-11_stable-057-release_Codex.md) records
the exact build, installed checks, archive, publication and public readback.
Four expanded context/guard tests failed on the retained pre-feature baseline;
they remain documented. New workflow checks use real Harness/tool execution and
a scripted model. General model quality, full-window generation and GPU
concurrency remain unqualified, and `releaseQualified=false` is retained.

The user explicitly requested formal distribution: v0.5.7 is selected by GitHub
Latest and the mutable latest alias. All older releases, assets and immutable tags
remain unchanged. The records below describe their original versions and
publication state, including the former stable 0.5.1 designation.

## Preview 0.5.6-local.1 — side questions and RustXNG Full research

This app-only preview adds `/btw`, `/deepresearch` and an unmodified official
RustXNG 0.31.1 Windows executable using the user's requested Full profile.
The signed native 0.5.5 engine and Control 0.5.6 retain their own earlier source
identities and qualification. No new native performance or numerical result is
claimed. RustXNG returned nonempty results in two documentation searches on
2026-11-11; this does not qualify arbitrary queries or generated answers.

The new [implementation and delivery record](../_docs/2026-11-11_btw-deepresearch-rustxng_Codex.md)
contains current test, build, overwrite, archive and publication evidence.
Previous passing and failing evidence below remains attached to its original
artifacts. `releaseQualified=false` and stable 0.5.1 / GitHub Latest remain.

## Preview 0.5.5-local.1 — native repairs, folder history and browser tools

[Release and assets](https://github.com/zapabob/Strata/releases/tag/v0.5.5-local.1) ·
[Quick install](../INSTALL.md) · [Full install/rollback](HAKUA_INSTALL.md).
Product0.5.5-local.1, Control0.5.6, native0.5.5. App/Control source is
35863deadec1bfa0fc7f250c111d219bb3344b57; independently frozen native source is
b995af139cc823537489b886bc1e8602cf10301c. Official0.2.13 intake staysf33c8c30.
The measured machine is RTX5060Ti16GB/32GB RAM, driver621.14, CUDA13.3.73, sm_120.

The final native resource guard passes with **zero violations**, unchanged
allowances/no passing baseline. All21 final kernel/numerical checks passed,
including actual repaired routes and DFlash byte equality. CPU/GPU/memcheck and
protected real-Q36 bounded7/legacy7/restart2 short requests passed at4096 capacity.
Five-job incremental builds retained the old caches/evidence. W4A8 remains opt-in;
extra weight reads include slower measured cases, so no general speedup is claimed.

The installer passed13 production self-tests and29544 installed payload hashes.
Backup4078 files includes three explicitly reviewed large non-weight files
(1,229,812,132 bytes) in private storage. A first backup size refusal and first
activation-fixture failure are retained. Fresh activation used the exact live
observed Q36 config saved before normal teardown, with required signatures and
full test-start. Old components/settings/models remained. Actual installed
WebView2 checked Q36, Prism plain and Harness-agent replies before/after restart,
normal child exits, workspace CRUD and confirmed26-chat bulk deletion. Bulk
deletion includes archive/beyond20 rows but retains other folders, files and chats
created after confirmation. Normal startup passed open/no-debug/unloaded.

Local agents receive capability-aware proactive search guidance. Browser-enabled
profiles use isolated Chrome/Edge, current accessibility snapshots and coordinate
tools. Actual MCP input/submit/coordinate checks and27 Harness/tool/permission
tests passed. Text-only models omit image responses. Manual/Plan permission,
provider/key and unrelated tool preferences remain. This PC's browser/search
switches were enabled by the user's request. Public search is **not qualified**:
Bing result links were unconfirmed and DuckDuckGo presented human verification.
An initial title-only public assertion was invalidated; no challenge was bypassed.
Broad autonomous-model search-choice quality is not established by scripted tests.

**releaseQualified=false** remains explicit. Full-window generation, general/
low-precision quality, positive Qwen3.8 projection, other GPUs/backends and absent
pointer-table MMQ coverage are not promoted. The historical054 Prism warm-up
timeout and its unconfirmed backend cause remain separate from the reproduced
agent/lifecycle fixes. Older failed full-discovery/localization/memory-UI results
stay retained. This Windows package still has no native image input or separate
native DeepSeek engine.

Installer112445444 bytes, SHA
`c2eac64146485720f2fef61e5fe3a2824e48fa15b1c1887a03761fa0ac90becf`.
Bundle1654538418 bytes, SHA
`de11e7c34f54401395b6a28292c5336f2a94791d6778dd9ea612e4a29919b6b7`.
The real7z/tar bundle uses Copy mode and every inner byte/hash was read back.
Its first readback stopped at the unchanged3GiB RAM floor during a running user
model. That drained failure remains; user-approved normal unload preceded the
complete passing continuation. No installer/native rebuild was substituted.

The EXE has no Authenticode/Tauri updater signature. Fork engine/feed signatures
are separate. Stable0.5.1/Latest and all older release assets are retained.
The preview uses explicit immutable URLs and signed feed sequence20261111000100;
older feeds are refused by anti-replay even across sources. Use retained installed
component rollback or separate stable/preview state roots. See the
[repair record](../_docs/2026-11-10_release-055-repair_OpenCode.md),
[feature/installed record](../_docs/2026-11-10_install-sidebar-web_OpenCode.md) and
[publication record](../_docs/2026-11-11_preview-055-release_OpenCode.md).

## Preview 0.5.4-local.1 — official 0.2.13 integration

[Release and assets](https://github.com/zapabob/Strata/releases/tag/v0.5.4-local.1) ·
[Installation](HAKUA_INSTALL.md). This GitHub prerelease distributes the exact
locally tested installer and component bytes. The Windows app follows upstream's
Tauri NSIS layout with a separate signed engine. A real tar.7z bundle contains
the same EXE, engine/Control packages, signed feed, checksums, SOURCE.json,
QUALITY.json and bilingual INSTALL.md. The EXE has no Authenticode or Tauri app
updater signature. Stable0.5.1/Latest and older published assets remain separate.

Installer112415417 bytes, SHA256
`ae05518656ee5afdf92665f5dd71a06a69425ae1957a9ac31468d8a4a1790750`.
Bundle1654599858 bytes, SHA256
`018dc44f6393f939b14c21c2a863e8e2813c46a8f25dd6929ce75c228d49a340`.
Seven-Zip Copy mode avoids recompressing already compressed inner files; complete
streamed extraction matched every member's bytes/hash. Initial LZMA packaging
stopped at the unchanged3GiB RAM floor; that failed attempt remains recorded.
The low-memory owned continuation passed with zero remaining child processes.

The source worktree contains upstream `f33c8c30` / `v0.2.13` with the fork's
API adaptations. Personal versions: product0.5.4-local.1, Control0.5.5,
native0.5.4. App/Control source85a78234 and native source7bb24974 have separately
bound artifacts. Five-job builds, protected bounded/legacy/restart short native
requests, CPU/GPU/sanitizer checks, signed component test-start, backed-up
overwrite and actual installed WebView2 Q36/Prism/workspace checks passed.
Prism passed on rerun after one retained restart warm-up timeout. The installed
payload matched29544 files and preserved4028 backed-up user files. Normal startup
was checked open without debugging, unloaded, with settings retained.
The fresh native resource guard still measured12 violations; releaseQualified=false.
Full-window generation and general/low-precision quality remain unqualified.
See [the integration record](UPSTREAM_0213.md). Earlier0.5.3 and local.2 results
retain their original source/artifact identities. Public stable remains0.5.1.
Full server discovery timed out; six language catalogs retain pre-existing
missing keys; a supplemental source-memory browser fixture did not launch.
These are not marked passing by publication. Public package preparation and
download readback are tracked in the [publication log](../_docs/2026-11-10_preview-release_OpenCode.md).

## Windows release: 0.5.1

Product, Control and the native CUDA package are `0.5.1`. The immutable release
is `v0.5.1`; the `latest` alias and GitHub Latest currently select it. The release
page records publication and anonymous-download checks. Metal source remains
`0.5.0-metal`; PrismML/TurboQuant and Hakua-memory retain their package versions.
The previous 0.5.0 results are retained below as historical evidence.

The Host now starts without loading the remembered or newly discovered model.
The selected model and library remain saved; **Load** starts the chosen model.
Two real-Host regression scenarios pass. A local hotfix in the installed 0.5.0
app was checked through its actual WebView2 with Playwright: ten seconds of
startup remained unloaded and stopped, with no page errors or registry changes.
The later 0.5.1 NSIS package was extracted, compared with the staged payload,
self-tested and checked through real WebView2. Its in-place installation and
manual startup checks passed while preserving the user profile and models.
The installed payload matched 29,462 archive files and preserved all 665
backed-up user files. Ten one-second startup observations stayed unloaded;
the normal profile selected Host, shell, Control and signed engine 0.5.1.

The native engine reports model and expert-pack verification progress. Its
Windows sequential verification uses bounded aligned I/O where supported. After
a complete SHA-256 and CRC check, unchanged NTFS files can reuse a local receipt
protected with Windows DPAPI. File identity, size, change time and USN must match;
the header and tensor geometry are still checked. Missing, changed or invalid
receipts fall back to full verification. `STRATA_Q36_VERIFY_PACK=1` forces the
full check. Model files and expert packs are retained.

CPU expert-store and transfer-lifetime tests pass, including changed content
with restored modification timestamps, invalid receipts, geometry mismatch and
an active writer. Full verification of the real UD-Q4_K_XL model and its expert
pack passed in 380.320 seconds; reusing the unchanged, DPAPI-protected receipt
took 0.004 seconds in the same CPU probe. These timings measure verification,
not application startup. The earlier 720-second probe timeout remains recorded.
The source-bound Windows CUDA 13.3 build completed. Eight CUDA/kernel/cache
parity tests passed on the RTX 5060 Ti 16 GB, driver 621.01. The real
UD-Q4_K_XL model reached READY in 41.250 seconds with a 4096-cell context,
five expert slots per layer, a 512-row prefill chunk and MTP depth 2. The
unchanged-file verification receipt was reused. Six short requests passed:
first owner, same-owner prefix reuse, a different owner, missing owner,
decode cancellation, and clearing after cancellation. The same-owner request
reused four tokens; the other requests reused none. These used four or five
synthetic input token IDs, not a full 4096-token prompt or a quality benchmark.
The lowest sampled available RAM during native qualification was
4.48 GiB, above the unchanged 3 GiB floor. The runtime hashes and model/pack
sizes and modification timestamps matched before and after the trial.

The NSIS payload, thirteen production self-tests, packaged WebView2 checks,
in-place installation and signed component import passed. These results do not
qualify a 98K or 131K context window, FP16 KV, or CUDA 13.5. The initial installer
build stopped because Windows PowerShell treated an informational stderr line
as an error; the PowerShell 7 retry passed. The digest expectation test was
updated for the measured engine and its eleven cases passed on rerun; the other
thirteen focused host cases passed in the first run.

CUDA 13.5 evaluation is separate from the startup repair. The opt-in
`tools/bench_cuda135.cu` measures the existing transfer modes and FP16/BF16 GEMMs
with bounded allocations and numerical checks. It requires CUDA 13.5 and
`sm_120`. No faster default or CUDA 13.5 product qualification is claimed until
the measurements and native-model checks pass.

The first library-only comparison passed six serial runs (three per cuBLAS
version) on the RTX 5060 Ti, driver 621.01. Both sides used the same CUDA 13.5.37
benchmark executable; only the cuBLAS DLLs changed from toolkit 13.3 to 13.5.
For N=4096, K=2048, median FP16 T=32 time changed from 0.027806 to 0.018528 ms;
BF16 T=1 changed from 0.043107 to 0.026534 ms. FP16 T=1 regressed from 0.027714
to 0.031571 ms. Eight 1 MiB host-to-device copies took about 1.26--1.27 ms
across the three transfer modes. Numerical fixtures and byte comparisons passed.
These are small synthetic measurements, not model throughput or a compiler A/B
comparison. They do not justify enabling every new path by default.

### 0.5.1 build identities

Native source is fixed at `00ce436f69a984304cbbd8da276da0ec1de9b3a3`.
`strata.exe` SHA-256 is
`e75716a9d5deb6289e87741d42e7003ba82b7696f6121d5203e615d9fe596a96`.
Host and shell source is fixed at `15fa71ef5e0bab3b4669c711d55eb6ba43e1cec9`.
The installer is 112,006,044 bytes, SHA-256
`8af65aec986a22999a85ae9d84ef6c85746848c173b9df2e08fbd98cc89fdf33`.
Subsequent documentation-only commits do not change these payloads.

## Windows release: 0.5.0

Publication policy: release 0.5.0 with the known limits recorded below. Native
Qwen3.6 startup is not qualified: the first 4K trial stopped for RAM pressure;
the second did not reach READY within six minutes. No private-cache requests ran.
Publishing this version does not change those failed or NOT_RUN results. The
prepared 0.4.1 packaging-only candidate was superseded by this integration.

Product, native CUDA engine and Control are independently versioned as `0.5.0`.
The immutable release tag is `v0.5.0`; `latest` selected that commit when 0.5.0 was published.
The [release page](https://github.com/zapabob/Strata/releases/tag/v0.5.0) records
publication and public-download verification. Existing historical assets remain unchanged. Metal source is `0.5.0-metal`, with no newly qualified
Metal package. PrismML/TurboQuant remain `0.4.0`; Hakua-memory remains `0.3.5`.

Official source is fixed at `9f7d977d220e9fdf2ec53cf2620cdadac1cfb24e`, including
upstream `0.2.11`, the subsequent Docker packaging correction and supporter credits
in all eight READMEs. Executable source is unchanged from the reviewed upstream
`5d37280db1f70ef6b55d69ac1d9ad8630d7a62f0`. The preceding upstream checkpoint was
`ffa80d73133ea5c548aefe57889a16e8085a938f`.

### Changes and preservation

The integration adds per-chat file storage, conflict copies, active-turn
reattachment, quiet-stream heartbeats, IME Enter handling, bounded previews,
renderer recovery, Unsloth v3.1 model sets and updated Control split-engine APIs.
Hakua memory, existing-GGUF discovery, private conversation ownership, mapped
expert packs, bounded Windows I/O, CRC validation and the fork's signed feed
are retained. Product, Control and engine version sources remain independent.

Security-related updates include Pillow 13 with a restricted image decoder list,
constant-time shared-key comparison, malformed Host-header handling, stream
ownership and cancellation isolation, image prompts passed through private files,
benchmark redaction, pinned-download completion checks and loopback startup probes.
These describe included changes, not a claim that all vulnerabilities are absent.

A reproduced incoming chat-store defect is also repaired: after a chat write
succeeds but its index write fails, restart now reads the authoritative chat
envelopes. A stale index cannot reset a revision or hide a stale-window conflict.
The focused failure-injection suite passes 13 tests. Startup reads all saved
chat files; large-history performance is not qualified by this result.

### Build identities

The native build uses CUDA 13.3, Visual Studio 2022 x64 and `sm_120`, with one
build job. ggml is pinned to `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
The source receipt is from `11ba67bec6f89ff998cf58293364f9a7c45bde6a`.
`strata.exe` SHA-256 is
`28c4dc3991fa05b900fa956c351720121088d9041202d6acefd9b7af2749370e`.
The NSIS build uses Host/shell source `df8762ff97819333f3416c48922b09965863f964`,
Node 24.22.2, CPython 3.12.15 and Pillow 13.0.0. Later README and installation
documentation changes do not alter these packaged executable sources.
The installer is 111,992,152 bytes, SHA-256
`fa1ab27695c13b7f251c9d9362d0f3649507f91d04194692c436810f6824a778`.

Control was built at `d6b6e9b9ddd9d7a35888e4eeaa5c13f7e368c421`. Later fixes
affect native/Host sources, tests and documentation; the packaged Control source
is checked file by file against the final checkout. The signed engine component
retains PrismML/TurboQuant 0.4.0 binary identities and their original provenance.

### Release evidence

| Check | Result and scope |
| --- | --- |
| Native CUDA ALL_BUILD | PASS after correcting the cache-prefill local-variable scope; the failed first build is retained. The Windows W16 parity test environment setter was also corrected and compiled. |
| CPU-only native CTest | 8 passed. This does not execute CUDA kernels. |
| Terminal, download, setup and runtime checks | 266 passed, 2 skipped across the focused batches. |
| Server integration | 558 passed, 5 skipped. Windows/POSIX and symlink privilege skips are retained; one passing closed-console fixture emitted a background-writer warning. |
| Host full suite with the real bundled Harness and scripted model responses | Initial run: 1,741 passed, 1 failed, 34 skipped out of 1,776 tests. The failure was Windows command-line length in a Markdown test; its input now uses stdin. The 28-test Markdown/preview rerun passed. The full suite was not repeated. |
| Qwen3.6 launch contract | 11 passed with the new source-bound executable identity. This registers mapped-pack capability; it is not an inference measurement. |
| Actual packaged Python/Pillow runtime | 80 passed, 2 skipped, plus 39 passing subtests using CPython 3.12.15 and Pillow 13.0.0. |
| TypeScript and production Host build | PASS. An initial build with a deliberately small 512 MiB JavaScript heap hit that limit; the qualified build used 1,024 MiB and passed. |
| Source Playwright UI | 10 passed, including 96K/128K configuration, saved profiles, responsive layouts and navigation; no model inference. |
| Production npm audit | No known advisories in the 8 production packages at the recorded check. This does not cover every bundled component. |
| GitHub hosted CI | BLOCKED: workflow dispatch returned HTTP 422 because Actions was disabled for this repository. Local checks are reported separately. |
| NSIS extraction and production self-tests | PASS: the payload matches the build sources and staged private runtimes; 13/13 production self-tests passed. The shell differs from the unbundled build only at Tauri's documented UNK-to-NSS bundle marker. |
| Packaged WebView2 with Playwright | 15/15 passed on the exact extracted NSIS shell, using an isolated profile. Existing Qwen3.6, Hikari and both Swift shards kept their original paths and reported zero model-download bytes; repeated import retained the same identity. Hikari displayed its name/PQ2_0, 96K/128K and CPU MoE settings survived save/reload, three viewport widths had no horizontal overflow, and four navigation routes worked. No page or console errors were recorded. No model inference was started. |
| Fresh CUDA parity | 5/5 passed on the RTX 5060 Ti in 100.22 s: native dense/expert parity, K-quant and v3.1 MMQ fixtures, and the native cache bridge. The conditional Q5_0 MMQ omission below still applies. |
| Bounded native Qwen3.6 startup and private-cache requests | ABORTED before READY. With context 4096, mapped-pack, 5 expert slots and a 512-token prefill chunk, available system RAM fell from 9.07 GiB at the start of the guarded batch to 2.42 GiB. The unchanged 3 GiB floor stopped the owned process tree after 268.99 s, including the preceding parity tests. None of the six planned private-cache requests ran. This is a resource-pressure stop, not a successful model-start or cache-isolation result. |
| Cleanup and runtime restoration | PASS: the guarded Job had zero active processes after termination. The interrupted runtime was restored with its exact original command line, model, 98,304 context and one slot, and its owned listener returned healthy. The stop-to-restoration interval was about 6 min 35 s. Model and expert-pack size/modification-time checks were unchanged; this is not a full model hash recheck. |
| Explicitly authorised second trial | CUDA parity again passed 5/5 in 96.06 s. The same native Qwen3.6 configuration did not return READY within the 360-second startup deadline, so it was stopped. RAM did not trigger the stop: available RAM began at 10.10 GiB and stayed at or above 5.96 GiB in the recorded samples. None of the six cache requests ran. This separates a startup timeout from the first trial's RAM-pressure abort; it does not identify the exact slow operation. |
| Second-trial telemetry and restoration | The native process reached 4.61 GiB working set and 9.06 GiB private bytes; these process-level figures do not identify individual allocations. The Job ended with zero active processes, runtime file hashes were unchanged, and model/pack size and modification time were unchanged. Exact original runtime settings and healthy listener were restored in about 9 min 23 s after the stop. Resource floors were unchanged in both trials. |
| Local component/feed signatures and tar.7z readback | PASS: the component and feed verify against the pinned signing key. All 10 tar members were streamed back from the 7z and matched their input size and SHA-256. Twelve separate public assets are prepared locally. This does not activate or publish the feed. |
| Anonymous public downloads | Performed after publication; consult the release page for the final download/signature result. The tagged source records pre-publication checks and does not claim that a future download has passed. Native startup and cache qualification remain unsuccessful regardless of publication. |

Read-only comparison found `src/qwen35/expert_store.cpp` and
`src/qwen35/q36_engine.cu` byte-identical to the preceding fork checkpoint
`91f5057b`. Model hashing and pack verification already use bounded 1 MiB reads;
pack SHA-256 and CRC share one payload pass. This rules out a direct edit regression
in those two files, not differences in linked code, build settings, driver behaviour
or other workloads. The last startup marker was `buffers:end`; the logs do not
identify the exact unfinished expert-placement substage. No speculative native
change was made from these observations.

The packaged local-model settings screen opened in 235 ms in this single check,
while unrelated native discovery was still pending. This measures screen access,
not model loading, prefill or token generation. The extracted-package test does
not replace the user's normal installation or establish an NSIS upgrade result.

The setup uses `Strata-0.5.0-windows-x64-setup.exe` (NSIS). The real tar then 7z
archive `Strata-0.5.0-windows-x64.tar.7z` contains the identical EXE, signed
CUDA component, Control ZIP, signed feed, install scripts, checksums and INSTALL.md.
Models, personal data and signing keys are excluded. Windows app updates remain
manual: neither Authenticode nor a Tauri app updater signature is claimed.

The locally verified tar.7z is 1,645,068,517 bytes, SHA-256
`4f58d543134c95512fc38f8fa5f9f43dd6e40878c571794587924a72535a8fe1`.
The engine tar.gz is 1,466,966,308 bytes, SHA-256
`531b2b311af907c5e32fa276789522f94ac75dcea5490af708d39bebf3004d90`.
The Control ZIP is 73,419,032 bytes, SHA-256
`01ee730bc5e496ccf91e2a147e9cb36dc3123acd283fb9234667649d7fc6f705`.

### Qualification limits

Upstream NVIDIA MMQ and automatic prompt-chunk changes require new local checks.
The build leaves `STRATA_ORCA_Q4KS_MMQ` off; a passing v3.1 parity executable alone
does not qualify its conditionally omitted Q5_0 MMQ path. The W16 parity binary
is built but its large fixture is NOT_RUN on this 16 GB GPU. The native Qwen3.8
cache-prefill integration and 0.5.0 throughput are not qualified by a small cache
bridge or Qwen3.6 request. Prior 0.4.1 measurements below do not qualify 0.5.0
bytes or default settings.
Native 98K/131K full-window generation remains unqualified. HIP, SYCL and Metal
builds and hardware runs are NOT_RUN on this Windows CUDA release machine.
The previous PrismML full-window observations retain their original scope.

## Previous beta qualification

## Windows beta: 0.4.1-beta.1

Product/tag: `0.4.1-beta.1` / `v0.4.1-beta.1`. Native engine and Control are
`0.4.1`; unchanged PrismML/TurboQuant packages remain `0.4.0`, Hakua-memory
`0.3.5`, and Metal source `0.4.0-metal`. Versions follow SemVer 2.0.0; dates stay
in the release notes. Published historical assets are not replaced.

The Windows app and signed engine are built and installed in the normal C-drive
locations. The local checks below pass within their stated scope. Native full-window
generation remains unqualified. Use the explicit
[beta release page](https://github.com/zapabob/Strata/releases/tag/v0.4.1-beta.1)
for asset availability; the normal latest feed retains its earlier signed release.
See [installation](HAKUA_INSTALL.md) and
[version policy](../packaging/UPDATES.md#version-and-asset-names).

Published on 2026-11-05 at 12:40:24 UTC as a prerelease with `latest=false`.
The release tag points to `9e6f515af3c707562e99b3f904c64da9e9c69e8d`.
Its seven assets are the setup EXE, CUDA component archive, Control ZIP, signed
feed, feed signature, Terminal install manifest and SHA256SUMS.txt. All seven
anonymous downloads returned HTTP 200 and matched the local sizes and SHA256
digests. The downloaded feed signature also verified with the pinned key.
The normal latest release remains `v0.4.0-preview.20261101.1`; no historical
asset or stable-feed pointer was replaced. These publication checks do not
extend the hardware or full-context qualification described below.

The repair includes meaningful names for generic GGUF metadata, existing local
GGUF discovery, resumable-pack disk accounting, layer progress, cancellation of
image downloads, consistent engine/monitor state after failure, and one runtime
preflight per load. All eight README languages and nine GUI catalogs describe the
affected behavior. Model weights and expert packs are retained.

Qwen3.6's new private cache is bound to an authenticated conversation. It is
cleared for a changed or absent owner, cancellation, errors and five idle minutes.
The Host propagates chat ownership through the actual Harness adapter without
mutating shared profiles. The two Host branches that rejected Qwen3.6 private-cache
requests before startup have been removed. The server now checks the running
engine's INFO capability before accepting requests; private-cache support is not
granted by a binary digest list. Saved refusal messages use the current translated
engine-update instructions.
Initial support is one request at a time. It does not enable shared prompt caches.

### App and Control checks on 2026-11-05

The CRC follow-up app built from `6fdb857bec44a83a4bba82ad752062e788625966` passed comparison
of 29,329 packaged files and all 13 isolated shell self-tests. These source
inventory comparisons exclude generated Python bytecode caches. The installer
is 113,080,604 bytes with SHA256
`4d78da03b848ff985c5887b1080b6d0227edcda902a278961e99dc9fbf06864d`.
Control remains at source `7092b802c3f6da72c0244cd84e45e1e738220eb6`.
It then completed an in-place update at `%LOCALAPPDATA%\Programs\Strata`.
The installed Host, Control, Node, Harness and notices match the qualified
payload. All four user-data inventories match their pre-installation SHA256
inventories (3,979 files). Both existing shortcuts point to the C-drive app.

Playwright drove that installed app's actual WebView2 window with a separate
test profile. All 15 checks passed: existing Qwen, Hikari and split Swift file
references, repeated-import identity, meaningful model labels, 96K/128K
selection, CPU MoE settings, saved profiles after reload, navigation and widths
1440, 768 and 375. No model was loaded during this UI check. Registration took
285 / 148 / 516 ms respectively; these are metadata-registration observations,
not weight-loading or generation times. The registered local-model screen
opened in 262 ms while native catalog discovery was still pending.

After native activation, a separate Playwright observation of the installed
WebView2 Settings > Engine page showed 0.4.1 as In use. The installed Control's
engine list independently returned that version as active and compatible.
The observation used an isolated profile and a minimised window; it sent no
model-load request and had no page or console errors. Two earlier observers
left the Settings route before the assertion and did not establish a UI result.

The same repair avoids a CUDA card-ordering probe on a single-GPU computer.
Multi-GPU and mixed-vendor detection retain their prior path. Two observations
of the old installed Control's native discovery took 24,807 and 24,984 ms;
the corresponding new staged-Control observation took 7,040 ms while Hermes
was running. These are observations on this PC, not a general speed guarantee.
The GPU-role regression suite passed 18 tests, the existing ModelRun suite
passed six, and the source UI passed 11 checks with native discovery deliberately
held pending. These UI and discovery checks do not qualify inference.

Before the CRC follow-up rebuild, the installed app completed a real GUI/Harness/search round trip through
an API connection to the existing local Hikari PQ2_0 server. Playwright selected
the model in the app and sent the prompt from its message box. The bundled
Harness 0.2.1-alpha.3 received a model-generated `web_search` call, returned live
DuckDuckGo results, and displayed the model's final answer with an official
source link. Search took 1,066 ms; the Host reported 107,587 ms for the turn,
including queue/startup and generation. Context capacity was 98,304; this was
a short request. Page and console errors were zero. No engine was restarted.
This qualifies the installed GUI/Harness loop through a local API connection;
it does not independently qualify the Strata-owned supervisor route. An earlier
attempt timed out while waiting for server status before sending a prompt; a
second attempt watched the wrong HTTP path and supplied no stream evidence.

### Measurements before the final version rebuild

Windows, RTX 5060 Ti 16 GB, 32 GB RAM, CUDA 13.3, sm_120;
Qwen3.6-35B-A3B UD-Q4_K_XL, mapped expert pack, FP32 KV, MTP 2,
4096 context capacity, 256-token prefill chunks, no startup warmup.
These measurements are tied to the listed development binaries, not a future
installer or an unmeasured full-length context.

The first cache-isolation probe passed 15 cases on executable
`7a0801aaab8752d3ef6a6370a42b05294552df6f0147b50c3983d028ea801073`.
READY was 318838 ms versus an earlier 799047 ms observation. Disk/page-cache state
was not controlled across those runs; this is not a repeated speed qualification.

The next executable, `335600683f10801b694fee0f10e45a7600da418289365e7501b52abc61db9341`,
compared the default transfer path with both `STRATA_Q36_PIN_PIPELINE=1` and
`STRATA_Q36_STREAM_RESIDENT=1`. Generated tokens matched exactly in all four cases.

| Synthetic case | Prompt tokens | Default total ms | Opt-in total ms |
| --- | ---: | ---: | ---: |
| Arithmetic | 31 | 72059 | 25323 |
| Python function | 35 | 75942 | 29791 |
| Web-search tool call | 291 | 100615 | 52429 |
| Longer prefill | 1596 | 42166 | 19830 |

READY was 337231 / 243690 ms. Whole-GPU peak used memory was 15796 / 15842 MiB,
including other applications. This single ordered pair has uncontrolled cache
warmth and desktop load. Some short decode timings regressed. Both switches stay
off by default; no general speed or quality claim follows from four prompts.
The longer fixture was originally named `prefill-1024`; its actual prompt is 1596
tokens, as recorded here.

The bounded verifier retains full model/payload SHA256 and per-layer CRC checks,
uses one payload pass, and releases the original GGUF mapping after expert
placement. CUDA transfers retain two bounded pinned slots with per-slot completion
events. No weight format, KV precision or context default was reduced.

The installed signed PrismML binary completed a real Hikari/PQ2_0 tool round trip
with context capacity 131072 and the production privacy flags: startup 15165 ms,
model-generated `web_search`, live DuckDuckGo sources, final official-source URL,
and engine still alive. This proves that bounded model/API/search path, not the
final installed GUI/Harness path or full-window quality.

### Full-window and final-candidate observations on 2026-11-05

On the same RTX 5060 Ti 16 GB / 32 GB RAM PC, the unchanged signed PrismML
runtime completed both full-window cases below. Each evaluated all 130,792
input tokens at capacity 131,072, generated 47 tokens, and recovered all three
synthetic lookup codes placed far apart in the input. These are one-run retrieval
checks, not broad quality tests or repeated speed comparisons. Compilation ran
in the background and disk/page-cache warmth was not controlled.

| Model and placement | Prompt time | Reported prompt rate | Generation time | Reported generation rate | Sampled whole-GPU peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B UD-Q4_K_XL, CPU MoE 28 layers | 1,724,306.98 ms | 75.8519 tokens/s | 26,094.379 ms | 1.76283 tokens/s | 11,586 MiB |
| Hikari 27B PQ2_0, all GPU | 257,130.314 ms | 508.66037 tokens/s | 1,803.198 ms | 25.51023 tokens/s | 12,740 MiB |

Both use Q8 K/V, Flash Attention, batch 1024, microbatch 256, six CPU threads,
one request slot, no automatic fit or context shift, and no shared prompt cache.
The generation rates above are the runtime's reported rates. See
[the 16 GB context guide](WINDOWS_16GB_CONTEXT.md) for the settings and limits.

Native Qwen3.6 executable
`2dd5dbcd2874767096b9c1eb4ac4132a96f7979bcd019ec6b779f23c9dbacc0f`
passed 18 actual private-cache checks at capacity 131,072, prefill 512, MTP 2
and FP32 KV, with the mapped expert pack and both opt-in transfer switches.
The cases covered same-owner reuse, owner changes and absence, invalid requests,
decode/prefill cancellation, and expiration after 305 idle seconds. READY took
243,092 ms; sampled whole-GPU peak was 15,702 MiB. These short requests do not
establish native full-window quality or qualify a later executable automatically.

Swift IQ2_XS native input of 98,016 tokens at capacity 98,304 failed with Windows
access violation `0xc0000005`. A null vector access in the KVA early-out path was
identified: its stream plan is absent below the 1024-token streaming threshold.
Three stream-plan release sites are now conditional on that plan being active.
The rebuilt executable
`83959903089c536a16a85cb14261070b05d1cb892f42d6b059bab61b238b21e7`
passed separate state-only and MoE-skip regressions with 912 input tokens,
512-token chunks, a partial final chunk, and two generated tokens. The diagnostic
KVA settings deliberately exercise those branches and do not measure retrieval
quality. The full repair run reached 96,256 processed tokens, but the PC became
unresponsive and the user manually restarted it. No completion or answer was
recorded. Earlier samples showed about 350 MiB of available RAM; the cause of
the host hang is not established. The same configuration is not qualified for
98K use and is not being retried unchanged.

A subsequent source-bound CUDA build completed with SHA256
`c86b5d36e5371d8276bc73d553d2be0149a674ccc74541fa51d7eb88a2ab55a0`.
Its `kva_test` passes; runtime qualification of these exact bytes remains
separate. The focused KVA fix is submitted as
[upstream draft PR #5273](https://github.com/Niko1221/Strata/pull/5273).
The PR contains only the three guards and records the missing exact-upstream
CUDA, HIP and SYCL validation.

The exact c86b5d36 native executable reached READY at capacity 131,072 in
692,031 ms and passed 13 short private-cache checks. During the 256-token
prefill-cancellation case, available physical memory fell below the diagnostic
3 GiB reserve. The retained process handle stopped only that engine after the
two-second grace period; its final sample had 2,758,021,120 available bytes.
This run did not finish the cancellation or idle-expiry checks and is not a
passing qualification. Hermes was restored with its original executable,
command line, model, context and slot count, and its health endpoint returned OK.

Inspection found that whole-layer uploads copy from the full mapped pack into
the bounded pinned slots. Reading those mapped source pages can still grow the
process working set. The opt-in `STRATA_Q36_FILE_STAGING=1` path
reads the same open file directly into those slots and retains the existing
CUDA completion events. CPU byte, range, concurrent-offset and file-replacement
checks pass. The bounded real-GPU run below also passes; full-window and repeated
speed qualification are still pending.
Model files, weight precision, KV precision and context capacity are unchanged.

An earlier native executable
`5e37033b11e6478920450f39a3b0057c2ca6d4116b1e196a1b507bf344d4fbd0`
completed 18 short private-cache checks at capacity 131,072. All ten comparable
deterministic cases matched the earlier c86b5d36 token results. The run covered
owner changes, missing and invalid owners, error recovery, decode cancellation,
final-prefill cancellation and expiry after five idle minutes. It used the
existing Qwen3.6 UD-Q4_K_XL GGUF and expert pack, five expert slots, 512-token
prefill chunks, MTP 2, no startup warmup, and all three opt-in transfer switches
(`PIN_PIPELINE`, `STREAM_RESIDENT`, `FILE_STAGING`, each with the `STRATA_Q36_`
prefix). Model and pack sizes and modification times were unchanged.

READY took 286,968 ms and the complete bounded run took 724,672 ms. Peak process
RSS was 5,806,473,216 bytes; minimum available physical RAM was 11,843,059,712 bytes.
There were no resource aborts and the owned engine exited and was reaped. These
are short-input observations, not a full-window result. Initial available RAM
and filesystem-cache warmth differed from the earlier run, so they do not prove
a controlled speedup. Final-prefill cancellation took 88,156 ms: state was cleared
before the acknowledgement, but prompt cancellation is not yet responsive.

The 1,466,619,912-byte CUDA archive has SHA256
`538578c8dd275408d6966e1404b1ae3339ed0574a9a0b3fc3eb27a99c5ceaed6`.
Its manifest and beta feed were signed and verified with the fork's pinned
component key. The installed Control imported the archive into a separate empty
profile, verified every file and selected engine 0.4.1. With no model registered
in that profile, the import test ran `--version` only; it is distinct from the
18-request native test. It did not change the production component selection.

A later attempt used the normal C-drive component store and the installed
Control's real one-token startup check. Its diagnostic context was 4,096 and its
startup limit was eight minutes. The Qwen3.6 model was still loading when that
limit expired. That attempt did not switch engines: the active component stayed
0.4.0, the saved 131,072 configuration was unchanged, and the existing GGUF and
expert pack retained their sizes and modification times. No resource-pressure
abort occurred; minimum available RAM was 7,929,376,768 bytes. The complete owned
test tree ended, and Hermes returned to the identical Hikari command line and
98,304 context. The interruption-to-restoration interval was about 11 minutes
25 seconds. This failed normal startup is separate from the earlier short
native protocol tests and binary-only import.

The startup verification reads the full 22,853,663,008-byte GGUF and validates
the full 20,127,612,928-byte expert pack. The revised CRC implementation keeps that
coverage and the existing IEEE CRC32 format, but processes eight bytes per
iteration. On the Ryzen 5 4500, six alternating 64 MiB samples had median CRC
times of 131.17 ms before and 31.74 ms after, about 4.1 times faster
for this checksum calculation. 16,390 comparison cases and a known CRC vector
passed. The implementation's pack-format, corruption, reuse, bounded-read and
transfer-lifetime CPU tests pass, including independent bitwise CRC checks.
Core/C++ static analysis passes. The later installed startup observation is
reported below; it is not a controlled before/after speed comparison. Model and
KV precision, context capacity and all hash checks remain unchanged.

The CRC follow-up native executable has SHA256
`e79b9791bdfdacd1e5f20324b8e80978af287cc1dc464d22231da8819de0f7fc`.
Its CUDA build, 138-file component integrity check and pinned-key signature
check pass. The signed archive is 1,466,638,193 bytes, SHA256
`d305c6e2f9f367a8ca1d09052daa1e09134d5802a4fead261d1923c4d86dbfa0`;
the signed manifest is
`de9ff76a8c69e97d4de19a031ccb390eb775ec8059cae7a28b118d9286f7018b`.
The matching beta feed, sequence 20261105121136, is signed and verified with the
fork's pinned key. It names this exact archive and Control ZIP SHA256
`23a94b37df6faadf7c56f20786c071541882587b5b63153c9dac619fc6ab6244`.
It has no app updater entry. The earlier 18-request isolation result belongs to
the previous executable; those 18 checks were not repeated on this CRC rebuild.

The installed Control then imported this exact archive into the normal C-drive
component store, verified its signature and every file, loaded the existing
Qwen3.6 GGUF/pack, and completed its real one-token startup check. Control reported
the model loaded at 289 seconds. The 11-token prompt took 28,672 ms; one generated
token is insufficient for a sustained decode-rate measurement. This test used
4,096 context, 512-token chunks, MTP 2, FP32 KV, five expert slots, no warmup,
private conversation caching and the three opt-in transfer switches above.
Legacy K/V quantisation and spin8 settings in the saved config are ignored by
this native engine; they did not change its FP32 KV precision.

The active component is now 0.4.1. The normal saved context remains 131,072 and
the existing GGUF and expert-pack sizes and modification times are unchanged.
The owned test tree exited completely. Minimum available physical RAM was
6,467,551,232 bytes, with no resource abort. Hermes was restored through its
existing scheduled launcher with the exact original command line, model,
98,304 context and one slot, and its health endpoint returned OK. Total
interruption was 635.78 seconds. This passes bounded installed startup, not
native full-window generation, repeated speed qualification or all-model support.

GitHub Actions returned `Actions has been disabled for this repository` when
the validation branch was dispatched. The exact source head had no workflow
runs on the later check, despite the permissions API reporting enabled. This
remains unresolved. No exact-head cloud-test pass is claimed; this beta's
validation is limited to the recorded local checks.

Three Qwen3.6 GDN kernel candidates passed their bitwise parity checks but did
not show a consistent timing improvement. None is enabled in the engine. No
weight precision, KV precision or context capacity was reduced to obtain these
results. HIP/SYCL/Metal builds, native full-window generation/quality, repeated
statistical speed qualification and the existing CUDA spill baseline remain
unqualified. The final build's source/hash receipt, installer checks and installed
GUI/Harness tool round trip are separate requirements.

## Historical record: 0.4.0-preview.20261103.7

Product tag: **0.4.0-preview.20261103.7**. Use the explicit release page, not GitHub `/releases/latest`.

- Release URL: [v0.4.0-preview.20261103.7](https://github.com/zapabob/Strata/releases/tag/v0.4.0-preview.20261103.7). Published with `gh release create --latest=false`. GitHub’s latest pointer stays [v0.4.0-preview.20261101.1](https://github.com/zapabob/Strata/releases/tag/v0.4.0-preview.20261101.1).
- App assets on this tag: setup EXE and `SHA256SUMS.txt` only (no matching 7z). Setup EXE SHA-256 `5e8b8fff82b6e63dea48bf5c1907950911700c9bf30673444e5d2489a87c0336`, 113064953 bytes. The installer has no Windows code signature.
- Local silent install: a bare empty tree can return NSIS exit 2. Known fix: create the install folder and empty managed dirs (`control`, `host`, `node`, `harness`, `engine-bundle`, `local-backends`, `notices`), then run **one** `/S` (do not wipe/`rmdir` a live tree while setup runs). After repair, DisplayVersion `0.4.0-preview.20261103.7`; ProductVersion on `strata-desktop.exe` matches `.7` (SHA-256 `58426441dcd64fddcbca565d64f59841cbedcb8c972c8dd7b44d294071010b4a`). Recorded InstallLocation on this machine: `H:\Programs\Strata-0.4.0-preview.20261103.7` (restored from the manufacturer install key; do not delete that folder during ship). Concurrent setup/`rmdir` races left a leftover empty `%LOCALAPPDATA%\Programs\Strata` tree; leave it alone rather than wiping.
- Private Node may need a one-shot copy from the NSIS cargo `release\node` staging if a raced `/S` left `node\` empty; that is a copy into the install, not a folder wipe.
- Unsigned Qwen3.6 side-by-side component: `0.4.0-q36-windows-x86_64-cuda-cuda13` (no picture reader; `signature` is null). Hotspot-profile `strata.exe` SHA-256 `907341c74d5a9f26c784e4aaad3460e376d4cabd84635c5d032879c59b08d150`. Bundle `native-build.json` SHA-256 `cd235fdf812d43c45e0a0881073bee97483c5aa50c65bbefb877bd1e8e8dbd8b` (metadata entry, not the engine entrypoint). This is not a signed package.
- Mapped-pack stays off unless the launch asks for it. Allowed resolved `strata.exe` digests: `907341c74d5a9f26c784e4aaad3460e376d4cabd84635c5d032879c59b08d150` (hotspot-profile opt-in rebuild; T07 capture PASS), `64226fde7905821148c78b7de85ede4d291c4df8184d80866dccba014d7c02b7` (help-honesty rebuild; mapped-live READY + T03 PASS + T04 observational pair recorded), `6fe110650940d0c19f09909d769393769650045e6bc6eab29ccc49734dbcab29` (pin-pool wire), `dd53df07789e0fefcb87e33acab54f5bfe1faf312ef6bad515924736b715c5b5`, and the older `512d7f8bba66b57fbe0f8b90807825d3b3172a86dd039452bc29e685ca27d046`. Older digests stay allowed. A digest in model settings does not unlock the flag.
- Signed CUDA component digest `b5e4d5255a48882334b27ab643c22c9c1b54e4580f82b402176a9ca5661bf6f1` on **v0.4.0-preview.20261101.1** was not overwritten. PrismML and TurboQuant stay in that signed component.
- Defaults recorded for this line: legacy host arena; context 65536; FP32 KV; MTP 2; prefill auto.
- T04 adoption is KEEP_DEFAULT (bounded policy NOT_RUN). T06 scheduler still BLOCKED. T07 hotspot **profile** captured (`STRATA_Q36_HOTSPOT_PROFILE`; dominant host bucket gdn); kernel opt-in NOT_RUN. T08 refuse PASS / measured FP16 NOT_RUN. Startup time and token speed claims are not made beyond recorded observational notes. The CUDA spill guard did not pass the reviewed baseline.

Install steps: [HAKUA_INSTALL.md](HAKUA_INSTALL.md). The detailed measured publication below for **0.4.0-preview.20261101.1** is not a measurement of this 20261103.7 app build.

## Historical measured record: 0.4.0-preview.20261101.1

GUI/Product recorded in this section: **0.4.0-preview.20261101.1**, native engine/Control **0.4.0**, Hakua-memory **0.3.5**. Integrated upstream 0.2.9 is frozen at `c62d5e3a15b40ea51c96aae9e8d36928965f53ab`. Local package checks are complete. Public release and anonymous download verification are complete; see the publication record in this section. Keep product `.6` and signed-component `.1` identities apart. An unsigned package is never a signed one.

### Distribution and installation (.20261101.1)

Use the [fork release](https://github.com/zapabob/Strata/releases/tag/v0.4.0-preview.20261101.1) for the signed CUDA / feed assets from this publication, and [HAKUA_INSTALL.md](HAKUA_INSTALL.md) for the current app installer on `.6`. Following the official Windows architecture, the app contains GUI/Host, private Node 24.22.2, CPython 3.12.15/Control, Hakua-memory and Harness. GPU runtimes are a separate signed component, downloaded explicitly from the preparation screen or imported from a file. Native CUDA, PrismML/PQ2_0 and TurboQuant are included in that component. Full model weights are separate. System Node/Python, pip and a compiler are unnecessary.

This Windows EXE is code-unsigned and uses manual app updates. Feed/component Ed25519 signatures and file checks do not constitute Windows executable signing. The own-fork default feed preserves explicit alternate feeds, uses flat GitHub asset URLs and a matching ZIP Control/bootstrap manifest. It has engine/Control entries and no automatic app-update entry.

### Final measured checks (.20261101.1)

The complete Terminal/Control and Hakua-memory run covered 2123 passed tests, 73 conditional skips and 367 passed subtests. The first complete run retained two environmental failures when the legacy Windows installer correctly refused C: below its unchanged six-GiB reserve. Its entire installer module then passed 17 tests using the existing H-drive fixture. Test sources and reserves were unchanged. The complete Host/UI suite passed 1320, with 31 conditional skips and zero failures, from the SSD source checkout using the actual installed private Node and Harness, concurrency 1 and original timeouts. Earlier interrupted H-drive runs and their failures remain recorded; they are not counted as passes.

The final EXE's 29292-file managed inventory and the actual installed 29286-file inventory match the intended source/resources, excluding generated CPython bytecode. The extracted production shell passed all 13 self-tests with System32-only PATH and the actual user's conflicting Node CA settings. The direct app update preserved all four complete data SHA256 inventories (3724 files), the selected signed engine and tray=true. The 7z passed integrity, exact three-file membership, re-extraction and installer-hash checks. Normal installed startup passed with the user's NODE_OPTIONS and NODE_USE_SYSTEM_CA retained; global settings were unchanged. The old October 27 cancelling operation recovered to cancelled and no active engine jobs remained.

Two normal-user failures were reproduced and corrected before publication. User-level OpenSSL CA options combined with NODE_USE_SYSTEM_CA=1 made Node exit 9 before Host readiness. The shell now removes only the conflicting system-CA variable from the child environment when OpenSSL was explicitly selected; it preserves certificate verification, other Node options, extra CA certificates and global settings. Desktop Rust tests pass 31, including five CA cases and the exit-9 diagnostic. A persisted cancelling load previously blocked every new operation after restart. Startup recovery now settles it as cancelled; all three cancellation regression tests pass, including the recorded failing pre-fix case. Personal job records were not manually edited.

Native engine SHA256 `b5e4d5255a48882334b27ab643c22c9c1b54e4580f82b402176a9ca5661bf6f1` is unchanged. Complete CTest passed 221 with eight conditional skips, 229 registered. The reviewed 433-entry CUDA13.3/sm120 source/compiler guard and 24 generic/optimized case hashes pass with unchanged resource budgets. This exact engine passed real arithmetic, authenticated independent memory consent, synthetic save, unload/reload, recall and unchanged read-only DB bytes: context4096, 512 expert-cache slots, prefill32, spec2, no MTP, unchanged 300s chat limit. Free dedicated VRAM was14836MiB before loading,9433MiB after first load and9796MiB after reload. Both native ports closed and Host exited0.

PrismML Hikari PQ2_0 (Q8_0 K/V, Quarter GPU) and TurboQuant Qwen3-1.7B-Q8_0 (FP16 K/Turbo3 V) passed actual memory save, full Host/model restart and recall at4096 context. Both checked per-turn independent consent, authentication, unchanged read-only DB and owned-process cleanup. Ordinary TurboQuant GUI chat passed at8192 context, including correct answers, Stop117ms, resend and four viewport sizes. These exact executable hashes match the signed component and installed selection. Earlier browser UI evidence and physical WebView2 rendering/tray/quit confirmation are separate from shell self-tests. The user also confirmed that the final installed shell rendered correctly and the previous engine-operation error was absent; native GUI automation was unavailable.

Earlier supplementary runs passed setup975 with two skips/359 subtests, server1349 registered with30 skips/no failures, desktop39 with five skips/eight subtests and six actual NSIS cleanup/locked/link/uninstall/data-preservation scenarios. These earlier results were not rerun after the final startup fixes; fresh Rust and Host results are identified separately. The approved Hermes GPU loan was restored with health, model/context and synthetic-response evidence. Its later CPU backend automatically replaced itself at port56271; current HTTP200 and the existing model server8080 were checked, and the redundant restore guard was stopped to prevent a duplicate backend.

### Package identities (.20261101.1)

Current public app installer (rebuild uploaded 2026-11-02): workspace `8870f25309ca7142a44b0042b1d16f337e4487af` (Hikari Strata-engine load / agent / 16GB hybrid VRAM tooling). Embedded Control source remains `a684b50d5c149718fdc538d71c97c8389760347d`. Signed engine/Control/feed assets are unchanged from the 2026-11-01 publication. Installed NSIS-patched shell SHA256: `a52b2eff958eed250f43bb7fe5a21b384ff792da59e5010aaa185033e8d60f4c`. Personal memories, keys and full inference model weights are excluded. The licensed small experimental control vector remains off by default.

Earlier first-publication identities (superseded for the three app assets below, retained for audit): app source `be0960a0707ab2e7fc1a90d5a7888ffd149c9858`; native source `4a6a54141ff6adabf64a16117b5ee0f8a80aff93`; shell SHA256 `ee0621a49e38aae3f3b40584a2c2fd77f1636c2f892d3e9f64751dbebff70284`; GUI receipt SHA256 `11433b751da7e7310ef6005c241323e7f7b374d8b9101995195b6fb07270d6be`.

| Asset | Bytes | SHA256 |
| --- | ---: | --- |
| `Hakua-Strata-0.4.0-preview.20261101.1-windows-x64-sm120.7z` | 113749050 | `7d02e384a502600b0e2b2143cbb73ebc8c71893af8a2fd8532b74ad823bb6fbe` |
| `SHA256SUMS.txt` | 720 | `411ab24fd7579daca3a1f73d240de65718399b1de3511b5f416988a32e7c73f6` |
| `Strata_0.4.0-preview.20261101.1_x64-setup.exe` | 113758075 | `9682b59774fa1b552d9ba09c7d0baed0028f0b7b104cfeec4c9b0a3d7eda2a26` |
| `install-manifest.txt` | 135 | `91bb97a42fce7b463dfdf28298ab5f66583b4ecee845c93ed23f29fd41d49b9d` |
| `strata-control-0.4.0-windows-x86_64.zip` | 73041197 | `2f550ea2e0a5b5693b7f42b642538dd545e46e5da94ddd4666fdd4261c4f7c6f` |
| `strata-engine-0.4.0-windows-x86_64-cuda-cuda13.tar.gz` | 1418256311 | `3972253734ce936527d75c6048bad7b7bffbcefad32085cdca6143d367ba0e84` |
| `strata-feed.json` | 1381 | `98d6a3c7b01cc0dc1dc9acb062539bac794348221200ef6e5254289ab1f3331e` |
| `strata-feed.json.sig` | 151 | `d51204e14739662b7ab2791748c3ed761d268248ebc41c3c9f5f01f57d684603` |

### Published download verification (.20261101.1)

First published 2026-11-01T19:36:42Z, release ID430808779, source tag `v0.4.0-preview.20261101.1` at `a35bb92c8fb3f53ff75ebccf1fa457519d6e0b77`. All eight public assets were downloaded without authentication and matched the then-qualified local size, SHA256 and GitHub asset digest. The latest-release pointer, latest/download feed, Ed25519 feed signature, sequence20261101160145 and flat engine/Control package URLs passed at 2026-11-01T19:39:33.566443+00:00. That latest pointer still points here after later `--latest=false` preview tags.

Installer rebuild upload 2026-11-02T09:51Z (JST 18:51): replaced only `Strata_0.4.0-preview.20261101.1_x64-setup.exe`, `Hakua-Strata-0.4.0-preview.20261101.1-windows-x64-sm120.7z` and `SHA256SUMS.txt` on the same tag via `gh release upload --clobber`. Signed engine/Control/feed/install-manifest digests were left unchanged. Anonymous curl downloads of the three replaced assets matched the staged SHA256 and GitHub digests at 2026-11-02T09:58:59Z. Local gates for this rebuild: CUDA engine rebuild + `verify-release` + NSIS `NSIS_BUILD_EXIT=0` + `/S /UPDATE` DisplayVersion `0.4.0-preview.20261101.1` + smoke `strata-desktop` start. Full Host/Python suite and 13 production-shell self-tests were not re-run for this upload; earlier 2026-11-01 suite evidence remains that of the first publication payload.

### Limits and retained failures (.20261101.1)

Windows x64/CUDA sm_120 preview, measured on RTX5060Ti. Native text is included; native image input and the separate native DeepSeek engine are absent. Other GPUs/platforms,131k context and simultaneous large-model coexistence require separate checks. Earlier89MiB VRAM refusals and memory timeouts remain recorded. TurboQuant K modes produced corrupted real answers and remain disabled; use K=FP16 and V=TURBO2/3/4. PQ2_0 and TurboQuant are separate executables. Qwen3-0.6B/Turbo3 did not produce a useful memory summary and is not memory-qualified; the1.7B combination passed.

GitHub CI is NOT_RUN. Real workflow dispatches returned HTTP422, Actions disabled, including the exact a684b50d source dispatch. Repository settings were not changed and no remote pass is inferred. Older all-in-one and split installers, including the f91df6ce candidate that failed normal-user startup, were superseded before publication. Their retained measurements do not identify this release. See the [engineering journal](../_docs/2026-11-01_hakua-official-distribution_Codex.md).

## Historical measured record: 0.3.3-preview.20261030.1

The following sections preserve the earlier package's evidence. Their installer hashes, test counts and GUI observations apply to 0.3.3 only. They are not measurements of 0.4.0-preview.20261103.7, 0.4.0-preview.20261103.7 or 0.4.0-preview.20261101.1.

GUI/Product: **0.3.3-preview.20261030.1**. Native engine/Control: **0.3.3**. Hakua-memory: **0.3.5**.

This Windows preview integrates upstream 0.2.8 at `870aedd9e1ebe1c91f94da93b59af17729f6c5c0`, the release's fixed upstream source cut. Implementation commit: `767b92fdc24aaacda58812aef2fe87869109a308`; native source was merged in `6b1139108b62e08174286357bacb57d7525b564f`. Local package qualification and downloaded-release verification are complete. [Published release](https://github.com/zapabob/Strata/releases/tag/v0.3.3-preview.20261030.1), source tag `336f074b772197ce9ec249b9586c0dab9269c369`, published 2026-10-30 02:12:17 UTC. At that publication it was the fork's latest release; its title and these instructions identify it as an unsigned preview. [Installation instructions](HAKUA_INSTALL.md).

### What changed (0.3.3)

The previous published installer had GUI 0.3.2-preview.20261029.2 and Control 0.3.1. Its engine-bundle contained two placeholder files without an engine executable. Build and bundle hooks now reject stale GUI assets, stale Control source, missing Hakua tools, changed native source/build identity, wrong component versions and changed DLL payloads. They also check the Harness revision, official Node executable and licence, approved public artwork, and third-party notices. The installer carries the newly built CUDA engine and separate PrismML and TurboQuant runtimes.

Managed readiness now selects an accepted operator credential instead of a restricted memory principal. The restricted principal still cannot read shared telemetry. Windows stale-runtime recovery now retries only sharing/lock violations for at most one second; other permission errors still propagate. Qwen4exp Swift discovery accepts supported IQ quantization names when no layout metadata is available, while unsupported variants remain blocked.

### Current source tests (0.3.3)

| Suite | Recorded result |
| --- | --- |
| Host r13, real staged Harness enabled | 1,297 passed; zero failed; 30 conditional skips; 1,327 tests. |
| Python full r20: Terminal, Hakua, models and setup | 2,994 passed; zero failed; 76 conditional skips; 716 passing subtests; 2,483.50 s. |
| Focused stale-state recovery and model discovery | 49 passed; zero failed. |
| Server r3 | 1,265 passed; zero failed; 28 skips; 1,293 registered tests. |
| Rust unit tests, unchanged Rust implementation | 25 passed; zero failed. Final packaged self-test: 13 of 13 passed. |
| Native CTest r5 | 219 passed; zero failed; eight conditional skips; 227 registered tests. |
| Public packaging/data/native integrity cases | 13 passed; zero failed. |
| Fault injection r12 | Eight mutations detected after successful baselines. |
| Public UI r6, TypeScript r3, release guard r8 | Passed. |
| Remote GitHub CI | Not run. Workflow dispatch returned HTTP 422: Actions is disabled for this repository. |

The native skips are qsa_slots_parity, iq_parity_fixtures, iq_parity, prefill_fused_iq_test_ws, expert_multi_test, native_hadamard_test, tq_cpu_parity and tq_cpu_parity_avx2. These require their optional fixtures or dependencies and are not counted as passes. The Windows ISA gate passed in the complete r5 run, checking all 121 wide functions.

Independent CPU GGML/F32 PLE captures passed the unchanged CUDA tolerances, with worst relative stage error 1.713e-3. The source/compiler/ggml-bound resource budget passed: all 432 nonzero stack/local entries matched the earlier measured values, with no added, removed or increased entries. The general allowance table is unchanged.

Python r18 found a Windows sharing-lock failure and a Swift-name discovery failure. Both were corrected before the focused green run and r20. Native r4 was invoked without the required PLE fixture path; the targeted rerun and complete r5 passed with the explicit verified fixture. Failed diagnostics and interrupted runs remain in the implementation record and are not counted as successful qualifications.

### Real inference and memory (0.3.3)

The Windows test machine has an RTX 5060 Ti, driver 621.01, Ryzen 5 4500 and 32 GiB RAM. Native memory tests use Qwen3.8-Flash-Next Q2_0, a 4,096-token context, one request slot, file-backed experts and a fictional temporary profile.

The current native executable reports 0.3.3; SHA-256: `a1bf844ce33ae68990d606fe31c2a9faa9da2dc0f3a24b10627f32eefea44403`. Current Host/Control/native r4 passed two loads, exact arithmetic response 4, memory-page consent controls, restricted telemetry HTTP403, a real SQLite memory write, full unload, a fresh-process recall of Cerulean and unchanged DB bytes for the read-only request. The monitor confirmed the first engine was absent before the second was created, and all 95 recorded owned process identities were absent after cleanup. Both cuBLAS 13 DLLs came from the engine's own package under System32-only PATH.

During concurrent build/test load, readiness took 596,809 ms and 515,688 ms, and the recall request took approximately 230 seconds. These are successful but slow runs, not a claim of interactive-speed startup. The earlier r3 diagnostic's second load exceeded its six-minute driver limit. R4 changed only that diagnostic limit to fifteen minutes and tracked loading instances for cleanup; source, payload and assertions were unchanged. The earlier pre-0.2.8 binary is not used as evidence for this result.

PrismML PQ2_0 passed two fresh-process load, arithmetic, streaming, stop and restart rounds using Hikari-LoRA-fixed-scale-PQ2_0.gguf. TurboQuant K=FP16 with V=turbo2, turbo3 and turbo4 each passed two strict arithmetic, named-greeting, stream, stop and restart rounds using Qwen3-0.6B-Q8_0. K=q8_0 with V=turbo3 also passed. These runtime payloads are unchanged by the upstream integration.

TurboQuant K modes produced corrupted responses and are excluded from the packaged capabilities. The earlier symmetric TURBO4 diagnostic accepted a numeral inside a degraded response and is superseded by strict checks. PrismML PQ2_0 and TurboQuant are separate executables; PQ2_0 with TurboQuant cache quantization in one process is not offered. Their chat paths do not share the native Hakua memory sidebar. [Backend details](LOCAL_LLAMA_BACKENDS.md).

These checks do not establish general model quality, a speed improvement, multiuser isolation, long-context performance or every optional engine feature. Loading and prefill can take minutes on this PC.

### Package integrity (0.3.3)

The package uses official Node 24.22.2, SHA-256 `1c331340fa6cac5e4294725959070a143fa0507195a78ad77eed7943f719aac9`. The reviewed Hakua wheel is selected by exact URI and hash, with hashed dependency installation at build time. App startup does not install Python packages or compile code. Windows credential files use a protected owner-only DACL. The public UI uses approved pinned Lucide assets, an owned MIT icon and system fonts.

There are 94 inventoried third-party notice files. The unchanged upstream 483,520-byte control vector is included, off by default, with its full Qwen Community License 1.0. [Provenance](../data/experimental-speed-projection/PROVENANCE.json) pins its source and licence bytes. Vector SHA-256: `ef0724c5b79297e481017be85769832bc26eb220c984ecba2b0e1b0c8426312b`. The guard permits exactly this declared GGUF and rejects other GGUFs or live profile data in Control. Complete inference model weight packs, personal memories and user keys are not distributed.

### Installer and publication (0.3.3)

R6 completed a fresh full Tauri compilation. Its first bundle guard stopped on a transient Prism execution/source probe failure; the unchanged executable and unchanged probe then passed. R7 repeated the complete bundle guard and produced NSIS from that fresh shell. R8 qualification passed extracted-source/payload comparison of 29,324 application files and all 13 packaged self-tests under System32-only PATH. An independent normal-window extraction had a responsive Strata window; its PE product version and API both reported 0.3.3-preview.20261030.1, and Control reported engine 0.3.3. All 15 recorded owned window/child process identities were absent after cleanup. The build-wait helper's earlier 30-minute orchestration limit and the successful repeat are retained in the implementation record.

The outer 7z passed its integrity test and actual re-extraction. It contains exactly the installer, INSTALL.md and an inner installer checksum. The re-extracted installer bytes match the standalone EXE.

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| Strata_0.3.3-preview.20261030.1_x64-setup.exe | 799,979,767 | `478743a7cbbe633d09101102f2df4be7d4d254bf4119a251ab19c7f1fd4a3533` |
| Hakua-Strata-0.3.3-preview.20261030.1-windows-x64-sm120.7z | 793,967,757 | `0958e9d748a0772cfaaa70b64dfcb9470524dea8c36f18f309827e8c80588373` |

All three uploaded assets were downloaded again into a fresh directory. Their bytes match the local EXE/7z/checksum files and GitHub's SHA-256 digests. Publishing preserved the same asset IDs and digests. Release ID: `428581199`; EXE asset `690609303`; 7z asset `690609304`; checksum asset `690609312`. SHA256SUMS.txt is 239 bytes, SHA-256 `c52aaeba06c54bd811ca8e26077aa54ed4136e3fd250eeadf1c700f3fa204df9`.

The release source tag is fixed at the documentation/qualification commit above. Its production runtime is implementation `767b92fd`; later commits containing only publication receipts do not change that runtime. Remote CI is unrun because the repository rejects dispatch with Actions disabled; the suite results in this document are local results. The repository policy was not changed. Earlier R5 diagnostic installer/7z artifacts are excluded from publication.

The deliverables are `Strata_0.3.3-preview.20261030.1_x64-setup.exe`, `Hakua-Strata-0.3.3-preview.20261030.1-windows-x64-sm120.7z` and `SHA256SUMS.txt`. The 7z contains the same installer and instructions; it is not a portable app.

The current normal window and API passed the observations above. CUA could not start its native app-server, and its in-app browser provider was unavailable. Visual interaction and clicks are not recorded as passed. Upgrade through the user's real uninstall registration has not been run. The existing installed copy has not been replaced during scratch tests.

This is an unsigned manual-update preview. An owned signed automatic-update feed/key is not qualified. Windows x64/CUDA sm_120 on the stated hardware is the runtime qualification scope. HIP, SYCL, Linux, other GPUs and Metal require separate evidence; Metal's source version is not a tested Metal artifact.

### Test cleanup incident (0.3.3)

A preexisting planning test recursively removed part of the real H-drive Strata build/dependency area. That cleanup was removed. Node tests now use a dedicated scratch tree with recursive-cleanup guards, including parent-junction escapes. Known pinned sources and the reviewed experimental Prism patch were reconstructed; both preexisting worktrees had no tracked deletions. Current release inputs were rebuilt or rehashed independently. Complete recovery of every old untracked byte or build output is not established. The correction and limits remain in the [implementation record](../_docs/2026-10-30_hakua-release-integrity_Codex.md).
