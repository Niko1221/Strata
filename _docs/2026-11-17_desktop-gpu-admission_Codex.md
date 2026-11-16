# Installed acceptance desktop GPU admission repair

The release-only installed real-model controller previously rejected every PID
returned by NVIDIA query-compute-apps. The retained original observation showed
24 explicitly inspected Windows desktop processes reported as C+G and about
14811 MiB free. Its original admission failed. C+G includes compute contexts;
the repair does not redefine it as graphics-only or infer safety from a name.

Scope began on clean primary main c7a5303283e969233506605dac499cada1293108.
Only the release acceptance adapter, its shared observer, focused tests and this
record changed. Native/app product sources, prior source-selection/GPU pins,
root's private installed-Q38 harness, resource guards, installed software and
public release assets are outside this change. No model, compiler, download,
installation, process termination or publication was performed.

The new `_docs/stable-060-20261116/gpu-admission.py` implements the observation
and policy shared by Python entry admission, Node immediately before load and
reload, each runtime attestation, and final receipt replay. Explicit
`--gpu-desktop-baseline` and `--gpu-desktop-baseline-sha256` options bind the
private pre-existing baseline. Omitting both preserves strict admission without
a baseline. Supplying only one, a relative path or an invalid hash fails.
There is no baseline creation, regeneration, process-name allowlist or retry.

Each observed baseline PID must retain its pinned parent PID, CIM creation
timestamp, executable path, executable SHA-256 and C+G classification. Its
disappearance is allowed. Unknown additional processes require a live ancestry
chain to the exact owned worker, including creation order and the native
creation FILETIME of every chain member. PID reuse, inaccessible identity,
changed file/path/time/class, unknown C+G or compute owner, inconsistent NVIDIA
enumeration and another GPU fail closed. Native observation uses only
PROCESS_QUERY_LIMITED_INFORMATION and closes its handles.

An actual CPU identity test exposed that CIM truncates the last 100-nanosecond
FILETIME digit. The controller now records exact GetProcessTimes FILETIME in
addition to the original CIM timestamp. Native/CIM timestamps must agree at
CIM's measured precision; worker-to-Job and ancestry comparisons retain exact
native FILETIME. No tolerance is applied to the Job root identity. The failed
test is retained in native-desktop-gpu-final-r4. A subsequent stale negative
fixture failed because it changed CIM time without changing its synthetic
native time; native-desktop-gpu-native-time-r5 is also retained. The fixture was
corrected to represent an actual reused parent, with both timestamps advanced.

The original entry floor remains 8192 MiB and runtime reserve remains 768 MiB.
The policy uses the lower free-memory value from CSV and XML observations.
RAM 3 GiB, commit 4 GiB, H: 10 GiB/C: 5 GiB floors, Job containment, deadlines
and five-job ceiling are unchanged. Powershell observation uses the absolute
Windows executable because the production sanitized PATH omits its folder;
the original command-argument ancestry probe uses that same explicit path.

Receipts retain observedPIDs for all GPU processes, computePIDs for all C/C+G
contexts including the baseline, desktopPIDs for pinned identities, ownedPIDs
for verified attempt descendants and unapprovedPIDs, which must be empty for a
successful classification. Raw NVIDIA XML/query data and process identities
are replayed by the common validator rather than trusting an empty computed
list. Runtime attestation must additionally contain the actual owned compute
PID with the separately verified Prism executable/hash. sharedDesktopGPU and
performance/concurrency UNQUALIFIED remain explicit. Existing quality flags,
including releaseQualified=false, are preserved.

The helper joins ADAPTERS and the receipt's pinned source inputs. The interpreter
is separately hashed and bound to Node's helper invocation and the final
receipt. Baseline pins also bind to the original worker CLI. Private baseline
and raw observations remain under H:/Strata/evidence, or the private attempt
root; they are not copied into tracked source or shipped/public assets.
Failed/partial observations are retained without automatically retrying.

Verification used the existing custom owned-Job runner and trusted Python
3.12.10 interpreter, rather than changing dependencies or environments. This
is the Python SOP's custom-wrapper exception: the wrapper establishes the
required containment, trusted Node, filtered environment and resource floors.
Every command's exact argument vector, complete output and guard outcome are
in H:/Strata/evidence/upstream-sync-20261116/receipts. The task report at
`.superpowers/sdd/2026-11-16_upstream-sync-plan_Codex/native-q38-desktop-gpu-admission-report.md`
lists the command names and final results.

PASS: native-desktop-gpu-focused-final-r7 ran the new focused GPU policy module
and only three directly affected real-model consumer/worker test selections:
42 passed, zero failures/skips, 2.20 seconds. This was not a replay of the
earlier broad 42-case suite or ten-case GPU qualification. Its guard observed
native exit 0, containment before resume, termination confirmed, Job active
after 0, zero dropped output, minimum available RAM 13446742016 bytes and
peak Job commit 146399232 bytes. Representative tests cover baseline equality
and disappearance, pure compute/unknown C+G, path/hash/time/PID changes,
baseline tamper, owned/reused/unrelated ancestry, unavailable identity,
NVIDIA enumeration disagreement, both VRAM floors and receipt tampering.

PASS: actual read-only preflight with the original baseline SHA
3e4bf851763bcd8b36e19860484ead23a5c70115d7816eca9f7226228b13f57f:
24 observed PIDs, 24 pinned desktop identities, zero owned GPU descendants,
14810 MiB free. All 24 remain reported as C+G compute contexts. Final native
identity evidence is gpu-desktop-admission-preflight-r3.json, SHA-256
e55181245bd2241158c16ea791ac9e67f1c553c0d06868db15c26e719009c5ef.
Its guard native-desktop-gpu-live-preflight-r2 ended native exit 0/Job 0 with
minimum RAM 13891444736 bytes. Earlier passing preflight-r2 is retained too.

PASS: new helper/test Black formatting and flake8 checks, actual current-process
CIM/native identity with a PATH containing only Windows/System32 and Windows,
and Node syntax validation. The restricted-PATH case is recorded separately
as native-desktop-gpu-restricted-path-r8; Node syntax is
native-desktop-gpu-node-syntax-r2. No broader tests were repeated after the
last absolute-path-only observation change.

Canonical prompt, PC/project AGENTS, common/start-gate/skill-selection/security/
application/Python/TypeScript SOPs, milspec-codex-standard and
incremental-implementation were applied. Self-review-event: ticket_completed
and qa_defect; review-only governance decision. The actual CIM precision test
caught and prevented a false final Job rejection. No governance or memory was
changed. The remaining action is independent review followed by root's final
source freeze/build/install/real-model acceptance/public readback. This change
does not qualify inference, general quality, full-window performance or GPU
concurrency. Process exit/update or unobservable ancestry can intentionally
stop a later attempt; inspect retained evidence and obtain a fresh explicit
policy ruling rather than broadening the baseline automatically.
