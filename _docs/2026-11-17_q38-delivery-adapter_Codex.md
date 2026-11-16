# Q38 delivery adapter implementation log

The bounded package adapter starts from clean primary main963b6a19 and retains
the separately compiled6ee Q38 executable706867da. It fixes dependency receipt
identity collision and CMake blank/comment parsing, adds frozen portable Q38
provenance plus the actual MSVC redistributable closure, and extends existing
package/source checks and notices. The report records requirements, decisions,
changed producers/consumers, commands, evidence, limits and controller handoff:
[full adapter report](../.superpowers/sdd/2026-11-16_upstream-sync-plan_Codex/native-q38-delivery-adapter-report.md).

Applied instructions are the canonical preimplementation prompt; PC, Codex and
project AGENTS; common engineering, start gate, skill selection, security,
application, Python and TypeScript SOPs. Skills used are milspec-codex-standard,
build-manager and agent-self-review-evolution. Existing custom guarded Python
3.12 is the Python SOP wrapper exception; no dependencies were installed. No
formal MILSPEC certification is asserted.

Final focused Python22 and Node9 tests passed with drained owned Jobs and zero
RAM-floor breach samples. Real Q38/CRT PE closure and actual ephemeral signed
Control sibling resolution are covered; no production key was accessed. Black,
Flake8 and diff whitespace checks passed. The first failed new Node filename
fixture and all earlier evidence remain retained. Notice regression evidence is
linked in the full report. Build/sign/install/publication remain root-owned.

Self-review-event: ticket_completed and qa_defect, q38-delivery-adapter. Decision:
review-only; the direct regression catches the actual filename validation defect
and complete PE dependency inspection catches the missing CRTs. Existing controls
are sufficient; no governance or memory edits. Source rollback is a scoped commit
revert. PQ format acceptance remains2/4; no full-window or general-quality claim.
