# Hakua 0.6.0 preparation helpers

These are new preparation recipes, not release receipts. No build, packaging, signing, installation, model run, download or publication was executed by this task. `releaseQualified=false` remains mandatory. Qwen3.8 prototype work and missing/deleted Qwen3.6 weights are outside this delivery; no model is copied or downloaded.

The controller owns the final reviewed commit, native compilation/binding, quality records, installed-app checks, backup/standard overwrite, tar.7z delivery and public release. The helper derives its checkout from its own tracked location. It accepts only `C:/Users/downl/Documents/ChatGPT/Strata` on `main`, or the temporary migration checkout `H:/Strata/worktrees/upstream-sync-20261116` on `codex/upstream-sync-20261116`. Other roots, mismatched branches and reparse/path aliases are refused. Every execution still requires an explicit clean full commit; no release build is implied by migration to main.

## Requirements and implementation

R1: frozen identity. `release.py freeze` captures the exact clean commit/tree and SHA-256 of every tracked physical file. Every other phase independently requires the same explicit clean current commit and 0.6.0 version files. The native binding must name the actual executing checkout. Moving the helper from H: to C: does not rebind the old H: CUDA build or certify its bytes as a new build.

R2: unchanged dependencies. `dependencies` verifies the reviewed Harness commit and output-budget patch, its missing-import check, RustXNG Full version/source/binary/licences, local-backend manifests and explicitly supplied pinned public assets. It inventories managed Harness/local-backends before and after copying and compares every destination hash. When both physical resource paths are exactly the primary resource root, it instead verifies twice and compares complete inventories before and after, recording `verified-in-place` and `copied:false`. Other destinations must be absent or contain only tracked scaffolding; existing payload causes refusal. This exception never applies to Control, notices or native component replacement. Node 24.22.3 executable and licence are independently pinned. No model/profile copying or recursive delete/move is implemented.

R3: fresh Terminal/Control. `control` requires an existing CPython 3.12.15 archive matching the reviewed source pin and a flat wheelhouse accepted by the current 25-package hash lock. It copies these inputs into the fresh attempt, invokes the production Terminal packager with the explicit reviewed commit and offline wheels, stages a new private Control directory with the production stager, validates version/lock/installed packages/COMMIT, fills the integration's scaffold-only Control resources and packs a new Control ZIP. Production tools may remove their own fresh extraction scratch; these targets are prechecked absent under the new attempt. Nothing is installed into the global interpreter or the user's app.

R4: current notices. The existing production `stage-notices.py` still insists on the historical external ggml repository commit. This task does not edit it. `notices` follows its collection pattern while taking the actual vendored `third_party/strata-ggml/LICENSE`, current Node licence, new Host dependency notices, fresh Control Python/distribution licences, current libstrix attribution and Qwen licence. Existing Microsoft documents come from an explicitly preserved physical `--prior-notices` directory under release `inputs`, with `--prior-notices-manifest-sha256` supplied from its verified preservation record. Complete membership, manifest bytes, listed file hashes and current vendor pins must match. The preserved input must not overlap the resource target or attempt. The controller must preserve the old resource tree and prepare a fresh/scaffold-only target before execution; this resolves the former circular dependency on that same occupied target. This phase also requires an explicit CUDA toolkit directory. No claim of a supply-chain audit is made by copying these bytes.

R5: fresh native signing/package. The controller provides a fresh unsigned native package and the binding receipt specified below. `verify.mjs native` checks both executables, exact native-build bytes, current native sources, actual vendored ggml identity, current source commit and the production verifier. `component` uses production composition with current verified local-backends/notices, copies into a fresh signing directory, passes only an external key path to the production signer, verifies using bundled trusted keys and full payload verification, checks that only the manifest changed, checks DLL coverage, packs a new engine tar.gz and stages the exact signed tree. It carries no historical numerical/model-test counts. Integrity verification is not a quality qualification.

R6: app NSIS. `app` requires the fresh native receipt and a clean source commit. It uses the integration's production Tauri build hook and release verifier, pinned Node, locked/offline Cargo, a fresh H: Cargo target, public assets, explicit native source/manifest identity and five compiler jobs. The standard NSIS installer is copied to the official-style filename with SHA-256 verification. It does not install it, sign it with Authenticode, or create the final delivery tar.7z.

All mutating phases require `--execute`. Plans do not inspect private keys or create the release directory. Each phase runs inside its own Windows Job, admitted with unchanged available RAM 3 GiB, commit headroom 4 GiB, H: free 10 GiB and C: free 5 GiB floors. The guard observes those floors during the phase and drains only its owned descendants. Five compiler jobs and one test job are explicit. No process memory cap is invented. Receipts are published only after successful observed child exit and zero remaining owned processes. Every attempt uses a fresh phase/rN directory; failures are retained, never retried automatically. Component signing suppresses all signer output and never reads private key contents itself.

## Later controller invocation sequence

Review and commit changes with the final source first; the clean-source gate intentionally refuses uncommitted changes. Use PowerShell from primary C: main. The H: integration worktree has been removed. The placeholders below are required verified inputs, not claims that all preparation has completed. Existing managed-resource preservation and dependency-cache prerequisites still apply; this path migration does not execute staging or make existing payloads replaceable.

```powershell
$py = 'C:/Users/downl/AppData/Local/Programs/Python/Python312/python.exe'
$helper = 'C:/Users/downl/Documents/ChatGPT/Strata/_docs/stable-060-20261116/release.py'
$commit = '<final reviewed full 40-character commit>'
$pythonCache = 'H:/Strata/evidence/upstream-sync-20261116/control-inputs-v2/python-cache'
$wheels = 'H:/Strata/evidence/upstream-sync-20261116/control-inputs-v2/wheels'
$assetArgs = @('--public-assets', 'H:/Strata/releases/stable-060-20261116-codex/inputs/public-assets',
  '--public-assets-receipt', 'H:/Strata/releases/stable-060-20261116-codex/inputs/lucide-sources/provenance.json',
  '--public-assets-receipt-sha256', 'c1b03b0352525867d5f6bb7864790f8ca0e2d11f28731c1de73d3c16322c3abf')
$cuda = '<actual CUDA toolkit root with EULA.txt and LICENSE>'
$priorNotices = 'H:/Strata/releases/stable-060-20261116-codex/inputs/prior-notices'
$priorNoticeHash = '<NOTICES.json SHA-256 from the verified preserved copy>'
$native = '<fresh root-built unsigned 0.6.0 native component folder>'
$binding = '<fresh controller binding receipt JSON>'
$keyPath = '<absolute existing signing-key path outside all source/release trees>'
uv run --offline --no-project --python $py python -B $helper freeze --reviewed-commit $commit @assetArgs
uv run --offline --no-project --python $py python -B $helper freeze --reviewed-commit $commit @assetArgs --execute
uv run --offline --no-project --python $py python -B $helper dependencies --reviewed-commit $commit @assetArgs --execute
uv run --offline --no-project --python $py python -B $helper control --reviewed-commit $commit --python-cache $pythonCache --wheelhouse $wheels @assetArgs --execute
uv run --offline --no-project --python $py python -B $helper notices --reviewed-commit $commit --cuda-dir $cuda --prior-notices $priorNotices --prior-notices-manifest-sha256 $priorNoticeHash @assetArgs --execute
uv run --offline --no-project --python $py python -B $helper component --reviewed-commit $commit --native-component $native --native-binding $binding --key $keyPath @assetArgs --execute
uv run --offline --no-project --python $py python -B $helper app --reviewed-commit $commit --native-binding $binding @assetArgs --execute
```

Default attempt is `r1`; use `--attempt r2` only after reviewing the failed receipt and fresh resource state. A failed phase may have filled managed integration resources. Those bytes are retained and the helper will refuse to replace them; the controller must inventory and preserve them before any new staging attempt. It must not reuse a failed receipt as success.

Expected output root is `H:/Strata/releases/stable-060-20261116-codex`. Each phase has its own `reservation.json`, `owned-job.json`, `pending.json` and final `receipt.json`. The Control phase carries Terminal archive, Control archive and source/package inventories; the component phase carries unsigned/signed component and archive assets; the app phase carries its fresh Cargo output and `Strata-0.6.0-windows-x64-setup.exe`. Published names/metadata, signing of the feed, previous DeepSeek feed-member handling, all README final publication wording and final tar.7z packaging remain controller responsibilities.

The native binding receipt contract is:

```json
{
  "sourceCommit": "<final reviewed full commit>",
  "sourceRoot": "C:/Users/downl/Documents/ChatGPT/Strata",
  "releaseQualified": false,
  "nativeBuildSHA256": "<sha256 of fresh native-build.json>",
  "binaries": {
    "strata.exe": "<sha256 of fresh main executable>",
    "strata-q36.exe": "<sha256 of fresh Q36 executable>"
  }
}
```

The controller must derive this from its actual successfully guarded build/bind execution and current bytes. Writing hashes alone is not proof of compilation. Existing `tools/build_hakua_engine.mjs` binds the main executable; this extra receipt also binds the actual Q36 executable. The unsigned package's `build_id` must equal the commit, `native-build.json` must contain current source inventory and `recipe.ggml.mode=vendored`, and the native package must carry both executables. The compose helper may append its standard `-hakua-runtimes-v1` build-id suffix.

## Verification performed

`uv run --offline --no-project --python C:/Users/downl/AppData/Local/Programs/Python/Python312/python.exe python -B _docs/stable-060-20261116/test_release.py -v` initially passed six boundary tests. Independent review then found inherited `TAURI_CONFIG` could alter a production build. The original implementer removed it from the actual child-environment boundary and added a cleared synthetic-environment regression. The final seven cases passed; scoped re-review marked SPEC and QUALITY ADDRESSED. They cover default nonmutation, malformed attempts, worker bypass, incomplete commits, existing payload refusal and the configuration boundary. They do not execute build or signing code.

The reviewed final `release.py` SHA-256 is `d852f33b096d2ffc627d2d4935510b47b64e32f545525cde976cb6698e455bad`; `verify.mjs` is `f4604f4e950c56c8c7ca73bda143b0618d5d4e18789bba9f35bf7bbf6ba53324`. Private fix-review artifacts are retained outside the source tree under `H:/Strata/evidence/upstream-sync-20261116/release-helper-private-v1`; no environment values are included in publication artifacts.

Python AST parsing passed. The `--help` command and default `app` plan passed. Node `--check` passed for `verify.mjs`. Black formatting at line length 110 and flake8 at maximum line length 140 passed using already installed tools through offline uv. uv emitted pre-existing dangling dist-info warnings; no dependency installation was attempted.

Read-only actual dependency verification passed against `C:/Users/downl/Documents/ChatGPT/Strata/apps/desktop/src-tauri/resources`, using the staged Node 24.22.3 executable and this new `verify.mjs`. It returned `ok=true`, `mode=dependencies`, `releaseQualified=false`. This confirms the checks described in R2 at verification time, not post-copy staging, fresh build or runtime inference. The planned release output directory was checked absent after the tests.

The separately authorised Metal workflow correction is staged and documented in `fix1-report.md`, `metal-fix1.diff` and its SHA-256 sidecar. Actionlint passed without optional shellcheck/pyflakes. The controller reported independent scoped re-review PASS. Hosted Mac compilation remains NOT_RUN here.

NOT_RUN: all `--execute` phases, new Terminal/Control packaging, notices staging, signed native packaging, NSIS build, archive extraction/readback, GPU/model inference, installed app QA, installation/rollback and public publication. Explicit unresolved inputs are the final reviewed commit, current flat wheelhouse, actual CUDA root, fresh native component/binding receipt and signing-key path. The known cached Python archive location above was inventoried by filename only; its contents will be rehashed against the source pin by the Control phase before use.

## Applied standards and retained risks

Read the canonical `C:/Users/downl/_docs/prompts/CLAUDE-FABLE-5.md`, PC-wide/home and Codex instructions and integration AGENTS. Selected common engineering, Implementation Start Gate, Skill Selection, Security, Application Development, Python and TypeScript SOPs. Applied `milspec-codex-standard`, `build-manager`, `evidence-changelog-closeout` and review-only `agent-self-review-evolution`. Current versions/pins were taken from current source and existing receipts, not inferred from old release claims. No formal MILSPEC certification is asserted.

The changes are restricted to this new helper directory and the separately authorised Metal workflow. No old recipe/result, primary source, runtime profile, key or model was changed. Source/data rollback is retention: failed attempts stay in their unique directory and no automated rollback/deletion runs. Full package runtime and owner acceptance are still controller gates. The guard depends on the repository's existing Windows Job implementation; these tests do not exercise job containment or memory-pressure behavior.

Self-review-event: ticket_completed for preparation artifacts only; qa_defect for the independent Metal workflow finding. Decision: review-only for governance. The useful narrow prevention is explicit dry-run/overwrite boundary tests and actionlint; no AGENTS/SOP/skill modifications are warranted.

## Migration follow-up: primary main and protected signing-key location

This follow-up changes only `release.py`, `test_release.py` and this report, against source `4282fce334e0d6c1843c74bca5d4063bd784bc05`. `ROOT` is derived from `HERE.parents[1]`; a fixed root/branch allowlist and physical-path checks enforce the two approved migration locations. `HERE` retains the lexical absolute path until validation so symlink/junction aliases cannot be silently resolved into an approved path.

The key-location check no longer rejects the whole H:/Strata parent. It requires an existing absolute regular file through a path without reparse points, resolves its location, and rejects containment within either approved source checkout, any H:/Strata/worktrees tree, any H:/Strata/releases tree, or the actual release output. A protected private-keys sibling is therefore permitted without copying it. Only metadata/path checks occur in this validator; key contents are neither read nor printed. Production signature verification with bundled trusted public keys and the inherited `TAURI_CONFIG` exclusion remain unchanged.

Four focused synthetic cases were added for accepted and mismatched checkout/branch pairs, path-alias refusal, protected-sibling versus source/release key containment, and linked-key refusal before resolution/read. They clear the test environment and never open a real key. All existing seven tests plus these four passed: 11 total. Python AST, Black check (110 columns) and flake8 (140 maximum columns) passed. No dependency, build, signing, model, installation, publication or Git commit command ran in this follow-up.

The existing `_docs/upstream-sync-20261116/check.py` guard ran `helper-primary-migration-tests-v1` with `python -B _docs/stable-060-20261116/test_release.py -v`, and `helper-primary-migration-static-v1` with AST parsing followed by the Black and flake8 commands. Receipts are under `H:/Strata/evidence/upstream-sync-20261116/receipts/`; command-log SHA-256 values are `d88e0bb0be7576613f2bed099c8d59859e3cfea0982bf9138372237789c2adc3` and `817b9e6364f77bdf917dde4e09dacf2161c1760c90bb6f61d6ba961ce8eecb82`, respectively. Both owned Jobs finished with exit 0, `termination_confirmed=true` and `job_active_after=0`. No owned test/static-check job remains active.

The exact code diff is available with `git diff 4282fce334e0d6c1843c74bca5d4063bd784bc05 -- _docs/stable-060-20261116/release.py _docs/stable-060-20261116/test_release.py`. It contains 77 insertions and 13 deletions before this report update. Final physical SHA-256 values: `release.py` = `19cee832591dd72ff55dfa2995d9051ecb147aa527348c95bfffd2c1446db947`; `test_release.py` = `af9ba409476ef512ceed6c31210c55b691df3173bbbf9795816ba8453c2b4337`.

Release execution from C: remains NOT_RUN. In particular, existing managed resources cannot be overwritten by rerunning staging, and the currently pinned Host dependency cache remains a separately verified prerequisite. The controller owns source integration, push and exact-target cleanup; none is performed by this helper task.

## Explicit prepared inputs after cleanup

This follow-up is based on clean C main `ed2c923a1da45c55906e643e24b8b66a799b2e2c`. It removes the helper's dependency on the deleted historical public asset directory and receipt. It does not reconstruct that receipt or claim recovery of its acquisition history. The controller prepared new inputs with unchanged `packaging/full_assets.py`; the explicit example above names its new receipt. Executing dependencies, component or app now requires all three asset arguments. Other phases accept the same arguments to bind the complete preparation sequence.

The helper requires existing absolute physical paths under the release output's `inputs` directory, rejects reparse points and unsafe inventory names, verifies the caller's expected receipt SHA-256, pinned Lucide commit, exact membership and every output hash. It records the full identity in the parent reservation, checks it in the owned worker before and after its phase, and checks again before publishing the final receipt. The Node verifier receives the same explicit paths and expected hash and runs production `verifyPublicAssets`. This is new input verification, not a replacement for source acquisition evidence. The expected receipt hash must come from the controller's reviewed preparation record. Before/after checks detect drift; these checks do not provide filesystem immutability against a concurrent privileged writer.

Only unchanged Harness/local-backends may use the exact-primary verified-in-place branch. Fresh Control, notices and signed native payloads retain the existing overwrite refusal. The Host dependency prerequisite now names `H:/Strata/evidence/upstream-sync-20261116/host-deps-060/node_modules`, checks the physical target, and requires Vite 6.4.6. Both notices and app phases enforce it. The controller must prepare the primary dependency entrypoint and reviewed lock; this helper never installs dependencies or changes that route. The controller reports that primary Node is now a physical pinned 24.22.3 payload. Fresh C-source native binding and preservation/preparation of existing Control/notices/native staging remain separate required gates.

Validation used the existing C-root `check.py` owned-Job guard through offline uv and Python 3.12. Initial 20 Python boundary tests passed. The added actual Node boundary case initially failed with exit 134 because an entirely empty Windows environment did not initialise Node; the test now supplies only the fixed OS root in an otherwise synthetic environment. Final `helper-inputs-fix-tests-v3` passed all 21 tests, retaining the original 11. It covers CLI-to-worker-to-Node propagation, expected receipt identity, changed bytes/membership/receipt, traversal and linked/outside input refusal, double in-place verification and drift detection, separate-target overwrite refusal, and exact Host cache/version checks. Actual Node accepts the synthetic valid inventory and rejects both a different receipt hash and changed asset bytes. No caller environment dump or real signing key is used.

Final `helper-inputs-fix-static-v2` passed Python AST parsing, Black check at 110 columns, flake8 at 140 columns and Node syntax checking. The final test command-log SHA-256 is `025778a19d3052d50fdcb77bf5ef195f1d97916a0e352c8414ed56fa3a7359c6`; the static command-log SHA-256 is `42952929f98ff284b84ffc33f5fca4e517620f9b2ceb3fca1375e504affa8195`. Guard receipts record successful termination and zero remaining owned processes. RAM/commit/H/C floors remain 3/4/10/5 GiB, compiler jobs five and test concurrency one. Existing offline uv dangling dist-info warnings remain unchanged.

Only this report, `release.py`, `verify.mjs` and `test_release.py` were changed. Exact before/after files, baseline diff, hashes and this task's guard receipts are frozen under `H:/Strata/private-backups/cleanup-20261116/release-helper-inputs-fix-v1`. Applied the previously selected common/start/skill/security/Python engineering SOPs, build-manager and evidence-changelog-closeout; this does not assert formal MILSPEC certification. NOT_RUN: any actual release phase, current dependency preparation, native/product build, signing, installation, GPU/model qualification or publication. `releaseQualified=false` remains unchanged. No commit or push was performed by this implementation task.

## Fix round: preserved notices input

The controller's execution preparation found a real circular requirement in notices on C main: the phase read its old notices from the same resource directory where generic staging correctly refused existing payload. This round is based on `a8139c8c702f520000347f3c13ed40e935a51c05`. The original implementer read the subagent-driven-development fix-loop instructions and repaired the source/target separation. Only `release.py`, `test_release.py` and this report changed; generic `stage()` and `verify.mjs` remain unchanged.

The two new prior-notices arguments are required for notices. They are optional for other phases; when neither is passed, unrelated phases do not examine any prior input. When provided, the helper checks absolute physical containment beneath release `inputs`, no reparse points, no target/attempt overlap, the expected manifest SHA-256, safe relative member names, exact membership and every listed hash, and the current three Microsoft vendor pins. Only the manifest itself and optional `.gitignore` scaffold may be outside the listed manifest entries. Both are included in the complete reserved identity. The parent reservation, worker start, worker end and final receipt all bind this identity. The notices phase checks it again after copying. These checks detect drift but do not create filesystem immutability against a privileged concurrent writer.

Read-only inspection before this fix confirmed the current C source notices had 95 listed files, all matching; 97 total files included only `.gitignore` and `NOTICES.json` as extras. All three current Microsoft pins matched. The inspected manifest SHA-256 was `fc038fd1f512bfaa40fc8598387ddbeac7c0c7fb8e0ad4abd2b26d5b24584ce7`. This inspection is not a receipt for a backup that has not yet been made. The controller still owns verified preservation of that tree, the explicit input copy and fresh resource-target preparation. No primary resource was copied, deleted, moved or overwritten by this fix task.

The existing guarded test command is `uv run --offline --no-project --python C:/Users/downl/AppData/Local/Programs/Python/Python312/python.exe python -B _docs/upstream-sync-20261116/check.py notice-input-fix-tests-v2 python -B _docs/stable-060-20261116/test_release.py -v`. All 30 tests passed, retaining the prior 21. The nine new synthetic tests execute notices against temporary licences and a separate preserved input with no old primary notices, verify its fresh output, prove occupied-target refusal and byte preservation, and cover optionality, CLI propagation, complete scaffold/manifest identity, bad receipt hash, traversal, changed membership/payload/vendor bytes, reserved scaffold drift and target/attempt/outside/reparse refusal. The synthetic test environment is cleared and no real key or profile is accessed.

`notice-input-fix-static-v1` used the same offline uv and existing guard for Black formatting/check at 110 columns, Python AST parsing and flake8 at 140 columns; all passed. The static command-log SHA-256 is `2c2ee2bf44822b0f183bb93f371b086e89c2a73d470e2c9a72310735a980d5c3`. Exact commands, outputs, before/after sources, diff and hashes are retained under `H:/Strata/private-backups/cleanup-20261116/notice-input-fix-v1`. The guard retains RAM/commit/H/C floors 3/4/10/5 GiB, five compiler jobs and test concurrency one, and confirmed termination with no remaining owned jobs. This task ran no product build, actual release phase, native/GPU test, signing, installation, commit or push. Independent scoped re-review is the controller's next gate.
