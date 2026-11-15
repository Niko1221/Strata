# Hakua 0.6.0 preparation helpers

These are new preparation recipes, not release receipts. No build, packaging, signing, installation, model run, download or publication was executed by this task. `releaseQualified=false` remains mandatory. Qwen3.8 prototype work and missing/deleted Qwen3.6 weights are outside this delivery; no model is copied or downloaded.

The controller owns the final reviewed integration commit, native compilation/binding, quality records, installed-app checks, backup/standard overwrite, tar.7z delivery and public release. These helpers use the integration checkout at `H:/Strata/worktrees/upstream-sync-20261116`, never primary C: main. The controller subsequently discovered official changes through `baf16363b4403ef0ed4f538f5d5ed11c43fb9535`; therefore no commit was hard-coded as the accepted source. Every execution requires the final clean full commit on `codex/upstream-sync-20261116`.

## Requirements and implementation

R1: frozen identity. `release.py freeze` captures the exact clean commit/tree and SHA-256 of every tracked physical file. Every other phase independently requires the same explicit clean current commit and 0.6.0 version files. This recipe deliberately selects the integration checkout as the native source root, which matches the controller's existing CUDA build. It does not create a detached source worktree or pretend an old build belongs to one.

R2: unchanged dependencies. `dependencies` verifies the reviewed Harness commit and output-budget patch, its missing-import check, RustXNG Full version/source/binary/licences, local-backend manifests and pinned public assets. It inventories managed Harness/local-backends before and after copying and compares every destination hash. Node 24.22.3 executable and licence are independently pinned. Model/profile trees and primary resources are not modified. Destinations must be absent or contain only tracked scaffolding; existing payload causes refusal. No recursive delete/move is implemented by this helper.

R3: fresh Terminal/Control. `control` requires an existing CPython 3.12.15 archive matching the reviewed source pin and a flat wheelhouse accepted by the current 25-package hash lock. It copies these inputs into the fresh attempt, invokes the production Terminal packager with the explicit reviewed commit and offline wheels, stages a new private Control directory with the production stager, validates version/lock/installed packages/COMMIT, fills the integration's scaffold-only Control resources and packs a new Control ZIP. Production tools may remove their own fresh extraction scratch; these targets are prechecked absent under the new attempt. Nothing is installed into the global interpreter or the user's app.

R4: current notices. The existing production `stage-notices.py` still insists on the historical external ggml repository commit. This task does not edit it. `notices` follows its collection pattern while taking the actual vendored `third_party/strata-ggml/LICENSE`, current Node licence, new Host dependency notices, fresh Control Python/distribution licences, current libstrix attribution and Qwen licence. Existing Microsoft documents are copied only after both inventory and reviewed vendor hashes match. This phase requires an explicit CUDA toolkit directory. No claim of a supply-chain audit is made by copying these bytes.

R5: fresh native signing/package. The controller provides a fresh unsigned native package and the binding receipt specified below. `verify.mjs native` checks both executables, exact native-build bytes, current native sources, actual vendored ggml identity, current source commit and the production verifier. `component` uses production composition with current verified local-backends/notices, copies into a fresh signing directory, passes only an external key path to the production signer, verifies using bundled trusted keys and full payload verification, checks that only the manifest changed, checks DLL coverage, packs a new engine tar.gz and stages the exact signed tree. It carries no historical numerical/model-test counts. Integrity verification is not a quality qualification.

R6: app NSIS. `app` requires the fresh native receipt and a clean source commit. It uses the integration's production Tauri build hook and release verifier, pinned Node, locked/offline Cargo, a fresh H: Cargo target, public assets, explicit native source/manifest identity and five compiler jobs. The standard NSIS installer is copied to the official-style filename with SHA-256 verification. It does not install it, sign it with Authenticode, or create the final delivery tar.7z.

All mutating phases require `--execute`. Plans do not inspect private keys or create the release directory. Each phase runs inside its own Windows Job, admitted with unchanged available RAM 3 GiB, commit headroom 4 GiB, H: free 10 GiB and C: free 5 GiB floors. The guard observes those floors during the phase and drains only its owned descendants. Five compiler jobs and one test job are explicit. No process memory cap is invented. Receipts are published only after successful observed child exit and zero remaining owned processes. Every attempt uses a fresh phase/rN directory; failures are retained, never retried automatically. Component signing suppresses all signer output and never reads private key contents itself.

## Later controller invocation sequence

Review and commit these helpers with the final source first: the directory is currently untracked, so a clean-source gate will intentionally refuse execution until the controller includes it in the source cut or otherwise deliberately freezes its own reviewed arrangement. Use PowerShell from the integration checkout. The placeholders below are required verified inputs, not claims that those files already exist.

```powershell
$py = 'C:/Users/downl/AppData/Local/Programs/Python/Python312/python.exe'
$helper = 'H:/Strata/worktrees/upstream-sync-20261116/_docs/stable-060-20261116/release.py'
$commit = '<final reviewed full 40-character commit>'
$pythonCache = 'H:/Strata/releases/stable-059-20261113-codex/app-059-control-r2/packaging-cache'
$wheels = '<existing verified flat wheelhouse for the current 25-package lock>'
$cuda = '<actual CUDA toolkit root with EULA.txt and LICENSE>'
$native = '<fresh root-built unsigned 0.6.0 native component folder>'
$binding = '<fresh controller binding receipt JSON>'
$keyPath = '<absolute existing signing-key path outside all source/release trees>'
uv run --offline --no-project --python $py python -B $helper freeze --reviewed-commit $commit
uv run --offline --no-project --python $py python -B $helper freeze --reviewed-commit $commit --execute
uv run --offline --no-project --python $py python -B $helper dependencies --reviewed-commit $commit --execute
uv run --offline --no-project --python $py python -B $helper control --reviewed-commit $commit --python-cache $pythonCache --wheelhouse $wheels --execute
uv run --offline --no-project --python $py python -B $helper notices --reviewed-commit $commit --cuda-dir $cuda --execute
uv run --offline --no-project --python $py python -B $helper component --reviewed-commit $commit --native-component $native --native-binding $binding --key $keyPath --execute
uv run --offline --no-project --python $py python -B $helper app --reviewed-commit $commit --native-binding $binding --execute
```

Default attempt is `r1`; use `--attempt r2` only after reviewing the failed receipt and fresh resource state. A failed phase may have filled managed integration resources. Those bytes are retained and the helper will refuse to replace them; the controller must inventory and preserve them before any new staging attempt. It must not reuse a failed receipt as success.

Expected output root is `H:/Strata/releases/stable-060-20261116-codex`. Each phase has its own `reservation.json`, `owned-job.json`, `pending.json` and final `receipt.json`. The Control phase carries Terminal archive, Control archive and source/package inventories; the component phase carries unsigned/signed component and archive assets; the app phase carries its fresh Cargo output and `Strata-0.6.0-windows-x64-setup.exe`. Published names/metadata, signing of the feed, previous DeepSeek feed-member handling, all README final publication wording and final tar.7z packaging remain controller responsibilities.

The native binding receipt contract is:

```json
{
  "sourceCommit": "<final reviewed full commit>",
  "sourceRoot": "H:/Strata/worktrees/upstream-sync-20261116",
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
