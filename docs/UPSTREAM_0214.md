# Official 0.2.14 integration in the Hakua fork

The 0.6.0 source candidate integrates official Strata through
[baf16363b4403ef0ed4f538f5d5ed11c43fb9535](https://github.com/Niko1221/Strata/commit/baf16363b4403ef0ed4f538f5d5ed11c43fb9535).
The official release is [v0.2.14](https://github.com/Niko1221/Strata/releases/tag/v0.2.14),
published on 2026-11-16 JST. The same release later received the Terminal
0.2.14.1 / Control 0.1.60.1 drafter fixes. App 0.2.14 and engine 0.1.60
remain independent from these and from this fork's 0.6.0 numbering. Some official platform
components retain 0.1.59; the complete API asset inventory is preserved in the
integration evidence. A source version does not prove a package was built,
installed or published.

The fork input is 6f13c7555a7d50289f0e3587ab780627c65a8000. Its merge base with
official source is f33c8c30588ff00e24b6b63bad0bf6a14da19081. The frozen intake
initially contained 900 official-only commits and 447 fork-only commits, followed
by seven additional official fixes before the source cut. Python scripts
orchestrate three-way merges from immutable blobs. Line-ending normalization
prevents CRLF from turning a small semantic conflict into a whole-file choice.

## What changes

The intake includes official networking timeouts/retries, drafting selection and
missing-drafter recovery, prompt kernels, owned memory handling and platform
fixes. It also vendors ggml in the source tree. Ordinary builds no longer need
to download a separate llama.cpp checkout. The official default omits the MMQ
prompt path; `STRATA_MMQ=ON` retains that optional path. Local integration builds
also compile it to check the retained fork behavior. No performance improvement
is asserted from source changes or compilation.

Hakua retains side-question subagents, research, workflow learning/application,
private skills and memory, workspace/sidebar operations and signed fork update
trust. Native bounded expert memory, pack integrity, supported Spin8/cache
options and P0 observations remain present. The actual feature and regression
results are recorded separately from the source merge.

## DeepSeek compatibility

Official retirement of DeepSeek paths is not applied as deletion of the fork's
compatibility extension. Protocol, catalog, setup and native source support are
retained with adapters around the new official interfaces. The Models screen
keeps the family visible with an unavailable reason where appropriate.

Loading requires the selected configuration's actual dedicated engine file,
checked before effects and again before dispatch. A generic Qwen engine cannot
establish DeepSeek availability. File presence still leaves ABI and execution
compatibility unverified. The Windows fork does not distribute or qualify a
native DeepSeek engine. A Qwen GGUF with DeepSeek in its filename remains Qwen
when its architecture metadata says so.

## Dependency updates and limits

The Windows Python runtime lock was queried for all 25 dependencies, with no
skipped queries and no known vulnerabilities returned at the audit time.
The Host npm lock moves Vite 6.4.3 to 6.4.6; the isolated updated dependency tree
passed type and production-build checks. The desktop npm audit returned zero
advisories for its 12 dependencies. These are dated advisory results, not proof
that no vulnerability exists.

The private Node runtime and all platform archive pins move from 24.22.2 to
24.22.3, following the [official security patch release](https://nodejs.org/en/blog/release/v24.22.3/)
and [November security notice](https://nodejs.org/en/blog/vulnerability/november-2026-security-releases/).
The Windows archive and executable match the published SHA-256 values. Fresh
Host regression, type and sealed-build checks passed using that runtime.
PGP verification was not performed; the existing installed runtime was not
modified during source validation.

The Cargo lock audit queried 525 registry packages and found 16 advisory IDs,
including aliases and maintenance notices. Targeted updates move the Tauri
updater to 2.14.1 and ZIP dependency to 9.6.0, rustix to 1.1.6, rustls-webpki
to 0.103.16 and webpki-root-certs to 1.0.10. The new 528-package audit leaves six
IDs covering GLib and maintenance notices. The Windows target graph contains
the repaired ZIP/certificate dependency versions and excludes those remaining
flagged packages. The inherited Linux GLib issue and unmaintained GTK-related
dependencies are not silently waived or presented as fixed. No new Linux app
distribution is qualified here.

Primary references: [Tauri updater dependency declaration](https://github.com/tauri-apps/plugins-workspace/blob/v2/plugins/updater/Cargo.toml),
[RustSec certificate verification advisory](https://rustsec.org/advisories/RUSTSEC-2026-0382.html),
[RustSec rustix advisory](https://rustsec.org/advisories/RUSTSEC-2026-0486.html),
[RustSec ZIP advisory](https://rustsec.org/advisories/RUSTSEC-2026-0368.html),
[OSV batch API](https://google.github.io/osv.dev/post-v1-querybatch/).

## Validation and delivery

See the [integration record](../_docs/2026-11-16_upstream-integration_Codex.md),
[Python compatibility record](../_docs/2026-11-16_upstream-python-compatibility_Codex.md)
and [Host compatibility record](../_docs/2026-11-16_upstream-host-compatibility_Codex.md).
Current installer/installation status belongs in [the fork release record](HAKUA_RELEASE.md).
The [0.2.13 record](UPSTREAM_0213.md) keeps its original historical sources.

RAM, commit, GPU reserve and disk floors remain unchanged. Five compiler jobs
are the maximum on this PC. General model quality, full-window inference and
GPU concurrency remain unqualified; `releaseQualified=false`. The installer
has no Windows Authenticode signature; component/feed signatures are separate.
