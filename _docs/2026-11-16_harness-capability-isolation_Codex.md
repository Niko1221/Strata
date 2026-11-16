# Harness capability and explicit instructions-home isolation prerequisite

Prepared on the primary C: checkout at base
b24d25ead08be850851c42291daad25d9255c9d8. This is a source prerequisite for
bounded installed-app acceptance, not real-model acceptance, installation or
release evidence. No original app, original instruction contents, GPU, model,
external network, signing, publication, commit or push was used.

Disabling Files previously removed the preset's file tools but left the inherited
Harness working_directory tool registered. That tool could change cwd, after
which the ordinary instruction loader included the new directory's AGENTS.md in
the next request. The actual pinned Harness reproduced this with a synthetic
sibling directory and scripted loopback inference. Source profile generation now
adds a disabled override for the inherited tool only when Files is off. With
Files on, the generated profile is unchanged; the ordinary directory tool still
works. No permission preset, filesystem sandbox or security UI was added.

The Harness environment now carries only an explicitly named nonblank
DSH_AGENTS_HOME path from its base environment. Other DSH_* variables and ambient
credentials remain filtered, while the existing trusted extra environment keeps
its established precedence. Absent, empty and whitespace-only overrides preserve
the SDK's ordinary user-global .agents fallback. The isolated WebView environment
and packaged self-test environment create and select an owned .agents directory;
USERPROFILE remains available to Edge. This does not universally replace the
normal user's global instructions.

Normal-start identity now records the allowlisted DSH_AGENTS_HOME choice and the
effective agents root. Relative overrides resolve against the actual Harness
data/agent/run cwd, current-user tilde expands using the observed USERPROFILE,
nonblank whitespace is preserved, and blank values select USERPROFILE/.agents.
Original 0.5.9 capture uses its legacy filtering semantics after verifying the
observed host and shell version. It refuses a nonblank ambient override that
would select a different root in 0.6.0. An absent/blank override or an explicit
root equal to the legacy default preserves the selection. Future normal-start
verification compares the effective root as well as observed environment paths.
The stopped original process was not used to infer the current override state;
a fresh original-profile capture is still required.

The corrected actual Harness regression proves that Files off excludes
working_directory from model schemas and rejects a stale call without sending
the synthetic sibling instruction to the model. Stale read, write, run_code and
subagent requests also fail. An ordinary bounded MCP CSV read still succeeds.
With an explicit agents override, only the synthetic override instruction is
sent. Removing that override and enabling Files creates a fresh normal runtime:
working_directory works and the synthetic global instruction is restored. These
are real Harness/tool mechanics with synthetic model responses, not model
quality or OS filesystem confinement claims.

Validation uses the offline Python 3.12 runner and pinned private Node 24.22.3
through check.py, one test process at a time. real-model-prerequisite-r2 passed
all 49 Python tests, including five nested Node cases, in 59.35 seconds.
real-model-prerequisite-node-r1 passed all three focused profile/environment
Node cases without skips. The earlier prerequisite-r1 actual Harness cases
passed; one Python whitespace identity assertion failed because its special case
was in the wrong loop. That implementation error was corrected and its failed
receipt retained. The later legacy-version capture correction is validated in
real-model-prerequisite-delivery-r3: all 49 delivery cases passed in 35.22 seconds,
including the two new legacy/default root checks. Black formatted the touched
Python files at line length 110, and
git diff --check passed.

The successful r2 Job had containment before resume, native exit 0, confirmed
termination, zero remaining processes and zero dropped output bytes. Minimum
available RAM was 6,350,049,280 bytes, commit headroom 19,996,737,536 bytes and
H free 47,316,520,960 bytes. Last sampled C free was 33,237,594,112 bytes. The
unchanged floors are RAM 3 GiB, commit 4 GiB, H 10 GiB and C 5 GiB, with no
diagnostic commit cap. The test Job included existing delivery stress fixtures;
it does not claim a GPU protective stop or any real model resource measurement.

Prior boundary evidence is immutable under real-model-producer-v1 and
real-model-producer-scope-v2 in
H:/Strata/private-backups/cleanup-20261116. The prerequisite handoff is frozen
separately as real-model-prerequisite-v1 with exact before/after bytes, scoped
diff, source status, test/guard receipts, typed capability proof and SHA-256
manifest. Independent review is required before the parent commits this scope.
The full real-model.py/real-model.mjs producer and validate_real_model consumer
remain the next task after that prerequisite review; none is claimed complete.
