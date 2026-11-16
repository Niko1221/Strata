"""Explicit, replayable hunk rulings for the frozen c775 intake; private output only."""
import hashlib
import json
from pathlib import Path
import re

OUT = Path('H:/Strata/private-backups/cleanup-20261116/upstream-c775-intake-v1')
if (OUT / 'applied.json').exists():
    raise RuntimeError('Preserve applied intake evidence; resolve a new preparation directory instead')
PATTERN = re.compile(r'^<<<<<<< fork\n(.*?)^\|\|\|\|\|\|\| base\n(.*?)^=======\n(.*?)^>>>>>>> official\n', re.M | re.S)
manifest = json.loads((OUT / 'preview.json').read_text(encoding='utf-8'))
decisions = {}


def replace(text, old, new):
    assert text.count(old) == 1, old[:100]
    return text.replace(old, new)


def ruling(name, i, ours, base, theirs):
    if name.startswith('README'):
        return ours, 'Upstream changes only its installer version; preserve the factual fork 0.5.9 delivery text pending separately measured publication.'
    if name in ('CMakeLists.txt', 'apps/desktop/package.json', 'apps/desktop/src-tauri/Cargo.lock', 'apps/desktop/src-tauri/Cargo.toml', 'apps/host/package.json', 'apps/host/package-lock.json', 'strata_cli/__init__.py'):
        return ours, 'Preserve independent fork 0.6.0 source version; upstream version is recorded separately.'
    if name == 'AGENTS.md':
        return ours.replace('Keep upstream Niko1221 0.2.14 at\nbaf16363b4403ef0ed4f538f5d5ed11c43fb9535', 'Keep upstream Niko1221 0.2.15 at\nc77572806996349b49995e4f98d41b5294d36a21'), 'Advance official reference while retaining all fork delivery evidence and resource requirements.'
    if name == '.github/workflows/snapshot.yml':
        return (ours if i == 0 else theirs), 'Keep Windows 45/Linux 30 minute limits; adopt upstream matrix timeout scale 2/1, equivalent to retained command limits.'
    if name == 'cmake/backends/cpu.cmake':
        return ours, 'Both sides implement the same bounded 8 MiB draft-policy test stack; keep existing comment and option.'
    if name in ('src/core/conversation_buffer_pool_test.cpp', 'src/core/conversation_segment_pool_test.cpp'):
        return theirs, 'Use upstream helper factoring with the same Windows aligned allocation/free pair; retain other merged fork assertions.'
    if name == 'src/kernels/cpu/card_node_test.cpp':
        return ours, 'Portable fake directory names retain all parser assertions on Windows instead of skipping them; BDF formatting remains separately tested.'
    if name in ('tests/terminal/test_tui2_023_versions.py', 'apps/host/tests/qa1-storage.test.mjs'):
        return ours, 'Retain stronger shared fork VERSION consistency assertions rather than hard-code upstream release numbers.'
    if name == 'apps/host/tests/packaged-control.test.mjs':
        return ours.replace('engineLogPollMs = 200', 'engineLogPollMs = 200, installPoll = null'), 'Combine isolated log-poll fixture control and incoming installer-poll fixture control.'
    if name == 'apps/host/src/lib/download-eta.ts':
        return ours.splitlines(keepends=True)[0] + theirs, 'Retain non-download phase exclusion and add checking-files own progress.'
    if name == 'apps/host/src/locales/engine-fr.json':
        value = json.loads('{'+theirs+'}')
        return '\n'.join('  '+json.dumps(k, ensure_ascii=False)+': '+json.dumps(v, ensure_ascii=False)+(',' if j<len(value)-1 else '') for j,(k,v) in enumerate(value.items()))+'\n', 'Take incoming French wording while retaining repository JSON indentation.'
    if name == 'docs/CAPABILITIES.md':
        return theirs, 'Official-reference capability header advances independently of fork versions.'
    if name == 'docs/INSTALL.md':
        return ours.replace('version 0.2.14', 'version 0.2.15'), 'Retain fork installation boundary and update official-reference version.'
    if name == 'docs/data/release.json':
        fork = json.loads('{'+ours+'}')
        official = json.loads('{'+theirs+'}')
        fork['upstream'] = {**official, 'source_commit': manifest['refs']['official']}
        return json.dumps(fork, ensure_ascii=False, indent=2)[2:-2]+'\n', 'Preserve independent fork candidate and historical assets; update nested upstream manifest including reused-engine provenance.'
    if name == 'packaging/README.md':
        return ours.replace('app and Terminal 0.2.14, Control and engines 0.1.60', 'app and Terminal 0.2.15, Control 0.1.61, engines 0.1.60'), 'Retain fork fresh-build, signing and packaging requirements; update official-reference component split.'
    if name == 'tools/capability_table.py':
        return ours, 'Existing formatter already separates differing Control/engine versions and additionally supports independent Terminal versions.'
    if name == 'tools/feed_model_install_gate.py':
        return ours[:ours.index('    if backend == "cuda"')] + theirs.replace('    elif "--resident-budget-gib" not in args:', '    elif backend == "cuda" and argument_value(args, "--resident-budget-gib") is None:'), 'Combine component-deferred resolution and no-draft/no-vision checks with Metal mmap validation; CUDA-only resident budget remains backend aware.'
    if name == 'tools/release_manifest.py':
        if i == 0:
            return ours.splitlines(keepends=True)[0]+''.join(theirs.splitlines(keepends=True)[1:]), 'Retain fork AGENTS reference pattern; adopt separate official Control/engine documentation fields.'
        return '    ("packaging/README.md", r"Upstream reference: app and Terminal " + V + r", Control " + V + r", engines " + V + r"\\.", ["product_version", "control_version", "engine_version"]),\n', 'Update fork-prefixed packaging reference pattern for separate official component versions.'
    if name == 'tools/test_release_manifest.py':
        return ours+theirs, 'Retain independent fork version test and add incoming reused-engine provenance negative cases.'
    if name == 'tools/test_qimage.py':
        return ours, 'Existing tests normalize to LF then CRLF and assert both variants; incoming normalization fix is already covered more strongly.'
    if name == 'tools/test_te_tokenizer.py':
        return ours.replace('write_text("\\n".join(lines) + "\\n")', 'write_text("\\n".join(lines) + "\\n", newline="\\n")'), 'Add incoming exact-LF cases writer while retaining both LF/CRLF tokenizer fixtures.'
    if name == 'tools/test_dit_rope.py':
        return (ours+'import platform\n' if i == 1 else ours), 'Retain strict seven-case independently pinned C++ oracle; incoming float64 reference is supplemental only and cannot replace exact acceptance.'
    if name == 'apps/host/server/index.mjs':
        if i == 0:
            return ours, 'Both sides prohibit automatic loading; preserve existing fork comment.'
        needle = '      const loadConfig={ beginner: store.get().beginner, expert: engine.params(), serving: store.get().serving };'
        if needle not in ours:
            raise ValueError('Inspect load route indentation before adding explicit backend admission')
        return ours.replace(needle, needle+"\n      if (!['PrismML', 'TurboQuant'].includes(localBackendChoice(loadConfig.expert))) {\n        return error(res, 409, 'Strata runs models on its own engine. Choose a verified PrismML or TurboQuant adapter explicitly for this imported model.');\n      }"), 'Reject generic imports before stopping any loaded model; preserve verified explicit adapter KV/preflight/capacity lifecycle.'
    if name == 'apps/host/server/supervisor.mjs':
        if i != 1:
            return ours, 'Retain explicit adapter lifecycle, RAM guard, private memory, props capacity and startup race ownership; generic discovery is removed separately.'
        return '''/* Explicit adapters only: never search environment, PATH, Homebrew or generic GGUF components. */
export function findBinary({ platform = process.platform, env = process.env, componentsDir = componentsDirectory({ env, platform }), backend = 'Automatic', installRoot } = {}) {
  if (!['PrismML', 'TurboQuant'].includes(backend)) return null;
  return findLocalBackend(backend, { componentsDir, platform, installRoot });
}

export async function resolveBinary({ updateState = true, backend = 'Automatic' } = {}) {
  const found = findBinary({ backend });
  if (found && updateState) {
    state.binary = found.path;
    state.binarySource = found.source;
  }
  return found;
}

export async function probeCapabilities({ backend = 'Automatic', manifestPath = process.env.STRATA_BONSAI_MANIFEST ?? null } = {}) {
  const binary = await resolveBinary({ updateState: false, backend });
  if (binary) await verifyLocalBackend(binary, backend);
  return probeBackendCapabilities(binary?.path, { manifestPath });
}

''', 'Replace automatic/env/PATH/Homebrew/generic-component discovery with an explicit PrismML/TurboQuant allowlist.'
    if name == 'apps/host/server/native-control.mjs':
        if i == 0:
            return '\n'.join(theirs.splitlines()[:2])+'\n', 'Add incoming offline-drafter note; preserve fork shared vision-flag helper import.'
        if i == 1:
            return replace(theirs, "packaged&&args.vision===undefined?", "packaged&&args.vision===undefined&&args.family!=='deepseek41'?"), 'Add explicit MTP off while retaining DeepSeek-specific vision default exclusion.'
        if i == 2:
            return theirs.splitlines(keepends=True)[0]+''.join(ours.splitlines(keepends=True)[1:]), 'Add offline-note clearing after drafter recovery; preserve trusted build-config parameters and canonical runtime executable binding.'
        return ours, 'Retain stronger read-current-and-merge persistence, which preserves concurrent drafting and KV choices without overwriting user edits.'
    raise ValueError('Unreviewed conflict '+name)


for name, row in manifest['rows'].items():
    text = (OUT / 'merge' / name / 'candidate').read_text(encoding='utf-8')
    notes = []
    def resolve_match(match):
        output, reason = ruling(name, len(notes), *match.groups())
        notes.append({'hunk': len(notes), 'reason': reason})
        return output
    text, count = PATTERN.subn(resolve_match, text)
    assert count == row['conflicts'], (name, count, row['conflicts'])
    if name == 'apps/host/server/supervisor.mjs':
        text = replace(text, ' * Engine supervisor (what is left of it): the engine status the Monitor and the engine page read, its log ring buffer, the parser of the\n * engine\'s Prometheus metrics, and the translation of the tuning configuration into flags for the command preview.', ' * Engine supervisor: explicit adapter process ownership, status, log ring buffer,\n * Prometheus metrics and tuning configuration translated into command flags.')
        text = replace(text, '  throw new Error(state.error);\n}', "  log('supervisor', 'engine ready');\n  return status();\n}")
        text = replace(text, " * It starts nothing: the llama.cpp `llama-server` it used to find and start is gone (see start() below). Models run on the Strata engine.", " * Models normally run on the Strata engine. Only explicitly selected, verified PrismML/TurboQuant adapters use this lifecycle.")
        text = replace(text, "import { existsSync, readFileSync, realpathSync, readdirSync, statSync } from 'node:fs';", "import { existsSync, statSync } from 'node:fs';")
        text = replace(text, "import { dirname, join, resolve } from 'node:path';", "import { join, resolve } from 'node:path';")
        text = replace(text, "import { which } from './util.mjs';\n", '')
        text = replace(text, '  const backend=localBackendChoice(config?.expert);', "  const backend=localBackendChoice(config?.expert);\n  if(!['PrismML','TurboQuant'].includes(backend))throw new Error('Strata runs models on its own engine. Select a verified PrismML or TurboQuant adapter explicitly.');")
        text = replace(text, "  const runtime=backend==='Automatic'?null:await verifyLocalBackend(binary,backend);\n  if(runtime)requireRuntimeKv(runtime,kv);\n  else if([kv.k,kv.v].some(type=>type.startsWith('turbo')))throw new Error('Choose the verified TurboQuant llama-server backend before using a TurboQuant K/V cache.');", '  const runtime=await verifyLocalBackend(binary,backend);\n  requireRuntimeKv(runtime,kv);')
    if name == 'apps/host/server/index.mjs':
        text = replace(text, '#!/usr/bin/env node\n', "#!/usr/bin/env node\nimport { localBackendChoice } from './local-backends.mjs';\n")
    if name == 'tools/test_dit_rope.py':
        text = replace(text, '    @unittest.skipUnless(GLIBC, "the recorded sha256 are sd.cpp\'s tables on glibc, and this C library rounds cosf / sinf "\n                                "differently in the last bit (test_tables_close_to_float64 checks every float here)")\n', '')
        text = replace(text, '    exe = td / "strata-dit-rope"', '    exe = td / ("strata-dit-rope.exe" if os.name == "nt" else "strata-dit-rope")')
    if name == 'strata_control/runtime.py':
        text = replace(text, '    if pending_ok and stat.S_ISREG(details.st_mode) and details.st_size == 0:\n        return None\n', '')
        text = replace(text, '        raise ProfileError("runtime state file is not private or is invalid", 5)\n', '        raise ProfileError("runtime state file is not private or is invalid", 5)\n    if pending_ok and details.st_size == 0:\n        return None\n')
        notes.append({'hunk': 'clean-boundary', 'reason': 'Wait for an empty owned startup lease only after normal regular-file/private/size validation; empty files never bypass privacy admission.'})
    if name == 'src/kernels/cpu/card_node_test.cpp':
        text = replace(text, '    expect(cpu::card_node_pci_node(root.string(), "0000:99:00.0") == -1, "pci device missing: -1");\n#endif', '    expect(cpu::card_node_pci_node(root.string(), "0000:99:00.0") == -1, "pci device missing: -1");')
    target = OUT / 'resolved' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = text.encode('utf-8')
    target.write_bytes(raw)
    decisions[name] = {'reason': '; '.join(x['reason'] for x in notes) or 'Clean diff3 merge of pinned inputs; fork boundary inspection recorded in intake report.', 'hunks': notes, 'sha256': hashlib.sha256(raw).hexdigest()}
(OUT / 'decisions.json').write_text(json.dumps(decisions, indent=2), encoding='utf-8')
print(json.dumps({'resolved': len(decisions), 'conflicted': sum(bool(x['hunks']) for x in decisions.values())}))
