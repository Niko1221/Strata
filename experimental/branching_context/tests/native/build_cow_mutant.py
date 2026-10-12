"""Build isolated ownership/state mutants from an existing native build.

Only a copied translation unit and, when needed, a copied archive change.
The normal backend, oracle and their source files remain byte-for-byte unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native_root', type=Path)
    parser.add_argument('build', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--mutation', choices=('shared-write', 'shared-scales', 'stale-gdn', 'drop-leak', 'cold-gdn', 'prefix-copy'), default='shared-write')
    args = parser.parse_args()
    root, build, output = (p.resolve() for p in (args.native_root, args.build, args.output))
    output.mkdir(parents=True, exist_ok=False)
    mutations = {
        'shared-write': ('src/core/branch_pages.cpp',
                        'if (table_->pages[i].use_count()==1) {',
                        'if (true) { // TEST MUTANT: shared physical write'),
        'shared-scales': ('src/core/branch_pages.cpp',
                          'if (table_->pages[i].use_count()==1) {',
                          'if (table_->pages[i].use_count()==1 || bytes==64) { // TEST MUTANT: shared INT8 scales'),
        'stale-gdn': ('src/core/conversation_state.cpp',
                     '!copy(ss.gdn_state, c.gdn.data(), c.gdn.size(), error)',
                     'false /* TEST MUTANT: retain the previous lease GDN state */'),
        'drop-leak': ('src/core/branch_driver.cpp',
                      'if (command == "drop") images[index].reset();',
                      'if (command == "drop") { /* TEST MUTANT: retain the evicted image */ }'),
        'cold-gdn': ('src/core/branch_session.cpp',
                     'auto next_logits = frontier; // Allocate before touching private device state.',
                     'auto next_logits = frontier; // Allocate before touching private device state.\n'
                     '        std::fill(image.live.gdn.begin(), image.live.gdn.end(), uint8_t{0}); // TEST MUTANT: lose cold GDN'),
        'prefix-copy': ('src/core/branch_session.cpp',
                        'if (!b->memory.prepare_write(first*b->per_cell, (end-first)*b->per_cell)) {',
                        'if (!b->memory.prepare_write(0, end*b->per_cell)) { // TEST MUTANT: copy completed prefix'),
    }
    relative, old, replacement = mutations[args.mutation]
    original = root / relative
    archive = build / 'libstrata_engine.a'
    binary = build / 'strata-branch'
    protected = {str(p): sha(p) for p in (original, archive, binary)}
    record = dict(status='BUILDING', started=time.time(), commands=[],
                  protected_inputs=protected,
                  mutation=args.mutation)

    def run(command):
        start = time.time()
        result = subprocess.run(command, cwd=build, capture_output=True, text=True)
        row = dict(command=command, cwd=str(build), exit_code=result.returncode,
                   started=start, finished=time.time(), stdout=result.stdout, stderr=result.stderr)
        record['commands'].append(row)
        if result.returncode:
            raise RuntimeError(result.stderr)
        return result.stdout

    try:
        source = original.read_text()
        assert source.count(old) == 1
        mutant = output / original.name
        mutant.write_text(source.replace(old, replacement))
        record['source'] = dict(relative_path=relative, original_sha256=sha(original), mutant_sha256=sha(mutant))
        link = shlex.split((build / 'CMakeFiles/strata-branch.dir/link.txt').read_text())
        direct_object = 'CMakeFiles/strata-branch.dir/' + relative + '.o'
        direct = direct_object in link
        target = 'strata-branch' if direct else 'strata_engine'
        record['linkage'] = 'executable-object' if direct else 'engine-archive'
        if direct:
            assert link.count(direct_object) == 1
            protected[str(build / direct_object)] = sha(build / direct_object)
        flags = {}
        for line in (build / ('CMakeFiles/' + target + '.dir/flags.make')).read_text().splitlines():
            if line.startswith(('CXX_DEFINES =', 'CXX_INCLUDES =', 'CXX_FLAGS =')):
                name, value = line.split('=', 1)
                flags[name.strip()] = shlex.split(value)
        member = original.name + '.o'
        if not direct:
            members = run(['/usr/bin/ar', 't', str(archive)]).splitlines()
            assert members.count(member) == 1
        object_path = output / member
        run(['/usr/bin/c++'] + flags['CXX_DEFINES'] + flags['CXX_INCLUDES'] + flags['CXX_FLAGS']
            + ['-I', str(original.parent), '-o', str(object_path), '-c', str(mutant)])
        if direct:
            link[link.index(direct_object)] = str(object_path)
        else:
            mutant_archive = output / archive.name
            shutil.copy2(archive, mutant_archive)
            run(['/usr/bin/ar', 'r', str(mutant_archive), str(object_path)])
            run(['/usr/bin/ranlib', str(mutant_archive)])
            link = [str(mutant_archive) if token == archive.name else token for token in link]
            record['mutant_archive_sha256'] = sha(mutant_archive)
        link[link.index('-o') + 1] = str(output / ('strata-branch-' + args.mutation + '-mutant'))
        link = ['-Wl,--dependency-file=' + str(output / 'link.d')
                if token.startswith('-Wl,--dependency-file=') else token for token in link]
        run(link)
        assert {name: sha(name) for name in protected} == protected
        record.update(status='BUILT', binary=link[link.index('-o') + 1],
                      binary_sha256=sha(link[link.index('-o') + 1]),
                      mutant_object_sha256=sha(object_path))
    except BaseException as error:
        record.update(status='FAIL', error=repr(error))
        raise
    finally:
        record['finished'] = time.time()
        (output / 'build-receipt.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(status=record['status'], receipt=str(output / 'build-receipt.json'))))


if __name__ == '__main__':
    main()
