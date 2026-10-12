"""Rebuild every branch-page ABI consumer in a separate, measured CUDA stage.

The base build is read-only. Its actual CMake flags/link lines are reused; the
source overlay and starting engine archive are explicit, hashed inputs.
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
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base_build', type=Path)
    parser.add_argument('base_engine', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    base, engine, source, output = (p.resolve() for p in
                                   (args.base_build, args.base_engine, args.source, args.output))
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='BUILDING', commands=[], source_sha256={
        str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*')) if p.is_file()},
        protected_inputs={}, binaries={})
    receipt = output / 'build-receipt.json'

    def save():
        receipt.write_text(json.dumps(report, indent=2) + '\n')

    def protect(path):
        path = path.resolve()
        if str(path) not in report['protected_inputs']:
            report['protected_inputs'][str(path)] = sha(path)

    def run(command):
        start = time.time()
        result = subprocess.run([str(v) for v in command], cwd=base,
                                text=True, capture_output=True)
        report['commands'].append(dict(command=[str(v) for v in command], cwd=str(base),
                                       started=start, finished=time.time(),
                                       exit_code=result.returncode,
                                       stdout=result.stdout, stderr=result.stderr))
        save()
        if result.returncode:
            raise RuntimeError(f'command {len(report["commands"])} failed: {result.stderr}')

    def flags(target, language):
        path = base / f'CMakeFiles/{target}.dir/flags.make'
        protect(path)
        values = {}
        for line in path.read_text().splitlines():
            if ' = ' in line:
                key, value = line.split(' = ', 1)
                values[key] = shlex.split(value)
        result = ['-I', str(source / 'include')]
        for kind in ('DEFINES', 'INCLUDES', 'FLAGS'):
            result += values[f'{language}_{kind}']
        for i, item in enumerate(result):
            if item == '--options-file':
                protect(base / result[i + 1])
        return result

    def compile_file(relative, target='strata_engine'):
        language = 'CUDA' if relative.endswith('.cu') else 'CXX'
        compiler = '/usr/local/cuda/bin/nvcc' if language == 'CUDA' else '/usr/bin/c++'
        obj = output / (Path(relative).name + '.o')
        run([compiler, *flags(target, language), '-o', obj, '-c', source / relative])
        return obj

    def link(name, objects, template='branch_pages_test'):
        path = base / f'CMakeFiles/{template}.dir/link.txt'
        protect(path)
        original = shlex.split(path.read_text())
        command, inserted = [], False
        for item in original:
            if item.endswith('.o'):
                if not inserted:
                    command += [str(p) for p in objects]
                    inserted = True
            elif item.startswith('-Wl,--dependency-file='):
                command.append(f'-Wl,--dependency-file={output / (name + ".link.d")}')
            elif item == template:
                command.append(str(output / name))
            elif item == 'libstrata_engine.a':
                command.append(str(output / 'libstrata_engine.a'))
            else:
                command.append(item)
                if item.endswith(('.a', '.so')) and (base / item).is_file():
                    protect(base / item)
        if not inserted:
            raise ValueError('link template had no objects')
        run(command)
        report['binaries'][name] = dict(path=str(output / name), sha256=sha(output / name))

    try:
        protect(engine)
        protect(base / 'libstrata_engine.a')
        protect(base / 'strata-branch')
        shutil.copyfile(engine, output / 'libstrata_engine.a')
        objects = [compile_file('src/core/' + name) for name in
                   ('branch_digest.cu', 'branch_pages.cpp', 'branch_session.cpp')]
        run(['/usr/bin/ar', 'r', output / 'libstrata_engine.a', *objects])
        run(['/usr/bin/ranlib', output / 'libstrata_engine.a'])
        driver = compile_file('src/core/branch_driver.cpp', 'strata-branch')
        generate = base / 'CMakeFiles/strata-branch.dir/src/program/generate.cpp.o'
        protect(generate)
        link('strata-branch', [generate, driver], 'strata-branch')
        for name in ('branch_pages_test', 'branch_pages_stress', 'branch_pages_readonly',
                     'branch_pages_graph', 'branch_session_test', 'branch_audit_test',
                     'branch_lifecycle_test',
                     'branch_integrity_test', 'branch_digest_gpu', 'branch_pressure_test',
                     'branch_pressure_helper'):
            ext = '.cu' if name in ('branch_pages_readonly', 'branch_pages_graph') else '.cpp'
            target = name if (base / f'CMakeFiles/{name}.dir/flags.make').exists() else 'branch_pages_test'
            obj = compile_file('tests/core/' + name + ext, target)
            link(name, [obj], target)
        for name in ('branch_digest_host', 'branch_handles_test'):
            run(['/usr/bin/c++', '-O3', '-std=c++20', '-I', source / 'include',
                 source / ('tests/core/' + name + '.cpp'), '-o', output / name])
            report['binaries'][name] = dict(path=str(output / name), sha256=sha(output / name))
        report['replacement_archive_sha256'] = sha(output / 'libstrata_engine.a')
        report['status'] = 'COMPILED_NOT_EXECUTED'
    except Exception as error:
        report['status'] = 'BUILD_FAILED'
        report['error'] = repr(error)
        raise
    finally:
        changed = [p for p, before in report['protected_inputs'].items() if sha(p) != before]
        report['changed_protected_inputs'] = changed
        if changed:
            report['status'] = 'INPUT_CHANGED'
        save()
        if changed:
            raise RuntimeError(f'Protected build input changed: {changed}')
    print(json.dumps(dict(status=report['status'], commands=len(report['commands']),
                          binaries=report['binaries'], receipt_sha256=sha(receipt))))


if __name__ == '__main__':
    main()
