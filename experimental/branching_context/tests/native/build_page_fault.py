"""Add a real byte-fault injector to a separate test-only native executable."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('cmake_build', type=Path)
    parser.add_argument('engine_archive', type=Path)
    parser.add_argument('normal_binary', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source, build, archive, normal, output = (p.resolve() for p in vars(args).values())
    output.mkdir(parents=True, exist_ok=False)
    original = source / 'src/core/branch_driver.cpp'
    injector = Path(__file__).with_name('branch_page_fault.inc')
    protected = {str(p): sha(p) for p in (original, archive, normal, injector)}
    report = dict(status='BUILDING', commands=[], protected_inputs=protected,
                  purpose='actual CUDA byte damage; production logic otherwise unchanged')
    receipt = output / 'build-receipt.json'

    def run(command):
        started = time.time()
        result = subprocess.run([str(v) for v in command], cwd=build, capture_output=True, text=True)
        report['commands'].append(dict(command=[str(v) for v in command], cwd=str(build),
            started=started, finished=time.time(), exit_code=result.returncode,
            stdout=result.stdout, stderr=result.stderr))
        if result.returncode:
            raise RuntimeError(result.stderr)

    try:
        text = original.read_text()
        anchor = '        error.clear();\n'
        assert text.count(anchor) == 1
        altered = output / 'branch_driver.cpp'
        altered.write_text('#include <cuda_runtime.h>\n' + text.replace(anchor, anchor + injector.read_text() + '\n'))
        report['injected_source_sha256'] = sha(altered)
        flags = {}
        for line in (build / 'CMakeFiles/strata-branch.dir/flags.make').read_text().splitlines():
            if line.startswith(('CXX_DEFINES =', 'CXX_INCLUDES =', 'CXX_FLAGS =')):
                key, value = line.split('=', 1)
                flags[key.strip()] = shlex.split(value)
        obj = output / 'branch_driver.cpp.o'
        run(['/usr/bin/c++', '-I', source / 'include', *flags['CXX_DEFINES'],
             *flags['CXX_INCLUDES'], *flags['CXX_FLAGS'], '-o', obj, '-c', altered])
        link = shlex.split((build / 'CMakeFiles/strata-branch.dir/link.txt').read_text())
        old = 'CMakeFiles/strata-branch.dir/src/core/branch_driver.cpp.o'
        assert link.count(old) == 1
        link[link.index(old)] = str(obj)
        link = [str(archive) if item == 'libstrata_engine.a' else
                f'-Wl,--dependency-file={output / "link.d"}' if item.startswith('-Wl,--dependency-file=')
                else item for item in link]
        binary = output / 'strata-branch-page-fault'
        link[link.index('-o') + 1] = str(binary)
        run(link)
        assert {p: sha(p) for p in protected} == protected
        report.update(status='COMPILED_NOT_EXECUTED', binary=str(binary), binary_sha256=sha(binary))
    except BaseException as error:
        report.update(status='BUILD_FAILED', error=repr(error))
        raise
    finally:
        receipt.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], binary_sha256=report['binary_sha256'],
                          receipt_sha256=sha(receipt))))


if __name__ == '__main__':
    main()
