"""Measure both pinned engines resident together before calibrating a new layout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from context_time_travel.real import BranchingModel, FreshOracle
from calibration_inputs import execution_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw = args.profile.read_bytes()
    profile = json.loads(raw)
    assert profile['backend'] == 'strata'
    command = profile['execution']['sut_command']
    slots = int(command[command.index('--expert-cache') + 1])
    report = dict(status='RUNNING', pid=os.getpid(), started=time.time(),
                  scope='Dual-engine resource admission only; no numerical gate qualification',
                  input_profile_sha256=hashlib.sha256(raw).hexdigest(),
                  sut_command=command, oracle_command=profile['execution']['oracle_command'],
                  expert_cache_slots=slots, hardware=profile['hardware'],
                  model_weights_sha256=profile['model_weights_sha256'], gpu_samples=[])
    for key, name in [('CTT_NATIVE_TRACE', 'native.jsonl'), ('CTT_ORACLE_TRACE', 'oracle.jsonl')]:
        os.environ[key] = str(output / name)
    backend = oracle = None
    stop = threading.Event()
    sampling_errors = []

    def save():
        temporary = output / 'progress.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(output / 'progress.json')

    def sample():
        text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                        '--format=csv,noheader,nounits'], text=True, timeout=10)
        processes = []
        for line in text.splitlines():
            pid, memory = (part.strip() for part in line.split(',', 1))
            proc = Path('/proc', pid)
            try:
                parent = next(row.split(':', 1)[1].strip()
                              for row in (proc / 'status').read_text().splitlines() if row.startswith('PPid:'))
                processes.append(dict(pid=int(pid), parent=int(parent),
                                      executable=str((proc / 'exe').resolve(strict=True)), used_mib=int(memory)))
            except FileNotFoundError:
                continue  # The process finished between the driver and /proc reads.
        return dict(time=time.time(), processes=processes)

    def collect():
        try:
            while not stop.is_set():
                report['gpu_samples'].append(sample())
                stop.wait(0.5)
        except BaseException as error:
            sampling_errors.append(repr(error))

    try:
        save()
        if sample()['processes']:
            raise RuntimeError('Admission requires an otherwise idle GPU')
        backend = BranchingModel.load_backend(profile)
        oracle = FreshOracle(profile)
        with backend.lease() as lease:
            backend.consume(lease, (14,))
            image = backend.export(lease)
            frontier = backend.logits(lease)
        report['retained_image_audit'] = backend.audit(image)
        before = backend._refresh_stats()
        worker = threading.Thread(target=collect)
        worker.start()
        try:
            oracle.forward((15,))
        finally:
            stop.set()
            worker.join()
        if sampling_errors:
            raise RuntimeError('Memory instrumentation failed: ' + repr(sampling_errors))
        expected_oracle = str(Path(profile['execution']['oracle_command'][0]).resolve())
        simultaneous = [row for row in report['gpu_samples']
                        if any(p['pid'] == backend.process.pid for p in row['processes'])
                        and any(p['parent'] == os.getpid() and p['executable'] == expected_oracle
                                for p in row['processes'])]
        if not simultaneous:
            raise RuntimeError('No direct sample observed both owned engines resident')
        report['simultaneous_samples'] = len(simultaneous)
        report['sampled_max_combined_gpu_mib'] = max(sum(p['used_mib'] for p in row['processes'])
                                                   for row in simultaneous)
        with backend.lease() as lease:
            backend.restore(lease, image, history=(14,), frontier=frontier)
            assert backend.position(lease) == 0 and backend.logits(lease) == frontier
        after = backend._refresh_stats()
        assert after['forward_tokens'] == before['forward_tokens'] == 1
        assert oracle.last['computed_tokens'] == 1 and oracle.last['position_max'] == 0
        report.update(before_restore=before, after_restore=after, oracle_work=oracle.last)
        for name, executable in [('sut_binary_sha256', command[0]),
                                 ('oracle_binary_sha256', profile['execution']['oracle_command'][0])]:
            report[name] = hashlib.sha256(Path(executable).read_bytes()).hexdigest()
        backend.close()
        assert backend.process.returncode == 0
        report['status'] = 'PASS'
        settings = execution_settings(report)
        assert settings['kv_format'] == profile['execution']['kv_dtype']
        if settings['kv_format'] == 'INT8':
            assert {'k_scale', 'v_scale'} <= {r['kind'] for r in report['retained_image_audit']['regions']}
        assert args.profile.read_bytes() == raw
    except BaseException as error:
        report.update(status='FAIL', error=repr(error))
        raise
    finally:
        if oracle is not None:
            oracle.close()
        if backend is not None:
            backend.close()
        report['finished'] = time.time()
        save()
        print(json.dumps(dict(status=report['status'], receipt=str(output / 'progress.json'))), flush=True)


if __name__ == '__main__':
    main()
