"""Actual host export failure in the process that owns the backend allocation."""
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from context_time_travel.real import BranchingModel
from allocation_probe import HostAllocationProbe
from recovery_observation import record


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    raw = Path(os.environ['CTT_PROFILE']).read_bytes()
    original = json.loads(raw)
    profile = copy.deepcopy(original)
    probe = None
    binding = None
    if profile.get('backend') == 'strata':
        directory = Path(os.environ.get('CTT_RECOVERY_OBSERVATIONS',
                                        'evidence/runs/recovery-observations.jsonl')).resolve().parent
        directory.mkdir(parents=True, exist_ok=True)
        artifact = Path(tempfile.mkdtemp(prefix='native-allocation-', dir=directory))
        probe = HostAllocationProbe(artifact, profile['execution']['sut_command'])
        # Explicit fault-instrumented execution identity. The model contract,
        # native executable, native arguments, oracle and tolerances stay fixed.
        profile['execution']['sut_command'] = probe.command
        profile['native_build']['sha256'].update(probe.fingerprints)
        profile['execution_profile_id'] = digest(dict(build=profile['native_build'],
            execution=profile['execution'], hardware=profile['hardware']))
        profile.pop('gate', None)
        profile.update(status='FAULT_INSTRUMENTATION_ONLY', numerical_gate_qualified=False)
        location = artifact / 'fault-execution-profile.json'
        location.write_text(json.dumps(profile, indent=2) + '\n')
        binding = dict(parent_profile_sha256=hashlib.sha256(raw).hexdigest(),
            parent_execution_profile_id=original['execution_profile_id'],
            fault_profile_path=str(location), fault_profile_sha256=hashlib.sha256(location.read_bytes()).hexdigest(),
            fault_execution_profile_id=profile['execution_profile_id'],
            scope='Syscall-traced fault probe only; no independent numerical qualification or cache compatibility transfer')

    backend = BranchingModel.load_backend(profile)
    old = resource.getrlimit(resource.RLIMIT_AS)
    allocation_bytes = []
    before = None
    after = None
    failed = False
    try:
        sut = BranchingModel(sys.argv[1], backend, profile)
        if probe:
            probe.bind(backend.process)
            # First establish the retained root with unlimited native memory.
            # Inject at the child's export after the suffix was really consumed.
            sut.next_logits(sys.argv[2])
            root_length = len(sut.history(sys.argv[2]))

        def exhaust(point, lease):
            nonlocal before
            if point != 'before_native_export':
                return
            if probe:
                before = backend._refresh_stats()
                if before['consumed'] != root_length + 1:
                    raise RuntimeError('Allocation fault did not follow the consumed suffix')
                probe.arm()
            else:
                allocation_bytes.append(int(backend.lib.ctt_state_size(lease)))
                ctypes.CDLL(None).malloc_trim(0)
                vm = int(Path('/proc/self/statm').read_text().split()[0]) * os.sysconf('SC_PAGE_SIZE')
                resource.setrlimit(resource.RLIMIT_AS, (vm + 1024 * 1024, old[1]))

        sut._allocation_fault = exhaust
        try:
            sut.feed(sys.argv[2], profile['fixtures']['suffix_a'][:1])
        except MemoryError:
            failed = True
        finally:
            if probe:
                probe.restore_limit()
            else:
                resource.setrlimit(resource.RLIMIT_AS, old)
        if probe:
            actual_failures = probe.allocation_failures()
            allocation_bytes = [row['requested_bytes'] for row in actual_failures]
            if failed and not actual_failures:
                raise RuntimeError('Native MemoryError has no observed failed OS allocation')
            after = backend._refresh_stats()
            if failed and (before is None or after['captures'] != before['captures']):
                raise RuntimeError('Failed export published a native image')
    finally:
        try:
            if probe:
                probe.restore_limit()
            else:
                resource.setrlimit(resource.RLIMIT_AS, old)
        finally:
            backend.close()
    observation = dict(allocation_failed=failed, allocation_bytes=allocation_bytes,
        counters=dict(backend.stats),
        scope='Actual host export allocation failure; no GPU-OOM or system-RAM exhaustion claim')
    if probe:
        receipt = probe.receipt()
        if not receipt['native_process_exited'] or receipt['tracer_returncode'] != 0:
            raise RuntimeError('Fault-probe native process did not close cleanly')
        observation.update(native_probe=receipt, execution_binding=binding,
            native_before_failure=before, native_after_failure=after,
            allocation_bytes_scope='Actual failed mmap/mremap request lengths including OS/allocator rounding')
    record('allocation-failure', **observation)
    print(json.dumps(observation))


if __name__ == '__main__':
    main()
