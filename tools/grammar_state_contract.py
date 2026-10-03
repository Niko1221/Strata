"""Bounded client-side example: grammar proposes; application state authorizes.

Only in-memory toy task state changes. No jobs, shell commands, network tools or
production authorization system are implemented. Strata owns neither this state
nor these effects. The application compares full snapshots, not fingerprints.
"""
from dataclasses import dataclass
import hashlib
import json
import re
from threading import Lock


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,32}', value):
        raise ValueError('task/principal IDs must use 1..32 ASCII letters, digits, underscores or hyphens')
    return value


def task_ids(values):
    if not isinstance(values, tuple) or len(values) > 32:
        raise ValueError('the example supports a tuple of at most 32 task IDs')
    for value in values:
        identifier(value)
    if len(set(values)) != len(values):
        raise ValueError('duplicate task ID')
    return tuple(sorted(values))


@dataclass(frozen=True)
class Snapshot:
    revision: int
    principal: str
    available: tuple[str, ...]

    def __post_init__(self):
        if type(self.revision) is not int or not 0 <= self.revision < 2**63:
            raise ValueError('invalid application revision')
        identifier(self.principal)
        if task_ids(self.available) != self.available:
            raise ValueError('snapshot tasks must be sorted')

    def fingerprint(self):
        raw = json.dumps([self.revision, self.principal, self.available], separators=(',', ':')).encode('ascii')
        return 'task-snapshot-v1-sha256:' + hashlib.sha256(raw).hexdigest()


def allowed_commands(snapshot):
    return ('WAIT',) + tuple('START ' + name for name in snapshot.available)


def derive_grammar(snapshot):
    # Bounded domain names cannot inject quotes, escapes or grammar rules. This
    # deliberately rejects arbitrary names rather than pretending to escape them.
    return 'root ::= ' + ' | '.join('"' + value + '"' for value in allowed_commands(snapshot)) + '\n'


def verify_candidate(snapshot, candidate):
    # Independent application verification, not the native grammar mechanism.
    # No trimming/repair; exact full strings only, under this frozen domain state.
    if not isinstance(candidate, str) or candidate not in allowed_commands(snapshot):
        raise ValueError('candidate is not an allowed complete command')
    return candidate


class StaleDecision(ValueError):
    pass


class TaskState:
    def __init__(self, tasks, grants):
        self._available = set(task_ids(tasks))
        self._grants = {}
        if not isinstance(grants, dict) or not 1 <= len(grants) <= 16:
            raise ValueError('the example supports 1..16 principals')
        for principal, allowed in grants.items():
            self._grants[identifier(principal)] = set(task_ids(allowed))
            if not self._grants[principal] <= self._available:
                raise ValueError('grant names an unknown task')
        self._tasks = frozenset(self._available)
        self._revision, self._lock = 0, Lock()

    def _snapshot(self, principal):
        identifier(principal)
        if principal not in self._grants:
            raise ValueError('unknown principal')
        return Snapshot(self._revision, principal, tuple(sorted(self._available & self._grants[principal])))

    def snapshot(self, principal):
        with self._lock:
            return self._snapshot(principal)

    def replace_grant(self, principal, allowed):
        # A toy external state change. In a real application only its authorized
        # administration path may call this; the model has no such callback.
        principal, allowed = identifier(principal), set(task_ids(allowed))
        with self._lock:
            if principal not in self._grants or not allowed <= self._tasks:
                raise ValueError('unknown principal or task')
            if self._revision == 2**63 - 1:
                raise ValueError('application revision exhausted')
            self._grants[principal] = allowed
            self._revision += 1

    def apply(self, principal, snapshot, candidate):
        # The caller supplies authenticated identity independently of model text.
        # Revision + current permission + transition share one application lock.
        with self._lock:
            current = self._snapshot(principal)
            if current != snapshot:
                raise StaleDecision('state or authorization changed; derive a new contract')
            verify_candidate(current, candidate)
            if candidate.startswith('START '):
                if self._revision == 2**63 - 1:
                    raise ValueError('application revision exhausted')
                self._available.remove(candidate[6:])
                self._revision += 1
            return self._snapshot(principal)


if __name__ == '__main__':
    state = TaskState(('task-a', 'task-b'), {'alice': ('task-a', 'task-b')})
    frozen = state.snapshot('alice')
    print('Synthetic candidate demonstration; no inference or external effects.')
    print('state.snapshot', frozen, frozen.fingerprint())
    print('state.derive_language\n' + derive_grammar(frozen), end='')
    print('state.apply', state.apply('alice', frozen, 'START task-a'))
    try:
        state.apply('alice', frozen, 'START task-b')
    except StaleDecision as exc:
        print('state.check_revision rejected:', exc)
