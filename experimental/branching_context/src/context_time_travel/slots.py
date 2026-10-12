"""FIFO branch-reference slots for the backend's single execution workspace."""
from collections import deque
from dataclasses import dataclass
import threading
import time

from .store import BranchHead, Conflict, InvalidOperation


@dataclass(frozen=True)
class Work:
    slot: int
    head: BranchHead
    kind: str
    argument: object
    operation_id: str


@dataclass(frozen=True)
class Result:
    head: BranchHead
    output: object


class BranchSlots:
    def __init__(self, model, count):
        if type(count) is not int or not 1 <= count <= 8:
            raise InvalidOperation('Slot count must be between one and eight')
        self.model = model
        self.branches = [None] * count
        self.pending = deque()
        self.active = None
        self.lock = threading.Lock()
        self.stats = dict(completed=0, failed=0, execution_seconds=0.0,
                          execution_mode='serialized')

    def _slot(self, slot):
        if type(slot) is not int or not 0 <= slot < len(self.branches):
            raise InvalidOperation('Invalid branch slot')

    def bind(self, slot, branch):
        self._slot(slot)
        self.model.store.head(branch)
        with self.lock:
            if any(index != slot and name == branch for index,name in enumerate(self.branches)):
                raise InvalidOperation('Branch is already bound to another slot')
            if (self.active is not None and self.active.slot == slot) or any(
                    task.slot == slot for task in self.pending):
                raise InvalidOperation('Cannot reassign a busy branch slot')
            self.branches[slot] = branch

    def submit(self, slot, kind, argument, operation_id):
        self._slot(slot)
        if kind not in ('feed', 'generate'):
            raise InvalidOperation('Unknown slot transition')
        if kind == 'feed':
            argument = tuple(argument)
        elif type(argument) is not int or argument < 0:
            raise InvalidOperation('Invalid generation bound')
        if not isinstance(operation_id, str) or not operation_id:
            raise InvalidOperation('A slot operation identity is required')
        with self.lock:
            if self.branches[slot] is None:
                raise InvalidOperation('Unbound branch slot')
            head = self.model.store.head(self.branches[slot])
            task = Work(slot, head, kind, argument, operation_id)
            self.pending.append(task)
            return task

    def cancel(self, task):
        """Remove only this queued job; active execution retains its owner."""
        with self.lock:
            if task is self.active:
                raise InvalidOperation('Cannot release an active execution job')
            for index, queued in enumerate(self.pending):
                if queued is task:
                    del self.pending[index]
                    return True
            return False

    def run_next(self):
        with self.lock:
            if self.active is not None:
                raise InvalidOperation('The serialized execution workspace is busy')
            if not self.pending:
                return None
            task = self.pending.popleft()
            self.active = task
        started = time.monotonic()
        try:
            if self.model.store.head(task.head.name) != task.head:
                raise Conflict('Queued branch revision changed')
            if task.kind == 'feed':
                target = self.model.feed(task.head.tip, task.argument)
                output = None
            else:
                output = self.model.generate(task.head.tip, task.argument)
                target = output.snapshot_id
            head = self.model.store.move(task.head.name, task.head.revision,
                                         target, task.operation_id)
            with self.lock:
                self.stats['completed'] += 1
            return Result(head, output)
        except BaseException:
            with self.lock:
                self.stats['failed'] += 1
            raise
        finally:
            with self.lock:
                self.active = None
                self.stats['execution_seconds'] += time.monotonic() - started
