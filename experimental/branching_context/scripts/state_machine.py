"""Reproducible action programs and failure-signature-preserving reduction.

run_program uses the pinned real backend and a separate fresh-process oracle.
Helper tests alone do not execute inference or qualify a gate.
"""
import random
import hashlib
import json
from pathlib import Path
import struct
import time


OPERATIONS = ('append', 'fork', 'reseed', 'generate', 'rewind', 'checkpoint',
              'evict', 'evaluate', 'retry', 'release', 'gc')


class LogicalHistory:
    """Test oracle: tuples and a graph, with no Store/backend imports or files."""
    def __init__(self):
        self.nodes = {}
        self.heads = {}
        self.receipts = {}
        self.pins = {}
        self.released = set()

    def add(self, identity, parent, tokens):
        if parent is not None and parent not in self.nodes:
            raise KeyError(parent)
        node = (parent, tuple(tokens))
        if identity in self.nodes and self.nodes[identity] != node:
            raise ValueError('Immutable identity changed')
        if identity == parent:
            raise ValueError('Self-parent cycle')
        self.nodes[identity] = node

    def history(self, identity):
        segments = []
        while identity is not None:
            identity, tokens = self.nodes[identity]
            segments.append(tokens)
        return tuple(token for segment in reversed(segments) for token in segment)

    def root(self, identity):
        while self.nodes[identity][0] is not None:
            identity = self.nodes[identity][0]
        return identity

    def fork(self, name, identity):
        self.nodes[identity]
        if name in self.heads:
            raise ValueError('Duplicate branch')
        self.heads[name] = (identity, 0)
        return self.heads[name]

    def move(self, name, expected, identity, operation):
        request = (name, expected, identity)
        if operation in self.receipts:
            previous, result = self.receipts[operation]
            if previous != request:
                raise ValueError('Operation payload changed')
            return result
        current, revision = self.heads[name]
        if revision != expected or self.root(current) != self.root(identity):
            raise ValueError('Stale revision or foreign root')
        result = (identity, revision+1)
        self.heads[name] = result
        self.receipts[operation] = (request, result)
        return result

    def release(self, identity):
        self.nodes[identity]
        self.released.add(identity)

    def protect(self, name, identity):
        self.nodes[identity]
        self.pins[name] = identity

    def unprotect(self, name):
        del self.pins[name]

    def collect(self, root):
        if self.root(root) != root:
            raise ValueError('Collection needs a root')
        marked = set(self.nodes)-self.released
        marked.add(root)
        marked.update(self.pins.values())
        marked.update(identity for identity, revision in self.heads.values())
        marked.update(result[0] for request, result in self.receipts.values())
        pending = list(marked)
        while pending:
            parent = self.nodes[pending.pop()][0]
            if parent is not None and parent not in marked:
                marked.add(parent)
                pending.append(parent)
        deleted = {identity for identity in self.nodes if identity not in marked and self.root(identity) == root}
        for identity in deleted:
            del self.nodes[identity]
        self.released.difference_update(deleted)
        return deleted


def make_program(seed, steps, token_ids):
    if type(seed) is not int or type(steps) is not int or steps < 0:
        raise ValueError('Integer seed and nonnegative step count required')
    tokens = tuple(token_ids)
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError('Nonempty nonnegative token fixture required')
    rng = random.Random(seed)
    known = ['root']
    program = []
    for index in range(steps):
        output = 's' + str(index)
        program.append(dict(op=rng.choice(OPERATIONS), source=rng.choice(known),
                            out=output, token=rng.choice(tokens), seed=rng.getrandbits(64)))
        known.append(output)
    return program


def minimize(program, reproduce):
    """Return a deletion-minimal subsequence preserving the original failure.

    reproduce returns a stable failure signature, or None for a passing program.
    Invalid reduced programs must return a distinct signature or None. Unexpected
    harness errors propagate; they must never be counted as the original failure.
    This finds a local deletion minimum, not a globally shortest program.
    """
    candidate = list(program)
    signature = reproduce(candidate)
    if signature is None:
        raise ValueError('Cannot shrink a passing program')
    partitions = 2
    while candidate:
        width = max(1, (len(candidate)+partitions-1)//partitions)
        reduced = False
        for start in range(0, len(candidate), width):
            trial = candidate[:start] + candidate[start+width:]
            if reproduce(trial) == signature:
                candidate = trial
                partitions = max(2, partitions-1)
                reduced = True
                break
        if reduced:
            continue
        if width == 1:
            break
        partitions = min(len(candidate), partitions*2)
    return candidate


class ProgramFailure(AssertionError):
    pass


def run_program(profile, directory, program, *, seed):
    """Execute real operations and save measurements; never substitute a backend."""
    from context_time_travel.real import BranchingModel, FreshOracle
    from context_time_travel.numerics import compare_logits
    from context_time_travel.store import NotFound
    from context_time_travel.evaluation import evaluate

    if profile.get('calibration_status') != 'FROZEN':
        raise ValueError('Frozen numerical profile required')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = dict(schema=1, seed=seed, program=program, status='RUNNING', completed_operations=0,
                  executed_operations=[], boundaries=[], max_absolute_error=0.0,
                  model_contract_id=profile['model_contract_id'],
                  execution_profile_id=profile['execution_profile_id'])
    model = LogicalHistory()
    aliases, immutable = {}, {}
    backend = oracle = sut = None
    started = time.monotonic()

    def require(condition, signature):
        if not condition: raise ProgramFailure(signature)

    def remember(identity, parent, inputs):
        model.add(identity, parent, inputs)
        immutable.setdefault(identity, sut.snapshot_bytes(identity))

    def check(identity):
        history = model.history(identity)
        require(sut.history(identity) == history, 'canonical-history')
        require(sut.consumed(identity) == len(history), 'consumed-frontier')
        actual, expected = sut.next_logits(identity), oracle.forward(history)
        result = compare_logits(actual, expected, atol=profile['atol'], rtol=profile['rtol'])
        report['max_absolute_error'] = max(report['max_absolute_error'], result.max_absolute_error)
        report['boundaries'].append(dict(snapshot=identity, tokens=list(history),
                                         max_absolute_error=result.max_absolute_error))
        if not result.pass_elementwise or not result.top1_stable_within_tolerance:
            for name, values in [('actual', actual), ('expected', expected)]:
                (directory/(name+'.f32')).write_bytes(struct.pack('<'+str(len(values))+'f', *values))
            raise ProgramFailure('next-logits')
        for retained in model.nodes:
            require(sut.snapshot_bytes(retained) == immutable[retained], 'snapshot-mutated')
        for name, (tip, revision) in model.heads.items():
            head = sut.store.head(name)
            require((head.tip, head.revision) == (tip, revision), 'branch-reference')

    try:
        backend = BranchingModel.load_backend(profile)
        oracle = FreshOracle(profile)
        sut = BranchingModel(directory/'store', backend, profile)
        prefix = tuple(profile['fixtures']['prefix'])
        root = sut.create(prefix)
        remember(root, None, prefix)
        aliases['root'] = root
        sut.store.fork('main', root); model.fork('main', root)
        check(root)
        for index, row in enumerate(program):
            report['current_operation'] = index
            if row['source'] not in aliases or row['out'] in aliases or row['op'] not in OPERATIONS:
                raise ValueError('Invalid program dependency or operation')
            source = aliases[row['source']]
            if source not in model.nodes:
                try: sut.store.get_snapshot(source)
                except NotFound: pass
                else: raise ProgramFailure('collected-snapshot-resurrected')
                aliases[row['out']] = source
                report.setdefault('collected_source_checks', 0)
                report['collected_source_checks'] += 1
                report['completed_operations'] += 1
                continue
            op, token, output = row['op'], row['token'], source
            if op == 'append':
                output = sut.feed(source, [token]); remember(output, source, [token])
            elif op in ('generate', 'reseed'):
                stream = 'program-'+str(row['seed'])
                scope = sut.start_generation(source, dict(allowed=[token]), stream)
                remember(scope, source, []); check(scope)
                if op == 'reseed':
                    output = sut.set_random_stream(scope, stream+'-reset')
                    remember(output, scope, [])
                    state = sut.controller(output)
                    require(state['next_draw'] == 0 and state['stream_key'] == hashlib.sha256((stream+'-reset').encode()).hexdigest(), 'reseed-state')
                else:
                    generated = sut.generate(scope, 1)
                    require(generated.accepted_tokens == (token,), 'forced-generation')
                    output = generated.snapshot_id; remember(output, scope, [token])
                    state = sut.controller(output)
                    require(state['next_draw'] == 1 and state['accepted_in_scope'] == 1, 'generation-draw-frontier')
            elif op == 'fork':
                sut.store.fork(row['out'], source); model.fork(row['out'], source)
            elif op in ('rewind', 'retry'):
                target = root if op == 'rewind' else source
                revision = model.heads['main'][1]
                operation = row['out']+'-move'
                expected = model.move('main', revision, target, operation)
                actual = sut.store.move('main', revision, target, operation)
                require((actual.tip, actual.revision) == expected, 'move-receipt')
                if op == 'retry':
                    require(sut.store.move('main', revision, target, operation) == actual, 'retry-receipt')
                output = target
            elif op == 'checkpoint': sut.checkpoint(source)
            elif op == 'evict': sut.evict_all()
            elif op == 'release':
                sut.store.release_history(source); model.release(source)
            elif op == 'gc':
                expected = model.collect(root)
                require(set(sut.store.collect_history(root)['deleted']) == expected, 'collection-reachability')
                output = root
            elif op == 'evaluate':
                code = "result={'mandatory':{'valid':'PASS'},'score':len(candidate['tokens']),'unresolved':[]}"
                spec = dict(task_root=root, evaluator=dict(name='length', version='1', source=code),
                            fixtures={}, metric=dict(name='length', order='maximize'), seed=row['seed'],
                            mandatory=['valid'], timeout_seconds=5)
                value = evaluate(sut.store, source, spec)
                require(value['outcomes']['mandatory'] == {'valid':'PASS'} and
                        value['outcomes']['score'] == len(model.history(source)), 'evaluation-result')
                model.protect(row['out'], source)
            aliases[row['out']] = output
            check(output)
            report['executed_operations'].append(op)
            report['completed_operations'] += 1
        report['status'] = 'PASS'
    except Exception as error:
        report['status'] = 'FAIL'
        report['failure'] = dict(type=type(error).__name__, signature=str(error))
    finally:
        report['seconds'] = time.monotonic()-started
        if oracle is not None:
            report.update(oracle_requests=oracle.requests, oracle_tokens_computed=oracle.tokens_computed)
            oracle.close()
        if backend is not None:
            report['backend_counters'] = dict(backend.stats)
            try:
                if sut is not None: sut.evict_all()
            finally: backend.close()
        (directory/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
