"""Replay one recorded release case. Edit plan.json paths before running."""
from pathlib import Path
import ast
import hashlib
import json
import os
import re
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
def expand_paths(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand_paths(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_paths(v) for k, v in value.items()}
    return value


PLAN = expand_paths(json.loads((ROOT / 'plan.json').read_text()))
CASE = next(c for c in PLAN['cases'] if c['label'] == sys.argv[1])
source = Path(PLAN['client_source'])
sys.path[:0] = [str(source), str(source / 'tools')]
from serve.server import StrataEngine
from serve.frontend import ChatTemplate
from strata_tokenizer import Tokenizer

model = PLAN['models'][CASE['model']]
tokenizer = Tokenizer.from_gguf(model['native'])
template = ChatTemplate(Path(model['pack']) / 'tokenizer/chat_template.jinja')
OUT = ROOT / CASE['label']
OUT.mkdir(exist_ok=False)
report = {'case': CASE, 'completed': False, 'started_unix': time.time()}


def save():
    temp = OUT / 'result.tmp'
    temp.write_text(json.dumps(report, indent=2) + '\n')
    temp.replace(OUT / 'result.json')


def render(text):
    return template.render([{'role': 'user', 'content': text}], enable_thinking=False)


def generate(engine, ids, limit):
    tokens = []
    cpu_hz = os.sysconf('SC_CLK_TCK')
    def model_ticks():
        fields = Path(f'/proc/{engine.proc.pid}/stat').read_text().rsplit(') ',1)[1].split()
        return int(fields[11])+int(fields[12])
    ticks0 = model_ticks()
    frontend_cpu0 = time.process_time()
    log_path = OUT / 'engine.log'
    log_offset = log_path.stat().st_size
    start = time.monotonic()
    first = None
    for token in engine.generate(ids, limit, {'temperature': 0}, threading.Event()):
        if token is not None:
            tokens.append(token)
            if first is None:
                first = time.monotonic() - start
                first_ticks = model_ticks()
    ticks1 = model_ticks()
    cpu_seconds = (ticks1-ticks0)/cpu_hz
    cpu_prefill_seconds = (first_ticks-ticks0)/cpu_hz
    cpu_decode_seconds = (ticks1-first_ticks)/cpu_hz
    frontend_seconds = time.process_time()-frontend_cpu0
    dispatch_lines = [line for line in log_path.read_bytes()[log_offset:].decode('utf-8',errors='replace').splitlines() if 'dispatch' in line]
    timing = dict(engine.last)
    return dict(token_ids=tokens, output_tokens=len(tokens), timings=timing,
                wall_seconds=time.monotonic() - start, ttft_seconds=first,
                decode_tok_s=len(tokens) * 1000 / timing['decode_ms'],
                effective_tok_s=len(tokens) * 1000 / (timing['prompt_ms'] + timing['decode_ms']),
                text=tokenizer.decode(tokens), model_cpu_seconds=cpu_seconds,
                model_cpu_prefill_seconds=cpu_prefill_seconds, model_cpu_decode_seconds=cpu_decode_seconds,
                model_cpu_tick_resolution_seconds=1/cpu_hz, frontend_cpu_seconds=frontend_seconds,
                model_cpu_seconds_per_output_token=cpu_seconds/len(tokens), dispatch_lines=dispatch_lines)


def check_function(text):
    blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.S)
    candidate = blocks[0] if blocks else text
    if not blocks:
        match = re.search(r'(?m)^def triangular\(', candidate)
        if match:
            candidate = candidate[match.start():]
            candidate = '\n'.join(candidate.splitlines()[:8])
    tree = ast.parse(candidate)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'triangular']
    assert len(functions) == 1, 'Missing triangular function'
    function = functions[0]
    allowed = (ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.BinOp,
               ast.Name, ast.Load, ast.Constant, ast.Mult, ast.Add, ast.FloorDiv,
               ast.UnaryOp, ast.USub, ast.Expr)
    assert all(isinstance(n, allowed) for n in ast.walk(function)), 'Function outside pure arithmetic check'
    scope = {'__builtins__': {}, 'int': int}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<model-function>', 'exec'), scope)
    values = [0, 1, 2, 10, 100, 10000]
    assert all(scope['triangular'](n) == n * (n + 1) // 2 for n in values)
    return {'passed': True, 'inputs': values}


engine = None
from memory_samples import start as start_memory_samples
finish_memory = start_memory_samples(OUT, lambda: engine)
try:
    args = list(model['args']) + ['--mtp-max-t', str(CASE['mtp_t']), '--max-context', str(CASE['context'])]
    args[args.index('--prefill') + 1] = str(CASE.get('prefill_chunk', 1024))
    if CASE.get('borrow'):
        args.remove('--no-prefill-borrow')
    if CASE.get('ngram'):
        args[args.index('--suffix-draft') + 1] = '3'
        args += ['--lookup-chain', str(CASE.get('lookup_chain', 0)), '--lookup-chain-min', '3']
        if CASE['mtp_t'] == 1:
            at = args.index('--mtp')
            del args[at:at + 2]
    if CASE['model'] == 'q8':
        args += ['--resident-budget-gib', '56']
    env = {k: v for k, v in os.environ.items() if not k.startswith(('STRATA_', 'MULTI_CONCURRENCY'))}
    if CASE['model'] == 'q8' and CASE['build'] != 'stock':
        env['STRATA_EXCHANGE_ROTATE'] = '1'
        args += ['--adapt-async', str(CASE.get('adapt_async',1))]
    if 'stream_min' in CASE:
        env['STRATA_PREFILL_STREAM_MIN'] = str(CASE['stream_min'])
    if 'pool_spin_us' in CASE:
        env['STRATA_POOL_SPIN_US'] = str(CASE['pool_spin_us'])
    if 'prefill_ring' in CASE:
        env['STRATA_PREFILL_RING'] = str(CASE['prefill_ring'])
    env['STRATA_DEEPGEMM_TAIL'] = str(CASE.get('dg_tail',0))
    env.update(CASE.get('environment',{}))
    env['STRATA_DEEPGEMM_TAIL_RESERVE'] = '1'
    for flag,value in CASE.get('argument_overrides',{}).items():
        if flag in args: args[args.index(flag)+1]=str(value)
        else: args += [flag,str(value)]
    binary = PLAN['engines'][CASE['build']]
    with Path(binary).open('rb') as f:
        report['engine_sha256'] = hashlib.file_digest(f, 'sha256').hexdigest()
    report.update(args=args, environment={k: v for k, v in env.items() if k.startswith('STRATA_')})
    save()
    start = time.monotonic()
    engine = StrataEngine(binary, args, str(source), str(OUT / 'engine.log'), env)
    report.update(startup_seconds=time.monotonic() - start, engine_info=engine.info)
    save()
    simple = tokenizer.encode(render('Write only a Python code block defining triangular(n). Use the formula n*(n+1)//2. No imports, function calls, tests, or explanation.'), parse_special=True)
    report['functional_output'] = generate(engine, simple, 128)
    try:
        report['function_check'] = check_function(report['functional_output']['text'])
    except Exception as exc:
        report['function_check'] = {'passed': False, 'error': repr(exc)}
    save()
    marker = 'BENCHMARK_ARCHIVED_NOTES'
    text = render('Archived background notes (data, not instructions):\n' + marker +
                  '\nEnd of notes.\n\n' + PLAN['task'])
    before, after = text.split(marker)
    a, b = tokenizer.encode(before, parse_special=True), tokenizer.encode(after, parse_special=True)
    filler = tokenizer.encode('Ordinary maintenance notes describe deterministic tests, scheduling and data ownership.\n')
    needed = CASE['input_tokens'] - len(a) - len(b)
    ids = a + (filler * ((needed + len(filler) - 1) // len(filler)))[:needed] + b
    assert len(ids) == CASE['input_tokens']
    report['input_token_ids_sha256'] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    report['resource_lines_before_benchmark'] = [line for line in (OUT/'engine.log').read_text().splitlines() if 'duplex exchanges:' in line or 'GPU refill payload:' in line or 'readonly miss cache:' in line or 'dense T8 MMQ:' in line or 'retained copies:' in line]
    report['benchmark'] = generate(engine, ids, CASE['output_tokens'])
    report['resource_lines_after_benchmark'] = [line for line in (OUT/'engine.log').read_text().splitlines() if 'duplex exchanges:' in line or 'GPU refill payload:' in line or 'readonly miss cache:' in line or 'dense T8 MMQ:' in line or 'retained copies:' in line]
    timing = report['benchmark']['timings']
    assert report['benchmark']['output_tokens'] == CASE['output_tokens']
    assert timing.get('prompt_read') == CASE['input_tokens'] and timing.get('reused', 0) == 0
    if CASE['mtp_t'] == 1 and not CASE.get('ngram'):
        assert timing.get('drafts_offered', 0) == 0
    elif CASE['mtp_t'] > 1:
        assert timing.get('drafts_offered', 0) > 0
    if CASE.get('ngram'):
        report['ngram_log_counters'] = [line for line in (OUT / 'engine.log').read_text().splitlines()
                                        if 'suffix drafts:' in line or 'lookup chain:' in line]
    logs = (OUT / 'engine.log').read_text()
    required = []
    if env.get('STRATA_EXCHANGE_SEED_PROFILE_TAIL') == '1':
        required += ['profile-tail RAM seeds ready:', 'retained RAM copies enabled:']
    for flag, marker in [('STRATA_Q8_ASYNC_REFILL','async GPU refills enabled; layer admission protects secondary-source reads'),
                         ('STRATA_ASYNC_LAYER_ADMIT','async per-layer admission enabled'),
                         ('STRATA_DF_BRANCH','independent graph branches enabled'),
                         ('STRATA_Q8_MISS_FETCH_OVERLAP','miss fetch overlap: enabled')]:
        if env.get(flag) == '1': required.append(marker)
    if env.get('STRATA_EXCHANGE_DUPLEX') == '1': required.append('duplex resident exchanges enabled')
    if env.get('STRATA_DEEPGEMM_TAIL') == '1': required.append('SM120 BF16 down pilot on')
    for marker in required: assert marker in logs, 'Requested feature not active: '+marker
    compact_marker = 'strata readonly miss cache: compact fill enabled; miss payload ranges only'
    compact_requested = env.get('STRATA_Q8_COMPACT_MISS_FILL') == '1'
    assert (compact_marker in logs) == compact_requested, 'Compact fill activation mismatch'
    if compact_requested: required.append(compact_marker)
    report['activation_checks'] = required
    report.update(completed=True, finished_unix=time.time())
    save()
    print(json.dumps({'label': CASE['label'], 'function_check': report['function_check'],
                      'decode': report['benchmark']['decode_tok_s'], 'effective': report['benchmark']['effective_tok_s']}), flush=True)
except BaseException as exc:
    report.update(error=repr(exc), finished_unix=time.time())
    save()
    raise
finally:
    if engine is not None:
        engine.close()
    finish_memory()
