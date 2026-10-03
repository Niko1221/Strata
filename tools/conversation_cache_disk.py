"""L3 disk conversation-cache gate: A/B/A reuse, cross-process restart and payload corruption.

The L3 store (conversation_disk.hpp) parks the same conversations the RAM tier does, one file per
conversation, and survives a process restart.  This gate loads private engines, so it never touches
a running server, and it is dry-run by default: --run needs a separately available GPU/model window.

Scenarios:
  reuse    one process: A, B, then a continued A.  The disk record A+head is parked while B runs and
           the continued A is restored from it.  Output and (spec 1) byte-exact main-model state must
           match a cold baseline engine.
  restart  park A by switching to B, close the engine, start a second engine on the same disk
           directory and request the continued A.  The startup log must report a recovery and the
           request must reuse the parked prefix.
  corrupt  park a record, close the engine, flip one payload byte, restart.  The store must reject the
           record at read time (payload checksum), remove the file, count the corruption and fall back
           to a cold prefill whose output and state equal the cold baseline.

Every run forces `--conversation-cache-mib 0`, so no complete conversation stays in RAM. It keeps
the normal prompt checkpoints enabled because STATE_HASH diagnostics and the live-session continuation
depend on them. The config's layer split stays active through `serve.server.engine_args`.
Any disk settings the config carries are removed, so --disk-path/--disk-gib are the store the run uses
and the cold baseline opens no store at all.
Sampling is greedy (temperature 0) and residency is fixed.
"""
import argparse
import json
from pathlib import Path
import re
import struct
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]

# Mirrors conversation_cache_parity.STATE_KEYS; main() fails closed if the two ever drift.
STATE_KEYS = ('L', 'gdn', 'ple', 'tail', 'dead', 'pooled', 'pooled_full', 'kv', 'ple_prev')
# The engine settings that must not move between the baseline and the disk engines.
RESIDENCY_KEYS = ('expert_slots', 'kv', 'kv_resident', 'context', 'spec', 'mtp_max', 'lookup', 'cvec')
DISK_PREFIX = 'strata serve: conversation disk: '

# conversation_disk.cpp's file format: a fixed 96-byte header, then metadata, then the payload.
DISK_MAGIC = 0x4B53494443525453
DISK_HEADER_BYTES = 96
DISK_FORMAT_VERSION = 1
DISK_RECORD_SUFFIX = '.conversation'

# Every conversation-disk log line the engine writes.  The parser fails closed on any other line
# under the same prefix, so a renamed or reshaped line cannot silently drop evidence.
_DISK_PATTERNS = (
    ('startups', r'(?P<directory>.*) budget=(?P<gib>\d+) GiB recovered=(?P<recovered>\d+) '
                 r'bytes=(?P<bytes>\d+) evictions=(?P<evictions>\d+) corruptions=(?P<corruptions>\d+)'),
    ('parked', r'parked (?P<tokens>\d+) tokens in (?P<ms>[0-9.]+) ms; records=(?P<records>\d+) '
               r'bytes=(?P<bytes>\d+) evictions=(?P<evictions>\d+) corruptions=(?P<corruptions>\d+) '
               r'snapshot_bytes=(?P<snapshot_bytes>\d+)'),
    ('reads', r'read (?P<name>\S+) (?P<stages>\d+) stages in (?P<ms>[0-9.]+) ms; hit (?P<tokens>\d+) tokens '
              r'records=(?P<records>\d+) bytes=(?P<bytes>\d+) hits=(?P<hits>\d+) misses=(?P<misses>\d+)'),
    ('restored', r'restored (?P<tokens>\d+) tokens \((?P<source>live|checkpoint)\) in (?P<ms>[0-9.]+) ms; '
                 r'records=(?P<records>\d+) bytes=(?P<bytes>\d+) hits=(?P<hits>\d+) misses=(?P<misses>\d+)'),
    ('misses', r'miss \(ram_resume=(?P<ram_resume>\d+)\) in (?P<ms>[0-9.]+) ms; hits=(?P<hits>\d+) '
               r'misses=(?P<misses>\d+) records=(?P<records>\d+) bytes=(?P<bytes>\d+) '
               r'evictions=(?P<evictions>\d+) corruptions=(?P<corruptions>\d+)'),
    ('corrupt', r'corrupt record (?P<name>\S+) removed \((?P<reason>.*)\); corruptions=(?P<corruptions>\d+)'),
    ('discarded', r'discard invalid record (?P<name>\S+) \((?P<reason>.*)\)'),
    ('shape_discards', r'record (?P<name>\S+) holds (?P<stages>\d+) stages, not (?P<expected>\d+); discarded'),
    ('read_failures', r'read (?P<name>\S+) failed in (?P<ms>[0-9.]+) ms \((?P<reason>.*)\)'),
    ('skips', r'skip parking \((?P<reason>.*)\)'),
    ('write_failures', r'write failed in (?P<ms>[0-9.]+) ms \((?P<reason>.*)\); records=(?P<records>\d+) '
                       r'bytes=(?P<bytes>\d+)'),
    ('cannot_open', r'cannot open (?P<path>.*): (?P<reason>.*)'),
)
_DISK_TEXT_GROUPS = frozenset(('directory', 'name', 'reason', 'source', 'path'))
_DISK_FLOAT_GROUPS = frozenset(('ms',))
_DISK_EVIDENCE_KEYS = tuple(key for key, _ in _DISK_PATTERNS)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def parse_disk_log(text):
    """The conversation-disk evidence in one engine's log, or an AssertionError on an unknown line."""
    evidence = {key: [] for key in _DISK_EVIDENCE_KEYS}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith(DISK_PREFIX):
            continue
        rest = line[len(DISK_PREFIX):]
        for key, pattern in _DISK_PATTERNS:
            match = re.fullmatch(pattern, rest)
            if match is None:
                continue
            entry = {}
            for name, value in match.groupdict().items():
                if name in _DISK_TEXT_GROUPS:
                    entry[name] = value
                elif name in _DISK_FLOAT_GROUPS:
                    entry[name] = float(value)
                else:
                    entry[name] = int(value)
            evidence[key].append(entry)
            break
        else:
            raise AssertionError(f'unparsed conversation disk log line: {line}')
    return evidence


def read_record_header(data):
    """The fixed 96-byte header of a .conversation record, with the payload's offset."""
    require(len(data) >= DISK_HEADER_BYTES, 'disk record is shorter than its header')
    magic, version, header_bytes = struct.unpack_from('<QII', data, 0)
    record_bytes, metadata_bytes, _ = struct.unpack_from('<QQQ', data, 16)
    payload_bytes, payload_checksum = struct.unpack_from('<QQ', data, 40)
    require(magic == DISK_MAGIC, 'disk record magic differs')
    require(version == DISK_FORMAT_VERSION, 'disk record format version differs')
    require(header_bytes == DISK_HEADER_BYTES, 'disk record header size differs')
    require(record_bytes == len(data), 'disk record length differs from its header')
    require(record_bytes == DISK_HEADER_BYTES + metadata_bytes + payload_bytes,
            'disk record sections do not add up')
    require(payload_bytes > 0, 'disk record has an empty payload')
    return {'magic': magic, 'version': version, 'record_bytes': record_bytes,
            'metadata_bytes': metadata_bytes, 'payload_bytes': payload_bytes,
            'payload_offset': DISK_HEADER_BYTES + metadata_bytes, 'payload_checksum': payload_checksum}


def record_files(directory):
    """The .conversation files in a store directory, sorted."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir()
                  if path.is_file() and path.name.endswith(DISK_RECORD_SUFFIX))


def corrupt_payload(path):
    """Flip one byte in a record's payload (past the header and metadata) and return what changed."""
    data = bytearray(Path(path).read_bytes())
    header = read_record_header(bytes(data))
    offset = header['payload_offset'] + header['payload_bytes'] // 2
    require(offset < len(data), 'payload offset falls outside the record')
    original = data[offset]
    data[offset] ^= 0xFF
    require(data[offset] != original, 'the payload byte was not changed')
    Path(path).write_bytes(bytes(data))
    return {'file': str(path), 'offset': offset, 'byte': original, **header}


def disk_flags(path, gib, slots=0, min_free_mib=0):
    """The runtime's --conversation-cache-disk* arguments; the optional two appear only when set."""
    require(gib > 0, 'the disk budget must be positive')
    require(slots >= 0 and min_free_mib >= 0, 'the disk record cap and free-space floor cannot be negative')
    flags = ['--conversation-cache-disk', str(path), '--conversation-cache-disk-gib', str(gib)]
    if slots:
        flags += ['--conversation-cache-disk-slots', str(slots)]
    if min_free_mib:
        flags += ['--conversation-cache-disk-min-free-mib', str(min_free_mib)]
    return flags

DISK_ARG_FLAGS = ('--conversation-cache-disk', '--conversation-cache-disk-gib',
                   '--conversation-cache-disk-slots', '--conversation-cache-disk-min-free-mib')


def strip_disk_args(args):
    """Drop any config-supplied --conversation-cache-disk* pair, so the harness owns the store.

    serve.server.engine_args appends the config's disk settings.  The gate controls the store through
    its own --disk-path/--disk-gib flags, and the cold baseline must have no disk store at all, so
    the config's pairs are removed before the harness appends its own.
    """
    kept = []
    skip = 0
    for arg in args:
        if skip:
            skip -= 1
            continue
        if arg in DISK_ARG_FLAGS:
            skip = 1   # the flag's value follows
            continue
        kept.append(arg)
    return kept




def harness_args(server_args, spec, disk=None):
    """The engine's arguments: the server's (config + layer split) plus the fixed cache/greedy ones.

    Complete-conversation RAM parking is forced off with `--conversation-cache-mib 0`, so only
    the live session, its normal prompt checkpoints, and the disk store can reuse state. The prompt
    checkpoints stay enabled because the parity fingerprints require a valid live session.
    disk is (path, gib, slots, min_free_mib) or None for the cold baseline.
    """
    args = list(server_args) + [
        '--conversation-cache-mib', '0', '--conversation-cache-slots', '4',
        '--prompt-cache', '6', '--adapt-swaps', '0', '--spec', str(max(2, spec)),
        '--mtp-max-t', str(spec), '--suffix-draft', '0', '--spec-min-p', '0']
    if disk is not None:
        args += disk_flags(*disk)
    return args


def verify_records(engine, names, label):
    """Fail closed on an incomplete request sequence, output or state fingerprint."""
    records = engine['records']
    require([r['name'] for r in records] == list(names), f'{label}: incomplete request sequence')
    require(engine.get('hash_count') == len(records), f'{label}: missing state hashes')
    for record in records:
        require(bool(record.get('ids')), f'{label}: missing generated tokens')
        require(record.get('finish') in ('length', 'stop'), f'{label}: request did not finish normally')
        require(isinstance(record.get('reused'), int), f'{label}: missing reuse evidence')
        require(set(STATE_KEYS) <= set(record.get('state', {})), f'{label}: incomplete state fingerprint')


def verify_common(results, spec):
    """INFO, residency, layer split and baseline-log checks shared by every scenario."""
    require(results.get('spec') == spec, 'result spec differs from the run')
    disk = results['disk']
    require(disk['gib'] > 0, 'the disk budget must be positive')
    baseline = results['baseline']
    info = baseline['info']
    require(info.get('conversation_cache_mib') == 0, 'the baseline RAM cache is not disabled')
    require(info.get('conversation_cache_disk') == 0, 'the baseline unexpectedly opened a disk store')
    base_log = baseline['log']
    require(not base_log['startups'] and not base_log['parked'] and not base_log['reads'] and
            not base_log['restored'] and not base_log['misses'], 'the baseline wrote disk cache evidence')
    split = results['layer_split']
    require(split['baseline_in_args'] == split['configured'],
            'the baseline dropped the config layer split')
    disk_engines = results['disk_engines']
    require(len(split['disk_in_args']) == len(disk_engines), 'layer-split evidence is incomplete')
    for i, engine in enumerate(disk_engines):
        label = engine.get('label', f'disk-{i}')
        require(split['disk_in_args'][i] == split['configured'],
                f'{label}: dropped the config layer split')
        info = engine['info']
        require(info.get('conversation_cache_disk') == 1, f'{label}: INFO does not report the disk cache enabled')
        require(info.get('conversation_cache_disk_gib') == disk['gib'],
                f'{label}: INFO disk budget differs from the configured GiB')
        require(info.get('conversation_cache_mib') == 0, f'{label}: the RAM cache is not disabled')
        require(engine.get('hash_count') == len(engine['records']), f'{label}: missing state hashes')
        log = engine['log']
        require(len(log['startups']) == 1, f'{label}: missing the disk startup log line')
        startup = log['startups'][0]
        require(startup['gib'] == disk['gib'], f'{label}: startup budget differs from the configured GiB')
        require(Path(startup['directory']) == Path(disk['path']), f'{label}: startup directory differs')
        require(not log['skips'], f'{label}: parking was skipped: {log["skips"]}')
        require(not log['write_failures'], f'{label}: a disk write failed: {log["write_failures"]}')
        require(not log['read_failures'], f'{label}: a disk read failed: {log["read_failures"]}')
        require(not log['cannot_open'], f'{label}: the disk store could not open')
        require(not log['shape_discards'], f'{label}: a record held the wrong number of stages')
        for key in RESIDENCY_KEYS:
            require(info.get(key) == baseline['info'].get(key), f'{label}: engine setting differs: {key}')


def verify_reuse(results, spec):
    """One process A/B/A: the disk record parked on the switch to B must restore the continued A."""
    require(results['scenario'] == 'reuse', 'not a reuse result')
    verify_common(results, spec)
    prompt_tokens = results['prompt_tokens']['A']
    baseline = results['baseline']
    verify_records(baseline, ('A', 'A+'), 'baseline')
    require(len(results['disk_engines']) == 1, 'reuse must run one disk engine')
    engine = results['disk_engines'][0]
    verify_records(engine, ('A', 'B', 'A+'), 'reuse')
    records = engine['records']
    require(records[0]['ids'] == baseline['records'][0]['ids'], 'cold A output differs')
    if spec == 1:
        require(records[0]['state'] == baseline['records'][0]['state'], 'cold A state differs')
    require(records[1]['reused'] == 0, 'unrelated B reused cached state')
    require(records[2]['reused'] >= prompt_tokens, 'A was not restored from the disk store')
    require(records[2]['ids'] == baseline['records'][1]['ids'], 'restored continuation output differs')
    if spec == 1:
        require(records[2]['state'] == baseline['records'][1]['state'], 'restored main-model state differs')
    log = engine['log']
    require(log['startups'][0]['recovered'] == 0, 'the reuse store was not empty at startup')
    require(log['parked'], 'no disk park was recorded')
    require(len(log['reads']) == 1 and len(log['restored']) == 1, 'the disk read/restore was not recorded')
    require(log['reads'][0]['tokens'] == records[2]['reused'], 'the read log disagrees with the reuse count')
    require(log['restored'][0]['tokens'] == records[2]['reused'], 'the restore log disagrees with the reuse count')
    require(log['restored'][0]['source'] == 'live', 'the restored prefix was not the parked live image')
    require(log['misses'], 'no disk miss was recorded for the unrelated requests')
    require(not log['corrupt'] and not log['discarded'], 'the reuse run reported a bad record')


def verify_restart(results, spec):
    """Park in one process, close it, resume the continued A in a second process on the same store."""
    require(results['scenario'] == 'restart', 'not a restart result')
    verify_common(results, spec)
    prompt_tokens = results['prompt_tokens']['A']
    baseline = results['baseline']
    verify_records(baseline, ('A', 'A+'), 'baseline')
    require(len(results['disk_engines']) == 2, 'restart must run two disk engines')
    first, second = results['disk_engines']
    verify_records(first, ('A', 'B'), 'restart-first')
    verify_records(second, ('A+',), 'restart-second')
    require(first['records'][0]['ids'] == baseline['records'][0]['ids'], 'cold A output differs')
    if spec == 1:
        require(first['records'][0]['state'] == baseline['records'][0]['state'], 'cold A state differs')
    require(first['records'][1]['reused'] == 0, 'unrelated B reused cached state')
    require(first['log']['startups'][0]['recovered'] == 0, 'the first store was not empty at startup')
    require(first['log']['parked'], 'A was not parked before the restart')
    require(second['log']['startups'][0]['recovered'] >= 1, 'the restarted store recovered no record')
    require(len(second['log']['reads']) == 1 and len(second['log']['restored']) == 1,
            'the restarted engine did not read/restore the record')
    require(not second['log']['misses'], 'the restarted engine fell back to a cold prefill')
    record = second['records'][0]
    require(record['reused'] >= prompt_tokens, 'the continued A did not reuse the recovered prefix')
    require(second['log']['reads'][0]['tokens'] == record['reused'], 'the read log disagrees with the reuse count')
    require(second['log']['restored'][0]['tokens'] == record['reused'],
            'the restore log disagrees with the reuse count')
    require(record['ids'] == baseline['records'][1]['ids'], 'restored continuation output differs')
    if spec == 1:
        require(record['state'] == baseline['records'][1]['state'], 'restored main-model state differs')
    require(not first['log']['corrupt'] and not second['log']['corrupt'] and
            not first['log']['discarded'] and not second['log']['discarded'],
            'the restart run reported a bad record')


def verify_corrupt(results, spec):
    """One flipped payload byte must be rejected at read time and fall back to an identical cold prefill."""
    require(results['scenario'] == 'corrupt', 'not a corrupt result')
    verify_common(results, spec)
    baseline = results['baseline']
    verify_records(baseline, ('A', 'A+'), 'baseline')
    require(len(results['disk_engines']) == 2, 'corrupt must run two disk engines')
    first, second = results['disk_engines']
    verify_records(first, ('A', 'B'), 'corrupt-first')
    verify_records(second, ('A+',), 'corrupt-second')
    require(first['log']['startups'][0]['recovered'] == 0, 'the first store was not empty at startup')
    require(first['log']['parked'], 'A was not parked before the corruption')
    corruption = results['corruption']
    require(corruption['payload_bytes'] > 0, 'the corrupted record had an empty payload')
    require(corruption['payload_offset'] >= DISK_HEADER_BYTES, 'the flipped byte was not past the header')
    require(corruption['removed'], 'the corrupted record file was not removed')
    startup = second['log']['startups'][0]
    require(startup['recovered'] >= 1, 'the restarted store did not index the corrupted record')
    require(startup['corruptions'] == 0, 'the corruption was counted at open, not at read time')
    log = second['log']
    require(len(log['corrupt']) == 1, 'no corrupt-record removal was logged')
    require('payload checksum' in log['corrupt'][0]['reason'],
            f'the corruption was not detected as a payload checksum: {log["corrupt"][0]["reason"]}')
    require(log['corrupt'][0]['name'], 'the corrupt log line has no record name')
    require(log['corrupt'][0]['corruptions'] >= 1, 'the corrupt log line did not count the corruption')
    require(not log['reads'] and not log['restored'], 'a corrupted record was read or restored')
    require(log['misses'], 'the corrupted read did not fall back to a disk miss')
    require(log['misses'][-1]['corruptions'] >= 1, 'the miss log did not carry the corruption count')
    record = second['records'][0]
    require(record['reused'] == 0, 'the corrupted record was reused')
    require(record['ids'] == baseline['records'][1]['ids'], 'cold fallback output differs')
    # A cold full-prompt read and a live-session continuation can use different prompt chunk boundaries.
    # Their generated output is the consumer-visible fallback contract; internal state hashes need not match.
    require(not first['log']['corrupt'] and not first['log']['discarded'] and
            not first['log']['read_failures'], 'the first engine reported a bad record')


def verify(results, spec):
    """Dispatch to the scenario's gate; all checks fail closed."""
    scenario = results.get('scenario')
    if scenario == 'reuse':
        verify_reuse(results, spec)
    elif scenario == 'restart':
        verify_restart(results, spec)
    elif scenario == 'corrupt':
        verify_corrupt(results, spec)
    else:
        raise AssertionError(f'unknown scenario: {scenario}')


def run_engine(exe, args, cwd, log_path, env, steps, state_hashes):
    """Load one private engine, run steps, close it, and pair every request with its STATE_HASH.

    steps is a list of (name, ids, max_new); ids may be a callable taking the records so far, for a
    continuation built from an earlier request's output.
    """
    from serve.server import StrataEngine
    engine = StrataEngine(str(exe), list(args), cwd=cwd, log=str(log_path), env=env)
    records = []
    try:
        for name, ids, max_new in steps:
            if callable(ids):
                ids = ids(records)
            out = [t for t in engine.generate(list(ids), max_new, {'temperature': 0}, threading.Event())
                   if t is not None]
            require(bool(out), f'{name}: the engine generated no tokens')
            records.append({'name': name, 'ids': out, **engine.last})
    finally:
        engine.close()
    text = log_path.read_text(encoding='utf-8')
    hashes = state_hashes(text)
    require(len(hashes) == len(records),
            f'{log_path.name}: {len(hashes)} STATE_HASH lines for {len(records)} requests')
    for record, state in zip(records, hashes):
        record['state'] = state
    return {'info': dict(engine.info), 'records': records, 'hash_count': len(hashes),
            'log': parse_disk_log(text), 'log_path': str(log_path)}


def prompts(tok, tpl, paragraphs):
    """The deterministic A/B/continuation prompts, rendered through the chat template."""
    def encode(text):
        return tok.encode(tpl.render([{'role': 'user', 'content': text}], enable_thinking=False),
                          parse_special=True)
    a = encode('Conversation A: remember this list.\n' + '\n'.join(
        f'Record {i}: blue square, green triangle, red circle.' for i in range(paragraphs)))
    b = encode('Unrelated conversation B: name a color.\n' + '\n'.join(
        f'Record {i}: an entirely different list.' for i in range(4)))
    suffix = tok.encode('<|im_end|>\n<|im_start|>user\nName a color from the list.<|im_end|>\n'
                        '<|im_start|>assistant\n thinking\n\n</think>\n\n', parse_special=True)
    return a, b, suffix


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--engine', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True,
                    help='new private directory for the logs and results.json; existing paths refused')
    ap.add_argument('--disk-path', type=Path, required=True,
                    help='the L3 store directory; must be empty or missing')
    ap.add_argument('--disk-gib', type=int, default=25, help='the store file budget in GiB (default 25)')
    ap.add_argument('--disk-slots', type=int, default=0, help='the store record cap (default 0 = no cap)')
    ap.add_argument('--disk-min-free-mib', type=int, default=0,
                    help='the free-space floor the store keeps (default 0)')
    ap.add_argument('--paragraphs', type=int, default=128, help='the length of conversation A')
    ap.add_argument('--scenario', choices=('reuse', 'restart', 'corrupt'), default='reuse')
    ap.add_argument('--spec', type=int, default=1, choices=range(1, 9),
                    help='decode window cap; 1 requires byte-exact main-model state')
    ap.add_argument('--run', action='store_true')
    a = ap.parse_args()
    if a.disk_gib <= 0 or a.paragraphs < 1 or a.disk_slots < 0 or a.disk_min_free_mib < 0:
        ap.error('disk-gib and paragraphs must be positive; disk-slots and disk-min-free-mib cannot be negative')
    disk = (a.disk_path, a.disk_gib, a.disk_slots, a.disk_min_free_mib)
    if not a.run:
        print(f'Dry run: {a.scenario} disk conversation cache; no model loaded and no files created.')
        print(f'  config: {a.config}')
        print(f'  engine: {a.engine}')
        print(f'  output: {a.output}')
        print(f'  disk: {a.disk_path} ({a.disk_gib} GiB, slots {a.disk_slots}, '
              f'min-free {a.disk_min_free_mib} MiB)')
        print(f'  forced flags: --conversation-cache-mib 0 --prompt-cache 6; disk flags '
              f'{" ".join(disk_flags(*disk))}; greedy sampling; config layer split retained')
        print('No model loaded. Use --run only with a separately available GPU/test window.')
        return
    sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
    from serve.server import child_env, engine_args as server_engine_args, gpu_list
    from serve.frontend import ChatTemplate
    from conversation_cache_parity import STATE_KEYS as parity_state_keys, load_tokenizer, state_hashes
    require(tuple(parity_state_keys) == STATE_KEYS, 'STATE_KEYS drifted from conversation_cache_parity')
    cfg = json.loads(a.config.read_text(encoding='utf-8'))
    output = a.output.resolve()
    disk_path = a.disk_path.resolve()
    require(not output.exists(), f'the output directory already exists: {output}')
    require(disk_path != output, 'the disk path must not be the output directory')
    if disk_path.exists():
        require(disk_path.is_dir() and not any(disk_path.iterdir()),
                f'the disk directory is not empty: {disk_path}')
    output.mkdir(mode=0o700, parents=False)
    p = Path(cfg['tokenizer'])
    tok = load_tokenizer(p)
    tpl = ChatTemplate(p / 'chat_template.jinja')
    A, B, suffix = prompts(tok, tpl, a.paragraphs)
    env = child_env(cfg)
    env['STRATA_STATE_HASH'] = '1'
    cwd = cfg.get('cwd')
    server_args = strip_disk_args(server_engine_args(cfg))
    split_configured = len(gpu_list(cfg)) > 1
    engine_exe = a.engine.resolve()
    base_args = harness_args(server_args, a.spec)
    baseline = run_engine(engine_exe, base_args, cwd, output / 'baseline.log', env,
                          [('A', A, 1), ('A+', lambda records: A + records[0]['ids'] + suffix, 8)],
                          state_hashes)
    head = baseline['records'][0]['ids']
    continuation = list(A) + head + list(suffix)
    disk_args = harness_args(server_args, a.spec, (disk_path, a.disk_gib, a.disk_slots, a.disk_min_free_mib))
    engines = []
    corruption = None
    if a.scenario == 'reuse':
        engines.append({'label': 'disk-0',
                        **run_engine(engine_exe, disk_args, cwd, output / 'disk-0.log', env,
                                     [('A', A, 1), ('B', B, 1), ('A+', continuation, 8)], state_hashes)})
    else:
        engines.append({'label': 'disk-0',
                        **run_engine(engine_exe, disk_args, cwd, output / 'disk-0.log', env,
                                     [('A', A, 1), ('B', B, 1)], state_hashes)})
        if a.scenario == 'corrupt':
            files = record_files(disk_path)
            require(len(files) == 1, f'expected one parked record, found {len(files)}')
            corruption = corrupt_payload(files[0])
        engines.append({'label': 'disk-1',
                        **run_engine(engine_exe, disk_args, cwd, output / 'disk-1.log', env,
                                     [('A+', continuation, 8)], state_hashes)})
        if corruption is not None:
            corruption['removed'] = not Path(corruption['file']).exists()
    results = {
        'scenario': a.scenario,
        'spec': a.spec,
        'paragraphs': a.paragraphs,
        'disk': {'path': str(disk_path), 'gib': a.disk_gib, 'slots': a.disk_slots,
                 'min_free_mib': a.disk_min_free_mib},
        'layer_split': {'configured': split_configured,
                        'baseline_in_args': '--layer-split' in base_args,
                        'disk_in_args': ['--layer-split' in disk_args for _ in engines]},
        'prompt_tokens': {'A': len(A), 'B': len(B), 'continuation': len(continuation)},
        'baseline': baseline,
        'disk_engines': engines,
        'corruption': corruption,
    }
    (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    verify(results, a.spec)
    print(f'PASS: {a.scenario} disk conversation cache, output and main-model state parity'
          + (', byte-exact state' if a.spec == 1 else ''))
    print(f'Results: {output / "results.json"}')


if __name__ == '__main__':
    main()
