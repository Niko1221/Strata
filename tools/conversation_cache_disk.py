"""Private-engine NVMe restart, corruption, identity and admission gate.

Dry-run by default. Six sequential model loads require an exclusive GPU window.
The test only mutates snapshots and tokenizer copies in a new private directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from conversation_cache_parity import STATE_KEYS, engine_args, load_tokenizer, require, state_hashes
from conversation_cache_soak import answer_text
from serve.frontend import ChatTemplate
from serve.server import StrataEngine, child_env


def evidence(text):
    return {
        'verified': [dict(draft=h, cells=int(c), mode=int(m), source=s, resident=int(r)) for h, c, m, s, r in re.findall(
            r'SNAPSHOT_VERIFY draft=([0-9a-f]{16}) cells=(\d+) mode=(\d+) source=(ram|disk) resident=(\d+)', text)],
        'draft_prefill': [dict(path=p, mode=int(m), cells=int(c)) for p, m, c in re.findall(
            r'DRAFT_PREFILL path=(batched|token) mode=(\d+) cells=(\d+)', text)],
        'identity_mismatches': re.findall(r'snapshot identity differs: ([A-Za-z0-9_./:-]+)', text),
        'spills': text.count('disk cache: spilled '),
        'promotion_skips': text.count('disk cache: promotion skipped '),
        'integrity_failures': text.count('snapshot integrity check failed'),
        'parks': [dict(parked=int(n), bytes=int(b)) for n, b in re.findall(
            r'conversation cache: parked \d+ tokens .*?parked=(\d+) bytes=(\d+)', text)],
    }


def verify(results):
    phases = results['phases']
    require(set(phases) == {'baseline', 'producer', 'restart', 'admission', 'foreign', 'corrupt'}, 'incomplete disk lifecycle')
    baseline, producer = phases['baseline'], phases['producer']
    require([r['name'] for r in baseline['records']] == ['A', 'warm', 'B', 'cold'], 'incomplete baseline controls')
    require([r['name'] for r in producer['records']] == ['A', 'B', 'C', 'return'], 'incomplete eviction sequence')
    warm, cold = baseline['records'][1], baseline['records'][3]
    require(cold['reused'] == 0, 'cold control unexpectedly reused active state')
    require(len(baseline['records'][0]['ids']) == len(producer['records'][0]['ids']) == 1, 'initial A must consume exactly its prompt')
    require(producer['records'][0]['ids'] == baseline['records'][0]['ids'], 'initial A differs')
    require(producer['disk_counts'][0:2] == [0, 0] and producer['disk_counts'][2] > 0,
            'disk writes must start on RAM eviction, not on continuing turns or ordinary parking')
    require(producer['disk_counts'][-1] == producer['files_after_close'] > 0, 'shutdown unexpectedly changed disk retention')
    require(producer['evidence']['spills'] > 0, 'no disk spill observed')
    require(baseline['info']['conversation_cache_mib'] == baseline['info']['conversation_disk_mib'] == 0,
            'baseline cache configuration differs')
    for name, phase in phases.items():
        for key in ('context', 'kv', 'kv_resident', 'expert_slots', 'spec', 'mtp_max', 'lookup', 'cvec', 'pcie_frac'):
            require(phase['info'][key] == baseline['info'][key], f'{name}: inference setting differs: {key}')
        if name != 'baseline':
            require(phase['info']['conversation_cache_mib'] == results['cache_mib'] and
                    phase['info']['conversation_cache_slots'] == 1 and
                    phase['info']['conversation_disk_mib'] == results['disk_mib'] and
                    phase['info']['conversation_disk_slots'] == 8, f'{name}: cache configuration differs')
        for record in phase['records']:
            require(record['ids'] and record['finish'] in ('length', 'stop'), f'{name}: request did not complete')
            require(set(STATE_KEYS) <= record['state'].keys(), f'{name}: missing main-state fingerprint')
        require(all(p['parked'] <= 1 and p['bytes'] <= results['cache_mib'] * 1024 * 1024
                    for p in phase['evidence']['parks']), f'{name}: RAM retention exceeded its budget')
        require(len(phase['disk_bytes']) == len(phase['records']) and
                all(0 <= n <= results['disk_mib'] * 1024 * 1024 for n in phase['disk_bytes']),
                f'{name}: missing or over-budget disk occupancy')
        require(len(phase['disk_counts']) == len(phase['records']) and all(0 <= n <= 8 for n in phase['disk_counts']),
                f'{name}: missing or over-budget disk entry count')
    mode = 0 if results['draft_path'] == 'batched' else 2
    path = 'batched' if mode == 0 else 'token'
    require(any(p['path'] == path and p['mode'] == mode and p['cells'] > 0
                for p in producer['evidence']['draft_prefill']), 'requested draft prefill path never ran')
    require(phases['admission']['info']['conversation_cache_min_free_mib'] == results['denial_floor_mib'],
            'admission phase did not use the requested floor')
    require(results['foreign_asset_before'] != results['foreign_asset_after'], 'tokenizer identity was not changed')
    require('asset/tokenizer/vocab.json' in phases['foreign']['evidence']['identity_mismatches'],
            'foreign tokenizer rejection did not identify the first changed asset')
    require(results['corrupted_files'] > 0, 'no persisted files were corrupted')
    fingerprints = []
    for name in ('producer', 'restart', 'admission', 'foreign', 'corrupt'):
        phase = phases[name]
        record = phase['records'][-1]
        require(record['name'] == 'return' and (name == 'producer' or len(phase['records']) == 1), 'invalid consumer sequence')
        reference = warm if name in ('producer', 'restart') else cold
        require(answer_text(reference['text']) == results['expected'].casefold(), 'baseline known answer is wrong')
        require(answer_text(record['text']) == results['expected'].casefold(), f'{name}: known answer is wrong')
        require(record['ids'] == reference['ids'], f'{name}: continuation token parity differs')
        require(record['state'] == reference['state'], f'{name}: main-model state parity differs')
        verified = [v for v in phase['evidence']['verified'] if v['source'] == 'disk']
        if name in ('producer', 'restart'):
            require(record['reused'] >= results['prompt_tokens'], f'{name}: full disk prefix not restored')
            require(len(verified) == 1 and verified[0]['mode'] == mode and verified[0]['cells'] >= results['prompt_tokens'],
                    f'{name}: missing draft read-back proof for the requested layout')
            if mode == 2:
                require(0 < verified[0]['resident'] < results['prompt_tokens'],
                        f'{name}: restored prompt did not exceed the actual draft ring')
            fingerprints.append(verified[0]['draft'])
        else:
            require(record['reused'] == 0 and not verified, f'{name}: incompatible/unadmitted file reused')
    require(fingerprints[0] == fingerprints[1], 'persisted draft payload changed across restart')
    require(phases['admission']['evidence']['promotion_skips'] > 0, 'no disk staging admission rejection observed')
    require(phases['corrupt']['evidence']['integrity_failures'] > 0, 'no integrity rejection observed')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--engine', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cache-mib', type=int, default=8192)
    ap.add_argument('--disk-mib', type=int, default=16384)
    ap.add_argument('--paragraphs', type=int, default=128)
    ap.add_argument('--draft-path', choices=('batched', 'ring'), required=True)
    ap.add_argument('--resident-cells', type=int, default=32768)
    ap.add_argument('--min-free-mib', type=int, default=2560)
    ap.add_argument('--run', action='store_true')
    a = ap.parse_args()
    limit = (2**63 - 1) // (1024 * 1024)
    if not 0 < a.cache_mib <= limit or not 0 < a.disk_mib <= limit or a.paragraphs < 1 or a.resident_cells < 4:
        ap.error('positive in-range cache/disk budgets, paragraphs and resident-cells >= 4 are required')
    if not 0 <= a.min_free_mib <= limit:
        ap.error('min-free-mib must be a nonnegative engine-range integer')
    if not a.run:
        print(f'Dry run: {a.draft_path} draft KV; six sequential engines; restart, admission, tokenizer identity and corruption.')
        print('No model loaded. --run requires an exclusive GPU/model window.')
        return
    cfg = json.loads(a.config.read_text(encoding='utf-8'))
    a.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    output = a.output.resolve()
    tokenizer = output / 'tokenizer'
    tokenizer.mkdir(mode=0o700)
    for name in ('vocab.json', 'merges.txt', 'token_type.json', 'chat_template.jinja'):
        source = Path(cfg['tokenizer']) / name
        if name == 'chat_template.jinja' and not source.exists():
            source = ROOT / 'serve/chat_template.jinja'
        shutil.copyfile(source, tokenizer / name)
    tok = load_tokenizer(tokenizer)
    tpl = ChatTemplate(tokenizer / 'chat_template.jinja')
    expected = 'AZURE-314159'
    def encode(text):
        return tok.encode(text, parse_special=True)
    def prompt(label, code):
        content = f'Conversation {label}. Remember the exact code {code}.\n'
        content += '\n'.join(f'Record {i}: blue square, green triangle, red circle.' for i in range(a.paragraphs))
        content += f'\nThe exact code to remember is {code}. Reply OK.'
        return encode(tpl.render([{'role': 'user', 'content': content}], enable_thinking=False))
    A, B, C = prompt('A', expected), prompt('B', 'BRONZE-271828'), prompt('C', 'CORAL-161803')
    if a.draft_path == 'ring':
        # The draft ring is window + 4*spec + 64 cells, then page-rounded.
        # Reject a short test before loading any model; actual residency is also
        # required in the read-back evidence, since the runtime may fall back.
        window = 32768
        for i, arg in enumerate(cfg['args'][:-1]):
            if arg == '--mtp-window':
                window = int(cfg['args'][i + 1])
        require(window > 0 and len(A) > max(a.resident_cells, window + 128),
                'ring gate needs a prompt beyond main residency and the draft window; increase --paragraphs')
    suffix = encode('<|im_end|>\n<|im_start|>user\nWhat exact code did I ask you to remember? '
                    'Reply with only that code.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
    disk = output / 'disk'
    env = child_env(cfg)
    env.update(STRATA_STATE_HASH='1', STRATA_SNAPSHOT_VERIFY='1', STRATA_MTP_BATCH='1')
    results = {'phases': {}, 'expected': expected, 'prompt_tokens': len(A), 'cache_mib': a.cache_mib,
               'disk_mib': a.disk_mib, 'draft_path': a.draft_path, 'denial_floor_mib': limit}
    continuation = None
    def files():
        return sorted((disk / 'strata-conversations-v2').glob('*.snap'))
    def save():
        (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    for phase in ('baseline', 'producer', 'restart', 'admission', 'foreign', 'corrupt'):
        vocab = tokenizer / 'vocab.json'
        if phase == 'foreign':
            results['foreign_asset_before'] = hashlib.sha256(vocab.read_bytes()).hexdigest()
            with vocab.open('ab') as stream:
                stream.write(b'\n')  # valid JSON with identical tokenizer semantics
            results['foreign_asset_after'] = hashlib.sha256(vocab.read_bytes()).hexdigest()
        if phase == 'corrupt':
            with vocab.open('r+b') as stream:
                stream.seek(-1, 2)
                stream.truncate()  # undo only this test's appended newline
            damaged = files()
            require(damaged, 'producer left no snapshots to corrupt')
            for path in damaged:
                with path.open('r+b') as stream:
                    stream.seek(-1, 2)
                    byte = stream.read(1)
                    stream.seek(-1, 2)
                    stream.write(bytes([byte[0] ^ 1]))
            results['corrupted_files'] = len(damaged)
        args = engine_args(cfg, 0 if phase == 'baseline' else a.cache_mib, 1)
        args += ['--conversation-cache-slots', '1', '--conversation-cache-disk', str(disk),
                 '--conversation-cache-disk-mib', '0' if phase == 'baseline' else str(a.disk_mib),
                 '--conversation-cache-disk-slots', '8', '--conversation-cache-tokenizer', str(tokenizer),
                 '--conversation-cache-template', str(tokenizer / 'chat_template.jinja'),
                 '--conversation-cache-min-free-mib', str(limit if phase == 'admission' else a.min_free_mib),
                 '--kv-resident', str(0 if a.draft_path == 'batched' else a.resident_cells)]
        log = output / f'{phase}.log'
        engine = StrataEngine(str(a.engine.resolve()), args, cwd=cfg.get('cwd'), log=str(log), env=env)
        entry = {'info': dict(engine.info), 'records': [], 'disk_counts': [], 'disk_bytes': []}
        results['phases'][phase] = entry
        def generate(ids, count, name):
            tokens = [t for t in engine.generate(ids, count, {'temperature': 0}, threading.Event()) if t is not None]
            entry['records'].append({'name': name, 'ids': tokens, 'text': tok.decode(tokens), **engine.last})
            entry['disk_counts'].append(len(files()))
            entry['disk_bytes'].append(sum(path.stat().st_size for path in files()))
            return tokens
        try:
            if phase == 'baseline':
                head = generate(A, 1, 'A')
                continuation = A + head + suffix
                generate(continuation, 24, 'warm')
                generate(B, 1, 'B')
                generate(continuation, 24, 'cold')
            elif phase == 'producer':
                generate(A, 1, 'A')
                generate(B, 1, 'B')
                generate(C, 1, 'C')
                generate(continuation, 24, 'return')
            else:
                generate(continuation, 24, 'return')
        finally:
            engine.close()
            engine.proc.wait(timeout=30)  # close() may have killed a slow child; wait before the next model load
            engine.log.close()
            save()
        text = log.read_text(encoding='utf-8')
        hashes = state_hashes(text)
        require(len(hashes) == len(entry['records']), f'{phase}: missing state hashes')
        for record, state in zip(entry['records'], hashes):
            record['state'] = state
        entry['evidence'] = evidence(text)
        entry['files_after_close'] = len(files())
        save()
    verify(results)
    print('PASS: eviction-only persistence, restart, draft read-back, known answer, main-state parity, admission and foreign/corrupt fallback')
    print(f'Results: {output / "results.json"}')


if __name__ == '__main__':
    main()
