"""Offline checks for the disk conversation-cache gate's parsing and verification (no model loaded)."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from tools.conversation_cache_disk import (DISK_MAGIC, STATE_KEYS, corrupt_payload, disk_flags,
                                           harness_args, parse_disk_log, read_record_header,
                                           strip_disk_args, verify, verify_corrupt, verify_reuse,
                                           verify_restart)

STARTUP = ('strata serve: conversation disk: /tmp/store budget=25 GiB recovered=1 bytes=4096 '
           'evictions=0 corruptions=0')
PARKED = ('strata serve: conversation disk: parked 1234 tokens in 12.5 ms; records=1 bytes=4096 '
          'evictions=0 corruptions=0 snapshot_bytes=4096')
READ = ('strata serve: conversation disk: read c0a1-1234 1 stages in 1.5 ms; hit 1234 tokens '
        'records=1 bytes=4096 hits=1 misses=0')
RESTORED = ('strata serve: conversation disk: restored 1234 tokens (live) in 2.5 ms; records=1 '
            'bytes=4096 hits=1 misses=0')
MISS = ('strata serve: conversation disk: miss (ram_resume=0) in 0.5 ms; hits=0 misses=1 records=0 '
        'bytes=0 evictions=0 corruptions=0')
CORRUPT = ('strata serve: conversation disk: corrupt record c0a1-1234 removed '
           '(corrupt conversation record: payload checksum); corruptions=1')


def synthetic_record(metadata=b'm' * 8, payload=b'p' * 16):
    header = struct.pack('<QII', DISK_MAGIC, 1, 96)
    header += struct.pack('<QQQ', 96 + len(metadata) + len(payload), len(metadata), 0)
    header += struct.pack('<QQ', len(payload), 0)
    header += b'\0' * 32 + b'\0' + b'\0' * 7
    return header + metadata + payload


def state(name):
    return {key: name for key in STATE_KEYS}


def record(name, ids, reused):
    return {'name': name, 'ids': list(ids), 'finish': 'length', 'reused': reused, 'state': state(name)}


def info(disk=1, gib=25):
    return {'conversation_cache_mib': 0, 'conversation_cache_slots': 4, 'conversation_cache_min_free_mib': 0,
            'conversation_cache_disk': disk, 'conversation_cache_disk_gib': gib,
            'conversation_cache_disk_records': 1, 'conversation_cache_disk_bytes': 100,
            'conversation_cache_disk_recoveries': 0, 'conversation_cache_disk_corruptions': 0,
            'expert_slots': 100, 'kv': 'int8', 'kv_resident': 32768, 'context': 4096,
            'spec': 2, 'mtp_max': 1, 'lookup': 0, 'cvec': 'none'}


def empty_log(**overrides):
    evidence = parse_disk_log('')
    evidence.update(overrides)
    return evidence


def startup(gib=25, recovered=0, directory='/store', corruptions=0):
    return {'directory': directory, 'gib': gib, 'recovered': recovered, 'bytes': 0, 'evictions': 0,
            'corruptions': corruptions}


def engine_evidence(label, records, engine_info, log):
    return {'label': label, 'info': engine_info, 'records': records, 'log': log,
            'hash_count': len(records), 'log_path': label + '.log'}


def evidence(scenario, prompt_tokens=1000):
    park = {'tokens': prompt_tokens + 1, 'records': 1, 'bytes': 100, 'evictions': 0, 'corruptions': 0,
            'snapshot_bytes': 100}
    read = {'name': 'c0-1', 'stages': 1, 'tokens': prompt_tokens + 1, 'records': 1, 'bytes': 100,
            'hits': 1, 'misses': 0}
    restore = {'tokens': prompt_tokens + 1, 'source': 'live', 'records': 1, 'bytes': 100, 'hits': 1,
               'misses': 0}
    miss = {'ram_resume': 0, 'hits': 0, 'misses': 1, 'records': 0, 'bytes': 0, 'evictions': 0,
            'corruptions': 0}
    results = {
        'scenario': scenario, 'spec': 1, 'paragraphs': 8,
        'disk': {'path': '/store', 'gib': 25, 'slots': 0, 'min_free_mib': 0},
        'layer_split': {'configured': False, 'baseline_in_args': False, 'disk_in_args': []},
        'prompt_tokens': {'A': prompt_tokens, 'B': 3, 'continuation': prompt_tokens + 5},
        'baseline': {'info': info(disk=0, gib=0), 'records': [record('A', [1], 0),
                                                             record('A+', [1, 2], prompt_tokens + 1)],
                     'log': empty_log(), 'hash_count': 2, 'log_path': 'baseline.log'},
        'disk_engines': [], 'corruption': None,
    }
    if scenario == 'reuse':
        log = empty_log(startups=[startup()], parked=[park], reads=[read], restored=[restore],
                        misses=[miss, miss])
        results['disk_engines'] = [engine_evidence('disk-0', [
            record('A', [1], 0), record('B', [9], 0), record('A+', [1, 2], prompt_tokens + 1)],
            info(), log)]
    else:
        log0 = empty_log(startups=[startup()], parked=[park], misses=[miss, miss])
        first = engine_evidence('disk-0', [record('A', [1], 0), record('B', [9], 0)], info(), log0)
        if scenario == 'restart':
            log1 = empty_log(startups=[startup(recovered=1)], reads=[read], restored=[restore])
            second = engine_evidence('disk-1', [record('A+', [1, 2], prompt_tokens + 1)], info(), log1)
        else:
            corrupt_miss = dict(miss, records=1, bytes=100, corruptions=1)
            log1 = empty_log(startups=[startup(recovered=1)],
                             corrupt=[{'name': 'c0-1',
                                       'reason': 'corrupt conversation record: payload checksum',
                                       'corruptions': 1}], misses=[corrupt_miss])
            second = engine_evidence('disk-1', [record('A+', [1, 2], 0)], info(), log1)
            results['corruption'] = {'file': '/store/c0-1.conversation', 'offset': 112, 'byte': 112,
                                     'payload_offset': 104, 'payload_bytes': 16, 'metadata_bytes': 8,
                                     'record_bytes': 120, 'magic': DISK_MAGIC, 'version': 1,
                                     'payload_checksum': 0, 'removed': True}
        results['disk_engines'] = [first, second]
    results['layer_split']['disk_in_args'] = [False] * len(results['disk_engines'])
    return results


class DiskLogParsing(unittest.TestCase):
    def test_every_line_kind_parses(self):
        text = '\n'.join(['strata serve: READY 4096', STARTUP, PARKED, READ, RESTORED, MISS, CORRUPT,
                          'strata serve: conversation disk: skip parking (no live prefix)',
                          'strata serve: conversation disk: write failed in 1.0 ms (disk full); records=0 bytes=0'])
        evidence = parse_disk_log(text)
        self.assertEqual(evidence['startups'][0]['gib'], 25)
        self.assertEqual(evidence['startups'][0]['recovered'], 1)
        self.assertEqual(evidence['startups'][0]['directory'], '/tmp/store')
        self.assertEqual(evidence['parked'][0]['snapshot_bytes'], 4096)
        self.assertEqual(evidence['parked'][0]['ms'], 12.5)
        self.assertEqual(evidence['reads'][0], {'name': 'c0a1-1234', 'stages': 1, 'ms': 1.5,
                                                'tokens': 1234, 'records': 1, 'bytes': 4096,
                                                'hits': 1, 'misses': 0})
        self.assertEqual(evidence['restored'][0]['source'], 'live')
        self.assertEqual(evidence['misses'][0]['ram_resume'], 0)
        self.assertEqual(evidence['corrupt'][0]['corruptions'], 1)
        self.assertEqual(evidence['skips'], [{'reason': 'no live prefix'}])
        self.assertEqual(evidence['write_failures'][0]['records'], 0)

    def test_unknown_disk_line_fails_closed(self):
        with self.assertRaises(AssertionError):
            parse_disk_log('strata serve: conversation disk: a line the parser does not know')

    def test_other_lines_ignored(self):
        evidence = parse_disk_log('strata serve: conversation cache: parked 1 tokens\nREADY 4096\n')
        self.assertTrue(all(not values for values in evidence.values()))


class DiskRecordFormat(unittest.TestCase):
    def test_header_and_payload_offsets(self):
        header = read_record_header(synthetic_record())
        self.assertEqual(header['magic'], DISK_MAGIC)
        self.assertEqual(header['metadata_bytes'], 8)
        self.assertEqual(header['payload_bytes'], 16)
        self.assertEqual(header['payload_offset'], 104)
        self.assertEqual(header['record_bytes'], 120)

    def test_header_rejects_bad_magic_and_length(self):
        data = bytearray(synthetic_record())
        data[0] ^= 0xFF
        with self.assertRaises(AssertionError):
            read_record_header(bytes(data))
        with self.assertRaises(AssertionError):
            read_record_header(synthetic_record()[:-1])

    def test_corrupt_payload_flips_one_byte_inside_the_payload(self):
        with tempfile.TemporaryDirectory(prefix='strata-disk-') as directory:
            path = Path(directory) / 'c0-1.conversation'
            path.write_bytes(synthetic_record())
            changed = corrupt_payload(path)
            self.assertEqual(changed['offset'], 112)
            self.assertEqual(changed['payload_bytes'], 16)
            data = path.read_bytes()
            self.assertEqual(data[:104], synthetic_record()[:104])
            self.assertNotEqual(data[112], synthetic_record()[112])
            self.assertEqual(data[113:], synthetic_record()[113:])

    def test_corrupt_payload_needs_a_payload(self):
        with tempfile.TemporaryDirectory(prefix='strata-disk-') as directory:
            path = Path(directory) / 'c0-1.conversation'
            path.write_bytes(synthetic_record(payload=b''))
            with self.assertRaises(AssertionError):
                corrupt_payload(path)


class DiskArguments(unittest.TestCase):
    def test_flags_include_optional_only_when_set(self):
        self.assertEqual(disk_flags('/d', 25), ['--conversation-cache-disk', '/d',
                                                '--conversation-cache-disk-gib', '25'])
        self.assertEqual(disk_flags('/d', 25, 3, 128),
                         ['--conversation-cache-disk', '/d', '--conversation-cache-disk-gib', '25',
                          '--conversation-cache-disk-slots', '3',
                          '--conversation-cache-disk-min-free-mib', '128'])
        with self.assertRaises(AssertionError):
            disk_flags('/d', 0)

    def test_harness_args_disable_both_ram_tiers_and_keep_the_server_args(self):
        args = harness_args(['--native', 'pack', '--layer-split', 'auto'], 1, ('/d', 25, 0, 0))
        self.assertEqual(args[:3], ['--native', 'pack', '--layer-split'])
        self.assertEqual(args[args.index('--conversation-cache-mib') + 1], '0')
        self.assertEqual(args[args.index('--prompt-cache') + 1], '6')
        self.assertIn('--conversation-cache-disk', args)
        self.assertNotIn('--conversation-cache-disk-slots', args)
        baseline = harness_args(['--native', 'pack'], 1)
        self.assertNotIn('--conversation-cache-disk', baseline)
        self.assertEqual(baseline[baseline.index('--conversation-cache-mib') + 1], '0')

    def test_config_supplied_disk_args_are_stripped(self):
        args = strip_disk_args(['--native', 'pack', '--conversation-cache-disk', '/cfg',
                                '--conversation-cache-disk-gib', '4',
                                '--conversation-cache-disk-slots', '2',
                                '--conversation-cache-disk-min-free-mib', '8',
                                '--layer-split', 'auto'])
        self.assertEqual(args, ['--native', 'pack', '--layer-split', 'auto'])
        self.assertEqual(strip_disk_args(['--prompt-cache', '6']), ['--prompt-cache', '6'])


class DiskGate(unittest.TestCase):
    def test_valid_evidence_passes(self):
        for scenario in ('reuse', 'restart', 'corrupt'):
            with self.subTest(scenario=scenario):
                verify(evidence(scenario), 1)

    def test_common_failures(self):
        cases = {
            'disk disabled in INFO': lambda d: d['disk_engines'][0]['info'].update(conversation_cache_disk=0),
            'gib mismatch': lambda d: d['disk_engines'][0]['info'].update(conversation_cache_disk_gib=1),
            'ram cache enabled': lambda d: d['disk_engines'][0]['info'].update(conversation_cache_mib=4096),
            'residency mismatch': lambda d: d['disk_engines'][0]['info'].update(expert_slots=1),
            'baseline has a disk store': lambda d: d['baseline']['info'].update(conversation_cache_disk=1),
            'baseline wrote disk logs': lambda d: d['baseline']['log'].update(startups=[startup()]),
            'missing startup log': lambda d: d['disk_engines'][0]['log'].update(startups=[]),
            'startup budget differs': lambda d: d['disk_engines'][0]['log']['startups'][0].update(gib=1),
            'dropped layer split': lambda d: d['layer_split'].update(configured=True),
            'missing state hash': lambda d: d['disk_engines'][0].update(hash_count=99),
            'empty output': lambda d: d['disk_engines'][0]['records'][0]['ids'].clear(),
            'missing state': lambda d: d['disk_engines'][0]['records'][0]['state'].clear(),
            'skipped parking': lambda d: d['disk_engines'][0]['log'].update(skips=[{'reason': 'no'}]),
            'unknown scenario': lambda d: d.update(scenario='other'),
        }
        for name, mutate in cases.items():
            with self.subTest(case=name):
                data = evidence('reuse')
                mutate(data)
                with self.assertRaises(AssertionError):
                    verify(data, 1)

    def test_reuse_failures(self):
        cases = {
            'no reuse': lambda d: d['disk_engines'][0]['records'][2].update(reused=0),
            'reuse below the prompt': lambda d: d['disk_engines'][0]['records'][2].update(reused=1),
            'continuation output differs': lambda d: d['disk_engines'][0]['records'][2].update(ids=[7]),
            'continuation state differs': lambda d: d['disk_engines'][0]['records'][2]['state'].update(gdn='x'),
            'cold A output differs': lambda d: d['disk_engines'][0]['records'][0].update(ids=[7]),
            'unrelated B reused': lambda d: d['disk_engines'][0]['records'][1].update(reused=5),
            'no read log': lambda d: d['disk_engines'][0]['log'].update(reads=[]),
            'no park log': lambda d: d['disk_engines'][0]['log'].update(parked=[]),
            'read log disagrees': lambda d: d['disk_engines'][0]['log']['reads'][0].update(tokens=1),
            'corrupt record': lambda d: d['disk_engines'][0]['log'].update(corrupt=[{'name': 'c', 'reason': 'x', 'corruptions': 1}]),
            'store not empty at startup': lambda d: d['disk_engines'][0]['log']['startups'][0].update(recovered=1),
            'missing request': lambda d: d['disk_engines'][0]['records'].pop(),
            'two disk engines': lambda d: d['disk_engines'].append(d['disk_engines'][0]),
        }
        for name, mutate in cases.items():
            with self.subTest(case=name):
                data = evidence('reuse')
                mutate(data)
                with self.assertRaises(AssertionError):
                    verify_reuse(data, 1)

    def test_restart_failures(self):
        cases = {
            'no recovery': lambda d: d['disk_engines'][1]['log']['startups'][0].update(recovered=0),
            'no reuse after restart': lambda d: d['disk_engines'][1]['records'][0].update(reused=0),
            'cold prefill after restart': lambda d: d['disk_engines'][1]['log'].update(
                misses=[{'ram_resume': 0, 'hits': 0, 'misses': 1, 'records': 1, 'bytes': 100,
                         'evictions': 0, 'corruptions': 0}]),
            'no read after restart': lambda d: d['disk_engines'][1]['log'].update(reads=[]),
            'first store not empty': lambda d: d['disk_engines'][0]['log']['startups'][0].update(recovered=1),
            'A was not parked': lambda d: d['disk_engines'][0]['log'].update(parked=[]),
            'continuation output differs': lambda d: d['disk_engines'][1]['records'][0].update(ids=[7]),
            'continuation state differs': lambda d: d['disk_engines'][1]['records'][0]['state'].update(kv='x'),
            'one disk engine': lambda d: d['disk_engines'].pop(),
        }
        for name, mutate in cases.items():
            with self.subTest(case=name):
                data = evidence('restart')
                mutate(data)
                with self.assertRaises(AssertionError):
                    verify_restart(data, 1)

    def test_corrupt_failures(self):
        cases = {
            'no corruption log': lambda d: d['disk_engines'][1]['log'].update(corrupt=[]),
            'wrong corruption reason': lambda d: d['disk_engines'][1]['log']['corrupt'][0].update(reason='header'),
            'file not removed': lambda d: d['corruption'].update(removed=False),
            'no payload': lambda d: d['corruption'].update(payload_bytes=0),
            'no recovery at startup': lambda d: d['disk_engines'][1]['log']['startups'][0].update(recovered=0),
            'corruption counted at open': lambda d: d['disk_engines'][1]['log']['startups'][0].update(corruptions=1),
            'record reused': lambda d: d['disk_engines'][1]['records'][0].update(reused=5),
            'read a corrupted record': lambda d: d['disk_engines'][1]['log'].update(
                reads=[{'name': 'c0-1', 'stages': 1, 'tokens': 1, 'records': 1, 'bytes': 1, 'hits': 1, 'misses': 0}]),
            'no miss after corruption': lambda d: d['disk_engines'][1]['log'].update(misses=[]),
            'cold fallback output differs': lambda d: d['disk_engines'][1]['records'][0].update(ids=[7]),
            'A was not parked': lambda d: d['disk_engines'][0]['log'].update(parked=[]),
        }
        for name, mutate in cases.items():
            with self.subTest(case=name):
                data = evidence('corrupt')
                mutate(data)
                with self.assertRaises(AssertionError):
                    verify_corrupt(data, 1)

    def test_spec_two_skips_byte_exact_state(self):
        data = evidence('reuse')
        data['spec'] = 2
        data['disk_engines'][0]['records'][2]['state'].update(gdn='different')
        verify_reuse(data, 2)


class DiskDryRun(unittest.TestCase):
    def test_dry_run_does_not_read_missing_config_or_create_paths(self):
        with tempfile.TemporaryDirectory(prefix='strata-disk-dry-') as directory:
            p = Path(directory)
            result = subprocess.run([
                sys.executable, str(Path(__file__).with_name('conversation_cache_disk.py')),
                '--config', str(p / 'missing.json'), '--engine', str(p / 'missing'),
                '--output', str(p / 'not-created'), '--disk-path', str(p / 'store'),
                '--scenario', 'reuse'],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('no model loaded', result.stdout)
            self.assertIn('--conversation-cache-mib 0', result.stdout)
            self.assertFalse((p / 'not-created').exists())
            self.assertFalse((p / 'store').exists())

    def test_dry_run_rejects_a_nonpositive_budget(self):
        with tempfile.TemporaryDirectory(prefix='strata-disk-dry-') as directory:
            p = Path(directory)
            result = subprocess.run([
                sys.executable, str(Path(__file__).with_name('conversation_cache_disk.py')),
                '--config', str(p / 'missing.json'), '--engine', str(p / 'missing'),
                '--output', str(p / 'not-created'), '--disk-path', str(p / 'store'),
                '--disk-gib', '0'],
                capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('disk-gib', result.stderr)


class DiskResultsJson(unittest.TestCase):
    def test_evidence_is_json_serializable(self):
        for scenario in ('reuse', 'restart', 'corrupt'):
            with self.subTest(scenario=scenario):
                json.dumps(evidence(scenario))


if __name__ == '__main__':
    unittest.main()
