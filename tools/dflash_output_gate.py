#!/usr/bin/env python3
"""Real-weight output/cache gate. Never accepts disabled caching as a cache pass.

Uses an existing configuration; runs private, sequential native-protocol engines.
All requested arguments, resolved INFO, RESUME/DONE lines and token IDs are saved.
Exit 1 means a requirement failed, not an infrastructure failure.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from dflash_sampling import Engine, replace
from strata_tokenizer import Tokenizer


class ObservedEngine(Engine):
    def __init__(self, *args, **kwargs):
        self.info = {}
        self.protocol = []
        super().__init__(*args, **kwargs)

    def line(self):
        line = super().line()
        if line.startswith('INFO '):
            self.info = dict(p.split('=', 1) for p in line.split()[1:] if '=' in p)
        if line.startswith(('INFO ', 'RESUME ', 'DONE ', 'ERR')):
            self.protocol.append(line)
        return line

    def request(self, ids, sampling, count):
        result = super().request(ids, sampling, count)
        result['requested'] = count
        if 'done' in result:
            fields = result['done'].split()
            result['reused'] = int(fields[8])
            result['read'] = int(fields[14]) if len(fields) > 14 else None
        return result


def first_difference(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                min(len(a), len(b)) if len(a) != len(b) else None)


def evaluate(results, selected, require_cache=True):
    checks = {}
    for label, arm in results.items():
        records = arm['records']
        status = {}
        status['fixed_expert_residency'] = all(
            arm['info'].get(key) == '2000' for key in ('expert_slots', 'expert_slots_primary'))
        if require_cache:
            status['cold_control_missed'] = records['cache-cold'].get('reused',-1) == 0
            status['cache_enabled'] = int(arm['info'].get('conversation_cache_mib','0'))==2048
            for key in ('cache-live','cache-restored','cache-checkpoint'):
                status[key+'-hit'] = records[key].get('reused',0)>0
                status[key+'-cold-parity'] = records[key]['tokens']==records['cache-cold']['tokens']
        profile, mode = label.split('-',1)
        reference = results.get(profile+'-ar')
        status['target_reference_present'] = reference is not None
        if reference:
            status['target_configuration_matches'] = all(
                key in arm['info'] and key in reference['info'] and
                arm['info'][key] == reference['info'][key]
                for key in ('expert_slots', 'expert_slots_primary', 'context', 'kv', 'spec'))
            if require_cache:
                for key in ('cache-live','cache-restored','cache-checkpoint','cache-cold'):
                    status['target-'+key] = records[key]['tokens']==reference['records'][key]['tokens']
            for name in selected:
                for kind in ('greedy','sampled'):
                    key = name+'-'+kind
                    a = reference['records'][key]['tokens']
                    b = records[key]['tokens']
                    status[key] = dict(equal=a==b, first_difference=first_difference(a,b),
                                       target_length=len(a),candidate_length=len(b))
        status['complete_requests'] = all('error' not in r and len(r['tokens']) == r['requested']
                                          for r in records.values() if isinstance(r,dict) and 'requested' in r)
        checks[label] = status
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=ROOT/'strata-dflash2-test.json')
    ap.add_argument('--engine', type=Path, default=ROOT/'build-dflash2/strata')
    ap.add_argument('--plain-draft', type=Path, default=ROOT.parent/'Strata-data/dflash/drafter-Q8_0.gguf')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--profiles', nargs='+', choices=['fixed', 'mixed'], default=['fixed', 'mixed'])
    ap.add_argument('--modes', nargs='+', choices=['ar', 'dflash', 'dflash2'], default=['ar', 'dflash', 'dflash2'])
    ap.add_argument('--probe-position', type=int)
    ap.add_argument('--cache-only', action='store_true')
    ap.add_argument('--cache-prefix-token', type=int)
    ap.add_argument('--only', nargs='+')
    o = ap.parse_args()
    out = o.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(o.config.read_text())
    original = list(cfg['args'])
    pack = Path(original[original.index('--pack')+1])/'tokenizer'
    vocab = json.loads((pack/'vocab.json').read_text())
    tokens = [None]*len(vocab)
    for text, idx in vocab.items():
        tokens[idx] = text
    tok = Tokenizer(tokens, (pack/'merges.txt').read_text().split('\n'),
                    json.loads((pack/'token_type.json').read_text()))
    def prompt(text):
        return tok.encode('<|im_start|>user\n'+text+'<|im_end|>\n<|im_start|>assistant\n<think>\n', parse_special=True)
    prompts = {
        'code': prompt('Write a Python function that merges overlapping intervals. Include a clear explanation and unit tests.'),
        'math': prompt('Solve step by step: A train travels 180 km at 60 km/h, then 120 km at 80 km/h. What is its average speed for the entire trip?'),
        'prose': prompt('Explain how a CPU cache works, including locality, cache lines, and why sequential memory access is usually faster.'),
        'long': prompt(('Reference notes: CPU caches store recently accessed memory. Sequential access improves locality.\n'*150)+'\nSummarize these notes and give a Python example of sequential array access.')
    }
    selected = [] if o.cache_only else [k for k in prompts if not o.only or k in o.only]
    (out/'prompts.json').write_text(json.dumps(prompts, indent=2))
    results = {}
    for profile in o.profiles:
        for mode in o.modes:
            label = profile+'-'+mode
            args = original.copy()
            for key in ('--mtp', '--dflash', '--dflash-head', '--dflash-vocab', '--pcie-frac'):
                replace(args, key, None)
            if mode != 'ar':
                draft = o.plain_draft if mode == 'dflash' else Path(original[original.index('--dflash')+1])
                replace(args, '--dflash', str(draft.resolve()))
                replace(args, '--dflash-block', '3')
            else:
                for key in ('--dflash-block', '--dflash-window'):
                    replace(args, key, None)
            for key, value in (('--spec','8'), ('--spec-min-p','0'), ('--expert-cache','2000'),
                ('--prefill','512'), ('--max-context','8192'), ('--kv-resident','8192'),
                ('--adapt-every','0'), ('--suffix-draft','0'), ('--lookup-chain','0'),
                ('--eos-ids','248319'), ('--prompt-cache','6'), ('--prompt-cache-every','128'),
                ('--prompt-cache-root','0'), ('--conversation-cache-mib','2048'),
                ('--conversation-cache-slots','4'), ('--conversation-cache-min-tokens','1'),
                ('--conversation-cache-min-free-mib','0')):
                replace(args, key, value)
            if profile == 'fixed':
                replace(args, '--pcie-frac', '0')
            env = dict(os.environ, STRATA_PREFILL_CPU_SHARE='0')
            env.pop('STRATA_DF_EVENTS', None)
            if o.probe_position is not None:
                env['STRATA_DF_TARGET_PROBE'] = str(o.probe_position)
                env['STRATA_DF_TARGET_LOGITS'] = str(out/f'{label}-logits.bin')
                Path(env['STRATA_DF_TARGET_LOGITS']).unlink(missing_ok=True)
            (out/f'{label}-command.json').write_text(json.dumps([str(o.engine.resolve()), *args, '--serve'], indent=2))
            print('START',label,flush=True)
            engine = ObservedEngine(o.engine.resolve(), args, ROOT, env, out/f'{label}.log')
            records = {}
            try:
                for name in selected:
                    for kind, sampling, count in (
                        ('greedy','temperature=0 seed=1234 ckpt=0',128),
                        ('sampled','temperature=0.7 seed=3456 top_k=40 top_p=0.9 min_p=0 ckpt=0',64)):
                        key = name+'-'+kind
                        r = engine.request(prompts[name], sampling, count)
                        r['prompt_tokens'] = len(prompts[name])
                        records[key] = r
                        print(label,key,len(r['tokens']),r.get('accepted'),r.get('offered'),flush=True)
                if o.only is None:
                    a = prompt('Read these notes and explain them: '+('CPU caches exploit locality. '*60))
                    b = prompt('Find the pattern in this different sequence: '+('2 4 8 16 32 64. '*60))
                    # A predetermined continuation lets us run its cold control
                    # BEFORE any checkpoint with this prefix exists.
                    cont = a + tok.encode('Let us reason carefully.\nContinue the explanation with a Python example.',parse_special=True)
                    if o.cache_prefix_token is not None:
                        cont = a + [o.cache_prefix_token] + tok.encode('\nContinue the explanation with a Python example.',parse_special=True)
                    records['cache-cold'] = engine.request(cont,'temperature=0 seed=1234 ckpt=0',16)
                    records['cache-A'] = engine.request(a,'temperature=0 seed=1234',1)
                    for key, ids, count in (
                        ('cache-live',cont,16), ('cache-B',b,1),
                        ('cache-restored',cont,16), ('cache-checkpoint',cont,16)):
                        records[key] = engine.request(ids,'temperature=0 seed=1234',count)
                    records['cache-inputs'] = dict(A=a,B=b,continuation=cont)
                    records['cache-prompts'] = dict(A=len(a),B=len(b),continuation=len(cont))
            finally:
                engine.close()
            results[label] = dict(info=engine.info,records=records,protocol=engine.protocol)
            (out/'results.json').write_text(json.dumps(results,indent=2))
    checks = evaluate(results, selected, o.only is None)
    (out/'checks.json').write_text(json.dumps(checks,indent=2))
    good = all((value['equal'] if isinstance(value,dict) else value)
               for arm in checks.values() for value in arm.values())
    print('PASS' if good else 'FAIL: inspect checks.json',flush=True)
    return 0 if good else 1


if __name__ == '__main__':
    raise SystemExit(main())
