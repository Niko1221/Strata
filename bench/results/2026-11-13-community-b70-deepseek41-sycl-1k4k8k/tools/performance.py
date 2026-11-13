import json
from pathlib import Path
import statistics
from http_client import stream_chat

ROOT = Path(__file__).resolve().parents[1]
run = ROOT / 'runs/performance01'
run.mkdir(exist_ok=False)
corpus = json.loads((ROOT / 'config/performance.json').read_text())
warmups = json.loads((ROOT / 'config/warmups.json').read_text())
for case in warmups:
    result = stream_chat(case['body'])
    (run / (case['id'] + '-excluded.json')).write_text(json.dumps(result, indent=2) + '\n')
    assert result['status'] == 200 and result['done'] and result['error'] is None, case['id']
    assert result['usage']['prompt_tokens'] == case['expected_prompt_tokens'], result['usage']
    print(case['id'], 'excluded warmup', result['usage'], flush=True)
rows = []
with (run / 'results.jsonl').open('x') as output:
    for case in corpus:
        result = stream_chat(case['body'])
        timings, usage = result.get('timings') or {}, result.get('usage') or {}
        measured = {'prompt_tok_s': timings.get('prompt_per_second'),
                    'decode_tok_s': timings.get('predicted_per_second'),
                    'ttft_s': result['ttft_s'], 'total_s': result['duration_s']}
        valid = (result['status'] == 200 and result['done'] and result['error'] is None
                 and result['finish_reason'] in ['stop', 'length']
                 and usage.get('prompt_tokens') == case['expected_prompt_tokens']
                 and all(isinstance(x, (int, float)) and x > 0 for x in measured.values()))
        row = {k: case[k] for k in ['id', 'kind', 'repetition', 'expected_prompt_tokens', 'expected_output_tokens']}
        row.update({'measurement_valid': bool(valid), 'measured': measured, **result})
        rows.append(row)
        output.write(json.dumps(row, ensure_ascii=False) + '\n')
        output.flush()
        (run / (case['id'] + '.json')).write_text(json.dumps(row, ensure_ascii=False, indent=2) + '\n')
        print(case['id'], 'valid=' + str(valid), measured, 'usage=', usage,
              'finish=' + str(result['finish_reason']), flush=True)
        if result['error'] or result['status'] != 200:
            break
groups = {}
for kind in ['1024', '4096', '8192']:
    group = [x for x in rows if x['kind'] == kind and x['measurement_valid']]
    metrics = {}
    for metric in ['prompt_tok_s', 'decode_tok_s', 'ttft_s', 'total_s']:
        values = [x['measured'][metric] for x in group]
        metrics[metric] = {'values': values, 'median': statistics.median(values) if values else None,
                           'min': min(values) if values else None, 'max': max(values) if values else None}
    groups[kind] = {'valid_runs': len(group), 'metrics': metrics,
                    'capped': sum(x['finish_reason'] == 'length' for x in group),
                    'actual_output_tokens': [x['usage']['completion_tokens'] for x in group]}
summary = {'kind': 'absolute performance; no comparator', 'context': 16384,
           'requested': len(corpus), 'executed': len(rows),
           'valid': sum(x['measurement_valid'] for x in rows),
           'warmups_excluded': len(warmups), 'groups': groups}
(run / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2), flush=True)
raise SystemExit(0 if summary['valid'] == len(corpus) else 2)
