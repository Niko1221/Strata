"""Paired short-context CPU/GPU miss-share experiment; Python standard library only."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8080')
    p.add_argument('--model', default='qwen3.8-flash-next')
    p.add_argument('--engine-log', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--fractions', default='1,.5,.25,0')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--startup-only', action='store_true')
    args = p.parse_args()
    fractions = [float(x) for x in args.fractions.split(',')]
    if args.repeats < 1 or not fractions or any(not 0 <= x <= 1 for x in fractions):
        p.error('positive repeats and fractions between zero and one are required')
    if args.startup_only and len(fractions) != 1:
        p.error('--startup-only requires one fraction describing the startup configuration')
    protocol = json.loads(Path(__file__).with_name('protocol.json').read_text(encoding='utf-8'))
    # Refuse to overwrite an existing experiment, including a partially failed one.
    args.out.mkdir(parents=True, exist_ok=False)

    def save(name, data):
        (args.out / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    save('protocol.json', protocol)
    save('run-settings.json', dict(created=dt.datetime.now(dt.timezone.utc).isoformat(),
                                  fractions=fractions, repeats=args.repeats, startup_only=args.startup_only))
    records = []
    for repeat in range(1, args.repeats + 1):
        order = fractions if repeat % 2 else fractions[::-1]
        for fraction in order:
            for test, prompt in enumerate(protocol['prompts'], 1):
                name = f'r{repeat}-pcie{fraction:g}-test{test}'
                payload = dict(model=args.model, messages=[dict(role='user', content=prompt)],
                               max_tokens=protocol['max_tokens'], temperature=protocol['temperature'],
                               seed=protocol['seed'], cache_prompt=False,
                               chat_template_kwargs=dict(enable_thinking=False))
                if not args.startup_only:
                    payload['strata_tune'] = dict(pcie_frac=fraction)
                save(name + '-request.json', payload)
                offset = args.engine_log.stat().st_size
                start = time.perf_counter()
                request = urllib.request.Request(args.url.rstrip('/') + '/v1/chat/completions',
                    data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                    headers={'Content-Type': 'application/json; charset=utf-8'})
                with urllib.request.urlopen(request, timeout=600) as response:
                    answer = json.load(response)
                elapsed = time.perf_counter() - start
                save(name + '-response.json', answer)
                with args.engine_log.open('rb') as log_file:
                    log_file.seek(offset)
                    log = log_file.read().decode('utf-8', errors='replace')
                (args.out / (name + '-engine.log')).write_text(log, encoding='utf-8')
                match = re.search(r'decode expert compute: (\d+) CPU jobs, (\d+) RAM-to-GPU experts', log)
                if not match or '0 blob reads from the file' not in log:
                    raise RuntimeError('missing expert counters or unconfirmed expert-file residency')
                cpu_jobs, copies = map(int, match.groups())
                if fraction == 1 and cpu_jobs != 0:
                    raise RuntimeError('GPU reference fell back to CPU experts')
                if fraction == 0 and (copies != 0 or cpu_jobs == 0):
                    raise RuntimeError('CPU-miss mode was not engaged')
                timing = answer['timings']
                if timing['cache_n'] != 0 or timing['predicted_n'] != protocol['max_tokens']:
                    raise RuntimeError('request reused a cache or did not reach the fixed token limit')
                content = answer['choices'][0]['message']['content']
                row = dict(repeat=repeat, pcie_frac=fraction, test=test, elapsed_seconds=elapsed,
                           generation_tps=timing['predicted_n'] * 1000 / timing['predicted_ms'],
                           prompt_tps=timing['prompt_per_second'], output_tokens=timing['predicted_n'],
                           drafts=timing['draft_n'], accepted=timing['draft_n_accepted'],
                           cpu_expert_jobs=cpu_jobs, ram_gpu_experts=copies,
                           answer_sha256=hashlib.sha256(content.encode('utf-8')).hexdigest())
                records.append(row)
                save('runs.json', records)
                print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
