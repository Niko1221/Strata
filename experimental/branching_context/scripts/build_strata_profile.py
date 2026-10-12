"""Pin measured Strata files/settings; incomplete calibration never gets tolerances."""
import argparse
import array
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

from calibration_inputs import execution_settings


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finalize_existing_profile(profile_path, calibration_path, output):
    """Attach complete measured calibration without reattributing a pinned build."""
    from context_time_travel.strata_backend import _validate_execution, _validate_model
    profile = json.loads(Path(profile_path).read_text())
    baseline = json.loads(Path(calibration_path).read_text())
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    lengths = [1,31,127,255,256,257,511,1024,4096]
    require(baseline.get('status') == 'FROZEN', 'Calibration is not complete')
    require(baseline.get('calibration_lengths') == lengths and baseline.get('repetitions') == 3,
            'Calibration matrix differs from the predeclared matrix')
    rows = baseline.get('commands', [])
    require(len(rows) == 27 and {(r['length'],r['repeat']) for r in rows} ==
            {(n,r) for n in lengths for r in range(3)}, 'Incomplete calibration forwards')
    _validate_execution(profile)
    _validate_model(profile)
    admission_path = Path(baseline['resource_admission'])
    require(sha(admission_path) == baseline['resource_admission_sha256'], 'Admission receipt changed')
    admission = json.loads(admission_path.read_text())
    settings = execution_settings(admission)
    command = profile['execution']['oracle_command']
    require(command == baseline['command'] == admission['oracle_command'], 'Oracle command mismatch')
    require(profile['execution']['sut_command'] == admission['sut_command'], 'Branch command mismatch')
    require(sha(command[0]) == baseline['oracle_sha256'] == admission['oracle_binary_sha256'],
            'Oracle binary mismatch')
    require(sha(admission['sut_command'][0]) == admission['sut_binary_sha256'], 'Branch binary mismatch')
    require(baseline['hardware'] == profile['hardware'], 'Hardware mismatch')
    # Commands and binary hashes above bind the actual operator. These two
    # historical descriptions name that same corrected T=1 executable; retain
    # both descriptions without changing the pinned execution identity.
    descriptions = (baseline['operator'], profile['execution']['operator'])
    require(descriptions[0] == descriptions[1] or descriptions == (
        'Verifier.run(T=1); successful recurrent commit; next-token frontier',
        'Verifier.run(T=1); bound GPU expert plan; fixed PCIe share 0.55; recurrent commit; next-token frontier'),
        'operator mismatch')
    for key in ('environment','capacity'):
        require(baseline[key] == profile['execution'][key], key+' mismatch')
    require(baseline['kv_format'] == settings['kv_format'] == profile['execution']['kv_dtype'], 'KV mismatch')
    require(baseline['expert_cache_slots'] == settings['expert_cache_slots'], 'Expert cache mismatch')
    require(set(baseline['model_files']) == set(profile['model_files']), 'Model file set mismatch')
    for name,item in baseline['model_files'].items():
        for key in ('bytes','sha256','mtime_ns','ctime_ns'):
            require(item[key] == profile['model_files'][name][key], 'Model receipt mismatch: '+name)
    observed = {}
    for row in rows:
        n, repeat = row['length'], row['repeat']
        require(row['exit_code'] == 0, 'Failed forward')
        token_path = Path(row['stdout']).with_suffix('.tokens')
        require(row['command'] == command+['--tokens-file',str(token_path)], 'Forward command mismatch')
        ids = [100+i*17 % 137 for i in range(n)]
        require(token_path.read_text() == ','.join(map(str,ids)), 'Token fixture mismatch')
        counters = json.loads(Path(row['stdout']).read_text())
        require(counters == row['native_counters'], 'Saved counters differ')
        require(counters['forward_tokens'] == counters['consumed'] == n and counters['position_max'] == n-1,
                'Prefix computation mismatch')
        raw = Path(row['stdout']).with_suffix('.f32').read_bytes()
        require(len(raw) == row['bytes'] == counters['logits_bytes'] == 248320*4, 'Frontier size mismatch')
        require(hashlib.sha256(raw).hexdigest() == row['logits_sha256'], 'Frontier hash mismatch')
        values = array.array('f',raw)
        if sys.byteorder != 'little':
            values.byteswap()
        require(all(map(math.isfinite,values)), 'Nonfinite frontier')
        observed[n,repeat] = values
    measured = {(n,r): max(abs(x-y) for x,y in zip(observed[n,0],observed[n,r]))
                for n in lengths for r in (1,2)}
    comparisons = baseline['comparisons']
    require(len(comparisons) == 18 and {(c['length'],c['repeat']) for c in comparisons} == set(measured),
            'Incomplete repeat comparisons')
    for comparison in comparisons:
        n,r = comparison['length'],comparison['repeat']
        require(comparison['exact_token_ids'] == [100+i*17 % 137 for i in range(n)], 'Comparison fixture mismatch')
        require(comparison['max_absolute_error'] == measured[n,r], 'Recorded error differs from raw logits')
    worst = max(measured.values())
    require(baseline['baseline_max_absolute_error'] == worst and baseline['atol'] == 2*worst
            and baseline['rtol'] == 0, 'Tolerance is not the frozen baseline-derived rule')
    profile.update(model_files=baseline['model_files'],calibration_path=str(Path(calibration_path).resolve()),
        calibration_operator_description=baseline['operator'],
        calibration_sha256=sha(calibration_path),calibration_status='FROZEN',
        admission_path=str(admission_path),admission_sha256=sha(admission_path),
        atol=baseline['atol'],rtol=0,baseline_max_absolute_error=worst,
        tolerance_rule=baseline['tolerance_rule'],numerical_gate_qualified=False)
    with Path(output).open('x') as stream:
        json.dump(profile,stream,indent=2)
        stream.write('\n')
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native_root', type=Path)
    parser.add_argument('calibration', type=Path)
    parser.add_argument('source_receipt', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--admission', type=Path)
    args = parser.parse_args()
    root = args.native_root.resolve()
    calibration = json.loads(args.calibration.read_text())
    admission_path = args.admission or root/'build/q8-dual-engine-admission.json'
    admission = json.loads(admission_path.read_text())
    settings = execution_settings(admission)
    assert settings['kv_format'] == calibration['kv_format']
    assert settings['capacity'] == calibration['capacity']
    assert admission['expert_cache_slots'] == calibration['expert_cache_slots']
    assert admission['oracle_command'] == calibration['command']
    assert sha(admission['oracle_command'][0]) == calibration['oracle_sha256']
    assert sha(admission['sut_command'][0]) == admission['sut_binary_sha256']
    previous_path = Path(calibration['prior_executed_hash_run'])
    previous = json.loads(previous_path.read_text())
    for name, item in calibration['model_files'].items():
        stat = Path(name).stat()
        assert (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) == (item['bytes'], item['mtime_ns'], item['ctime_ns']), name
        assert previous['model_files'][name]['sha256'] == item['sha256'], name
    helper = root/'tools/gguf_reader.py'
    spec = importlib.util.spec_from_file_location('strata_profile_gguf', helper)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    command = admission['sut_command']
    model_path = command[command.index('--native') + 1]
    md = module.GGUFFile(Path(model_path)).metadata
    assert md['general.architecture'] == 'qwen4exp'
    assert md['tokenizer.ggml.model'] == 'gpt2'
    eos = [md['tokenizer.ggml.tokens'].index(token) for token in ('<|endoftext|>', '<|im_end|>')]
    weights = {name: dict(bytes=item['bytes'], sha256=item['sha256']) for name, item in calibration['model_files'].items()}
    model_sha = digest(weights)
    contract = dict(weights_sha256=model_sha, architecture=md['general.architecture'],
        rope_base=str(md['qwen4exp.rope.freq_base']), yarn='model-default',
        tokenizer_identity=model_sha, template_identity='exact-token-ids/no-template',
        attention='causal-qsa-indexed', positions='absolute-zero-based', input_kind='token_ids',
        adapters=[], state_kind='qwen4exp-gdn-qsa-ple-v1')
    source = json.loads(args.source_receipt.read_text())
    for name, expected in source['source_sha256'].items():
        assert sha(root / name) == expected, name
    build = dict(revision=source['source_commit'], source_sha256=source['source_sha256'],
        sha256={name: sha(name) for name in (command[0], admission['oracle_command'][0], str(helper))})
    execution = dict(sut_command=command, oracle_command=admission['oracle_command'],
        capacity=calibration['capacity'], kv_dtype=calibration['kv_format'],
        operator=calibration['operator'], environment=calibration['environment'], eos_tokens=eos)
    # Disjoint from the synthetic calibration histories, fixed before API tests.
    fixtures = dict(prefix=[1 + (i * 13 % 71) for i in range(33)], suffix_a=[71, 72, 73], suffix_b=[81, 82, 83])
    profile = dict(backend='strata', model_path=model_path, model_weights_sha256=model_sha,
        model_files=calibration['model_files'], model_hash_receipt=str(previous_path),
        model_hash_receipt_sha256=sha(previous_path), model_contract=contract,
        model_contract_id=digest(contract), backend_revision=build['revision'], native_build=build,
        gguf_reader=str(helper), execution=execution, hardware=calibration['hardware'],
        execution_profile_id=digest(dict(build=build, execution=execution, hardware=calibration['hardware'])),
        fixtures=fixtures, token_fixtures_sha256=digest(fixtures),
        oracle_binary=admission['oracle_command'][0], oracle_revision=calibration['oracle_source_revision'],
        calibration_path=str(args.calibration.resolve()), calibration_status=calibration['status'],
        admission_path=str(admission_path.resolve()), admission_sha256=sha(admission_path),
        numerical_gate_qualified=False)
    if calibration['status'] == 'FROZEN':
        profile.update(atol=calibration['atol'], rtol=calibration['rtol'],
            baseline_max_absolute_error=calibration['baseline_max_absolute_error'],
            tolerance_rule=calibration['tolerance_rule'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        json.dump(profile, out, indent=2)
        out.write('\n')
    print(json.dumps(dict(path=str(args.output), sha256=sha(args.output),
        model_contract_id=profile['model_contract_id'], execution_profile_id=profile['execution_profile_id'],
        calibration_status=profile['calibration_status'], numerical_gate_qualified=False)))


if __name__ == '__main__':
    main()
