"""Validate the executed admission receipt before selecting numerical settings."""


def _options(command):
    if not isinstance(command, list) or not command or any(not isinstance(x, str) for x in command):
        raise ValueError('Invalid executed command')
    result = {}
    index = 1
    while index < len(command):
        name = command[index]
        if not name.startswith('--'):
            raise ValueError('Expected an option name')
        if name in result:
            raise ValueError('Duplicate executed option: ' + name)
        index += 1
        if name in ('--mmap-experts', '--kv-grow'):
            result[name] = True
        else:
            if index == len(command) or command[index].startswith('--'):
                raise ValueError('Missing option value: ' + name)
            result[name] = command[index]
            index += 1
    # Inspected native Options::kv default; no inference from weight format.
    result.setdefault('--kv', 'fp16')
    return result


def execution_settings(admission):
    if admission.get('status') != 'PASS':
        raise ValueError('Executed dual-engine admission must be PASS')
    sut = _options(admission['sut_command'])
    oracle = _options(admission['oracle_command'])
    if '--tokens' in oracle or '--tokens-file' in oracle:
        raise ValueError('Fresh oracle input must be supplied separately for each full history')
    # The branch pipe accepts this unused startup argument; its first consumed
    # token comes from a measured feed command. The fresh executable never does.
    sut.pop('--tokens', None)
    if sut != oracle:
        raise ValueError('SUT and oracle execution settings differ')
    labels = dict(fp16='FP16', int8='INT8', q4_0='Q4_0', k8v4='K8V4')
    if sut['--kv'] not in labels:
        raise ValueError('Unsupported K/V format')
    try:
        capacity = int(sut['--max-context'])
        slots = int(sut['--expert-cache'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('Missing or invalid capacity/cache setting') from error
    if capacity <= 0 or slots <= 0:
        raise ValueError('Invalid capacity/cache setting')
    if type(admission.get('expert_cache_slots')) is not int or admission['expert_cache_slots'] != slots:
        raise ValueError('Admission cache count differs from executed commands')
    return dict(kv_format=labels[sut['--kv']], capacity=capacity, expert_cache_slots=slots)
