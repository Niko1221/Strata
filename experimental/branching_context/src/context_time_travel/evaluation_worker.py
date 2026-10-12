"""Private bubblewrap entry point; no store or backend imports."""
import json
import random
import resource
from types import MappingProxyType


def freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(freeze(item) for item in value)
    return value


def main():
    resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
    with open('/input.json', 'rb') as stream:
        payload = json.load(stream)
    namespace = {'candidate': freeze(payload['candidate']),
                 'fixtures': MappingProxyType({k: bytes.fromhex(v) for k, v in payload['fixtures'].items()}),
                 'rng': random.Random(payload['seed']), '__name__': '__evaluator__'}
    exec(compile(payload['source'], '<evaluator>', 'exec'), namespace)
    print(json.dumps(namespace['result'], allow_nan=False))


if __name__ == '__main__':
    main()
