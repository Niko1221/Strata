"""Shared stub for the launcher-contract tests: a fake `python` that answers the GPU probes
(hipinfo.py) and the model resolver (hfmodel.py) without a GPU, a card, or a cache.  Imported
by test_launcher_contract.py and test_runtime_contract.py; answers mirror the real offline
behavior verified in test_hfmodel_release.py, so a launcher test sees what the real resolver
would answer for this machine's card and the three named releases."""

STUB_PY = """import os, sys
if sys.argv[1].endswith('hipinfo.py'):
    if '--arch' in sys.argv: print('gfx1101')
    elif '--render-node' in sys.argv: print('/dev/dri/renderD128')
    elif '--vram' in sys.argv: print('12272 1500 10772')
    elif '--reserve-mib' in sys.argv: print('768')
elif sys.argv[1].endswith('hfmodel.py'):
    import shlex
    a = sys.argv
    def flag(name):
        return a[a.index(name) + 1] if name in a else ''
    model, repo, rel = flag('--model'), flag('--repo'), flag('--release')
    fam = rel or ('swift' if 'ukisai' in repo else 'coder' if model == 'IQ1_M' else 'qwen')
    if '--print' in a and a[a.index('--print') + 1] == 'release':
        print(fam); sys.exit(0)
    if model == 'IQ1_M' and fam != 'coder':      # the coder-only rule, stubbed as hfmodel dies
        print('hfmodel: IQ1_M is only released for the expert-pruned coder release', file=sys.stderr)
        sys.exit(1)
    if '--print' in a and a[a.index('--print') + 1] == 'shell':
        tag = {'qwen': '', 'swift': 'swift-', 'coder': 'coder-'}[fam]
        name = {'qwen': 'qwen3.8-flash-next', 'swift': 'swift-1.5',
                'coder': 'qwen3.8-flash-next-coder'}[fam]
        lic = 'Swift Open License 1.0: https://huggingface.co/ukisai/x' if fam == 'swift' else ''
        keys = {'MODEL': model, 'CACHED': 1,
                'REPO': {'swift': 'ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF',
                         'coder': 'ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF',
                         'qwen': 'ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF'}[fam],
                'FAMILY': fam, 'PACK_TAG': tag,
                'MODEL_NAME_DEFAULT': f'{name}-{model.lower()}', 'LICENSE': lic,
                'HF_CACHE': '/cache', 'SHARD1': '/cache/s1.gguf', 'SHARD2': '/cache/s2.gguf',
                'PLE_FILE': '/cache/s2.gguf', 'FILES': 2, 'DOWNLOAD_GB': 75.8, 'ARENA_GB': 43,
                'RAM_GB': 60, 'HF_INCLUDE': f'{model}/*', 'MTP_REPO': '', 'MTP_FILE': '',
                'MTP_GGUF': '', 'MTP_GGUF_CACHED': 0, 'MTP_DOWNLOAD_GB': 0}
        for k, v in keys.items():
            print(f'STRATA_{k}={shlex.quote(str(v))}')
else:
    os.execv(sys.executable, [sys.executable] + sys.argv[1:])
"""


def stub_python(path):
    import sys  # noqa: PLC0415 - tiny helper, keeps the stub in one place
    """Write the stub as an executable named `python` (what PYTHON= points at)."""
    path.write_text(f"#!{sys.executable}\n" + STUB_PY)
    path.chmod(0o755)
