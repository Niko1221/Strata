"""Compare global and domain expert profiles on held-out requests, without a GPU.

Manifest: {"samples": [{"id": "c00", "domain": "code", "trace": "run.trace.gz",
"start": 0, "end": 4224, "decode_entries": 480}, ...]}.
Byte ranges refer to the uncompressed --dump-routing file. decode_entries is the
engine's exact decode lookups + offloaded entries, not the generated token count.
Small prompt windows may precede decoding in a serve trace; they are not scored.

    python tools/evaluate_domain_profiles.py MANIFEST --slots 1582 --out REPORT
        [--folds 5 --seed 20261109 --profiles-dir NEW_DIRECTORY]

The optional profile bank can be used with the existing --expert-profile flag.
Coverage on fixed routing traces does not measure speed or output quality.

Copyright (c) 2026 Francisco Angulo de Lafuente. MIT licensed.
ORCID: https://orcid.org/0009-0001-1634-7063
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

import make_profile as mp


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def decode_counts(data, entries, n_expert=mp.N_EXPERT):
    """Count only an exact, complete-window decode suffix of a request byte range."""
    entries = positive_int(entries, "decode_entries")
    if len(data) < 8 or len(data) % 4:
        raise ValueError("empty or truncated routing record")
    raw = np.frombuffer(data, dtype='<i4')
    k = int(raw[1])
    if not 1 <= k <= min(n_expert, 64):
        raise ValueError("invalid experts-per-token count")
    record = 2 + 2*k
    if raw.size % record:
        raise ValueError("truncated routing record")
    raw = raw.reshape(-1, record)
    if np.any(raw[:, 1] != k):
        raise ValueError("experts-per-token count changes within the trace")
    layers, ids = raw[:, 0], raw[:, 2:2+k]
    if np.any((layers < 0) | (layers >= mp.N_LAYER)) or np.any((ids < 0) | (ids >= n_expert)):
        raise ValueError("layer or expert outside the model geometry")
    changes = np.flatnonzero(np.diff(layers) != 0) + 1
    starts, ends = np.r_[0, changes], np.r_[changes, len(layers)]
    if len(starts) % mp.N_LAYER:
        raise ValueError("incomplete routing window")
    group_layers = layers[starts].reshape(-1, mp.N_LAYER)
    sizes = (ends-starts).reshape(-1, mp.N_LAYER)
    if np.any(group_layers != np.arange(mp.N_LAYER)) or np.any(sizes != sizes[:, :1]):
        raise ValueError("window layers are missing, reordered, or have unequal token counts")
    per_window = sizes[:, 0]*mp.N_LAYER*k
    cumulative = np.cumsum(per_window[::-1])
    matches = np.flatnonzero(cumulative == entries)
    if len(matches) != 1:
        raise ValueError("decode_entries does not select a complete-window suffix")
    first_window = len(per_window)-1-int(matches[0])
    first_record = int(starts[first_window*mp.N_LAYER])
    selected = raw[first_record:]
    counts = np.zeros((mp.N_LAYER, n_expert), dtype=np.int64)
    np.add.at(counts, (np.repeat(selected[:, 0], k), selected[:, 2:2+k].ravel()), 1)
    if int(counts.sum()) != entries:
        raise ValueError("decode entry count did not survive parsing")
    return counts, dict(decode_entries=entries, prompt_entries=int(ids.size-entries), windows=len(per_window)-first_window)


def load_samples(manifest, n_expert=mp.N_EXPERT):
    document = json.loads(Path(manifest).read_text(encoding='utf-8'))
    rows = document['samples']
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest has no samples")
    files, intervals, identities, output = {}, {}, set(), []
    for row in rows:
        identity, domain = row['id'], row['domain']
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError("request IDs must be nonempty and unique")
        if not isinstance(domain, str) or not domain:
            raise ValueError("domain labels must be nonempty strings")
        identities.add(identity)
        path = (Path(manifest).resolve().parent / row['trace']).resolve()
        if path not in files:
            files[path] = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
        data = files[path]
        start, end = row['start'], row['end']
        if any(isinstance(x, bool) or not isinstance(x, int) for x in (start,end)) or not 0 <= start < end <= len(data) or start % 4 or end % 4:
            raise ValueError("invalid request byte range")
        if any(start < b and a < end for a,b in intervals.get(path, [])):
            raise ValueError("request byte ranges overlap")
        intervals.setdefault(path, []).append((start,end))
        counts, audit = decode_counts(data[start:end], row['decode_entries'], n_expert)
        output.append(dict(id=identity,domain=domain,counts=counts,audit=audit))
    return output


def ranked(scores):
    return np.argsort(-scores.ravel(), kind='stable')


def quota_profile(scores, quotas):
    n_expert = scores.shape[1]
    return np.concatenate([layer*n_expert+np.argsort(-scores[layer],kind='stable')[:int(n)] for layer,n in enumerate(quotas)])


def evaluate(samples, slots, folds=5, seed=20261109):
    slots = positive_int(slots, "slots")
    folds = positive_int(folds, "folds")
    shape = samples[0]['counts'].shape
    if slots > shape[0]*shape[1] or folds < 2 or folds > len(samples):
        raise ValueError("slots or fold count exceeds the available data")
    groups = sorted({r['domain'] for r in samples})
    buckets = [[] for _ in range(folds)]
    for domain in groups:
        indices = [i for i,r in enumerate(samples) if r['domain']==domain]
        if len(indices) < 2:
            raise ValueError("each domain needs at least two independent requests")
        # Stable identities make the split independent of the manifest's row order.
        indices.sort(key=lambda i:hashlib.sha256(f"{seed}:{samples[i]['id']}".encode()).digest())
        for j,i in enumerate(indices):
            buckets[j%folds].append(i)
    distribution = np.array([r['counts']/r['counts'].sum() for r in samples])
    results = []
    for fold,test in enumerate(buckets):
        train = sorted((i for i in range(len(samples)) if i not in test), key=lambda i:samples[i]['id'])
        global_score = distribution[train].sum(axis=0)
        global_top = ranked(global_score)[:slots]
        quotas = np.bincount(global_top//shape[1], minlength=shape[0])
        for i in test:
            group_train = [j for j in train if samples[j]['domain']==samples[i]['domain']]
            if not group_train:
                raise ValueError("a held-out domain has no training requests")
            score = distribution[group_train].sum(axis=0)
            domain_top = ranked(score)[:slots]
            constrained = quota_profile(score, quotas)
            counts = samples[i]['counts'].ravel()
            coverage = lambda pairs: float(counts[pairs].sum()/counts.sum())
            results.append(dict(id=samples[i]['id'],domain=samples[i]['domain'],fold=fold,
                training_ids=[samples[j]['id'] for j in train],domain_training_ids=[samples[j]['id'] for j in group_train],
                global_coverage=coverage(global_top),domain_coverage=coverage(domain_top),
                domain_same_layer_quotas_coverage=coverage(constrained),**samples[i]['audit']))
    results.sort(key=lambda r:r['id'])
    mean = lambda field:float(np.mean([r[field] for r in results]))
    return dict(kind='held_out_static_routing_coverage',weighting='one request, one vote',slots=slots,folds=folds,seed=seed,
        requests=len(results),mean_global=mean('global_coverage'),mean_domain=mean('domain_coverage'),
        mean_domain_same_layer_quotas=mean('domain_same_layer_quotas_coverage'),
        domain_minus_global_pp=100*(mean('domain_coverage')-mean('global_coverage')),
        domain_same_quotas_minus_global_pp=100*(mean('domain_same_layer_quotas_coverage')-mean('global_coverage')),
        rows=results,limits=['Static coverage does not measure adaptive caching, copy cost, speed or output quality.',
            'Known domain labels do not measure a classifier.', 'Profiles exclude every request in the evaluated fold.',
            'The caller must assign related prompts or repeated conversations to the same independent request identity.'])


def write_bank(samples, directory, slots):
    """Build deployable profiles from all input requests, for future evaluation only."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    samples = sorted(samples,key=lambda r:r['id'])
    dist = np.array([r['counts']/r['counts'].sum() for r in samples])
    score = dist.sum(axis=0)
    global_rank = ranked(score)
    n_expert = score.shape[1]
    quotas = np.bincount(global_rank[:slots]//n_expert,minlength=mp.N_LAYER)
    index = dict(author='Francisco Angulo de Lafuente',orcid='https://orcid.org/0009-0001-1634-7063',
                 training_ids=[r['id'] for r in samples],slots=slots,global_profile=None,domains={})
    for number,domain in enumerate([None]+sorted({r['domain'] for r in samples})):
        selected = list(range(len(samples))) if domain is None else [i for i,r in enumerate(samples) if r['domain']==domain]
        score = dist[selected].sum(axis=0)
        top = global_rank[:slots] if domain is None else quota_profile(score,quotas)
        # Keep the selected layer capacities at `slots`, and useful prefixes at smaller budgets.
        top = top[np.argsort(-score.ravel()[top],kind='stable')]
        used = set(top.tolist())
        order = np.r_[top,[i for i in ranked(score) if i not in used]].astype(int)
        path = directory/f'profile-{number:02d}.bin'
        pairs = [(int(i//n_expert),int(i%n_expert)) for i in order]
        mp.write_profile(path,pairs,n_expert)
        if mp.read_profile(path,n_expert) != pairs:
            raise ValueError("profile did not survive a round trip")
        info = dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),requests=len(selected))
        if domain is None:
            index['global_profile'] = info
        else:
            index['domains'][domain] = info
    (directory/'index.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8',newline='\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('manifest')
    parser.add_argument('--slots',type=int,required=True)
    parser.add_argument('--folds',type=int,default=5)
    parser.add_argument('--seed',type=int,default=20261109)
    parser.add_argument('--out',required=True)
    parser.add_argument('--profiles-dir')
    args = parser.parse_args()
    samples = load_samples(args.manifest)
    report = evaluate(samples,args.slots,args.folds,args.seed)
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    if args.profiles_dir:
        write_bank(samples,args.profiles_dir,args.slots)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__ == '__main__':
    main()
