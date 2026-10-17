import json, sys, os
E='.'
arms=sys.argv[1:] or sorted(d for d in os.listdir(E) if os.path.isdir(f'{E}/{d}'))
print(f"{'arm':18s} " + ' '.join(f"{t:>9s}pp {t:>7s}tg" for t in ('4K','32K','128K')))
for a in arms:
    p=f'{E}/{a}/summary.json'
    if not os.path.exists(p): continue
    s=json.load(open(p)); row=[]
    for t in ('4096','32768','128000'):
        v=s.get(t)
        if not v: row.append(f"{'-':>11s} {'-':>9s}"); continue
        pp=v['prefill_tok_s']['median']; tg=v['decode_tok_s']['median']
        row.append(f"{pp:11.0f} {tg:9.1f}" if pp and tg else f"{str(v.keys())[:20]:>21s}")
    print(f"{a:18s} "+' '.join(row))
