import sys
ref=open(sys.argv[1]).read().splitlines()
for p in sys.argv[2:]:
    run=open(p).read().splitlines()
    same=sum(a==b for a,b in zip(ref,run))
    print(f"{p}: {len(run)} rows vs {len(ref)}, {same} identical lines, bit for bit {'yes' if same==len(ref)==len(run) else 'NO'}")
