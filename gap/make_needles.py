#!/usr/bin/env python3
"""Build single-needle 130K battery files (needs a strata server on :12345 for --tokenize).
Writes gap/tokens/b{05,25,50,75,95}.txt: filler (t32768 repeated) with one needle sentence
at depth d and the query tail last. Recall check: needle's code tokens appear in `output`."""
import urllib.request, json
def tok(s):
    req = urllib.request.Request('http://127.0.0.1:12345/tokenize',
        data=json.dumps({'content': s}).encode(), headers={'content-type': 'application/json'})
    return json.loads(urllib.request.urlopen(req, timeout=5).read())['tokens']
filler = [int(x) for x in open('gap/tokens/t32768.txt').read().split() if x.lstrip('-').isdigit()]
query = tok("Question: what is the secret launch code? Answer: The code is")
L = 130000
for d, code in ((5,'ALPHA-9182'), (25,'BRAVO-4417'), (50,'CHARLIE-6305'), (75,'DELTA-2761'), (95,'ECHO-8840')):
    needle = tok(f"The secret launch code for this experiment is {code}. Remember it.")
    pos = int((L - len(needle) - len(query)) * d / 100)
    ids = (filler * 5)[:pos] + needle + (filler * 5)[:L - pos - len(needle) - len(query)] + query
    open(f'gap/tokens/b{d:02d}.txt', 'w').write(' '.join(map(str, ids)))
