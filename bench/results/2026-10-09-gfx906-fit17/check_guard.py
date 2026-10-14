#!/usr/bin/env python3
"""CPU boundary proof; not a substitute for GPU graph-transition parity."""
def arms(rows):
    short = rows[-1] <= 24576
    fit = max(n // 4 for n in rows) + 1 <= 1024 * 17
    return short, not short and fit, not short and not fit

def check(rows):
    selected = arms(rows)
    assert sum(selected) == 1
    if selected[1]:
        assert all(((n // 4 + 1) + 1023) // 1024 <= 17 for n in rows)

for n in range(1, 270336):
    check([n])
for rows in ([24576,24577], [69631,69632], [69632,69631],
             [200000,65536], [200000,24576], [65536]*8):
    check(rows)
assert arms([69631]) == (False,True,False)
assert arms([69632]) == (False,False,True)  # empty tail still takes a block
assert arms([200000,65536]) == (False,False,True)
assert arms([200000,24576]) == (True,False,False)
print('PASS: 270341 guard cases; inclusive scalar limit 69631 cells')
