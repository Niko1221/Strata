"""Does the new grammar really refuse every form param_end treats as a closer?"""
import sys; sys.path.insert(0,'.')
from serve import grammar_tools as gt
rules=[]
gt.exclude_seq(gt.PARAM_END, "v", rules)
text="\n".join(rules)
print(text[:600])
print("\n=== does 'v' accept a value containing '</parameter>'?")
# the automaton's state names encode matched length: reaching len('</parameter>') refuses the char
print("total states:", len([r for r in rules if '::=' in r]))
