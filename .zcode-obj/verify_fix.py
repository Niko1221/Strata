"""After the PARAM_END exclusion, can a value still hold a closer the parser honours?"""
import sys; sys.path.insert(0,'.')
from serve import grammar_tools as gt
rules=[]
gt.exclude_seq(gt.PARAM_END, "v", rules)
# the automaton must have NO accepting path through "</parameter>":
# state v-11 takes ">" for nothing -> the closer completes and is refused
terminal = [r for r in rules if r.startswith("v-11 ::=")]
print("terminal rule:", terminal[0])
ok = ">" not in terminal[0].split("::=")[1]
print("closer's last char refused:", ok)
# and it must still accept ordinary text
print("restart from every state:", all("| \"\"" in r or r.endswith('""') for r in rules))
