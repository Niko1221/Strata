"""A nested-object param goes through the JSON converter, whose char* allows '</parameter>'."""
import sys; sys.path.insert(0,'.')
from serve import grammar_tools as gt, grammar_schema as gs
from pathlib import Path
TOOLS=[{'name':'edit','parameters':{'type':'object','properties':{
  'patch':{'type':'object','properties':{'old':{'type':'string'},'new':{'type':'string'}},'required':['old','new']},
  'mode':{'type':'string'}},'required':['patch','mode'],'additionalProperties':False}}]
txt=gt.build_tool_grammar(TOOLS,'required',None,False, lambda s: gs.schema_to_grammar(Path('build-release'),s))
# does any rule inside the patch subtree exclude the closer?
sub=[l for l in txt.splitlines() if 'edit-0-patch' in l]
print(f"patch subtree: {len(sub)} rules; any exclusion chain (12+ states)?",
      any('patch-string' in l for l in sub))
for l in sub:
    if 'char' in l or 'string ::=' in l: print("  ", l[:130])
