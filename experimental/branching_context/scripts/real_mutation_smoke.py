"""T02 six-mutant campaign against unchanged real acceptance assertions.

Test-only monkeypatches deliberately violate contracts; GPU/oracle remain real.
The shared-state mutant corrupts a published checkpoint fixture, not GPU COW.
"""
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for directory in ('src','tests/gates/G1','tests/gates/G2'):
    sys.path.insert(0,str(ROOT/directory))
import test_continuation
import test_controller
import test_crashes
import test_recovery
from context_time_travel.real import BranchingModel, Checkpoint
from context_time_travel.store import Store

original_feed = BranchingModel.feed
original_generate = BranchingModel.generate
original_move = Store.move

def contaminate(self, snapshot, ids):
    if len(ids)==1 and hasattr(self,'_mutant_sibling'):
        snapshot = self._mutant_sibling
    result = original_feed(self,snapshot,ids)
    if len(ids)>1:
        self._mutant_sibling = result
    return result

def unconsumed(self, snapshot, ids):
    return snapshot

def reset_rng(self, snapshot, count):
    decoder = self.controller(snapshot)
    decoder['next_draw'] = 0
    return original_generate(self,self._controller_transition(snapshot,decoder),count)

def bypass_aba(self, name, revision, target, operation):
    return original_move(self,name,self.head(name).revision,target,operation)

def trust_corrupt(self, snapshot, history):
    manifest = json.loads((self.checkpoint_directory/(snapshot+'.json')).read_text())['manifest']
    payload = (self.checkpoint_directory/(manifest['payload_sha256']+'.state')).read_bytes()
    n = manifest['state_bytes']
    frontier = struct.unpack('<'+str(self.backend.vocab_size)+'f',payload[n:])
    return Checkpoint(manifest['model_contract_id'],manifest['execution_profile_id'],len(history),
        payload[:n],frontier,Checkpoint.digest(payload[:n],frontier))

def overwrite_parent(self, snapshot, ids):
    result = original_feed(self,snapshot,ids)
    saved = self.cache[snapshot]
    frontier = self.cache[result].frontier
    self.cache[snapshot] = replace(saved,frontier=frontier,checksum=Checkpoint.digest(saved.blob,frontier))
    return result

cases = [
    ('wrong-prefix',BranchingModel,'feed',contaminate,test_continuation.ContinuationTests,'test_g1_002_fresh_suffix_b'),
    ('unconsumed-frontier',BranchingModel,'feed',unconsumed,test_continuation.ContinuationTests,'test_g1_001_fresh_suffix_a'),
    ('reset-rng-draw-state',BranchingModel,'generate',reset_rng,test_controller.ControllerTests,'test_g1_019_chunked_generate_equivalence'),
    ('aba-stale-revision',Store,'move',bypass_aba,test_crashes.CrashTests,'test_g2_010_aba_pointer_revision'),
    ('bad-checkpoint-hash',BranchingModel,'_load_checkpoint',trust_corrupt,test_recovery.RecoveryTests,'test_g2_003_checkpoint_payload_corruption'),
    ('published-shared-state-overwrite',BranchingModel,'feed',overwrite_parent,test_continuation.ContinuationTests,'test_g1_003_empty_suffix_frontier')]
rows = []
for name,owner,method,mutant,cls,test in cases:
    stream = io.StringIO()
    with patch.object(owner,method,mutant):
        result = unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.TestSuite([cls(test)]))
    killed = result.testsRun==1 and len(result.failures)==1 and not result.errors and not result.skipped
    rows.append(dict(mutant=name,test_id=cls.__module__+'.'+cls.__name__+'.'+test,
        killed=killed,tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),output=stream.getvalue()))
    print(name,'KILLED' if killed else 'NOT QUALIFIED',flush=True)
profile = json.loads(Path(os.environ['CTT_PROFILE']).read_text())
report = dict(status='PASS' if all(row['killed'] for row in rows) else 'FAIL',
    scope='Six T02 mutations with real inference; not the twelve-mutant G5 campaign; no physical GPU-page mutation claim',
    rows=rows,mutation_kills=sum(row['killed'] for row in rows),
    model_weights_sha256=profile['model_weights_sha256'],execution_profile_id=profile['execution_profile_id'],
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(ROOT/'evidence/runs/real-mutation-smoke.json').write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(0 if report['status']=='PASS' else 1)
