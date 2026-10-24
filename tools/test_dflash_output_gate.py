import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
"""Fail-closed checks for the real-weight output/cache gate; no GPU required."""
import copy
import unittest
from dflash_output_gate import evaluate, first_difference


def fixture():
    record = dict(tokens=[1, 2], requested=2, reused=10)
    records = {key: copy.deepcopy(record) for key in
               ('code-greedy','code-sampled','cache-live','cache-restored','cache-checkpoint','cache-cold')}
    records['cache-cold']['reused']=0
    return dict(info={'conversation_cache_mib':'2048', 'expert_slots':'2000',
                      'expert_slots_primary':'2000', 'context':'8192', 'kv':'int8', 'spec':'8'},records=records)


class Gate(unittest.TestCase):
    def test_changed_expert_placement_invalidates_comparison(self):
        r={'fixed-ar':fixture(),'fixed-dflash2':fixture()}
        r['fixed-dflash2']['info']['expert_slots']='1999'
        c=evaluate(r,['code'])['fixed-dflash2']
        self.assertFalse(c['fixed_expert_residency'])
        self.assertFalse(c['target_configuration_matches'])

    def test_prompt_read_field_is_not_offloaded_expert_count(self):
        from dflash_output_gate import ObservedEngine
        from unittest.mock import patch
        done = 'DONE 2 12 1.0 2.0 length 0 0 10 0 0 0 0 0.0 2 123'
        with patch('dflash_sampling.Engine.request', return_value={'tokens':[1,2], 'done':done}):
            r=ObservedEngine.__new__(ObservedEngine).request([0]*12,'temperature=0',2)
        self.assertEqual(r['reused'],10)
        self.assertEqual(r['read'],2)

    def test_exact_output_and_cache(self):
        r={'fixed-ar':fixture(),'fixed-dflash2':fixture()}
        c=evaluate(r,['code'])
        self.assertTrue(all(v['equal'] if isinstance(v,dict) else v for a in c.values() for v in a.values()))

    def test_disabled_cache_and_no_hit_are_failures(self):
        r={'fixed-ar':fixture(),'fixed-dflash2':fixture()}
        r['fixed-dflash2']['info']['conversation_cache_mib']='0'
        r['fixed-dflash2']['records']['cache-restored']['reused']=0
        c=evaluate(r,['code'])['fixed-dflash2']
        self.assertFalse(c['cache_enabled'])
        self.assertFalse(c['cache-restored-hit'])
        r['fixed-ar']['records']['cache-cold']['reused']=10
        self.assertFalse(evaluate(r,['code'])['fixed-ar']['cold_control_missed'])

    def test_missing_target_and_truncation_are_failures(self):
        r={'fixed-dflash2':fixture()}
        r['fixed-dflash2']['records']['code-greedy']['tokens']=[]
        c=evaluate(r,['code'])['fixed-dflash2']
        self.assertFalse(c['target_reference_present'])
        self.assertFalse(c['complete_requests'])

    def test_changed_output_is_not_a_near_tie_without_evidence(self):
        r={'fixed-ar':fixture(),'fixed-dflash2':fixture()}
        r['fixed-dflash2']['records']['code-sampled']['tokens']=[1,3]
        c=evaluate(r,['code'])['fixed-dflash2']
        self.assertFalse(c['code-sampled']['equal'])
        self.assertEqual(c['code-sampled']['first_difference'],1)
        self.assertEqual(first_difference([1],[1,2]),1)
        self.assertIsNone(first_difference([1,2],[1,2]))


if __name__=='__main__': unittest.main()
